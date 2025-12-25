import json
import logging
import math
import time
from typing import Any, Dict

import numpy as np
from fastapi import Depends, FastAPI, Request, status
from fastapi.responses import PlainTextResponse
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from app.db import engine, get_db
from app.inference import load_artifacts, predict_one
from app.models import Base, RequestHistory

logger = logging.getLogger(__name__)

app = FastAPI(title="CHRONOS ML Service")

BAD_REQUEST_TEXT = "bad request"
MODEL_FAILED_TEXT = "model could not process the data"


@app.on_event("startup")
def _startup() -> None:
    Base.metadata.create_all(bind=engine)


def _plain(text: str, code: int) -> PlainTextResponse:
    return PlainTextResponse(text, status_code=code)


def _bad_request() -> PlainTextResponse:
    return _plain(BAD_REQUEST_TEXT, status.HTTP_400_BAD_REQUEST)


def _parse_constant(const: str) -> str:
    # Called by json.loads(parse_constant=...) for: Infinity, -Infinity, NaN :contentReference[oaicite:1]{index=1}
    if const == "Infinity":
        return "Infinity"
    if const == "-Infinity":
        return "-Infinity"
    if const == "NaN":
        return "NaN"
    return const


def _make_json_safe(x: Any) -> Any:
    if isinstance(x, float):
        if math.isnan(x):
            return "NaN"
        if x == float("inf"):
            return "Infinity"
        if x == float("-inf"):
            return "-Infinity"
        return x

    if isinstance(x, dict):
        out: Dict[str, Any] = {}
        for k, v in x.items():
            out[str(k)] = _make_json_safe(v)
        return out

    if isinstance(x, list):
        return [_make_json_safe(v) for v in x]

    return x


def _json_dumps_safe(x: Any) -> str:
    return json.dumps(_make_json_safe(x), ensure_ascii=False)


def _safe_json_loads(s: str) -> Any:
    try:
        obj = json.loads(s, parse_constant=_parse_constant)
        return _make_json_safe(obj)
    except Exception:
        return s


def _validate_features(features: Any) -> bool:
    if not isinstance(features, dict):
        return False

    _, meta = load_artifacts()
    raw_cols = meta.get("raw_cols", [])
    if not isinstance(raw_cols, list) or not raw_cols:
        return False

    for c in raw_cols:
        if c not in features:
            return False
        v = features.get(c)
        if v is None:
            return False
        try:
            float(v)
        except Exception:
            return False

    return True


@app.post("/forward")
async def forward(request: Request, db: Session = Depends(get_db)):
    content_type = (request.headers.get("content-type") or "").lower()
    if "application/json" not in content_type:
        return _bad_request()

    try:
        payload: Any = await request.json()
    except Exception:
        return _bad_request()

    if not isinstance(payload, dict):
        return _bad_request()

    features = payload.get("features")
    if not _validate_features(features):
        return _bad_request()

    status_code = status.HTTP_200_OK
    error_text = ""
    response_obj: Any

    t0 = time.perf_counter()
    try:
        pred, _model_ms = predict_one(features)
        response_obj = {"prediction": float(pred)}
    except Exception:
        logger.exception("Model inference failed")
        status_code = status.HTTP_403_FORBIDDEN
        error_text = MODEL_FAILED_TEXT
        response_obj = MODEL_FAILED_TEXT

    duration_ms = (time.perf_counter() - t0) * 1000.0

    # add elapsed_ms to success response
    if status_code == status.HTTP_200_OK and isinstance(response_obj, dict):
        response_obj["elapsed_ms"] = float(duration_ms)

    hist = RequestHistory(
        path="/forward",
        status_code=int(status_code),
        duration_ms=float(duration_ms),
        headers_json=_json_dumps_safe(dict(request.headers)),
        body_json=_json_dumps_safe(payload),
        response_json=_json_dumps_safe(response_obj) if not isinstance(response_obj, str) else response_obj,
        error_text=error_text,
    )
    db.add(hist)
    db.commit()

    if status_code == status.HTTP_403_FORBIDDEN:
        return _plain(MODEL_FAILED_TEXT, status_code)

    return response_obj


@app.get("/history")
def history(limit: int = 200, offset: int = 0, db: Session = Depends(get_db)):
    if limit < 0 or offset < 0:
        return _bad_request()
    if limit > 2000:
        limit = 2000

    stmt = select(RequestHistory).order_by(desc(RequestHistory.id)).limit(limit).offset(offset)
    rows = db.execute(stmt).scalars().all()

    out = []
    for r in rows:
        out.append(
            {
                "id": r.id,
                "created_at": r.created_at.isoformat(),
                "path": r.path,
                "status_code": r.status_code,
                "duration_ms": r.duration_ms,
                "headers": _safe_json_loads(r.headers_json) if r.headers_json else {},
                "body": _safe_json_loads(r.body_json) if r.body_json else None,
                "response": _safe_json_loads(r.response_json) if r.response_json else None,
                "error": r.error_text or "",
            }
        )

    return {"items": out, "limit": limit, "offset": offset, "count": len(out)}


@app.get("/stats")
def stats(db: Session = Depends(get_db)):
    stmt = (
        select(RequestHistory.status_code, RequestHistory.duration_ms, RequestHistory.body_json)
        .where(RequestHistory.path == "/forward")
    )
    rows = db.execute(stmt).all()

    total = len(rows)
    success_rows = [r for r in rows if int(r.status_code) == 200]
    failed_rows = total - len(success_rows)

    durations_all = np.array([float(r.duration_ms or 0.0) for r in rows], dtype=float)
    durations_ok = np.array([float(r.duration_ms or 0.0) for r in success_rows], dtype=float)

    def _quantiles(arr: np.ndarray):
        if arr.size == 0:
            return {"mean": None, "p50": None, "p95": None, "p99": None}
        p50, p95, p99 = np.percentile(arr, [50, 95, 99]).tolist()
        return {"mean": float(arr.mean()), "p50": float(p50), "p95": float(p95), "p99": float(p99)}

    _, meta = load_artifacts()
    expected = set(meta.get("raw_cols", []))

    num_features_list = []
    missing_list = []
    payload_bytes_list = []

    for r in rows:
        body_str = r.body_json or ""
        payload_bytes_list.append(len(body_str.encode("utf-8")))

        try:
            body = json.loads(body_str, parse_constant=_parse_constant) if body_str else {}
            feats = body.get("features", {})
            if isinstance(feats, dict):
                keys = set(feats.keys())
                num_features_list.append(len(keys))
                missing_list.append(len(expected - keys) if expected else 0)
            else:
                num_features_list.append(0)
                missing_list.append(len(expected) if expected else 0)
        except Exception:
            num_features_list.append(0)
            missing_list.append(len(expected) if expected else 0)

    return {
        "requests_total": total,
        "requests_success": len(success_rows),
        "requests_failed": failed_rows,
        "expected_raw_cols": sorted(list(expected)),
        "latency_ms_all": _quantiles(durations_all),
        "latency_ms_success": _quantiles(durations_ok),
        "input_num_features": _quantiles(np.array(num_features_list, dtype=float)),
        "input_missing_features": _quantiles(np.array(missing_list, dtype=float)),
        "input_payload_bytes": _quantiles(np.array(payload_bytes_list, dtype=float)),
    }
