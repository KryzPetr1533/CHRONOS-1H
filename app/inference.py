import json
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import pandas as pd


@lru_cache(maxsize=1)
def load_artifacts(
    model_path: str = "artifacts/best_enet_B_aggr.joblib",
    meta_path: str = "artifacts/best_enet_B_aggr.meta.json",
) -> Tuple[Any, Dict[str, Any]]:
    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    model = joblib.load(model_path)
    return model, meta


def predict_one(features: Dict[str, float]) -> Tuple[float, float]:
    model, meta = load_artifacts()

    raw_cols = list(meta["raw_cols"])
    row = {k: features.get(k) for k in raw_cols}
    X = pd.DataFrame([row], columns=raw_cols)

    t0 = time.perf_counter()
    y = model.predict(X)
    dt_ms = (time.perf_counter() - t0) * 1000.0

    return float(y[0]), float(dt_ms)
