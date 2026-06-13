from __future__ import annotations
from typing import Any, Mapping

_MAX_PARAM_LEN = 500
_BATCH_SIZE = 100


def _truncate(value: str) -> str:
    if len(value) <= _MAX_PARAM_LEN:
        return value
    return value[: _MAX_PARAM_LEN - 3] + '...'


def flatten_mapping(data: Any, prefix: str = '') -> dict[str, str]:
    out: dict[str, str] = {}
    if data is None:
        return out
    if isinstance(data, Mapping):
        for key, value in data.items():
            key_s = str(key)
            path = f'{prefix}.{key_s}' if prefix else key_s
            if isinstance(value, Mapping):
                out.update(flatten_mapping(value, path))
            elif isinstance(value, (list, tuple)):
                out[path] = _truncate(str(value))
            elif value is not None and not isinstance(value, (dict,)):
                out[path] = _truncate(str(value))
        return out
    if prefix:
        out[prefix] = _truncate(str(data))
    return out


def safe_log_params(params: Mapping[str, Any]) -> None:
    from chronos_ts.mlflow_client import get_mlflow
    mlflow = get_mlflow()
    cleaned: dict[str, str] = {}
    for key, value in params.items():
        if value is None:
            continue
        k = str(key).replace('/', '_')[:250]
        cleaned[k] = _truncate(str(value))
    items = list(cleaned.items())
    for i in range(0, len(items), _BATCH_SIZE):
        mlflow.log_params(dict(items[i : i + _BATCH_SIZE]))
