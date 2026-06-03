from __future__ import annotations
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SHADOW_NAMES = ('mlflow',)
_SHADOW_PATHS = tuple((p for p in (_REPO_ROOT / 'mlflow', _REPO_ROOT / 'infra' / 'mlflow') if p.is_dir()))


def _is_real_mlflow(mod: Any) -> bool:
    return mod is not None and callable(getattr(mod, 'set_tracking_uri', None))


def _purge_shadow_module() -> None:
    mod = sys.modules.get('mlflow')
    if mod is not None and _is_real_mlflow(mod):
        return
    for key in list(sys.modules):
        if key == 'mlflow' or key.startswith('mlflow.'):
            del sys.modules[key]


def get_mlflow():
    _purge_shadow_module()
    import mlflow
    if not _is_real_mlflow(mlflow):
        shadow_hint = ''
        if (_REPO_ROOT / 'mlflow').is_dir():
            shadow_hint = (
                f' Remove or rename {_REPO_ROOT / "mlflow"} (Docker stack lives in infra/mlflow/).'
            )
        raise ImportError(
            'The pip package "mlflow" was shadowed by a local mlflow/ directory on PYTHONPATH.'
            + shadow_hint
        )
    return mlflow
