from __future__ import annotations
import io
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Optional
import numpy as np
import pandas as pd

def _mlflow():
    from chronos_ts.mlflow_client import get_mlflow
    return get_mlflow()

def _log_metric_dict(metrics_by_split: dict, prefix_split: bool = True, name_prefix: str = '') -> None:
    mlflow = _mlflow()
    for split, m in metrics_by_split.items():
        if not isinstance(m, dict):
            continue
        for k, v in m.items():
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                continue
            fv = float(v)
            if math.isnan(fv) or math.isinf(fv):
                continue
            key = f'{name_prefix}{k}' if name_prefix else (f'{split}_{k}' if prefix_split else k)
            mlflow.log_metric(key, fv)

def log_hydra_config(cfg: Any) -> None:
    try:
        from omegaconf import OmegaConf
        from chronos_ts.mlflow_params import flatten_mapping, safe_log_params
        container = OmegaConf.to_container(cfg, resolve=True)
        params = flatten_mapping(container, prefix='cfg')
        if params:
            safe_log_params(params)
    except Exception as exc:
        _mlflow().log_param('hydra_params_error', str(exc)[:500])

def configure_mlflow(tracking_uri: str='http://localhost:5050', experiment_name: str='chronos-1h-classification', s3_endpoint_url: str='http://localhost:9000', aws_access_key_id: str='admin', aws_secret_access_key: str='password') -> str:
    if os.environ.get('MLFLOW_S3_ENDPOINT_URL'):
        s3_endpoint_url = os.environ['MLFLOW_S3_ENDPOINT_URL']
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = s3_endpoint_url
    os.environ['AWS_ACCESS_KEY_ID'] = aws_access_key_id
    os.environ['AWS_SECRET_ACCESS_KEY'] = aws_secret_access_key
    mlflow = _mlflow()
    mlflow.set_tracking_uri(tracking_uri)
    exp = mlflow.set_experiment(experiment_name)
    print(f'MLflow URI : {tracking_uri}')
    print(f'Experiment : {experiment_name}  (id={exp.experiment_id})')
    print(f'S3 endpoint: {s3_endpoint_url}')
    return exp.experiment_id

def set_global_seed(seed: int=42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass

def log_classification_run(result: dict[str, Any], model: Any, X_test: pd.DataFrame, label_maker: Any, run_cfg: Any, register_as: Optional[str]=None, promote_to_prd: bool=False, hydra_cfg: Any=None) -> Optional[str]:
    mlflow = _mlflow()
    from mlflow.models import infer_signature
    from chronos_ts.mlflow_params import safe_log_params
    if hydra_cfg is not None:
        log_hydra_config(hydra_cfg)
    flat_params = _flatten_params(result, run_cfg)
    safe_log_params(flat_params)
    _log_metric_dict(result.get('metrics', {}), prefix_split=True)
    for split in ('val', 'test'):
        for bl_name, bl_metrics in (result.get('baselines', {}).get(split) or {}).items():
            if isinstance(bl_metrics, dict):
                _log_metric_dict({split: bl_metrics}, prefix_split=False, name_prefix=f'baseline_{split}_{bl_name}_')
    _log_confusion_png(result, label_maker)
    _log_coverage_curve(result, label_maker)
    _log_prediction_sample(result)
    _log_label_description(label_maker)
    _log_data_provenance(run_cfg)
    registered_version = None
    try:
        sample = X_test.head(5)
        proba = model.predict_proba(sample)
        signature = infer_signature(sample, proba)
        model_info = _log_model_artifact(model, sample, signature, register_as)
        if register_as and model_info.registered_model_version:
            registered_version = model_info.registered_model_version
            if promote_to_prd:
                _set_prd_alias(register_as, registered_version)
        elif register_as:
            run_id = mlflow.active_run().info.run_id
            mv = mlflow.register_model(f'runs:/{run_id}/model', register_as)
            registered_version = mv.version
            if promote_to_prd:
                _set_prd_alias(register_as, registered_version)
    except Exception as exc:
        err = str(exc)[:500]
        mlflow.log_param('model_log_error', err)
        print(f'[MLflow model registration failed: {err}]')
    return registered_version


def _log_model_artifact(model: Any, sample: pd.DataFrame, signature: Any, register_as: Optional[str]):
    mlflow = _mlflow()
    model_module = type(model).__module__
    kwargs = dict(artifact_path='model', signature=signature, input_example=sample)
    if 'catboost' in model_module:
        import mlflow.catboost
        return mlflow.catboost.log_model(cb_model=model, **kwargs)
    import mlflow.sklearn
    return mlflow.sklearn.log_model(sk_model=model, **kwargs)

def _set_prd_alias(model_name: str, version: str) -> None:
    from mlflow.tracking import MlflowClient
    client = MlflowClient()
    client.set_model_version_tag(model_name, version, 'env', 'PRD')
    client.set_registered_model_alias(model_name, 'prd', version)
    print(f"Registered {model_name} v{version} with tag env=PRD and alias 'prd'")

def _flatten_params(result: dict, run_cfg: Any) -> dict:
    params: dict[str, Any] = {}
    for k, v in (result.get('best_params_') or result.get('best_params') or {}).items():
        params[f'hp_{k}'] = str(v)
    if result.get('cv_best_score') is not None:
        params['cv_best_score'] = result['cv_best_score']
    lm = result.get('label_maker', {})
    cfg_d = lm.get('config', {}) if isinstance(lm, dict) else {}
    for k, v in cfg_d.items():
        params[f'label_{k}'] = str(v)
    params['seed'] = getattr(run_cfg, 'seed', 42)
    params['model_name'] = getattr(run_cfg, 'model_name', 'unknown')
    params['label_family'] = getattr(run_cfg, 'label_family', '?')
    params['data_csv'] = getattr(run_cfg, 'data_csv', '?')
    params['cv_splits'] = getattr(run_cfg, 'cv_splits', '?')
    params['abstention_threshold'] = getattr(run_cfg, 'abstention_threshold', '?')
    params['top_k_pct'] = getattr(run_cfg, 'top_k_pct', '?')
    params['n_features'] = result.get('feature_cols') and len(result['feature_cols'])
    params['n_classes'] = result.get('n_classes', '?')
    sizes = result.get('split_sizes') or {}
    for split_name, count in sizes.items():
        params[f'split_{split_name}_rows'] = count
    cfg = result.get('config', {})
    split = cfg.get('split', {}) if isinstance(cfg, dict) else {}
    if isinstance(split, dict):
        params['train_frac'] = split.get('train_frac', '?')
        params['val_frac'] = split.get('val_frac', '?')
        params['test_frac'] = split.get('test_frac', '?')
    elif hasattr(split, 'train_frac'):
        params['train_frac'] = split.train_frac
        params['val_frac'] = split.val_frac
        params['test_frac'] = split.test_frac
    return {k: str(v) for k, v in params.items() if v is not None}

def _log_confusion_png(result: dict, label_maker: Any) -> None:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        mlflow = _mlflow()
        cm = result['metrics']['test'].get('confusion_matrix')
        names = label_maker.class_names()
        if cm is None:
            return
        arr = np.array(cm, dtype=float)
        fig, ax = plt.subplots(figsize=(max(4, len(names)), max(3, len(names))))
        im = ax.imshow(arr, cmap='Blues')
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion matrix (test)')
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, int(arr[i, j]), ha='center', va='center', color='white' if arr[i, j] > arr.max() / 2 else 'black')
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        mlflow.log_image(buf.read(), 'confusion_matrix.png')
        plt.close(fig)
    except Exception:
        pass

def _log_coverage_curve(result: dict, label_maker: Any) -> None:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        mlflow = _mlflow()
        preds_path = _find_predictions_csv(result)
        if preds_path is None:
            return
        df = pd.read_csv(preds_path)
        proba_cols = [c for c in df.columns if c.startswith('proba_')]
        if not proba_cols:
            return
        max_proba = df[proba_cols].max(axis=1)
        thresholds = np.linspace(0.3, 0.95, 30)
        coverages, accs = ([], [])
        for t in thresholds:
            mask = max_proba >= t
            coverages.append(mask.mean())
            if mask.sum() > 0:
                accs.append((df.loc[mask, 'y_true'] == df.loc[mask, 'y_pred']).mean())
            else:
                accs.append(float('nan'))
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(coverages, accs, marker='o', markersize=4)
        ax.axhline(1 / label_maker.n_classes(), ls='--', color='gray', label='random')
        ax.set_xlabel('Coverage (fraction predicted)')
        ax.set_ylabel('Accuracy')
        ax.set_title('Confident accuracy vs coverage (test)')
        ax.legend()
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        mlflow.log_image(buf.read(), 'coverage_curve.png')
        plt.close(fig)
    except Exception:
        pass

def _find_predictions_csv(result: dict) -> Optional[str]:
    cfg = result.get('config', {})
    out_dir = cfg.get('output_dir')
    model = cfg.get('model_name', '')
    family = cfg.get('label', {}).get('target_family', '') if isinstance(cfg.get('label'), dict) else ''
    if out_dir and model and family:
        p = Path(out_dir) / f'{model}_{family}_test_predictions.csv'
        if p.exists():
            return str(p)
    return None

def _log_prediction_sample(result: dict) -> None:
    try:
        mlflow = _mlflow()
        preds_path = _find_predictions_csv(result)
        if preds_path and Path(preds_path).exists():
            df = pd.read_csv(preds_path)
            mlflow.log_artifact(preds_path, 'predictions')
    except Exception:
        pass

def _log_label_description(label_maker: Any) -> None:
    try:
        mlflow = _mlflow()
        desc = label_maker.describe()
        buf = io.BytesIO(json.dumps(desc, indent=2, default=str).encode())
        mlflow.log_text(buf.read().decode(), 'label_config.json')
    except Exception:
        pass

def _log_data_provenance(run_cfg: Any) -> None:
    try:
        mlflow = _mlflow()
        prov = {'data_csv': getattr(run_cfg, 'data_csv', 'unknown'), 'source': 'Binance BTCUSDT public futures API', 'endpoint': 'fapi.binance.com/fapi/v1/klines (1h)', 'features': 'chronos_ts.dataset.ExperimentDatasetBuilder (lag/rolling)', 'seed': getattr(run_cfg, 'seed', 42)}
        mlflow.log_text(json.dumps(prov, indent=2), 'data_provenance.json')
    except Exception:
        pass