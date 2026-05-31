"""
MLflow connection and logging helpers.

Usage from a notebook:
    from chronos_ts.tracking import configure_mlflow, log_classification_run
    configure_mlflow()
    with mlflow.start_run(run_name="catboost_vol_regime"):
        log_classification_run(result, model, X_test, label_maker, run_cfg)

Usage from scripts/train_mlflow.py:
    configure_mlflow(cfg.mlflow.tracking_uri, cfg.mlflow.experiment_name)
"""
from __future__ import annotations

import io
import json
import os
import random
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


# ------------------------------------------------------------------ #
# Connection
# ------------------------------------------------------------------ #

def configure_mlflow(
    tracking_uri: str = "http://localhost:5050",
    experiment_name: str = "chronos-1h-classification",
    s3_endpoint_url: str = "http://localhost:9000",
    aws_access_key_id: str = "admin",
    aws_secret_access_key: str = "password",
) -> str:
    """
    Set env vars and tracking URI. Returns the experiment id.
    Call once at the top of every notebook / script.
    """
    # MinIO runs inside docker network as 'minio:9000'; from the host use localhost.
    if "minio:9000" in s3_endpoint_url:
        s3_endpoint_url = "http://localhost:9000"

    os.environ["MLFLOW_S3_ENDPOINT_URL"] = s3_endpoint_url
    os.environ["AWS_ACCESS_KEY_ID"] = aws_access_key_id
    os.environ["AWS_SECRET_ACCESS_KEY"] = aws_secret_access_key

    import mlflow
    mlflow.set_tracking_uri(tracking_uri)
    exp = mlflow.set_experiment(experiment_name)
    print(f"MLflow URI : {tracking_uri}")
    print(f"Experiment : {experiment_name}  (id={exp.experiment_id})")
    print(f"S3 endpoint: {s3_endpoint_url}")
    return exp.experiment_id


# ------------------------------------------------------------------ #
# Seed
# ------------------------------------------------------------------ #

def set_global_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


# ------------------------------------------------------------------ #
# Logging helpers
# ------------------------------------------------------------------ #

def log_classification_run(
    result: dict[str, Any],
    model: Any,
    X_test: pd.DataFrame,
    label_maker: Any,
    run_cfg: Any,
    register_as: Optional[str] = None,
    promote_to_prd: bool = False,
) -> Optional[str]:
    """
    Log a ClassificationTrainer result to the *active* MLflow run.

    Call inside `with mlflow.start_run(...):`

    Parameters
    ----------
    result       : dict returned by ClassificationTrainer.run()
    model        : fitted sklearn-compatible model
    X_test       : test feature DataFrame (for signature inference)
    label_maker  : fitted LabelMaker instance
    run_cfg      : RunConfig or ClassificationConfig (has .seed, .model_name, etc.)
    register_as  : model name in registry (None = skip registration)
    promote_to_prd: if True, set tag env=PRD and alias 'prd'

    Returns
    -------
    Registered model version (str) or None
    """
    import mlflow
    import mlflow.sklearn
    from mlflow.models import infer_signature

    # --- Params ---
    flat_params = _flatten_params(result, run_cfg)
    mlflow.log_params(flat_params)

    # --- Metrics (train / val / test + baselines) ---
    for split in ("train", "val", "test"):
        m = result["metrics"].get(split, {})
        for k, v in m.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                mlflow.log_metric(f"{split}_{k}", float(v))

    for split in ("val", "test"):
        bl = result.get("baselines", {}).get(split, {}).get("majority_class", {})
        for k, v in bl.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                mlflow.log_metric(f"baseline_{split}_{k}", float(v))

    # --- Artifacts ---
    _log_confusion_png(result, label_maker)
    _log_coverage_curve(result, label_maker)
    _log_prediction_sample(result)
    _log_label_description(label_maker)
    _log_data_provenance(run_cfg)

    # --- Model ---
    registered_version = None
    try:
        sample = X_test.head(5)
        proba = model.predict_proba(sample)
        signature = infer_signature(sample, proba)
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=sample,
            registered_model_name=register_as,
        )
        if register_as and model_info.registered_model_version:
            registered_version = model_info.registered_model_version
            if promote_to_prd:
                _set_prd_alias(register_as, registered_version)
    except Exception as exc:
        mlflow.log_param("model_log_error", str(exc)[:200])

    return registered_version


def _set_prd_alias(model_name: str, version: str) -> None:
    import mlflow
    from mlflow.tracking import MlflowClient
    client = MlflowClient()
    client.set_model_version_tag(model_name, version, "env", "PRD")
    client.set_registered_model_alias(model_name, "prd", version)
    print(f"Registered {model_name} v{version} with tag env=PRD and alias 'prd'")


def _flatten_params(result: dict, run_cfg: Any) -> dict:
    params: dict[str, Any] = {}

    # best hyperparams from grid search
    for k, v in (result.get("best_params_") or result.get("best_params") or {}).items():
        params[f"hp_{k}"] = str(v)

    # label config
    lm = result.get("label_maker", {})
    cfg_d = lm.get("config", {})
    for k, v in cfg_d.items():
        params[f"label_{k}"] = str(v)

    # split / training config
    params["seed"] = getattr(run_cfg, "seed", 42)
    params["model_name"] = getattr(run_cfg, "model_name", "unknown")
    params["cv_splits"] = getattr(run_cfg, "cv_splits", "?")
    params["n_features"] = result.get("feature_cols") and len(result["feature_cols"])
    params["n_classes"] = result.get("n_classes", "?")

    split = result.get("config", {}).get("split", {})
    if isinstance(split, dict):
        params["train_frac"] = split.get("train_frac", "?")
        params["val_frac"] = split.get("val_frac", "?")
        params["test_frac"] = split.get("test_frac", "?")

    return {k: str(v) for k, v in params.items() if v is not None}


def _log_confusion_png(result: dict, label_maker: Any) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import mlflow

        cm = result["metrics"]["test"].get("confusion_matrix")
        names = label_maker.class_names()
        if cm is None:
            return
        arr = np.array(cm, dtype=float)
        fig, ax = plt.subplots(figsize=(max(4, len(names)), max(3, len(names))))
        im = ax.imshow(arr, cmap="Blues")
        ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title("Confusion matrix (test)")
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, int(arr[i, j]), ha="center", va="center",
                        color="white" if arr[i, j] > arr.max() / 2 else "black")
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100)
        buf.seek(0)
        mlflow.log_image(buf.read(), "confusion_matrix.png")
        plt.close(fig)
    except Exception:
        pass


def _log_coverage_curve(result: dict, label_maker: Any) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import mlflow

        preds_path = _find_predictions_csv(result)
        if preds_path is None:
            return
        df = pd.read_csv(preds_path)
        proba_cols = [c for c in df.columns if c.startswith("proba_")]
        if not proba_cols:
            return

        max_proba = df[proba_cols].max(axis=1)
        thresholds = np.linspace(0.3, 0.95, 30)
        coverages, accs = [], []
        for t in thresholds:
            mask = max_proba >= t
            coverages.append(mask.mean())
            if mask.sum() > 0:
                accs.append((df.loc[mask, "y_true"] == df.loc[mask, "y_pred"]).mean())
            else:
                accs.append(float("nan"))

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(coverages, accs, marker="o", markersize=4)
        ax.axhline(1 / label_maker.n_classes(), ls="--", color="gray", label="random")
        ax.set_xlabel("Coverage (fraction predicted)"); ax.set_ylabel("Accuracy")
        ax.set_title("Confident accuracy vs coverage (test)")
        ax.legend(); plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100); buf.seek(0)
        mlflow.log_image(buf.read(), "coverage_curve.png")
        plt.close(fig)
    except Exception:
        pass


def _find_predictions_csv(result: dict) -> Optional[str]:
    cfg = result.get("config", {})
    out_dir = cfg.get("output_dir")
    model = cfg.get("model_name", "")
    family = cfg.get("label", {}).get("target_family", "") if isinstance(cfg.get("label"), dict) else ""
    if out_dir and model and family:
        p = Path(out_dir) / f"{model}_{family}_test_predictions.csv"
        if p.exists():
            return str(p)
    return None


def _log_prediction_sample(result: dict) -> None:
    try:
        import mlflow
        preds_path = _find_predictions_csv(result)
        if preds_path and Path(preds_path).exists():
            df = pd.read_csv(preds_path)
            mlflow.log_artifact(preds_path, "predictions")
    except Exception:
        pass


def _log_label_description(label_maker: Any) -> None:
    try:
        import mlflow
        desc = label_maker.describe()
        buf = io.BytesIO(json.dumps(desc, indent=2, default=str).encode())
        mlflow.log_text(buf.read().decode(), "label_config.json")
    except Exception:
        pass


def _log_data_provenance(run_cfg: Any) -> None:
    try:
        import mlflow
        prov = {
            "data_csv": getattr(run_cfg, "data_csv", "unknown"),
            "source": "Binance BTCUSDT public futures API",
            "endpoint": "fapi.binance.com/fapi/v1/klines (1h)",
            "features": "chronos_ts.dataset.ExperimentDatasetBuilder (lag/rolling)",
            "seed": getattr(run_cfg, "seed", 42),
        }
        mlflow.log_text(json.dumps(prov, indent=2), "data_provenance.json")
    except Exception:
        pass
