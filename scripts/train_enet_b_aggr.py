from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import hydra
import joblib
import numpy as np
import pandas as pd
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from chronos_ml.transformers import ChronosFeatureEngineerB

logger = logging.getLogger(__name__)


def time_split(
    df: pd.DataFrame, train_frac: float, val_frac: float, test_frac: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assert abs((train_frac + val_frac + test_frac) - 1.0) < 1e-9, "fractions must sum to 1.0"

    df = df.sort_values("ts").reset_index(drop=True)
    n = len(df)

    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    n_test = n - n_train - n_val

    assert min(n_train, n_val, n_test) > 0, "split too small"

    return (
        df.iloc[:n_train].copy(),
        df.iloc[n_train : n_train + n_val].copy(),
        df.iloc[n_train + n_val :].copy(),
    )


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    std = float(np.std(y_true)) if np.std(y_true) > 0 else np.nan

    corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else np.nan
    r2 = float(r2_score(y_true, y_pred))

    sign_true = np.sign(y_true)
    sign_pred = np.sign(y_pred)
    mask = (sign_true != 0) & (sign_pred != 0)
    dir_acc = float((sign_true[mask] == sign_pred[mask]).mean()) if mask.sum() else np.nan

    return {
        "rmse": rmse,
        "mae": mae,
        "nrmse": rmse / std if std and not np.isnan(std) else np.nan,
        "mae_over_std": mae / std if std and not np.isnan(std) else np.nan,
        "r2": r2,
        "corr": corr,
        "directional_accuracy": dir_acc,
    }


@hydra.main(version_base=None, config_path="../configs", config_name="train_enet_b_aggr")
def main(cfg: DictConfig) -> None:
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    data_path = Path(to_absolute_path(cfg.data_path))
    df = pd.read_csv(data_path, parse_dates=["ts"]).sort_values("ts").reset_index(drop=True)

    for col in ("fundingRate", "markPrice", "premium_close"):
        if col in df.columns:
            df[col] = df[col].ffill()

    df["target_log_ret_1h"] = df["log_ret_1h"].shift(-1)

    raw_cols = list(cfg.columns.required_raw)
    df = df.dropna(subset=raw_cols + ["target_log_ret_1h"]).reset_index(drop=True)

    df_train, df_val, df_test = time_split(
        df,
        float(cfg.split.train_frac),
        float(cfg.split.val_frac),
        float(cfg.split.test_frac),
    )

    X_train = df_train[raw_cols]
    y_train = df_train["target_log_ret_1h"].to_numpy(dtype=float)

    X_val = df_val[raw_cols]
    y_val = df_val["target_log_ret_1h"].to_numpy(dtype=float)

    X_test = df_test[raw_cols]
    y_test = df_test["target_log_ret_1h"].to_numpy(dtype=float)

    estimator = Pipeline(
        [
            ("fe", ChronosFeatureEngineerB(required_raw=raw_cols, output_cols=list(cfg.columns.output_cols))),
            ("scaler", StandardScaler()),
            ("model", ElasticNet(max_iter=10000)),
        ]
    )

    param_grid = {
        "model__alpha": list(cfg.grid.alpha),
        "model__l1_ratio": list(cfg.grid.l1_ratio),
    }

    cv = TimeSeriesSplit(n_splits=int(cfg.split.cv_splits))
    gcv = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=cv,
        n_jobs=int(cfg.n_jobs),
        refit=True,
    )

    gcv.fit(X_train, y_train)
    best_model: Pipeline = gcv.best_estimator_

    y_val_pred = best_model.predict(X_val)
    y_test_pred = best_model.predict(X_test)

    metrics: Dict[str, Any] = {
        "best_params": gcv.best_params_,
        "cv_best_score_neg_mse": float(gcv.best_score_),
        "val": regression_metrics(y_val, y_val_pred),
        "test": regression_metrics(y_test, y_test_pred),
        "n_rows": {"train": int(len(df_train)), "val": int(len(df_val)), "test": int(len(df_test))},
        "ts_range": {"min": str(df["ts"].min()), "max": str(df["ts"].max())},
    }

    artifacts_dir = Path(to_absolute_path(cfg.artifacts_dir))
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    model_path = artifacts_dir / "best_enet_B_aggr.joblib"
    metrics_path = artifacts_dir / "best_enet_B_aggr.metrics.json"
    meta_path = artifacts_dir / "best_enet_B_aggr.meta.json"

    joblib.dump(best_model, model_path)

    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    meta = {
        "model_path": str(model_path),
        "raw_cols": raw_cols,
        "output_cols": list(cfg.columns.output_cols),
        "target": "target_log_ret_1h",
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("Saved: %s", model_path)
    logger.info("Saved: %s", metrics_path)
    logger.info("Saved: %s", meta_path)


if __name__ == "__main__":
    main()
