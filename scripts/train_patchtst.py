from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from neuralforecast import NeuralForecast
from neuralforecast.models import PatchTST
from neuralforecast.losses.pytorch import MAE


DATA_CSV = Path("outputs/datasets/btcusdt_patchtst.csv")
OUT_DIR = Path("outputs/models/patchtst_core")
OUT_DIR.mkdir(parents=True, exist_ok=True)

H = 1
INPUT_SIZE = 168
VAL_SIZE = 3707
TEST_SIZE = 3708

HIST_EXOG = [
    "premium_chg",
    "fundingRate",
    "vol_chg",
    "trades_chg",
    "taker_buy_share",
    "rv_24",
    "fundingRate_missing",
    "taker_buy_share_missing",
]

FUTR_EXOG = [
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
]


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    std = float(np.std(y_true)) if np.std(y_true) > 0 else np.nan

    pearson = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else np.nan
    try:
        spearman = float(pd.Series(y_true).corr(pd.Series(y_pred), method="spearman"))
    except Exception:
        spearman = np.nan

    sign_true = np.sign(y_true)
    sign_pred = np.sign(y_pred)
    mask = (sign_true != 0) & (sign_pred != 0)
    dir_acc = float((sign_true[mask] == sign_pred[mask]).mean()) if mask.sum() else np.nan

    abs_pred = np.abs(y_pred)
    cutoff = np.quantile(abs_pred, 0.8) if len(abs_pred) else np.nan
    top_mask = abs_pred >= cutoff if np.isfinite(cutoff) else np.zeros_like(abs_pred, dtype=bool)
    top_dir_acc = float((sign_true[top_mask] == sign_pred[top_mask]).mean()) if top_mask.sum() > 0 else np.nan

    strat_ret = np.sign(y_pred) * y_true
    strat_mean = float(np.mean(strat_ret))
    strat_std = float(np.std(strat_ret))
    strat_sharpe = float(np.sqrt(24 * 365) * strat_mean / strat_std) if strat_std > 0 else np.nan

    return {
        "rmse": rmse,
        "mae": mae,
        "nrmse": rmse / std if std and not np.isnan(std) else np.nan,
        "mae_over_std": mae / std if std and not np.isnan(std) else np.nan,
        "r2": float(r2_score(y_true, y_pred)),
        "pearson_corr": pearson,
        "spearman_corr": spearman,
        "directional_accuracy": dir_acc,
        "top20_directional_accuracy": top_dir_acc,
        "sign_strategy_mean": strat_mean,
        "sign_strategy_sharpe": strat_sharpe,
    }


def main() -> None:
    df = pd.read_csv(DATA_CSV, parse_dates=["ds"]).sort_values("ds").reset_index(drop=True)

    model = PatchTST(
        h=H,
        input_size=INPUT_SIZE,
        hist_exog_list=HIST_EXOG,
        futr_exog_list=FUTR_EXOG,
        encoder_layers=3,
        n_heads=8,
        hidden_size=128,
        linear_hidden_size=256,
        dropout=0.1,
        loss=MAE(),
        valid_loss=MAE(),
        scaler_type="robust",
        max_steps=1500,
        val_check_steps=100,
        early_stop_patience_steps=5,
        batch_size=256,
        windows_batch_size=1024,
        random_seed=42,
        enable_progress_bar=True,
        logger=False,
    )

    nf = NeuralForecast(models=[model], freq="H")

    # rolling CV over the last test segment, with validation segment reserved internally
    cv_df = nf.cross_validation(
        df=df,
        val_size=VAL_SIZE,
        test_size=TEST_SIZE,
        step_size=1,
        n_windows=None,
        refit=False,
        verbose=1,
    )

    pred_col = [c for c in cv_df.columns if c not in {"unique_id", "ds", "cutoff", "y"}][0]

    # align naive baseline: previous observed return
    full = df[["ds", "y"]].copy()
    full["y_prev"] = full["y"].shift(1)
    cv_df = cv_df.merge(full[["ds", "y_prev"]], on="ds", how="left")

    # split by cutoff into val-like and test-like regions
    # NeuralForecast returns predictions for the held-out region; we keep one metrics block.
    y_true = cv_df["y"].to_numpy(dtype=float)
    y_pred = cv_df[pred_col].to_numpy(dtype=float)
    y_base = cv_df["y_prev"].to_numpy(dtype=float)

    result = {
        "data_csv": str(DATA_CSV),
        "model": "PatchTST",
        "h": H,
        "input_size": INPUT_SIZE,
        "hist_exog": HIST_EXOG,
        "futr_exog": FUTR_EXOG,
        "n_predictions": int(len(cv_df)),
        "ts_range": {
            "min": str(cv_df["ds"].min()),
            "max": str(cv_df["ds"].max()),
        },
        "baseline": regression_metrics(y_true, y_base),
        "patchtst": regression_metrics(y_true, y_pred),
    }

    cv_df.to_csv(OUT_DIR / "patchtst_cv_predictions.csv", index=False)
    (OUT_DIR / "patchtst_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()