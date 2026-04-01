from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from arch import arch_model


DATA_CSV = Path("outputs/datasets/btcusdt_core_vol_small.csv")
OUT_DIR = Path("outputs/models/garch_vol_small")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SCALE = 100.0  # helps optimization stability


@dataclass
class SplitConfig:
    train_frac: float = 0.7
    val_frac: float = 0.15
    test_frac: float = 0.15


def time_split(df: pd.DataFrame, cfg: SplitConfig):
    df = df.sort_values("ts").reset_index(drop=True)
    n = len(df)
    n_train = int(n * cfg.train_frac)
    n_val = int(n * cfg.val_frac)
    n_test = n - n_train - n_val
    if min(n_train, n_val, n_test) <= 0:
        raise ValueError("split too small")
    return (
        df.iloc[:n_train].copy(),
        df.iloc[n_train:n_train + n_val].copy(),
        df.iloc[n_train + n_val:].copy(),
    )


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    std = float(np.std(y_true)) if np.std(y_true) > 0 else np.nan

    pearson = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else np.nan
    spearman = float(pd.Series(y_true).corr(pd.Series(y_pred), method="spearman"))

    return {
        "rmse": rmse,
        "mae": mae,
        "nrmse": rmse / std if std and not np.isnan(std) else np.nan,
        "mae_over_std": mae / std if std and not np.isnan(std) else np.nan,
        "r2": float(r2_score(y_true, y_pred)),
        "pearson_corr": pearson,
        "spearman_corr": spearman,
    }


def make_arch_model(kind: str, y_scaled: pd.Series):
    kind = kind.lower()
    if kind == "garch":
        return arch_model(
            y_scaled,
            mean="Zero",
            vol="GARCH",
            p=1,
            o=0,
            q=1,
            dist="normal",
            rescale=False,
        )
    if kind == "gjr":
        return arch_model(
            y_scaled,
            mean="Zero",
            vol="GARCH",
            p=1,
            o=1,
            q=1,
            dist="normal",
            rescale=False,
        )
    if kind == "egarch":
        return arch_model(
            y_scaled,
            mean="Zero",
            vol="EGARCH",
            p=1,
            o=1,
            q=1,
            dist="normal",
            rescale=False,
        )
    raise ValueError(f"Unsupported model kind: {kind}")


def fit_model(kind: str, train_returns_scaled: pd.Series):
    am = make_arch_model(kind, train_returns_scaled)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = am.fit(disp="off", update_freq=0, show_warning=False)
    return res


def get_param(params: pd.Series, name: str, default: float = 0.0) -> float:
    return float(params[name]) if name in params.index else float(default)


def next_variance(kind: str, params: pd.Series, prev_eps: float, prev_sigma2: float) -> float:
    eps = 1e-12
    prev_sigma2 = max(float(prev_sigma2), eps)

    omega = get_param(params, "omega")
    alpha = get_param(params, "alpha[1]")
    beta = get_param(params, "beta[1]")
    gamma = get_param(params, "gamma[1]", 0.0)

    if kind == "garch":
        sigma2_next = omega + alpha * (prev_eps ** 2) + beta * prev_sigma2

    elif kind == "gjr":
        indicator = 1.0 if prev_eps < 0 else 0.0
        sigma2_next = omega + (alpha + gamma * indicator) * (prev_eps ** 2) + beta * prev_sigma2

    elif kind == "egarch":
        sigma_prev = np.sqrt(prev_sigma2)
        z_prev = prev_eps / max(sigma_prev, eps)
        expected_abs_z = np.sqrt(2.0 / np.pi)  # normal innovation
        log_sigma2_next = (
            omega
            + beta * np.log(prev_sigma2)
            + alpha * (abs(z_prev) - expected_abs_z)
            + gamma * z_prev
        )
        sigma2_next = float(np.exp(log_sigma2_next))
    else:
        raise ValueError(f"Unsupported model kind: {kind}")

    return max(float(sigma2_next), eps)


def forecast_sigma_path(
    kind: str,
    params: pd.Series,
    prev_eps: float,
    prev_sigma2: float,
    eval_returns_scaled: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    preds = []

    for r_t in eval_returns_scaled:
        sigma2_t = next_variance(kind, params, prev_eps=prev_eps, prev_sigma2=prev_sigma2)
        preds.append(np.sqrt(sigma2_t) / SCALE)

        # after observing current return, move state forward
        prev_sigma2 = sigma2_t
        prev_eps = float(r_t)

    return np.asarray(preds, dtype=float), float(prev_eps), float(prev_sigma2)


def previous_abs_baseline(train_or_prev_last: float, eval_returns_unscaled: np.ndarray) -> np.ndarray:
    if len(eval_returns_unscaled) == 0:
        return np.asarray([], dtype=float)
    out = np.empty(len(eval_returns_unscaled), dtype=float)
    out[0] = abs(train_or_prev_last)
    if len(eval_returns_unscaled) > 1:
        out[1:] = np.abs(eval_returns_unscaled[:-1])
    return out


def main() -> None:
    df = pd.read_csv(DATA_CSV, parse_dates=["ts"]).sort_values("ts").reset_index(drop=True)
    df = df[["ts", "log_ret_1h", "abs_ret"]].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    split_cfg = SplitConfig()
    df_train, df_val, df_test = time_split(df, split_cfg)

    train_ret = df_train["log_ret_1h"].to_numpy(dtype=float)
    val_ret = df_val["log_ret_1h"].to_numpy(dtype=float)
    test_ret = df_test["log_ret_1h"].to_numpy(dtype=float)

    train_scaled = pd.Series(train_ret * SCALE)
    val_scaled = val_ret * SCALE
    test_scaled = test_ret * SCALE

    y_val = np.abs(val_ret)
    y_test = np.abs(test_ret)

    baseline_val = previous_abs_baseline(train_ret[-1], val_ret)
    baseline_test = previous_abs_baseline(np.concatenate([train_ret, val_ret])[-1], test_ret)

    candidates = ["garch", "gjr", "egarch"]
    search_rows = []
    best = None
    best_val_rmse = np.inf

    for kind in candidates:
        try:
            res = fit_model(kind, train_scaled)
            params = res.params.copy()
            last_sigma2 = float(res.conditional_volatility.iloc[-1] ** 2)
            last_eps = float(train_scaled.iloc[-1])

            pred_val, _, _ = forecast_sigma_path(
                kind=kind,
                params=params,
                prev_eps=last_eps,
                prev_sigma2=last_sigma2,
                eval_returns_scaled=val_scaled,
            )

            val_rmse = float(np.sqrt(mean_squared_error(y_val, pred_val)))
            row = {
                "model_kind": kind,
                "val_rmse": val_rmse,
                "val_mae": float(mean_absolute_error(y_val, pred_val)),
            }
            search_rows.append(row)

            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best = {
                    "kind": kind,
                    "val_pred": pred_val,
                }
                print(f"new best: {kind} val_rmse={val_rmse:.8f}")

        except Exception as e:
            print("skip", kind, "->", repr(e))

    if best is None:
        raise RuntimeError("No GARCH-family model fit succeeded")

    # Refit best on train+val and forecast test
    trainval_ret = np.concatenate([train_ret, val_ret])
    trainval_scaled = pd.Series(trainval_ret * SCALE)

    best_res = fit_model(best["kind"], trainval_scaled)
    best_params = best_res.params.copy()
    best_last_sigma2 = float(best_res.conditional_volatility.iloc[-1] ** 2)
    best_last_eps = float(trainval_scaled.iloc[-1])

    pred_test, _, _ = forecast_sigma_path(
        kind=best["kind"],
        params=best_params,
        prev_eps=best_last_eps,
        prev_sigma2=best_last_sigma2,
        eval_returns_scaled=test_scaled,
    )

    result = {
        "data_csv": str(DATA_CSV),
        "best_model": best["kind"],
        "split": asdict(split_cfg),
        "n_rows": {
            "train": int(len(df_train)),
            "val": int(len(df_val)),
            "test": int(len(df_test)),
        },
        "baseline": {
            "val": regression_metrics(y_val, baseline_val),
            "test": regression_metrics(y_test, baseline_test),
        },
        "model": {
            "val": regression_metrics(y_val, best["val_pred"]),
            "test": regression_metrics(y_test, pred_test),
        },
        "ts_range": {
            "train": {"min": str(df_train["ts"].min()), "max": str(df_train["ts"].max())},
            "val": {"min": str(df_val["ts"].min()), "max": str(df_val["ts"].max())},
            "test": {"min": str(df_test["ts"].min()), "max": str(df_test["ts"].max())},
        },
    }

    pd.DataFrame(search_rows).sort_values("val_rmse").to_csv(
        OUT_DIR / "garch_search_results.csv", index=False
    )
    pd.DataFrame(
        {
            "ts": df_test["ts"],
            "y_true_abs_ret": y_test,
            "y_pred_sigma": pred_test,
            "y_pred_baseline": baseline_test,
        }
    ).to_csv(OUT_DIR / "garch_test_predictions.csv", index=False)

    (OUT_DIR / "garch_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()