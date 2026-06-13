from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

@dataclass
class PredictionMetrics:
    rmse: float
    mae: float
    nrmse: float
    mae_over_std: float
    r2: float
    pearson_corr: float
    spearman_corr: float
    directional_accuracy: float
    top20_directional_accuracy: float
    sign_strategy_mean: float
    sign_strategy_sharpe: float

    def to_dict(self) -> dict[str, float]:
        return self.__dict__.copy()

def _safe_corr(y_true: np.ndarray, y_pred: np.ndarray, method: str='pearson') -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float('nan')
    if method == 'pearson':
        return float(np.corrcoef(y_true, y_pred)[0, 1])
    if method == 'spearman':
        order_true = np.argsort(np.argsort(y_true))
        order_pred = np.argsort(np.argsort(y_pred))
        return float(np.corrcoef(order_true, order_pred)[0, 1])
    raise ValueError(f'Unsupported correlation method: {method}')

def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    std = float(np.std(y_true)) if np.std(y_true) > 0 else float('nan')
    sign_true = np.sign(y_true)
    sign_pred = np.sign(y_pred)
    nz = (sign_true != 0) & (sign_pred != 0)
    directional_accuracy = float((sign_true[nz] == sign_pred[nz]).mean()) if nz.any() else float('nan')
    abs_pred = np.abs(y_pred)
    cutoff = np.nanquantile(abs_pred, 0.8) if len(abs_pred) else float('nan')
    hi = abs_pred >= cutoff
    top20_directional_accuracy = float((sign_true[hi] == sign_pred[hi]).mean()) if hi.any() else float('nan')
    pnl = np.sign(y_pred) * y_true
    pnl_std = np.std(pnl)
    sign_strategy_sharpe = float(np.mean(pnl) / pnl_std * np.sqrt(24 * 365)) if pnl_std > 0 else float('nan')
    metrics = PredictionMetrics(rmse=rmse, mae=mae, nrmse=float(rmse / std) if np.isfinite(std) and std > 0 else float('nan'), mae_over_std=float(mae / std) if np.isfinite(std) and std > 0 else float('nan'), r2=float(r2_score(y_true, y_pred)), pearson_corr=_safe_corr(y_true, y_pred, method='pearson'), spearman_corr=_safe_corr(y_true, y_pred, method='spearman'), directional_accuracy=directional_accuracy, top20_directional_accuracy=top20_directional_accuracy, sign_strategy_mean=float(np.mean(pnl)), sign_strategy_sharpe=sign_strategy_sharpe)
    return metrics.to_dict()