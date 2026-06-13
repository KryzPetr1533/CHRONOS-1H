from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
import itertools
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
DATA_CSV = Path('outputs/datasets/btcusdt_core_mean_small.csv')
OUT_DIR = Path('outputs/models/sarimax_mean_small')
OUT_DIR.mkdir(parents=True, exist_ok=True)

@dataclass
class SplitConfig:
    train_frac: float = 0.7
    val_frac: float = 0.15
    test_frac: float = 0.15

def time_split(df: pd.DataFrame, cfg: SplitConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.sort_values('ts').reset_index(drop=True)
    n = len(df)
    n_train = int(n * cfg.train_frac)
    n_val = int(n * cfg.val_frac)
    n_test = n - n_train - n_val
    if min(n_train, n_val, n_test) <= 0:
        raise ValueError('split too small')
    return (df.iloc[:n_train].copy(), df.iloc[n_train:n_train + n_val].copy(), df.iloc[n_train + n_val:].copy())

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    std = float(np.std(y_true)) if np.std(y_true) > 0 else np.nan
    pearson = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else np.nan
    try:
        spearman = float(pd.Series(y_true).corr(pd.Series(y_pred), method='spearman'))
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
    return {'rmse': rmse, 'mae': mae, 'nrmse': rmse / std if std and (not np.isnan(std)) else np.nan, 'mae_over_std': mae / std if std and (not np.isnan(std)) else np.nan, 'r2': float(r2_score(y_true, y_pred)), 'pearson_corr': pearson, 'spearman_corr': spearman, 'directional_accuracy': dir_acc, 'top20_directional_accuracy': top_dir_acc, 'sign_strategy_mean': strat_mean, 'sign_strategy_sharpe': strat_sharpe}

def fit_scaler(train_df: pd.DataFrame, exog_cols: list[str]) -> StandardScaler | None:
    if not exog_cols:
        return None
    scaler = StandardScaler()
    scaler.fit(train_df[exog_cols])
    return scaler

def transform_exog(df: pd.DataFrame, exog_cols: list[str], scaler: StandardScaler | None) -> pd.DataFrame | None:
    if not exog_cols:
        return None
    arr = scaler.transform(df[exog_cols]) if scaler is not None else df[exog_cols].to_numpy()
    return pd.DataFrame(arr, columns=exog_cols, index=df.index)

def fit_and_forecast(train_df: pd.DataFrame, future_df: pd.DataFrame, order: tuple[int, int, int], trend: str, exog_cols: list[str]) -> np.ndarray:
    scaler = fit_scaler(train_df, exog_cols)
    exog_train = transform_exog(train_df, exog_cols, scaler)
    exog_future = transform_exog(future_df, exog_cols, scaler)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model = SARIMAX(endog=train_df['target_log_ret_1h'].astype(float), exog=exog_train, order=order, trend=trend, enforce_stationarity=False, enforce_invertibility=False)
        res = model.fit(disp=False)
    fc = res.get_forecast(steps=len(future_df), exog=exog_future)
    return np.asarray(fc.predicted_mean, dtype=float)

def main() -> None:
    df = pd.read_csv(DATA_CSV, parse_dates=['ts']).sort_values('ts').reset_index(drop=True)
    exog_cols = ['premium_chg', 'fundingRate', 'vol_chg', 'trades_chg', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'fund_cycle_sin', 'fund_cycle_cos']
    keep = ['ts', 'target_log_ret_1h', 'log_ret_1h'] + exog_cols
    df = df[keep].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    split_cfg = SplitConfig()
    df_train, df_val, df_test = time_split(df, split_cfg)
    y_val = df_val['target_log_ret_1h'].to_numpy(dtype=float)
    y_test = df_test['target_log_ret_1h'].to_numpy(dtype=float)
    baseline_val = df_val['log_ret_1h'].to_numpy(dtype=float)
    baseline_test = df_test['log_ret_1h'].to_numpy(dtype=float)
    orders = [(p, 0, q) for p, q in itertools.product([0, 1, 2, 3], [0, 1, 2])]
    trends = ['n', 'c']
    candidate_sets = {'ar_only': [], 'arx_exog': exog_cols}
    search_rows = []
    best = None
    best_score = np.inf
    for model_kind, cols in candidate_sets.items():
        for order in orders:
            if order == (0, 0, 0):
                continue
            for trend in trends:
                try:
                    pred_val = fit_and_forecast(train_df=df_train, future_df=df_val, order=order, trend=trend, exog_cols=cols)
                    rmse = float(np.sqrt(mean_squared_error(y_val, pred_val)))
                    row = {'model_kind': model_kind, 'order': order, 'trend': trend, 'val_rmse': rmse}
                    search_rows.append(row)
                    if rmse < best_score:
                        best_score = rmse
                        best = {'model_kind': model_kind, 'order': order, 'trend': trend, 'exog_cols': cols, 'val_pred': pred_val}
                        print('new best:', best['model_kind'], best['order'], best['trend'], 'rmse=', rmse)
                except Exception as e:
                    print('skip', model_kind, order, trend, '->', repr(e))
    if best is None:
        raise RuntimeError('No SARIMAX model fit succeeded')
    df_trainval = pd.concat([df_train, df_val], axis=0).reset_index(drop=True)
    pred_test = fit_and_forecast(train_df=df_trainval, future_df=df_test, order=best['order'], trend=best['trend'], exog_cols=best['exog_cols'])
    result = {'data_csv': str(DATA_CSV), 'best_model': {'model_kind': best['model_kind'], 'order': list(best['order']), 'trend': best['trend'], 'exog_cols': best['exog_cols']}, 'split': asdict(split_cfg), 'n_rows': {'train': int(len(df_train)), 'val': int(len(df_val)), 'test': int(len(df_test))}, 'ts_range': {'train': {'min': str(df_train['ts'].min()), 'max': str(df_train['ts'].max())}, 'val': {'min': str(df_val['ts'].min()), 'max': str(df_val['ts'].max())}, 'test': {'min': str(df_test['ts'].min()), 'max': str(df_test['ts'].max())}}, 'baseline': {'val': regression_metrics(y_val, baseline_val), 'test': regression_metrics(y_test, baseline_test)}, 'model': {'val': regression_metrics(y_val, best['val_pred']), 'test': regression_metrics(y_test, pred_test)}}
    pd.DataFrame(search_rows).sort_values('val_rmse').to_csv(OUT_DIR / 'sarimax_search_results.csv', index=False)
    pd.DataFrame({'ts': df_test['ts'], 'y_true': y_test, 'y_pred': pred_test, 'y_pred_baseline': baseline_test}).to_csv(OUT_DIR / 'sarimax_test_predictions.csv', index=False)
    (OUT_DIR / 'sarimax_metrics.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
if __name__ == '__main__':
    main()
