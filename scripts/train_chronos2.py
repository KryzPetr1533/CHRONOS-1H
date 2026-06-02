from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
DATA_CSV = Path('outputs/datasets/chronos2_panel.csv')
OUT_DIR = Path('outputs/models/chronos2_core')
OUT_DIR.mkdir(parents=True, exist_ok=True)
PREDICTION_LENGTH = 1
KNOWN_COVS = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos']

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

def to_tsdf(df: pd.DataFrame) -> TimeSeriesDataFrame:
    return TimeSeriesDataFrame.from_data_frame(df, id_column='item_id', timestamp_column='timestamp')

def last_rows_per_item(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.groupby('item_id', group_keys=False).tail(n)

def flatten_backtests(pred_list, tgt_list, pred_col='mean'):
    rows = []
    for pred, tgt in zip(pred_list, tgt_list):
        pred_df = pred.reset_index()
        tgt_df = tgt.reset_index()
        tgt_tail = tgt_df.groupby('item_id', group_keys=False).tail(PREDICTION_LENGTH)
        merged = pred_df[['item_id', 'timestamp', pred_col]].merge(tgt_tail[['item_id', 'timestamp', 'target']], on=['item_id', 'timestamp'], how='inner')
        rows.append(merged)
    out = pd.concat(rows, ignore_index=True)
    out = out.sort_values(['item_id', 'timestamp']).reset_index(drop=True)
    return out

def main() -> None:
    df = pd.read_csv(DATA_CSV, parse_dates=['timestamp']).sort_values(['item_id', 'timestamp']).reset_index(drop=True)
    unique_ts = np.array(sorted(df['timestamp'].unique()))
    n_ts = len(unique_ts)
    train_end = int(n_ts * 0.7)
    val_end = int(n_ts * 0.85)
    train_cut = unique_ts[train_end - 1]
    val_cut = unique_ts[val_end - 1]
    train_df = df[df['timestamp'] <= train_cut].copy()
    full_df = df.copy()
    train_ts = to_tsdf(train_df)
    full_ts = to_tsdf(full_df)
    n_test_windows = min(256, int(n_ts - val_end))
    val_step_size = 24
    predictor = TimeSeriesPredictor(target='target', prediction_length=PREDICTION_LENGTH, known_covariates_names=KNOWN_COVS, eval_metric='RMSE', freq='h', path=str(OUT_DIR))
    predictor.fit(train_data=train_ts, hyperparameters={'Chronos2': [{'ag_args': {'name_suffix': 'SmallZeroShot'}, 'model_path': 'autogluon/chronos-2-small', 'context_length': 512, 'batch_size': 128, 'cross_learning': True}, {'ag_args': {'name_suffix': 'SmallFineTuned'}, 'model_path': 'autogluon/chronos-2-small', 'context_length': 512, 'batch_size': 128, 'cross_learning': True, 'fine_tune': True, 'fine_tune_mode': 'lora', 'fine_tune_lr': 1e-05, 'fine_tune_steps': 500, 'fine_tune_batch_size': 32, 'fine_tune_context_length': 512, 'eval_during_fine_tune': True, 'fine_tune_eval_max_items': 64}]}, enable_ensemble=False, num_val_windows=1, refit_full=False, verbosity=2)
    model_names = predictor.model_names()
    results = {}
    targets = predictor.backtest_targets(full_ts, num_val_windows=n_test_windows, val_step_size=val_step_size)
    for model_name in model_names:
        print(f'backtesting model: {model_name}')
        preds = predictor.backtest_predictions(full_ts, model=model_name, num_val_windows=n_test_windows, val_step_size=val_step_size, use_cache=False)
        merged = flatten_backtests(preds, targets, pred_col='mean')
        baseline = full_df[['item_id', 'timestamp', 'log_ret_1h']].copy()
        merged = merged.merge(baseline, on=['item_id', 'timestamp'], how='left')
        y_true = merged['target'].to_numpy(dtype=float)
        y_pred = merged['mean'].to_numpy(dtype=float)
        y_base = merged['log_ret_1h'].to_numpy(dtype=float)
        metrics = {'baseline': regression_metrics(y_true, y_base), 'model': regression_metrics(y_true, y_pred), 'n_predictions': int(len(merged))}
        results[model_name] = metrics
        merged.rename(columns={'mean': 'y_pred', 'target': 'y_true', 'log_ret_1h': 'y_pred_baseline'}).to_csv(OUT_DIR / f'{model_name}_backtest_predictions.csv', index=False)
    leaderboard = predictor.leaderboard(train_ts)
    leaderboard.to_csv(OUT_DIR / 'leaderboard.csv', index=False)
    final = {'data_csv': str(DATA_CSV), 'prediction_length': PREDICTION_LENGTH, 'n_items': int(df['item_id'].nunique()), 'n_rows': int(len(df)), 'n_unique_timestamps': int(n_ts), 'train_cutoff': str(train_cut), 'val_cutoff': str(val_cut), 'n_test_backtest_windows': int(n_test_windows), 'model_results': results, 'model_names': model_names}
    (OUT_DIR / 'chronos2_metrics.json').write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(final, ensure_ascii=False, indent=2))
if __name__ == '__main__':
    main()
