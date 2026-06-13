from __future__ import annotations
from pathlib import Path
import copy
import itertools
import json
import numpy as np
import pandas as pd
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import RMSE
from pytorch_forecasting.models.timexer import TimeXer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
DATA_CSV = Path('outputs/datasets/btcusdt_timexer.csv')
OUT_DIR = Path('outputs/models/timexer_core_tuned')
OUT_DIR.mkdir(parents=True, exist_ok=True)
ENC_LEN = 168
PRED_LEN = 1
BATCH_SIZE = 256
ACCUMULATE = 4
MAX_EPOCHS = 40

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

def make_datasets(df: pd.DataFrame):
    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    training = TimeSeriesDataSet(df[df.time_idx < train_end], time_idx='time_idx', target='target', group_ids=['series_id'], max_encoder_length=ENC_LEN, max_prediction_length=PRED_LEN, static_categoricals=[], static_reals=[], time_varying_known_reals=['time_idx', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos'], time_varying_unknown_reals=['target', 'log_ret_1h', 'premium_chg', 'fundingRate', 'vol_chg', 'trades_chg', 'taker_buy_share', 'rv_24', 'fundingRate_missing', 'taker_buy_share_missing'], target_normalizer=GroupNormalizer(groups=['series_id']), add_relative_time_idx=True, add_target_scales=True, add_encoder_length=True, allow_missing_timesteps=False)
    validation = TimeSeriesDataSet.from_dataset(training, df[df.time_idx < val_end], min_prediction_idx=train_end, stop_randomization=True)
    test = TimeSeriesDataSet.from_dataset(training, df, min_prediction_idx=val_end, stop_randomization=True)
    return (training, validation, test, train_end, val_end, n)

def main() -> None:
    seed_everything(42, workers=True)
    df = pd.read_csv(DATA_CSV, parse_dates=['ds']).sort_values('time_idx').reset_index(drop=True)
    training, validation, test, train_end, val_end, n = make_datasets(df)
    train_loader = training.to_dataloader(train=True, batch_size=BATCH_SIZE, num_workers=4)
    val_loader = validation.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=4)
    test_loader = test.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=4)
    search_space = [{'learning_rate': 0.0003, 'hidden_size': 64, 'n_heads': 4, 'e_layers': 2, 'd_ff': 256, 'dropout': 0.2, 'patch_length': 24}, {'learning_rate': 0.001, 'hidden_size': 128, 'n_heads': 4, 'e_layers': 2, 'd_ff': 512, 'dropout': 0.2, 'patch_length': 24}, {'learning_rate': 0.0003, 'hidden_size': 128, 'n_heads': 8, 'e_layers': 3, 'd_ff': 512, 'dropout': 0.2, 'patch_length': 24}, {'learning_rate': 0.001, 'hidden_size': 256, 'n_heads': 8, 'e_layers': 2, 'd_ff': 1024, 'dropout': 0.1, 'patch_length': 24}, {'learning_rate': 0.0003, 'hidden_size': 128, 'n_heads': 4, 'e_layers': 2, 'd_ff': 512, 'dropout': 0.3, 'patch_length': 12}, {'learning_rate': 0.0003, 'hidden_size': 128, 'n_heads': 4, 'e_layers': 2, 'd_ff': 512, 'dropout': 0.2, 'patch_length': 48}]
    best = None
    best_val_rmse = np.inf
    search_rows = []
    for i, cfg in enumerate(search_space, start=1):
        run_dir = OUT_DIR / f'run_{i:02d}'
        run_dir.mkdir(parents=True, exist_ok=True)
        early_stop = EarlyStopping(monitor='val_loss', patience=8, mode='min')
        ckpt = ModelCheckpoint(dirpath=run_dir, filename='best', monitor='val_loss', mode='min', save_top_k=1)
        logger = CSVLogger(save_dir=str(run_dir), name='logs')
        trainer = Trainer(max_epochs=40, accelerator='gpu' if torch.cuda.is_available() else 'cpu', devices=1, accumulate_grad_batches=ACCUMULATE, gradient_clip_val=0.5, gradient_clip_algorithm='norm', callbacks=[early_stop, ckpt], logger=logger, enable_model_summary=False, deterministic=False)
        model = TimeXer.from_dataset(training, loss=RMSE(), learning_rate=cfg['learning_rate'], context_length=ENC_LEN, prediction_length=PRED_LEN, hidden_size=cfg['hidden_size'], n_heads=cfg['n_heads'], e_layers=cfg['e_layers'], d_ff=cfg['d_ff'], dropout=cfg['dropout'], patch_length=cfg['patch_length'], optimizer='adam', optimizer_params={}, weight_decay=0.0001, reduce_on_plateau_patience=3, reduce_on_plateau_reduction=2.0, reduce_on_plateau_min_lr=1e-06)
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        best_model = TimeXer.load_from_checkpoint(ckpt.best_model_path)
        val_raw = best_model.predict(val_loader, return_y=True)
        y_val = val_raw.y[0].detach().cpu().numpy().reshape(-1)
        p_val = val_raw.output.detach().cpu().numpy().reshape(-1)
        val_metrics = regression_metrics(y_val, p_val)
        search_rows.append({**cfg, **{f'val_{k}': v for k, v in val_metrics.items()}})
        print('run', i, cfg, 'val_rmse', val_metrics['rmse'], 'val_dir_acc', val_metrics['directional_accuracy'])
        if val_metrics['rmse'] < best_val_rmse:
            test_raw = best_model.predict(test_loader, return_y=True)
            y_test = test_raw.y[0].detach().cpu().numpy().reshape(-1)
            p_test = test_raw.output.detach().cpu().numpy().reshape(-1)
            baseline_val = df.iloc[train_end:val_end]['log_ret_1h'].to_numpy(dtype=float)[:len(y_val)]
            baseline_test = df.iloc[val_end:]['log_ret_1h'].to_numpy(dtype=float)[:len(y_test)]
            best_val_rmse = val_metrics['rmse']
            best = {'cfg': copy.deepcopy(cfg), 'val_metrics': val_metrics, 'test_metrics': regression_metrics(y_test, p_test), 'baseline_val': regression_metrics(y_val, baseline_val), 'baseline_test': regression_metrics(y_test, baseline_test), 'y_test': y_test, 'p_test': p_test, 'baseline_test_pred': baseline_test}
    pd.DataFrame(search_rows).sort_values('val_rmse').to_csv(OUT_DIR / 'timexer_search_results.csv', index=False)
    pd.DataFrame({'y_true': best['y_test'], 'y_pred': best['p_test'], 'y_pred_baseline': best['baseline_test_pred']}).to_csv(OUT_DIR / 'timexer_test_predictions.csv', index=False)
    result = {'data_csv': str(DATA_CSV), 'model': 'TimeXer', 'encoder_length': ENC_LEN, 'prediction_length': PRED_LEN, 'batch_size': BATCH_SIZE, 'accumulate_grad_batches': ACCUMULATE, 'n_rows': {'total': int(n), 'train': int(train_end), 'val': int(val_end - train_end), 'test': int(n - val_end)}, 'best_params': best['cfg'], 'baseline': {'val': best['baseline_val'], 'test': best['baseline_test']}, 'timexer': {'val': best['val_metrics'], 'test': best['test_metrics']}}
    (OUT_DIR / 'timexer_metrics.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
if __name__ == '__main__':
    main()
