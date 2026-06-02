from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
DATA_CSV = Path('outputs/datasets/btcusdt_core_vol_small.csv')
OUT_DIR = Path('outputs/models/har_vol_small')
OUT_DIR.mkdir(parents=True, exist_ok=True)

@dataclass
class SplitConfig:
    train_frac: float = 0.7
    val_frac: float = 0.15
    test_frac: float = 0.15

def time_split(df: pd.DataFrame, cfg: SplitConfig):
    df = df.sort_values('ts').reset_index(drop=True)
    n = len(df)
    n_train = int(n * cfg.train_frac)
    n_val = int(n * cfg.val_frac)
    n_test = n - n_train - n_val
    return (df.iloc[:n_train].copy(), df.iloc[n_train:n_train + n_val].copy(), df.iloc[n_train + n_val:].copy())

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    std = float(np.std(y_true)) if np.std(y_true) > 0 else np.nan
    pearson = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else np.nan
    spearman = float(pd.Series(y_true).corr(pd.Series(y_pred), method='spearman'))
    return {'rmse': rmse, 'mae': mae, 'nrmse': rmse / std if std and (not np.isnan(std)) else np.nan, 'mae_over_std': mae / std if std and (not np.isnan(std)) else np.nan, 'r2': float(r2_score(y_true, y_pred)), 'pearson_corr': pearson, 'spearman_corr': spearman}

def main() -> None:
    df = pd.read_csv(DATA_CSV, parse_dates=['ts']).sort_values('ts').reset_index(drop=True)
    df['target_abs_ret_1h'] = df['abs_ret'].shift(-1)
    feature_cols = ['abs_ret', 'rv_6', 'rv_24', 'rv_72', 'volume', 'num_trades', 'fundingRate', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos']
    keep = ['ts', 'target_abs_ret_1h'] + feature_cols
    df = df[keep].replace([np.inf, -np.inf], np.nan)
    if 'fundingRate' in df.columns:
        df['fundingRate'] = df['fundingRate'].ffill()
    df = df.dropna().reset_index(drop=True)
    split_cfg = SplitConfig()
    df_train, df_val, df_test = time_split(df, split_cfg)
    X_train = df_train[feature_cols]
    y_train = df_train['target_abs_ret_1h'].to_numpy(dtype=float)
    X_val = df_val[feature_cols]
    y_val = df_val['target_abs_ret_1h'].to_numpy(dtype=float)
    X_test = df_test[feature_cols]
    y_test = df_test['target_abs_ret_1h'].to_numpy(dtype=float)
    y_val_baseline = df_val['abs_ret'].to_numpy(dtype=float)
    y_test_baseline = df_test['abs_ret'].to_numpy(dtype=float)
    candidates = {'linear': Pipeline([('imputer', SimpleImputer(strategy='median')), ('model', LinearRegression())]), 'ridge_0.1': Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler()), ('model', Ridge(alpha=0.1))]), 'ridge_1.0': Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler()), ('model', Ridge(alpha=1.0))]), 'ridge_10.0': Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler()), ('model', Ridge(alpha=10.0))])}
    best_name = None
    best_model = None
    best_val_rmse = np.inf
    search_rows = []
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        pred_val = model.predict(X_val)
        val_rmse = float(np.sqrt(mean_squared_error(y_val, pred_val)))
        search_rows.append({'model': name, 'val_rmse': val_rmse})
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_name = name
            best_model = model
    pred_val = best_model.predict(X_val)
    pred_test = best_model.predict(X_test)
    result = {'data_csv': str(DATA_CSV), 'best_model': best_name, 'split': asdict(split_cfg), 'feature_cols': feature_cols, 'n_rows': {'train': int(len(df_train)), 'val': int(len(df_val)), 'test': int(len(df_test))}, 'baseline': {'val': regression_metrics(y_val, y_val_baseline), 'test': regression_metrics(y_test, y_test_baseline)}, 'model': {'val': regression_metrics(y_val, pred_val), 'test': regression_metrics(y_test, pred_test)}, 'ts_range': {'train': {'min': str(df_train['ts'].min()), 'max': str(df_train['ts'].max())}, 'val': {'min': str(df_val['ts'].min()), 'max': str(df_val['ts'].max())}, 'test': {'min': str(df_test['ts'].min()), 'max': str(df_test['ts'].max())}}}
    pd.DataFrame(search_rows).sort_values('val_rmse').to_csv(OUT_DIR / 'har_search_results.csv', index=False)
    pd.DataFrame({'ts': df_test['ts'], 'y_true': y_test, 'y_pred': pred_test, 'y_pred_baseline': y_test_baseline}).to_csv(OUT_DIR / 'har_test_predictions.csv', index=False)
    (OUT_DIR / 'har_metrics.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
if __name__ == '__main__':
    main()
