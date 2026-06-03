from __future__ import annotations
import copy
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = REPO_ROOT / 'outputs/datasets/btcusdt_core_seq_small.csv'
DEFAULT_OUT = REPO_ROOT / 'outputs/models/seq_core_small'
SEARCH_FULL = [{'model_kind': 'gru', 'lookback': 48, 'hidden_dim': 32, 'dropout': 0.0}, {'model_kind': 'gru', 'lookback': 168, 'hidden_dim': 32, 'dropout': 0.0}, {'model_kind': 'lstm', 'lookback': 48, 'hidden_dim': 32, 'dropout': 0.0}, {'model_kind': 'lstm', 'lookback': 168, 'hidden_dim': 32, 'dropout': 0.0}]
SEARCH_QUICK = [{'model_kind': 'gru', 'lookback': 48, 'hidden_dim': 32, 'dropout': 0.0}]

def set_seed(seed: int=42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

@dataclass
class SplitConfig:
    train_frac: float = 0.7
    val_frac: float = 0.15
    test_frac: float = 0.15

@dataclass
class TrainConfig:
    epochs: int = 25
    batch_size: int = 256
    lr: float = 0.001
    weight_decay: float = 1e-05
    patience: int = 5

class SeqDataset(Dataset):

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (self.X[idx], self.y[idx])

class RNNRegressor(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, model_kind: str='gru', dropout: float=0.0):
        super().__init__()
        self.model_kind = model_kind.lower()
        if self.model_kind == 'gru':
            self.rnn = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True, dropout=0.0)
        elif self.model_kind == 'lstm':
            self.rnn = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True, dropout=0.0)
        else:
            raise ValueError(f'Unsupported model_kind: {model_kind}')
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, x):
        out, _ = self.rnn(x)
        last = out[:, -1, :]
        pred = self.head(last).squeeze(-1)
        return pred

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

def split_indices(n: int, cfg: SplitConfig):
    n_train = int(n * cfg.train_frac)
    n_val = int(n * cfg.val_frac)
    n_test = n - n_train - n_val
    if min(n_train, n_val, n_test) <= 0:
        raise ValueError('split too small')
    train_end = n_train
    val_end = n_train + n_val
    return (train_end, val_end)

def make_positions(n: int, train_end: int, val_end: int, lookback: int):
    start = lookback - 1
    train_pos = np.arange(start, train_end)
    val_pos = np.arange(max(train_end, start), val_end)
    test_pos = np.arange(max(val_end, start), n)
    return (train_pos, val_pos, test_pos)

def build_seq_arrays(features_2d: np.ndarray, targets_1d: np.ndarray, positions: np.ndarray, lookback: int):
    X, y = ([], [])
    for pos in positions:
        left = pos - lookback + 1
        if left < 0:
            continue
        window = features_2d[left:pos + 1]
        if window.shape[0] != lookback:
            continue
        target = targets_1d[pos]
        if np.isnan(window).any() or np.isnan(target):
            continue
        X.append(window)
        y.append(target)
    return (np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32))

def train_one_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, y_scaler: StandardScaler, device: torch.device, cfg: TrainConfig):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best_state = None
    best_val_rmse = np.inf
    best_val_pred = None
    epochs_without_improve = 0
    for epoch in range(cfg.epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
        model.eval()
        val_preds, val_true = ([], [])
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                pred = model(xb).cpu().numpy()
                val_preds.append(pred)
                val_true.append(yb.numpy())
        val_pred_scaled = np.concatenate(val_preds)
        val_true_scaled = np.concatenate(val_true)
        val_pred = y_scaler.inverse_transform(val_pred_scaled.reshape(-1, 1)).ravel()
        val_true_orig = y_scaler.inverse_transform(val_true_scaled.reshape(-1, 1)).ravel()
        val_rmse = float(np.sqrt(mean_squared_error(val_true_orig, val_pred)))
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = copy.deepcopy(model.state_dict())
            best_val_pred = val_pred.copy()
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
        print(f'epoch={epoch + 1:02d} val_rmse={val_rmse:.8f}')
        if epochs_without_improve >= cfg.patience:
            break
    model.load_state_dict(best_state)
    return (model, best_val_rmse, best_val_pred)

def predict_model(model: nn.Module, loader: DataLoader, y_scaler: StandardScaler, device: torch.device) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy()
            preds.append(pred)
    pred_scaled = np.concatenate(preds)
    return y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()

def fit_feature_scaler(df: pd.DataFrame, feature_cols: list[str], train_end: int) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(df.iloc[:train_end][feature_cols])
    return scaler

def fit_target_scaler(df: pd.DataFrame, target_col: str, train_positions: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(df.iloc[train_positions][[target_col]])
    return scaler

def train_seq_regression(data_csv: Path | str=DEFAULT_DATA, out_dir: Path | str=DEFAULT_OUT, seed: int=42, epochs: int=25, patience: int=5, batch_size: int=256, quick: bool=False) -> dict:
    data_csv = Path(data_csv).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if not data_csv.is_file():
        raise FileNotFoundError(f'Missing dataset: {data_csv}. Run: make build-regression-datasets')
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)
    print('data:', data_csv)
    print('out:', out_dir)
    df = pd.read_csv(data_csv, parse_dates=['ts']).sort_values('ts').reset_index(drop=True)
    target_col = 'target_log_ret_1h'
    feature_cols = [c for c in df.columns if c not in ['ts', target_col]]
    df = df[['ts', target_col] + feature_cols].replace([np.inf, -np.inf], np.nan)
    if 'fundingRate' in df.columns:
        df['fundingRate'] = df['fundingRate'].ffill()
    df = df.dropna().reset_index(drop=True)
    n = len(df)
    split_cfg = SplitConfig()
    train_end, val_end = split_indices(n, split_cfg)
    search_space = SEARCH_QUICK if quick else SEARCH_FULL
    train_cfg = TrainConfig(epochs=epochs, patience=patience, batch_size=batch_size)
    best = None
    best_val_rmse = np.inf
    search_rows = []
    for cfg_row in search_space:
        lookback = cfg_row['lookback']
        train_pos, val_pos, test_pos = make_positions(n, train_end, val_end, lookback)
        if len(train_pos) == 0 or len(val_pos) == 0 or len(test_pos) == 0:
            continue
        x_scaler = fit_feature_scaler(df, feature_cols, train_end)
        y_scaler = fit_target_scaler(df, target_col, train_pos)
        X_all = x_scaler.transform(df[feature_cols])
        y_all = y_scaler.transform(df[[target_col]]).ravel()
        X_train, y_train = build_seq_arrays(X_all, y_all, train_pos, lookback)
        X_val, y_val = build_seq_arrays(X_all, y_all, val_pos, lookback)
        X_test, y_test = build_seq_arrays(X_all, y_all, test_pos, lookback)
        train_ds = SeqDataset(X_train, y_train)
        val_ds = SeqDataset(X_val, y_val)
        test_ds = SeqDataset(X_test, y_test)
        train_loader = DataLoader(train_ds, batch_size=train_cfg.batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_ds, batch_size=train_cfg.batch_size, shuffle=False, drop_last=False)
        test_loader = DataLoader(test_ds, batch_size=train_cfg.batch_size, shuffle=False, drop_last=False)
        model = RNNRegressor(input_dim=X_train.shape[-1], hidden_dim=cfg_row['hidden_dim'], model_kind=cfg_row['model_kind'], dropout=cfg_row['dropout']).to(device)
        print('\ntraining', cfg_row)
        model, val_rmse, val_pred = train_one_model(model=model, train_loader=train_loader, val_loader=val_loader, y_scaler=y_scaler, device=device, cfg=train_cfg)
        y_val_orig = y_scaler.inverse_transform(y_val.reshape(-1, 1)).ravel()
        val_metrics = regression_metrics(y_val_orig, val_pred)
        row = {**cfg_row, 'val_rmse': val_metrics['rmse'], 'val_pearson': val_metrics['pearson_corr'], 'val_dir_acc': val_metrics['directional_accuracy']}
        search_rows.append(row)
        if val_metrics['rmse'] < best_val_rmse:
            best_val_rmse = val_metrics['rmse']
            test_pred = predict_model(model, test_loader, y_scaler, device)
            y_test_orig = y_scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
            baseline_val = df.iloc[val_pos]['log_ret_1h'].to_numpy(dtype=float)
            baseline_test = df.iloc[test_pos]['log_ret_1h'].to_numpy(dtype=float)
            best = {'cfg': cfg_row, 'model_state': copy.deepcopy(model.state_dict()), 'x_scaler': x_scaler, 'y_scaler': y_scaler, 'feature_cols': feature_cols, 'lookback': lookback, 'val_positions': val_pos, 'test_positions': test_pos, 'val_true': y_val_orig, 'val_pred': val_pred, 'test_true': y_test_orig, 'test_pred': test_pred, 'baseline_val': baseline_val, 'baseline_test': baseline_test}
    if best is None:
        raise RuntimeError('No sequence model fit succeeded')
    result = {'data_csv': str(data_csv), 'best_model': best['cfg'], 'split': asdict(split_cfg), 'seed': seed, 'n_rows': {'total': int(len(df)), 'train_target_positions': int(train_end), 'val_target_positions': int(len(best['val_positions'])), 'test_target_positions': int(len(best['test_positions']))}, 'feature_cols': best['feature_cols'], 'baseline': {'val': regression_metrics(best['val_true'], best['baseline_val']), 'test': regression_metrics(best['test_true'], best['baseline_test'])}, 'model': {'val': regression_metrics(best['val_true'], best['val_pred']), 'test': regression_metrics(best['test_true'], best['test_pred'])}, 'ts_range': {'train': {'min': str(df.iloc[:train_end]['ts'].min()), 'max': str(df.iloc[:train_end]['ts'].max())}, 'val': {'min': str(df.iloc[train_end:val_end]['ts'].min()), 'max': str(df.iloc[train_end:val_end]['ts'].max())}, 'test': {'min': str(df.iloc[val_end:]['ts'].min()), 'max': str(df.iloc[val_end:]['ts'].max())}}}
    pd.DataFrame(search_rows).sort_values('val_rmse').to_csv(out_dir / 'seq_search_results.csv', index=False)
    preds_path = out_dir / 'seq_test_predictions.csv'
    pd.DataFrame({'ts': df.iloc[best['test_positions']]['ts'].to_numpy(), 'y_true': best['test_true'], 'y_pred': best['test_pred'], 'y_pred_baseline': best['baseline_test']}).to_csv(preds_path, index=False)
    metrics_path = out_dir / 'seq_metrics.json'
    metrics_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result
