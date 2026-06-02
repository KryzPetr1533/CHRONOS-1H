from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
INPUT = Path('outputs/datasets/btcusdt_core_tabular.csv')
OUT_DIR = Path('outputs/datasets')
OUT_DIR.mkdir(parents=True, exist_ok=True)

def ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values('ts').reset_index(drop=True)
    if 'abs_ret' not in df.columns:
        df['abs_ret'] = df['log_ret_1h'].abs()
    if 'rv_6' not in df.columns:
        df['rv_6'] = df['log_ret_1h'].rolling(6).std()
    if 'rv_24' not in df.columns:
        df['rv_24'] = df['log_ret_1h'].rolling(24).std()
    if 'rv_72' not in df.columns:
        df['rv_72'] = df['log_ret_1h'].rolling(72).std()
    if 'premium_chg' not in df.columns and 'premium_close' in df.columns:
        df['premium_chg'] = df['premium_close'].diff()
    if 'vol_chg' not in df.columns and 'volume' in df.columns:
        df['vol_chg'] = np.log1p(df['volume']).diff()
    if 'trades_chg' not in df.columns and 'num_trades' in df.columns:
        df['trades_chg'] = np.log1p(df['num_trades']).diff()
    if 'taker_buy_share' not in df.columns and {'taker_buy_base', 'volume'}.issubset(df.columns):
        denom = df['volume'].replace(0, np.nan)
        df['taker_buy_share'] = df['taker_buy_base'] / denom
    return df

def save_dataset(df: pd.DataFrame, name: str, feature_cols: list[str]) -> None:
    keep = ['ts', 'target_log_ret_1h'] + feature_cols
    out = df[keep].copy()
    if 'fundingRate' in out.columns:
        out['fundingRate'] = out['fundingRate'].ffill()
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna().reset_index(drop=True)
    csv_path = OUT_DIR / f'{name}.csv'
    meta_path = OUT_DIR / f'{name}.meta.json'
    out.to_csv(csv_path, index=False)
    meta = {'name': name, 'n_rows': int(len(out)), 'columns': list(out.columns), 'ts_min': str(out['ts'].min()) if len(out) else None, 'ts_max': str(out['ts'].max()) if len(out) else None}
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'saved {csv_path} rows={len(out)}')

def main() -> None:
    df = pd.read_csv(INPUT, parse_dates=['ts'])
    df = ensure_cols(df)
    mean_small = ['log_ret_1h', 'premium_chg', 'fundingRate', 'vol_chg', 'trades_chg', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'fund_cycle_sin', 'fund_cycle_cos']
    vol_small = ['log_ret_1h', 'abs_ret', 'rv_6', 'rv_24', 'rv_72', 'volume', 'num_trades', 'fundingRate', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos']
    seq_small = ['log_ret_1h', 'premium_chg', 'fundingRate', 'vol_chg', 'trades_chg', 'taker_buy_share', 'rv_24', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos']
    for name, cols in [('btcusdt_core_mean_small', mean_small), ('btcusdt_core_vol_small', vol_small), ('btcusdt_core_seq_small', seq_small)]:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f'{name}: missing columns {missing}')
        save_dataset(df, name, cols)
if __name__ == '__main__':
    main()
