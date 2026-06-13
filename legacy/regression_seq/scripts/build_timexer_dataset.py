from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
INPUT = Path('outputs/datasets/btcusdt_core_seq_small.csv')
OUT_DIR = Path('outputs/datasets')
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / 'btcusdt_timexer.csv'
OUT_META = OUT_DIR / 'btcusdt_timexer.meta.json'

def main() -> None:
    df = pd.read_csv(INPUT, parse_dates=['ts']).sort_values('ts').reset_index(drop=True)
    out = pd.DataFrame({'series_id': 'BTCUSDT', 'time_idx': np.arange(len(df), dtype=int), 'ds': pd.to_datetime(df['ts'], utc=True).dt.tz_convert(None), 'target': df['target_log_ret_1h'].astype(float), 'log_ret_1h': df['log_ret_1h'].astype(float), 'premium_chg': df['premium_chg'].astype(float), 'fundingRate': df['fundingRate'].astype(float), 'vol_chg': df['vol_chg'].astype(float), 'trades_chg': df['trades_chg'].astype(float), 'taker_buy_share': df['taker_buy_share'].astype(float), 'rv_24': df['rv_24'].astype(float), 'hour_sin': df['hour_sin'].astype(float), 'hour_cos': df['hour_cos'].astype(float), 'dow_sin': df['dow_sin'].astype(float), 'dow_cos': df['dow_cos'].astype(float)})
    out['fundingRate_missing'] = out['fundingRate'].isna().astype(int)
    out['taker_buy_share_missing'] = out['taker_buy_share'].isna().astype(int)
    out['fundingRate'] = out['fundingRate'].ffill().fillna(0.0)
    out['taker_buy_share'] = out['taker_buy_share'].ffill().fillna(0.0)
    num_cols = [c for c in out.columns if c not in ['series_id', 'ds']]
    out[num_cols] = out[num_cols].replace([np.inf, -np.inf], np.nan)
    out = out.dropna().reset_index(drop=True)
    out.to_csv(OUT_CSV, index=False)
    meta = {'csv': str(OUT_CSV), 'n_rows': int(len(out)), 'columns': list(out.columns), 'target': 'target', 'group_id': 'series_id', 'time_idx': 'time_idx', 'known_reals': ['time_idx', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos'], 'unknown_reals': ['target', 'log_ret_1h', 'premium_chg', 'fundingRate', 'vol_chg', 'trades_chg', 'taker_buy_share', 'rv_24', 'fundingRate_missing', 'taker_buy_share_missing'], 'ds_min': str(out['ds'].min()) if len(out) else None, 'ds_max': str(out['ds'].max()) if len(out) else None}
    OUT_META.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'saved: {OUT_CSV}')
    print(json.dumps(meta, ensure_ascii=False, indent=2))
if __name__ == '__main__':
    main()
