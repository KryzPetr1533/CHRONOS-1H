from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
INPUT = Path('outputs/datasets/btcusdt_core_seq_small.csv')
OUT_DIR = Path('outputs/datasets')
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / 'btcusdt_patchtst.csv'
OUT_META = OUT_DIR / 'btcusdt_patchtst.meta.json'
HIST_EXOG = ['premium_chg', 'fundingRate', 'vol_chg', 'trades_chg', 'taker_buy_share', 'rv_24', 'fundingRate_missing', 'taker_buy_share_missing']
FUTR_EXOG = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos']

def main() -> None:
    df = pd.read_csv(INPUT, parse_dates=['ts']).sort_values('ts').reset_index(drop=True)
    ts = pd.to_datetime(df['ts'], utc=True).dt.tz_convert(None)
    out = pd.DataFrame({'unique_id': 'BTCUSDT', 'ds': ts, 'y': df['log_ret_1h'].astype(float), 'premium_chg': df['premium_chg'].astype(float), 'fundingRate': df['fundingRate'].astype(float), 'vol_chg': df['vol_chg'].astype(float), 'trades_chg': df['trades_chg'].astype(float), 'taker_buy_share': df['taker_buy_share'].astype(float), 'rv_24': df['rv_24'].astype(float), 'hour_sin': df['hour_sin'].astype(float), 'hour_cos': df['hour_cos'].astype(float), 'dow_sin': df['dow_sin'].astype(float), 'dow_cos': df['dow_cos'].astype(float)})
    out['fundingRate_missing'] = out['fundingRate'].isna().astype(int)
    out['taker_buy_share_missing'] = out['taker_buy_share'].isna().astype(int)
    out['fundingRate'] = out['fundingRate'].ffill().fillna(0.0)
    out['taker_buy_share'] = out['taker_buy_share'].ffill().fillna(0.0)
    for col in ['y'] + HIST_EXOG + FUTR_EXOG:
        out[col] = pd.to_numeric(out[col], errors='coerce')
    out = out.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    out.to_csv(OUT_CSV, index=False)
    meta = {'csv': str(OUT_CSV), 'n_rows': int(len(out)), 'columns': list(out.columns), 'hist_exog': HIST_EXOG, 'futr_exog': FUTR_EXOG, 'ds_min': str(out['ds'].min()) if len(out) else None, 'ds_max': str(out['ds'].max()) if len(out) else None}
    OUT_META.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'saved: {OUT_CSV}')
    print(json.dumps(meta, ensure_ascii=False, indent=2))
if __name__ == '__main__':
    main()
