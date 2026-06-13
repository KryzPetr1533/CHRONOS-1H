#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np

REPO = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description='EDA summary for btcusdt_clf dataset')
    parser.add_argument('--csv', default='outputs/datasets/btcusdt_clf_core.csv')
    parser.add_argument('--out', default='outputs/reports/clf_eda_summary.md')
    args = parser.parse_args()
    path = REPO / args.csv
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, parse_dates=['ts'])
    df = df.sort_values('ts').reset_index(drop=True)
    lines = [
        '# Classification dataset EDA',
        '',
        f'**File:** `{args.csv}`',
        '',
        '## Shape',
        f'- Rows: {len(df):,}',
        f'- Columns: {len(df.columns)}',
        f'- Date range: {df["ts"].min()} → {df["ts"].max()}',
        '',
        '## Missingness (top 15)',
    ]
    miss = df.isna().mean().sort_values(ascending=False).head(15)
    for col, pct in miss.items():
        lines.append(f'- `{col}`: {pct:.2%}')
    if 'log_ret_1h' in df.columns:
        r = df['log_ret_1h'].dropna()
        lines.extend([
            '',
            '## Returns (`log_ret_1h`)',
            f'- mean: {r.mean():.6f}',
            f'- std: {r.std():.6f}',
            f'- min / max: {r.min():.6f} / {r.max():.6f}',
        ])
    if 'log_ret_1h_lag_1' in df.columns and 'log_ret_1h' in df.columns:
        diff = (df['log_ret_1h_lag_1'] - df['log_ret_1h'].shift(1)).abs().max()
        lines.append(f'- lag_1 vs shift(1) max |diff|: {diff:.2e} (expect ~0)')
    lines.extend(['', '## Column dtypes', '```', df.dtypes.value_counts().to_string(), '```'])
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    meta = {'rows': len(df), 'cols': len(df.columns), 'ts_min': str(df['ts'].min()), 'ts_max': str(df['ts'].max())}
    (out.with_suffix('.json')).write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(f'Wrote {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
