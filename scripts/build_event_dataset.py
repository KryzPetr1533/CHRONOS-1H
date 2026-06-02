from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chronos_ts.bars import VolumeBarBuilder, DollarBarBuilder
from chronos_ts.event_features import EventFeatureBuilder, EventFeatureConfig
from chronos_ts.labels import LabelConfig, LabelMaker
from chronos_ts.splits import TimeRangeSplitConfig, time_fraction_split

class EventDatasetBuilder:

    def __init__(self, raw_1m_dir: str='data/raw_fine/BTCUSDT/1m', source_1h_csv: str='data/btcusdt_1h_merged.csv', output_dir: str='outputs/datasets', min_history_bars: int=200):
        self.raw_1m_dir = Path(raw_1m_dir)
        self.source_1h = Path(source_1h_csv)
        self.output_dir = Path(output_dir)
        self.min_history = min_history_bars

    def run(self, force_proxy: bool=False, force_1m: bool=False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        use_proxy = self._should_use_proxy(force_proxy, force_1m)
        if use_proxy:
            print('Mode: 1h-proxy (no 1m klines found). Run `python scripts/fetch_fine_data.py` to get real data.')
            df_1m = self._synthesise_1m_from_1h()
        else:
            print('Mode: 1m klines from data/raw_fine/')
            df_1m = self._load_1m_klines()
        print(f'1m bars: {len(df_1m):,} rows')
        vb_raw = VolumeBarBuilder().build(df_1m)
        db_raw = DollarBarBuilder().build(df_1m)
        print(f'Volume bars: {len(vb_raw)}  Dollar bars: {len(db_raw)}')
        feat_builder = EventFeatureBuilder(EventFeatureConfig(lag_bars=[1, 2, 3, 6, 12, 24], rolling_windows=[6, 24, 72], ts_col='bar_end'))
        vb_feat = feat_builder.build(vb_raw)
        db_feat = feat_builder.build(db_raw)
        vb_feat = self._add_targets(vb_feat, ts_col='bar_end')
        db_feat = self._add_targets(db_feat, ts_col='bar_end')
        vb_feat = vb_feat.iloc[self.min_history:].reset_index(drop=True)
        db_feat = db_feat.iloc[self.min_history:].reset_index(drop=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        vb_path = self.output_dir / 'btcusdt_volume_bars.csv'
        db_path = self.output_dir / 'btcusdt_dollar_bars.csv'
        vb_feat.to_csv(vb_path, index=False)
        db_feat.to_csv(db_path, index=False)
        self._write_meta(vb_feat, vb_path, 'volume', use_proxy)
        self._write_meta(db_feat, db_path, 'dollar', use_proxy)
        print(f'Saved: {vb_path}  ({len(vb_feat)} rows, {vb_feat.shape[1]} cols)')
        print(f'Saved: {db_path}  ({len(db_feat)} rows, {db_feat.shape[1]} cols)')
        self._sanity_check(vb_feat, 'volume')
        self._sanity_check(db_feat, 'dollar')
        return (vb_feat, db_feat)

    def _should_use_proxy(self, force_proxy, force_1m) -> bool:
        if force_proxy:
            return True
        if force_1m:
            return False
        parquets = list(self.raw_1m_dir.glob('*.parquet')) if self.raw_1m_dir.exists() else []
        return len(parquets) == 0

    def _load_1m_klines(self) -> pd.DataFrame:
        parquets = sorted(self.raw_1m_dir.glob('*.parquet'))
        frames = [pd.read_parquet(p) for p in parquets]
        df = pd.concat(frames, ignore_index=True)
        df = df.sort_values('ts').reset_index(drop=True)
        if 'dollar_volume' not in df.columns:
            df['dollar_volume'] = df['volume'] * df['close'].astype(float)
        return df

    def _synthesise_1m_from_1h(self) -> pd.DataFrame:
        df_1h = pd.read_csv(self.source_1h, parse_dates=['ts'])
        df_1h = df_1h.sort_values('ts').reset_index(drop=True)
        rng = np.random.default_rng(42)
        rows = []
        for _, row in df_1h.iterrows():
            ts_start = pd.Timestamp(row['ts'])
            o = float(row['open']) if not pd.isna(row.get('open', np.nan)) else float(row['close'])
            h = float(row['high']) if not pd.isna(row.get('high', np.nan)) else o * 1.001
            lo = float(row['low']) if not pd.isna(row.get('low', np.nan)) else o * 0.999
            c = float(row['close'])
            vol = float(row.get('volume', 1.0))
            n_trades = int(row.get('num_trades', 60)) if not pd.isna(row.get('num_trades', np.nan)) else 60
            buy_base = float(row.get('taker_buy_base', vol * 0.5)) if not pd.isna(row.get('taker_buy_base', np.nan)) else vol * 0.5
            prices = np.linspace(o, c, 60) + rng.normal(0, (h - lo) / 6, 60)
            prices = np.clip(prices, lo, h)
            vol_split = rng.dirichlet(np.ones(60)) * vol
            buy_split = rng.dirichlet(np.ones(60)) * buy_base
            for i in range(60):
                minute_ts = ts_start + pd.Timedelta(minutes=i)
                p = prices[i]
                v = vol_split[i]
                rows.append({'ts': minute_ts, 'open': p, 'high': p * (1 + rng.uniform(0, 0.0002)), 'low': p * (1 - rng.uniform(0, 0.0002)), 'close': p, 'volume': v, 'dollar_volume': v * p, 'n_trades': max(1, n_trades // 60), 'taker_buy_base': buy_split[i]})
        df = pd.DataFrame(rows)
        df['ts'] = pd.to_datetime(df['ts'], utc=True)
        return df.sort_values('ts').reset_index(drop=True)

    def _add_targets(self, df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
        if 'log_ret_bar' not in df.columns and 'close' in df.columns:
            df['log_ret_bar'] = np.log(df['close'] / df['close'].shift(1))
        return df

    def _write_meta(self, df: pd.DataFrame, path: Path, bar_type: str, is_proxy: bool) -> None:
        meta = {'bar_type': bar_type, 'is_proxy_from_1h': is_proxy, 'n_rows': len(df), 'n_cols': df.shape[1], 'ts_min': str(df['bar_end'].min() if 'bar_end' in df.columns else ''), 'ts_max': str(df['bar_end'].max() if 'bar_end' in df.columns else ''), 'note': 'PROXY dataset: 1h bars decomposed into synthetic 1m bars. For real results run scripts/fetch_fine_data.py first.' if is_proxy else 'Real 1m klines from Binance FAPI'}
        path.with_suffix('.meta.json').write_text(json.dumps(meta, indent=2, default=str), encoding='utf-8')

    def _sanity_check(self, df: pd.DataFrame, bar_type: str) -> None:
        print(f'\n{bar_type} bars sanity check:')
        if 'bar_end' in df.columns:
            ts = pd.to_datetime(df['bar_end'])
            ok = ts.is_monotonic_increasing
            print(f"  sorted: {('OK' if ok else 'FAIL')}")
        if 'log_ret_bar' in df.columns:
            nn = df['log_ret_bar'].notna().sum()
            print(f'  log_ret_bar: {nn}/{len(df)} non-null')
        print(f'  shape: {df.shape}')

def main() -> None:
    parser = argparse.ArgumentParser(description='Build event-bar feature datasets.')
    parser.add_argument('--from-1h', action='store_true', help='Force 1h proxy mode')
    parser.add_argument('--from-1m', action='store_true', help='Require real 1m klines')
    parser.add_argument('--output-dir', default='outputs/datasets')
    args = parser.parse_args()
    EventDatasetBuilder(output_dir=args.output_dir).run(force_proxy=args.from_1h, force_1m=args.from_1m)
if __name__ == '__main__':
    main()