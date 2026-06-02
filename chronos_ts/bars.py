from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd
BAR_COLS = ['bar_start', 'bar_end', 'open', 'high', 'low', 'close', 'volume', 'dollar_volume', 'n_trades', 'taker_buy_base', 'dt_seconds']

def _empty_bar_df() -> pd.DataFrame:
    return pd.DataFrame(columns=BAR_COLS)

class _BarBuilder:

    def _validate_input(self, df: pd.DataFrame) -> pd.DataFrame:
        required = {'ts', 'open', 'high', 'low', 'close', 'volume'}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f'Missing columns: {missing}')
        df = df.sort_values('ts').reset_index(drop=True)
        return df

    def _finalize_bar(self, rows: list, last_bar_end: pd.Timestamp) -> pd.DataFrame:
        if not rows:
            return _empty_bar_df()
        df = pd.DataFrame(rows, columns=BAR_COLS)
        assert (df['bar_end'] <= last_bar_end).all(), 'Bar ends past data end — possible look-ahead leak'
        return df

    def _assert_no_overlap(self, bars: pd.DataFrame) -> None:
        if len(bars) < 2:
            return
        ends = bars['bar_end'].values[:-1]
        starts = bars['bar_start'].values[1:]
        assert (ends <= starts).all(), 'Overlapping bars detected'

class TimeBarBuilder(_BarBuilder):

    def __init__(self, freq: str='1h'):
        self.freq = freq

    def build(self, df_1m: pd.DataFrame) -> pd.DataFrame:
        df = self._validate_input(df_1m)
        df = df.set_index('ts')
        agg = df.resample(self.freq).agg(open=('open', 'first'), high=('high', 'max'), low=('low', 'min'), close=('close', 'last'), volume=('volume', 'sum'), dollar_volume=('dollar_volume' if 'dollar_volume' in df.columns else 'volume', 'sum'), n_trades=('n_trades' if 'n_trades' in df.columns else 'volume', 'count'), taker_buy_base=('taker_buy_base' if 'taker_buy_base' in df.columns else 'volume', 'sum')).dropna(subset=['open']).reset_index()
        bars = pd.DataFrame({'bar_start': agg['ts'], 'bar_end': agg['ts'] + pd.Timedelta(self.freq), 'open': agg['open'], 'high': agg['high'], 'low': agg['low'], 'close': agg['close'], 'volume': agg['volume'], 'dollar_volume': agg.get('dollar_volume', agg['volume'] * agg['close']), 'n_trades': agg['n_trades'], 'taker_buy_base': agg['taker_buy_base'], 'dt_seconds': pd.Timedelta(self.freq).total_seconds()})
        return bars

class VolumeBarBuilder(_BarBuilder):

    def __init__(self, volume_threshold: Optional[float]=None):
        self.volume_threshold = volume_threshold

    def build(self, df_1m: pd.DataFrame) -> pd.DataFrame:
        df = self._validate_input(df_1m)
        df = self._enrich(df)
        threshold = self.volume_threshold or self._calibrate(df)
        bars = []
        self._accumulate(df, threshold, bars, kind='volume')
        out = self._finalize_bar(bars, df['ts'].iloc[-1])
        self._assert_no_overlap(out)
        return out

    def _calibrate(self, df: pd.DataFrame) -> float:
        return float(df['volume'].sum() / (len(df) / 60))

    def _enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'dollar_volume' not in df.columns:
            df = df.copy()
            df['dollar_volume'] = df['volume'] * df['close'].astype(float)
        if 'n_trades' not in df.columns:
            df['n_trades'] = 1
        if 'taker_buy_base' not in df.columns:
            df['taker_buy_base'] = df['volume'] * 0.5
        return df

    def _accumulate(self, df: pd.DataFrame, threshold: float, bars: list, kind: str) -> None:
        cum_vol = cum_dollar = cum_trades = cum_buy = 0.0
        bar_open = bar_high = bar_low = None
        bar_start = None
        for _, row in df.iterrows():
            v = float(row['volume'])
            d = float(row['dollar_volume'])
            n = float(row.get('n_trades', 1))
            b = float(row.get('taker_buy_base', v * 0.5))
            price = float(row['close'])
            if bar_start is None:
                bar_start = row['ts']
                bar_open = float(row['open'])
                bar_high = float(row['high'])
                bar_low = float(row['low'])
            bar_high = max(bar_high, float(row['high']))
            bar_low = min(bar_low, float(row['low']))
            cum_vol += v
            cum_dollar += d
            cum_trades += n
            cum_buy += b
            limit = cum_vol if kind == 'volume' else cum_dollar
            if limit >= threshold:
                dt_s = (row['ts'] - bar_start).total_seconds() + 60
                bars.append([bar_start, row['ts'], bar_open, bar_high, bar_low, price, cum_vol, cum_dollar, int(cum_trades), cum_buy, dt_s])
                cum_vol = cum_dollar = cum_trades = cum_buy = 0.0
                bar_start = None

class DollarBarBuilder(VolumeBarBuilder):

    def __init__(self, dollar_threshold: Optional[float]=None):
        super().__init__(volume_threshold=dollar_threshold)

    def build(self, df_1m: pd.DataFrame) -> pd.DataFrame:
        df = self._validate_input(df_1m)
        df = self._enrich(df)
        threshold = self.volume_threshold or self._calibrate_dollar(df)
        bars = []
        self._accumulate(df, threshold, bars, kind='dollar')
        out = self._finalize_bar(bars, df['ts'].iloc[-1])
        self._assert_no_overlap(out)
        return out

    def _calibrate_dollar(self, df: pd.DataFrame) -> float:
        return float(df['dollar_volume'].sum() / (len(df) / 60))

class ImbalanceBarBuilder(_BarBuilder):

    def __init__(self, imbalance_threshold: Optional[float]=None):
        self.imbalance_threshold = imbalance_threshold

    def build(self, df_1m: pd.DataFrame) -> pd.DataFrame:
        df = self._validate_input(df_1m)
        if 'taker_buy_base' not in df.columns:
            df = df.copy()
            df['taker_buy_base'] = df['volume'] * 0.5
        if 'dollar_volume' not in df.columns:
            df = df.copy()
            df['dollar_volume'] = df['volume'] * df['close'].astype(float)
        if 'n_trades' not in df.columns:
            df['n_trades'] = 1
        total_vol = df['volume'].astype(float)
        buy_vol = df['taker_buy_base'].astype(float)
        sell_vol = (total_vol - buy_vol).clip(lower=0)
        imbalance = buy_vol - sell_vol
        threshold = self.imbalance_threshold or float(imbalance.abs().mean() * 10)
        bars = []
        cum_imb = cum_vol = cum_dollar = cum_trades = cum_buy = 0.0
        bar_open = bar_high = bar_low = None
        bar_start = None
        for i, row in df.iterrows():
            v = float(row['volume'])
            d = float(row['dollar_volume'])
            n = float(row.get('n_trades', 1))
            b = float(row['taker_buy_base'])
            price = float(row['close'])
            imb = float(imbalance.iloc[i])
            if bar_start is None:
                bar_start = row['ts']
                bar_open = float(row['open'])
                bar_high = float(row['high'])
                bar_low = float(row['low'])
            bar_high = max(bar_high, float(row['high']))
            bar_low = min(bar_low, float(row['low']))
            cum_vol += v
            cum_dollar += d
            cum_trades += n
            cum_buy += b
            cum_imb += imb
            if abs(cum_imb) >= threshold:
                dt_s = (row['ts'] - bar_start).total_seconds() + 60
                bars.append([bar_start, row['ts'], bar_open, bar_high, bar_low, price, cum_vol, cum_dollar, int(cum_trades), cum_buy, dt_s])
                cum_imb = cum_vol = cum_dollar = cum_trades = cum_buy = 0.0
                bar_start = None
        out = self._finalize_bar(bars, df['ts'].iloc[-1])
        self._assert_no_overlap(out)
        return out