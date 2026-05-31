"""
Feature engineering on top of event bars (volume/dollar/time bars from 1m klines).

Usage:
    from chronos_ts.event_features import EventFeatureBuilder
    builder = EventFeatureBuilder()
    df_features = builder.build(bars_df, external_df=eth_bars)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class EventFeatureConfig:
    lag_bars: List[int] = field(default_factory=lambda: [1, 2, 3, 6, 12, 24, 48])
    rolling_windows: List[int] = field(default_factory=lambda: [6, 24, 72])
    include_ohlc_vol: bool = True       # Parkinson / Garman-Klass
    include_order_flow: bool = True     # imbalance, run length
    include_calendar: bool = True
    include_external: bool = True       # external asset covariates
    ts_col: str = "bar_end"


class EventFeatureBuilder:
    """
    Build a feature matrix from event bars.

    Callable from notebooks and scripts:
        df = EventFeatureBuilder().build(bars, external_df=eth_bars)
    """

    def __init__(self, config: Optional[EventFeatureConfig] = None):
        self.config = config or EventFeatureConfig()

    def build(
        self,
        bars: pd.DataFrame,
        external_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        bars        : event bar DataFrame (standard BAR_COLS schema from bars.py)
        external_df : optional DataFrame with a 'ts' column and return/vol columns
                      from an external asset (e.g. ETHUSDT bars, already built).

        Returns
        -------
        Feature DataFrame indexed by bar_end, with log_ret_bar computed.
        """
        df = bars.copy()
        ts_col = self.config.ts_col

        # Ensure time sorted
        df = df.sort_values(ts_col).reset_index(drop=True)

        # Base return
        df["log_ret_bar"] = np.log(df["close"] / df["close"].shift(1))

        if self.config.include_ohlc_vol:
            df = self._add_ohlc_volatility(df)

        if self.config.include_order_flow:
            df = self._add_order_flow(df)

        if self.config.include_calendar:
            df = self._add_calendar(df, ts_col)

        if self.config.include_external and external_df is not None:
            df = self._add_external(df, external_df, ts_col)

        df = self._add_lag_features(df)
        df = self._add_rolling_features(df)

        return df

    # ------------------------------------------------------------------ #

    def _add_ohlc_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parkinson and Garman-Klass estimators — strictly intra-bar (causal)."""
        hi = np.log(df["high"].astype(float))
        lo = np.log(df["low"].astype(float))
        op = np.log(df["open"].astype(float))
        cl = np.log(df["close"].astype(float))

        # Parkinson: uses high-low range only
        df["rv_parkinson"] = ((hi - lo) ** 2) / (4 * np.log(2))

        # Garman-Klass: uses all four prices
        df["rv_garman_klass"] = (
            0.5 * (hi - lo) ** 2
            - (2 * np.log(2) - 1) * (cl - op) ** 2
        ).clip(lower=0)

        # Intra-bar realized vol proxy: dt-scaled
        df["dt_scaled_rv"] = df["rv_parkinson"] * np.sqrt(3600 / df["dt_seconds"].clip(lower=1))

        return df

    def _add_order_flow(self, df: pd.DataFrame) -> pd.DataFrame:
        """Signed-volume imbalance and buy/sell run length."""
        total_vol = df["volume"].astype(float)
        buy_vol   = df.get("taker_buy_base", total_vol * 0.5).astype(float)
        sell_vol  = (total_vol - buy_vol).clip(lower=0)

        df["buy_vol_share"]  = (buy_vol / total_vol.replace(0, np.nan)).fillna(0.5)
        df["vol_imbalance"]  = buy_vol - sell_vol
        df["net_flow"]       = df["vol_imbalance"].cumsum()

        # Run length: consecutive bars where buy_vol_share > 0.5
        direction = (df["buy_vol_share"] > 0.5).astype(int) * 2 - 1  # ±1
        run = direction.groupby((direction != direction.shift()).cumsum()).cumcount() + 1
        df["buy_run_length"] = np.where(direction > 0, run, 0)
        df["sell_run_length"] = np.where(direction < 0, run, 0)

        # Inter-bar time (already in dt_seconds)
        if "dt_seconds" in df.columns:
            df["log_dt_seconds"] = np.log1p(df["dt_seconds"])

        return df

    def _add_calendar(self, df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
        ts = pd.to_datetime(df[ts_col])
        hour = ts.dt.hour
        dow  = ts.dt.dayofweek
        fund = hour % 8  # 8h funding cycle

        df["hour_sin"]         = np.sin(2 * np.pi * hour / 24)
        df["hour_cos"]         = np.cos(2 * np.pi * hour / 24)
        df["dow_sin"]          = np.sin(2 * np.pi * dow / 7)
        df["dow_cos"]          = np.cos(2 * np.pi * dow / 7)
        df["fund_cycle_sin"]   = np.sin(2 * np.pi * fund / 8)
        df["fund_cycle_cos"]   = np.cos(2 * np.pi * fund / 8)
        return df

    def _add_external(
        self, df: pd.DataFrame, ext: pd.DataFrame, ts_col: str
    ) -> pd.DataFrame:
        """
        Merge external asset features (e.g. ETHUSDT) onto the main bar DataFrame.

        External df must have a 'ts' column. We lag by >=1 bar using merge_asof
        so there is no look-ahead: we only use the last *completed* external bar
        that ended before this bar started.
        """
        ext = ext.copy().sort_values("ts" if "ts" in ext.columns else ext.columns[0])
        ext_ts_col = "ts" if "ts" in ext.columns else ext.columns[0]

        # Compute external return + vol if not present
        if "log_ret_bar" not in ext.columns and "close" in ext.columns:
            ext["ext_log_ret"] = np.log(ext["close"] / ext["close"].shift(1))
        else:
            ext = ext.rename(columns={"log_ret_bar": "ext_log_ret"})

        if "rv_parkinson" in ext.columns:
            ext = ext.rename(columns={"rv_parkinson": "ext_rv_parkinson"})

        keep = [ext_ts_col] + [c for c in ["ext_log_ret", "ext_rv_parkinson"] if c in ext.columns]
        ext_slim = ext[keep].rename(columns={ext_ts_col: "_ext_ts"})

        # Use merge_asof: for each main bar's bar_start, find the last completed ext bar
        bar_start_col = "bar_start" if "bar_start" in df.columns else ts_col
        merged = pd.merge_asof(
            df.sort_values(bar_start_col).reset_index(),
            ext_slim.sort_values("_ext_ts"),
            left_on=bar_start_col,
            right_on="_ext_ts",
            direction="backward",
        ).set_index("index").sort_index()
        merged = merged.drop(columns=["_ext_ts"], errors="ignore")
        return merged

    def _lag_candidates(self, df: pd.DataFrame) -> List[str]:
        candidates = [
            "log_ret_bar", "vol_imbalance", "buy_vol_share",
            "rv_parkinson", "rv_garman_klass", "dt_scaled_rv",
            "volume", "dollar_volume", "log_dt_seconds",
            "ext_log_ret", "ext_rv_parkinson",
        ]
        return [c for c in candidates if c in df.columns]

    def _rolling_candidates(self, df: pd.DataFrame) -> List[str]:
        candidates = [
            "log_ret_bar", "vol_imbalance", "buy_vol_share",
            "rv_parkinson", "dt_scaled_rv", "volume", "dollar_volume",
            "ext_log_ret",
        ]
        return [c for c in candidates if c in df.columns]

    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        candidates = self._lag_candidates(df)
        frames = [df]
        for col in candidates:
            for lag in self.config.lag_bars:
                frames.append(df[[col]].shift(lag).rename(columns={col: f"{col}_lag_{lag}"}))
        return pd.concat(frames, axis=1)

    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        candidates = self._rolling_candidates(df)
        frames = [df]
        for col in candidates:
            for w in self.config.rolling_windows:
                roll = df[col].rolling(window=w, min_periods=w)
                frames.append(pd.DataFrame({
                    f"{col}_rmean_{w}": roll.mean(),
                    f"{col}_rstd_{w}":  roll.std(),
                }))
        # Realized vol from log_ret_bar
        if "log_ret_bar" in df.columns:
            for w in self.config.rolling_windows:
                frames.append(pd.DataFrame({
                    f"rv_bar_{w}": df["log_ret_bar"].rolling(w, min_periods=w).std(),
                    f"abs_ret_rmean_{w}": df["log_ret_bar"].abs().rolling(w, min_periods=w).mean(),
                }))
        return pd.concat(frames, axis=1)
