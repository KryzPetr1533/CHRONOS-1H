from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class DatasetBuildConfig:
    input_csv: str
    output_csv: str
    profile: str = "core"  # core | rich
    ts_col: str = "ts"
    price_col: str = "close"
    current_return_col: str = "log_ret_1h"
    target_col: str = "target_log_ret_1h"
    forecast_horizon: int = 1
    forward_fill_cols: list[str] = field(default_factory=lambda: ["premium_close", "fundingRate", "markPrice"])
    lag_hours: list[int] = field(default_factory=lambda: [1, 2, 3, 6, 12, 24, 48, 72, 168])
    rolling_windows: list[int] = field(default_factory=lambda: [6, 24, 72, 168])
    min_history_hours: int = 168
    dropna_target: bool = True
    include_calendar: bool = True
    include_core_derived: bool = True
    include_rich_derived: bool = True

    @property
    def core_source_cols(self) -> list[str]:
        return [
            "close",
            "volume",
            "quote_volume",
            "num_trades",
            "taker_buy_base",
            "taker_buy_quote",
            "premium_close",
            "fundingRate",
            "log_ret_1h",
        ]

    @property
    def rich_extra_source_cols(self) -> list[str]:
        return [
            "buyVol",
            "sellVol",
            "buySellRatio",
            "taker_imbalance",
            "sumOpenInterest",
            "sumOpenInterestValue",
        ]


class ExperimentDatasetBuilder:
    """Builds modeling-ready tabular datasets for BTCUSDT hourly forecasting."""

    def __init__(self, config: DatasetBuildConfig):
        self.config = config

    def build(self) -> pd.DataFrame:
        df = pd.read_csv(self.config.input_csv, parse_dates=[self.config.ts_col])
        df = df.sort_values(self.config.ts_col).reset_index(drop=True)

        if self.config.current_return_col not in df.columns:
            price = df[self.config.price_col].astype(float)
            df[self.config.current_return_col] = np.log(price / price.shift(1))

        for col in self.config.forward_fill_cols:
            if col in df.columns:
                df[col] = df[col].ffill()

        df[self.config.target_col] = df[self.config.current_return_col].shift(-self.config.forecast_horizon)
        df = self._add_derived_features(df)
        df = self._add_calendar_features(df)
        df = self._add_lagged_features(df)
        df = self._add_rolling_features(df)

        keep_cols = self._final_columns(df)
        out = df[keep_cols].copy()

        if self.config.dropna_target:
            out = out.dropna(subset=[self.config.target_col])

        # Drop early rows that cannot have full lag/rolling history.
        out = out.iloc[self.config.min_history_hours :].reset_index(drop=True)

        required_for_profile = self._required_base_columns()
        out = out.dropna(subset=[c for c in required_for_profile if c in out.columns]).reset_index(drop=True)
        return out

    def save(self, df: pd.DataFrame) -> None:
        output_path = Path(self.config.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

    def _required_base_columns(self) -> list[str]:
        cols = list(self.config.core_source_cols)
        if self.config.profile == "rich":
            cols += self.config.rich_extra_source_cols
        return [self.config.ts_col, self.config.target_col] + cols

    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.config.include_core_derived:
            return df

        if {"markPrice", "close"}.issubset(df.columns):
            df["mark_minus_close"] = df["markPrice"] - df["close"]

        if {"taker_buy_base", "volume"}.issubset(df.columns):
            denom = df["volume"].replace(0, np.nan)
            df["taker_buy_share"] = (df["taker_buy_base"] / denom).fillna(0.0)

        if "premium_close" in df.columns:
            df["premium_chg"] = df["premium_close"].diff()

        if "sumOpenInterest" in df.columns:
            df["oi_pct"] = df["sumOpenInterest"].pct_change().replace([np.inf, -np.inf], np.nan)

        if "volume" in df.columns:
            df["vol_chg"] = np.log1p(df["volume"].clip(lower=0)).diff()

        if "num_trades" in df.columns:
            df["trades_chg"] = np.log1p(df["num_trades"].clip(lower=0)).diff()

        if self.config.include_rich_derived:
            if {"buyVol", "sellVol"}.issubset(df.columns):
                denom_bs = (df["buyVol"] + df["sellVol"]).replace(0, np.nan)
                df["buy_share"] = (df["buyVol"] / denom_bs).fillna(0.0)
                df["net_taker_flow"] = df["buyVol"] - df["sellVol"]

            if {"sumOpenInterest", "sumOpenInterestValue"}.issubset(df.columns):
                denom_oi = df["sumOpenInterest"].replace(0, np.nan)
                df["avg_open_price"] = (df["sumOpenInterestValue"] / denom_oi).fillna(0.0)

            if {"sumOpenInterest", "volume"}.issubset(df.columns):
                denom_vol = df["volume"].replace(0, np.nan)
                df["oi_to_volume"] = (df["sumOpenInterest"] / denom_vol).fillna(0.0)

        return df

    def _add_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.config.include_calendar:
            return df

        ts = df[self.config.ts_col]
        hour = ts.dt.hour
        dow = ts.dt.dayofweek
        fund_cycle = hour % 8

        df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
        df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
        df["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
        df["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
        df["fund_cycle_sin"] = np.sin(2 * np.pi * fund_cycle / 8.0)
        df["fund_cycle_cos"] = np.cos(2 * np.pi * fund_cycle / 8.0)
        return df

    def _lag_feature_candidates(self, df: pd.DataFrame) -> list[str]:
        candidates = [
            "log_ret_1h",
            "premium_close",
            "premium_chg",
            "fundingRate",
            "mark_minus_close",
            "volume",
            "quote_volume",
            "num_trades",
            "taker_buy_share",
            "vol_chg",
            "trades_chg",
            "taker_imbalance",
            "buy_share",
            "net_taker_flow",
            "sumOpenInterest",
            "sumOpenInterestValue",
            "oi_pct",
            "avg_open_price",
            "oi_to_volume",
        ]
        return [c for c in candidates if c in df.columns]

    def _rolling_feature_candidates(self, df: pd.DataFrame) -> list[str]:
        candidates = [
            "log_ret_1h",
            "premium_close",
            "premium_chg",
            "fundingRate",
            "volume",
            "quote_volume",
            "num_trades",
            "taker_buy_share",
            "taker_imbalance",
            "oi_pct",
            "vol_chg",
            "trades_chg",
        ]
        return [c for c in candidates if c in df.columns]

    def _add_lagged_features(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self._lag_feature_candidates(df):
            for lag in self.config.lag_hours:
                df[f"{col}_lag_{lag}"] = df[col].shift(lag)
        return df

    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self._rolling_feature_candidates(df):
            for window in self.config.rolling_windows:
                roll = df[col].rolling(window=window, min_periods=window)
                df[f"{col}_roll_mean_{window}"] = roll.mean()
                df[f"{col}_roll_std_{window}"] = roll.std()

        if "log_ret_1h" in df.columns:
            abs_ret = df["log_ret_1h"].abs()
            for window in self.config.rolling_windows:
                df[f"abs_ret_roll_mean_{window}"] = abs_ret.rolling(window=window, min_periods=window).mean()
                df[f"rv_{window}"] = df["log_ret_1h"].rolling(window=window, min_periods=window).std()
        return df

    def _final_columns(self, df: pd.DataFrame) -> list[str]:
        base = [self.config.ts_col, self.config.price_col, self.config.current_return_col, self.config.target_col]
        selected = set(base)
        selected.update(c for c in self.config.core_source_cols if c in df.columns)
        if self.config.profile == "rich":
            selected.update(c for c in self.config.rich_extra_source_cols if c in df.columns)

        derived_candidates = [
            "mark_minus_close",
            "taker_buy_share",
            "premium_chg",
            "oi_pct",
            "vol_chg",
            "trades_chg",
            "buy_share",
            "net_taker_flow",
            "avg_open_price",
            "oi_to_volume",
            "hour_sin",
            "hour_cos",
            "dow_sin",
            "dow_cos",
            "fund_cycle_sin",
            "fund_cycle_cos",
        ]
        selected.update(c for c in derived_candidates if c in df.columns)
        selected.update(c for c in df.columns if "_lag_" in c or "_roll_" in c or c.startswith("rv_"))

        ordered = [self.config.ts_col] + [c for c in df.columns if c in selected and c != self.config.ts_col]
        return ordered

    def metadata(self, df: pd.DataFrame) -> dict:
        feature_cols = [c for c in df.columns if c not in {self.config.ts_col, self.config.target_col}]
        return {
            "profile": self.config.profile,
            "n_rows": int(len(df)),
            "n_features": int(len(feature_cols)),
            "target_col": self.config.target_col,
            "feature_cols": feature_cols,
            "ts_min": str(df[self.config.ts_col].min()),
            "ts_max": str(df[self.config.ts_col].max()),
            "config": asdict(self.config),
        }
