from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class LabelConfig:
    target_family: str = "direction"
    # direction | large_move | vol_regime | horizon_dir | return_token
    return_col: str = "log_ret_1h"
    horizon: int = 1                  # bars ahead; also vol window for vol_regime
    dead_zone_sigma: float = 0.1      # fraction of train std dropped in direction
    move_quantile: float = 0.80       # upper tail threshold for large_move
    n_tokens: int = 5                 # bins for return_token
    vol_window: int = 6               # rolling window for forward realised vol

    def to_dict(self) -> dict:
        return asdict(self)


class LabelMaker:
    """
    Builds classification labels from log-return column.

    Must be .fit() on the TRAINING split only. .transform() can then be called
    on any split (train / val / test) without leakage, because bin edges were
    estimated on training data alone.
    """

    def __init__(self, config: LabelConfig):
        self.config = config
        self._edges: Optional[np.ndarray] = None   # quantile / tercile edges
        self._train_std: Optional[float] = None
        self._fitted = False

    # ------------------------------------------------------------------
    # Fit (on train only)
    # ------------------------------------------------------------------

    def fit(self, train_df: pd.DataFrame) -> "LabelMaker":
        ret = train_df[self.config.return_col].dropna().to_numpy(dtype=float)
        family = self.config.target_family

        if family == "direction":
            self._train_std = float(np.std(ret))

        elif family == "large_move":
            fwd = self._forward_returns(train_df)
            abs_fwd = np.abs(fwd.dropna().to_numpy())
            self._edges = np.array([np.nanquantile(abs_fwd, self.config.move_quantile)])

        elif family == "vol_regime":
            fwd_rv = self._forward_rv(train_df).dropna().to_numpy()
            self._edges = np.nanquantile(fwd_rv, [1 / 3, 2 / 3])

        elif family == "horizon_dir":
            # no edge to estimate; just need std for optional dead zone
            self._train_std = float(np.std(ret))

        elif family == "return_token":
            fwd = self._forward_returns(train_df).dropna().to_numpy()
            # quantile edges → near-uniform classes
            qs = np.linspace(0, 1, self.config.n_tokens + 1)
            self._edges = np.nanquantile(fwd, qs[1:-1])  # n_tokens-1 internal edges

        else:
            raise ValueError(f"Unknown target_family: {family!r}")

        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Transform (any split after fit)
    # ------------------------------------------------------------------

    def transform(self, df: pd.DataFrame) -> pd.Series:
        assert self._fitted, "Call .fit(train_df) before .transform()"
        family = self.config.target_family

        if family == "direction":
            return self._make_direction(df)
        if family == "large_move":
            return self._make_large_move(df)
        if family == "vol_regime":
            return self._make_vol_regime(df)
        if family == "horizon_dir":
            return self._make_horizon_dir(df)
        if family == "return_token":
            return self._make_return_token(df)
        raise ValueError(f"Unknown target_family: {family!r}")

    def label_name(self) -> str:
        return f"y_{self.config.target_family}"

    def class_names(self) -> List[str]:
        family = self.config.target_family
        if family in ("direction", "large_move", "horizon_dir"):
            return ["down", "up"]
        if family == "vol_regime":
            return ["low_vol", "mid_vol", "high_vol"]
        if family == "return_token":
            return [f"tok_{i}" for i in range(self.config.n_tokens)]
        raise ValueError(family)

    def n_classes(self) -> int:
        return len(self.class_names())

    def describe(self) -> dict:
        return {
            "config": self.config.to_dict(),
            "edges": self._edges.tolist() if self._edges is not None else None,
            "train_std": self._train_std,
            "class_names": self.class_names(),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _forward_returns(self, df: pd.DataFrame) -> pd.Series:
        """Sum of log-returns over the next `horizon` bars."""
        ret = df[self.config.return_col].astype(float)
        if self.config.horizon == 1:
            return ret.shift(-1)
        # rolling sum over the next H bars
        return ret.shift(-1).rolling(window=self.config.horizon, min_periods=self.config.horizon).sum().shift(-(self.config.horizon - 1))

    def _forward_rv(self, df: pd.DataFrame) -> pd.Series:
        """Realised volatility (std) of the next `vol_window` bars."""
        ret = df[self.config.return_col].astype(float)
        w = self.config.vol_window
        return ret.shift(-1).rolling(window=w, min_periods=w).std().shift(-(w - 1))

    def _make_direction(self, df: pd.DataFrame) -> pd.Series:
        fwd = self._forward_returns(df)
        threshold = (self.config.dead_zone_sigma * self._train_std) if self._train_std else 0.0
        label = pd.Series(np.nan, index=df.index, name=self.label_name())
        up_mask = fwd > threshold
        down_mask = fwd < -threshold
        label[up_mask] = 1
        label[down_mask] = 0
        # rows inside dead-zone stay NaN and are dropped at training time
        return label.astype("Int64")

    def _make_large_move(self, df: pd.DataFrame) -> pd.Series:
        fwd = self._forward_returns(df).abs()
        threshold = float(self._edges[0])
        label = (fwd >= threshold).astype("Int64")
        label[fwd.isna()] = pd.NA
        return label.rename(self.label_name())

    def _make_vol_regime(self, df: pd.DataFrame) -> pd.Series:
        fwd_rv = self._forward_rv(df)
        low, high = float(self._edges[0]), float(self._edges[1])
        conditions = [fwd_rv < low, fwd_rv <= high]
        choices = [0, 1]
        arr = np.select(conditions, choices, default=2)
        label = pd.Series(arr, index=df.index, name=self.label_name(), dtype="Int64")
        label[fwd_rv.isna()] = pd.NA
        return label

    def _make_horizon_dir(self, df: pd.DataFrame) -> pd.Series:
        fwd = self._forward_returns(df)
        threshold = (self.config.dead_zone_sigma * self._train_std) if self._train_std else 0.0
        label = pd.Series(np.nan, index=df.index, name=self.label_name())
        label[fwd > threshold] = 1
        label[fwd < -threshold] = 0
        return label.astype("Int64")

    def _make_return_token(self, df: pd.DataFrame) -> pd.Series:
        fwd = self._forward_returns(df)
        arr = np.digitize(fwd.to_numpy(dtype=float), bins=self._edges)
        # np.digitize gives values 0..n_tokens (inclusive)
        # cap at n_tokens-1 to stay within class range
        arr = np.clip(arr, 0, self.config.n_tokens - 1)
        label = pd.Series(arr, index=df.index, name=self.label_name(), dtype="Int64")
        label[fwd.isna()] = pd.NA
        return label
