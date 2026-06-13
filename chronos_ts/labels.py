from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Optional
import numpy as np
import pandas as pd

@dataclass
class LabelConfig:
    target_family: str = 'direction'
    return_col: str = 'log_ret_1h'
    horizon: int = 1
    dead_zone_sigma: float = 0.1
    move_quantile: float = 0.8
    n_tokens: int = 5
    vol_window: int = 6

    def to_dict(self) -> dict:
        return asdict(self)

class LabelMaker:

    def __init__(self, config: LabelConfig):
        self.config = config
        self._edges: Optional[np.ndarray] = None
        self._train_std: Optional[float] = None
        self._fitted = False

    def fit(self, train_df: pd.DataFrame) -> 'LabelMaker':
        ret = train_df[self.config.return_col].dropna().to_numpy(dtype=float)
        family = self.config.target_family
        if family == 'direction':
            self._train_std = float(np.std(ret))
        elif family == 'large_move':
            fwd = self._forward_returns(train_df)
            abs_fwd = np.abs(fwd.dropna().to_numpy())
            self._edges = np.array([np.nanquantile(abs_fwd, self.config.move_quantile)])
        elif family == 'vol_regime':
            fwd_rv = self._forward_rv(train_df).dropna().to_numpy()
            self._edges = np.nanquantile(fwd_rv, [1 / 3, 2 / 3])
        elif family == 'horizon_dir':
            self._train_std = float(np.std(ret))
        elif family == 'return_token':
            fwd = self._forward_returns(train_df).dropna().to_numpy()
            qs = np.linspace(0, 1, self.config.n_tokens + 1)
            self._edges = np.nanquantile(fwd, qs[1:-1])
        else:
            raise ValueError(f'Unknown target_family: {family!r}')
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.Series:
        assert self._fitted, 'Call .fit(train_df) before .transform()'
        family = self.config.target_family
        if family == 'direction':
            return self._make_direction(df)
        if family == 'large_move':
            return self._make_large_move(df)
        if family == 'vol_regime':
            return self._make_vol_regime(df)
        if family == 'horizon_dir':
            return self._make_horizon_dir(df)
        if family == 'return_token':
            return self._make_return_token(df)
        raise ValueError(f'Unknown target_family: {family!r}')

    def label_name(self) -> str:
        return f'y_{self.config.target_family}'

    def class_names(self) -> List[str]:
        family = self.config.target_family
        if family in ('direction', 'large_move', 'horizon_dir'):
            return ['down', 'up']
        if family == 'vol_regime':
            return ['low_vol', 'mid_vol', 'high_vol']
        if family == 'return_token':
            return [f'tok_{i}' for i in range(self.config.n_tokens)]
        raise ValueError(family)

    def n_classes(self) -> int:
        return len(self.class_names())

    def describe(self) -> dict:
        return {'config': self.config.to_dict(), 'edges': self._edges.tolist() if self._edges is not None else None, 'train_std': self._train_std, 'class_names': self.class_names()}

    def _forward_returns(self, df: pd.DataFrame) -> pd.Series:
        ret = df[self.config.return_col].astype(float)
        if self.config.horizon == 1:
            return ret.shift(-1)
        return ret.shift(-1).rolling(window=self.config.horizon, min_periods=self.config.horizon).sum().shift(-(self.config.horizon - 1))

    def _forward_rv(self, df: pd.DataFrame) -> pd.Series:
        ret = df[self.config.return_col].astype(float)
        w = self.config.vol_window
        return ret.shift(-1).rolling(window=w, min_periods=w).std().shift(-(w - 1))

    def _make_direction(self, df: pd.DataFrame) -> pd.Series:
        fwd = self._forward_returns(df)
        threshold = self.config.dead_zone_sigma * self._train_std if self._train_std else 0.0
        label = pd.Series(np.nan, index=df.index, name=self.label_name())
        up_mask = fwd > threshold
        down_mask = fwd < -threshold
        label[up_mask] = 1
        label[down_mask] = 0
        return label.astype('Int64')

    def _make_large_move(self, df: pd.DataFrame) -> pd.Series:
        fwd = self._forward_returns(df).abs()
        threshold = float(self._edges[0])
        label = (fwd >= threshold).astype('Int64')
        label[fwd.isna()] = pd.NA
        return label.rename(self.label_name())

    def _make_vol_regime(self, df: pd.DataFrame) -> pd.Series:
        fwd_rv = self._forward_rv(df)
        low, high = (float(self._edges[0]), float(self._edges[1]))
        conditions = [fwd_rv < low, fwd_rv <= high]
        choices = [0, 1]
        arr = np.select(conditions, choices, default=2)
        label = pd.Series(arr, index=df.index, name=self.label_name(), dtype='Int64')
        label[fwd_rv.isna()] = pd.NA
        return label

    def _make_horizon_dir(self, df: pd.DataFrame) -> pd.Series:
        fwd = self._forward_returns(df)
        threshold = self.config.dead_zone_sigma * self._train_std if self._train_std else 0.0
        label = pd.Series(np.nan, index=df.index, name=self.label_name())
        label[fwd > threshold] = 1
        label[fwd < -threshold] = 0
        return label.astype('Int64')

    def _make_return_token(self, df: pd.DataFrame) -> pd.Series:
        fwd = self._forward_returns(df)
        arr = np.digitize(fwd.to_numpy(dtype=float), bins=self._edges)
        arr = np.clip(arr, 0, self.config.n_tokens - 1)
        label = pd.Series(arr, index=df.index, name=self.label_name(), dtype='Int64')
        label[fwd.isna()] = pd.NA
        return label