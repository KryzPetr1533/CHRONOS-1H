from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)


class ChronosFeatureEngineerB(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        required_raw: Sequence[str],
        output_cols: Sequence[str],
    ):
        self.required_raw = required_raw
        self.output_cols = output_cols

    def fit(self, X, y=None):
        return self

    @staticmethod
    def _safe_div(numer: pd.Series, denom: pd.Series, fill: float = 0.0) -> pd.Series:
        denom = denom.astype(float).where(lambda s: s != 0.0, np.nan)
        numer = numer.astype(float)
        return (numer / denom).fillna(fill)

    def transform(self, X):
        assert isinstance(X, pd.DataFrame), "ChronosFeatureEngineerB expects pandas.DataFrame"

        missing = [c for c in self.required_raw if c not in X.columns]
        if missing:
            logger.error("Missing required columns: %s", missing)
        assert not missing, f"Missing required columns: {missing}"

        df = X.copy()

        df["mark_minus_close"] = df["markPrice"].astype(float) - df["close"].astype(float)
        df["taker_buy_share"] = self._safe_div(df["taker_buy_base"], df["volume"])

        denom_bs = df["buyVol"].astype(float) + df["sellVol"].astype(float)
        df["buy_share"] = self._safe_div(df["buyVol"], denom_bs)

        df["net_taker_flow"] = df["buyVol"].astype(float) - df["sellVol"].astype(float)
        df["avg_open_price"] = self._safe_div(df["sumOpenInterestValue"], df["sumOpenInterest"])
        df["oi_to_volume"] = self._safe_div(df["sumOpenInterest"], df["volume"])

        out = df.loc[:, list(self.output_cols)].astype(float)
        return out
