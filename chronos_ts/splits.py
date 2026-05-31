from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class TimeRangeSplitConfig:
    train_frac: float = 0.7
    val_frac: float = 0.15
    test_frac: float = 0.15

    def validate(self) -> None:
        total = self.train_frac + self.val_frac + self.test_frac
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"Fractions must sum to 1.0, got {total}")


def time_fraction_split(df: pd.DataFrame, config: TimeRangeSplitConfig, ts_col: str = "ts") -> dict[str, pd.DataFrame]:
    config.validate()
    df = df.sort_values(ts_col).reset_index(drop=True)
    n = len(df)
    n_train = int(n * config.train_frac)
    n_val = int(n * config.val_frac)
    n_test = n - n_train - n_val
    if min(n_train, n_val, n_test) <= 0:
        raise ValueError(f"Split too small: train={n_train}, val={n_val}, test={n_test}")

    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train : n_train + n_val].copy()
    test_df = df.iloc[n_train + n_val :].copy()
    return {"train": train_df, "val": val_df, "test": test_df}
