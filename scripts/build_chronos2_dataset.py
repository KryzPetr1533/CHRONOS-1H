from __future__ import annotations

from pathlib import Path
import glob
import json
import numpy as np
import pandas as pd

IN_GLOB = "outputs/datasets/*timexer.csv"
OUT_CSV = Path("outputs/datasets/chronos2_panel.csv")
OUT_META = Path("outputs/datasets/chronos2_panel.meta.json")
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

KNOWN_COVS = ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]

PAST_COVS = [
    "log_ret_1h",
    "premium_chg",
    "fundingRate",
    "vol_chg",
    "trades_chg",
    "taker_buy_share",
    "rv_24",
    "fundingRate_missing",
    "taker_buy_share_missing",
]

def infer_item_id(df: pd.DataFrame, path: Path) -> str:
    if "series_id" in df.columns and df["series_id"].nunique() == 1:
        return str(df["series_id"].iloc[0])
    stem = path.stem.lower()
    return stem.replace("_timexer", "").replace("btcusdt_", "BTCUSDT_")

def main() -> None:
    paths = [Path(p) for p in glob.glob(IN_GLOB)]
    if not paths:
        raise FileNotFoundError(f"No input files matched {IN_GLOB}")

    frames = []
    for path in sorted(paths):
        df = pd.read_csv(path, parse_dates=["ds"]).sort_values("time_idx").reset_index(drop=True)
        item_id = infer_item_id(df, path)

        # minimal required columns
        cols = ["ds", "target"] + KNOWN_COVS + PAST_COVS
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"{path} missing columns: {missing}")

        out = pd.DataFrame({
            "item_id": item_id,
            "timestamp": pd.to_datetime(df["ds"], utc=True).dt.tz_convert(None),
            "target": pd.to_numeric(df["target"], errors="coerce"),
        })

        for c in KNOWN_COVS + PAST_COVS:
            out[c] = pd.to_numeric(df[c], errors="coerce")

        frames.append(out)

    panel = pd.concat(frames, ignore_index=True)

    # sparse features
    if "fundingRate" in panel.columns:
        panel["fundingRate_missing"] = panel["fundingRate"].isna().astype(int)
        panel["fundingRate"] = panel.groupby("item_id")["fundingRate"].ffill().fillna(0.0)

    if "taker_buy_share" in panel.columns:
        panel["taker_buy_share_missing"] = panel["taker_buy_share"].isna().astype(int)
        panel["taker_buy_share"] = panel.groupby("item_id")["taker_buy_share"].ffill().fillna(0.0)

    panel = panel.replace([np.inf, -np.inf], np.nan).dropna().sort_values(["item_id", "timestamp"]).reset_index(drop=True)

    panel.to_csv(OUT_CSV, index=False)

    meta = {
        "csv": str(OUT_CSV),
        "n_rows": int(len(panel)),
        "n_items": int(panel["item_id"].nunique()),
        "items": sorted(panel["item_id"].unique().tolist())[:50],
        "known_covariates_names": KNOWN_COVS,
        "past_covariates": PAST_COVS,
        "timestamp_min": str(panel["timestamp"].min()),
        "timestamp_max": str(panel["timestamp"].max()),
    }
    OUT_META.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(meta, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()