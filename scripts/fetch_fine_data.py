"""
Fetch fine-grained Binance futures data (1m klines + optional aggTrades).

CLI:
    python scripts/fetch_fine_data.py --symbol BTCUSDT --days 30
    python scripts/fetch_fine_data.py --symbol BTCUSDT --days 7 --fetch-trades
    python scripts/fetch_fine_data.py --symbols BTCUSDT ETHUSDT --days 14

Notebook / script import:
    from scripts.fetch_fine_data import FineDataFetcher
    df = FineDataFetcher().run(symbol='BTCUSDT', days=30)
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

BASE_URL = "https://fapi.binance.com"


class FineDataFetcher:
    """
    Downloads Binance Futures public data.

    Parameters
    ----------
    output_dir   : root directory for partitioned parquet files
    request_delay: seconds between API calls (rate-limit safety)
    """

    def __init__(
        self,
        output_dir: str = "data/raw_fine",
        request_delay: float = 0.3,
    ):
        self.output_dir = Path(output_dir)
        self.request_delay = request_delay

    # ------------------------------------------------------------------ #

    def run(
        self,
        symbol: str = "BTCUSDT",
        days: int = 30,
        end_date: Optional[datetime] = None,
        fetch_trades: bool = False,
    ) -> pd.DataFrame:
        """
        Download `days` days of 1m klines ending at `end_date` (default: now UTC).

        Writes:
          data/raw_fine/<SYMBOL>/1m/YYYY-MM-DD.parquet  (one file per day)
          data/raw_fine/<SYMBOL>/provenance.json

        Returns the concatenated DataFrame.
        """
        if end_date is None:
            end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days)

        print(f"Fetching {symbol} 1m klines: {start_date.date()} → {end_date.date()} ({days} days)")

        frames = []
        current = start_date
        while current.date() < end_date.date():
            day_df = self._fetch_day_klines(symbol, current)
            if day_df is not None and len(day_df) > 0:
                self._save_parquet(day_df, symbol, "1m", current)
                frames.append(day_df)
            current += timedelta(days=1)

        if fetch_trades:
            print(f"Fetching {symbol} aggTrades (last {days} days) ...")
            self._fetch_aggtrades(symbol, start_date, end_date)

        full_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        self._write_provenance(symbol, start_date, end_date, len(full_df), fetch_trades)

        print(f"Done: {len(full_df):,} 1m bars saved to {self.output_dir}/{symbol}/1m/")
        return full_df

    # ------------------------------------------------------------------ #
    # Klines
    # ------------------------------------------------------------------ #

    def _fetch_day_klines(self, symbol: str, day: datetime) -> Optional[pd.DataFrame]:
        start_ms = int(datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp() * 1000)
        end_ms   = start_ms + 86_400_000  # +24h

        rows = []
        limit = 1000
        open_time = start_ms

        while open_time < end_ms:
            batch = self._get_klines(symbol, "1m", open_time, min(open_time + limit * 60_000, end_ms), limit)
            if not batch:
                break
            rows.extend(batch)
            open_time = batch[-1][0] + 60_000
            if len(batch) < limit:
                break
            time.sleep(self.request_delay)

        if not rows:
            return None

        cols = ["open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_volume", "n_trades",
                "taker_buy_base", "taker_buy_quote", "ignore"]
        df = pd.DataFrame(rows, columns=cols)
        df["ts"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        for c in ["open", "high", "low", "close", "volume", "quote_volume",
                  "taker_buy_base", "taker_buy_quote"]:
            df[c] = df[c].astype(float)
        df["n_trades"] = df["n_trades"].astype(int)
        return df[["ts", "open", "high", "low", "close", "volume", "quote_volume",
                   "n_trades", "taker_buy_base", "taker_buy_quote"]].copy()

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=30))
    def _get_klines(self, symbol, interval, start_time, end_time, limit):
        r = requests.get(
            f"{BASE_URL}/fapi/v1/klines",
            params={"symbol": symbol, "interval": interval,
                    "startTime": start_time, "endTime": end_time, "limit": limit},
            timeout=15,
        )
        r.raise_for_status()
        return r.json()

    # ------------------------------------------------------------------ #
    # aggTrades (optional, heavy)
    # ------------------------------------------------------------------ #

    def _fetch_aggtrades(self, symbol: str, start: datetime, end: datetime) -> None:
        """Fetch aggTrades and save as parquet. Warning: very large for long ranges."""
        out_dir = self.output_dir / symbol / "aggtrades"
        out_dir.mkdir(parents=True, exist_ok=True)

        start_ms = int(start.timestamp() * 1000)
        end_ms   = int(end.timestamp() * 1000)
        batch_ms = 3_600_000  # 1-hour batches to stay within API limits

        current = start_ms
        all_frames = []
        while current < end_ms:
            batch_end = min(current + batch_ms, end_ms)
            rows = self._get_aggtrades(symbol, current, batch_end)
            if rows:
                df = pd.DataFrame(rows)
                df.columns = ["agg_id", "price", "qty", "first_id", "last_id",
                              "timestamp", "is_buyer_maker", "is_best_match"]
                df["ts"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                all_frames.append(df)
            current = batch_end
            time.sleep(self.request_delay)

        if all_frames:
            merged = pd.concat(all_frames, ignore_index=True)
            path = out_dir / f"{start.strftime('%Y-%m-%d')}_{end.strftime('%Y-%m-%d')}.parquet"
            merged.to_parquet(path, index=False)
            print(f"  aggTrades saved: {len(merged):,} rows → {path}")

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=30))
    def _get_aggtrades(self, symbol, start_time, end_time):
        r = requests.get(
            f"{BASE_URL}/fapi/v1/aggTrades",
            params={"symbol": symbol, "startTime": start_time, "endTime": end_time, "limit": 1000},
            timeout=15,
        )
        r.raise_for_status()
        return r.json()

    # ------------------------------------------------------------------ #
    # Persistence helpers
    # ------------------------------------------------------------------ #

    def _save_parquet(self, df: pd.DataFrame, symbol: str, interval: str, day: datetime) -> None:
        out_dir = self.output_dir / symbol / interval
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{day.strftime('%Y-%m-%d')}.parquet"
        df.to_parquet(path, index=False)

    def _write_provenance(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        n_rows: int,
        fetch_trades: bool,
    ) -> None:
        prov = {
            "symbol": symbol,
            "endpoint": f"{BASE_URL}/fapi/v1/klines",
            "interval": "1m",
            "start_utc": start.isoformat(),
            "end_utc": end.isoformat(),
            "n_rows": n_rows,
            "aggtrades_fetched": fetch_trades,
            "pulled_at": datetime.now(timezone.utc).isoformat(),
        }
        path = self.output_dir / symbol / "provenance.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(prov, indent=2), encoding="utf-8")
        print(f"Provenance: {path}")

    # ------------------------------------------------------------------ #
    # Load helpers
    # ------------------------------------------------------------------ #

    def load(self, symbol: str = "BTCUSDT", interval: str = "1m") -> pd.DataFrame:
        """Load all saved parquet files for a symbol + interval."""
        data_dir = self.output_dir / symbol / interval
        files = sorted(data_dir.glob("*.parquet")) if data_dir.exists() else []
        if not files:
            raise FileNotFoundError(f"No data at {data_dir}")
        return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #

def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Binance 1m klines (+ optional aggTrades).")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--symbols", nargs="*", help="Fetch multiple symbols")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--output-dir", default="data/raw_fine")
    parser.add_argument("--fetch-trades", action="store_true")
    args = parser.parse_args()

    symbols = args.symbols or [args.symbol]
    fetcher = FineDataFetcher(output_dir=args.output_dir)
    for sym in symbols:
        fetcher.run(symbol=sym, days=args.days, fetch_trades=args.fetch_trades)


if __name__ == "__main__":
    main()
