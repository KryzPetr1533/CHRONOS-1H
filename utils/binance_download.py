from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

FAPI_BASE = os.getenv("BINANCE_FAPI_BASE", "https://fapi.binance.com")
VISION_BASE = os.getenv("BINANCE_VISION_BASE", "https://data.binance.vision")

# ----------------------------
# Helpers
# ----------------------------
def interval_to_ms(interval: str) -> int:
    mapping = {
        "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
        "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000, "6h": 21_600_000,
        "8h": 28_800_000, "12h": 43_200_000, "1d": 86_400_000,
    }
    if interval not in mapping:
        raise ValueError(f"Unsupported interval: {interval}")
    return mapping[interval]

def to_ms(dt: datetime) -> int:
    return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)

def from_ms(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)

def parse_date(s: Optional[str]) -> Optional[datetime]:
    if s is None:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        return datetime.fromisoformat(s.replace("Z", "")).astimezone(timezone.utc)

def save_df(df: pd.DataFrame, path_csv: os.PathLike | str) -> None:
    os.makedirs(os.path.dirname(str(path_csv)), exist_ok=True)
    df.to_csv(path_csv, index=False)
    try:
        df.to_parquet(str(path_csv).rsplit(".", 1)[0] + ".parquet", index=False)
    except Exception:
        pass

def make_session() -> requests.Session:
    s = requests.Session()
    retry_strategy = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=32, pool_maxsize=32)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers.update({"User-Agent": "btc-forecast-utils/1.0"})
    return s

# ----------------------------
# Client
# ----------------------------
@dataclass
class BinancePublic:
    session: Optional[requests.Session] = None

    def __post_init__(self):
        self.s = self.session or make_session()

    # Low-level GET with retry
    @retry(wait=wait_exponential(multiplier=1, min=1, max=60),
           stop=stop_after_attempt(7),
           retry=retry_if_exception_type(requests.RequestException))
    def _get(self, url: str, params: Dict[str, Any]) -> Any:
        r = self.s.get(url, params=params, timeout=30)
        if r.status_code == 429:
            # Respect Retry-After if present
            try:
                import time
                time.sleep(int(r.headers.get("Retry-After", "5")))
            finally:
                r.raise_for_status()
        r.raise_for_status()
        return r.json()

    # ---------- KLINES ----------
    def fetch_klines(
        self, symbol: str, interval: str, start: datetime, end: Optional[datetime] = None,
        limit: int = 1500,
    ) -> pd.DataFrame:
        """USDS-M klines. limit<=1500. GET /fapi/v1/klines"""
        # Docs: :contentReference[oaicite:3]{index=3}
        end = end or datetime.now(timezone.utc)
        start_ms, end_ms = to_ms(start), to_ms(end)
        out: List[List[Any]] = []
        url = f"{FAPI_BASE}/fapi/v1/klines"

        while True:
            params = {"symbol": symbol, "interval": interval,
                      "startTime": start_ms, "endTime": end_ms, "limit": limit}
            data = self._get(url, params)
            if not data:
                break
            out.extend(data)
            last_open = data[-1][0]
            next_open = last_open + interval_to_ms(interval)
            if next_open > end_ms:
                break
            start_ms = next_open

        cols = ["open_time","open","high","low","close","volume","close_time","quote_volume",
                "num_trades","taker_buy_base","taker_buy_quote","ignore"]
        df = pd.DataFrame(out, columns=cols)
        for c in ["open","high","low","close","volume","quote_volume","taker_buy_base","taker_buy_quote"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
        return df

    # ---------- PREMIUM INDEX KLINES ----------
    def fetch_premium_index_klines(
        self, symbol: str, interval: str, start: datetime, end: Optional[datetime] = None,
        limit: int = 1500,
    ) -> pd.DataFrame:
        """GET /fapi/v1/premiumIndexKlines"""
        # Docs: :contentReference[oaicite:4]{index=4}
        url = f"{FAPI_BASE}/fapi/v1/premiumIndexKlines"
        end = end or datetime.now(timezone.utc)
        start_ms, end_ms = to_ms(start), to_ms(end)
        out: List[List[Any]] = []

        while True:
            data = self._get(url, {
                "symbol": symbol, "interval": interval,
                "startTime": start_ms, "endTime": end_ms, "limit": limit
            })
            if not data:
                break
            out.extend(data)
            last = data[-1][0]
            next_open = last + interval_to_ms(interval)
            if next_open > end_ms:
                break
            start_ms = next_open

        cols = ["open_time","open","high","low","close","volume","close_time","quote_volume",
                "num_trades","taker_buy_base","taker_buy_quote","ignore"]
        df = pd.DataFrame(out, columns=cols)
        for c in ["open","high","low","close","volume","quote_volume","taker_buy_base","taker_buy_quote"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
        return df

    # ---------- FUNDING ----------
    def fetch_funding_rates(
        self, symbol: str, start: datetime, end: Optional[datetime] = None, limit: int = 1000
    ) -> pd.DataFrame:
        """GET /fapi/v1/fundingRate (ascending order, max 1000 per page)"""
        # Docs: :contentReference[oaicite:5]{index=5}
        url = f"{FAPI_BASE}/fapi/v1/fundingRate"
        end = end or datetime.now(timezone.utc)
        start_ms, end_ms = to_ms(start), to_ms(end)
        out: List[Dict[str, Any]] = []

        while True:
            chunk = self._get(url, {"symbol": symbol, "startTime": start_ms, "endTime": end_ms, "limit": limit})
            if not chunk:
                break
            out.extend(chunk)
            last = int(chunk[-1]["fundingTime"])
            if last >= end_ms or len(chunk) < limit:
                break
            start_ms = last + 1

        df = pd.DataFrame(out)
        if not df.empty:
            df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
            for c in ["fundingRate","markPrice"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
        return df

    # ---------- TAKER FLOWS ----------
    def fetch_taker_buy_sell_volume(
        self, symbol: str, period: str = "1h", limit: int = 500,
        start: Optional[datetime] = None, end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        USDⓈ-M taker buy/sell volume ratio & volumes (~30 days).
        Primary: GET /futures/data/takerlongshortRatio (buyVol, sellVol, buySellRatio).
        """
        # Docs: :contentReference[oaicite:6]{index=6}
        url_usdm = f"{FAPI_BASE}/futures/data/takerlongshortRatio"
        params: Dict[str, Any] = {"symbol": symbol, "period": period, "limit": limit}
        if start: params["startTime"] = to_ms(start)
        if end:   params["endTime"]   = to_ms(end)

        data = self._get(url_usdm, params)
        df = pd.DataFrame(data)
        if df.empty:
            return df
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        for c in ["buyVol","sellVol","buySellRatio"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df

    def fetch_taker_long_short_ratio(
        self, symbol: str, period: str = "1h", limit: int = 500,
        start: Optional[datetime] = None, end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Global Long/Short Account Ratio (~30 days).
        GET /futures/data/globalLongShortAccountRatio
        """
        # Docs: :contentReference[oaicite:7]{index=7}
        url = f"{FAPI_BASE}/futures/data/globalLongShortAccountRatio"
        params: Dict[str, Any] = {"symbol": symbol, "period": period, "limit": limit}
        if start: params["startTime"] = to_ms(start)
        if end:   params["endTime"]   = to_ms(end)
        data = self._get(url, params)
        df = pd.DataFrame(data)
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            for c in ["longShortRatio","longAccount","shortAccount"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
        return df

    # ---------- OPEN INTEREST ----------
    def fetch_open_interest_now(self, symbol: str) -> pd.DataFrame:
        """GET /fapi/v1/openInterest"""
        # Docs: :contentReference[oaicite:8]{index=8}
        url = f"{FAPI_BASE}/fapi/v1/openInterest"
        data = self._get(url, {"symbol": symbol})
        return pd.DataFrame([data])

    def fetch_open_interest_hist(
        self, symbol: str, period: str = "1h", limit: int = 500,
        start: Optional[datetime] = None, end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """GET /futures/data/openInterestHist (history window ~30d)"""
        # Docs: :contentReference[oaicite:9]{index=9}
        url = f"{FAPI_BASE}/futures/data/openInterestHist"
        params: Dict[str, Any] = {"symbol": symbol, "period": period, "limit": limit}
        if start: params["startTime"] = to_ms(start)
        if end:   params["endTime"]   = to_ms(end)
        data = self._get(url, params)
        df = pd.DataFrame(data)
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            for c in ["sumOpenInterest","sumOpenInterestValue"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
        return df

    # ---------- data.binance.vision monthly 1h ----------
    def fetch_vision_monthly_klines_1h(
        self, symbol: str, start_month: str, end_month: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Download monthly ZIPs with 1h klines from data.binance.vision and concatenate.
        Month format: 'YYYY-MM'.
        """
        # Directory structure reference: :contentReference[oaicite:10]{index=10}
        import io, zipfile

        def month_iter(start_m: str, end_m: Optional[str]):
            y, m = map(int, start_m.split("-"))
            end_dt = (datetime.now(timezone.utc) if end_m is None
                      else datetime.strptime(end_m, "%Y-%m").replace(tzinfo=timezone.utc))
            cur = datetime(y, m, 1, tzinfo=timezone.utc)
            while cur <= end_dt:
                yield cur.strftime("%Y-%m")
                # +1 month
                y2, m2 = (cur.year + (1 if cur.month == 12 else 0), 1 if cur.month == 12 else cur.month + 1)
                cur = cur.replace(year=y2, month=m2)

        frames: List[pd.DataFrame] = []
        base_prefix = f"{VISION_BASE}/data/futures/um/monthly/klines/{symbol}/1h"
        for mon in month_iter(start_month, end_month):
            url = f"{base_prefix}/{symbol}-1h-{mon}.zip"
            try:
                resp = self.s.get(url, timeout=60)
                resp.raise_for_status()
                with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                    csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
                    if not csv_names:
                        continue
                    with zf.open(csv_names[0]) as fh:
                        df = pd.read_csv(fh, header=None)
                        df.columns = ["open_time","open","high","low","close","volume","close_time","quote_volume",
                                      "num_trades","taker_buy_base","taker_buy_quote","ignore"]
                        for c in ["open","high","low","close","volume","quote_volume","taker_buy_base","taker_buy_quote"]:
                            df[c] = pd.to_numeric(df[c], errors="coerce")
                        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
                        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
                        frames.append(df)
            except Exception as e:
                print(f"Skip {url}: {e}")

        if not frames:
            return pd.DataFrame(columns=["open_time","open","high","low","close","volume","close_time","quote_volume",
                                         "num_trades","taker_buy_base","taker_buy_quote","ignore"])
        return pd.concat(frames, ignore_index=True).sort_values("open_time").reset_index(drop=True)
