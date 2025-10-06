# CHRONOS-1H

## Task

Forecast the **next 1-hour log-return** of **BTCUSDT** using **public Binance data** and deliver results via a **Telegram bot**.

## Plan

1. **Get data from sources (public, no keys)**
  Possible sources:
  * Historical spot/futures market data from `data.binance.vision` (daily/monthly dumps).
  * Futures OHLCV (1h bars) from `GET /fapi/v1/klines`.
  * Premium index klines (perp–spot basis) from `GET /fapi/v1/premiumIndexKlines`.
  * Funding rate history from  `GET /fapi/v1/fundingRate`.
  * Taker buy/sell volume (flow imbalance, latest ~30 days) from `GET /futures/data/takerBuySellVol` (and related taker ratio endpoints).
  * Open interest & OI stats from `GET /fapi/v1/openInterest`, `GET /futures/data/openInterestHist`. 

2. **Estimate baseline**
**Random Walk** (predict 0 return) and **Linear Regression**.

3. **Fit and finetune models**
  Models to try:
  * **Gradient Boosting** (XGBoost/LightGBM/CatBoost) on hourly tabular features.
  * **LSTM** on rolling windows of hourly features (e.g., last 48–168 hours).

4. **Make telegram bot as API surface**
Api supports theese commands:
    * `/start` — register & brief help.
    * `/predict` — returns latest **1h forecast** (`pred_return`, `prob_up`, timestamp).
    * `/subscribe` — opt-in to **hourly push** when a new bar closes.
    * `/unsubscribe` — stop notifications.
    * `/status` — model version & data freshness.
