# Classification dataset EDA

**File:** `outputs/datasets/btcusdt_clf_core.csv`

## Shape
- Rows: 24,714
- Columns: 302
- Date range: 2023-01-08 00:00:00+00:00 → 2025-11-02 17:00:00+00:00

## Missingness (top 15)
- `oi_pct_lag_168`: 98.66%
- `oi_pct_roll_std_168`: 98.66%
- `oi_pct_roll_mean_168`: 98.66%
- `sumOpenInterest_lag_168`: 98.66%
- `sumOpenInterestValue_lag_168`: 98.66%
- `taker_imbalance_lag_168`: 98.66%
- `net_taker_flow_lag_168`: 98.66%
- `taker_imbalance_roll_std_168`: 98.65%
- `taker_imbalance_roll_mean_168`: 98.65%
- `oi_pct_lag_72`: 98.28%
- `oi_pct_roll_std_72`: 98.27%
- `sumOpenInterest_lag_72`: 98.27%
- `oi_pct_roll_mean_72`: 98.27%
- `sumOpenInterestValue_lag_72`: 98.27%
- `net_taker_flow_lag_72`: 98.27%

## Returns (`log_ret_1h`)
- mean: 0.000076
- std: 0.005007
- min / max: -0.064628 / 0.054145
- lag_1 vs shift(1) max |diff|: 0.00e+00 (expect ~0)

## Column dtypes
```
float64                300
datetime64[ns, UTC]      1
int64                    1
```
