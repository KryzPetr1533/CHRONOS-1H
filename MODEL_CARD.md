# MODEL CARD — chronos_1h_prd

## Model identity

| Field | Value |
|---|---|
| Registry name | `chronos_1h_prd` |
| Alias | `prd` |
| Tag | `env=PRD` |
| MLflow experiment | `chronos-1h-classification` |
| Load URI | `models:/chronos_1h_prd@prd` |
| Run ID | `30df7f27995441b0a610ae808053a3ef` |
| Registry version | `3` (`make train-final`, 2026-06-03) |

## What it does

Predicts the **volatility regime** of the next 1-hour BTCUSDT candle as one of three classes:
- `0` — `low_vol`: next-bar realized volatility in the bottom tercile (train-set)
- `1` — `mid_vol`: middle tercile
- `2` — `high_vol`: top tercile

This is a **classification** model, not a raw return forecast. Directional (mean-return) forecasting was found to be near-white-noise (R²≈0 across all models — see README §7.1).

## Algorithm

**CatBoostClassifier** with:
- `auto_class_weights=Balanced`
- `loss_function=MultiClass`
- Grid-searched on `TimeSeriesSplit(cv=3)` over depth, learning_rate, n_estimators, l2_leaf_reg
- Features: 300 tabular features from `outputs/datasets/btcusdt_clf_core.csv` (lag/rolling on OHLCV, funding rate, premium index, taker buy share, OI)
- Temporal split: 70% train / 15% val / 15% test (no shuffle — time order preserved)

## Selection rationale

| Target | Test ROC-AUC | vs Baseline |
|---|---|---|
| `vol_regime` (chosen) | **~0.69–0.72** | baseline (majority) = 0.333 balanced-acc |
| `large_move` | ~0.62 | baseline = 0.500 |
| `direction` | ~0.506 | baseline = 0.500 |

Volatility is the only **forecastable** signal in the 1h BTCUSDT data — confirmed by HAR regression (R²=0.085, Pearson=0.33) and the classification ROC-AUC above. Direction is effectively unpredictable at the 1h horizon.

CatBoost is selected over LogReg for nonlinear feature interactions and better handling of the feature-null pattern in this dataset.

## Key metrics (`make train-final`, core dataset)

| Metric | Test | Val | Majority baseline (test) |
|---|---|---|---|
| balanced_accuracy | **0.527** | 0.580 | 0.333 |
| MCC | 0.267 | 0.377 | 0.000 |
| ROC-AUC (macro OvR) | **0.716** | 0.770 | — |
| trading_coverage | 0.20 | 0.26 | — |
| trading_hit_rate | 0.50 | 0.52 | — |

MLflow UI: http://localhost:5050/#/experiments/1/runs/30df7f27995441b0a610ae808053a3ef

## Data

- Source: Binance BTCUSDT public futures API (`fapi.binance.com/fapi/v1/klines`, 1h)
- Date range: 2023-01-01 → present (~24,700 rows)
- Feature engineering: `chronos_ts.dataset.ExperimentDatasetBuilder` (lag_hours=[1,2,3,6,12,24,48,72,168], rolling_windows=[6,24,72,168])
- Label: `LabelMaker(vol_regime, vol_window=6)` — edges fit on training split only

## Reproducibility

- Seed: 42 (CatBoost + numpy + random)
- Feature matrix checksum: see `outputs/datasets/btcusdt_clf_core.meta.json`
- Label bin edges: logged as `label_config.json` artifact in MLflow

## Known limitations

1. **Regime transition errors**: model fails at abrupt vol shocks (news events). Requires real-time external signals not available in Phase 1.
2. **Boundary ambiguity**: low/mid and mid/high boundaries are quantile-based; bars near the threshold are inherently uncertain.
3. **Sparse late-in-data features**: `mark_minus_close_lag_*` and similar columns are dropped due to >95% nulls in training data.
4. **No tick data**: Phase 2 will add event bars and microstructure features that may improve the representation.
