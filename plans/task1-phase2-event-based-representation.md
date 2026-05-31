# Task 1 · Phase 2 — Event-based representation, microstructure features, tokenized sequence models

> **Builds directly on Phase 1.** Phase 1 fixed *what we predict* (classification /
> tokenized labels) on the existing 1h time grid. Phase 2 fixes *how we represent
> the data*: replace the rigid, uneven hourly grid with **event bars** (volume /
> dollar / imbalance bars), add real **microstructure features**, bring in
> **external market context**, and add a **tokenized causal-sequence model** that
> predicts the next return-token. The Phase-1 `LabelMaker`, `clf_metrics`,
> `ClassificationTrainer`, Hydra tree, and baselines are all **reused unchanged** —
> only the *feature matrix* and one new model type are added.
>
> **Why this needs new data:** true volume/dollar bars, "time since last trade",
> and order-flow imbalance cannot be reconstructed from 1h OHLCV. They require
> finer Binance data (1m klines and/or aggTrades). That ingestion is the gate for
> this phase and is why it was deferred from Phase 1.

---

## 1. Outcomes / definition of done

- A reproducible **fine-data ingestion** script (1m klines + aggTrades for BTCUSDT,
  optionally the panel assets).
- **Event-bar builders**: time, volume, dollar, and (optional) imbalance bars.
- A **microstructure + cross-asset feature layer** on top of event bars.
- Phase-1 labels/metrics/trainer re-run on the event-bar representation; leaderboard
  compares **1h-grid (P1)** vs **event-bar (P2)** head-to-head.
- A **tokenized causal-sequence model** (small Transformer/GRU over discretized
  return tokens) predicting the next token, evaluated with the Phase-1 token metrics.
- Optional **multi-asset panel** sharing one covariate schema (the README §7.3
  scaling direction), enabling cross-learning for Chronos-2.

---

## 2. Data ingestion (the gate)

### New file: `scripts/fetch_fine_data.py`
- Pulls, from Binance public futures (`fapi`) endpoints, with retry/backoff
  (project already depends on `tenacity`, `tqdm`, `requests`):
  - **1m klines** (`/fapi/v1/klines`, interval `1m`) — for volume/dollar bars and
    intrabar realized vol.
  - **aggTrades** (`/fapi/v1/aggTrades`) — for signed trade flow, time-since-trade,
    and true tick-derived imbalance. (Heavier; make it optional behind a flag.)
- Writes partitioned parquet under `data/raw_fine/BTCUSDT/{1m,aggtrades}/...`.
- Documents provenance (symbol, endpoint, date range, pull timestamp) into a
  sidecar JSON — feeds Task 2's "describe the data / how it was obtained" requirement.

**Reproducibility:** record exact endpoint params + UTC range; data is append-only
and content-addressed by date partition. Note: aggTrades for the full 2023→ range
is large; default the script to a recent window and make the range a config field.

---

## 3. Event-bar builders

### New file: `chronos_ts/bars.py`
Each builder consumes 1m (or aggTrade) records and emits variable-width bars with a
consistent schema (`bar_start`, `bar_end`, OHLC, `volume`, `dollar_volume`,
`n_trades`, `buy_volume`, `sell_volume`, `dt_seconds`).

| Builder | Rule | Addresses (user ask) |
|---|---|---|
| `TimeBarBuilder` | fixed interval (sanity baseline = current 1h) | — |
| `VolumeBarBuilder` | close a bar when cumulative `volume ≥ V` | "volume bands" |
| `DollarBarBuilder` | close a bar when cumulative `dollar_volume ≥ D` | "dollar bars" |
| `ImbalanceBarBuilder` *(optional)* | close on cumulative signed-flow imbalance threshold (López de Prado style) | "order imbalance" |

`V`/`D` thresholds chosen so the *average* bar ≈ 1h of activity (keeps sample size
comparable to the 24.7k-row baseline and keeps Phase-1 horizons meaningful).

---

## 4. Microstructure + cross-asset feature layer

### New file: `chronos_ts/event_features.py`
Extends the Phase-1 feature philosophy (the existing `dataset.py` lag/rolling
machinery is reused where possible) with event-aware features:

- **Returns over multiple horizons** — bar-count and clock-time horizons.
- **Rolling statistics** — mean/std/skew/kurt over N bars (reuse `dataset.py`
  `_add_rolling_features` idiom).
- **Volatility characteristics** — realized vol from 1m sub-bars within each event
  bar; Parkinson & Garman–Klass estimators from OHLC; vol-of-vol.
- **Momentum** — multi-horizon return signs / z-scored cumulative returns.
- **Lag features** — reuse `dataset.py` `_add_lagged_features`.
- **Spread / volume / order flow** — `dt_seconds` (**time since last trade /
  inter-bar time**), signed-volume imbalance, taker buy share (already derived in
  `dataset.py:111`), buy/sell run length, VPIN-like bucketed imbalance.
- **External market indices** — ETHUSDT (and optionally BNB/SOL/XRP/DOGE per README
  §7.3) returns/vol as covariates; BTC dominance proxy; cross-asset funding/OI.

All features are **strictly causal** (computed from data up to `bar_end`), and the
Phase-1 train-only fitting discipline (§2 of P1) is enforced for any normalization.

---

## 5. Re-run Phase-1 stack on event bars (no new modelling code)

- New `conf/data/btcusdt_dollar.yaml`, `conf/data/btcusdt_volume.yaml` data groups
  pointing at the event-bar feature matrices.
- Run `scripts/train_classifier.py data=btcusdt_dollar label=vol_regime model=catboost`
  etc. — the Phase-1 trainer/labels/metrics are untouched.
- Extend `notebooks/classification_report.ipynb` to a **representation comparison**:
  for each target family, `1h-grid` vs `volume-bar` vs `dollar-bar` on identical
  metrics. This directly answers the user's hypothesis that the bottleneck is
  representation/features, not models.

---

## 6. Tokenized causal-sequence model (the "tokenization" idea, extended)

Phase 1 produced `return_token` *labels*. Phase 2 adds a model that consumes a
**sequence of past tokens** (plus a few continuous covariates) and predicts the
**next token** — i.e. a tiny language-model-style head over discretized returns.

### New file: `scripts/train_token_transformer.py`
- Vocabulary = Phase-1 `LabelMaker(return_token)` bins (train-only edges).
- Inputs: window of past `K` tokens + optional continuous covariates per step.
- Model: small causal Transformer (or GRU fallback) → softmax over vocab.
- Trained with cross-entropy; evaluated with the **same** `clf_metrics`
  (multiclass) + token-level confident-accuracy and the abstention trading metric.
- Reuses `splits.py` for the temporal split and the artifact-saving convention.

This reuses the **existing** sequence-model scripts as a starting point rather than
writing from scratch: `scripts/train_seq.py` (GRU/LSTM) and `scripts/train_timexer.py`
are adapted from regression heads to a classification/token head.

---

## 7. Optional: multi-asset panel for Chronos-2 (README §7.3 scaling bet)

- `scripts/build_panel_dataset.py` builds a long-format panel (BTC + ETH + BNB +
  SOL + XRP + DOGE) with one shared covariate schema and the Phase-1 labels.
- Adapt the existing `scripts/build_chronos2_dataset.py` / `scripts/train_chronos2.py`
  to the **classification/token** target and cross-learning across items.
- Keep the volatility model alongside as a confidence signal (README §7.2).

This is explicitly optional/stretch; the event-bar + tokenized-sequence work is the
core of Phase 2.

---

## 8. Legacy moves (executed only when this phase starts)

By Phase 2 the **1h-grid regression sequence/transformer scripts are superseded** by
their classification/token reworks. Move the originals once the reworks pass smoke
tests (so we always keep a working reference until then):

| Move to `legacy/regression_seq/` | Why | Replaced by |
|---|---|---|
| `scripts/train_patchtst.py` | regression PatchTST; README notes impl rejected future exog | dropped (no replacement) — documented dead-end |
| `scripts/train_timexer.py` (original) | regression TimeXer (dir-acc 0.486) | classification/token TimeXer rework |
| `scripts/train_seq.py` (original) | regression GRU/LSTM (dir-acc 0.493) | `scripts/train_token_transformer.py` |
| `scripts/build_patchtst_dataset.py`, `build_timexer_dataset.py` | regression dataset builders | event-bar + token builders |
| superseded regression notebooks (`notebooks/train.ipynb` regression cells) | regression EDA/training | `classification_report.ipynb` |

**Kept:** `train_chronos2.py` / `build_chronos2_dataset.py` (adapted, not removed —
panel route), `train_har_vol.py` / `train_garch.py` (volatility confidence signal),
all Phase-1 modules. `legacy/README.md` updated with each move.

---

## 9. Step-by-step execution checklist
1. `scripts/fetch_fine_data.py` → pull 1m klines (aggTrades behind a flag); write provenance JSON.
2. `chronos_ts/bars.py` → time/volume/dollar (+optional imbalance) builders + tests that bars are causal & non-overlapping.
3. `chronos_ts/event_features.py` → microstructure + cross-asset features (reuse `dataset.py` lag/rolling helpers).
4. New `conf/data/*` groups for volume/dollar bars.
5. Re-run Phase-1 classifier across representations; extend the report to a representation comparison.
6. `scripts/train_token_transformer.py` (adapt `train_seq.py`); evaluate with `clf_metrics`.
7. (Optional) panel dataset + Chronos-2 classification adaptation.
8. `legacy/` moves (§8) + `legacy/README.md` update; refresh root `README.md`.

## 10. Risks & mitigations
- **aggTrades volume/cost** → default to a recent window; make range a config; volume/dollar bars from 1m klines work without aggTrades for a first pass.
- **Event bars change sample size/horizon meaning** → calibrate `V`/`D` to ≈1h average; report bar-count distribution.
- **Cross-asset look-ahead** → align all assets on `bar_end`; lag external indices by ≥1 bar.
- **Tokenized model overfit** → small model, dropout, early stopping on val log-loss; compare to Phase-1 tabular token classifier as a baseline.

## 11. Deliverables
`scripts/fetch_fine_data.py`, `chronos_ts/bars.py`, `chronos_ts/event_features.py`,
`conf/data/btcusdt_{volume,dollar}.yaml`, `scripts/train_token_transformer.py`,
extended `classification_report.ipynb` (representation comparison),
`outputs/reports/phase2_representation_leaderboard.csv`, (optional) panel scripts,
populated `legacy/regression_seq/**`, updated `README.md`.
</content>
