# CHRONOS-1H — Improvement Plans

This folder contains **four sequential plans** covering two tasks. Each task is
split into **Phase 1 (foundation / minimal viable change)** and **Phase 2 (builds
on Phase 1)**. Execute Phase 1 fully, then Phase 2.

| # | File | Task | Phase |
|---|------|------|-------|
| 1 | [`task1-phase1-classification-reframing.md`](task1-phase1-classification-reframing.md) | Reframe the ML problem | Foundation: classification + tokenization on existing 1h data |
| 2 | [`task1-phase2-event-based-representation.md`](task1-phase2-event-based-representation.md) | Reframe the ML problem | Build-on: event bars, microstructure features, tokenized sequence models |
| 3 | [`task2-phase1-mlflow-notebook-versioning.md`](task2-phase1-mlflow-notebook-versioning.md) | Experiment versioning | Foundation: MLflow + S3 stack, notebook-driven flow, PRD model |
| 4 | [`task2-phase2-hydra-cli-bonus.md`](task2-phase2-hydra-cli-bonus.md) | Experiment versioning | Build-on (bonus): Hydra config + CLI `.py` training & PRD inference |

## Why these two tasks are coupled

Task 1 changes **what** we model (a classification/tokenization problem with a
real, defensible signal — volatility/direction — instead of near-white-noise mean
return). Task 2 changes **how** we track and ship it (MLflow + S3 versioning).

The bridge between them is **Hydra**: `hydra-core==1.3.2` is already listed in
`requirements.txt` but is currently **unused** (configs are loaded with a manual
`yaml.safe_load`). Task 1 Phase 1 introduces a Hydra config tree for the new
classification trainer. Task 2 Phase 2 (the bonus) then reuses that exact tree to
run experiments from the CLI with MLflow logging. Doing Task 1 Phase 1 first makes
the Task 2 bonus almost free.

**Recommended global order:** T1-P1 → T2-P1 → T1-P2 → T2-P2. (You get a tracked,
versioned model early, then deepen the modelling, then automate.) The plans are
written so they can also be done strictly 1→2→3→4.

## The `legacy/` folder (deferred — not created yet)

Per the request, *everything that gets replaced* is moved into a top-level
`legacy/` folder rather than deleted, preserving git history via `git mv`. The
folder mirrors the original layout so provenance is obvious:

```
legacy/
  regression_mean/        # next-hour-return regression scripts/configs (T1-P1)
    configs/
    scripts/
  template_tutorials/     # apple/cnn/bert MLflow demo notebooks + demo data (T2-P1)
  notebooks_regression/   # superseded regression notebooks (T1-P1/P2)
  README.md               # index: what moved, from where, why, and what replaced it
```

Each plan has a **"Legacy moves"** section listing the exact files, the reason, and
the replacement. Nothing is moved until the relevant phase is approved and started.
**No files have been moved as part of writing these plans.**

## Ground truth captured during exploration (so the plans are accurate)

- **Data is hourly bars**, not ticks. `data/btcusdt_1h_merged.csv` = 24,883 rows
  (2023-01-01 →), columns: OHLCV, `quote_volume`, `num_trades`, taker buy
  base/quote, `log_ret_1h`, `premium_close`, `fundingRate`, `markPrice`, `buyVol`,
  `sellVol`, `buySellRatio`, `taker_imbalance`, `sumOpenInterest`,
  `sumOpenInterestValue`. True dollar/volume/tick bars and "time since last trade"
  therefore require **finer source data** (1m klines / aggTrades) — that is exactly
  why it is deferred to Task 1 **Phase 2**, not Phase 1.
- **Current target** (`chronos_ts/dataset.py`): `target_log_ret_1h = log_ret_1h(t+1)`
  — a one-step-ahead **regression** target. EDA + every model (Ridge, CatBoost,
  SARIMAX, GRU, TimeXer, Chronos-2) confirm it is ≈ white noise (directional
  accuracy 0.49–0.53, R² ≈ 0). The one positive result is **HAR volatility**
  (test R² ≈ 0.085, Pearson ≈ 0.33). This is the empirical basis for reframing.
- **Trainer** (`chronos_ts/trainer.py`) is regression-only (`RegressorMixin`,
  `mean_squared_error` CV). It needs a classification sibling, not a rewrite.
- **Metrics** (`chronos_ts/metrics.py`) already compute directional accuracy and a
  sign-strategy Sharpe — useful, but framed around a continuous prediction. A
  classification metrics module is needed.
- **Serving app** (`app/`) loads `artifacts/best_enet_B_aggr.joblib` (a regression
  ElasticNet) via FastAPI `/forward`. It is out of scope for the modelling change
  but is the eventual consumer of the PRD model — noted where relevant.
- **MLflow template** (`template-docker-mlflow-s3/`) is a **separate git repo**
  (own `.git`) with Postgres + MLflow `v3.12.0` + MinIO + `mc` auto-bucket. Its
  notebooks already demonstrate the full `log_model → register → set PRD tag →
  set 'prd' alias → load models:/<name>@prd` cycle. Task 2 reuses this stack and
  this exact pattern, pointed at the CHRONOS model.
</content>
