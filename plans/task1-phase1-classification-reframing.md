# Task 1 · Phase 1 — Reframe to classification + return tokenization (on existing 1h data)

> **Goal of this phase:** stop predicting the (near-white-noise) next-hour mean
> return as a regression, and instead predict **labels that are actually
> learnable**: direction, large-move, volatility regime, multi-horizon direction,
> and a discretized "return token". All on the data we *already have*
> (`data/btcusdt_1h_merged.csv`) — **no new data ingestion** (that is Phase 2).
>
> **Why now:** EDA + every regression model in the README converge on the same
> result — one-step return mean is ~unpredictable (dir-acc 0.49–0.53, R²≈0), while
> **volatility is forecastable** (HAR: R²≈0.085, Pearson≈0.33). Reframing the
> *target* is the highest-leverage change and is independent of new data.

---

## 1. Outcomes / definition of done

- A new label layer turns the merged dataset into **5 target families** (below).
- A classification trainer mirrors the existing `TabularTrainer` API but for
  classifiers, with proper time-series CV, class weighting, and probability
  calibration.
- A classification metrics module (accuracy, balanced-acc, F1, ROC/PR-AUC, log
  loss, MCC, confusion matrix, **confident-subset directional accuracy**, and a
  trading metric with an abstention band).
- **Hydra** config tree drives the new trainer (first real use of the already-listed
  `hydra-core`), establishing the structure reused by Task 2 Phase 2.
- A comparison report/notebook ranking each (target family × model) vs honest
  baselines.
- Superseded mean-return regression configs/scripts moved to `legacy/`.

**Success bar (realistic, not aspirational):** beat the relevant baseline on the
**test** split by a margin that survives the val split too, on at least the
*volatility-regime* and *large-move* targets. Direction is allowed to stay weak —
the point is to *measure it honestly* and surface where signal exists.

---

## 2. New target families (the core of the reframe)

All derived from `log_ret_1h` and its forward shift. Implemented in a new
`chronos_ts/labels.py`. A **dead-zone** (configurable) avoids labelling noise as
signal.

| Family | Target | Definition | Type |
|---|---|---|---|
| `direction` | `y_dir` | `sign(ret_{t+1})`, dropping `|ret_{t+1}| < ε` rows (dead-zone) | binary |
| `large_move` | `y_move` | `1` if `|ret_{t+1}| > τ` else `0`; `τ` = rolling quantile (e.g. 80th pct of trailing `|ret|`) | binary |
| `vol_regime` | `y_vol` | tercile of forward realized vol `rv_{t+1..t+H}` → {low, mid, high} | 3-class |
| `horizon_dir` | `y_dir_H` | `sign(Σ ret_{t+1..t+H})` for `H ∈ {1,3,6,12,24}` | binary × H |
| `return_token` | `y_tok` | quantile-binned `ret_{t+1}` into `K` buckets (e.g. K=5/7/9) — **the "tokenization" idea** | K-class |

Design notes:
- **Leakage discipline:** all thresholds/quantiles (`τ`, tercile edges, token bin
  edges) are fit **on the training split only** and applied forward. `labels.py`
  exposes `fit(train_df)` → edges, then `transform(df)`. This is the single most
  important correctness property of the phase.
- **Class balance:** report base rates; pass `class_weight="balanced"` (or
  CatBoost `auto_class_weights`) by default; for `return_token`, quantile binning
  yields near-uniform classes by construction.
- **Dead-zone `ε`** for `direction` defaults to a small fraction of train-set
  return std (e.g. `0.1·σ`); also produce the no-dead-zone variant for comparison.

### New file: `chronos_ts/labels.py`
```python
@dataclass
class LabelConfig:
    target_family: str          # direction|large_move|vol_regime|horizon_dir|return_token
    return_col: str = "log_ret_1h"
    horizon: int = 1            # H for horizon_dir / vol_regime window
    dead_zone_sigma: float = 0.1
    move_quantile: float = 0.8
    n_tokens: int = 5
    vol_window: int = 6

class LabelMaker:
    def fit(self, train_df) -> "LabelMaker": ...   # learn edges on TRAIN only
    def transform(self, df) -> pd.Series: ...       # produce label column
    def class_names(self) -> list[str]: ...
    def describe(self) -> dict: ...                 # edges, base rates -> for MLflow later
```

---

## 3. New classifier trainer (sibling to `TabularTrainer`, not a rewrite)

### New file: `chronos_ts/classification.py`
- `ClassifierFactory.make(name)` for: `logreg` (Pipeline: median impute → scale →
  `LogisticRegression`), `catboost` (`CatBoostClassifier`), `lightgbm`
  (`LGBMClassifier`), `xgboost` (`XGBClassifier`). Mirror the existing
  `ModelFactory` style in `trainer.py:54`.
- `ClassifierFactory.default_param_grid(name)` — analogous to the existing grids.
- `ClassificationTrainer.run()` mirrors `TabularTrainer.run()` (`trainer.py:150`)
  but:
  - reuses `time_fraction_split` (`splits.py`) for the train/val/test split,
  - reuses `TimeSeriesSplit` for the inner CV, **scoring on log-loss or balanced
    accuracy** instead of `neg_mean_squared_error`,
  - wraps the final estimator in `CalibratedClassifierCV` (prefit, on val) so
    probabilities are usable for the abstention/trading metric,
  - persists: `*_model.joblib`, `*_metrics.json`, `*_test_predictions.csv`
    (now with `y_true`, `y_pred`, `proba_*`), and `*_confusion.csv`.

Reuse, don't duplicate: `splits.py`, the imputation/scaling idiom, and the JSON/
artifact-saving block from `trainer.py:209-227`. Factor the shared split + feature
selection (`trainer.py:155-166`) into a small helper imported by both trainers if
convenient, but a parallel class is acceptable and lower-risk.

---

## 4. Classification metrics

### New file: `chronos_ts/clf_metrics.py`
`evaluate_classification(y_true, y_pred, proba, class_names)` →
- `accuracy`, `balanced_accuracy`, `f1_macro`, `f1_weighted`, `mcc`,
- `roc_auc` / `pr_auc` (binary; one-vs-rest macro for multiclass),
- `log_loss`,
- **`confident_directional_accuracy`** — accuracy on the top-X% most confident
  predictions (`|p−0.5|` for binary), the honest analogue of the existing
  `top20_directional_accuracy` in `metrics.py:55`,
- **trading metric with abstention** — only act when
  `max_class_proba ≥ θ`; report coverage, hit-rate, and the sign-strategy mean /
  Sharpe (reuse the Sharpe annualization idiom in `metrics.py:60-62`),
- `confusion_matrix` (returned as a labelled DataFrame, saved as CSV + later a PNG).

---

## 5. Honest baselines (must beat these, per target)

Implemented as part of the trainer's report (no separate model files needed):

| Target | Baselines |
|---|---|
| `direction` / `horizon_dir` | majority class; "persist previous direction" (momentum); always-up (BTC drift) |
| `large_move` | majority class; "previous bar was a large move" (vol clustering persistence) |
| `vol_regime` | majority class; "persist previous regime" |
| `return_token` | majority bucket; empirical-frequency sampler (log-loss reference) |

These mirror — and make explicit — the baseline comparison the README already does
for regression. They are also the **baseline** that Task 2 requires for the
"compare with baseline" deliverable.

---

## 6. Hydra config tree (first real use of `hydra-core`)

Create a `conf/` tree (Hydra's convention) — this is the structure Task 2 Phase 2
reuses verbatim:
```
conf/
  config.yaml                 # defaults: data, label, model, split
  data/btcusdt_core.yaml      # input_csv, feature columns, ts_col, drop_cols
  label/direction.yaml        # target_family + LabelConfig fields
  label/large_move.yaml
  label/vol_regime.yaml
  label/horizon_dir.yaml
  label/return_token.yaml
  model/logreg.yaml
  model/catboost.yaml
  model/lightgbm.yaml
  split/default.yaml          # 0.7/0.15/0.15, cv_splits
```
The existing plain-YAML loader in `scripts/train_model.py:12` stays for the legacy
regression path; the new classifier script uses `@hydra.main`.

### New file: `scripts/train_classifier.py`
```python
@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # build LabelMaker(cfg.label) -> fit on train -> attach label column
    # ClassificationTrainer(cfg) -> run() -> print + save metrics/artifacts
```
Run: `python scripts/train_classifier.py label=vol_regime model=catboost`
Sweep: `python scripts/train_classifier.py -m label=direction,large_move,vol_regime model=logreg,catboost`

---

## 7. Comparison report

A new `notebooks/classification_report.ipynb` (or `scripts/report_classification.py`)
that loads all `*_metrics.json`, builds a leaderboard (target × model × split),
plots confusion matrices and confident-accuracy-vs-coverage curves, and writes
`outputs/reports/phase1_leaderboard.csv`. This is also the input to Task 2's
"select & justify the best model" step.

---

## 8. Legacy moves (executed only when this phase starts)

The **mean-return regression formulation is what gets replaced.** Move with
`git mv`; keep volatility work (it is reused for `vol_regime`).

| Move to `legacy/regression_mean/` | Why | Replaced by |
|---|---|---|
| `configs/train_ridge_core.yaml` | next-hour-return regression | `conf/label/*` + `train_classifier.py` |
| `configs/train_catboost_core.yaml` | next-hour-return regression | same |
| `configs/train_catboost_rich.yaml` | next-hour-return regression (tiny-sample) | same |
| `configs/train_enet_b_aggr.yaml` | regression for the serving ElasticNet | superseded once PRD classifier exists (T2) |
| `scripts/train_sarimax.py` | mean-return AR model (README §5.2, didn't work) | `vol_regime`/`direction` classifiers |
| `scripts/train_enet_b_aggr.py` | mean-return regression (app model) | PRD classifier (Task 2) |

**Kept (NOT moved):**
- `chronos_ts/trainer.py`, `metrics.py`, `splits.py`, `dataset.py` — reused.
- `scripts/train_har_vol.py`, `scripts/train_garch.py` — **volatility** models;
  feed/benchmark the `vol_regime` target.
- `scripts/train_model.py` + `configs/build_*.yaml` — dataset building is still needed.
- `scripts/train_seq.py`, `train_timexer.py`, `train_patchtst.py`,
  `train_chronos2.py` — deferred to Phase 2 (tokenized-sequence reuse), **not**
  moved yet; a note is added to `legacy/README.md` that they are "parked for P2".
- `app/` — untouched this phase.

`legacy/README.md` records each move (from → reason → replacement).

---

## 9. Step-by-step execution checklist

1. Add `chronos_ts/labels.py` (`LabelConfig`, `LabelMaker`) with train-only edge fitting + unit checks for no-leakage.
2. Add `chronos_ts/clf_metrics.py`.
3. Add `chronos_ts/classification.py` (`ClassifierFactory`, `ClassificationTrainer`).
4. Create the `conf/` Hydra tree.
5. Add `scripts/train_classifier.py` (`@hydra.main`).
6. Smoke-run each target family with `logreg` + `catboost`; confirm artifacts land in `outputs/models/<run>/`.
7. Build `notebooks/classification_report.ipynb` → leaderboard CSV.
8. Do the `legacy/` moves (§8) via `git mv`; write `legacy/README.md`.
9. Update root `README.md` with a "Reframing (Phase 1)" section + how to run.

## 10. Risks & mitigations
- **Leakage via label edges** → enforce `fit(train)`-only; add an assertion test.
- **Imbalance illusions** → always report balanced-accuracy + base rates, never raw accuracy alone.
- **Overfitting the dead-zone/threshold** → treat `ε`, `τ`, `K` as config; report sensitivity in the leaderboard.
- **Reusing regression CV scoring by accident** → CV scorer is explicitly log-loss/balanced-acc.

## 11. Deliverables
`chronos_ts/labels.py`, `chronos_ts/clf_metrics.py`, `chronos_ts/classification.py`,
`conf/**`, `scripts/train_classifier.py`, `notebooks/classification_report.ipynb`,
`outputs/reports/phase1_leaderboard.csv`, populated `legacy/regression_mean/**`,
updated `README.md`.
</content>
