# Task 2 · Phase 1 — MLflow + S3 experiment versioning (notebook-driven)

> **Goal of this phase:** stand up the MLflow + MinIO(S3) stack, connect a research
> notebook to it, integrate MLflow logging into CHRONOS training, run a final
> tracked experiment for the **best CHRONOS model** (from Task 1), register it with
> the **PRD** tag, and complete the full analysis checklist (error analysis,
> baseline comparison, robustness). A separate clean notebook loads the PRD model
> and makes a test prediction.
>
> This phase covers **every required (non-bonus) item** in the task brief. The bonus
> (Hydra + CLI `.py` scripts) is Task 2 Phase 2.
>
> **Reuse:** `template-docker-mlflow-s3/` already provides a working
> Postgres + MLflow `v3.12.0` + MinIO + auto-bucket stack and notebooks that
> demonstrate the exact `log_model → register → set PRD tag → set 'prd' alias →
> load models:/<name>@prd` pattern (see `1_mlflow_catboost_tutorial.ipynb`). We
> adapt that pattern to the CHRONOS model — we do **not** reinvent the infra.

---

## 1. Outcomes / definition of done (mapped to the brief)

| Brief requirement | Delivered by |
|---|---|
| Local MLflow in Docker | adapted `docker-compose.yml` (MLflow `v3.12.0`) |
| Local/cloud S3 bucket | MinIO `mlflow-bucket` (auto-created by `mc` service) |
| Notebook ↔ MLflow connection | `notebooks/mlflow_setup.ipynb` (tracking URI + S3 endpoint localhost-rewrite) |
| Select best model + justify | §5, using the Task-1 leaderboard |
| Record final version + PRD tag | §6 (model registry: `env=PRD` tag + `prd` alias) |
| Retrain with MLflow + log final experiment(s) | §4–6 |
| Log params (hyperparams) + metrics (train/val/test) | §4 |
| Save artifacts to S3 (model, plots, prediction examples) | §4 |
| Reproducibility (seed, params, data description) | §7 |
| Error analysis (categories + 10–20 examples) | §8 |
| Baseline comparison | §9 (reuses Task-1 Phase-1 baselines) |
| Robustness (perturb inputs, observe) | §10 |
| Clean notebook loads PRD model + predicts | §11 |

---

## 2. Stand up the infrastructure

Decision: **bring the compose stack into the CHRONOS project** so training and
tracking live together. Recommended layout:
```
mlflow/                      # copied & trimmed from template-docker-mlflow-s3
  docker-compose.yml
  .env
  build/                     # Dockerfile, init.sql, basic_auth.ini (if auth used)
  data/                      # minio_data/, mlflow/  (bind mounts, gitignored)
```
Steps:
1. Copy `template-docker-mlflow-s3/{docker-compose.yml,.env,build/}` into `mlflow/`.
   The template is a *separate git repo*; copy files (don't nest the repo).
2. `cd mlflow && docker compose up -d --build`; verify with `docker compose ps`.
3. Confirm endpoints: MLflow UI `http://localhost:5050`, MinIO console
   `http://localhost:9001` (admin/password), bucket `mlflow-bucket` present.
4. Sanity: `mlflow.search_experiments()` from the notebook returns without error.

Notes captured from the template that matter:
- The server runs with `--serve-artifacts --artifacts-destination s3://mlflow-bucket/mlflow`,
  so clients log artifacts via the MLflow proxy (`mlflow-artifacts:/`) — avoids the
  `minio:9000` host-resolution issue called out in the template README.
- `minio:9000` resolves only inside the docker network; the notebook must rewrite
  the S3 endpoint to `http://localhost:9000` (the template notebook already shows
  this — replicate it).

---

## 3. Notebook ↔ MLflow connection

### New file: `notebooks/mlflow_setup.ipynb` (or a reusable `chronos_ts/tracking.py` helper)
- Loads `.env` (`python-dotenv`), sets:
  - `MLFLOW_TRACKING_URI = http://localhost:5050`
  - `AWS_ACCESS_KEY_ID/SECRET` = admin/password
  - `MLFLOW_S3_ENDPOINT_URL` rewritten `minio:9000 → localhost:9000`
- `mlflow.set_experiment("chronos-1h-classification")`.
- Factor this into `chronos_ts/tracking.py::configure_mlflow()` so Phase 2's CLI
  scripts reuse the identical connection logic (DRY across phases).

Add `mlflow>=2.15.1` (server is v3.12; pin a compatible client — the template
README explicitly warns about client/server skew) and `boto3` to `requirements.txt`.

---

## 4. Integrate MLflow into training & log the final experiment

Wrap the **Task-1 `ClassificationTrainer`** run in an MLflow run (in the notebook
for this phase; via Hydra CLI in Phase 2). Log:

- **Params:** model hyperparameters (the `best_params` already produced by the
  trainer), `LabelConfig` fields (target family, horizon, thresholds, `n_tokens`),
  split fractions, CV folds, feature count, **`seed`**.
- **Metrics:** the full `clf_metrics` dict for **train, val, and test** (prefix
  `train_/val_/test_`), plus baseline metrics for the same splits.
- **Artifacts → S3:**
  - the trained model (`mlflow.sklearn.log_model` / `mlflow.catboost.log_model`
    with `infer_signature` + `input_example`, exactly as the template notebook does),
  - **confusion matrix PNG**, **learning/calibration curves**, **confident-accuracy-
    vs-coverage curve**,
  - **prediction examples** CSV (a sample of `y_true`, `y_pred`, class probabilities),
  - the resolved config + `LabelMaker.describe()` (edges/base-rates) as JSON,
  - `mlflow.log_input(mlflow.data.from_pandas(...))` for dataset lineage.

One "final experiment" = the chosen best (target family × model). Optionally log the
small grid of candidates as sibling runs so the registry/UI shows the comparison.

---

## 5. Select the best model & justify

- Load `outputs/reports/phase1_leaderboard.csv` (Task-1 P1) into the notebook.
- **Selection criteria (documented in the notebook):**
  - statistical: balanced-accuracy / MCC / PR-AUC on **test**, *and* that the test
    edge is corroborated on **val** (no single-split flukes — the README explicitly
    flags the tiny-sample CatBoost-rich result as untrustworthy; we avoid that trap).
  - business: usefulness of the *trading metric with abstention* (coverage × hit-rate)
    and interpretability/latency for the serving `app/`.
- Likely winner given current evidence: a **`vol_regime`** or **`large_move`**
  classifier (volatility is the one forecastable signal per README §7.2), not a raw
  direction model. The notebook states the choice and the numbers behind it.
- **Record the final version** in a short `MODEL_CARD.md` (chosen run id, version,
  metrics, rationale).

---

## 6. Register with the PRD tag

Replicate the template's registry pattern against the CHRONOS model:
```python
model_info = mlflow.<flavor>.log_model(..., registered_model_name="chronos_1h_prd")
client.set_model_version_tag("chronos_1h_prd", version, "env", "PRD")
client.set_registered_model_alias("chronos_1h_prd", "prd", version)
```
So the final model is retrievable as `models:/chronos_1h_prd@prd` and tagged `PRD`.

---

## 7. Reproducibility

- Fix and **log** a global `seed` (the codebase already uses `random_seed=42` in
  CatBoost / `random_state=42` — centralize it into the config and log it).
- Log the resolved training config + library versions (`mlflow` autolog or explicit
  `mlflow.log_dict`).
- **Data description:** log how the dataset was obtained — for Phase-1 data, the
  build config from `configs/build_core.yaml` + source = Binance public futures; for
  Phase-2 data, the provenance JSON from `fetch_fine_data.py`. Store as an artifact.

---

## 8. Error analysis

In `notebooks/error_analysis.ipynb`:
- **Typical error categories** for the chosen classifier, e.g.: errors clustered in
  **high-volatility regimes**, around **funding timestamps** (8h cycle — features
  exist), at **regime transitions**, or on **low-confidence** predictions.
- Pull **10–20 specific misclassified test examples**; for each show the features,
  predicted vs true class, probability, and the surrounding price context.
- Explain causes (irreducible noise vs fixable feature gaps). Distinguish errors
  that **cannot** be corrected (genuine market randomness — the core README finding)
  from those that **could** (missing microstructure features → motivates Task-1 P2).

---

## 9. Compare with baseline

- **Baseline** = the honest baselines defined in Task-1 Phase-1 §5 (majority class /
  persist-previous-direction / vol-clustering persistence) — a simple model/heuristic,
  exactly as the brief asks.
- Compare final vs baseline on the selected metrics (balanced-acc, MCC, trading
  metric). Log both to the same MLflow experiment for side-by-side UI comparison.
- **Interpret the difference** briefly in the notebook (where the model adds value,
  where it only matches the heuristic).

---

## 10. Robustness check

- Apply small perturbations to test inputs: additive Gaussian noise scaled to each
  feature's std (e.g. 0.5%, 1%, 5%); feature dropout (set one feature to its median);
  a 1-bar time shift.
- Measure prediction stability: class-flip rate, mean |Δproba|, metric degradation.
- **Record observations** as a table/plot artifact in MLflow and a short written
  conclusion (which features the model is most sensitive to → ties back to error
  analysis & robustness).

---

## 11. Clean PRD-inference notebook

### New file: `notebooks/prd_predict.ipynb`
- Minimal, fresh notebook: configure tracking (§3), then
  `model = mlflow.pyfunc.load_model("models:/chronos_1h_prd@prd")`,
  load a few held-out rows, `model.predict(...)`, display predictions.
- Asserts the loaded PRD model reproduces the logged test predictions (a smoke check
  on reproducibility). This satisfies the brief's final required item.

---

## 12. Legacy moves (executed only when this phase starts)

| Move to `legacy/template_tutorials/` | Why | Replaced by |
|---|---|---|
| `template-docker-mlflow-s3/1_mlflow_catboost_tutorial.ipynb` | apple-quality demo; pattern absorbed into CHRONOS notebooks | `notebooks/mlflow_setup.ipynb` + training notebook |
| `template-docker-mlflow-s3/2_mlflow_cnn_tutorial.ipynb` | unrelated CNN demo | — |
| `template-docker-mlflow-s3/3_mlflow_bert_tiny_tutorial.ipynb` | unrelated BERT demo | — |
| `template-docker-mlflow-s3/data/apple_quality.csv` | demo dataset | CHRONOS datasets |

**Kept:** `docker-compose.yml`, `.env`, `build/`, `administration.ipynb` (copied
into `mlflow/`, actively used). The template's own `.git` is not copied. Record moves
in `legacy/README.md`.

---

## 13. Step-by-step execution checklist
1. Copy compose stack → `mlflow/`; `docker compose up -d --build`; verify UI + bucket.
2. `chronos_ts/tracking.py::configure_mlflow()` + `notebooks/mlflow_setup.ipynb`; confirm `search_experiments()`.
3. Add `mlflow`, `boto3`, `python-dotenv` (present in template) to `requirements.txt`.
4. Notebook: retrain best Task-1 classifier inside `mlflow.start_run`; log params/metrics/artifacts to S3.
5. Select & justify best model; write `MODEL_CARD.md`.
6. Register `chronos_1h_prd`; set `env=PRD` tag + `prd` alias.
7. `notebooks/error_analysis.ipynb` (10–20 examples), baseline comparison, robustness — log artifacts.
8. `notebooks/prd_predict.ipynb` loads `models:/chronos_1h_prd@prd` and predicts.
9. `legacy/template_tutorials/` moves + `legacy/README.md`; update root `README.md` with an MLflow section.

## 14. Risks & mitigations
- **Client/server version skew** → pin `mlflow` client compatible with server `v3.12.0` (template README §"Примечание по версиям").
- **`minio:9000` not resolvable from host** → endpoint rewrite to `localhost:9000` + rely on `--serve-artifacts` proxy.
- **Writing to deleted default experiment (`experiment 0`)** → always `set_experiment` to a named experiment (template troubleshooting §2).
- **Artifacts not landing in S3** → verify `artifact_uri` is `mlflow-artifacts:/...` and check the bucket in MinIO console.

## 15. Deliverables
`mlflow/**`, `chronos_ts/tracking.py`, `notebooks/mlflow_setup.ipynb`,
training-with-MLflow notebook, `notebooks/error_analysis.ipynb`,
`notebooks/prd_predict.ipynb`, `MODEL_CARD.md`, registered `chronos_1h_prd@prd`
(tag `PRD`), populated `legacy/template_tutorials/**`, updated `README.md`,
`requirements.txt`.
</content>
