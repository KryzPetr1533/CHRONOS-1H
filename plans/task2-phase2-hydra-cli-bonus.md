# Task 2 · Phase 2 — Bonus: Hydra-configured CLI training & PRD inference

> **Builds on Task 2 Phase 1.** Phase 1 did everything from notebooks. This phase
> implements the **bonus**: run experiments **not from a notebook** but from `.py`
> scripts via the **CLI**, with all hyperparameters controlled by **Hydra `.yaml`
> config** — and a CLI script that loads the **PRD**-tagged model and runs a test
> prediction.
>
> **Why this is cheap now:** Task-1 Phase-1 already created the `conf/` Hydra tree
> and `scripts/train_classifier.py` (`@hydra.main`); Task-2 Phase-1 already created
> `chronos_ts/tracking.py::configure_mlflow()`. Phase 2 mostly **composes** these:
> add MLflow logging into the Hydra training script and add a PRD-inference CLI.
>
> The brief: *"Experiments should be run not from a notebook, but from a .py script
> via the CLI, and hyperparameters should be controlled via a hydra config file.
> Test run of the trained model with the PRD tag also via a .py script via the CLI."*

---

## 1. Outcomes / definition of done

- A single CLI command trains the chosen model, logs the full experiment to MLflow,
  saves artifacts to S3, and (optionally) registers/promotes to `prd` — all driven
  by Hydra config, **zero notebook involvement**.
- Hydra **multirun** sweeps run a hyperparameter grid, one MLflow run per trial.
- A CLI command loads `models:/chronos_1h_prd@prd` and prints a test prediction.
- Reproducibility is strengthened: Hydra snapshots the fully-resolved config per run
  (`.hydra/`), and the same resolved config is logged to MLflow.

---

## 2. Hydra config: add an `mlflow` group

Extend the Phase-1 `conf/` tree:
```
conf/
  config.yaml            # defaults now include: - mlflow: local
  mlflow/local.yaml      # tracking_uri, experiment_name, s3 endpoint, registered_model_name, promote_to_prd: false
  experiment/final.yaml  # pins the chosen target+model+params for the "final" run
  seed.yaml              # global seed (logged)
```
`mlflow/local.yaml` mirrors `chronos_ts/tracking.py` defaults so the CLI and the
Phase-1 notebooks share one source of truth (override on the command line for a
remote/cloud tracking server later).

---

## 3. CLI training script (notebook → script)

### New file: `scripts/train_mlflow.py`
```python
@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    configure_mlflow(cfg.mlflow)                 # reuse Phase-1 helper
    set_global_seed(cfg.seed)                    # reproducibility
    label = LabelMaker(cfg.label).fit(train_df)  # reuse Task-1 P1
    result = ClassificationTrainer(cfg).run()    # reuse Task-1 P1
    with mlflow.start_run(run_name=cfg.experiment.name):
        mlflow.log_params(flatten(cfg))          # all hyperparams from Hydra
        mlflow.log_metrics(prefixed(result))     # train/val/test + baseline
        log_artifacts(result)                    # model, confusion PNG, curves, pred examples, config, data lineage
        if cfg.mlflow.promote_to_prd:
            register_and_promote(cfg.mlflow.registered_model_name)  # PRD tag + 'prd' alias
```
Usage:
- Single final run: `python scripts/train_mlflow.py experiment=final mlflow.promote_to_prd=true`
- Override hyperparams: `python scripts/train_mlflow.py model=catboost model.depth=8 label=vol_regime`
- **Multirun sweep:** `python scripts/train_mlflow.py -m model=logreg,catboost,lightgbm label=vol_regime,large_move`
  → one MLflow run per combination; the MLflow UI becomes the experiment leaderboard.

This script **supersedes the Phase-1 training notebook** for producing official
experiments; the notebooks remain only for exploration/analysis.

---

## 4. CLI PRD-inference script

### New file: `scripts/predict_prd.py`
```python
@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    configure_mlflow(cfg.mlflow)
    model = mlflow.pyfunc.load_model(f"models:/{cfg.mlflow.registered_model_name}@prd")
    sample = load_sample(cfg.predict.input_csv)  # a few held-out rows, or --input on CLI
    preds = model.predict(sample)
    print(to_json(preds))
```
Usage: `python scripts/predict_prd.py predict.input_csv=outputs/datasets/sample.csv`
- Mirrors `notebooks/prd_predict.ipynb` (Phase-1 §11) but headless — satisfies the
  bonus's second clause ("test run of the PRD model also via a .py CLI script").
- Add `conf/predict/default.yaml` (input path, n rows, output format).

---

## 5. Reproducibility upgrades (beyond Phase 1)
- Hydra writes the fully-resolved config + overrides to `outputs/<run>/.hydra/`;
  archive that directory as an MLflow artifact so any run is replayable with
  `python scripts/train_mlflow.py --config-dir <archived>/.hydra`.
- `set_global_seed(cfg.seed)` covers `random`, `numpy`, and model seeds in one place;
  the seed is both a Hydra param and an MLflow param.
- Log `pip freeze` / `uv.lock` hash as an artifact for environment pinning.

---

## 6. Legacy moves (executed only when this phase starts)

| Move to `legacy/notebook_training/` | Why | Replaced by |
|---|---|---|
| Phase-1 training-with-MLflow notebook (the cells that *train & register*) | superseded by CLI `train_mlflow.py` | `scripts/train_mlflow.py` |

**Kept:** `notebooks/mlflow_setup.ipynb`, `notebooks/error_analysis.ipynb`,
`notebooks/prd_predict.ipynb`, `notebooks/classification_report.ipynb` — these are
analysis/exploration, not official experiment runners, so they stay. If a notebook
mixes training and analysis, split it: move training cells to the script, keep
analysis cells in the notebook. Record in `legacy/README.md`.

---

## 7. Step-by-step execution checklist
1. Add `conf/mlflow/local.yaml`, `conf/experiment/final.yaml`, `conf/seed.yaml`, `conf/predict/default.yaml`; wire into `config.yaml` defaults.
2. `scripts/train_mlflow.py` (`@hydra.main` + `configure_mlflow` + `ClassificationTrainer` + MLflow logging + optional PRD promotion).
3. Verify: single run logs to MLflow with artifacts in S3; `-m` multirun produces N runs.
4. `scripts/predict_prd.py` loads `models:/chronos_1h_prd@prd` and prints a prediction.
5. Confirm the CLI-trained run is registered & promotable to `prd`, identical to the Phase-1 notebook result.
6. `legacy/notebook_training/` moves + `legacy/README.md`; document the CLI workflow in root `README.md` (and a `Makefile` target, e.g. `make train-final`, to match the existing Makefile style).

## 8. Risks & mitigations
- **Hydra changes CWD** (it `cd`s into the run dir) → use absolute paths / `hydra.utils.get_original_cwd()` for data and artifact paths.
- **Config drift between notebook (P1) and CLI (P2)** → both read the same `conf/` tree and `chronos_ts/tracking.py`; no duplicated constants.
- **Multirun flooding the registry** → only `experiment=final` sets `promote_to_prd=true`; sweep runs log but do not register.
- **`OmegaConf` types vs MLflow params** → flatten/stringify the config before `log_params`.

## 9. Deliverables
`conf/mlflow/local.yaml`, `conf/experiment/final.yaml`, `conf/seed.yaml`,
`conf/predict/default.yaml`, `scripts/train_mlflow.py`, `scripts/predict_prd.py`,
`set_global_seed` helper, a `Makefile` target, populated `legacy/notebook_training/**`,
updated `README.md`.
</content>
