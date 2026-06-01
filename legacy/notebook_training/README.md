# legacy/notebook_training/ — Training cells superseded by CLI scripts

This directory holds training-heavy notebook cells that were superseded by
`scripts/train_mlflow.py` (T2-P2 bonus). The analysis/exploration cells remain
in the active notebooks under `notebooks/`.

## What was moved here

| Source | Content | Replaced by |
|---|---|---|
| `notebooks/experiment_tracking.ipynb` (training cells) | The `ClassifierRunner.run()` + MLflow registration block | `python scripts/train_mlflow.py experiment=final mlflow.promote_to_prd=true` |

## What stays active

- `notebooks/mlflow_setup.ipynb` — verify stack connectivity (not training)
- `notebooks/experiment_tracking.ipynb` — leaderboard display, model justification, metrics summary
- `notebooks/error_analysis.ipynb` — error analysis, robustness (not training)
- `notebooks/prd_predict.ipynb` — inference only (not training)
- `notebooks/classification_report.ipynb` — report generation (not training)

## How to re-run training via CLI

```bash
# Start MLflow stack
make mlflow-up

# Train final model and register PRD
make train-final

# Sweep all label × model combinations
make train-clf-sweep

# Load PRD model and predict
make predict-prd
```
