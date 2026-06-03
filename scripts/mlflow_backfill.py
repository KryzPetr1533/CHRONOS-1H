#!/usr/bin/env python3
"""Push existing on-disk metrics to MLflow without retraining."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from chronos_ts.experiments import EXPERIMENT_NAMES, get_experiment
from chronos_ts.experiments.types import MLflowConfig

BACKFILL_DEFAULTS: dict[str, dict] = {
    'ridge': {'config_path': 'legacy/regression_mean/configs/train_ridge_core.yaml'},
    'catboost_reg': {'config_path': 'legacy/regression_mean/configs/train_catboost_core.yaml'},
    'seq': {'out_dir': 'outputs/models/seq_core_small'},
}

BACKFILL_ALL = ('seq', 'ridge', 'catboost_reg', 'har_vol', 'garch', 'timexer', 'chronos2')


def parse_args():
    p = argparse.ArgumentParser(description='Log existing training artifacts to MLflow (no retrain)')
    p.add_argument('experiment', nargs='?', help=f'One of: {", ".join(EXPERIMENT_NAMES)}')
    p.add_argument('--all', action='store_true', help=f'Backfill: {", ".join(BACKFILL_ALL)}')
    p.add_argument('--tracking-uri', default='http://localhost:5050')
    p.add_argument('--experiment-name', default=None)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def backfill_one(name: str, tracking_uri: str, experiment_name: str | None, seed: int) -> int:
    if name == 'classification':
        print('classification: use train_mlflow.py / make train-sweep', file=sys.stderr)
        return 1
    if name == 'patchtst':
        print('patchtst: no metrics file (training failed); skip', file=sys.stderr)
        return 1
    exp = get_experiment(name)
    defaults = BACKFILL_DEFAULTS.get(name, {})
    mlflow_cfg = MLflowConfig(tracking_uri=tracking_uri, experiment_name=experiment_name or exp.default_mlflow_experiment, seed=seed)
    result = exp.log_from_disk(mlflow_cfg, **defaults)
    print(f'OK {name}  run_name={result.run_name}  mlflow_run_id={result.mlflow_run_id}')
    return 0


def main() -> int:
    args = parse_args()
    names = list(BACKFILL_ALL) if args.all else [args.experiment] if args.experiment else []
    if not names:
        print('Pass an experiment name or --all', file=sys.stderr)
        return 1
    failed = 0
    for name in names:
        if name not in EXPERIMENT_NAMES:
            print(f'Unknown: {name}', file=sys.stderr)
            failed += 1
            continue
        try:
            code = backfill_one(name, args.tracking_uri, args.experiment_name, args.seed)
            failed += code
        except FileNotFoundError as exc:
            print(f'SKIP {name}: {exc}', file=sys.stderr)
            failed += 1
        except ModuleNotFoundError as exc:
            print(exc, file=sys.stderr)
            return 1
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
