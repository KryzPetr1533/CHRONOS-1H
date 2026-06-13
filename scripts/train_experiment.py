#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from chronos_ts.experiments import EXPERIMENT_NAMES

EXPERIMENT_DEFAULTS: dict[str, dict] = {
    'ridge': {'config_path': 'legacy/regression_mean/configs/train_ridge_core.yaml'},
    'catboost_reg': {'config_path': 'legacy/regression_mean/configs/train_catboost_core.yaml'},
    'seq': {'data_csv': 'outputs/datasets/btcusdt_core_seq_small.csv', 'out_dir': 'outputs/models/seq_core_small'},
}


def parse_args():
    p = argparse.ArgumentParser(description='Run a registered training experiment with optional MLflow logging')
    p.add_argument('experiment', help=f'One of: {", ".join(EXPERIMENT_NAMES)}')
    p.add_argument('--no-mlflow', action='store_true', help='Train only; skip MLflow')
    p.add_argument('--tracking-uri', default='http://localhost:5050')
    p.add_argument('--experiment-name', default=None, help='MLflow experiment (default: per trainer)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--quick', action='store_true', help='seq: single GRU config')
    p.add_argument('--epochs', type=int, default=None, help='seq: training epochs')
    p.add_argument('--config-path', default=None, help='tabular: YAML config path')
    p.add_argument('--promote-prd', action='store_true', help='classification: register PRD model')
    return p.parse_args()


def main() -> int:
    from chronos_ts.experiments import get_experiment
    from chronos_ts.experiments.types import MLflowConfig
    args = parse_args()
    if args.experiment not in EXPERIMENT_NAMES:
        print(f'Unknown experiment: {args.experiment}', file=sys.stderr)
        print('Choose from:', ', '.join(EXPERIMENT_NAMES), file=sys.stderr)
        return 1
    exp = get_experiment(args.experiment)
    defaults = EXPERIMENT_DEFAULTS.get(args.experiment, {})
    mlflow_cfg = None
    if not args.no_mlflow:
        mlflow_cfg = MLflowConfig(tracking_uri=args.tracking_uri, experiment_name=args.experiment_name or exp.default_mlflow_experiment, seed=args.seed, promote_to_prd=args.promote_prd)
    fit_kwargs = dict(defaults)
    if args.config_path:
        fit_kwargs['config_path'] = args.config_path
    if args.epochs is not None:
        fit_kwargs['epochs'] = args.epochs
    if args.quick:
        fit_kwargs['quick'] = True
    if args.experiment == 'classification':
        print('classification experiment: use scripts/train_mlflow.py (Hydra)', file=sys.stderr)
        return 1
    result = exp.run(mlflow=mlflow_cfg, seed=args.seed, **fit_kwargs)
    print('\n--- Done ---')
    print(f'experiment={result.name}  run_name={result.run_name}')
    if result.mlflow_run_id:
        print(f'mlflow_run_id={result.mlflow_run_id}')
    if result.kind == 'regression' and result.payload.get('model', {}).get('test'):
        print(json.dumps(result.payload['model']['test'], indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
