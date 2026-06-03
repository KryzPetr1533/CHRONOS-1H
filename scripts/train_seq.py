#!/usr/bin/env python3
from __future__ import annotations
import argparse
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from chronos_ts.experiments.types import MLflowConfig
from chronos_ts.experiments.seq import SeqRegressionExperiment
from chronos_ts.seq_regression import DEFAULT_DATA, DEFAULT_OUT


def main() -> int:
    p = argparse.ArgumentParser(description='GRU/LSTM sequence regression (wrapper around SeqRegressionExperiment)')
    p.add_argument('--data-csv', type=Path, default=DEFAULT_DATA)
    p.add_argument('--out-dir', type=Path, default=DEFAULT_OUT)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=25)
    p.add_argument('--patience', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--quick', action='store_true')
    p.add_argument('--mlflow', action='store_true')
    p.add_argument('--mlflow-experiment', default='chronos-1h-regression-seq')
    p.add_argument('--mlflow-tracking-uri', default='http://localhost:5050')
    args = p.parse_args()
    mlflow = None
    if args.mlflow:
        mlflow = MLflowConfig(tracking_uri=args.mlflow_tracking_uri, experiment_name=args.mlflow_experiment, seed=args.seed)
    SeqRegressionExperiment().run(mlflow=mlflow, data_csv=args.data_csv, out_dir=args.out_dir, seed=args.seed, epochs=args.epochs, patience=args.patience, batch_size=args.batch_size, quick=args.quick)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
