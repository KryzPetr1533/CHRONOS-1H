#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description='Register local classifier joblib as chronos_1h_prd@prd')
    parser.add_argument('--model-path', default='outputs/models/clf/catboost_vol_regime/catboost_vol_regime_model.joblib')
    parser.add_argument('--sample-csv', default='outputs/datasets/btcusdt_clf_core.csv')
    parser.add_argument('--metrics-path', default='outputs/models/clf/catboost_vol_regime/catboost_vol_regime_metrics.json')
    parser.add_argument('--registered-name', default='chronos_1h_prd')
    parser.add_argument('--promote', action='store_true', default=True)
    parser.add_argument('--tracking-uri', default='http://localhost:5050')
    parser.add_argument('--s3-endpoint', default='http://localhost:9000')
    args = parser.parse_args()

    from chronos_ts.tracking import configure_mlflow, _log_model_artifact, _set_prd_alias
    import mlflow
    import pandas as pd
    from mlflow.models import infer_signature

    configure_mlflow(
        tracking_uri=args.tracking_uri,
        s3_endpoint_url=args.s3_endpoint,
    )
    model_path = REPO_ROOT / args.model_path
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    model = joblib.load(model_path)
    metrics_path = REPO_ROOT / args.metrics_path
    feature_cols = json.load(open(metrics_path))['feature_cols']
    df = pd.read_csv(REPO_ROOT / args.sample_csv)
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f'{len(missing)} feature columns missing from {args.sample_csv}, e.g. {missing[:5]}')
    sample = df[feature_cols].dropna().head(5)
    proba = model.predict_proba(sample)
    signature = infer_signature(sample, proba)

    with mlflow.start_run(run_name='register_prd_from_artifacts'):
        _log_model_artifact(model, sample, signature, register_as=None)
        run_id = mlflow.active_run().info.run_id
        mv = mlflow.register_model(f'runs:/{run_id}/model', args.registered_name)
        print(f'Registered {args.registered_name} version {mv.version}')
        if args.promote:
            _set_prd_alias(args.registered_name, mv.version)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
