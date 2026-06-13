from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from chronos_ts.experiments.base import BaseExperiment
from chronos_ts.experiments.types import ExperimentResult
from chronos_ts.seq_regression import DEFAULT_DATA, DEFAULT_OUT, train_seq_regression


class SeqRegressionExperiment(BaseExperiment):
    name = 'seq'
    default_mlflow_experiment = 'chronos-1h-regression-seq'

    def fit(self, data_csv: str | Path=DEFAULT_DATA, out_dir: str | Path=DEFAULT_OUT, seed: int=42, epochs: int=25, patience: int=5, batch_size: int=256, quick: bool=False, **kwargs: Any) -> ExperimentResult:
        payload = train_seq_regression(data_csv=data_csv, out_dir=out_dir, seed=seed, epochs=epochs, patience=patience, batch_size=batch_size, quick=quick)
        best = payload['best_model']
        out = Path(out_dir)
        return self._result_from_payload(payload, out_dir=out, seed=seed, epochs=epochs)

    def result_from_artifacts(self, out_dir: str | Path=DEFAULT_OUT, **kwargs: Any) -> ExperimentResult:
        out = Path(out_dir)
        metrics_path = out / 'seq_metrics.json'
        if not metrics_path.is_file():
            raise FileNotFoundError(f'Expected {metrics_path}')
        payload = json.loads(metrics_path.read_text(encoding='utf-8'))
        return self._result_from_payload(payload, out_dir=out, seed=int(payload.get('seed', 42)), epochs=int(kwargs.get('epochs', 0)))

    @staticmethod
    def _result_from_payload(payload: dict, out_dir: Path, seed: int, epochs: int) -> ExperimentResult:
        best = payload['best_model']
        out = Path(out_dir)
        run_name = f"seq_{best['model_kind']}_lb{best['lookback']}"
        params = {'seed': seed, **best}
        if epochs:
            params['epochs'] = epochs
        artifacts = tuple(n for n in ('seq_metrics.json', 'seq_search_results.csv', 'seq_test_predictions.csv') if (out / n).is_file())
        return ExperimentResult(kind='regression', name='seq', run_name=run_name, data_csv=str(payload.get('data_csv', '')), output_dir=out, payload=payload, params=params, artifact_names=artifacts)
