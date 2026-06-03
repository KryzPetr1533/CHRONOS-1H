from __future__ import annotations
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
        return ExperimentResult(kind='regression', name=self.name, run_name=f"seq_{best['model_kind']}_lb{best['lookback']}", data_csv=str(payload['data_csv']), output_dir=out, payload=payload, params={'seed': seed, 'epochs': epochs, **best}, artifact_names=('seq_metrics.json', 'seq_search_results.csv', 'seq_test_predictions.csv'))
