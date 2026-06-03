from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import yaml
from chronos_ts.experiments.base import BaseExperiment
from chronos_ts.experiments.types import ExperimentResult
from chronos_ts.splits import TimeRangeSplitConfig
from chronos_ts.trainer import TabularTrainer, TrainConfig


class TabularRegressionExperiment(BaseExperiment):
    name = 'tabular'
    default_mlflow_experiment = 'chronos-1h-regression-tabular'

    def __init__(self, name: str | None=None, default_mlflow_experiment: str | None=None):
        if name:
            self.name = name
        if default_mlflow_experiment:
            self.default_mlflow_experiment = default_mlflow_experiment

    def fit(self, config_path: str | Path, **kwargs: Any) -> ExperimentResult:
        cfg = self._load_config(config_path)
        out = TabularTrainer(cfg).run()
        return self._result_from_payload(out['result'], cfg)

    def result_from_artifacts(self, config_path: str | Path, **kwargs: Any) -> ExperimentResult:
        cfg = self._load_config(config_path)
        metrics_path = Path(cfg.output_dir) / f'{cfg.model_name}_metrics.json'
        if not metrics_path.is_file():
            raise FileNotFoundError(f'Expected {metrics_path}')
        payload = json.loads(metrics_path.read_text(encoding='utf-8'))
        return self._result_from_payload(payload, cfg)

    def _result_from_payload(self, payload: dict, cfg: TrainConfig) -> ExperimentResult:
        out = Path(cfg.output_dir)
        names = (f'{cfg.model_name}_metrics.json', f'{cfg.model_name}_test_predictions.csv', f'{cfg.model_name}_model.joblib')
        artifacts = tuple(n for n in names if (out / n).is_file())
        return ExperimentResult(kind='regression', name=self.name, run_name=f'{cfg.model_name}_regression', data_csv=cfg.data_csv, output_dir=out, payload=payload, params={'model_name': cfg.model_name, **(payload.get('best_params') or {})}, artifact_names=artifacts)

    @staticmethod
    def _load_config(path: str | Path) -> TrainConfig:
        raw = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
        split_cfg = raw.pop('split', {})
        raw['split'] = TimeRangeSplitConfig(**split_cfg)
        return TrainConfig(**raw)
