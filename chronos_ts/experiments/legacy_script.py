from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path
from typing import Any
from chronos_ts.experiments.base import BaseExperiment
from chronos_ts.experiments.types import ExperimentResult

REPO_ROOT = Path(__file__).resolve().parents[2]


class LegacyScriptExperiment(BaseExperiment):
    def __init__(self, name: str, script: Path, output_dir: Path, metrics_file: str, default_mlflow_experiment: str, model_key: str | None=None):
        self.name = name
        self.script = Path(script)
        self.output_dir = Path(output_dir)
        self.metrics_file = metrics_file
        self.default_mlflow_experiment = default_mlflow_experiment
        self.model_key = model_key

    def fit(self, **kwargs: Any) -> ExperimentResult:
        if not self.script.is_file():
            raise FileNotFoundError(self.script)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, str(self.script)], cwd=str(REPO_ROOT), check=True)
        metrics_path = self.output_dir / self.metrics_file
        if not metrics_path.is_file():
            raise FileNotFoundError(f'Expected metrics at {metrics_path}')
        payload = json.loads(metrics_path.read_text(encoding='utf-8'))
        run_name = self._run_name(payload)
        params: dict[str, Any] = {'script': str(self.script)}
        extra = payload.get('best_params') or payload.get('best_model') or {}
        if isinstance(extra, dict):
            params.update(extra)
        elif extra:
            params['best_model'] = str(extra)
        artifacts = tuple(p.name for p in self.output_dir.iterdir() if p.is_file() and (p.suffix in ('.json', '.csv')))
        return ExperimentResult(kind='regression', name=self.name, run_name=run_name, data_csv=str(payload.get('data_csv', '')), output_dir=self.output_dir, payload=payload, params={k: str(v) for k, v in params.items()}, artifact_names=artifacts)

    def _run_name(self, payload: dict) -> str:
        if self.model_key and self.model_key in payload:
            return f'{self.name}_{self.model_key}'
        best = payload.get('best_model', self.name)
        if isinstance(best, dict):
            return self.name
        return f'{self.name}_{best}'

    def run(self, mlflow=None, run_name=None, **fit_kwargs):
        if self.name == 'chronos2' and mlflow is not None:
            return self._run_chronos2_mlflow(mlflow, **fit_kwargs)
        return super().run(mlflow=mlflow, run_name=run_name, **fit_kwargs)

    def _run_chronos2_mlflow(self, mlflow_cfg, **fit_kwargs):
        from chronos_ts.tracking import set_global_seed
        from chronos_ts.experiments.mlflow_logger import MlflowLogger
        set_global_seed(mlflow_cfg.seed)
        result = self.fit(**fit_kwargs)
        logger = MlflowLogger(mlflow_cfg)
        for model_name, metrics in result.payload.get('model_results', {}).items():
            sub_payload = {'data_csv': result.payload.get('data_csv'), 'model': metrics['model'], 'baseline': metrics['baseline']}
            sub = ExperimentResult(kind='regression', name=self.name, run_name=f'chronos2_{model_name}', data_csv=result.data_csv, output_dir=result.output_dir, payload=sub_payload, params={'model_name': model_name})
            logger.log(sub, run_name=sub.run_name)
        result.mlflow_run_id = 'multi'
        return result
