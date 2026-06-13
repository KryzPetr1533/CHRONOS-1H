from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

ExperimentKind = Literal['classification', 'regression']


@dataclass
class MLflowConfig:
    tracking_uri: str = 'http://localhost:5050'
    experiment_name: str = 'chronos-1h'
    s3_endpoint_url: str = 'http://localhost:9000'
    aws_access_key_id: str = 'admin'
    aws_secret_access_key: str = 'password'
    registered_model_name: str = 'chronos_1h_prd'
    promote_to_prd: bool = False
    seed: int = 42


@dataclass
class ExperimentResult:
    kind: ExperimentKind
    name: str
    run_name: str
    data_csv: str
    output_dir: Path
    payload: dict[str, Any]
    params: dict[str, Any] = field(default_factory=dict)
    artifact_names: tuple[str, ...] = ()
    mlflow_run_id: str | None = None
    registered_version: str | None = None

    def regression_model_metrics(self) -> dict[str, dict[str, Any]]:
        model = self.payload.get('model')
        if model:
            return model
        for key in ('timexer', 'har', 'garch'):
            if key in self.payload:
                return self.payload[key]
        if 'model_results' in self.payload:
            first = next(iter(self.payload['model_results'].values()), {})
            return {'test': first.get('model', first), 'val': first.get('model', first)}
        return {}

    def regression_baseline_metrics(self) -> dict[str, dict[str, Any]]:
        return self.payload.get('baseline', {})
