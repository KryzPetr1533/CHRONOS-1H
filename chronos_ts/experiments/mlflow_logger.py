from __future__ import annotations
from pathlib import Path
from typing import Any, Optional
from chronos_ts.experiments.types import ExperimentResult, MLflowConfig


class MlflowLogger:

    def __init__(self, config: MLflowConfig):
        self.config = config

    def log(self, result: ExperimentResult, run_name: Optional[str] = None, **kwargs: Any) -> str:
        try:
            from chronos_ts.tracking import configure_mlflow, log_classification_run, log_hydra_config, log_regression_run
        except ModuleNotFoundError as exc:
            if 'mlflow' in str(exc).lower():
                raise ModuleNotFoundError(
                    'Training finished but MLflow logging failed: mlflow is not installed. '
                    'Rebuild the dev image (`make rebuild`) or pip install -r requirements.txt'
                ) from exc
            raise
        configure_mlflow(tracking_uri=self.config.tracking_uri, experiment_name=self.config.experiment_name, s3_endpoint_url=self.config.s3_endpoint_url, aws_access_key_id=self.config.aws_access_key_id, aws_secret_access_key=self.config.aws_secret_access_key)
        name = run_name or result.run_name
        if result.kind == 'classification':
            extras = kwargs.get('classification_extras') or {}
            register_as = kwargs.get('register_as')
            promote_to_prd = bool(kwargs.get('promote_to_prd'))
            hydra_cfg = kwargs.get('hydra_cfg')
            with self._active_run(name):
                if hydra_cfg is not None:
                    log_hydra_config(hydra_cfg)
                version = log_classification_run(result=extras['result'], model=extras['model'], X_test=extras['X_test'], label_maker=extras['label_maker'], run_cfg=extras['run_cfg'], register_as=register_as or (self.config.registered_model_name if promote_to_prd else None), promote_to_prd=promote_to_prd, hydra_cfg=hydra_cfg)
                run_id = self._run_id()
            result.mlflow_run_id = run_id
            result.registered_version = version
            return run_id
        run_id = log_regression_run(result=result.payload, run_name=name, artifact_dir=result.output_dir, extra_params=result.params, artifact_names=result.artifact_names)
        result.mlflow_run_id = run_id
        return run_id

    def _active_run(self, run_name: str):
        from chronos_ts.mlflow_client import get_mlflow
        return get_mlflow().start_run(run_name=run_name)

    def _run_id(self) -> str:
        from chronos_ts.mlflow_client import get_mlflow
        run = get_mlflow().active_run()
        return run.info.run_id if run else ''
