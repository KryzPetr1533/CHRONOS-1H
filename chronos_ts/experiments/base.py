from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Optional
from chronos_ts.experiments.mlflow_logger import MlflowLogger
from chronos_ts.experiments.types import ExperimentResult, MLflowConfig


class BaseExperiment(ABC):
    name: str = 'base'
    default_mlflow_experiment: str = 'chronos-1h'

    @abstractmethod
    def fit(self, **kwargs: Any) -> ExperimentResult:
        raise NotImplementedError

    def run(self, mlflow: Optional[MLflowConfig] = None, run_name: Optional[str] = None, **fit_kwargs: Any) -> ExperimentResult:
        from chronos_ts.tracking import set_global_seed
        cfg = mlflow or MLflowConfig(experiment_name=self.default_mlflow_experiment)
        set_global_seed(cfg.seed)
        result = self.fit(**fit_kwargs)
        if mlflow is not None:
            extras = self._mlflow_extras(fit_kwargs, result)
            MlflowLogger(cfg).log(result, run_name=run_name, **extras)
        return result

    def _mlflow_extras(self, fit_kwargs: dict[str, Any], result: ExperimentResult) -> dict[str, Any]:
        return {}
