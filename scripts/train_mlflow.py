from __future__ import annotations
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
log = logging.getLogger(__name__)

@dataclass
class MLflowRunConfig:
    tracking_uri: str = 'http://localhost:5050'
    experiment_name: str = 'chronos-1h-classification'
    s3_endpoint_url: str = 'http://localhost:9000'
    aws_access_key_id: str = 'admin'
    aws_secret_access_key: str = 'password'
    registered_model_name: str = 'chronos_1h_prd'
    promote_to_prd: bool = False
    run_name: str = 'default_run'

class TrainWithMLflow:

    def run_direct(self, label_family: str='vol_regime', model_name: str='catboost', data_csv: str='outputs/datasets/btcusdt_clf_core.csv', output_base: str='outputs/models/clf', mlflow_cfg: Optional[MLflowRunConfig]=None, seed: int=42, cv_splits: int=3, param_grid: Optional[dict]=None, promote_to_prd: bool=False) -> dict[str, Any]:
        from chronos_ts.experiments.classification import ClassificationExperiment
        from chronos_ts.experiments.types import MLflowConfig as ExpMLflowConfig
        from scripts.train_classifier import RunConfig
        cfg_mlflow = mlflow_cfg or MLflowRunConfig(promote_to_prd=promote_to_prd)
        run_cfg = RunConfig(data_csv=data_csv, label_family=label_family, model_name=model_name, output_dir=output_base, cv_splits=cv_splits, seed=seed, param_grid=param_grid or {})
        exp_ml = ExpMLflowConfig(tracking_uri=cfg_mlflow.tracking_uri, experiment_name=cfg_mlflow.experiment_name, s3_endpoint_url=cfg_mlflow.s3_endpoint_url, aws_access_key_id=cfg_mlflow.aws_access_key_id, aws_secret_access_key=cfg_mlflow.aws_secret_access_key, registered_model_name=cfg_mlflow.registered_model_name, promote_to_prd=cfg_mlflow.promote_to_prd, seed=seed)
        exp_result = ClassificationExperiment().run(mlflow=exp_ml, run_cfg=run_cfg, register_as=cfg_mlflow.registered_model_name if cfg_mlflow.promote_to_prd else None, promote_to_prd=cfg_mlflow.promote_to_prd)
        out = dict(exp_result.payload)
        out['mlflow_run_id'] = exp_result.mlflow_run_id
        return out

    def run_from_hydra(self, cfg) -> dict[str, Any]:
        import hydra
        from chronos_ts.experiments.classification import ClassificationExperiment
        from chronos_ts.experiments.types import MLflowConfig
        from scripts.train_classifier import _build_run_config_from_hydra
        project_root = Path(hydra.utils.get_original_cwd())
        ml = cfg.mlflow
        run_cfg = _build_run_config_from_hydra(cfg)
        run_cfg.output_dir = str(project_root / cfg.output_base)
        register_as = ml.registered_model_name if ml.promote_to_prd else None
        mlflow_cfg = MLflowConfig(tracking_uri=ml.tracking_uri, experiment_name=ml.experiment_name, s3_endpoint_url=ml.s3_endpoint_url, aws_access_key_id=ml.aws_access_key_id, aws_secret_access_key=ml.aws_secret_access_key, registered_model_name=ml.registered_model_name, promote_to_prd=bool(ml.promote_to_prd), seed=int(cfg.seed))
        exp_result = ClassificationExperiment().run(mlflow=mlflow_cfg, run_cfg=run_cfg, register_as=register_as, promote_to_prd=bool(ml.promote_to_prd), hydra_cfg=cfg)
        result = dict(exp_result.payload)
        result['mlflow_run_id'] = exp_result.mlflow_run_id
        result['registered_version'] = exp_result.registered_version
        try:
            from chronos_ts.mlflow_client import get_mlflow
            mlflow = get_mlflow()
            hydra_cfg_path = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir) / '.hydra'
            if hydra_cfg_path.exists() and result.get('mlflow_run_id'):
                with mlflow.start_run(run_id=result['mlflow_run_id']):
                    mlflow.log_artifacts(str(hydra_cfg_path), 'hydra_config')
        except Exception:
            pass
        return result
try:
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base=None, config_path='../conf', config_name='config')
    def main(cfg: DictConfig) -> None:
        logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
        result = TrainWithMLflow().run_from_hydra(cfg)
        print('\n--- Test metrics ---')
        print(json.dumps(result['metrics']['test'], indent=2, default=str))
except ImportError:

    def main() -> None:
        print('hydra-core not installed; use TrainWithMLflow().run_direct() instead.')
if __name__ == '__main__':
    main()