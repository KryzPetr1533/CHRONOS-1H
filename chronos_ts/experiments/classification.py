from __future__ import annotations
from pathlib import Path
from typing import Any, Optional
import joblib
import pandas as pd
from chronos_ts.experiments.base import BaseExperiment
from chronos_ts.experiments.types import ExperimentResult
from chronos_ts.labels import LabelConfig, LabelMaker
from chronos_ts.splits import TimeRangeSplitConfig, time_fraction_split


class ClassificationExperiment(BaseExperiment):
    name = 'classification'
    default_mlflow_experiment = 'chronos-1h-classification'

    def __init__(self):
        self._mlflow_ctx: dict[str, Any] = {}

    def fit(self, run_cfg: Any, register_as: Optional[str]=None, promote_to_prd: bool=False, hydra_cfg: Any=None, **kwargs: Any) -> ExperimentResult:
        from scripts.train_classifier import ClassifierRunner
        raw = ClassifierRunner(run_cfg).run(mlflow_experiment=None, register_as=None, promote_to_prd=False, hydra_cfg=None)
        self._mlflow_ctx = self._build_mlflow_context(run_cfg, raw)
        self._mlflow_ctx['register_as'] = register_as
        self._mlflow_ctx['promote_to_prd'] = promote_to_prd
        self._mlflow_ctx['hydra_cfg'] = hydra_cfg
        out = Path(run_cfg.output_dir) / f'{run_cfg.model_name}_{run_cfg.label_family}'
        return ExperimentResult(kind='classification', name=self.name, run_name=f'{run_cfg.model_name}_{run_cfg.label_family}', data_csv=run_cfg.data_csv, output_dir=out, payload=raw, params={'model_name': run_cfg.model_name, 'label_family': run_cfg.label_family, 'seed': run_cfg.seed}, artifact_names=tuple())

    def _mlflow_extras(self, fit_kwargs: dict[str, Any], result: ExperimentResult) -> dict[str, Any]:
        return {'classification_extras': self._mlflow_ctx, 'register_as': self._mlflow_ctx.get('register_as'), 'promote_to_prd': self._mlflow_ctx.get('promote_to_prd', False), 'hydra_cfg': self._mlflow_ctx.get('hydra_cfg')}

    @staticmethod
    def _build_mlflow_context(run_cfg: Any, result: dict) -> dict[str, Any]:
        df = pd.read_csv(run_cfg.data_csv, parse_dates=[run_cfg.ts_col])
        df = df.sort_values(run_cfg.ts_col).reset_index(drop=True)
        split_cfg = TimeRangeSplitConfig(train_frac=run_cfg.train_frac, val_frac=run_cfg.val_frac, test_frac=run_cfg.test_frac)
        splits = time_fraction_split(df, split_cfg, ts_col=run_cfg.ts_col)
        label_cfg = LabelConfig(target_family=run_cfg.label_family, return_col=run_cfg.return_col, horizon=run_cfg.horizon, dead_zone_sigma=run_cfg.dead_zone_sigma, move_quantile=run_cfg.move_quantile, n_tokens=run_cfg.n_tokens, vol_window=run_cfg.vol_window)
        label_maker = LabelMaker(label_cfg)
        label_maker.fit(splits['train'])
        y_test_series = label_maker.transform(splits['test'])
        mask = y_test_series.notna()
        feature_cols = result['feature_cols']
        X_test = splits['test'].loc[mask, feature_cols]
        out_dir = Path(run_cfg.output_dir) / f'{run_cfg.model_name}_{run_cfg.label_family}'
        model_path = out_dir / f'{run_cfg.model_name}_{run_cfg.label_family}_model.joblib'
        model = joblib.load(model_path)
        return {'result': result, 'model': model, 'X_test': X_test, 'label_maker': label_maker, 'run_cfg': run_cfg}
