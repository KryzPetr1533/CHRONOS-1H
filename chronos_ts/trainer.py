from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any
import pandas as pd
import numpy as np
import joblib
import copy
from sklearn.base import RegressorMixin, clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, HuberRegressor, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ParameterGrid, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from .metrics import evaluate_predictions
from .splits import TimeRangeSplitConfig, time_fraction_split
try:
    from catboost import CatBoostRegressor
except Exception:
    CatBoostRegressor = None
try:
    from lightgbm import LGBMRegressor
except Exception:
    LGBMRegressor = None
try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None

@dataclass
class TrainConfig:
    data_csv: str
    output_dir: str
    model_name: str = 'ridge'
    ts_col: str = 'ts'
    target_col: str = 'target_log_ret_1h'
    drop_cols: list[str] = field(default_factory=lambda: ['ts', 'target_log_ret_1h'])
    split: TimeRangeSplitConfig = field(default_factory=TimeRangeSplitConfig)
    cv_splits: int = 5
    n_jobs: int = -1
    scoring: str = 'neg_mean_squared_error'
    baseline_col: str = 'log_ret_1h'
    param_grid: dict[str, list[Any]] = field(default_factory=dict)

class ModelFactory:

    @staticmethod
    def make(model_name: str) -> RegressorMixin | Pipeline:
        model_name = model_name.lower()
        if model_name == 'ridge':
            return Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler()), ('model', Ridge())])
        if model_name == 'enet':
            return Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler()), ('model', ElasticNet(max_iter=20000))])
        if model_name == 'huber':
            return Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler()), ('model', HuberRegressor())])
        if model_name == 'catboost':
            if CatBoostRegressor is None:
                raise ImportError('catboost is not installed')
            return CatBoostRegressor(loss_function='RMSE', verbose=False, random_seed=42, allow_writing_files=False)
        if model_name == 'lightgbm':
            if LGBMRegressor is None:
                raise ImportError('lightgbm is not installed')
            return LGBMRegressor(objective='regression', random_state=42, n_jobs=-1)
        if model_name == 'xgboost':
            if XGBRegressor is None:
                raise ImportError('xgboost is not installed')
            return XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)
        raise ValueError(f'Unsupported model_name: {model_name}')

    @staticmethod
    def default_param_grid(model_name: str) -> dict[str, list[Any]]:
        model_name = model_name.lower()
        if model_name == 'ridge':
            return {'model__alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
        if model_name == 'enet':
            return {'model__alpha': [0.0005, 0.001, 0.01, 0.1], 'model__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]}
        if model_name == 'huber':
            return {'model__epsilon': [1.1, 1.2, 1.35, 1.5], 'model__alpha': [1e-05, 0.0001, 0.001]}
        if model_name == 'catboost':
            return {'depth': [4, 6, 8], 'learning_rate': [0.02, 0.05, 0.1], 'n_estimators': [200, 500], 'l2_leaf_reg': [3, 10]}
        if model_name == 'lightgbm':
            return {'num_leaves': [15, 31, 63], 'learning_rate': [0.02, 0.05, 0.1], 'n_estimators': [200, 500], 'min_child_samples': [20, 50]}
        if model_name == 'xgboost':
            return {'max_depth': [3, 5, 7], 'learning_rate': [0.02, 0.05, 0.1], 'n_estimators': [200, 500], 'subsample': [0.8, 1.0]}
        raise ValueError(f'Unsupported model_name: {model_name}')

class TabularTrainer:

    def __init__(self, config: TrainConfig):
        self.config = config

    def run(self) -> dict[str, Any]:
        df = pd.read_csv(self.config.data_csv, parse_dates=[self.config.ts_col])
        df = df.sort_values(self.config.ts_col).reset_index(drop=True)
        splits = time_fraction_split(df, self.config.split, ts_col=self.config.ts_col)
        raw_feature_cols = [c for c in df.columns if c not in set(self.config.drop_cols)]
        feature_cols = []
        for c in raw_feature_cols:
            if splits['train'][c].notna().sum() == 0:
                continue
            feature_cols.append(c)
        X_train, y_train = self._xy(splits['train'], feature_cols)
        X_val, y_val = self._xy(splits['val'], feature_cols)
        X_test, y_test = self._xy(splits['test'], feature_cols)
        estimator = ModelFactory.make(self.config.model_name)
        grid = self.config.param_grid or ModelFactory.default_param_grid(self.config.model_name)
        tscv = TimeSeriesSplit(n_splits=self.config.cv_splits)
        search = self._manual_grid_search(estimator=estimator, param_grid=grid, X=X_train, y=y_train, cv=tscv)
        best_model = search['best_estimator_']
        y_val_pred = best_model.predict(X_val)
        y_test_pred = best_model.predict(X_test)
        y_val_baseline = self._baseline_predictions(splits['val'])
        y_test_baseline = self._baseline_predictions(splits['test'])
        result = {'config': self._serialize_config(), 'feature_cols': feature_cols, 'best_params': search['best_params_'], 'cv_best_score': float(search['best_score_']), 'n_rows': {k: int(len(v)) for k, v in splits.items()}, 'baseline': {'val': evaluate_predictions(y_val, y_val_baseline), 'test': evaluate_predictions(y_test, y_test_baseline)}, 'model': {'val': evaluate_predictions(y_val, y_val_pred), 'test': evaluate_predictions(y_test, y_test_pred)}, 'ts_range': {split_name: {'min': str(split_df[self.config.ts_col].min()), 'max': str(split_df[self.config.ts_col].max())} for split_name, split_df in splits.items()}}
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / f'{self.config.model_name}_model.joblib'
        metrics_path = output_dir / f'{self.config.model_name}_metrics.json'
        preds_path = output_dir / f'{self.config.model_name}_test_predictions.csv'
        joblib.dump(best_model, model_path)
        pd.DataFrame({self.config.ts_col: splits['test'][self.config.ts_col], 'y_true': y_test, 'y_pred': y_test_pred, 'y_pred_baseline': y_test_baseline}).to_csv(preds_path, index=False)
        import json
        metrics_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        return {'model_path': str(model_path), 'metrics_path': str(metrics_path), 'predictions_path': str(preds_path), 'result': result}

    def _xy(self, df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
        X = df[feature_cols].copy()
        y = df[self.config.target_col].to_numpy(dtype=float)
        return (X, y)

    def _baseline_predictions(self, df: pd.DataFrame) -> np.ndarray:
        if self.config.baseline_col not in df.columns:
            return np.zeros(len(df), dtype=float)
        return df[self.config.baseline_col].to_numpy(dtype=float)

    def _serialize_config(self) -> dict[str, Any]:
        data = asdict(self.config)
        if isinstance(self.config.split, TimeRangeSplitConfig):
            data['split'] = asdict(self.config.split)
        return data

    def _fresh_estimator(self, estimator):
        try:
            return clone(estimator)
        except Exception:
            return copy.deepcopy(estimator)

    def _manual_grid_search(self, estimator, param_grid: dict[str, list[Any]], X: pd.DataFrame, y: np.ndarray, cv: TimeSeriesSplit) -> dict[str, Any]:
        best_score = -np.inf
        best_params = None
        for params in ParameterGrid(param_grid):
            fold_scores = []
            for train_idx, valid_idx in cv.split(X):
                X_tr = X.iloc[train_idx]
                X_va = X.iloc[valid_idx]
                y_tr = y[train_idx]
                y_va = y[valid_idx]
                est = self._fresh_estimator(estimator)
                est.set_params(**params)
                est.fit(X_tr, y_tr)
                pred = est.predict(X_va)
                score = -mean_squared_error(y_va, pred)
                fold_scores.append(score)
            mean_score = float(np.mean(fold_scores))
            if mean_score > best_score:
                best_score = mean_score
                best_params = params
        best_estimator = self._fresh_estimator(estimator)
        best_estimator.set_params(**best_params)
        best_estimator.fit(X, y)
        return {'best_estimator_': best_estimator, 'best_params_': best_params, 'best_score_': best_score}