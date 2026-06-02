from __future__ import annotations
import copy
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, List, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin, clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import ParameterGrid, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from .clf_metrics import evaluate_classification, confusion_matrix_df
from .labels import LabelConfig, LabelMaker
from .splits import TimeRangeSplitConfig, time_fraction_split
try:
    from catboost import CatBoostClassifier
except Exception:
    CatBoostClassifier = None
try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None
try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

@dataclass
class ClassificationConfig:
    data_csv: str
    output_dir: str
    label: LabelConfig = field(default_factory=LabelConfig)
    model_name: str = 'logreg'
    ts_col: str = 'ts'
    drop_cols: List[str] = field(default_factory=lambda: ['ts'])
    split: TimeRangeSplitConfig = field(default_factory=TimeRangeSplitConfig)
    cv_splits: int = 5
    n_jobs: int = -1
    seed: int = 42
    param_grid: dict = field(default_factory=dict)
    abstention_threshold: float = 0.65
    top_k_pct: float = 0.2

class ClassifierFactory:

    @staticmethod
    def make(model_name: str, seed: int=42) -> ClassifierMixin:
        name = model_name.lower()
        if name == 'logreg':
            return Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler()), ('model', LogisticRegression(max_iter=2000, random_state=seed, n_jobs=-1))])
        if name == 'catboost':
            if CatBoostClassifier is None:
                raise ImportError('catboost is not installed')
            return CatBoostClassifier(loss_function='Logloss', eval_metric='AUC', verbose=False, random_seed=seed, allow_writing_files=False, auto_class_weights='Balanced')
        if name == 'lightgbm':
            if LGBMClassifier is None:
                raise ImportError('lightgbm is not installed')
            return LGBMClassifier(objective='binary', random_state=seed, n_jobs=-1, class_weight='balanced', verbose=-1)
        if name == 'xgboost':
            if XGBClassifier is None:
                raise ImportError('xgboost is not installed')
            return XGBClassifier(objective='binary:logistic', random_state=seed, n_jobs=-1, eval_metric='logloss')
        raise ValueError(f'Unsupported model_name: {model_name!r}')

    @staticmethod
    def default_param_grid(model_name: str) -> dict:
        name = model_name.lower()
        if name == 'logreg':
            return {'model__C': [0.01, 0.1, 1.0, 10.0]}
        if name == 'catboost':
            return {'depth': [4, 6], 'learning_rate': [0.03, 0.05], 'n_estimators': [200, 400], 'l2_leaf_reg': [3, 10]}
        if name == 'lightgbm':
            return {'num_leaves': [15, 31], 'learning_rate': [0.03, 0.05], 'n_estimators': [200, 400]}
        if name == 'xgboost':
            return {'max_depth': [3, 5], 'learning_rate': [0.03, 0.05], 'n_estimators': [200, 400]}
        raise ValueError(f'Unsupported model_name: {model_name!r}')

class ClassificationTrainer:

    def __init__(self, config: ClassificationConfig):
        self.config = config

    def run(self) -> dict[str, Any]:
        df = pd.read_csv(self.config.data_csv, parse_dates=[self.config.ts_col])
        df = df.sort_values(self.config.ts_col).reset_index(drop=True)
        splits = time_fraction_split(df, self.config.split, ts_col=self.config.ts_col)
        label_maker = LabelMaker(self.config.label)
        label_maker.fit(splits['train'])
        labeled = {}
        for split_name, split_df in splits.items():
            y = label_maker.transform(split_df)
            labeled[split_name] = (split_df, y)
        drop_cols_set = set(self.config.drop_cols) | {label_maker.label_name()}
        feature_cols = self._select_feature_cols(labeled['train'][0], drop_cols_set)
        results_split: dict[str, Any] = {}
        Xs, ys = ({}, {})
        for split_name, (split_df, y_series) in labeled.items():
            mask = y_series.notna()
            Xs[split_name] = split_df.loc[mask, feature_cols]
            ys[split_name] = y_series[mask].to_numpy(dtype=int)
            results_split[split_name] = {'n_rows': int(mask.sum()), 'dropped_rows': int((~mask).sum())}
        n_classes = label_maker.n_classes()
        estimator = self._make_estimator(n_classes)
        grid = self.config.param_grid or ClassifierFactory.default_param_grid(self.config.model_name)
        tscv = TimeSeriesSplit(n_splits=self.config.cv_splits)
        search = self._manual_grid_search(estimator, grid, Xs['train'], ys['train'], tscv)
        best_model = search['best_estimator_']
        baselines = self._compute_baselines(labeled, feature_cols, label_maker)
        metrics: dict[str, Any] = {}
        preds_test: Optional[pd.DataFrame] = None
        for split_name in ('train', 'val', 'test'):
            X, y = (Xs[split_name], ys[split_name])
            proba = np.atleast_2d(best_model.predict_proba(X))
            y_pred = np.asarray(best_model.predict(X), dtype=float).ravel().astype(int)
            _ret_col_name = self.config.label.return_col
            _available = labeled[split_name][0].columns
            ret_col = labeled[split_name][0].loc[labeled[split_name][1].notna(), _ret_col_name].to_numpy(dtype=float) if _ret_col_name in _available else None
            metrics[split_name] = evaluate_classification(y, y_pred, proba, class_names=label_maker.class_names(), return_col=ret_col, abstention_threshold=self.config.abstention_threshold, top_k_pct=self.config.top_k_pct)
            if split_name == 'test':
                proba_df = pd.DataFrame(proba, columns=[f'proba_{c}' for c in label_maker.class_names()])
                preds_test = pd.DataFrame({'y_true': y, 'y_pred': y_pred})
                preds_test = pd.concat([preds_test, proba_df], axis=1)
        result = {'config': asdict(self.config), 'label_maker': label_maker.describe(), 'feature_cols': feature_cols, 'n_classes': n_classes, 'best_params': search['best_params_'], 'cv_best_score': float(search['best_score_']), 'split_sizes': results_split, 'baselines': baselines, 'metrics': metrics}
        self._save_artifacts(result, best_model, preds_test, label_maker, Xs['test'], ys['test'])
        return result

    def _select_feature_cols(self, train_df: pd.DataFrame, drop_set: set, min_non_null_frac: float=0.05) -> List[str]:
        cols = []
        for c in train_df.columns:
            if c in drop_set:
                continue
            non_null = train_df[c].notna().mean()
            if non_null < min_non_null_frac:
                continue
            cols.append(c)
        return cols

    def _make_estimator(self, n_classes: int):
        name = self.config.model_name.lower()
        seed = self.config.seed
        if n_classes > 2:
            if name == 'catboost':
                if CatBoostClassifier is None:
                    raise ImportError('catboost is not installed')
                return CatBoostClassifier(loss_function='MultiClass', eval_metric='Accuracy', verbose=False, random_seed=seed, allow_writing_files=False)
            if name == 'lightgbm':
                if LGBMClassifier is None:
                    raise ImportError('lightgbm is not installed')
                return LGBMClassifier(objective='multiclass', num_class=n_classes, random_state=seed, n_jobs=-1, verbose=-1)
            if name == 'xgboost':
                if XGBClassifier is None:
                    raise ImportError('xgboost is not installed')
                return XGBClassifier(objective='multi:softprob', num_class=n_classes, random_state=seed, n_jobs=-1, eval_metric='mlogloss')
        return ClassifierFactory.make(name, seed=seed)

    def _fresh_estimator(self, estimator):
        try:
            return clone(estimator)
        except Exception:
            return copy.deepcopy(estimator)

    def _manual_grid_search(self, estimator, param_grid, X, y, cv) -> dict:
        best_score = np.inf
        best_params = None
        for params in ParameterGrid(param_grid):
            fold_scores = []
            for train_idx, valid_idx in cv.split(X):
                X_tr, X_va = (X.iloc[train_idx], X.iloc[valid_idx])
                y_tr, y_va = (y[train_idx], y[valid_idx])
                est = self._fresh_estimator(estimator)
                est.set_params(**params)
                est.fit(X_tr, y_tr)
                proba = est.predict_proba(X_va)
                try:
                    score = log_loss(y_va, proba)
                except Exception:
                    score = np.inf
                fold_scores.append(score)
            mean_score = float(np.mean(fold_scores))
            if mean_score < best_score:
                best_score = mean_score
                best_params = params
        best_estimator = self._fresh_estimator(estimator)
        best_estimator.set_params(**best_params)
        best_estimator.fit(X, y)
        return {'best_estimator_': best_estimator, 'best_params_': best_params, 'best_score_': best_score}

    def _compute_baselines(self, labeled, feature_cols, label_maker) -> dict:
        baselines: dict[str, Any] = {}
        y_train = labeled['train'][1].dropna().to_numpy(dtype=int)
        majority_class = int(np.bincount(y_train).argmax())
        for split_name in ('val', 'test'):
            split_df, y_series = labeled[split_name]
            mask = y_series.notna()
            y = y_series[mask].to_numpy(dtype=int)
            n = len(y)
            n_classes = label_maker.n_classes()
            y_maj = np.full(n, majority_class, dtype=int)
            proba_maj = np.zeros((n, n_classes), dtype=float)
            proba_maj[:, majority_class] = 1.0
            baselines[split_name] = {'majority_class': evaluate_classification(y, y_maj, proba_maj, class_names=label_maker.class_names())}
            _ret_col = self.config.label.return_col
            if _ret_col in split_df.columns and n_classes == 2:
                prev_ret = split_df.loc[mask, _ret_col].to_numpy(dtype=float)
                y_persist = (prev_ret > 0).astype(int)
                proba_persist = np.zeros((n, 2), dtype=float)
                proba_persist[y_persist == 1, 1] = 0.9
                proba_persist[y_persist == 0, 0] = 0.9
                proba_persist[y_persist == 1, 0] = 0.1
                proba_persist[y_persist == 0, 1] = 0.1
                baselines[split_name]['persistence'] = evaluate_classification(y, y_persist, proba_persist, class_names=label_maker.class_names())
        return baselines

    def _save_artifacts(self, result, model, preds_test, label_maker, X_test, y_test) -> None:
        out = Path(self.config.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        name = self.config.model_name
        family = self.config.label.target_family
        joblib.dump(model, out / f'{name}_{family}_model.joblib')
        (out / f'{name}_{family}_metrics.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, default=_json_default), encoding='utf-8')
        if preds_test is not None:
            preds_test.to_csv(out / f'{name}_{family}_test_predictions.csv', index=False)
        y_pred = model.predict(X_test)
        cm_df = confusion_matrix_df(y_test, y_pred, label_maker.class_names())
        cm_df.to_csv(out / f'{name}_{family}_confusion.csv')

def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f'Not serialisable: {type(obj)}')