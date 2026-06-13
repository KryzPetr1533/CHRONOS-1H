from __future__ import annotations
from pathlib import Path
from chronos_ts.experiments.base import BaseExperiment
from chronos_ts.experiments.classification import ClassificationExperiment
from chronos_ts.experiments.legacy_script import LegacyScriptExperiment
from chronos_ts.experiments.seq import SeqRegressionExperiment
from chronos_ts.experiments.tabular import TabularRegressionExperiment

REPO = Path(__file__).resolve().parents[2]


def _legacy(name: str, script: str, out: str, metrics: str, experiment: str, model_key: str | None=None) -> LegacyScriptExperiment:
    return LegacyScriptExperiment(name=name, script=REPO / script, output_dir=REPO / out, metrics_file=metrics, default_mlflow_experiment=experiment, model_key=model_key)


REGISTRY: dict[str, type[BaseExperiment] | BaseExperiment] = {
    'classification': ClassificationExperiment,
    'seq': SeqRegressionExperiment,
    'ridge': lambda: TabularRegressionExperiment(name='ridge', default_mlflow_experiment='chronos-1h-regression-ridge'),
    'catboost_reg': lambda: TabularRegressionExperiment(name='catboost_reg', default_mlflow_experiment='chronos-1h-regression-catboost'),
    'timexer': lambda: _legacy('timexer', 'legacy/regression_seq/scripts/train_timexer.py', 'outputs/models/timexer_core_tuned', 'timexer_metrics.json', 'chronos-1h-regression-timexer', model_key='timexer'),
    'patchtst': lambda: _legacy('patchtst', 'legacy/regression_seq/scripts/train_patchtst.py', 'outputs/models/patchtst_core', 'patchtst_metrics.json', 'chronos-1h-regression-patchtst', model_key='patchtst'),
    'chronos2': lambda: _legacy('chronos2', 'scripts/train_chronos2.py', 'outputs/models/chronos2_core', 'chronos2_metrics.json', 'chronos-1h-regression-chronos2'),
    'har_vol': lambda: _legacy('har_vol', 'scripts/train_har_vol.py', 'outputs/models/har_vol_small', 'har_metrics.json', 'chronos-1h-regression-har'),
    'garch': lambda: _legacy('garch', 'scripts/train_garch.py', 'outputs/models/garch_vol_small', 'garch_metrics.json', 'chronos-1h-regression-garch'),
}

EXPERIMENT_NAMES = tuple(REGISTRY.keys())


def get_experiment(name: str) -> BaseExperiment:
    if name not in REGISTRY:
        raise KeyError(f'Unknown experiment {name!r}. Choose from: {", ".join(EXPERIMENT_NAMES)}')
    entry = REGISTRY[name]
    if isinstance(entry, type):
        return entry()
    if callable(entry):
        return entry()
    return entry
