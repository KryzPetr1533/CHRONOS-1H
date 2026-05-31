"""
Load the PRD-tagged model from MLflow and run a test prediction.
Bonus deliverable for Task 2 Phase 2.

CLI (Hydra):
    python scripts/predict_prd.py
    python scripts/predict_prd.py predict.input_csv=outputs/datasets/btcusdt_clf_core.csv predict.n_rows=5

Notebook / script import:
    from scripts.predict_prd import PrdPredictor
    df_preds = PrdPredictor().predict(input_csv='outputs/datasets/btcusdt_clf_core.csv', n_rows=10)
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@dataclass
class PrdPredictorConfig:
    registered_model_name: str = "chronos_1h_prd"
    tracking_uri: str = "http://localhost:5050"
    s3_endpoint_url: str = "http://localhost:9000"
    aws_access_key_id: str = "admin"
    aws_secret_access_key: str = "password"
    metrics_path: str = "outputs/models/clf/catboost_vol_regime/catboost_vol_regime_metrics.json"


class PrdPredictor:
    """
    Load models:/chronos_1h_prd@prd and predict on new data.

    Callable from notebooks and plain scripts — no Hydra required.
    """

    def __init__(self, cfg: Optional[PrdPredictorConfig] = None):
        self.cfg = cfg or PrdPredictorConfig()
        self._model = None

    def load(self) -> None:
        """Explicitly load the PRD model (lazy-loaded on first predict() call)."""
        from chronos_ts.tracking import configure_mlflow
        import mlflow

        configure_mlflow(
            tracking_uri=self.cfg.tracking_uri,
            experiment_name="chronos-1h-classification",
            s3_endpoint_url=self.cfg.s3_endpoint_url,
            aws_access_key_id=self.cfg.aws_access_key_id,
            aws_secret_access_key=self.cfg.aws_secret_access_key,
        )
        uri = f"models:/{self.cfg.registered_model_name}@prd"
        print(f"Loading {uri} ...")
        self._model = mlflow.pyfunc.load_model(uri)
        print(f"Loaded. Flavors: {list(self._model.metadata.flavors.keys())}")

    def predict(
        self,
        input_csv: str = "outputs/datasets/btcusdt_clf_core.csv",
        n_rows: int = 10,
    ) -> pd.DataFrame:
        """
        Load input CSV, reconstruct feature matrix, run model.predict().

        Returns a DataFrame with ts, y_true (if available), y_pred columns.
        """
        if self._model is None:
            self.load()

        feature_cols = self._load_feature_cols()

        df = pd.read_csv(input_csv, parse_dates=["ts"] if "ts" in pd.read_csv(input_csv, nrows=0).columns else [])
        df = df.sort_values("ts").reset_index(drop=True) if "ts" in df.columns else df

        from chronos_ts.labels import LabelConfig, LabelMaker
        from chronos_ts.splits import TimeRangeSplitConfig, time_fraction_split

        split_cfg = TimeRangeSplitConfig(0.70, 0.15, 0.15)
        splits = time_fraction_split(df, split_cfg, ts_col="ts")

        lm = LabelMaker(LabelConfig(target_family="vol_regime"))
        lm.fit(splits["train"])

        y_test = lm.transform(splits["test"])
        mask   = y_test.notna()
        X_test = splits["test"].loc[mask, [c for c in feature_cols if c in splits["test"].columns]]

        sample = X_test.tail(n_rows).reset_index(drop=True)
        preds  = self._model.predict(sample).astype(int)

        class_names = lm.class_names()
        result = pd.DataFrame({
            "ts":     splits["test"].loc[mask, "ts"].tail(n_rows).values if "ts" in splits["test"].columns else range(n_rows),
            "y_true": [class_names[i] for i in y_test[mask].values.astype(int)[-n_rows:]],
            "y_pred": [class_names[i] for i in preds],
        })
        result["correct"] = result["y_true"] == result["y_pred"]
        return result

    def _load_feature_cols(self) -> list[str]:
        if Path(self.cfg.metrics_path).exists():
            return json.load(open(self.cfg.metrics_path))["feature_cols"]
        raise FileNotFoundError(
            f"Metrics file not found: {self.cfg.metrics_path}\n"
            "Run experiment_tracking.ipynb first to train and register the PRD model."
        )


# ------------------------------------------------------------------ #
# Hydra CLI entry point
# ------------------------------------------------------------------ #

try:
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base=None, config_path="../conf", config_name="config")
    def main(cfg: DictConfig) -> None:
        pred_cfg = PrdPredictorConfig(
            registered_model_name=cfg.mlflow.registered_model_name,
            tracking_uri=cfg.mlflow.tracking_uri,
            s3_endpoint_url=cfg.mlflow.s3_endpoint_url,
            aws_access_key_id=cfg.mlflow.aws_access_key_id,
            aws_secret_access_key=cfg.mlflow.aws_secret_access_key,
        )
        predictor = PrdPredictor(pred_cfg)
        import hydra as _h
        project_root = Path(_h.utils.get_original_cwd())

        result = predictor.predict(
            input_csv=str(project_root / cfg.predict.input_csv),
            n_rows=int(cfg.predict.n_rows),
        )
        print(result.to_string(index=False))
        print(f"\nAccuracy on sample: {result['correct'].mean():.2%}")

except ImportError:
    def main() -> None:
        print("hydra-core not installed; use PrdPredictor().predict() instead.")


if __name__ == "__main__":
    main()
