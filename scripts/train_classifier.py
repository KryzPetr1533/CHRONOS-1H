"""
Train a classification model on BTCUSDT data.

CLI (Hydra — from project root with PYTHONPATH=.):

    # Default: vol_regime + catboost on core data
    python scripts/train_classifier.py

    # Override label and model
    python scripts/train_classifier.py label=direction model=logreg

    # Multirun sweep
    python scripts/train_classifier.py -m label=direction,large_move,vol_regime model=logreg,catboost

Notebook / script import (no Hydra required):

    from scripts.train_classifier import ClassifierRunner, RunConfig
    result = ClassifierRunner.from_defaults(
        label_family="vol_regime",
        model_name="catboost",
    ).run()

    # Full control:
    cfg = RunConfig(
        data_csv="outputs/datasets/btcusdt_clf_core.csv",
        label_family="vol_regime",
        model_name="catboost",
        output_dir="outputs/models/clf/catboost_vol_regime",
    )
    result = ClassifierRunner(cfg).run()
"""
from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

log = logging.getLogger(__name__)


@dataclass
class RunConfig:
    """Flat config for one training run — easy to construct from a notebook."""
    data_csv: str = "outputs/datasets/btcusdt_clf_core.csv"
    label_family: str = "vol_regime"
    model_name: str = "catboost"
    output_dir: str = "outputs/models/clf"
    ts_col: str = "ts"
    drop_cols: list[str] = field(default_factory=lambda: ["ts", "target_log_ret_1h"])
    train_frac: float = 0.70
    val_frac: float = 0.15
    test_frac: float = 0.15
    cv_splits: int = 5
    seed: int = 42
    param_grid: dict = field(default_factory=dict)
    abstention_threshold: float = 0.65
    top_k_pct: float = 0.20
    # LabelConfig overrides
    return_col: str = "log_ret_1h"
    horizon: int = 1
    dead_zone_sigma: float = 0.1
    move_quantile: float = 0.80
    n_tokens: int = 5
    vol_window: int = 6


class ClassifierRunner:
    """
    End-to-end classification trainer.

    Usable from a notebook, a plain .py script, or wrapped by Hydra CLI.
    All logic lives here; train_classifier.py's main() is just a Hydra wrapper.
    """

    def __init__(self, cfg: RunConfig):
        self.cfg = cfg

    @classmethod
    def from_defaults(
        cls,
        label_family: str = "vol_regime",
        model_name: str = "catboost",
        data_csv: str = "outputs/datasets/btcusdt_clf_core.csv",
        **kwargs: Any,
    ) -> "ClassifierRunner":
        """Convenience constructor for notebooks."""
        return cls(RunConfig(
            data_csv=data_csv,
            label_family=label_family,
            model_name=model_name,
            **kwargs,
        ))

    def run(self) -> dict[str, Any]:
        """Train, evaluate, save artifacts. Returns the full result dict."""
        from chronos_ts.labels import LabelConfig
        from chronos_ts.splits import TimeRangeSplitConfig
        from chronos_ts.classification import ClassificationConfig, ClassificationTrainer

        c = self.cfg
        output_dir = str(Path(c.output_dir) / f"{c.model_name}_{c.label_family}")

        label_cfg = LabelConfig(
            target_family=c.label_family,
            return_col=c.return_col,
            horizon=c.horizon,
            dead_zone_sigma=c.dead_zone_sigma,
            move_quantile=c.move_quantile,
            n_tokens=c.n_tokens,
            vol_window=c.vol_window,
        )

        split_cfg = TimeRangeSplitConfig(
            train_frac=c.train_frac,
            val_frac=c.val_frac,
            test_frac=c.test_frac,
        )

        train_cfg = ClassificationConfig(
            data_csv=c.data_csv,
            output_dir=output_dir,
            label=label_cfg,
            model_name=c.model_name,
            ts_col=c.ts_col,
            drop_cols=c.drop_cols,
            split=split_cfg,
            cv_splits=c.cv_splits,
            seed=c.seed,
            param_grid=c.param_grid,
            abstention_threshold=c.abstention_threshold,
            top_k_pct=c.top_k_pct,
        )

        log.info("Training: model=%s  label=%s", c.model_name, c.label_family)
        trainer = ClassificationTrainer(train_cfg)
        result = trainer.run()
        self._print_summary(result)
        return result

    # ------------------------------------------------------------------ #

    def _print_summary(self, result: dict) -> None:
        for split_name in ("val", "test"):
            m = result["metrics"][split_name]
            print(
                f"[{split_name}]  balanced_acc={m.get('balanced_accuracy', float('nan')):.4f}"
                f"  mcc={m.get('mcc', float('nan')):.4f}"
                f"  roc_auc={m.get('roc_auc', float('nan')):.4f}"
                f"  coverage={m.get('trading_coverage', float('nan')):.2f}"
                f"  hit_rate={m.get('trading_hit_rate', float('nan')):.4f}"
            )
        for split_name in ("val", "test"):
            bl = result["baselines"].get(split_name, {})
            maj = bl.get("majority_class", {})
            print(
                f"[{split_name}] BASELINE majority_class"
                f"  balanced_acc={maj.get('balanced_accuracy', float('nan')):.4f}"
                f"  mcc={maj.get('mcc', float('nan')):.4f}"
            )
        print(f"\nArtifacts saved to: {Path(self.cfg.output_dir) / f'{self.cfg.model_name}_{self.cfg.label_family}'}")


# ------------------------------------------------------------------ #
# Hydra CLI entry point
# ------------------------------------------------------------------ #

def _build_run_config_from_hydra(cfg) -> RunConfig:
    """Convert an OmegaConf DictConfig into a flat RunConfig."""
    from omegaconf import OmegaConf
    import hydra

    project_root = Path(hydra.utils.get_original_cwd())
    param_grid = OmegaConf.to_container(cfg.param_grid, resolve=True) if cfg.get("param_grid") else {}

    return RunConfig(
        data_csv=str(project_root / cfg.data_csv),
        label_family=cfg.label.target_family,
        model_name=cfg.model_name,
        output_dir=str(project_root / cfg.output_base),
        ts_col=cfg.ts_col,
        drop_cols=list(cfg.drop_cols),
        train_frac=float(cfg.split.train_frac),
        val_frac=float(cfg.split.val_frac),
        test_frac=float(cfg.split.test_frac),
        cv_splits=int(cfg.cv_splits),
        seed=int(cfg.seed),
        param_grid=param_grid,
        abstention_threshold=float(cfg.abstention_threshold),
        top_k_pct=float(cfg.top_k_pct),
        return_col=cfg.label.get("return_col", "log_ret_1h"),
        horizon=int(cfg.label.get("horizon", 1)),
        dead_zone_sigma=float(cfg.label.get("dead_zone_sigma", 0.1)),
        move_quantile=float(cfg.label.get("move_quantile", 0.80)),
        n_tokens=int(cfg.label.get("n_tokens", 5)),
        vol_window=int(cfg.label.get("vol_window", 6)),
    )


try:
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base=None, config_path="../conf", config_name="config")
    def main(cfg: DictConfig) -> None:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
        run_cfg = _build_run_config_from_hydra(cfg)
        result = ClassifierRunner(run_cfg).run()
        print("\n--- Final result (test) ---")
        print(json.dumps(result["metrics"]["test"], indent=2, default=str))

except ImportError:
    def main() -> None:
        print("hydra-core not installed; use ClassifierRunner directly from Python.")


if __name__ == "__main__":
    main()
