from __future__ import annotations

import argparse
import json

import yaml

from chronos_ts.splits import TimeRangeSplitConfig
from chronos_ts.trainer import TrainConfig, TabularTrainer


def load_config(path: str) -> TrainConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    split_cfg = raw.pop("split", {})
    raw["split"] = TimeRangeSplitConfig(**split_cfg)
    return TrainConfig(**raw)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train tabular forecasting model.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    args = parser.parse_args()

    config = load_config(args.config)
    trainer = TabularTrainer(config)
    out = trainer.run()
    print(json.dumps(out["result"], ensure_ascii=False, indent=2))
    print(f"Saved model: {out['model_path']}")
    print(f"Saved metrics: {out['metrics_path']}")
    print(f"Saved predictions: {out['predictions_path']}")


if __name__ == "__main__":
    main()
