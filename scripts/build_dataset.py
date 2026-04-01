from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from chronos_ts.dataset import DatasetBuildConfig, ExperimentDatasetBuilder


def load_config(path: str) -> DatasetBuildConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return DatasetBuildConfig(**raw)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build modeling-ready BTCUSDT datasets.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    args = parser.parse_args()

    config = load_config(args.config)
    builder = ExperimentDatasetBuilder(config)
    df = builder.build()
    builder.save(df)

    metadata = builder.metadata(df)
    meta_path = Path(config.output_csv).with_suffix(".meta.json")
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved dataset: {config.output_csv}")
    print(f"Saved metadata: {meta_path}")
    print(json.dumps({k: metadata[k] for k in ['profile', 'n_rows', 'n_features', 'ts_min', 'ts_max']}, indent=2))


if __name__ == "__main__":
    main()
