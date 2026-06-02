from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chronos_ts.dataset import DatasetBuildConfig, ExperimentDatasetBuilder
from chronos_ts.labels import LabelConfig, LabelMaker
from chronos_ts.splits import TimeRangeSplitConfig, time_fraction_split

class ClfDatasetBuilder:

    def __init__(self, input_csv: str='data/btcusdt_1h_merged.csv', output_csv: str='outputs/datasets/btcusdt_clf_core.csv', lag_hours: list[int] | None=None, rolling_windows: list[int] | None=None, min_history_hours: int=168):
        self.input_csv = input_csv
        self.output_csv = output_csv
        self.lag_hours = lag_hours or [1, 2, 3, 6, 12, 24, 48, 72, 168]
        self.rolling_windows = rolling_windows or [6, 24, 72, 168]
        self.min_history_hours = min_history_hours

    def run(self) -> pd.DataFrame:
        df = self.build_only()
        self.sanity_check(df)
        self.preview_labels(df)
        self._print_transfer_instructions()
        return df

    def build_only(self) -> pd.DataFrame:
        config = DatasetBuildConfig(input_csv=self.input_csv, output_csv=self.output_csv, profile='core', lag_hours=self.lag_hours, rolling_windows=self.rolling_windows, min_history_hours=self.min_history_hours)
        builder = ExperimentDatasetBuilder(config)
        df = builder.build()
        builder.save(df)
        meta = builder.metadata(df)
        meta_path = Path(self.output_csv).with_suffix('.meta.json')
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')
        print(f'Feature matrix saved → {self.output_csv}')
        print(f"  rows={meta['n_rows']}  features={meta['n_features']}")
        print(f"  ts_min={meta['ts_min']}  ts_max={meta['ts_max']}")
        return df

    def sanity_check(self, df: pd.DataFrame) -> None:
        print('\n=== Dataset sanity check ===')
        errors: list[str] = []
        ts = pd.to_datetime(df['ts'])
        if not ts.is_monotonic_increasing:
            errors.append('FAIL: ts column is not monotonically increasing')
        else:
            print('OK  ts is sorted')
        n_dup = ts.duplicated().sum()
        if n_dup > 0:
            errors.append(f'FAIL: {n_dup} duplicate timestamps')
        else:
            print('OK  no duplicate timestamps')
        null_cols = [c for c in df.columns if df[c].isna().all()]
        if null_cols:
            errors.append(f'FAIL: all-null columns: {null_cols}')
        else:
            print(f'OK  no all-null columns ({df.shape[1]} total)')
        if 'target_log_ret_1h' not in df.columns:
            errors.append('FAIL: target_log_ret_1h column missing')
        else:
            n_valid = df['target_log_ret_1h'].notna().sum()
            print(f'OK  target_log_ret_1h: {n_valid}/{len(df)} non-null values')
        if 'log_ret_1h' not in df.columns:
            errors.append('FAIL: log_ret_1h column missing')
        else:
            print(f"OK  log_ret_1h present ({df['log_ret_1h'].notna().sum()} non-null)")
        if 'log_ret_1h_lag_1' in df.columns:
            expected = df['log_ret_1h'].shift(1)
            actual = df['log_ret_1h_lag_1']
            match = (expected.dropna() - actual.dropna()).abs().max()
            if match < 1e-10:
                print('OK  lag_1 matches shift(1) — no future leakage in lags')
            else:
                errors.append(f'FAIL: lag_1 mismatch vs shift(1): max diff = {match:.2e}')
        null_frac = df.isnull().mean()
        high_null = null_frac[null_frac > 0.1]
        if len(high_null):
            print(f"WARN: {len(high_null)} columns with >10% nulls (expected for 'rich' cols early in data):")
            for c, f in high_null.items():
                print(f'       {c}: {f:.1%} null')
        else:
            print('OK  no column has >10% nulls')
        if errors:
            print('\n=== ERRORS ===')
            for e in errors:
                print(' ', e)
            sys.exit(1)
        else:
            print('\n=== All checks passed ===')

    def preview_labels(self, df: pd.DataFrame) -> list[dict]:
        print('\n=== Label preview (train-split-only edge fitting) ===')
        split_cfg = TimeRangeSplitConfig(train_frac=0.7, val_frac=0.15, test_frac=0.15)
        splits = time_fraction_split(df, split_cfg, ts_col='ts')
        families = [LabelConfig(target_family='direction'), LabelConfig(target_family='large_move'), LabelConfig(target_family='vol_regime'), LabelConfig(target_family='horizon_dir', horizon=6), LabelConfig(target_family='return_token', n_tokens=5)]
        rows = []
        for cfg in families:
            lm = LabelMaker(cfg)
            lm.fit(splits['train'])
            y_train = lm.transform(splits['train']).dropna()
            y_test = lm.transform(splits['test']).dropna()
            n_classes = lm.n_classes()
            train_counts = np.bincount(y_train.astype(int), minlength=n_classes)
            test_counts = np.bincount(y_test.astype(int), minlength=n_classes)
            row = {'family': cfg.target_family, 'n_classes': n_classes, 'train_n': int(len(y_train)), 'test_n': int(len(y_test)), 'train_base_rates': {lm.class_names()[i]: float(c / train_counts.sum()) for i, c in enumerate(train_counts)}, 'test_base_rates': {lm.class_names()[i]: float(c / test_counts.sum()) for i, c in enumerate(test_counts)}, 'edges': lm._edges.tolist() if lm._edges is not None else None}
            rows.append(row)
            print(f'\n  {cfg.target_family} ({n_classes} classes)')
            print(f'    train n={len(y_train)}, test n={len(y_test)}')
            print(f"    train base rates: { {k: f'{v:.3f}' for k, v in row['train_base_rates'].items()}}")
            print(f"    test  base rates: { {k: f'{v:.3f}' for k, v in row['test_base_rates'].items()}}")
        out_path = Path('outputs/datasets/label_preview.json')
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(rows, indent=2, default=str), encoding='utf-8')
        print(f'\nLabel preview saved → {out_path}')
        return rows

    def _print_transfer_instructions(self) -> None:
        print(f'\n=== Ready for training ===')
        print(f'Transfer to another PC:')
        print(f'  {self.output_csv}')
        print(f"  {Path(self.output_csv).with_suffix('.meta.json')}")
        print(f'  outputs/datasets/label_preview.json')
        print(f'\nThen run on that PC:')
        print(f'  python scripts/train_classifier.py data=btcusdt_core label=vol_regime model=catboost')

def main() -> None:
    parser = argparse.ArgumentParser(description='Build classification-ready BTCUSDT dataset.')
    parser.add_argument('--input', default='data/btcusdt_1h_merged.csv')
    parser.add_argument('--output', default='outputs/datasets/btcusdt_clf_core.csv')
    args = parser.parse_args()
    ClfDatasetBuilder(input_csv=args.input, output_csv=args.output).run()
if __name__ == '__main__':
    main()