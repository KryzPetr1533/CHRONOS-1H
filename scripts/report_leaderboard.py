#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DEFAULT_GLOB = 'outputs/models/clf/*/*_metrics.json'
METRIC_KEYS = ('balanced_accuracy', 'mcc', 'roc_auc', 'trading_coverage', 'trading_hit_rate')


def load_row(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding='utf-8'))
    model = data.get('config', {}).get('model_name', path.parent.name.split('_')[0])
    family = data.get('config', {}).get('label', {}).get('target_family', '')
    if not family:
        parts = path.stem.replace('_metrics', '').split('_', 1)
        family = parts[1] if len(parts) > 1 else ''
    rows = []
    for split in ('val', 'test'):
        m = data.get('metrics', {}).get(split, {})
        row = {'model': model, 'target_family': family, 'split': split}
        for k in METRIC_KEYS:
            row[k] = m.get(k)
        bl = data.get('baselines', {}).get(split, {})
        maj = bl.get('majority_class', {})
        row['baseline_bal_acc'] = maj.get('balanced_accuracy')
        best_bl = max((v.get('balanced_accuracy', float('-inf')) for v in bl.values() if isinstance(v, dict)), default=float('nan'))
        row['best_baseline_bal_acc'] = best_bl if best_bl != float('-inf') else None
        row['beats_best_baseline'] = bool(m.get('balanced_accuracy', 0) > (best_bl if best_bl == best_bl else 0))
        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description='Build classification leaderboard CSV from metrics JSON files')
    parser.add_argument('--glob', default=DEFAULT_GLOB, help='Glob under repo root for *_metrics.json')
    parser.add_argument('--out', default='outputs/reports/phase1_leaderboard.csv')
    args = parser.parse_args()
    paths = sorted(REPO.glob(args.glob))
    if not paths:
        print(f'No metrics files matching {args.glob}')
        return 1
    rows: list[dict] = []
    for p in paths:
        try:
            rows.extend(load_row(p))
        except Exception as exc:
            print(f'Skip {p}: {exc}')
    df = pd.DataFrame(rows)
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f'Wrote {len(df)} rows → {out}')
    if 'test' in df['split'].values:
        test = df[df['split'] == 'test'].sort_values('balanced_accuracy', ascending=False)
        print(test[['model', 'target_family', 'balanced_accuracy', 'roc_auc', 'beats_best_baseline']].to_string(index=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
