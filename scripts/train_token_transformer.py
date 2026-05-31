"""
Tokenized causal-sequence model: predict next return-token from a window of past tokens.

CLI:
    python scripts/train_token_transformer.py
    python scripts/train_token_transformer.py --lookback 48 --n-tokens 5 --epochs 20

Notebook / script import:
    from scripts.train_token_transformer import TokenTransformerRunner
    result = TokenTransformerRunner(data_csv='outputs/datasets/btcusdt_clf_core.csv').run()
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@dataclass
class TokenTransformerConfig:
    data_csv: str = "outputs/datasets/btcusdt_clf_core.csv"
    output_dir: str = "outputs/models/clf"
    n_tokens: int = 5
    lookback: int = 48        # past bars used as context
    hidden: int = 64
    n_layers: int = 2
    dropout: float = 0.2
    batch_size: int = 256
    epochs: int = 30
    lr: float = 1e-3
    patience: int = 5         # early stopping
    seed: int = 42
    train_frac: float = 0.70
    val_frac: float = 0.15
    test_frac: float = 0.15
    ts_col: str = "ts"
    return_col: str = "log_ret_1h"


class TokenTransformerRunner:
    """
    Train a GRU-based token sequence model.

    The model treats the discretized return series as a vocabulary and
    predicts the next token (next-bar vol-regime or return quantile).
    Evaluated with the same clf_metrics as tabular classifiers.
    """

    def __init__(self, cfg: Optional[TokenTransformerConfig] = None, **kwargs):
        self.cfg = cfg or TokenTransformerConfig(**{k: v for k, v in kwargs.items()
                                                    if k in TokenTransformerConfig.__dataclass_fields__})

    def run(self) -> dict[str, Any]:
        import torch
        from chronos_ts.labels import LabelConfig, LabelMaker
        from chronos_ts.splits import TimeRangeSplitConfig, time_fraction_split
        from chronos_ts.clf_metrics import evaluate_classification

        c = self.cfg
        _seed_everything(c.seed)

        df = pd.read_csv(c.data_csv, parse_dates=[c.ts_col])
        df = df.sort_values(c.ts_col).reset_index(drop=True)

        split_cfg = TimeRangeSplitConfig(c.train_frac, c.val_frac, c.test_frac)
        splits = time_fraction_split(df, split_cfg, ts_col=c.ts_col)

        lm = LabelMaker(LabelConfig(target_family="return_token", n_tokens=c.n_tokens))
        lm.fit(splits["train"])

        tokens_all = lm.transform(df)

        train_end = len(splits["train"])
        val_end   = train_end + len(splits["val"])

        tokens_train = tokens_all.iloc[:train_end].dropna().astype(int).values
        tokens_val   = tokens_all.iloc[train_end:val_end].dropna().astype(int).values
        tokens_test  = tokens_all.iloc[val_end:].dropna().astype(int).values

        def make_sequences(tokens):
            X, y = [], []
            for i in range(len(tokens) - c.lookback):
                X.append(tokens[i: i + c.lookback])
                y.append(tokens[i + c.lookback])
            return np.array(X), np.array(y)

        X_tr, y_tr = make_sequences(tokens_train)
        X_va, y_va = make_sequences(tokens_val)
        X_te, y_te = make_sequences(tokens_test)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model  = _GRUTokenModel(vocab_size=c.n_tokens, hidden=c.hidden,
                                n_layers=c.n_layers, dropout=c.dropout).to(device)
        opt    = torch.optim.Adam(model.parameters(), lr=c.lr)
        crit   = torch.nn.CrossEntropyLoss()

        best_val_loss = np.inf
        best_state    = None
        no_improve    = 0

        for epoch in range(1, c.epochs + 1):
            model.train()
            train_loss = _run_epoch(model, opt, crit, X_tr, y_tr, c.batch_size, device, train=True)
            model.eval()
            val_loss   = _run_epoch(model, None, crit, X_va, y_va, c.batch_size, device, train=False)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= c.patience:
                    print(f"Early stop at epoch {epoch}  val_loss={val_loss:.4f}")
                    break

            if epoch % 5 == 0 or epoch == 1:
                print(f"Epoch {epoch:3d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        model.load_state_dict(best_state)

        metrics = {}
        for split_name, X, y in [("val", X_va, y_va), ("test", X_te, y_te)]:
            proba, y_pred = _predict(model, X, c.n_tokens, c.batch_size, device)
            m = evaluate_classification(y, y_pred, proba, class_names=lm.class_names())
            metrics[split_name] = m
            print(f"[{split_name}]  balanced_acc={m['balanced_accuracy']:.4f}  "
                  f"mcc={m['mcc']:.4f}  roc_auc={m['roc_auc']:.4f}")

        result = {
            "config": vars(c),
            "label_maker": lm.describe(),
            "metrics": metrics,
            "best_val_loss": float(best_val_loss),
            "n_params": sum(p.numel() for p in model.parameters()),
        }

        out = Path(c.output_dir) / "token_transformer"
        out.mkdir(parents=True, exist_ok=True)
        (out / "token_transformer_metrics.json").write_text(
            json.dumps(result, indent=2, default=str), encoding="utf-8"
        )
        torch.save(model.state_dict(), out / "token_transformer.pt")
        print(f"Saved to {out}")
        return result


# ------------------------------------------------------------------ #
# PyTorch model
# ------------------------------------------------------------------ #

class _GRUTokenModel(object.__class__):
    def __new__(cls, vocab_size, hidden, n_layers, dropout):
        import torch.nn as nn

        class GRUModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed  = nn.Embedding(vocab_size, hidden // 2)
                self.gru    = nn.GRU(hidden // 2, hidden, num_layers=n_layers,
                                     batch_first=True, dropout=dropout if n_layers > 1 else 0)
                self.drop   = nn.Dropout(dropout)
                self.head   = nn.Linear(hidden, vocab_size)

            def forward(self, x):
                e = self.embed(x)
                out, _ = self.gru(e)
                return self.head(self.drop(out[:, -1, :]))

        return GRUModel()


def _seed_everything(seed: int) -> None:
    import torch, random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def _run_epoch(model, opt, crit, X, y, batch_size, device, train: bool) -> float:
    import torch
    total_loss, n_batches = 0.0, 0
    idx = np.arange(len(X))
    if train:
        np.random.shuffle(idx)

    for i in range(0, len(idx), batch_size):
        batch_idx = idx[i: i + batch_size]
        Xb = torch.tensor(X[batch_idx], dtype=torch.long).to(device)
        yb = torch.tensor(y[batch_idx], dtype=torch.long).to(device)

        logits = model(Xb)
        loss   = crit(logits, yb)

        if train:
            opt.zero_grad(); loss.backward(); opt.step()

        total_loss += loss.item(); n_batches += 1

    return total_loss / max(n_batches, 1)


def _predict(model, X, n_tokens, batch_size, device):
    import torch
    import torch.nn.functional as F

    all_proba, all_pred = [], []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            Xb = torch.tensor(X[i: i + batch_size], dtype=torch.long).to(device)
            logits = model(Xb)
            proba  = F.softmax(logits, dim=-1).cpu().numpy()
            pred   = proba.argmax(axis=1)
            all_proba.append(proba); all_pred.append(pred)

    return np.vstack(all_proba), np.concatenate(all_pred)


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-csv",    default="outputs/datasets/btcusdt_clf_core.csv")
    parser.add_argument("--output-dir",  default="outputs/models/clf")
    parser.add_argument("--n-tokens",    type=int, default=5)
    parser.add_argument("--lookback",    type=int, default=48)
    parser.add_argument("--hidden",      type=int, default=64)
    parser.add_argument("--epochs",      type=int, default=30)
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    cfg = TokenTransformerConfig(
        data_csv=args.data_csv,
        output_dir=args.output_dir,
        n_tokens=args.n_tokens,
        lookback=args.lookback,
        hidden=args.hidden,
        epochs=args.epochs,
        seed=args.seed,
    )
    TokenTransformerRunner(cfg).run()


if __name__ == "__main__":
    main()
