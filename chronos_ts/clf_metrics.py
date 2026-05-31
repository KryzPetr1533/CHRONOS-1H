from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
)


def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    proba: np.ndarray,
    class_names: Optional[List[str]] = None,
    return_col: Optional[np.ndarray] = None,
    abstention_threshold: float = 0.65,
    top_k_pct: float = 0.20,
) -> dict:
    """
    Compute classification metrics for a single split.

    Parameters
    ----------
    y_true       : integer class labels
    y_pred       : predicted integer class labels
    proba        : probability array, shape (n, n_classes)
    class_names  : list of class names (for confusion matrix columns)
    return_col   : raw log-returns aligned to y_true (for trading metric)
    abstention_threshold : min max-class-proba to take a position
    top_k_pct    : top fraction by confidence for confident_directional_accuracy

    Returns
    -------
    dict with all metrics (JSON-serialisable floats and lists)
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    proba = np.asarray(proba, dtype=float)

    n_classes = proba.shape[1] if proba.ndim == 2 else 2
    is_binary = n_classes == 2

    # --- Core classification metrics ---
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    mcc = float(matthews_corrcoef(y_true, y_pred))

    try:
        ll = float(log_loss(y_true, proba))
    except Exception:
        ll = float("nan")

    # --- ROC-AUC / PR-AUC ---
    try:
        if is_binary:
            roc_auc = float(roc_auc_score(y_true, proba[:, 1]))
            pr_auc = float(average_precision_score(y_true, proba[:, 1]))
        else:
            roc_auc = float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro"))
            pr_auc = float("nan")
    except Exception:
        roc_auc = float("nan")
        pr_auc = float("nan")

    # --- Confident directional accuracy (top K% by max proba) ---
    max_proba = proba.max(axis=1)
    n_top = max(1, int(len(y_true) * top_k_pct))
    top_idx = np.argsort(max_proba)[-n_top:]
    confident_acc = float(accuracy_score(y_true[top_idx], y_pred[top_idx]))
    confident_bal_acc = float(balanced_accuracy_score(y_true[top_idx], y_pred[top_idx]))

    # --- Abstention trading metric (binary direction proxy) ---
    trading = _trading_metric(
        y_true, y_pred, proba, return_col, abstention_threshold, is_binary
    )

    # --- Confusion matrix ---
    cm = confusion_matrix(y_true, y_pred).tolist()
    labels = class_names or [str(i) for i in range(n_classes)]

    # --- Base rates ---
    unique, counts = np.unique(y_true, return_counts=True)
    base_rates = {labels[int(u)] if int(u) < len(labels) else str(u): int(c) / len(y_true)
                  for u, c in zip(unique, counts)}

    return {
        "n_samples": int(len(y_true)),
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "mcc": mcc,
        "log_loss": ll,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confident_accuracy": confident_acc,
        "confident_balanced_accuracy": confident_bal_acc,
        "confident_top_k_pct": top_k_pct,
        "abstention_threshold": abstention_threshold,
        **trading,
        "confusion_matrix": cm,
        "class_names": labels,
        "base_rates": base_rates,
    }


def _trading_metric(
    y_true, y_pred, proba, return_col, threshold, is_binary
) -> dict:
    """
    Sign-strategy metric: only trade when max_class_proba >= threshold.
    For binary: class 1 = long (+1), class 0 = short (-1).
    For multiclass: map predicted class to {+1, -1, 0} via top/bottom class.
    """
    max_proba = proba.max(axis=1)
    act_mask = max_proba >= threshold
    coverage = float(act_mask.mean())

    if return_col is None or not act_mask.any():
        return {
            "trading_coverage": coverage,
            "trading_hit_rate": float("nan"),
            "trading_sign_mean": float("nan"),
            "trading_sharpe_annual": float("nan"),
        }

    ret = np.asarray(return_col, dtype=float)

    if is_binary:
        positions = np.where(y_pred == 1, 1.0, -1.0)
    else:
        n_classes = proba.shape[1]
        # top class = long, bottom class = short, rest = flat
        positions = np.where(
            y_pred == n_classes - 1, 1.0,
            np.where(y_pred == 0, -1.0, 0.0)
        )

    positions_active = positions[act_mask]
    ret_active = ret[act_mask]
    pnl = positions_active * ret_active

    hit_rate = float((pnl > 0).mean()) if len(pnl) > 0 else float("nan")
    sign_mean = float(np.mean(pnl)) if len(pnl) > 0 else float("nan")
    pnl_std = float(np.std(pnl))
    sharpe = float(np.mean(pnl) / pnl_std * np.sqrt(24 * 365)) if pnl_std > 0 else float("nan")

    return {
        "trading_coverage": coverage,
        "trading_hit_rate": hit_rate,
        "trading_sign_mean": sign_mean,
        "trading_sharpe_annual": sharpe,
    }


def confusion_matrix_df(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]
) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred)
    return pd.DataFrame(cm, index=class_names, columns=class_names)
