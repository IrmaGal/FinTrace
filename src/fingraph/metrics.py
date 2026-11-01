"""Imbalance-aware, threshold-explicit binary classification evaluation."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score, f1_score, precision_score, recall_score,
    roc_auc_score,
)


def choose_f1_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    candidates = np.linspace(0.05, 0.95, 91)
    scores = [f1_score(y_true, probabilities >= t, zero_division=0) for t in candidates]
    return float(candidates[int(np.argmax(scores))])


def evaluate(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    y_true = np.asarray(y_true, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=float)
    if len(y_true) != len(probabilities) or len(y_true) == 0:
        raise ValueError("Predikcije i oznake moraju imati jednaku, nepraznu dužinu")
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("Metrike računamo isključivo na poznatim oznakama 0/1")
    if np.unique(y_true).size != 2:
        raise ValueError("Potrebne su obje klase za PR-AUC i ROC-AUC")
    predicted = probabilities >= threshold
    k = min(100, len(y_true))
    top_k = np.argsort(-probabilities)[:k]
    return {
        "n": len(y_true),
        "positives": int(y_true.sum()),
        "threshold": float(threshold),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "precision_at_100_or_less": float(y_true[top_k].mean()),
        "recall_at_100_or_less": float(y_true[top_k].sum() / y_true.sum()),
    }
