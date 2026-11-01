"""Chronological evaluation helpers for saved Elliptic++ test predictions.

These functions do NOT select thresholds, train models or use unlabeled nodes as
negatives. They verify test IDs/labels against the canonical transaction graph.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)

from .data import TransactionGraph
from .splits import TimeSplits

REQUIRED_PREDICTION_COLUMNS = (
    "txId", "time_step", "label", "probability_illicit"
)


def load_and_check_test_predictions(
    path: str | Path, graph: TransactionGraph, splits: TimeSplits
) -> pd.DataFrame:
    """Read predictions and verify exact equality with known test IDs/labels.

    All saved test rows must be present exactly once; no train, validation or
    unknown-labelled nodes are allowed. Joining uses txId, never row order.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(
            f"Nije pronađen {file_path}. Prvo pokreni odgovarajući model "
            "(ne prepisuj druge rezultate)."
        )
    frame = pd.read_csv(file_path)
    missing = set(REQUIRED_PREDICTION_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"{file_path}: nedostaju kolone {sorted(missing)}")
    if frame["txId"].isna().any() or frame["txId"].duplicated().any():
        raise ValueError(f"{file_path}: txId mora biti jedinstven i neprazan")
    for col in ["time_step", "label", "probability_illicit"]:
        frame[col] = pd.to_numeric(frame[col], errors="raise")
    if frame[["time_step", "label", "probability_illicit"]].isna().any().any():
        raise ValueError(f"{file_path}: nedostaju predikcije/vrijednosti")
    probs = frame["probability_illicit"].to_numpy(dtype=float)
    if not np.isfinite(probs).all() or ((probs < 0) | (probs > 1)).any():
        raise ValueError(f"{file_path}: skorovi moraju biti konačni i u rasponu [0,1]")

    expected = pd.DataFrame({
        "txId": graph.tx_ids[splits.test],
        "time_step": graph.times[splits.test],
        "label": graph.labels[splits.test],
    })
    if len(frame) != len(expected):
        raise ValueError(
            f"{file_path}: očekivano {len(expected)} testnih redova, "
            f"pronađeno {len(frame)}"
        )
    combined = expected.merge(
        frame[list(REQUIRED_PREDICTION_COLUMNS)],
        on="txId", how="left", validate="one_to_one", suffixes=("_true", "_saved"),
        indicator=True,
    )
    if (combined["_merge"] != "both").any():
        raise ValueError(f"{file_path}: nedostaje ili je pogrešan testni txId")
    if not (
        np.array_equal(combined["time_step_true"].to_numpy(), combined["time_step_saved"].to_numpy())
        and np.array_equal(combined["label_true"].to_numpy(), combined["label_saved"].to_numpy())
    ):
        raise ValueError(f"{file_path}: neodgovarajući time_step ili label za txId")
    return combined[["txId", "time_step_true", "label_true", "probability_illicit"]].rename(
        columns={"time_step_true": "time_step", "label_true": "label"}
    )


def evaluate_per_time(predictions: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Compute metrics per step using a fixed validation-chosen threshold.

    The returned table includes prevalence. PR-AUC/ROC-AUC are NaN whenever
    a time step has only one known class. They should not be interpreted as 0.
    """
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold mora biti u [0,1]")
    records = []
    for step, group in predictions.groupby("time_step", sort=True):
        y = group["label"].to_numpy(dtype=int)
        p = group["probability_illicit"].to_numpy(dtype=float)
        if not np.isin(y, (0, 1)).all():
            raise ValueError("Nepoznate oznake nisu dozvoljene u testnim metrikama")
        positive = p >= threshold
        tn, fp, fn, tp = confusion_matrix(y, positive, labels=[0, 1]).ravel()
        valid_auc = np.unique(y).size == 2
        records.append({
            "time_step": int(step),
            "n": int(len(y)),
            "positives": int(y.sum()),
            "positive_rate": float(y.mean()),
            "predicted_positive": int(positive.sum()),
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
            "precision": float(precision_score(y, positive, zero_division=0)),
            "recall": float(recall_score(y, positive, zero_division=0)),
            "f1": float(f1_score(y, positive, zero_division=0)),
            "pr_auc": float(average_precision_score(y, p)) if valid_auc else float("nan"),
            "roc_auc": float(roc_auc_score(y, p)) if valid_auc else float("nan"),
        })
    return pd.DataFrame(records).sort_values("time_step").reset_index(drop=True)


def joined_predictions(predictions_by_model: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Align model predictions by txId and reject inconsistent test sets."""
    if not predictions_by_model:
        raise ValueError("Nema predikcija za poređenje")
    combined = None
    for name, pred in predictions_by_model.items():
        current = pred.rename(columns={"probability_illicit": f"score_{name}"})
        if combined is None:
            combined = current.copy()
        else:
            merged = combined.merge(
                current, on="txId", how="outer", validate="one_to_one",
                suffixes=("", "_other"), indicator=True,
            )
            if len(merged) != len(combined) or (merged["_merge"] != "both").any():
                raise ValueError(f"Model {name} nema isti skup testnih transakcija")
            for col in ("label", "time_step"):
                if not np.array_equal(merged[col].to_numpy(), merged[f"{col}_other"].to_numpy()):
                    raise ValueError(f"Model {name} nema isti {col} za iste txId")
            combined = merged.drop(columns=["label_other", "time_step_other", "_merge"])
    return combined.sort_values("time_step").reset_index(drop=True)
