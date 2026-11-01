"""XGBoost baseline: train only on labeled nodes, time-ordered test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from .data import load_elliptic
from .metrics import choose_f1_threshold, evaluate
from .splits import make_time_splits


def run(data_dir: str, output_dir: str, seed: int = 42) -> dict:
    from xgboost import XGBClassifier

    graph = load_elliptic(data_dir)
    splits = make_time_splits(graph)

    # Fit on all past node features, not future information. Using the same
    # historical observations as the GNN reduces preprocessing differences.
    observed_train = graph.times <= splits.train_end
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    imputer.fit(graph.features[observed_train])
    x_train = imputer.transform(graph.features[splits.train])
    x_valid = imputer.transform(graph.features[splits.valid])
    x_test = imputer.transform(graph.features[splits.test])
    y_train = graph.labels[splits.train]

    negatives = int((y_train == 0).sum())
    positives = int((y_train == 1).sum())
    model = XGBClassifier(
        n_estimators=350, max_depth=5, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85, tree_method="hist",
        objective="binary:logistic", eval_metric="logloss",
        scale_pos_weight=negatives / positives,
        random_state=seed, n_jobs=2,
    )
    model.fit(x_train, y_train)

    p_valid = model.predict_proba(x_valid)[:, 1]
    threshold = choose_f1_threshold(graph.labels[splits.valid], p_valid)
    p_test = model.predict_proba(x_test)[:, 1]
    result = {
        "model": "XGBoost", "seed": seed,
        "train_end": splits.train_end, "valid_end": splits.valid_end,
        "validation": evaluate(graph.labels[splits.valid], p_valid, threshold),
        "test": evaluate(graph.labels[splits.test], p_test, threshold),
    }

    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame({
        "txId": graph.tx_ids[splits.test],
        "time_step": graph.times[splits.test],
        "label": graph.labels[splits.test],
        "probability_illicit": p_test,
    }).to_csv(dest / "test_predictions.csv", index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--output-dir", default="outputs/xgboost")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(run(args.data_dir, args.output_dir, args.seed), indent=2))


if __name__ == "__main__":
    main()
