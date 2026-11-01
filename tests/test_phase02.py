"""Run with: python -m unittest discover -s tests -v"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from fingraph.data import load_elliptic
from fingraph.splits import make_time_splits
from fingraph.temporal import (
    evaluate_per_time, joined_predictions, load_and_check_test_predictions,
)


class TemporalEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        # Each split contains both known classes, plus some unlabeled context.
        times = [1] * 8 + [35] * 6 + [42] * 6
        ids = np.arange(100, 120)
        classes = [1, 2, 3, 1, 2, 3, 1, 2] + [1, 2, 3, 1, 2, 3] + [1, 2, 3, 1, 2, 3]
        pd.DataFrame({
            "txId": ids, "Time step": times,
            "feature_a": np.arange(20, dtype=float),
            "feature_b": np.sin(np.arange(20, dtype=float)),
        }).to_csv(self.path / "txs_features.csv", index=False)
        pd.DataFrame({"txId": ids, "class": classes}).to_csv(
            self.path / "txs_classes.csv", index=False)
        pd.DataFrame({"txId1": ids[:-1], "txId2": ids[1:]}).to_csv(
            self.path / "txs_edgelist.csv", index=False)
        self.graph = load_elliptic(self.path)
        self.splits = make_time_splits(self.graph)
        self.expected = pd.DataFrame({
            "txId": self.graph.tx_ids[self.splits.test],
            "time_step": self.graph.times[self.splits.test],
            "label": self.graph.labels[self.splits.test],
            "probability_illicit": [0.9, 0.15, 0.8, 0.25],
        })

    def test_alignment_is_by_txid_not_row_index(self):
        path = self.path / "pred.csv"
        self.expected.sample(frac=1, random_state=42).to_csv(path, index=False)
        out = load_and_check_test_predictions(path, self.graph, self.splits)
        self.assertEqual(out["label"].tolist(), [1, 0, 1, 0])
        self.assertEqual(out["probability_illicit"].round(2).tolist(), [0.9, 0.15, 0.8, 0.25])

    def test_rejects_changed_labels_and_missing_ids(self):
        path = self.path / "bad.csv"
        changed = self.expected.copy()
        changed.loc[0, "label"] = 0
        changed.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "time_step ili label"):
            load_and_check_test_predictions(path, self.graph, self.splits)
        self.expected.iloc[:-1].to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "očekivano"):
            load_and_check_test_predictions(path, self.graph, self.splits)

    def test_temporal_metrics_and_merging(self):
        per_time = evaluate_per_time(self.expected, threshold=0.5)
        self.assertEqual(per_time.iloc[0]["n"], 4)
        self.assertEqual(per_time.iloc[0]["tp"], 2)
        self.assertEqual(per_time.iloc[0]["fp"], 0)
        self.assertAlmostEqual(per_time.iloc[0]["pr_auc"], 1.0)
        other = self.expected.copy()
        other["probability_illicit"] = 1.0 - other["probability_illicit"]
        out = joined_predictions({"A": self.expected, "B": other})
        self.assertEqual(len(out), 4)
        self.assertIn("score_A", out.columns)
        self.assertIn("score_B", out.columns)

    def test_empty_class_auc_is_nan(self):
        frame = pd.DataFrame({
            "time_step": [42, 42, 43, 43],
            "label": [0, 0, 0, 1],
            "probability_illicit": [0.2, 0.4, 0.1, 0.7],
        })
        res = evaluate_per_time(frame, 0.5)
        self.assertTrue(np.isnan(res.iloc[0]["pr_auc"]))
        self.assertTrue(np.isnan(res.iloc[0]["roc_auc"]))
        self.assertFalse(np.isnan(res.iloc[1]["pr_auc"]))

    def test_mlp_smoke_training(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch not installed")
        from fingraph.mlp import run
        out = self.path / "outputs" / "mlp"
        result = run(str(self.path), str(out), epochs=3, hidden_dim=8, patience=2)
        self.assertEqual(result["test"]["n"], 4)
        self.assertEqual(result["model"], "MLP")
        for filename in ["metrics.json", "training_history.csv", "preprocessing.joblib",
                         "mlp_checkpoint.pt", "test_predictions.csv", "validation_predictions.csv"]:
            self.assertTrue((out / filename).exists(), filename)
        saved = json.loads((out / "metrics.json").read_text())
        self.assertEqual(saved["test"]["n"], 4)


if __name__ == "__main__":
    unittest.main()
