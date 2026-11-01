"""Small synthetic fixture: tests do not depend on private or public AML data."""

from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fingraph.data import load_elliptic
from fingraph.audit import describe
from fingraph.metrics import choose_f1_threshold, evaluate
from fingraph.splits import edges_observed_until, make_time_splits


class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)
        self.ids = [101, 102, 103, 201, 202, 203, 301, 302, 303]
        pd.DataFrame({
            "txId": self.ids,
            "Time step": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "amount_like": [1, 2, np.nan, 3, 4, 5, 6, 7, 8],
            "feature_b": list(range(9)),
            # If a notebook mistakenly adds class here, loader must drop it.
            "class": [1, 2, 3, 1, 2, 3, 1, 2, 3],
        }).to_csv(self.folder / "txs_features.csv", index=False)
        pd.DataFrame({
            "txId": self.ids[::-1],
            "class": [3, 2, 1, 3, 2, 1, 3, 2, 1],
        }).to_csv(self.folder / "txs_classes.csv", index=False)
        pd.DataFrame({
            "txId1": [101, 201, 301, 102],
            "txId2": [102, 202, 302, 301],
        }).to_csv(self.folder / "txs_edgelist.csv", index=False)

    def test_mapping_labels_features_and_edges(self):
        graph = load_elliptic(self.folder)
        self.assertEqual(graph.num_nodes, 9)
        self.assertEqual(graph.num_edges, 4)
        self.assertEqual(graph.labels.tolist(), [1, 0, -1] * 3)
        self.assertEqual(graph.features.shape, (9, 2))
        self.assertNotIn("class", graph.feature_names)
        self.assertTrue(np.isnan(graph.features[2, 0]))
        self.assertEqual(graph.edge_index[:, 0].tolist(), [0, 1])

    def test_time_splits_and_edge_filter(self):
        graph = load_elliptic(self.folder)
        split = make_time_splits(graph, train_end=1, valid_end=2)
        self.assertEqual(int(split.train.sum()), 2)
        self.assertEqual(int(split.valid.sum()), 2)
        self.assertEqual(int(split.test.sum()), 2)
        observed = edges_observed_until(graph, 2)
        self.assertEqual(observed.shape[1], 2)

    def test_unknown_edges_fail_instead_of_disappearing(self):
        pd.DataFrame({"txId1": [101], "txId2": [999]}).to_csv(
            self.folder / "txs_edgelist.csv", index=False
        )
        with self.assertRaisesRegex(ValueError, "nepoznatim txId"):
            load_elliptic(self.folder)

    def test_duplicate_ids_fail(self):
        pd.DataFrame({"txId": [101, 101], "class": [1, 2]}).to_csv(
            self.folder / "txs_classes.csv", index=False
        )
        with self.assertRaisesRegex(ValueError, "jedinstven"):
            load_elliptic(self.folder)

    def test_metrics_known_labels_only(self):
        true = np.array([1, 1, 0, 0])
        probabilities = np.array([0.9, 0.75, 0.2, 0.1])
        threshold = choose_f1_threshold(true, probabilities)
        result = evaluate(true, probabilities, threshold)
        self.assertAlmostEqual(result["pr_auc"], 1.0)
        self.assertAlmostEqual(result["recall"], 1.0)
        with self.assertRaisesRegex(ValueError, "poznatim oznakama"):
            evaluate(np.array([1, -1, 0, 0]), probabilities, threshold)

    def test_audit_is_json_serializable(self):
        report = describe(str(self.folder))
        self.assertEqual(report["labels"], {"illicit": 3, "licit": 3, "unknown": 3})
        self.assertIn('"nodes": 9', json.dumps(report))


if __name__ == "__main__":
    unittest.main()
