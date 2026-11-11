
"""Tests for the fifth experiment."""

import unittest
from types import SimpleNamespace

import numpy as np

from fingraph.phase05 import (
    feature_indices,
    fit_processor,
    masks,
    FOLDS,
)


class Phase05Tests(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(42)

        self.graph = SimpleNamespace(
            times=np.array(
                [1] * 20 + [2] * 20 + [3] * 20
            ),
            labels=np.array(
                [0, 1] * 10
                + [0, 1] * 10
                + [0, 1] * 10
            ),
            feature_names=(
                "Local_feature_1",
                "Local_feature_2",
                "Aggregate_feature_1",
                "other",
            ),
            features=rng.uniform(
                0.1, 3, size=(60, 4)
            ).astype(np.float32),
        )

    def test_feature_groups(self):
        names = self.graph.feature_names

        self.assertEqual(
            feature_indices(
                names, "local_only"
            ).tolist(),
            [0, 1],
        )

        self.assertEqual(
            feature_indices(
                names, "aggregate_only"
            ).tolist(),
            [2],
        )

        self.assertEqual(
            feature_indices(
                names, "without_aggregate"
            ).tolist(),
            [0, 1, 3],
        )

        self.assertEqual(
            len(feature_indices(names, "all")),
            4,
        )
        
        self.assertEqual(
            feature_indices(
                self.graph.feature_names,
                "named_only",
            ).tolist(),
            [3],
        )

    def test_train_only_scaler(self):
        graph = self.graph

        processor = fit_processor(
            graph, 1, "all", "median_standard"
        )

        self.assertEqual(processor.fit_count, 20)

        before = processor.transform(
            graph.features[20:25]
        ).copy()

        graph.features[40:, 0] = 100000

        after = processor.transform(
            graph.features[20:25]
        )

        np.testing.assert_allclose(before, after)

    def test_log_robust(self):
        processor = fit_processor(
            self.graph,
            2,
            "local_only",
            "median_log_robust",
        )

        result = processor.transform(
            self.graph.features[20:]
        )

        self.assertEqual(result.shape, (40, 2))


    
    def test_new_preprocessing_variants(self):
        variants = (
            "median_robust",
            "median_quantile",
            "median_power",
        )

        for variant in variants:
            with self.subTest(variant=variant):
                processor = fit_processor(
                    self.graph,
                    2,
                    "all",
                    variant,
                )

                transformed = processor.transform(
                    self.graph.features
                )

                self.assertEqual(
                    transformed.shape,
                    (60, 4),
                )

                self.assertTrue(
                    np.isfinite(transformed).all()
                )

    def test_pca(self):
        processor = fit_processor(
            self.graph,
            2,
            "all_pca95",
            "median_standard",
        )

        explained = sum(
            processor.pca.explained_variance_ratio_
        )

        self.assertGreaterEqual(
            explained,
            0.95 - 1e-6,
        )

        result = processor.transform(
            self.graph.features
        )

        self.assertEqual(
            result.shape[1],
            processor.n_features,
        )

    def test_folds_are_chronological(self):
        self.assertEqual(len(FOLDS), 3)

        for train_end, valid_end in FOLDS.values():
            self.assertLess(train_end, valid_end)

    def test_masks(self):
        train, valid = masks(
            self.graph, 1, 2
        )

        self.assertEqual(train.sum(), 20)
        self.assertEqual(valid.sum(), 20)


if __name__ == "__main__":
    unittest.main()
