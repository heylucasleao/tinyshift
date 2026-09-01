# Copyright (c) 2024-2026 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License


import numpy as np
import pandas as pd
import pytest

from tinyshift.outlier.hbos import HBOS


class TestHBOS:
    def setup_method(self):
        self.hbos = HBOS()
        self.hbos.feature_dtypes = np.array([pd.CategoricalDtype(), np.float64])
        self.hbos.feature_distributions = [
            {1: 0.2, 2: 0.3, 3: 0.5},
            [
                np.array([6.25, 0.0, 6.25, 0.0, 12.5]),
                np.array([0.1, 0.14, 0.18, 0.22, 0.26, 0.3]),
            ],
        ]
        self.hbos.n_features = 2
        self.hbos.n_features_in_ = 2

    def test_compute_outlier_score_categorical(self):
        X = np.array([[1, 0], [2, 0], [3, 0], [4, 0]])
        scores = self.hbos._compute_outlier_score(X, 0)
        expected_scores = -np.log(np.array([0.2, 0.3, 0.5, 1e-9]) + 1e-9)
        np.testing.assert_array_almost_equal(scores, expected_scores, decimal=3)

    def test_compute_outlier_score_continuous(self):
        X = np.array([[0, 0.1], [0, 0.2], [0, 0.3], [0, 0.3]])
        scores = self.hbos._compute_outlier_score(X, 1)
        expected_scores = -np.log(np.array([6.25, 6.25, 12.5, 12.5]) + 1e-9)
        np.testing.assert_array_almost_equal(scores, expected_scores, decimal=3)

    def test_fit(self):
        hbos = HBOS()
        X = np.array([[1, 0.1], [2, 0.2], [3, 0.3], [4, 0.4]])
        hbos.fit(X)
        assert hbos.feature_dtypes is not None
        assert hbos.feature_distributions is not None
        assert len(hbos.feature_dtypes) == X.shape[1]
        assert len(hbos.feature_distributions) == X.shape[1]
        assert hasattr(hbos, "decision_scores_")
        assert hbos.n_features == X.shape[1]

    def test_dynamic_bins(self):
        hbos = HBOS(dynamic_bins=True)
        X = np.array([[1, 0.1], [2, 0.2], [3, 0.3], [4, 0.4], [5, 0.5]])
        hbos.fit(X)
        assert hbos.feature_distributions is not None
        assert len(hbos.feature_distributions) == X.shape[1]
        for distribution in hbos.feature_distributions:
            if isinstance(distribution, list):
                assert all(
                    isinstance(bin_edges, np.ndarray) for bin_edges in distribution
                )

    def test_decision_function(self):
        X = np.array([[1, 0.1], [2, 0.2], [3, 0.3], [4, 0.3]])
        scores = self.hbos.decision_function(X)
        expected_scores = -np.log(np.array([0.2, 0.3, 0.5, 1e-9]) + 1e-9) + -np.log(
            np.array([6.25, 6.25, 12.5, 12.5]) + 1e-9
        )
        np.testing.assert_array_almost_equal(scores, expected_scores, decimal=3)

    def test_fit_rejects_invalid_bins_strategy(self):
        hbos = HBOS()
        X = np.array([[1, 0.1], [2, 0.2], [3, 0.3], [4, 0.4]])

        with pytest.raises(ValueError, match="Invalid binning strategy"):
            hbos.fit(X, nbins="invalid-strategy")

    def test_decision_function_rejects_mismatched_dataframe_columns(self):
        hbos = HBOS()
        X = pd.DataFrame({"a": [1, 2, 3], "b": [0.1, 0.2, 0.3]})
        hbos.fit(X)

        with pytest.raises(ValueError, match="columns"):
            hbos.decision_function(pd.DataFrame({"x": [1, 2, 3], "y": [0.1, 0.2, 0.3]}))

    def test_refit_replaces_learned_distributions(self):
        hbos = HBOS().fit(np.array([[0.0], [1.0], [2.0]]), nbins=2)
        first_distribution = hbos.feature_distributions[0]
        hbos.fit(np.array([[10.0], [11.0], [12.0]]), nbins=3)
        assert len(hbos.feature_distributions) == 1
        assert hbos.feature_distributions[0] is not first_distribution
        assert len(hbos.feature_distributions[0][0]) == 3

    def test_auto_bins_are_computed_per_feature(self):
        X = np.column_stack((np.ones(100), np.arange(100, dtype=float)))
        hbos = HBOS().fit(X, nbins="auto")
        assert len(hbos.feature_distributions[0][0]) == 1
        assert len(hbos.feature_distributions[1][0]) > 1

    def test_values_outside_fitted_support_receive_higher_score(self):
        hbos = HBOS().fit(np.array([[0.0], [0.5], [1.0]]), nbins=2)
        scores = hbos.decision_function(np.array([[0.5], [100.0]]))
        assert scores[1] > scores[0]

    @pytest.mark.parametrize("quantile", [-0.1, 1.1, np.nan])
    def test_predict_rejects_invalid_quantile(self, quantile):
        hbos = HBOS().fit(np.array([[0.0], [1.0], [2.0]]))
        with pytest.raises(ValueError, match="quantile"):
            hbos.predict(np.array([[1.0]]), quantile=quantile)
