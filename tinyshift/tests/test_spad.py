# Copyright (c) 2024-2025 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License


import numpy as np
from tinyshift.outlier.spad import SPAD
import pandas as pd
import pytest


class TestSPAD:
    def setup_method(self):
        self.spad = SPAD()

        self.spad.feature_dtypes = np.array([pd.CategoricalDtype(), np.float64])
        self.spad.feature_distributions = [
            {1: 0.2, 2: 0.3, 3: 0.5},
            [
                np.array([6.25, 0.0, 6.25, 0.0, 12.5]),
                np.array([0.1, 0.14, 0.18, 0.22, 0.26, 0.3]),
            ],
        ]
        self.spad.n_features = 2

    def test_compute_outlier_score_categorical(self):
        X = np.array([[1, 0], [2, 0], [3, 0], [4, 0]])
        scores = self.spad._compute_outlier_score(X, 0)
        expected_scores = -np.log(np.array([0.2, 0.3, 0.5, 1e-9]) + 1e-9)
        np.testing.assert_array_almost_equal(scores, expected_scores, decimal=3)

    def test_compute_outlier_score_continuous(self):
        X = np.array([[0, 0.1], [0, 0.2], [0, 0.3], [0, 0.3]])
        scores = self.spad._compute_outlier_score(X, 1)
        expected_scores = -np.log(np.array([6.25, 6.25, 12.5, 12.5]) + 1e-9)
        np.testing.assert_array_almost_equal(scores, expected_scores, decimal=3)

    def test_fit(self):
        X = np.array([[1, 0.1], [2, 0.2], [3, 0.3], [4, 0.4]])
        spad = SPAD()
        spad.fit(X)
        assert isinstance(spad, SPAD)
        assert spad.n_features == X.shape[1]
        assert hasattr(spad, "decision_scores_")

    def test_fit_with_plus(self):
        X = np.array([[1, 0.1], [2, 0.2], [3, 0.3], [4, 0.4]])
        spad_plus = SPAD(plus=True)
        model = spad_plus.fit(X)
        assert isinstance(model, SPAD)
        assert model.n_features == X.shape[1] * 2
        assert hasattr(model, "decision_scores_")
        assert model.pca_model is not None

    def test_decision_function(self):
        X = np.array([[1, 0.1], [2, 0.2], [3, 0.3], [4, 0.4]])
        scores = self.spad.decision_function(X)
        assert scores.shape[0] == X.shape[0]

    def test_decision_function_with_plus(self):
        X = np.array([[1, 0.1], [2, 0.2], [3, 0.3], [4, 0.4]])
        spad_plus = SPAD(plus=True)
        spad_plus.fit(X)
        scores = spad_plus.decision_function(X)
        assert scores.shape[0] == X.shape[0]
