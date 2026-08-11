# Copyright (c) 2024-2026 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License


import numpy as np
from tinyshift.outlier.pca import PCAReconstructionError
import pytest


class TestPCAReconstructionError:
    def setup_method(self):
        self.model = PCAReconstructionError()

    def test_fit_sets_attributes(self):
        X = np.array([[1, 2], [3, 4], [5, 6]])
        self.model.fit(X)
        assert hasattr(self.model, "PCA")
        assert hasattr(self.model, "decision_scores_")

    def test_score_output(self):
        X = np.array([[1, 2], [3, 4], [5, 6]])
        self.model.fit(X)
        scores = self.model.decision_function(X)
        assert scores.shape[0] == X.shape[0]

    def test_decision_function_output(self):
        X = np.array([[1, 2], [3, 4], [5, 6]])
        self.model.fit(X)
        assert self.model.predict(X).shape[0] == X.shape[0]

    def test_fit_not_fitted(self):
        X = np.array([[1, 2], [3, 4], [5, 6]])

        with pytest.raises(ValueError):
            self.model.predict(X)

    def test_decision_function_requires_fit(self):
        X = np.array([[1, 2], [3, 4], [5, 6]])

        with pytest.raises(ValueError, match="fitted"):
            self.model.decision_function(X)
