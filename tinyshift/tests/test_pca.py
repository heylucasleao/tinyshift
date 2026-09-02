# Copyright (c) 2024-2026 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License


import numpy as np
import pytest

from tinyshift.outlier.pca import PCAReconstructionError


class TestPCAReconstructionError:
    def setup_method(self):
        self.model = PCAReconstructionError()

    def test_fit_sets_attributes(self):
        X = np.array([[1, 2], [3, 4], [5, 6]])
        self.model.fit(X)
        assert hasattr(self.model, "pca_")
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

    def test_n_components_is_an_estimator_parameter(self):
        model = PCAReconstructionError(n_components=1)
        assert model.get_params()["n_components"] == 1
        model.fit(np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]))
        assert model.pca_.n_components == 1

    def test_univariate_default_is_rejected(self):
        with pytest.raises(ValueError, match="at least two features"):
            self.model.fit(np.array([[0.0], [1.0], [2.0]]))

    @pytest.mark.parametrize("n_components", [0, 2, -1, 1.5, True])
    def test_invalid_n_components_is_rejected(self, n_components):
        model = PCAReconstructionError(n_components=n_components)
        with pytest.raises(ValueError, match="n_components"):
            model.fit(np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]))

    @pytest.mark.parametrize("quantile", [-0.1, 1.1, np.nan])
    def test_predict_rejects_invalid_quantile(self, quantile):
        self.model.fit(np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]))
        with pytest.raises(ValueError, match="quantile"):
            self.model.predict(np.array([[1.0, 1.0]]), quantile=quantile)
