# Copyright (c) 2024-2026 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License


import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
from tinyshift.preprocessing.multicollinearity import filter_features_by_vif
from tinyshift.preprocessing.residualizer import FeatureResidualizer
from tinyshift.preprocessing.scaler import RobustGaussianScaler


class TestMulticollinearity:
    def test_filter_features_by_vif_removes_correlated_features(self):
        rng = np.random.RandomState(0)
        X = np.column_stack(
            [
                rng.normal(size=50),
                rng.normal(size=50) + 0.9 * np.arange(50),
                rng.normal(size=50),
            ]
        )

        mask = filter_features_by_vif(X, threshold=5.0, n_jobs=1)

        assert mask.dtype == bool
        assert mask.shape == (3,)
        assert mask.sum() >= 1

    @pytest.mark.parametrize("threshold", [np.nan, np.inf, -np.inf, 0.5, True])
    def test_filter_features_by_vif_rejects_invalid_threshold(self, threshold):
        with pytest.raises(ValueError, match="threshold"):
            filter_features_by_vif(np.arange(12).reshape(4, 3), threshold=threshold)

    @pytest.mark.parametrize("n_jobs", [0, True, 1.5])
    def test_filter_features_by_vif_rejects_invalid_n_jobs(self, n_jobs):
        error = ValueError if n_jobs == 0 else TypeError
        with pytest.raises(error, match="n_jobs"):
            filter_features_by_vif(np.arange(12).reshape(4, 3), n_jobs=n_jobs)

    def test_filter_features_by_vif_removes_constant_feature(self):
        X = np.column_stack((np.ones(20), np.arange(20), np.arange(20) ** 2))

        mask = filter_features_by_vif(X, threshold=5.0, n_jobs=1)

        assert not mask[0]


class TestResidualizer:
    def test_feature_residualizer_fit_transform(self):
        X = np.column_stack(
            [
                np.linspace(0, 10, 20),
                np.linspace(0, 10, 20)
                + np.random.RandomState(1).normal(scale=0.1, size=20),
            ]
        )

        transformer = FeatureResidualizer(corrcoef=0.6)
        Xt = transformer.fit_transform(X)

        assert Xt.shape == X.shape
        assert np.isfinite(Xt).all()
        assert abs(np.corrcoef(Xt, rowvar=False)[0, 1]) < 0.1

    def test_feature_residualizer_has_sklearn_fitted_state_and_parameters(self):
        transformer = FeatureResidualizer(corrcoef=0.7, corr_type="pos")

        with pytest.raises(NotFittedError):
            check_is_fitted(transformer)
        cloned = clone(transformer)
        assert cloned.get_params() == {"corr_type": "pos", "corrcoef": 0.7}

    @pytest.mark.parametrize("corrcoef", [0.0, -0.1, 1.1, np.nan, True, "bad"])
    def test_feature_residualizer_rejects_invalid_corrcoef(self, corrcoef):
        with pytest.raises(ValueError, match="corrcoef"):
            FeatureResidualizer(corrcoef=corrcoef).fit(np.arange(12).reshape(6, 2))

    def test_feature_residualizer_validates_dataframe_column_order(self):
        X = pd.DataFrame({"a": np.arange(10), "b": np.arange(10) + 0.1})
        transformer = FeatureResidualizer().fit(X)

        with pytest.raises(ValueError, match="columns"):
            transformer.transform(X[["b", "a"]])

    def test_feature_residualizer_fit_accepts_pipeline_y(self):
        X = np.column_stack((np.arange(10), np.arange(10) + 0.1))

        transformer = FeatureResidualizer().fit(X, y=np.arange(10))

        assert transformer.transform(X).shape == X.shape

    def test_feature_residualizer_refit_clears_dataframe_schema(self):
        X = pd.DataFrame({"a": np.arange(10), "b": np.arange(10) + 0.1})
        transformer = FeatureResidualizer().fit(X)

        transformer.fit(X.to_numpy())

        assert not hasattr(transformer, "feature_names_in_")


class TestScaler:
    def test_robust_gaussian_scaler_fit_transform(self):
        X = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [100.0, 101.0]])

        scaler = RobustGaussianScaler()
        Xt = scaler.fit_transform(X)

        assert Xt.shape == X.shape
        assert np.isfinite(Xt).all()

    def test_scaler_has_sklearn_fitted_state_and_parameters(self):
        scaler = RobustGaussianScaler(winsorize_method="mad", power_method="box-cox")

        with pytest.raises(NotFittedError):
            check_is_fitted(scaler)
        assert clone(scaler).get_params() == {
            "power_method": "box-cox",
            "winsorize_method": "mad",
        }

    def test_scaler_rejects_non_positive_box_cox_input(self):
        with pytest.raises(ValueError, match="strictly positive"):
            RobustGaussianScaler(power_method="box-cox").fit(
                np.array([[0.0], [1.0], [2.0]])
            )

    def test_scaler_validates_dataframe_column_order(self):
        X = pd.DataFrame({"a": np.arange(1, 6), "b": np.arange(6, 11)})
        scaler = RobustGaussianScaler().fit(X)

        with pytest.raises(ValueError, match="columns"):
            scaler.transform(X[["b", "a"]])

    def test_scaler_fit_accepts_pipeline_y_and_exposes_bounds(self):
        X = np.array([[1.0], [2.0], [3.0], [100.0]])
        scaler = RobustGaussianScaler(winsorize_method="iqr").fit(
            X, y=np.arange(len(X))
        )

        assert len(scaler.winsorization_bounds_) == 1
        assert scaler.transform(X).shape == X.shape

    def test_scaler_handles_constant_feature(self):
        result = RobustGaussianScaler().fit_transform(np.ones((5, 1)))

        np.testing.assert_allclose(result, 0.0)

    def test_scaler_refit_clears_dataframe_schema(self):
        X = pd.DataFrame({"a": np.arange(1, 6), "b": np.arange(6, 11)})
        scaler = RobustGaussianScaler().fit(X)

        scaler.fit(X.to_numpy())

        assert not hasattr(scaler, "feature_names_in_")
