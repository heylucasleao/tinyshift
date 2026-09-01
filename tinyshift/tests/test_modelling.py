# Copyright (c) 2024-2026 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License


import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
from tinyshift.modelling.multicollinearity import filter_features_by_vif
from tinyshift.modelling.residualizer import FeatureResidualizer
from tinyshift.modelling.scaler import RobustGaussianScaler
from tinyshift.modelling.ts_features import (
    estimate_history_length,
    fourier_seasonality,
    relative_strength_index,
    standardize_returns,
)


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


class TestTimeSeriesFeatures:
    def test_relative_strength_index_returns_expected_shape(self):
        x = np.array([10.0, 11.0, 10.5, 12.0, 11.5, 13.0])

        result = relative_strength_index(x, rolling_window=2)

        assert result.shape == x.shape
        assert np.isfinite(result).all()

    def test_standardize_returns_inserts_nan_at_start(self):
        x = np.array([1.0, 2.0, 4.0, 8.0])

        result = standardize_returns(x)

        assert result.shape == (4,)
        assert np.isnan(result[0])
        assert np.isfinite(result[1:]).all()

    def test_standardize_returns_can_skip_standardization(self):
        x = np.array([1.0, 2.0, 4.0, 8.0])

        result = standardize_returns(x, standardize=False)

        assert result.shape == (4,)
        assert np.isnan(result[0])
        np.testing.assert_allclose(
            result[1:],
            np.log(np.array([2.0, 4.0, 8.0]) / np.array([1.0, 2.0, 4.0])),
        )

    def test_relative_strength_index_rejects_multidimensional_input(self):
        with pytest.raises(ValueError, match="1-dimensional"):
            relative_strength_index(np.array([[1.0, 2.0], [3.0, 4.0]]))

    def test_fourier_seasonality_adds_sine_and_cosine_features(self):
        df = pd.DataFrame({"ds": pd.date_range("2024-01-01", periods=7, freq="D")})

        output = fourier_seasonality(df, time_col="ds", seasonality=["weekly", "daily"])

        assert {"weekly_sin", "weekly_cos", "daily_sin", "daily_cos"}.issubset(
            output.columns
        )

    def test_estimate_history_length_uses_rule_of_thumb(self):
        assert estimate_history_length(seasonal_period=7, horizon=14) == 17

    def test_fourier_seasonality_raises_for_unknown_seasonality(self):
        df = pd.DataFrame({"ds": pd.date_range("2024-01-01", periods=3, freq="D")})

        with pytest.raises(ValueError, match="Unknown seasonality"):
            fourier_seasonality(df, time_col="ds", seasonality=["weekly", "unknown"])
