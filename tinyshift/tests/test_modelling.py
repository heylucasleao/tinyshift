# Copyright (c) 2024-2026 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License


import numpy as np
import pandas as pd
import pytest

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


class TestResidualizer:
    def test_feature_residualizer_fit_transform(self):
        X = np.column_stack(
            [
                np.linspace(0, 10, 20),
                np.linspace(0, 10, 20)
                + np.random.RandomState(1).normal(scale=0.1, size=20),
            ]
        )

        transformer = FeatureResidualizer()
        Xt = transformer.fit_transform(X, corrcoef=0.6)

        assert Xt.shape == X.shape
        assert np.isfinite(Xt).all()


class TestScaler:
    def test_robust_gaussian_scaler_fit_transform(self):
        X = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [100.0, 101.0]])

        scaler = RobustGaussianScaler()
        Xt = scaler.fit_transform(X)

        assert Xt.shape == X.shape
        assert np.isfinite(Xt).all()


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
