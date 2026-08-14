# Copyright (c) 2024-2026 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License


from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest

import tinyshift.utils.imports as imports_utils
from tinyshift.modelling.dmstl import DMSTLWrapper
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


class TestDMSTLWrapper:
    def test_process_components_and_model_columns(self, monkeypatch):
        monkeypatch.setattr(imports_utils, "check_extra", lambda extra_name: None)

        wrapper = DMSTLWrapper(
            mf_resid=SimpleNamespace(freq="D"),
            season_length=[7],
            trend_model=None,
            seasonal_model=None,
            log_transform=False,
        )
        wrapper.id_col_ = "unique_id"
        wrapper.time_col_ = "ds"

        components_df = pd.DataFrame(
            {
                "trend": [1.0, 2.0, 3.0, 4.0],
                "seasonal_7": [0.1, 0.2, 0.1, 0.2],
                "resid": [0.0, 0.0, 0.0, 0.0],
            }
        )

        trend, seasonal, residual = wrapper._process_components(components_df)
        cols = wrapper._get_model_cols(
            pd.DataFrame(
                {
                    "unique_id": [1],
                    "ds": [pd.Timestamp("2024-01-01")],
                    "y": [10.0],
                    "model_a": [11.0],
                }
            )
        )

        assert trend.shape == (4,)
        assert seasonal.shape == (4,)
        assert residual.shape == (4,)
        assert cols == ["y", "model_a"]

    def test_seasonal_config_resolves_per_uid_season_length(self, monkeypatch):
        monkeypatch.setattr(imports_utils, "check_extra", lambda extra_name: None)

        wrapper = DMSTLWrapper(
            mf_resid=SimpleNamespace(freq="D"),
            season_length={"sku-a": [7, 30]},
        )

        class SeasonalNaive:
            def __init__(self, season_length):
                self.season_length = season_length

        season_lengths, seasonal_models = wrapper._get_seasonal_config(
            "sku-a", SeasonalNaive
        )

        assert season_lengths == [7, 30]
        assert [model.season_length for model in seasonal_models] == [7, 30]
