# Copyright (c) 2024-2026 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License

import numpy as np
import pandas as pd
import pytest

import tinyshift.utils.imports as imports_utils
from tinyshift.modelling.dtl import DTLWrapper


class TestDTLWrapper:
    def test_initialization_preserves_sklearn_parameters(self, monkeypatch):
        monkeypatch.setattr(imports_utils, "check_extra", lambda extra_name: None)

        wrapper = DTLWrapper(
            residual_model_callable=lambda nlags, freq: None,
            freq="MS",
            trend_frac=0.3,
            robust=False,
            log_transform=True,
        )

        params = wrapper.get_params()
        assert params["freq"] == "MS"
        assert params["residual_model_callable"] is not None
        assert params["trend_frac"] == 0.3
        assert params["robust"] is False
        assert params["log_transform"] is True

    def test_trend_model_callable_creates_model_per_uid(self, monkeypatch):
        monkeypatch.setattr(imports_utils, "check_extra", lambda extra_name: None)

        created_models = []

        def trend_factory():
            model = object()
            created_models.append(model)
            return model

        wrapper = DTLWrapper(
            residual_model_callable=lambda nlags, freq: None,
            freq="MS",
            trend_model_callable=trend_factory,
        )

        model = wrapper._get_trend_config("series-a", lambda: object())

        assert model is created_models[0]
        assert len(created_models) == 1

    def test_trend_model_callable_rejects_model_instance(self, monkeypatch):
        monkeypatch.setattr(imports_utils, "check_extra", lambda extra_name: None)

        wrapper = DTLWrapper(
            residual_model_callable=lambda nlags, freq: None,
            freq="MS",
            trend_model_callable=object(),
        )

        with pytest.raises(TypeError, match="trend_model_callable"):
            wrapper._get_trend_config("series-a", lambda: object())

    def test_fit_stores_trend_and_residual_models_together(self, monkeypatch):
        monkeypatch.setattr(imports_utils, "check_extra", lambda extra_name: None)

        wrapper = DTLWrapper(
            residual_model_callable=lambda nlags, freq: None,
            freq="MS",
            trend_frac=0.5,
        )
        wrapper._fit_statsforecast = lambda models, values, dates, uid, freq: {
            "component": "trend",
            "uid": uid,
            "values": values,
        }
        captured = {}

        def fit_residual(group, residual, prediction_intervals, static_features):
            captured["static_features"] = static_features
            return {
                "component": "residual",
                "uid": group["unique_id"].iloc[0],
                "values": residual,
            }

        wrapper._fit_mlforecast = fit_residual

        frame = pd.DataFrame(
            {
                "unique_id": ["series-a"] * 8,
                "ds": pd.date_range("2024-01-01", periods=8, freq="MS"),
                "y": [1.0, 2.1, 2.8, 4.2, 5.1, 6.2, 7.0, 8.1],
            }
        )

        fitted = wrapper.fit(frame, static_features=["store_id"])

        assert fitted is wrapper
        assert set(wrapper.fitted_models_) == {"series-a"}
        assert wrapper.fitted_models_["series-a"]["trend"]["component"] == "trend"
        assert wrapper.fitted_models_["series-a"]["residual"]["component"] == "residual"
        assert captured["static_features"] == ["store_id"]
        assert not hasattr(wrapper, "trend_models_")

    def test_model_columns_exclude_id_and_time(self, monkeypatch):
        monkeypatch.setattr(imports_utils, "check_extra", lambda extra_name: None)

        wrapper = DTLWrapper(
            residual_model_callable=lambda nlags, freq: None,
            freq="MS",
        )
        wrapper.id_col_ = "unique_id"
        wrapper.time_col_ = "ds"

        predictions = pd.DataFrame(
            {
                "unique_id": ["series-a"],
                "ds": pd.date_range("2024-01-01", periods=1, freq="MS"),
                "LinearRegression": [1.0],
            }
        )

        assert wrapper._get_model_cols(predictions) == ["LinearRegression"]

    def test_residual_lags_support_manual_and_auto_configuration(self, monkeypatch):
        monkeypatch.setattr(imports_utils, "check_extra", lambda extra_name: None)
        calls = []

        def fake_select(values, **kwargs):
            calls.append((values, kwargs))
            return 4, 0.2, np.array([0.5, 0.4, 0.3, 0.2])

        monkeypatch.setattr("tinyshift.series.select_pami_lag", fake_select)

        wrapper = DTLWrapper(
            residual_model_callable=lambda nlags, freq: None,
            freq="MS",
            nlags={"series-a": "auto", "series-b": 2},
            pami_params={"max_tau": 12, "m": 3},
        )

        assert wrapper._get_residual_lags("series-b", np.arange(10)) == [1, 2]
        assert wrapper._get_residual_lags("series-a", np.arange(10)) == [4]
        assert calls[0][1] == {"max_tau": 12, "m": 3, "return_mode": "value_only"}

    def test_residual_factory_receives_lags_and_frequency(self, monkeypatch):
        monkeypatch.setattr(imports_utils, "check_extra", lambda extra_name: None)
        received = {}

        fit_arguments = {}

        class ResidualModel:
            def fit(self, *args, **kwargs):
                fit_arguments.update(kwargs)
                return self

        def residual_factory(nlags, freq):
            received.update(nlags=nlags, freq=freq)
            return ResidualModel()

        wrapper = DTLWrapper(
            residual_model_callable=residual_factory,
            freq="MS",
            nlags=3,
        )
        wrapper.freq_ = wrapper.freq
        wrapper.id_col_ = "unique_id"
        wrapper.time_col_ = "ds"
        wrapper.target_col_ = "y"
        wrapper.exog_cols_ = []

        group = pd.DataFrame(
            {
                "unique_id": ["series-a"] * 4,
                "ds": pd.date_range("2024-01-01", periods=4, freq="MS"),
                "y": [1.0, 2.0, 3.0, 4.0],
            }
        )

        wrapper._fit_mlforecast(group, np.zeros(4), static_features=["store_id"])

        assert received == {"nlags": [1, 2, 3], "freq": "MS"}
        assert fit_arguments["static_features"] == ["store_id"]
