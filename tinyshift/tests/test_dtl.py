# Copyright (c) 2024-2026 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import tinyshift.utils.imports as imports_utils
from tinyshift.modelling.dtl import DTLWrapper


class TestDTLWrapper:
    def test_initialization_preserves_sklearn_parameters(self, monkeypatch):
        monkeypatch.setattr(imports_utils, "check_extra", lambda extra_name: None)

        wrapper = DTLWrapper(
            mf_resid=SimpleNamespace(freq="MS"),
            trend_frac=0.3,
            robust=False,
            log_transform=True,
        )

        params = wrapper.get_params()
        assert params["mf_resid"].freq == "MS"
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
            mf_resid=SimpleNamespace(freq="MS"),
            trend_model_callable=trend_factory,
        )

        model = wrapper._get_trend_config("series-a", lambda: object())

        assert model is created_models[0]
        assert len(created_models) == 1

    def test_trend_model_callable_rejects_model_instance(self, monkeypatch):
        monkeypatch.setattr(imports_utils, "check_extra", lambda extra_name: None)

        wrapper = DTLWrapper(
            mf_resid=SimpleNamespace(freq="MS"),
            trend_model_callable=object(),
        )

        with pytest.raises(TypeError, match="trend_model_callable"):
            wrapper._get_trend_config("series-a", lambda: object())

    def test_fit_stores_trend_and_residual_models_together(self, monkeypatch):
        monkeypatch.setattr(imports_utils, "check_extra", lambda extra_name: None)

        wrapper = DTLWrapper(mf_resid=SimpleNamespace(freq="MS"), trend_frac=0.5)
        wrapper._fit_statsforecast = lambda models, values, dates, uid, freq: {
            "component": "trend",
            "uid": uid,
            "values": values,
        }
        wrapper._fit_mlforecast = lambda group, residual, prediction_intervals: {
            "component": "residual",
            "uid": group["unique_id"].iloc[0],
            "values": residual,
        }

        frame = pd.DataFrame(
            {
                "unique_id": ["series-a"] * 8,
                "ds": pd.date_range("2024-01-01", periods=8, freq="MS"),
                "y": [1.0, 2.1, 2.8, 4.2, 5.1, 6.2, 7.0, 8.1],
            }
        )

        fitted = wrapper.fit(frame)

        assert fitted is wrapper
        assert set(wrapper.fitted_models_) == {"series-a"}
        assert wrapper.fitted_models_["series-a"]["trend"]["component"] == "trend"
        assert wrapper.fitted_models_["series-a"]["residual"]["component"] == "residual"
        assert not hasattr(wrapper, "trend_models_")

    def test_model_columns_exclude_id_and_time(self, monkeypatch):
        monkeypatch.setattr(imports_utils, "check_extra", lambda extra_name: None)

        wrapper = DTLWrapper(mf_resid=SimpleNamespace(freq="MS"))
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
