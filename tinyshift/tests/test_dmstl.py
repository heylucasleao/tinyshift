# Copyright (c) 2024-2026 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License

from functools import partial
import numpy as np
import pandas as pd
import pytest

import tinyshift.utils.imports as imports_utils
from tinyshift.modelling.dmstl import DMSTLWrapper


class TestDMSTLWrapper:
    def test_process_components_and_model_columns(self, monkeypatch):
        monkeypatch.setattr(imports_utils, "check_extra", lambda extra_name: None)

        wrapper = DMSTLWrapper(
            season_length=[7],
            log_transform=False,
        )
        wrapper.id_col_ = "unique_id"
        wrapper.time_col_ = "ds"

        components_df = pd.DataFrame(
            {
                "trend": [1.0, 2.0, 3.0, 4.0],
                "seasonal_7": [0.1, 0.2, 0.1, 0.2],
                "seasonal_30": [1.0, 2.0, 1.0, 2.0],
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

        _, seasonal_parts, _ = wrapper._process_components(
            components_df, split_seasonal=True
        )
        assert len(seasonal_parts) == 2
        np.testing.assert_array_equal(seasonal_parts[0], [0.1, 0.2, 0.1, 0.2])
        np.testing.assert_array_equal(seasonal_parts[1], [1.0, 2.0, 1.0, 2.0])

    def test_seasonal_config_resolves_per_uid_season_length(self, monkeypatch):
        monkeypatch.setattr(imports_utils, "check_extra", lambda extra_name: None)

        wrapper = DMSTLWrapper(
            season_length={"sku-a": [7, 30]},
        )

        class SeasonalNaive:
            def __init__(self, season_length):
                self.season_length = season_length

        season_lengths, seasonal_models = wrapper._get_seasonal_config(
            "sku-a", np.arange(32, dtype=float), SeasonalNaive
        )

        assert season_lengths == [7, 30]
        assert [model.season_length for model in seasonal_models] == [7, 30]

    def test_seasonal_config_uses_detected_periods_for_auto(self, monkeypatch):
        monkeypatch.setattr(imports_utils, "check_extra", lambda extra_name: None)
        monkeypatch.setattr(
            "tinyshift.modelling.dmstl.detect_seasonal_periods",
            lambda series, **kwargs: [8, 16],
        )

        wrapper = DMSTLWrapper(
            season_length="auto",
        )

        class SeasonalNaive:
            def __init__(self, season_length):
                self.season_length = season_length

        season_lengths, seasonal_models = wrapper._get_seasonal_config(
            "sku-a", np.arange(32, dtype=float), SeasonalNaive
        )

        assert season_lengths == [8, 16]
        assert [model.season_length for model in seasonal_models] == [8, 16]

    def test_seasonal_config_accepts_model_factory(self, monkeypatch):
        monkeypatch.setattr(imports_utils, "check_extra", lambda extra_name: None)

        from statsforecast.models import AutoETS

        wrapper = DMSTLWrapper(
            season_length=[7, 30],
            seasonal_model_callable=lambda period: AutoETS(
                season_length=period,
                model="ZNM",
                alias=f"AutoETS-{period}",
            ),
        )

        season_lengths, seasonal_models = wrapper._get_seasonal_config(
            "sku-a", np.arange(32, dtype=float), AutoETS
        )

        assert season_lengths == [7, 30]
        assert [model.season_length for model in seasonal_models] == [7, 30]
        assert [model.model for model in seasonal_models] == ["ZNM", "ZNM"]

    def test_seasonal_config_rejects_model_instance(self, monkeypatch):
        monkeypatch.setattr(imports_utils, "check_extra", lambda extra_name: None)

        from statsforecast.models import AutoETS

        wrapper = DMSTLWrapper(
            season_length=[7, 30],
            freq="D",
            seasonal_model_callable=AutoETS(model="ZNA"),
        )

        with pytest.raises(TypeError, match="seasonal_model_callable"):
            wrapper._get_seasonal_config("sku-a", np.arange(32, dtype=float), AutoETS)

    def test_trend_model_callable_creates_model_per_uid(self, monkeypatch):
        monkeypatch.setattr(imports_utils, "check_extra", lambda extra_name: None)

        from statsforecast.models import AutoETS

        wrapper = DMSTLWrapper(
            season_length=7,
            trend_model_callable=lambda: AutoETS(model="ZZN"),
        )

        assert wrapper.trend_model_callable().model == "ZZN"

        default_callable = partial(AutoETS, model="ZZN")
        resolved_model = wrapper._get_trend_config("sku-a", default_callable)
        assert resolved_model.model == "ZZN"
        assert resolved_model is not default_callable()

    def test_trend_model_callable_rejects_model_instance(self, monkeypatch):
        monkeypatch.setattr(imports_utils, "check_extra", lambda extra_name: None)

        from statsforecast.models import AutoETS

        wrapper = DMSTLWrapper(
            season_length=7,
            freq="D",
            trend_model_callable=AutoETS(model="ZZN"),
        )

        with pytest.raises(TypeError, match="trend_model_callable"):
            wrapper.fit(pd.DataFrame({"unique_id": ["sku-a"], "ds": [1], "y": [1.0]}))

    def test_residual_lags_use_nlags_configuration(self, monkeypatch):
        monkeypatch.setattr(imports_utils, "check_extra", lambda extra_name: None)
        select_calls = []

        def fake_select_pami_lag(values, **kwargs):
            select_calls.append((values, kwargs))
            return 3, 0.2, np.array([0.5, 0.3, 0.2])

        monkeypatch.setattr(
            "tinyshift.modelling.dmstl.select_pami_lag", fake_select_pami_lag
        )

        wrapper = DMSTLWrapper(
            nlags={"sku-a": "auto", "sku-b": 2},
            pami_params={"max_tau": 10},
        )

        assert wrapper._get_residual_lags("sku-b", np.arange(8)) == [1, 2]
        assert wrapper._get_residual_lags("sku-a", np.arange(8)) == [3]
        assert len(select_calls) == 1
        assert select_calls[0][1] == {"max_tau": 10}
