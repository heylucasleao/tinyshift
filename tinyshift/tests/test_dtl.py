from unittest.mock import Mock

import pandas as pd
import pytest

from tinyshift.modelling import DTLWrapper
from tinyshift.modelling.dtl.global_ import DTLGlobalWrapper
from tinyshift.modelling.dtl.local_ import DTLLocalWrapper


class TestDTLWrapper:
    def test_initialization_preserves_sklearn_parameters(self):
        wrapper = DTLWrapper(
            residual_model_callable=lambda nlags, freq: None,
            freq="MS",
            trend_frac=0.3,
            robust=False,
            log_transform=True,
        )

        params = wrapper.get_params()
        assert params["mode"] == "global"
        assert params["freq"] == "MS"
        assert params["trend_frac"] == 0.3
        assert params["robust"] is False
        assert params["log_transform"] is True

    def test_mode_selects_local_or_global_strategy(self):
        assert isinstance(DTLWrapper(mode="local")._make_delegate(), DTLLocalWrapper)
        assert isinstance(DTLWrapper(mode="global")._make_delegate(), DTLGlobalWrapper)

    def test_invalid_mode_is_rejected(self):
        with pytest.raises(ValueError, match="mode"):
            DTLWrapper(mode="invalid")._make_delegate()

    def test_local_uses_one_residual_model_per_uid(self):
        calls = []

        def factory(nlags, freq):
            model = Mock()
            model.fit.return_value = model
            calls.append((nlags, freq))
            return model

        wrapper = DTLLocalWrapper(residual_model_callable=factory, freq="MS")
        wrapper.id_col_ = "unique_id"
        wrapper.time_col_ = "ds"
        wrapper.target_col_ = "y"
        wrapper.freq_ = "MS"
        wrapper.fitted_models_ = {"a": {}, "b": {}}
        frame = pd.DataFrame(
            {
                "unique_id": ["a", "a"],
                "ds": pd.date_range("2024-01-01", periods=2, freq="MS"),
                "y": [0.1, 0.2],
            }
        )

        wrapper._fit_residuals(
            [("a", frame, [1]), ("b", frame.assign(unique_id="b"), [1, 2])],
            None,
            ["store_id"],
        )

        assert calls == [([1], "MS"), ([1, 2], "MS")]
        assert all("residual" in wrapper.fitted_models_[uid] for uid in ("a", "b"))

    def test_global_uses_union_of_residual_lags_and_one_model(self):
        calls = []

        class ResidualModel:
            def fit(self, frame, **kwargs):
                self.frame = frame
                return self

        def factory(nlags, freq):
            calls.append((nlags, freq))
            return ResidualModel()

        wrapper = DTLGlobalWrapper(residual_model_callable=factory, freq="MS")
        wrapper.id_col_ = "unique_id"
        wrapper.time_col_ = "ds"
        wrapper.target_col_ = "y"
        wrapper.freq_ = "MS"
        frame = pd.DataFrame(
            {
                "unique_id": ["a", "a"],
                "ds": pd.date_range("2024-01-01", periods=2, freq="MS"),
                "y": [0.1, 0.2],
            }
        )

        wrapper._fit_residuals(
            [("a", frame, [1, 3]), ("b", frame.assign(unique_id="b"), [2, 3])],
            None,
            None,
        )

        assert calls == [([1, 2, 3], "MS")]
        assert len(wrapper.residual_mlforecast_.frame) == 4

    def test_fit_sorts_each_series_before_modeling(self, monkeypatch):
        captured_dates = []

        def fake_detrend(frame, **kwargs):
            return pd.DataFrame(
                {
                    "trend": frame["y"].to_numpy(),
                    "detrended": [0.0] * len(frame),
                },
                index=frame.index,
            )

        monkeypatch.setattr("tinyshift.modelling.dtl.base.detrend", fake_detrend)
        wrapper = DTLLocalWrapper(
            residual_model_callable=lambda nlags, freq: None,
            freq="MS",
            trend_model_callable=lambda: object(),
            nlags=1,
        )
        wrapper._fit_statsforecast = (
            lambda model, values, dates, uid: captured_dates.append(list(dates))
        )
        wrapper._fit_residuals = (
            lambda residuals, prediction_intervals, static_features: None
        )

        frame = pd.DataFrame(
            {
                "unique_id": ["series-a"] * 3,
                "ds": pd.to_datetime(["2024-03-01", "2024-01-01", "2024-02-01"]),
                "y": [3.0, 1.0, 2.0],
            }
        )

        wrapper.fit(frame)

        assert captured_dates == [
            list(pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]))
        ]
