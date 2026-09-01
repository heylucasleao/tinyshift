from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from tinyshift.forecasting import DTLWrapper
from tinyshift.forecasting.dtl.global_ import DTLGlobalWrapper
from tinyshift.forecasting.dtl.local_ import DTLLocalWrapper


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
        captured_rows = []

        def fake_detrend(frame, **kwargs):
            return pd.DataFrame(
                {
                    "trend": frame["y"].to_numpy(),
                    "detrended": [0.0] * len(frame),
                },
                index=frame.index,
            )

        monkeypatch.setattr("tinyshift.forecasting.dtl.base.detrend", fake_detrend)
        wrapper = DTLLocalWrapper(
            residual_model_callable=lambda nlags, freq: None,
            freq="MS",
            trend_model_callable=lambda: object(),
            nlags=1,
        )
        wrapper._fit_panel = lambda models, rows: captured_rows.append(rows)
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

        [(uid, _, dates)] = captured_rows[0]
        assert uid == "series-a"
        assert list(dates) == list(
            pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"])
        )

    def test_fit_batches_skus_sharing_default_trend_factory(self):
        from mlforecast import MLForecast
        from sklearn.linear_model import LinearRegression

        def residual_model_callable(nlags, freq):
            return MLForecast(models=[LinearRegression()], lags=nlags, freq=freq)

        dates = pd.date_range("2024-01-01", periods=20, freq="MS")
        frame = pd.concat(
            [
                pd.DataFrame(
                    {
                        "unique_id": uid,
                        "ds": dates,
                        "y": np.arange(20, dtype=float) * 0.1 + offset,
                    }
                )
                for uid, offset in [("a", 0.0), ("b", 5.0)]
            ],
            ignore_index=True,
        )

        wrapper = DTLLocalWrapper(
            residual_model_callable=residual_model_callable,
            freq="MS",
            nlags=1,
        )
        wrapper.fit(frame)

        assert (
            wrapper.fitted_models_["a"]["trend"] is wrapper.fitted_models_["b"]["trend"]
        )

        predictions = wrapper.predict(h=3)
        assert set(predictions["unique_id"]) == {"a", "b"}
