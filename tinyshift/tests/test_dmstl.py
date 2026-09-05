from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from tinyshift.forecasting import DMSTLWrapper
from tinyshift.forecasting.dmstl.global_ import DMSTLGlobalWrapper
from tinyshift.forecasting.dmstl.local_ import DMSTLLocalWrapper


class TestDMSTLWrapper:
    def test_mode_selects_local_or_global_strategy(self):
        assert isinstance(
            DMSTLWrapper(mode="local")._make_delegate(), DMSTLLocalWrapper
        )
        assert isinstance(
            DMSTLWrapper(mode="global")._make_delegate(), DMSTLGlobalWrapper
        )

    def test_invalid_mode_is_rejected(self):
        with pytest.raises(ValueError, match="mode"):
            DMSTLWrapper(mode="invalid")._make_delegate()

    def test_global_requires_one_residual_factory(self):
        wrapper = DMSTLGlobalWrapper(freq="D")

        with pytest.raises(ValueError, match="one callable"):
            wrapper._fit_residuals([], None, None)

    def test_local_uses_one_residual_model_per_uid(self):
        models = []

        def factory(nlags, freq):
            model = Mock()
            model.fit.return_value = model
            models.append((nlags, freq, model))
            return model

        wrapper = DMSTLLocalWrapper(
            residual_model_callable=factory,
            freq="D",
        )
        wrapper.id_col_ = "unique_id"
        wrapper.time_col_ = "ds"
        wrapper.target_col_ = "y"
        wrapper.freq_ = "D"
        wrapper.fitted_models_ = {"a": {}, "b": {}}

        frame = pd.DataFrame(
            {
                "unique_id": ["a", "a"],
                "ds": pd.date_range("2024-01-01", periods=2),
                "y": [0.1, 0.2],
            }
        )
        wrapper._fit_residuals(
            [("a", frame, [1]), ("b", frame.assign(unique_id="b"), [1, 2])],
            None,
            None,
        )

        assert [item[0] for item in models] == [[1], [1, 2]]
        assert all("residual" in wrapper.fitted_models_[uid] for uid in ("a", "b"))

    def test_global_uses_union_of_residual_lags_and_one_model(self):
        factory_calls = []

        class ResidualModel:
            def fit(self, frame, **kwargs):
                self.frame = frame
                return self

        def factory(nlags, freq):
            factory_calls.append((nlags, freq))
            return ResidualModel()

        wrapper = DMSTLGlobalWrapper(
            residual_model_callable=factory,
            freq="D",
        )
        wrapper.id_col_ = "unique_id"
        wrapper.time_col_ = "ds"
        wrapper.target_col_ = "y"
        wrapper.freq_ = "D"

        frame = pd.DataFrame(
            {
                "unique_id": ["a", "a"],
                "ds": pd.date_range("2024-01-01", periods=2),
                "y": [0.1, 0.2],
            }
        )
        wrapper._fit_residuals(
            [("a", frame, [1, 3]), ("b", frame.assign(unique_id="b"), [2, 3])],
            None,
            None,
        )

        assert factory_calls == [([1, 2, 3], "D")]
        assert len(wrapper.residual_mlforecast_.frame) == 4

    def test_fit_batches_skus_sharing_default_trend_and_seasonal_factories(self):
        from sklearn.linear_model import LinearRegression
        from mlforecast import MLForecast

        def residual_model_callable(nlags, freq):
            return MLForecast(models=[LinearRegression()], lags=nlags, freq=freq)

        dates = pd.date_range("2024-01-01", periods=28, freq="D")
        pattern = np.tile(np.arange(7, dtype=float), 4)
        frame = pd.concat(
            [
                pd.DataFrame(
                    {
                        "unique_id": uid,
                        "ds": dates,
                        "y": np.arange(28, dtype=float) * 0.1 + pattern + offset,
                    }
                )
                for uid, offset in [("a", 0.0), ("b", 5.0)]
            ],
            ignore_index=True,
        )

        wrapper = DMSTLLocalWrapper(
            residual_model_callable=residual_model_callable,
            freq="D",
            season_length=[7],
            nlags=1,
        )
        wrapper.fit(frame)

        assert (
            wrapper.fitted_models_["a"]["trend"] is wrapper.fitted_models_["b"]["trend"]
        )
        assert (
            wrapper.fitted_models_["a"]["seasonal"][0]
            is wrapper.fitted_models_["b"]["seasonal"][0]
        )

        predictions = wrapper.predict(h=3)
        assert set(predictions["unique_id"]) == {"a", "b"}
        assert (predictions["unique_id"] == "a").sum() == 3
        assert (predictions["unique_id"] == "b").sum() == 3

    def test_stabilization_rejects_unknown_method(self):
        wrapper = DMSTLLocalWrapper()

        with pytest.raises(ValueError, match="stabilization_method"):
            wrapper._stabilize(np.array([1.0]), "unknown", 0.5)

    def test_series_length_validation_raises_error_when_too_short(self):
        wrapper = DMSTLLocalWrapper(season_length=7)
        short_series = np.array(
            [112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118]
        )

        expected_msg = (
            "Series for unique_id 'sku_101' has length 12, which is too short "
            "for seasonal period 7 \(MSTL requires at least 14 observations\)\. "
            "Adjust your train window / step_size or set a smaller period for this SKU\."
        )

        with pytest.raises(ValueError, match=expected_msg):
            wrapper._resolve_seasonal_periods("sku_101", short_series)

    def test_multi_seasonality_series_length_validation_raises_error(self):
        wrapper = DMSTLLocalWrapper(season_length=[7, 12])
        # len(series) = 24 is less than 2 * sum([7, 12]) = 38
        series_24 = np.array(
            [
                112,
                118,
                132,
                129,
                121,
                135,
                148,
                148,
                136,
                119,
                104,
                118,
                115,
                126,
                141,
                135,
                125,
                149,
                170,
                170,
                158,
                133,
                114,
                140,
            ]
        )

        expected_msg = (
            "Series for unique_id 'sku_202' has length 24, which is too short "
            "for seasonal period 12 \(MSTL requires at least 24 observations\)\. "
            "Adjust your train window / step_size or set a smaller period for this SKU\."
        )

        with pytest.raises(ValueError, match=expected_msg):
            wrapper._resolve_seasonal_periods("sku_202", series_24)

    def test_seasonal_periods_format_validation(self):
        wrapper = DMSTLLocalWrapper(season_length=[1, 7])
        series = np.arange(30)

        with pytest.raises(
            ValueError, match="must contain integer periods greater than one"
        ):
            wrapper._resolve_seasonal_periods("sku_303", series)

    @patch("tinyshift.forecasting.dmstl.base.SeasonalPeriodDetector")
    def test_auto_detection_failure_raises_value_error(self, mock_detect):
        mock_detect.return_value.detect.return_value = []
        wrapper = DMSTLLocalWrapper(season_length="auto")
        series = np.array([10.0] * 12)

        with pytest.raises(
            ValueError, match="Could not automatically detect seasonal periods"
        ):
            wrapper._resolve_seasonal_periods("sku_404", series)
