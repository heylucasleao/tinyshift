# Copyright (c) 2024-2026 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License


import numpy as np
import pandas as pd
import pytest

from tinyshift.series.diagnostic import (
    detrend,
    detect_seasonal_periods,
    hurst_exponent,
    trend_significance,
    seasonal_significance,
    extract_mstl_components,
)
from tinyshift.series.forecastability import (
    foreca,
    adi_cv,
    sample_entropy,
    regularity_index,
    permutation_entropy,
    theoretical_limit,
    permutation_auto_mutual_information,
    select_pami_lag,
)
from tinyshift.series.interpolation import vi, hpi, hfi
from tinyshift.series.metric import (
    wape,
    pbias,
    score,
    rmae,
    fva_rmae,
    forecast_instability,
    economic_loss,
)
from tinyshift.series.outlier import hampel_filter, bollinger_bands
from tinyshift.series.stability import macv, mach, mascv, masch, rmsscv, rmssch
from statsmodels.tsa.seasonal import DecomposeResult


def test_economic_loss_aggregates_understock_and_overstock_by_id():
    df = pd.DataFrame(
        {
            "unique_id": ["a", "a", "b"],
            "y": [10.0, 5.0, 8.0],
            "model": [8.0, 7.0, 10.0],
            "cu": [3.0, 3.0, 2.0],
            "co": [1.0, 1.0, 4.0],
        }
    )

    result = economic_loss(
        df,
        models=["model"],
        cost_understock="cu",
        cost_overstock="co",
    )

    assert list(result.columns) == ["unique_id", "metric", "model"]
    assert result.set_index("unique_id").loc["a", "metric"] == "economic_loss"
    assert result.set_index("unique_id").loc["a", "model"] == pytest.approx(8.0)
    assert result.set_index("unique_id").loc["b", "model"] == pytest.approx(8.0)


def test_economic_loss_accepts_scalar_costs():
    df = pd.DataFrame(
        {
            "unique_id": ["a", "a"],
            "y": [10.0, 5.0],
            "model_a": [8.0, 7.0],
            "model_b": [12.0, 4.0],
        }
    )

    result = economic_loss(
        df,
        models=["model_a", "model_b"],
        cost_understock=3.0,
        cost_overstock=1.0,
    )

    assert result.loc[0, "model_a"] == pytest.approx(8.0)
    assert result.loc[0, "model_b"] == pytest.approx(5.0)


class TestDiagnostic:
    def test_detrend_returns_nixtla_columns_and_residual(self):
        frame = pd.DataFrame(
            {
                "unique_id": ["a"] * 20,
                "ds": pd.date_range("2024-01-01", periods=20, freq="D"),
                "y": np.linspace(1.0, 10.0, 20),
            }
        )

        result = detrend(frame, frac=0.3, robust=False)

        assert list(result.columns) == ["unique_id", "ds", "y", "trend", "detrended"]
        np.testing.assert_allclose(result["detrended"], result["y"] - result["trend"])

    def test_detrend_interpolates_missing_values_only_for_trend(self):
        series = pd.DataFrame(
            {
                "unique_id": ["a"] * 5,
                "ds": pd.date_range("2024-01-01", periods=5, freq="D"),
                "y": [1.0, np.nan, 3.0, 4.0, 5.0],
            }
        )

        result = detrend(series, frac=0.5)

        assert np.isnan(result.loc[1, "y"])
        assert np.isfinite(result.loc[1, "trend"])
        assert np.isnan(result.loc[1, "detrended"])

    def test_detrend_rejects_invalid_frac(self):
        with pytest.raises(ValueError, match="frac"):
            detrend(
                pd.DataFrame({"unique_id": ["a", "a"], "ds": [1, 2], "y": [1.0, 2.0]}),
                frac=0.0,
            )

    def test_detrend_rejects_non_dataframe_input(self):
        with pytest.raises(TypeError, match="DataFrame"):
            detrend([1.0, 2.0, 3.0])

    def test_detrend_accepts_nixtla_style_panel(self):
        frame = pd.DataFrame(
            {
                "unique_id": ["a", "a", "b", "b"],
                "ds": [2, 1, 1, 2],
                "y": [2.0, 1.0, 4.0, 5.0],
            }
        )

        result = detrend(frame, frac=1.0, robust=False)

        assert list(result.columns) == ["unique_id", "ds", "y", "trend", "detrended"]
        assert result[["unique_id", "ds", "y"]].equals(frame)
        assert result["trend"].notna().all()
        np.testing.assert_allclose(result["detrended"], result["y"] - result["trend"])

    def test_detrend_accepts_custom_nixtla_column_names(self):
        frame = pd.DataFrame(
            {
                "series_id": ["a"] * 4,
                "timestamp": [1, 2, 3, 4],
                "value": [1.0, 2.0, 3.0, 4.0],
            }
        )

        result = detrend(
            frame,
            frac=1.0,
            robust=False,
            id_col="series_id",
            time_col="timestamp",
            target_col="value",
        )

        assert result[["series_id", "timestamp", "value"]].equals(frame)
        np.testing.assert_allclose(
            result["detrended"], result["value"] - result["trend"]
        )

    def test_detrend_panel_rejects_missing_columns(self):
        with pytest.raises(ValueError, match="missing required columns"):
            detrend(pd.DataFrame({"unique_id": ["a"], "y": [1.0]}))

    def test_detect_seasonal_periods(self):
        x = np.sin(2 * np.pi * np.arange(32) / 8)
        periods = detect_seasonal_periods(x)
        assert 8 in periods
        assert periods == sorted(set(periods))

    def test_detect_seasonal_periods_ignores_missing_values(self):
        x = np.sin(2 * np.pi * np.arange(32) / 8)
        x[[3, 17]] = np.nan

        periods = detect_seasonal_periods(pd.Series(x), top_k=1)

        assert periods == [8]

    def test_detect_seasonal_periods_supports_panel_data(self):
        steps = np.arange(32)
        frame = pd.DataFrame(
            {
                "unique_id": ["weekly"] * 32 + ["biweekly"] * 32,
                "y": np.concatenate(
                    [
                        np.sin(2 * np.pi * steps / 8),
                        np.sin(2 * np.pi * steps / 16),
                    ]
                ),
            }
        )

        periods = detect_seasonal_periods(frame, top_k=1)

        assert periods["weekly"] == [8]
        assert periods["biweekly"] == [16]

    def test_detect_seasonal_periods_infers_single_numeric_target(self):
        steps = np.arange(32)
        frame = pd.DataFrame(
            {
                "unique_id": ["a"] * 32,
                "timestamp": pd.date_range("2024-01-01", periods=32),
                "value": np.sin(2 * np.pi * steps / 8),
            }
        )

        periods = detect_seasonal_periods(frame, top_k=1)

        assert periods == {"a": [8]}

    def test_hurst_exponent(self):
        x = np.cumsum(np.random.RandomState(0).normal(size=60))
        h, pvalue = hurst_exponent(x)
        assert np.isfinite(h)
        assert np.isfinite(pvalue)

    def test_trend_significance(self):
        x = np.linspace(0, 10, 40)
        r_squared, p_value = trend_significance(x)
        assert r_squared >= 0.0
        assert np.isfinite(p_value)

    def test_seasonal_significance(self):
        y = np.array([0, 1, 0, -1, 0, 1, 0, -1], dtype=float)
        strength, f_stat, p_value = seasonal_significance(
            y, y, np.zeros_like(y), period=4
        )
        assert 0.0 <= strength <= 1.0
        assert np.isfinite(f_stat)
        assert np.isfinite(p_value)

    def test_extract_mstl_components(self):
        result = DecomposeResult(
            observed=np.arange(6, dtype=float),
            trend=np.arange(6, dtype=float) + 0.1,
            seasonal=np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0]),
            resid=np.zeros(6, dtype=float),
            weights=None,
        )
        df = extract_mstl_components(result, periods=[4])
        assert list(df.columns) == ["data", "trend", "seasonal_4", "resid"]
        assert df.shape[0] == 6

    def test_extract_mstl_components_raises_for_wrong_period_count(self):
        result = DecomposeResult(
            observed=np.arange(6, dtype=float),
            trend=np.arange(6, dtype=float) + 0.1,
            seasonal=np.array(
                [[0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
            ),
            resid=np.zeros(6, dtype=float),
            weights=None,
        )
        with pytest.raises(ValueError):
            extract_mstl_components(result, periods=[4])

    def test_detect_seasonal_periods_raises_for_invalid_input(self):
        with pytest.raises(ValueError):
            detect_seasonal_periods(np.array([1.0, 2.0, 3.0]))

        with pytest.raises(ValueError):
            detect_seasonal_periods(np.array([1.0, 2.0, 3.0, 4.0]), top_k=0)

        with pytest.raises(ValueError, match="unique ID"):
            detect_seasonal_periods(pd.DataFrame({"y": [1.0, 2.0, 3.0, 4.0]}))

        with pytest.raises(ValueError, match="Could not infer"):
            detect_seasonal_periods(
                pd.DataFrame(
                    {
                        "unique_id": ["a"] * 4,
                        "first": [1.0, 2.0, 3.0, 4.0],
                        "second": [4.0, 3.0, 2.0, 1.0],
                    }
                )
            )


class TestForecastability:
    def test_foreca(self):
        x = np.sin(2 * np.pi * np.arange(64) / 8)
        omega = foreca(x)
        assert 0.0 <= omega <= 1.0 + 1e-12

    def test_adi_cv(self):
        x = np.array([0, 0, 1, 2, 0, 0, 1, 1], dtype=float)
        adi, cv = adi_cv(x)
        assert adi > 0.0
        assert cv >= 0.0

    def test_sample_entropy(self):
        x = np.array([0.0, 0.5, 1.0, 0.0, 0.5, 1.0], dtype=float)
        entropy = sample_entropy(x)
        assert np.isfinite(entropy)

    def test_regularity_index(self):
        x = np.array([0.0, 0.5, 1.0, 0.0, 0.5, 1.0], dtype=float)
        regularity = regularity_index(x)
        assert 0.0 < regularity <= 1.0

    def test_permutation_entropy(self):
        x = np.array([1, 2, 3, 4, 5, 6], dtype=float)
        pe = permutation_entropy(x)
        assert pe == pytest.approx(0.0)

    def test_theoretical_limit(self):
        x = np.array([1, 2, 3, 4, 5, 6], dtype=float)
        limit = theoretical_limit(x)
        assert limit == pytest.approx(1.0)

    def test_permutation_auto_mutual_information(self):
        x = np.array([0, 1, 0, 1, 0, 1], dtype=float)
        pami = permutation_auto_mutual_information(x)
        assert np.isfinite(pami)

    def test_select_pami_lag(self, monkeypatch):
        pami_values = {1: 0.8, 2: 0.4, 3: 0.7, 4: 0.2}

        def fake_pami(values, tau, m, delay, normalize):
            return pami_values[tau]

        monkeypatch.setattr(
            "tinyshift.series.forecastability.permutation_auto_mutual_information",
            fake_pami,
        )

        tau, value, values = select_pami_lag(np.arange(10), max_tau=4)

        assert tau == 2
        assert value == pytest.approx(0.4)
        np.testing.assert_allclose(values, [0.8, 0.4, 0.7, 0.2])

    def test_select_pami_lag_falls_back_to_global_minimum(self, monkeypatch):
        def fake_pami(values, tau, m, delay, normalize):
            return float(5 - tau)

        monkeypatch.setattr(
            "tinyshift.series.forecastability.permutation_auto_mutual_information",
            fake_pami,
        )

        tau, value, values = select_pami_lag(np.arange(10), max_tau=4)

        assert tau == 4
        assert value == pytest.approx(1.0)
        np.testing.assert_allclose(values, [4.0, 3.0, 2.0, 1.0])

    def test_select_pami_lag_rejects_short_series(self):
        with pytest.raises(ValueError):
            select_pami_lag(np.arange(3), m=3)


class TestInterpolation:
    def test_vi(self):
        fc = vi(np.array([1.0, 2.0]), np.array([3.0, 4.0]), 0.5)
        np.testing.assert_allclose(fc, np.array([2.0, 3.0]))

    def test_vi_with_scalar_input(self):
        fc = vi(np.array([1.0, 2.0]), np.array([3.0, 4.0]), 0.0)
        np.testing.assert_allclose(fc, np.array([1.0, 2.0]))

    def test_hpi(self):
        fc = hpi(np.array([1.0, 2.0, 3.0]), 0.5)
        np.testing.assert_allclose(fc, np.array([1.0, 1.5, 2.5]))

    def test_hfi(self):
        fc = hfi(np.array([1.0, 2.0, 3.0]), 0.5)
        np.testing.assert_allclose(fc, np.array([1.0, 1.5, 2.25]))


class TestMetric:
    def test_wape(self):
        df = pd.DataFrame(
            {
                "unique_id": ["A", "A", "B", "B"],
                "y": [10.0, 20.0, 8.0, 12.0],
                "model_a": [8.0, 24.0, 7.0, 12.0],
            }
        )
        result = wape(df, models=["model_a"])
        assert result.loc[0, "model_a"] == pytest.approx(20.0)

    def test_pbias(self):
        df = pd.DataFrame(
            {
                "unique_id": ["A", "A"],
                "y": [10.0, 20.0],
                "model_a": [8.0, 24.0],
            }
        )
        result = pbias(df, models=["model_a"])
        assert result.loc[0, "model_a"] == pytest.approx(6.6666666667)

    def test_score(self):
        df = pd.DataFrame(
            {
                "unique_id": ["A", "A"],
                "y": [10.0, 20.0],
                "model_a": [8.0, 24.0],
            }
        )
        result = score(df, models=["model_a"])
        assert result.loc[0, "model_a"] == pytest.approx(26.6666666667)

    def test_rmae(self):
        df = pd.DataFrame(
            {
                "unique_id": ["A", "A", "B", "B"],
                "y": [10.0, 20.0, 5.0, 15.0],
                "model_a": [9.0, 21.0, 4.0, 16.0],
                "baseline": [11.0, 19.0, 6.0, 14.0],
            }
        )
        result = rmae(df, models=["model_a"], baseline_col="baseline")
        assert result.loc[0, "model_a"] == pytest.approx(1.0)

    def test_fva_rmae(self):
        y_true = np.array([10.0, 20.0, 30.0, 40.0])
        y_pred = np.array([10.0, 20.0, 30.0, 39.0])
        value = fva_rmae(y_true, y_pred)
        assert np.isfinite(value)

    def test_forecast_instability(self):
        df = pd.DataFrame(
            {
                "unique_id": ["A", "A", "A"],
                "ds": [1, 2, 3],
                "model_a": [10.0, 12.0, 14.0],
            }
        )
        result = forecast_instability(df, models=["model_a"])
        assert result["metric"].eq("forecast_instability").all()
        assert result["model_a"].notna().all()


class TestOutlier:
    def test_hampel_filter(self):
        x = np.array([0.0, 1.0, 2.0, 100.0, 3.0, 4.0, 5.0])
        outliers = hampel_filter(x, window_size=3)
        assert outliers.dtype == bool
        assert outliers.sum() >= 1

    def test_bollinger_bands(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
        outliers = bollinger_bands(x, window_size=2)
        assert outliers.dtype == bool
        assert len(outliers) == len(x)


class TestStability:
    def test_macv(self):
        current = np.array([1.0, 2.0, 3.0])
        previous = np.array([0.0, 1.0, 2.0])
        assert macv(current, previous) == pytest.approx(1.0)

    def test_mach(self):
        x = np.array([1.0, 3.0, 2.0])
        assert mach(x) == pytest.approx(1.5)

    def test_mascv(self):
        y_train = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0], dtype=float)
        y_hat = np.array([1.0, 2.0, 3.0])
        y_hat_prev = np.array([0.0, 1.0, 2.0])
        value = mascv(y_train, y_hat, y_hat_prev, seasonality=2)
        assert np.isfinite(value)

    def test_masch(self):
        y_train = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0], dtype=float)
        y_hat = np.array([1.0, 2.0, 3.0])
        value = masch(y_train, y_hat, seasonality=2)
        assert value == pytest.approx(np.inf)

    def test_rmsscv(self):
        y_train = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0], dtype=float)
        y_hat = np.array([1.0, 2.0, 3.0])
        y_hat_prev = np.array([0.0, 1.0, 2.0])
        value = rmsscv(y_train, y_hat, y_hat_prev, seasonality=2)
        assert np.isfinite(value)

    def test_rmssch(self):
        y_train = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0], dtype=float)
        y_hat = np.array([1.0, 2.0, 3.0])
        value = rmssch(y_train, y_hat, seasonality=2)
        assert value == pytest.approx(np.inf)
