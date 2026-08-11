import numpy as np
import pandas as pd
import pytest

from tinyshift.series.diagnostic import (
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
)
from tinyshift.series.interpolation import vi, hpi, hfi
from tinyshift.series.metric import (
    wape,
    pbias,
    score,
    rmae,
    fva_rmae,
    forecast_instability,
)
from tinyshift.series.outlier import hampel_filter, bollinger_bands
from tinyshift.series.stability import macv, mach, mascv, masch, rmsscv, rmssch
from statsmodels.tsa.seasonal import DecomposeResult


class TestDiagnostic:
    def test_detect_seasonal_periods(self):
        x = np.sin(2 * np.pi * np.arange(32) / 8)
        periods = detect_seasonal_periods(x)
        assert len(periods) >= 0
        assert all(p > 1 for p in periods)

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


class TestInterpolation:
    def test_vi(self):
        fc = vi(np.array([1.0, 2.0]), np.array([3.0, 4.0]), 0.5)
        np.testing.assert_allclose(fc, np.array([2.0, 3.0]))

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
