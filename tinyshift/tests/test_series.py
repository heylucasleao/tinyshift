# Copyright (c) 2024-2026 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License


import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.seasonal import DecomposeResult

from tinyshift.series import IntermittencyAnalyzer, SeasonalPeriodDetector
from tinyshift.series.decomposition import detrend, extract_mstl_components
from tinyshift.series.diagnostic import (
    hurst_exponent,
    seasonal_significance,
    trend_significance,
)
from tinyshift.series.forecastability import (
    foreca,
    permutation_auto_mutual_information,
    permutation_entropy,
    regularity_index,
    sample_entropy,
    select_pami_lag,
    theoretical_limit,
)
from tinyshift.series.intermittency import IntermittencyAnalyzer as CanonicalAnalyzer
from tinyshift.series.interpolation import hfi, hpi, vi
from tinyshift.series.metric import (
    economic_loss,
    forecast_instability,
    fva_rmae,
    pbias,
    rmae,
    score,
    wape,
)
from tinyshift.series.outlier import bollinger_bands, hampel_filter
from tinyshift.series.profiler import SeriesProfiler


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
        underage_cost="cu",
        overage_cost="co",
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
        underage_cost=3.0,
        overage_cost=1.0,
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

    def test_seasonal_period_detector_detects_periods(self):
        x = np.sin(2 * np.pi * np.arange(32) / 8)
        frame = pd.DataFrame({"unique_id": "a", "ds": np.arange(32), "y": x})
        periods = SeasonalPeriodDetector().fit(frame).results_["a"]["candidate_periods"]
        assert 8 in periods
        assert periods == sorted(set(periods))

    def test_seasonal_period_detector_detects_period_two_at_nyquist(self):
        x = (-1.0) ** np.arange(32)

        frame = pd.DataFrame({"unique_id": "a", "ds": np.arange(32), "y": x})
        periods = (
            SeasonalPeriodDetector(top_k=1)
            .fit(frame)
            .results_["a"]["candidate_periods"]
        )
        assert periods == [2]

    def test_seasonal_period_detector_ignores_missing_values(self):
        x = np.sin(2 * np.pi * np.arange(32) / 8)
        x[[3, 17]] = np.nan

        frame = pd.DataFrame({"unique_id": "a", "ds": np.arange(32), "y": x})
        periods = (
            SeasonalPeriodDetector(top_k=1)
            .fit(frame)
            .results_["a"]["candidate_periods"]
        )

        assert periods == [8]

    def test_seasonal_period_detector_preserves_length_when_interpolating_gaps(self):
        x = np.sin(2 * np.pi * np.arange(32) / 8)
        x[[3, 17]] = np.nan
        frame = pd.DataFrame({"unique_id": "a", "ds": np.arange(32), "y": x})
        detector = SeasonalPeriodDetector(top_k=1).fit(frame)

        assert len(detector.results_["a"]["frequencies"]) == 17

    @pytest.mark.parametrize(
        ("fallback", "expected"),
        [(None, []), (12, [12]), ([7, 30], [7, 30])],
    )
    def test_seasonal_period_detector_uses_fallback_without_peaks(
        self, fallback, expected
    ):
        frame = pd.DataFrame({"unique_id": "a", "ds": np.arange(32), "y": np.ones(32)})
        periods = (
            SeasonalPeriodDetector(fallback=fallback)
            .fit(frame)
            .results_["a"]["candidate_periods"]
        )

        assert periods == expected

    def test_seasonal_period_detector_supports_panel_data(self):
        steps = np.arange(32)
        frame = pd.DataFrame(
            {
                "unique_id": ["weekly"] * 32 + ["biweekly"] * 32,
                "ds": list(range(32)) * 2,
                "y": np.concatenate(
                    [
                        np.sin(2 * np.pi * steps / 8),
                        np.sin(2 * np.pi * steps / 16),
                    ]
                ),
            }
        )

        results = SeasonalPeriodDetector(top_k=1).fit(frame).results_
        assert results["weekly"]["candidate_periods"] == [8]
        assert results["biweekly"]["candidate_periods"] == [16]

    def test_seasonal_period_detector_infers_single_numeric_target(self):
        steps = np.arange(32)
        frame = pd.DataFrame(
            {
                "unique_id": ["a"] * 32,
                "timestamp": pd.date_range("2024-01-01", periods=32),
                "value": np.sin(2 * np.pi * steps / 8),
            }
        )

        detector = SeasonalPeriodDetector(top_k=1).fit(
            frame,
            time_col="timestamp",
            target_col="value",
        )

        assert detector.results_["a"]["candidate_periods"] == [8]
        assert detector.time_col_ == "timestamp"
        assert detector.target_col_ == "value"

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

    def test_seasonal_period_detector_raises_for_invalid_input(self):
        with pytest.raises(TypeError, match="panel format"):
            SeasonalPeriodDetector().fit(np.array([1.0, 2.0, 3.0]))

        with pytest.raises(ValueError):
            SeasonalPeriodDetector(top_k=0).fit(np.array([1.0, 2.0, 3.0, 4.0]))

        with pytest.raises(ValueError, match="required columns"):
            SeasonalPeriodDetector().fit(pd.DataFrame({"y": [1.0, 2.0, 3.0, 4.0]}))

        with pytest.raises(ValueError, match="required columns"):
            SeasonalPeriodDetector().fit(
                pd.DataFrame(
                    {
                        "unique_id": ["a"] * 4,
                        "first": [1.0, 2.0, 3.0, 4.0],
                        "second": [4.0, 3.0, 2.0, 1.0],
                    }
                )
            )

    @pytest.mark.parametrize("fallback", [0, -1, True, [7, "30"]])
    def test_seasonal_period_detector_rejects_invalid_fallback(self, fallback):
        with pytest.raises(ValueError, match="fallback"):
            SeasonalPeriodDetector(fallback=fallback)

    @pytest.mark.parametrize("factor", [np.nan, np.inf, True])
    def test_seasonal_period_detector_rejects_invalid_noise_factor(self, factor):
        with pytest.raises(ValueError, match="noise_threshold_factor"):
            SeasonalPeriodDetector(noise_threshold_factor=factor)


class TestIntermittencyAnalyzer:
    def test_is_available_from_public_series_api(self):
        frame = pd.DataFrame({"unique_id": "a", "ds": np.arange(4), "y": [0, 1, 0, 1]})
        analyzer = IntermittencyAnalyzer().fit(frame)

        assert analyzer.results_["a"]["classification"] == "intermittent"
        assert CanonicalAnalyzer is IntermittencyAnalyzer

    def test_sorts_panel_by_id_and_time_and_returns_id_as_column(self):
        frame = pd.DataFrame(
            {
                "unique_id": ["a", "a", "a", "a"],
                "ds": [3, 1, 4, 2],
                "y": [1, 1, 0, 0],
            }
        )

        result = IntermittencyAnalyzer().fit(frame).profile()

        assert result.columns.tolist() == [
            "unique_id",
            "adi",
            "cv2",
            "zero_proportion",
            "interval_cv",
            "classification",
        ]
        np.testing.assert_array_equal(
            IntermittencyAnalyzer().fit(frame).results_["a"]["intervals"], [2]
        )

    def test_column_names_are_fit_parameters(self):
        frame = pd.DataFrame({"item": ["a", "a"], "date": [1, 2], "demand": [0.0, 1.0]})

        analyzer = IntermittencyAnalyzer().fit(
            frame, id_col="item", time_col="date", target_col="demand"
        )

        assert analyzer.profile()["item"].tolist() == ["a"]
        assert analyzer.id_col_ == "item"

    def test_profile_requires_fit(self):
        with pytest.raises(RuntimeError, match="fitted"):
            IntermittencyAnalyzer().profile()

        with pytest.raises(RuntimeError, match="fitted"):
            SeasonalPeriodDetector().profile()

    def test_seasonality_profile_returns_one_row_per_series(self):
        steps = np.arange(32)
        frame = pd.DataFrame(
            {
                "unique_id": ["a"] * 32 + ["b"] * 32,
                "ds": list(steps) * 2,
                "y": np.tile(np.sin(2 * np.pi * steps / 8), 2),
            }
        )

        profile = SeasonalPeriodDetector(top_k=1).fit(frame).profile()

        assert profile.to_dict("records") == [
            {"unique_id": "a", "candidate_periods": [8]},
            {"unique_id": "b", "candidate_periods": [8]},
        ]
        assert set(SeasonalPeriodDetector(top_k=1).fit(frame).results_["a"]) == {
            "candidate_periods",
            "frequencies",
            "power",
            "peaks",
        }

    @pytest.mark.parametrize("threshold", [np.nan, np.inf, True])
    def test_rejects_invalid_thresholds(self, threshold):
        with pytest.raises(ValueError, match="adi_threshold"):
            IntermittencyAnalyzer(adi_threshold=threshold)

        with pytest.raises(ValueError, match="cv2_threshold"):
            IntermittencyAnalyzer(cv2_threshold=threshold)


class TestSeriesProfiler:
    def test_returns_complete_panel_profile(self):
        steps = np.arange(64)
        frame = pd.concat(
            [
                pd.DataFrame(
                    {
                        "unique_id": unique_id,
                        "ds": steps[::-1],
                        "y": 3.0
                        + np.sin(2 * np.pi * steps[::-1] / period)
                        + 0.01 * steps[::-1],
                    }
                )
                for unique_id, period in [("a", 8), ("b", 16)]
            ],
            ignore_index=True,
        )

        profiler = SeriesProfiler(top_k=1).fit(frame)
        result = profiler.summary()

        assert result.columns.tolist() == [
            "unique_id",
            "adi",
            "cv2",
            "zero_prop",
            "interval_cv",
            "class",
            "foreca",
            "limit",
            "hurst",
            "trend_r2",
            "trend_pvalue",
            "spectral_conc",
            "candidate_periods",
        ]
        assert result["unique_id"].tolist() == ["a", "b"]
        assert result.set_index("unique_id").loc["a", "candidate_periods"] == [8]
        assert result.set_index("unique_id").loc["b", "candidate_periods"] == [16]
        assert (
            result.drop(columns=["unique_id", "class", "candidate_periods"])
            .apply(np.isfinite)
            .all()
            .all()
        )
        assert set(profiler.results_["a"]) == {
            "demand_occurrence",
            "predictability",
            "temporal_structure",
            "spectral_structure",
        }

    def test_accepts_custom_panel_column_names(self):
        steps = np.arange(64)
        frame = pd.DataFrame(
            {
                "item": "a",
                "date": steps,
                "demand": 2.0 + np.sin(2 * np.pi * steps / 8),
            }
        )

        result = (
            SeriesProfiler(top_k=1)
            .fit(
                frame,
                id_col="item",
                time_col="date",
                target_col="demand",
            )
            .summary()
        )

        assert result.loc[0, "item"] == "a"
        assert result.loc[0, "candidate_periods"] == [8]

    def test_rejects_series_too_short_for_hurst(self):
        frame = pd.DataFrame(
            {"unique_id": "a", "ds": np.arange(20), "y": np.arange(20.0)}
        )

        with pytest.raises(ValueError, match="at least 30"):
            SeriesProfiler().fit(frame)

    def test_summary_requires_fit(self):
        with pytest.raises(RuntimeError, match="fitted"):
            SeriesProfiler().summary()


class TestForecastability:
    def test_foreca(self):
        x = np.sin(2 * np.pi * np.arange(64) / 8)
        omega = foreca(x)
        assert 0.0 <= omega <= 1.0 + 1e-12

    def test_foreca_handles_constant_series(self):
        assert foreca(np.ones(8)) == pytest.approx(1.0)

    def test_foreca_rejects_non_finite_values(self):
        with pytest.raises(ValueError, match="finite"):
            foreca([1.0, np.nan, 2.0])

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

        tau, value, values = select_pami_lag(
            np.arange(10), max_tau=4, return_mode="value_only"
        )

        assert tau == 2
        assert value == pytest.approx(0.4)
        np.testing.assert_allclose(values, [0.8, 0.4, 0.7, 0.2])

    def test_select_pami_lag_rejects_missing_fallback(self, monkeypatch):
        def fake_pami(values, tau, m, delay, normalize):
            return float(5 - tau)

        monkeypatch.setattr(
            "tinyshift.series.forecastability.permutation_auto_mutual_information",
            fake_pami,
        )

        with pytest.raises(ValueError, match="no explicit 'fallback'"):
            select_pami_lag(np.arange(10), max_tau=4, return_mode="value_only")

    def test_select_pami_lag_uses_explicit_fallback(self, monkeypatch):
        def fake_pami(values, tau, m, delay, normalize):
            return float(5 - tau)

        monkeypatch.setattr(
            "tinyshift.series.forecastability.permutation_auto_mutual_information",
            fake_pami,
        )

        lags, value, values = select_pami_lag(
            np.arange(10), max_tau=4, fallback=2, return_mode="point"
        )

        assert lags == [2]
        assert value == pytest.approx(3.0)
        np.testing.assert_allclose(values, [4.0, 3.0, 2.0, 1.0])

    def test_select_pami_lag_returns_nan_for_out_of_bounds_fallback(self, monkeypatch):
        monkeypatch.setattr(
            "tinyshift.series.forecastability.permutation_auto_mutual_information",
            lambda values, tau, m, delay, normalize: float(5 - tau),
        )

        lag, value, _ = select_pami_lag(
            np.arange(10), max_tau=4, fallback=5, return_mode="value_only"
        )

        assert lag == 5
        assert np.isnan(value)

    def test_select_pami_lag_rejects_short_series(self):
        with pytest.raises(ValueError):
            select_pami_lag(np.arange(3), m=3)

    def test_select_pami_lag_return_modes(self, monkeypatch):
        pami_values = {1: 0.8, 2: 0.6, 3: 0.2, 4: 0.7, 5: 0.5}

        def fake_pami(values, tau, m, delay, normalize):
            assert normalize is True
            return pami_values[tau]

        monkeypatch.setattr(
            "tinyshift.series.forecastability.permutation_auto_mutual_information",
            fake_pami,
        )

        for return_mode, expected_lags in [
            ("range", [1, 2, 3]),
            ("point", [3]),
            ("short_term", [1, 2, 3]),
        ]:
            lags, value, values = select_pami_lag(
                np.arange(10),
                max_tau=5,
                normalize=True,
                return_mode=return_mode,
                short_term=2,
            )
            assert lags == expected_lags
            assert value == pytest.approx(0.2)
            np.testing.assert_allclose(values, [0.8, 0.6, 0.2, 0.7, 0.5])

        lag, value, values = select_pami_lag(
            np.arange(10), max_tau=5, normalize=True, return_mode="value_only"
        )
        assert lag == 3
        assert value == pytest.approx(0.2)
        np.testing.assert_allclose(values, [0.8, 0.6, 0.2, 0.7, 0.5])

    def test_select_pami_lag_clips_max_tau_to_valid_range(self, monkeypatch):
        evaluated_taus = []

        def fake_pami(values, tau, m, delay, normalize):
            evaluated_taus.append(tau)
            return float(tau)

        monkeypatch.setattr(
            "tinyshift.series.forecastability.permutation_auto_mutual_information",
            fake_pami,
        )

        _, _, values = select_pami_lag(
            np.arange(8), max_tau=100, m=3, delay=2, fallback=1
        )

        assert evaluated_taus == [1, 2, 3]
        np.testing.assert_allclose(values, [1.0, 2.0, 3.0])

    def test_select_pami_lag_rejects_invalid_return_mode(self, monkeypatch):
        monkeypatch.setattr(
            "tinyshift.series.forecastability.permutation_auto_mutual_information",
            lambda values, tau, m, delay, normalize: float(tau),
        )

        with pytest.raises(ValueError, match="Invalid return_mode"):
            select_pami_lag(np.arange(8), max_tau=3, fallback=1, return_mode="invalid")


class TestInterpolation:
    def test_vi(self):
        fc = vi(np.array([1.0, 2.0]), np.array([3.0, 4.0]), 0.5)
        np.testing.assert_allclose(fc, np.array([2.0, 3.0]))

    def test_vi_with_scalar_input(self):
        fc = vi(np.array([1.0, 2.0]), np.array([3.0, 4.0]), 0.0)
        np.testing.assert_allclose(fc, np.array([1.0, 2.0]))

    def test_vi_accepts_lists_and_rejects_different_shapes(self):
        np.testing.assert_allclose(vi([1.0, 2.0], [3.0, 4.0], 0.5), [2.0, 3.0])
        with pytest.raises(ValueError, match="same shape"):
            vi([1.0], [2.0, 3.0], 0.5)

    def test_hpi(self):
        fc = hpi(np.array([1.0, 2.0, 3.0]), 0.5)
        np.testing.assert_allclose(fc, np.array([1.0, 1.5, 2.5]))

    def test_hfi(self):
        fc = hfi(np.array([1.0, 2.0, 3.0]), 0.5)
        np.testing.assert_allclose(fc, np.array([1.0, 1.5, 2.25]))

    def test_horizontal_interpolation_preserves_fractional_results(self):
        np.testing.assert_allclose(hpi([0, 3], 0.5), [0.0, 1.5])
        np.testing.assert_allclose(hfi([0, 3, 3], 0.5), [0.0, 1.5, 2.25])


class TestMetric:
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

    def test_fva_rmae_moving_average_preserves_initial_window_behavior(self):
        y_true = np.array([1.0, 3.0, 5.0, 7.0])
        y_pred = np.array([1.0, 2.0, 4.0, 6.0])

        value = fva_rmae(
            y_true,
            y_pred,
            baseline_type="moving_average",
            window_size=2,
        )

        assert value == pytest.approx(3 / 7)

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

    def test_forecast_instability_isolates_missing_values_by_model(self):
        df = pd.DataFrame(
            {
                "unique_id": ["A"] * 4,
                "ds": [1, 2, 3, 4],
                "model_a": [10.0, np.nan, 12.0, 13.0],
                "model_b": [10.0, 11.0, 12.0, 13.0],
            }
        )

        result = forecast_instability(df, models=["model_a", "model_b"])

        assert result.loc[0, "model_a"] == pytest.approx(16.0)
        assert result.loc[0, "model_b"] == pytest.approx(17.3913043478)


class TestOutlier:
    def test_hampel_filter(self):
        x = np.array([0.0, 1.0, 2.0, 100.0, 3.0, 4.0, 5.0])
        outliers = hampel_filter(x, window_size=3)
        assert outliers.dtype == bool
        assert outliers.sum() >= 1

    def test_hampel_filter_always_returns_series(self):
        result = hampel_filter([1.0, 2.0], window_size=3)
        assert isinstance(result, pd.Series)
        assert result.tolist() == [False, False]

    def test_hampel_filter_uses_nan_aware_rolling_statistics(self):
        result = hampel_filter([1.0, np.nan, 1.0, 1.0, 100.0], window_size=3)
        assert result.tolist() == [False, False, False, False, True]

    @pytest.mark.parametrize("window_size", [True, 2, 3.5])
    def test_hampel_filter_rejects_invalid_window_size(self, window_size):
        with pytest.raises(ValueError, match="integer"):
            hampel_filter([1.0, 2.0, 3.0], window_size=window_size)

    def test_bollinger_bands(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
        outliers = bollinger_bands(x, window_size=2)
        assert outliers.dtype == bool
        assert len(outliers) == len(x)

    def test_bollinger_bands_preserves_population_standard_deviation(self):
        x = np.array([0.0, 2.0, 4.0])

        result = bollinger_bands(x, window_size=2, factor=0.8)

        assert result.tolist() == [True, True, True]
