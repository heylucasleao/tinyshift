# Copyright (c) 2024-2026 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License


import numpy as np
import pandas as pd
import pytest

from tinyshift.stats.bootstrap_bca import BootstrapBCA
from tinyshift.stats.statistical_interval import StatisticalInterval
from tinyshift.stats.utils import (
    assess_comparability,
    chebyshev_guaranteed_percentage,
    expanding_window,
    generate_lag,
    is_obsolete,
    jacknife,
    mad,
    remove_leading_zeros,
    rolling_window,
)


def test_chebyshev_guaranteed_percentage_matches_theoretical_bound():
    x = np.array([1, 2, 3, 4, 5])
    value = chebyshev_guaranteed_percentage(x, interval=(1, 5))

    assert value == pytest.approx(0.375, rel=1e-3)


def test_rolling_window_returns_expected_shape_and_values():
    x = np.array([1, 2, 3, 4])
    result = rolling_window(x, window_size=2, func=np.mean)

    assert result.shape == (4,)
    np.testing.assert_allclose(result, np.array([1.5, 1.5, 2.5, 3.5]))


def test_expanding_window_returns_expected_shape_and_values():
    x = np.array([1, 2, 3, 4])
    result = expanding_window(x, func=np.mean, window_size=2)

    assert result.shape == (4,)
    np.testing.assert_allclose(result, np.array([1.5, 1.5, 2.0, 2.5]))


def test_jackknife_and_mad_and_generate_lag():
    x = np.array([1, 3, 5, 7])

    jackknife_result = jacknife(x, func=np.mean)
    assert jackknife_result.shape == (4,)
    np.testing.assert_allclose(
        jackknife_result, np.array([5.0, 4.333333333333333, 3.6666666666666665, 3.0])
    )

    assert mad(x) == pytest.approx(2.0)

    lagged = generate_lag(x, lag=2)
    np.testing.assert_allclose(lagged[:2], np.array([np.nan, np.nan]))
    np.testing.assert_allclose(lagged[2:], np.array([4.0, 4.0]))


def test_remove_leading_zeros_and_is_obsolete():
    df = pd.DataFrame(
        {
            "ds": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "y": [0, 0, 5],
        }
    )

    cleaned = remove_leading_zeros(df)
    assert cleaned.iloc[0]["y"] == 5
    assert is_obsolete(df, days_obsoletes=1) is np.bool_(False)


def test_validation_paths_for_stats_utils():
    with pytest.raises(ValueError, match="window_size"):
        rolling_window([1, 2, 3], window_size=1)

    with pytest.raises(ValueError, match="1-dimensional"):
        rolling_window([[1], [2], [3]], window_size=2)

    with pytest.raises(ValueError, match="window_size"):
        expanding_window([1, 2], window_size=0)

    with pytest.raises(ValueError, match="larger"):
        expanding_window([1, 2], window_size=3)

    with pytest.raises(ValueError, match="1-dimensional"):
        jacknife([[1], [2], [3]])

    with pytest.raises(ValueError, match="one-dimensional"):
        generate_lag(np.array([[1, 2], [3, 4]]))


def test_assess_comparability_returns_expected_rows():
    df = pd.DataFrame(
        {
            "group": ["control", "control", "treatment", "treatment"],
            "feature": [1.0, 2.0, 4.0, 6.0],
        }
    )

    result = assess_comparability(df, features=["feature"], group_col="group")

    assert list(result.columns) == ["group", "feature", "cohen_d"]
    assert result.shape[0] == 1
    assert result.iloc[0]["group"] == "treatment"
    assert result.iloc[0]["cohen_d"] > 0


def test_statistical_interval_methods_and_compute_interval():
    x = np.array([1, 2, 3, 4, 5])

    lower, upper = StatisticalInterval.stddev_interval(x)
    expected_lower = np.mean(x) - 3 * np.std(x)
    expected_upper = np.mean(x) + 3 * np.std(x)
    assert lower == pytest.approx(expected_lower)
    assert upper == pytest.approx(expected_upper)

    iqr_lower, iqr_upper = StatisticalInterval.iqr_interval(x)
    assert iqr_lower < iqr_upper

    custom_lower, custom_upper = StatisticalInterval.compute_interval(
        x, lambda values: (np.min(values), np.max(values))
    )
    assert custom_lower == pytest.approx(1.0)
    assert custom_upper == pytest.approx(5.0)

    auto_lower, auto_upper = StatisticalInterval.compute_interval(x, "auto")
    assert auto_lower < auto_upper


def test_bootstrap_bca_interval_is_finite_and_contains_observed_statistic():
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    lower, upper = BootstrapBCA.compute_interval(
        data,
        confidence_level=0.95,
        statistic=np.mean,
        n_resamples=200,
        random_state=42,
    )

    observed = np.mean(data)
    assert np.isfinite(lower)
    assert np.isfinite(upper)
    assert lower < upper
    assert lower <= observed <= upper
