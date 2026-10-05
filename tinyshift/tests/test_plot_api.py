# Copyright (c) 2024-2026 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License


import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from tinyshift.forecasting.probabilistic.distribution import (
    GammaPredictiveDistribution,
    NegativeBinomialPredictiveDistribution,
)
from tinyshift.forecasting.probabilistic.forecast import PanelPredictiveForecast
from tinyshift.plot import ProbabilisticCalibrationPlot
from tinyshift.plot.calibration import beta_confidence_analysis
from tinyshift.plot.correlation import corr_heatmap
from tinyshift.plot.mstl import MSTLDiagnostics
from tinyshift.plot.power import power_curve


@pytest.mark.parametrize(
    ("func", "kwargs"),
    [
        (beta_confidence_analysis, {"alpha": 2.0, "beta_param": 3.0}),
        (corr_heatmap, {"X": np.arange(12).reshape(6, 2)}),
        (power_curve, {"effect_size": 0.5}),
    ],
    ids=["beta_confidence_analysis", "corr_heatmap", "power_curve"],
)
def test_plot_functions_return_figure_by_default(func, kwargs):
    fig = func(**kwargs)

    assert isinstance(fig, go.Figure)


def test_plot_functions_show_when_renderer_is_provided(monkeypatch):
    monkeypatch.setattr(
        go.Figure,
        "show",
        lambda self, *args, **kwargs: (args, kwargs),
    )

    result = power_curve(effect_size=0.5, fig_type="png")

    assert result == (("png",), {})


def test_mstl_class_reuses_fitted_components():
    series = np.sin(np.arange(80) / 3)
    diagnostics = MSTLDiagnostics(periods=7, nlags=10).fit(series)

    assert list(diagnostics.components_.columns) == [
        "data",
        "trend",
        "seasonal_7",
        "resid",
    ]
    assert list(diagnostics.summary().index) == [
        "trend",
        "residual_ljung_box",
        "seasonal_7",
    ]
    assert isinstance(diagnostics.plot(), go.Figure)


@pytest.mark.parametrize("periods", [True, 1, 0, -2, [], [7, 7], [7, 2.5]])
def test_mstl_rejects_invalid_periods(periods):
    with pytest.raises(ValueError):
        MSTLDiagnostics(periods=periods).fit(np.arange(80, dtype=float))


@pytest.mark.parametrize("nlags", [True, 0, -1, 2.5])
def test_mstl_rejects_invalid_nlags(nlags):
    with pytest.raises(ValueError, match="nlags must be a positive integer"):
        MSTLDiagnostics(periods=7, nlags=nlags)


def test_mstl_rejects_periods_too_large_for_the_series():
    with pytest.raises(ValueError, match="less than half.*invalid periods:.*365"):
        MSTLDiagnostics(periods=[7, 365]).fit(np.arange(80, dtype=float))


def _probabilistic_forecast(distribution):
    n_obs = len(distribution)
    frame = pd.DataFrame(
        {
            "unique_id": ["a"] * (n_obs // 2) + ["b"] * (n_obs - n_obs // 2),
            "ds": list(range(n_obs // 2)) + list(range(n_obs - n_obs // 2)),
            "lambda_t": distribution.means,
        }
    )
    return PanelPredictiveForecast(
        frame, distribution, "model", "unique_id", "ds"
    )


def test_probabilistic_calibration_continuous_plots_and_summary():
    means = np.full(20, 5.0)
    forecast = _probabilistic_forecast(
        GammaPredictiveDistribution(means, np.full(20, 2.0))
    )
    evaluation = forecast.to_frame()[["unique_id", "ds"]].assign(
        y=np.linspace(1.0, 10.0, 20)
    )
    diagnostics = ProbabilisticCalibrationPlot(forecast, evaluation)

    assert np.all((diagnostics.pit_ >= 0.0) & (diagnostics.pit_ <= 1.0))
    assert isinstance(diagnostics.pit_histogram(), go.Figure)
    assert isinstance(diagnostics.pit_acf(max_lag=3), go.Figure)
    assert isinstance(diagnostics.calibration_curve(threshold=5), go.Figure)
    summary = diagnostics.summary(threshold=5, max_lag=3)
    assert isinstance(summary, go.Figure)
    assert len(summary.data) == 4


def test_probabilistic_calibration_discrete_pit_is_reproducible():
    means = np.full(20, 4.0)
    forecast = _probabilistic_forecast(
        NegativeBinomialPredictiveDistribution(means, np.full(20, 3.0))
    )
    evaluation = forecast.to_frame()[["unique_id", "ds"]].assign(
        y=np.tile(np.arange(5), 4)
    )

    first = ProbabilisticCalibrationPlot(forecast, evaluation, random_state=7)
    second = ProbabilisticCalibrationPlot(forecast, evaluation, random_state=7)

    np.testing.assert_allclose(first.pit_, second.pit_)
    lower = forecast.distribution.cdf(evaluation["y"].to_numpy()[:, None] - 1).ravel()
    upper = forecast.distribution.cdf(evaluation["y"].to_numpy()[:, None]).ravel()
    assert np.all(first.pit_ >= lower)
    assert np.all(first.pit_ <= upper)


def test_probabilistic_calibration_uses_requested_renderer(monkeypatch):
    means = np.full(20, 5.0)
    forecast = _probabilistic_forecast(
        GammaPredictiveDistribution(means, np.full(20, 2.0))
    )
    evaluation = forecast.to_frame()[["unique_id", "ds"]].assign(
        y=np.linspace(1.0, 10.0, 20)
    )
    diagnostics = ProbabilisticCalibrationPlot(forecast, evaluation)
    monkeypatch.setattr(
        go.Figure,
        "show",
        lambda self, *args, **kwargs: (args, kwargs),
    )

    result = diagnostics.summary(threshold=5.0, fig_type="png")

    assert result == (("png",), {})
