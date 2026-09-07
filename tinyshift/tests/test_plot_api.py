# Copyright (c) 2024-2026 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License


import numpy as np
import plotly.graph_objects as go
import pytest

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
