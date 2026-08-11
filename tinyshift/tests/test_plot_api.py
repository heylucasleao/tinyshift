import numpy as np
import plotly.graph_objects as go
import pytest

from tinyshift.plot.calibration import beta_confidence_analysis
from tinyshift.plot.correlation import corr_heatmap
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
