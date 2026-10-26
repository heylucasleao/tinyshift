# Copyright (c) 2024-2026 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License


"""Visual diagnostics for local trend dynamics."""

from numbers import Integral, Real

import numpy as np
import pandas as pd

from tinyshift.utils.imports import requires_extra


@requires_extra("plot")
def trend_dynamics(
    df: pd.DataFrame,
    unique_id=None,
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
    window_length: int = 21,
    polyorder: int = 3,
    delta: float = 1.0,
    time_unit: str = "time step",
    show_observed: bool = True,
    fig_type: str | None = None,
    height: int = 850,
    width: int = 1200,
):
    """Plot a series' local trend, slope, and acceleration.

    A Savitzky-Golay local polynomial is evaluated at every observation. Its
    zeroth, first, and second derivatives form the three diagnostic panels.

    Parameters
    ----------
    df : pandas.DataFrame
        Long-format panel containing an ID, time, and numeric target column.
    unique_id : object, optional
        Series to plot. It may be omitted when ``df`` contains exactly one ID.
    id_col, time_col, target_col : str
        Names of the panel columns.
    window_length : int, default=21
        Odd number of observations in each local polynomial window.
    polyorder : int, default=3
        Polynomial degree. It must be at least 2 and less than
        ``window_length``.
    delta : float, default=1.0
        Constant spacing between observations, expressed in ``time_unit``.
    time_unit : str, default="time step"
        Unit displayed for slope and acceleration. For daily data, use
        ``time_unit="day"`` and ``delta=1``.
    show_observed : bool, default=True
        Overlay the observed series on the trend panel.
    fig_type : str, optional
        Plotly renderer passed to :meth:`plotly.graph_objects.Figure.show`.
        When omitted, the figure is returned.
    height, width : int
        Figure dimensions in pixels.

    Returns
    -------
    plotly.graph_objects.Figure or None
        Figure when ``fig_type`` is omitted; otherwise the result of ``show``.

    Notes
    -----
    Savitzky-Golay assumes equally spaced observations. The time column is
    checked for regular spacing after sorting. Derivatives near the boundaries
    are estimated by SciPy's default interpolating polynomial mode.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from scipy.signal import savgol_filter

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame in panel format.")
    required = [id_col, time_col, target_col]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}.")
    if isinstance(window_length, bool) or not isinstance(window_length, Integral):
        raise TypeError("window_length must be an odd integer.")
    if window_length < 3 or window_length % 2 == 0:
        raise ValueError("window_length must be an odd integer of at least 3.")
    if isinstance(polyorder, bool) or not isinstance(polyorder, Integral):
        raise TypeError("polyorder must be an integer.")
    if polyorder < 2 or polyorder >= window_length:
        raise ValueError("polyorder must be at least 2 and less than window_length.")
    if isinstance(delta, bool) or not isinstance(delta, Real) or not np.isfinite(delta):
        raise TypeError("delta must be a finite positive number.")
    if delta <= 0:
        raise ValueError("delta must be a finite positive number.")
    if not isinstance(time_unit, str) or not time_unit.strip():
        raise ValueError("time_unit must be a non-empty string.")

    ids = df[id_col].drop_duplicates()
    if unique_id is None:
        if len(ids) != 1:
            raise ValueError(
                "unique_id must be provided when df contains multiple series."
            )
        unique_id = ids.iloc[0]

    series = df.loc[df[id_col].eq(unique_id), [time_col, target_col]].copy()
    if series.empty:
        raise ValueError(f"No observations found for unique_id={unique_id!r}.")
    if series[[time_col, target_col]].isna().any().any():
        raise ValueError("Time and target values must not be missing.")
    if series[time_col].duplicated().any():
        raise ValueError("The selected series contains duplicate time values.")
    if not pd.api.types.is_numeric_dtype(series[target_col]):
        raise ValueError(f"Target column {target_col!r} must be numeric.")

    series = series.sort_values(time_col, kind="stable")
    values = series[target_col].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Target values must be finite.")
    if len(series) < window_length:
        raise ValueError(
            f"window_length={window_length} exceeds the selected series length "
            f"({len(series)})."
        )

    time_values = series[time_col]
    if pd.api.types.is_datetime64_any_dtype(time_values):
        coordinates = time_values.astype("int64").to_numpy()
    elif pd.api.types.is_numeric_dtype(time_values):
        coordinates = time_values.to_numpy(dtype=float)
    else:
        raise ValueError("Time column must be numeric or datetime-like.")
    gaps = np.diff(coordinates)
    if len(gaps) and (np.any(gaps <= 0) or not np.allclose(gaps, gaps[0])):
        raise ValueError(
            "trend_dynamics requires regularly spaced, strictly increasing times."
        )

    trend = savgol_filter(values, window_length, polyorder, deriv=0)
    slope = savgol_filter(values, window_length, polyorder, deriv=1, delta=float(delta))
    acceleration = savgol_filter(
        values, window_length, polyorder, deriv=2, delta=float(delta)
    )

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            "Observed and local trend",
            "Local slope",
            "Local acceleration",
        ),
    )
    if show_observed:
        fig.add_trace(
            go.Scatter(
                x=time_values,
                y=values,
                name="Observed",
                mode="lines",
                line={"color": "#9CA3AF", "width": 1.25},
            ),
            row=1,
            col=1,
        )
    fig.add_trace(
        go.Scatter(
            x=time_values,
            y=trend,
            name="Local trend",
            mode="lines",
            line={"color": "#2563EB", "width": 3},
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=time_values,
            y=slope,
            name="Slope",
            mode="lines",
            line={"color": "#059669", "width": 2},
            fill="tozeroy",
            fillcolor="rgba(5, 150, 105, 0.12)",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=time_values,
            y=acceleration,
            name="Acceleration",
            mode="lines",
            line={"color": "#D97706", "width": 2},
            fill="tozeroy",
            fillcolor="rgba(217, 119, 6, 0.12)",
        ),
        row=3,
        col=1,
    )
    fig.add_hline(y=0, line_dash="dot", line_color="#6B7280", row=2, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="#6B7280", row=3, col=1)
    fig.update_yaxes(title_text=target_col, row=1, col=1)
    fig.update_yaxes(title_text=f"{target_col} / {time_unit}", row=2, col=1)
    fig.update_yaxes(title_text=f"{target_col} / {time_unit}²", row=3, col=1)
    fig.update_xaxes(title_text=time_col, row=3, col=1)
    fig.update_layout(
        title=f"Local trend dynamics — {unique_id}",
        template="plotly_white",
        hovermode="x unified",
        height=height,
        width=width,
    )

    if fig_type is None:
        return fig
    return fig.show(fig_type)
