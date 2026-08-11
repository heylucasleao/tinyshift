# Copyright (c) 2024-2026 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License

from typing import Union, List, Optional
import numpy as np
import pandas as pd

from tinyshift.utils.imports import requires_extra


@requires_extra("plots")
def seasonal_decompose(
    X: Union[np.ndarray, List[float], pd.Series],
    periods: Union[int, List[int]],
    nlags: int = 10,
    height: int = 1200,
    width: int = 1300,
    fig_type: Optional[str] = None,
):
    """
    Performs seasonal decomposition of a time series using MSTL and plots the components.

    This function uses the MSTL (Multiple Seasonal-Trend decomposition using Loess) method
    from statsmodels to separate a time series into trend, seasonal, and residual components.
    It calculates trend significance, seasonal strength metrics, hypothesis tests for each
    seasonality, and performs the Ljung-Box test on residuals.

    Parameters
    ----------
    X : Union[np.ndarray, List[float], pd.Series]
        Input time series data.
    periods : int or list of int
        Period(s) of the seasonal components (e.g., 7 or [7, 365]).
    nlags : int, default=10
        Number of lags to use in the Ljung-Box test for residual autocorrelation.
    height : int, default=1200
        Figure height in pixels.
    width : int, default=1300
        Figure width in pixels.
    fig_type : str, optional
        Plotly figure output type passed to `fig.show()`. E.g.: 'json', 'html', 'notebook'.

    Returns
    -------
    plotly.graph_objects.Figure or None
        Returns the Plotly Figure object if `fig_type` is `None` or the result of `fig.show()`.

    Notes
    -----
    Interpretation of statistical tests and metrics in the Summary panel:

    1. **Trend Significance (R² and p-value)**:
       - Measures linear trend strength. High R² with p < 0.05 indicates a statistically
         significant linear trend in the time series.

    2. **Ljung-Box Test (Residual Autocorrelation)**:
       - **H0**: The residuals are independently distributed (white noise).
       - **Ha**: The residuals exhibit autocorrelation.
       - A p-value > 0.05 suggests that residuals behave as white noise, indicating
         that the model captured all major predictable patterns.

    3. **Seasonal Strength (F_s)**:
       - Calculated as: F_s = max(0, 1 - Var(Residuals) / Var(Seasonal + Residuals))
       - Range: [0, 1]. Values close to 1 represent strong seasonal patterns, while
         values near 0 indicate negligible seasonality.

    4. **Seasonal F-Test (F-Stat and p-value)**:
       - **H0**: The seasonal component does not explain significant variance.
       - **Ha**: The seasonal pattern is statistically significant.
       - A p-value < 0.05 indicates strong evidence of deterministic periodicity
         at the specified period.
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.tsa.seasonal import MSTL
    import plotly.subplots as sp
    import plotly.express as px
    import plotly.graph_objs as go
    from tinyshift.series import (
        trend_significance,
        seasonal_significance,
        extract_mstl_components,
    )

    period_list = [periods] if isinstance(periods, int) else list(periods)
    index = X.index if hasattr(X, "index") else list(range(len(X)))

    if not isinstance(X, pd.Series):
        X_series = pd.Series(np.asarray(X, dtype=np.float64))
    else:
        X_series = X.astype(np.float64)

    nlags = max(1, min(nlags, len(X_series) // 5))

    mstl = MSTL(X_series, periods=period_list)
    result_mstl = mstl.fit()
    components_df = extract_mstl_components(result_mstl, period_list)

    r_squared, p_value_trend = trend_significance(X_series.values)
    trend_summary = f"R²={r_squared:.4f}, p={p_value_trend:.4f}"

    ljung_box = acorr_ljungbox(components_df["resid"].dropna(), lags=[nlags])
    ljung_stat = ljung_box["lb_stat"].values[0]
    ljung_p = ljung_box["lb_pvalue"].values[0]
    ljung_summary = f"Stats={ljung_stat:.4f}, p={ljung_p:.4f}"

    summary_dict = {
        "Trend Significance": trend_summary,
        "Ljung-Box Test": ljung_summary,
    }

    y_detrended = X_series.values - components_df["trend"].values
    for p in period_list:
        s_col = f"seasonal_{p}"
        strength, f_stat, p_val_seas = seasonal_significance(
            y_detrended=y_detrended,
            seasonal_component=components_df[s_col].values,
            residuals=components_df["resid"].values,
            period=p,
        )
        summary_dict[f"Seasonality (Period {p})"] = (
            f"Strength={strength:.4f} | F-Test={f_stat:.4f}, p={p_val_seas:.4f}"
        )

    summary_html = "<br>".join([f"<b>{k}</b>: {v}" for k, v in summary_dict.items()])

    colors = px.colors.qualitative.T10
    subplot_titles = [
        col.capitalize().replace("_", " ") for col in components_df.columns
    ] + ["Summary"]

    fig = sp.make_subplots(
        rows=len(subplot_titles),
        cols=1,
        subplot_titles=subplot_titles,
    )

    for i, col in enumerate(components_df.columns):
        color = colors[i % len(colors)]
        fig.add_trace(
            go.Scatter(
                x=index,
                y=components_df[col],
                mode="lines",
                hovertemplate=f"{col.capitalize()}: %{{y}}<extra></extra>",
                line=dict(color=color),
            ),
            row=i + 1,
            col=1,
        )

    fig.add_trace(
        go.Scatter(x=[0], y=[0], text=[summary_html], mode="text", showlegend=False),
        row=len(subplot_titles),
        col=1,
    )

    fig.update_xaxes(visible=False, row=len(subplot_titles), col=1)
    fig.update_yaxes(visible=False, row=len(subplot_titles), col=1)

    fig.update_layout(
        title="Seasonal Decomposition (MSTL)",
        height=height,
        width=width,
        showlegend=False,
        hovermode="x",
    )

    return fig.show(fig_type)


@requires_extra("plots")
def stationarity_analysis(
    df: Union[pd.DataFrame, pd.Series],
    height: int = 1200,
    width: int = 1300,
    nlags: int = 30,
    fig_type: Optional[str] = None,
):
    """
    Creates interactive ACF and PACF plots with ADF test results for multiple series.

    This function generates a comprehensive diagnostic visualization to assess the
    stationarity and autocorrelation structure of multiple time series in a single panel.
    The plot includes the series itself, its autocorrelation function (ACF) and partial
    autocorrelation function (PACF), and a summary of the Augmented Dickey-Fuller (ADF)
    test results.

    Parameters
    ----------
    df : pandas.DataFrame, pandas.Series, or list
        Input data containing the time series. Can be:
        - DataFrame: Multiple columns will be analyzed
        - Series: Will be converted to single-column DataFrame
    height : int, default=1200
        Figure height in pixels.
    width : int, default=1300
        Figure width in pixels.
    nlags : int, default=30
        Number of lags to include in ACF and PACF calculations.
        Default is 30 or half the length of the series, whichever is smaller.
    fig_type : str, optional
        Plotly figure output type. Passed to `fig.show()`.
        E.g.: 'json', 'html', 'notebook'.

    Returns
    -------
    plotly.graph_objects.Figure or None
        Returns the Plotly Figure object if `fig_type` is `None` or the result
        of the `fig.show(fig_type)` call.

    Notes
    -----
    Statistical Diagnostics Overview:

    1. Augmented Dickey-Fuller (ADF) Test:
       - Evaluates the presence of a unit root (non-stationarity).
       - Null Hypothesis (H0): The series has a unit root (is non-stationary).
       - Alternative Hypothesis (H1): The series is stationary.
       - A low p-value (p < 0.05) rejects H0, indicating the
         series is stationary. A strong negative ADF statistic confirms rejection.

    2. Autocorrelation Function (ACF):
       - Measures linear correlation between the series and lagged values of itself.
       - Slow linear decay indicates non-stationarity or trend.
         In stationary series, significant spikes identify MA(q) process orders.
         Periodic spikes indicate seasonal components.

    3. Partial Autocorrelation Function (PACF):
       - Measures direct correlation between the series and lags, controlling for
         intermediate lags.
       - Direct cutoff after lag `p` identifies AR(p) process orders.

    Confidence bands are displayed on ACF and PACF plots at the ±1.96/√N level (95% CI).
    """
    from statsmodels.tsa.stattools import adfuller, acf, pacf
    import plotly.subplots as sp
    import plotly.express as px
    import plotly.graph_objs as go

    nlags = min(nlags, (len(df) // 2) - 1)

    if isinstance(df, pd.Series):
        series_name = df.name if df.name is not None else "Value"
        df = df.to_frame(name=series_name)

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "Input must be a pandas Series, pandas DataFrame, or a list (of lists)."
        )

    N = len(df.columns)
    colors = px.colors.qualitative.T10
    num_colors = len(colors)

    def create_acf_pacf_traces(X, nlags=30, color=None):
        """
        Helper function to create ACF and PACF traces with confidence intervals.
        """

        N = len(X)
        conf = 1.96 / np.sqrt(N)
        acf_vals = acf(X, nlags=nlags)
        pacf_vals = pacf(X, nlags=nlags, method="yw")

        acf_bar = go.Bar(
            x=list(range(len(acf_vals))),
            y=acf_vals,
            marker_color=color,
            name="ACF",
        )
        pacf_bar = go.Bar(
            x=list(range(len(pacf_vals))),
            y=pacf_vals,
            marker_color=color,
            name="PACF",
        )

        band_upper = go.Scatter(
            x=list(range(nlags + 1)),
            y=[conf] * (nlags + 1),
            mode="lines",
            line=dict(color="gray", dash="dash"),
            showlegend=False,
            name="Confidence Band",
        )
        band_lower = go.Scatter(
            x=list(range(nlags + 1)),
            y=[-conf] * (nlags + 1),
            mode="lines",
            line=dict(color="gray", dash="dash"),
            showlegend=False,
            name="Confidence Band",
        )

        return acf_bar, pacf_bar, band_upper, band_lower

    subplot_titles = []
    for var in df.columns:
        subplot_titles.extend([f"Series ({var})", f"ACF ({var})", f"PACF ({var})"])
    subplot_titles.extend(["ADF Results Summary", "", ""])

    fig = sp.make_subplots(rows=N + 1, cols=3, subplot_titles=subplot_titles)

    adf_results = {}

    for i, var in enumerate(df.columns, start=1):
        X = df[var].dropna()
        adf_stat, p_value = adfuller(X)[:2]
        adf_results[var] = f"ADF={adf_stat:.2f}, p={p_value:.4f}"
        color = colors[(i - 1) % num_colors]

        fig.add_trace(
            go.Scatter(
                x=X.index,
                y=X,
                mode="lines",
                name="Series",
                showlegend=False,
                line=dict(color=color),
            ),
            row=i,
            col=1,
        )

        acf_values, pacf_values, conf_up, conf_lo = create_acf_pacf_traces(
            X,
            color=color,
            nlags=nlags,
        )

        fig.add_trace(acf_values, row=i, col=2)
        fig.add_trace(pacf_values, row=i, col=3)
        fig.add_trace(conf_up, row=i, col=2)
        fig.add_trace(conf_lo, row=i, col=2)
        fig.add_trace(conf_up, row=i, col=3)
        fig.add_trace(conf_lo, row=i, col=3)

    adf_text = "<br>".join([f"<b>{k}</b>: {v}" for k, v in adf_results.items()])

    fig.add_trace(
        go.Scatter(
            x=[0],
            y=[0],
            text=[adf_text],
            mode="text",
            showlegend=False,
            name="Summary",
            hoverinfo="skip",
        ),
        row=N + 1,
        col=1,
    )

    fig.update_layout(
        title="ACF/PACF with ADF Summary",
        height=height,
        width=width,
        showlegend=False,
    )

    for row in range(1, N + 1):
        fig.update_xaxes(title_text="Index", row=row, col=1)
        fig.update_yaxes(title_text="Value", row=row, col=1)
        fig.update_xaxes(title_text="Lag", row=row, col=2)
        fig.update_xaxes(title_text="Lag", row=row, col=3)
        fig.update_yaxes(title_text="ACF", row=row, col=2)
        fig.update_yaxes(title_text="PACF", row=row, col=3)

    fig.update_xaxes(visible=False, row=N + 1, col=1)
    fig.update_yaxes(visible=False, row=N + 1, col=1)

    return fig.show(fig_type)


@requires_extra("plots")
def residual_analysis(
    df: Union[pd.DataFrame, pd.Series],
    height: int = 1200,
    width: int = 1300,
    nlags: int = 10,
    fig_type: Optional[str] = None,
):
    """
    Creates diagnostic plots for residual analysis including histogram, QQ-plot, and Ljung-Box test.

    This function generates a comprehensive residual diagnostic visualization to assess
    the distribution properties and autocorrelation structure of residuals from time series
    models. The plot includes the residual series itself, histogram, QQ-plot against normal
    distribution, and a summary of the Ljung-Box test results.

    Parameters
    ----------
    df : pandas.DataFrame, pandas.Series
        Input data containing the residual series. Can be:
        - DataFrame: Multiple columns will be analyzed as separate residual series
        - Series: Will be converted to single-column DataFrame
    height : int, default=1200
        Figure height in pixels.
    width : int, default=1300
        Figure width in pixels.
    nlags : int, default=10
        Number of lags to use in the Ljung-Box test for residual autocorrelation and ARCH test for heteroscedasticity.
        Default is set to 10 or 1/5th of the length of the series, whichever is smaller. (Rob J Hyndman rule of thumb for lag selection non-seasonal time series.)
    fig_type : str, optional
        Plotly figure output type. Passed to `fig.show()`.
        E.g.: 'json', 'html', 'notebook'.

    Returns
    -------
    plotly.graph_objects.Figure or None
        Returns the Plotly Figure object if `fig_type` is `None` or the result
        of the `fig.show(fig_type)` call.

    Notes
    -----
    Statistical Diagnostics Overview:

    1. Ljung-Box Test (Autocorrelation):
       - Evaluates whether residuals are independently distributed (white noise).
       - Null Hypothesis (H0): Residuals are i.i.d. (no autocorrelation up to lag `nlags`).
       - Alternative Hypothesis (H1): Residuals exhibit autocorrelation.
       - A high p-value (p > 0.05) fails to reject H0, confirming that
         residuals behave as white noise and no dynamic pattern was left unmodeled.

    2. Engle's LM-ARCH Test (Heteroscedasticity):
       - Assesses Autoregressive Conditional Heteroscedasticity (volatility clustering).
       - Null Hypothesis (H0): Residuals are homoscedastic (constant variance over time).
       - Alternative Hypothesis (H1): ARCH effects are present (time-varying variance).
       - A high p-value (p > 0.05) fails to reject H0, confirming
         homoscedasticity across time lags.

    3. Histogram & Density:
       - Visualizes the empirical distribution of residuals.
       - A symmetric, bell-shaped curve centered at zero.

    4. Quantile-Quantile (QQ) Plot:
       - Compares residual quantiles against standard normal distribution quantiles.
       - Points closely aligning along the red 45-degree theoretical
         line. Deviations at the tails indicate heavy tails or skewness.
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
    import scipy.stats
    import plotly.subplots as sp
    import plotly.express as px
    import plotly.graph_objects as go

    nlags = min(nlags, len(df) // 5)

    if isinstance(df, pd.Series):
        series_name = df.name if df.name is not None else "Value"
        df = df.to_frame(name=series_name)

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "Input must be a pandas Series, pandas DataFrame, or a list (of lists)."
        )

    N = len(df.columns)
    colors = px.colors.qualitative.T10
    num_colors = len(colors)

    subplot_titles = []
    for var in df.columns:
        subplot_titles.extend(
            [f"Series ({var})", f"Histogram ({var})", f"QQ-Plot ({var})"]
        )
    subplot_titles.extend(["Ljung-Box Results Summary", "LM-ARCH Results Summary", ""])

    fig = sp.make_subplots(rows=N + 1, cols=3, subplot_titles=subplot_titles)

    ljung_box_results = {}
    arch_results = {}

    for i, var in enumerate(df.columns, start=1):
        X = df[var].dropna()
        lb_stat, p_value = acorr_ljungbox(X, lags=[nlags], return_df=True).iloc[0]
        ljung_box_results[var] = f"LB={lb_stat:.2f}, p={p_value:.4f}"
        arch_stat, p_value = het_arch(X, nlags=nlags)[:2]
        arch_results[var] = f"ARCH={arch_stat:.2f}, p={p_value:.4f}"
        color = colors[(i - 1) % num_colors]

        fig.add_trace(
            go.Scatter(
                x=X.index,
                y=X,
                mode="lines",
                name="Series",
                showlegend=False,
                line=dict(color=color),
            ),
            row=i,
            col=1,
        )

        fig.add_trace(
            go.Histogram(
                x=X,
                marker_color=color,
                showlegend=False,
                opacity=0.7,
                name="Histogram",
            ),
            row=i,
            col=2,
        )

        (osm, osr), (slope, intercept, _) = scipy.stats.probplot(
            X, dist="norm", plot=None
        )

        qq_trace_points = go.Scatter(
            x=osm,
            y=osr,
            mode="markers",
            name="QQ-Plot",
            marker=dict(color="#1f77b4"),
            showlegend=False,
        )
        fig.add_trace(qq_trace_points, row=i, col=3)

        x_line = np.array([osm.min(), osm.max()])
        y_line = slope * x_line + intercept

        qq_trace_line = go.Scatter(
            x=x_line,
            y=y_line,
            mode="lines",
            name="Theoretical Line (Normal)",
            line=dict(color="red", dash="dash"),
            opacity=0.7,
            showlegend=False,
        )
        fig.add_trace(qq_trace_line, row=i, col=3)

    ljung_box_text = "<br>".join(
        [f"<b>{k}</b>: {v}" for k, v in ljung_box_results.items()]
    )
    arch_results_text = "<br>".join(
        [f"<b>{k}</b>: {v}" for k, v in arch_results.items()]
    )

    fig.add_trace(
        go.Scatter(
            x=[0],
            y=[0],
            text=[ljung_box_text],
            mode="text",
            showlegend=False,
            name="Summary",
            hoverinfo="skip",
        ),
        row=N + 1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=[0],
            y=[0],
            text=[arch_results_text],
            mode="text",
            showlegend=False,
            name="Summary",
            hoverinfo="skip",
        ),
        row=N + 1,
        col=2,
    )

    for row in range(1, N + 1):
        fig.update_xaxes(title_text="Index", row=row, col=1)
        fig.update_yaxes(title_text="Value", row=row, col=1)
        fig.update_xaxes(title_text="Residual", row=row, col=2)
        fig.update_yaxes(title_text="Frequency", row=row, col=2)
        fig.update_xaxes(title_text="Theorical Quantiles", row=row, col=3)
        fig.update_yaxes(title_text="Ordered Values", row=row, col=3)

    fig.update_xaxes(visible=False, row=N + 1, col=1)
    fig.update_yaxes(visible=False, row=N + 1, col=1)
    fig.update_xaxes(visible=False, row=N + 1, col=2)
    fig.update_yaxes(visible=False, row=N + 1, col=2)

    fig.update_layout(
        title="Histogram/QQ-Plot with Ljung-Box & LM-ARCH Summary",
        height=height,
        width=width,
        showlegend=False,
    )

    return fig.show(fig_type)


@requires_extra("plots")
def pami(
    X: Union[np.ndarray, List[float], pd.Series],
    nlags: int = 30,
    m: int = 3,
    delay: int = 1,
    normalize: bool = False,
    fig_type: Optional[str] = None,
    height: int = 600,
    width: int = 800,
):
    """
    Plots the Permutation Auto-Mutual Information (PAMI) for lags from 1 to nlags.

    This function calculates and visualizes the PAMI values across different time lags
    to help identify optimal lag values for time series analysis. PAMI measures the
    mutual information between a time series and its lagged versions using permutation
    patterns, which is useful for nonlinear time series analysis.

    Parameters
    ----------
    X : np.ndarray, list of float, or pd.Series
        Time series data to analyze.
    nlags : int, default=10
        Maximum lag to compute PAMI.
        Default is 30 or half the length of the series, whichever is smaller.

    m : int, default=3
        Embedding dimension for permutation patterns. Typically between 3-7.
    delay : int, default=1
        Embedding delay for time series reconstruction.
    normalize : bool, default=False
        Whether to normalize PAMI values to [0,1] range.
    fig_type : str, optional
        Plotly figure output type. Passed to `fig.show()`.
        E.g.: 'json', 'html', 'notebook'.
    height : int, default=600
        Figure height in pixels.
    width : int, default=800
        Figure width in pixels.

    Returns
    -------
    plotly.graph_objects.Figure or None
        Returns the Plotly Figure object if `fig_type` is `None` or the result
        of the `fig.show(fig_type)` call.

    Notes
    -----
    The plot includes:
    - Bar chart showing PAMI values for each lag
    - Confidence band at ±1.96/√N level (gray dashed line)
    - Local minima markers (red circles) indicating potential optimal lags

    Local minima in PAMI often correspond to meaningful time delays in the
    underlying dynamical system and can be used for lag selection in forecasting
    or embedding dimension analysis.
    """
    from scipy.signal import argrelextrema
    import plotly.graph_objects as go
    from tinyshift.series import permutation_auto_mutual_information

    nlags = min(nlags, (len(X) // 2) - 1)
    lags = np.arange(1, nlags + 1)
    pami_values = np.array(
        [
            permutation_auto_mutual_information(
                X, tau=lag, m=m, delay=delay, normalize=normalize
            )
            for lag in lags
        ]
    )
    local_minima = argrelextrema(pami_values, np.less)[0]
    offset = 0.01 * np.max(pami_values)

    min_lag = lags[local_minima]
    min_value = pami_values[local_minima]
    N = len(X)
    conf = 1.96 / np.sqrt(N)

    pami_bar = go.Bar(
        x=lags,
        y=pami_values,
        marker_color="#1f77b4",
        name="PAMI",
    )
    band_upper = go.Scatter(
        x=lags,
        y=[conf] * len(lags),
        mode="lines",
        line=dict(color="#949494", dash="dash"),
        showlegend=False,
        name="Confidence Band",
        hoverinfo="skip",
    )
    min_marker = go.Scatter(
        x=min_lag,
        y=min_value + offset,
        mode="markers",
        marker=dict(color="#d62728", size=4, symbol="circle"),
        name="Local Minima",
        hoverinfo="skip",
        showlegend=True,
    )

    fig = go.Figure([pami_bar, band_upper, min_marker])
    fig.update_layout(
        title=f"Permutation Auto-Mutual Information (PAMI) by Lag",
        xaxis_title="Lag",
        yaxis_title="PAMI",
        hovermode="x",
        height=height,
        width=width,
    )
    return fig.show(fig_type)


@requires_extra("plots")
def forest_plot(
    df: pd.DataFrame,
    feature: str,
    group_col: str,
    confidence: float = 0.95,
    fig_type: Optional[str] = None,
    height: int = 500,
    width: int = 700,
) -> go.Figure:
    """
    Creates a forest-style plot of group means with their confidence intervals.

    This function computes group means and Student's t-based confidence intervals
    for a numeric `feature` grouped by `group_col`, and returns an interactive
    Plotly figure in forest-plot style (mean points with horizontal error bars).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the data to analyze.
    feature : str
        Name of the numeric column to summarize (must exist in `df`).
    group_col : str
        Name of the categorical column used to group the data (must exist in `df`).
    confidence : float, default=0.95
        Confidence level for the intervals (between 0 and 1).

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive Plotly figure showing group means and confidence intervals in
        a forest-plot layout. Hover shows mean and CI bounds.

    Raises
    ------
    ValueError
        If confidence is not between 0 and 1 or if `df` is not a DataFrame.
    KeyError
        If `feature` or `group_col` are not columns in `df`.
    """
    import scipy.stats
    import plotly.express as px

    def t_confidence_interval(
        data: np.ndarray,
        confidence: float = 0.95,
    ) -> tuple:
        """
        Calculates the confidence interval (lower, upper) for the mean
        using the Student's t-distribution.
        """

        if isinstance(data, pd.Series):
            data = data.to_numpy()

        if len(data) < 2:
            return (np.nan, np.nan)

        return scipy.stats.t.interval(
            confidence, len(data) - 1, loc=np.mean(data), scale=scipy.stats.sem(data)
        )

    stats = df.groupby(group_col)[feature].agg(["mean"]).reset_index()

    intervals = df.groupby(group_col)[feature].apply(
        t_confidence_interval, confidence=confidence
    )
    intervals_df = intervals.apply(pd.Series).rename(
        columns={0: "ci_lower", 1: "ci_upper"}
    )
    intervals_df = intervals_df.reset_index()

    plot_data = pd.merge(stats, intervals_df, on=group_col)

    plot_data["error_minus"] = plot_data["mean"] - plot_data["ci_lower"]
    plot_data["error_plus"] = plot_data["ci_upper"] - plot_data["mean"]

    fig = px.scatter(
        plot_data,
        x=group_col,
        y="mean",
        error_y="error_plus",
        error_y_minus="error_minus",
        title=f"Mean and {int(confidence*100)}% Confidence Interval of {feature} by {group_col}",
        labels={"mean": f"Mean of {feature}", group_col: group_col},
    )

    fig.update_traces(
        marker_color="#2E86AB",
        name="Mean",
        hovertemplate=f"<b>{group_col}:</b> %{{x}}<br><b>Mean:</b> %{{y:.2f}}<br><b>CI:</b> [%{{customdata[0]:.2f}}, %{{customdata[1]:.2f}}] <extra></extra>",
        customdata=plot_data[["ci_lower", "ci_upper"]].values,
    )

    fig.update_layout(
        title_x=0.5,
        xaxis=dict(
            title_font=dict(size=14), tickfont=dict(size=12), gridcolor="lightgray"
        ),
        yaxis=dict(
            title_font=dict(size=14), tickfont=dict(size=12), gridcolor="lightgray"
        ),
        hovermode="x unified",
        height=height,
        width=width,
    )

    return fig.show(fig_type)
