# Copyright (c) 2024-2026 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License

from typing import List, Optional, Union

import numpy as np
import pandas as pd

from tinyshift.forecasting.dmstl.utils import extract_mstl_components, seasonal_strength
from tinyshift.series import harmonic_significance, trend_significance
from tinyshift.utils.imports import requires_extra

SeriesLike = Union[np.ndarray, List[float], pd.Series]


class MSTLDiagnostics:
    """
    Fit and inspect an MSTL decomposition for one time series.

    The decomposition is computed once by :meth:`fit` and retained for
    visualization and secondary diagnostics. The fitted components can be
    inspected directly or passed to stationarity and residual analyses without
    fitting MSTL again.

    Parameters
    ----------
    periods : int or list of int
        Seasonal period or periods used by MSTL. For example, ``7`` represents
        weekly seasonality and ``[7, 365]`` represents weekly and yearly
        seasonality for daily observations.
    nlags : int, default=10
        Maximum number of lags used by the decomposition-level Ljung-Box test.
        The effective value is limited to one fifth of the fitted series.

    Attributes
    ----------
    periods_ : list of int
        Normalized seasonal periods created by :meth:`fit`.
    observed_ : pandas.Series
        Numeric input series used for fitting, preserving the original index
        when the input is a pandas Series.
    model_ : statsmodels.tsa.seasonal.MSTL
        Fitted MSTL model object.
    result_ : statsmodels.tsa.seasonal.DecomposeResult
        Result returned by the MSTL fit.
    components_ : pandas.DataFrame
        Observed data, trend, one seasonal column per period, and residuals.
    statistics_ : pandas.DataFrame
        Trend, seasonal-strength, seasonal-significance, and residual
        Ljung-Box statistics indexed by metric name.

    Examples
    --------
    Fit a decomposition and reuse its components for diagnostics:

    >>> diagnostics = MSTLDiagnostics(periods=[7, 365]).fit(time_series)
    >>> diagnostics.components_.columns.tolist()
    ['data', 'trend', 'seasonal_7', 'seasonal_365', 'resid']
    >>> diagnostics.summary()
    >>> diagnostics.plot()
    >>> diagnostics.stationarity(columns=["data", "trend", "resid"])
    >>> diagnostics.residuals(nlags=20)

    Notes
    -----
    The class expects regularly sampled observations. Seasonal periods are
    expressed as numbers of observations rather than calendar units.

    See Also
    --------
    stationarity_analysis :
        Plot stationarity and autocorrelation diagnostics for fitted components.
    residual_analysis :
        Plot distribution and autocorrelation diagnostics for residuals.
    """

    def __init__(self, periods: Union[int, List[int]], nlags: int = 10) -> None:
        self.periods = periods
        self.nlags = nlags

    def fit(self, X: SeriesLike) -> "MSTLDiagnostics":
        """Fit MSTL and calculate decomposition-level statistics."""
        X_series = self._prepare_series(X)
        self.periods_ = self._normalize_periods()
        self.observed_ = X_series
        self.nlags_ = max(1, min(self.nlags, len(X_series) // 5))
        self.model_, self.result_ = self._fit_mstl(X_series)
        self.components_ = extract_mstl_components(self.result_, self.periods_)
        self.statistics_ = self._calculate_statistics()
        return self

    def _prepare_series(self, X: SeriesLike) -> pd.Series:
        if isinstance(X, pd.Series):
            return X.astype(np.float64)
        return pd.Series(np.asarray(X, dtype=np.float64))

    def _normalize_periods(self) -> List[int]:
        periods = (
            [self.periods] if isinstance(self.periods, int) else list(self.periods)
        )
        if not periods or any(not isinstance(period, int) for period in periods):
            raise ValueError("periods must be an integer or a list of integers.")
        return periods

    def _fit_mstl(self, X: pd.Series):
        from statsmodels.tsa.seasonal import MSTL

        model = MSTL(X, periods=self.periods_)
        return model, model.fit()

    def _calculate_statistics(self) -> pd.DataFrame:
        from statsmodels.stats.diagnostic import acorr_ljungbox

        _, r_squared, p_value_trend = trend_significance(self.observed_.values)
        ljung_box = acorr_ljungbox(
            self.components_["resid"].dropna(), lags=[self.nlags_]
        ).iloc[0]
        rows = [
            {
                "metric": "trend",
                "statistic": float(r_squared),
                "p_value": float(p_value_trend),
                "strength": np.nan,
            },
            {
                "metric": "residual_ljung_box",
                "statistic": float(ljung_box["lb_stat"]),
                "p_value": float(ljung_box["lb_pvalue"]),
                "strength": np.nan,
            },
        ]
        y_detrended = self.observed_.values - self.components_["trend"].values
        rows.extend(self._seasonality_statistics(y_detrended))
        return pd.DataFrame(rows).set_index("metric")

    def _seasonality_statistics(self, y_detrended: np.ndarray) -> List[dict]:
        rows = []
        for period in self.periods_:
            seasonal_column = f"seasonal_{period}"
            f_stat, p_value = harmonic_significance(y_detrended, period=period)
            rows.append(
                {
                    "metric": seasonal_column,
                    "statistic": float(f_stat),
                    "p_value": float(p_value),
                    "strength": seasonal_strength(
                        self.components_[seasonal_column].values,
                        self.components_["resid"].values,
                    ),
                }
            )
        return rows

    def _require_fitted(self) -> None:
        if not hasattr(self, "components_"):
            raise RuntimeError("MSTLDiagnostics must be fitted before use.")

    def summary(self) -> pd.DataFrame:
        """Return trend, seasonality and residual statistics."""
        self._require_fitted()
        return self.statistics_.copy()

    @requires_extra("plot")
    def plot(
        self,
        height: int = 1200,
        width: int = 1300,
        fig_type: Optional[str] = None,
    ):
        """Plot the fitted MSTL components and their statistical summary."""
        self._require_fitted()
        import plotly.express as px
        import plotly.graph_objs as go
        import plotly.subplots as sp

        component_columns = list(self.components_.columns)
        subplot_titles = [
            column.capitalize().replace("_", " ") for column in component_columns
        ] + ["Summary"]
        fig = sp.make_subplots(
            rows=len(subplot_titles), cols=1, subplot_titles=subplot_titles
        )
        self._add_component_traces(fig, go, px.colors.qualitative.T10)

        summary_row = len(subplot_titles)
        self._add_summary_trace(fig, go, summary_row)
        self._configure_figure(fig, summary_row, height, width)

        if fig_type is None:
            return fig
        return fig.show(fig_type)

    def _summary_lines(self) -> List[str]:
        lines = []
        for metric, values in self.statistics_.iterrows():
            if metric == "trend":
                label = f"Trend R²={values['statistic']:.4f}, p={values['p_value']:.4f}"
            elif metric == "residual_ljung_box":
                label = f"Ljung-Box stat={values['statistic']:.4f}, p={values['p_value']:.4f}"
            else:
                label = (
                    f"{metric}: strength={values['strength']:.4f}, "
                    f"F-test={values['statistic']:.4f}, p={values['p_value']:.4f}"
                )
            lines.append(label)
        return lines

    def _add_component_traces(self, fig, go, colors: List[str]) -> None:
        for row, column in enumerate(self.components_.columns, start=1):
            fig.add_trace(
                go.Scatter(
                    x=self.components_.index,
                    y=self.components_[column],
                    mode="lines",
                    hovertemplate=f"{column.capitalize()}: %{{y}}<extra></extra>",
                    line=dict(color=colors[(row - 1) % len(colors)]),
                    showlegend=False,
                ),
                row=row,
                col=1,
            )

    def _add_summary_trace(self, fig, go, row: int) -> None:
        fig.add_trace(
            go.Scatter(
                x=[0],
                y=[0],
                text=["<br>".join(self._summary_lines())],
                mode="text",
                showlegend=False,
            ),
            row=row,
            col=1,
        )

    def _configure_figure(self, fig, summary_row: int, height: int, width: int) -> None:
        fig.update_xaxes(visible=False, row=summary_row, col=1)
        fig.update_yaxes(visible=False, row=summary_row, col=1)
        fig.update_layout(
            title="Seasonal Decomposition (MSTL)",
            height=height,
            width=width,
            showlegend=False,
            hovermode="x",
        )

    def stationarity(self, columns: Optional[List[str]] = None, **kwargs):
        """Run stationarity diagnostics on selected fitted components."""
        self._require_fitted()
        from tinyshift.plot.diagnostic import stationarity_analysis

        columns = columns or ["data", "trend", "resid"]
        return stationarity_analysis(self.components_[columns], **kwargs)

    def residuals(self, **kwargs):
        """Run residual diagnostics on the fitted MSTL residual component."""
        self._require_fitted()
        from tinyshift.plot.diagnostic import residual_analysis

        return residual_analysis(self.components_["resid"], **kwargs)
