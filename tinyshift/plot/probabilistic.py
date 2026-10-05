# Copyright (c) 2024-2026 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License


from __future__ import annotations

import numpy as np
import pandas as pd

from tinyshift.forecasting.probabilistic.distribution import (
    DiscretePredictiveDistribution,
)
from tinyshift.utils.imports import requires_extra

__all__ = ["ProbabilisticCalibrationPlot"]


class ProbabilisticCalibrationPlot:
    """
    Inspect calibration for a row-aligned probabilistic panel forecast.

    The PIT histogram assesses distributional calibration, while the PIT ACF
    checks whether calibration errors retain temporal dependence. The event
    calibration curve compares forecast probabilities of ``Y > threshold``
    with observed exceedance frequencies.

    Parameters
    ----------
    forecast : PanelPredictiveForecast
        Probabilistic forecast exposing ``distribution`` and ``to_frame``.
        Its rows must identify unique series and timestamps and must be aligned
        with the underlying batch of predictive distributions.
    y_true : pandas.DataFrame
        Observed panel containing the identifier, timestamp, and numeric target
        columns. Every forecast row must have exactly one observed target.
    target_col : str, default="y"
        Name of the observed target column in ``y_true``.
    id_col : str, default="unique_id"
        Name of the series identifier column in both inputs.
    time_col : str, default="ds"
        Name of the timestamp column in both inputs.
    random_state : int or None, default=None
        Seed used by the randomized PIT for discrete distributions. It is
        ignored for continuous distributions.

    Attributes
    ----------
    forecast : PanelPredictiveForecast
        Forecast supplied at construction.
    pit_ : numpy.ndarray
        Row-aligned PIT values. Continuous forecasts use ``F(y)`` directly;
        discrete forecasts use the randomized PIT within each CDF jump.

    Examples
    --------
    Create the diagnostics once and reuse them in individual or combined
    plots:

    >>> diagnostics = ProbabilisticCalibrationPlot(forecast, evaluation_df)
    >>> diagnostics.pit_histogram(n_bins=10)
    >>> diagnostics.pit_acf(max_lag=20)
    >>> diagnostics.calibration_curve(threshold=15.0)
    >>> diagnostics.summary(threshold=15.0, max_lag=20)

    Notes
    -----
    The class is intended for out-of-sample forecasts. For a calibrated
    continuous predictive distribution, PIT values should be approximately
    uniform and serially independent. For a discrete predictive distribution,
    the randomized PIT is used so that the same reference remains applicable.

    The PIT ACF sorts rows by ``id_col`` and ``time_col`` and constructs lagged
    pairs within each series. It never links the final observation of one
    series to the first observation of another.

    See Also
    --------
    tinyshift.forecasting.probabilistic.TwoStageForecasterEvaluator :
        Numerical evaluation with interval scores and CRPS.
    """

    def __init__(
        self,
        forecast,
        y_true: pd.DataFrame,
        target_col: str = "y",
        id_col: str = "unique_id",
        time_col: str = "ds",
        random_state: int | None = None,
    ) -> None:
        if not hasattr(forecast, "distribution") or not hasattr(forecast, "to_frame"):
            raise TypeError("forecast must be a panel predictive forecast.")
        if not isinstance(y_true, pd.DataFrame):
            raise TypeError("y_true must be a pandas DataFrame.")

        self.forecast = forecast
        self.target_col = target_col
        self.id_col = id_col
        self.time_col = time_col
        self.random_state = random_state
        self._frame = self._align(y_true)
        self._observed = self._frame[target_col].to_numpy(dtype=float)
        self.pit_ = self._compute_pit()

    def _align(self, y_true: pd.DataFrame) -> pd.DataFrame:
        """Align observed targets to the forecast row order by panel keys."""
        keys = [self.id_col, self.time_col]
        forecast_frame = self.forecast.to_frame()
        for frame, required, name in (
            (forecast_frame, keys, "forecast"),
            (y_true, [*keys, self.target_col], "y_true"),
        ):
            missing = [column for column in required if column not in frame.columns]
            if missing:
                raise KeyError(f"Columns not found in {name}: {missing}")
            if frame.duplicated(keys).any():
                raise ValueError(f"{name} contains duplicate identifier/time rows.")

        aligned = forecast_frame[keys].merge(
            y_true[[*keys, self.target_col]],
            on=keys,
            how="left",
            validate="one_to_one",
            sort=False,
        )
        if aligned[self.target_col].isna().any():
            raise ValueError("y_true must contain a target for every forecast row.")
        try:
            observed = aligned[self.target_col].to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("y_true must contain numeric target values.") from exc
        if not np.all(np.isfinite(observed)):
            raise ValueError("y_true must contain only finite target values.")
        if len(aligned) != len(self.forecast.distribution):
            raise ValueError("Forecast and observed targets must have equal length.")
        return aligned

    def _row_cdf(
        self,
        values: np.ndarray,
    ) -> np.ndarray:
        """Evaluate one CDF value for each row-aligned distribution."""
        result = np.asarray(
            self.forecast.distribution.cdf(np.asarray(values)[:, None]), dtype=float
        )
        return result.reshape(-1)

    def _compute_pit(
        self,
    ) -> np.ndarray:
        """Compute continuous or randomized discrete row-wise PIT values."""
        distribution = self.forecast.distribution
        upper = self._row_cdf(self._observed)
        if not isinstance(distribution, DiscretePredictiveDistribution):
            return upper

        if np.any(self._observed != np.floor(self._observed)):
            raise ValueError(
                "Observed values must be integers for a discrete forecast."
            )
        lower = self._row_cdf(self._observed - 1.0)
        random = np.random.default_rng(self.random_state).uniform(size=len(upper))
        return lower + random * (upper - lower)

    @staticmethod
    def _validate_bins(
        n_bins: int,
    ) -> None:
        """Require at least two histogram or probability bins."""
        if isinstance(n_bins, bool) or not isinstance(n_bins, int) or n_bins < 2:
            raise ValueError("n_bins must be an integer greater than or equal to 2.")

    def _acf(
        self,
        max_lag: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate pooled correlations from within-series lagged PIT pairs."""
        if isinstance(max_lag, bool) or not isinstance(max_lag, int) or max_lag < 1:
            raise ValueError("max_lag must be a positive integer.")

        work = self._frame[[self.id_col, self.time_col]].copy()
        work["pit"] = self.pit_
        work = work.sort_values([self.id_col, self.time_col], kind="stable")
        correlations = [1.0]
        pair_counts = [len(work)]
        for lag in range(1, max_lag + 1):
            previous = work.groupby(self.id_col, observed=True)["pit"].shift(lag)
            valid = previous.notna()
            pair_counts.append(int(valid.sum()))
            if valid.sum() < 2:
                correlations.append(np.nan)
                continue
            x = previous[valid].to_numpy(dtype=float)
            y = work.loc[valid, "pit"].to_numpy(dtype=float)
            if np.std(x) == 0.0 or np.std(y) == 0.0:
                correlations.append(np.nan)
            else:
                correlations.append(float(np.corrcoef(x, y)[0, 1]))
        return (
            np.arange(max_lag + 1),
            np.asarray(correlations),
            np.asarray(pair_counts),
        )

    def _calibration_data(
        self,
        threshold: float,
        n_bins: int,
    ) -> pd.DataFrame:
        """Aggregate exceedance probabilities and outcomes into fixed bins."""
        self._validate_bins(n_bins)
        if not np.isscalar(threshold) or not np.isfinite(threshold):
            raise ValueError("threshold must be a finite scalar.")
        probabilities = np.asarray(
            self.forecast.distribution.sf(float(threshold)), dtype=float
        ).reshape(-1)
        data = pd.DataFrame(
            {
                "forecast_probability": probabilities,
                "observed_event": self._observed > float(threshold),
            }
        )
        data["bin"] = pd.cut(
            data["forecast_probability"],
            bins=np.linspace(0.0, 1.0, n_bins + 1),
            include_lowest=True,
        )
        result = (
            data.groupby("bin", observed=True)
            .agg(
                forecast_probability=("forecast_probability", "mean"),
                observed_frequency=("observed_event", "mean"),
                count=("observed_event", "size"),
            )
            .reset_index(drop=True)
        )
        standard_error = np.sqrt(
            result["observed_frequency"]
            * (1.0 - result["observed_frequency"])
            / result["count"]
        )
        result["ci95"] = 1.96 * standard_error
        return result

    @requires_extra("plot")
    def pit_histogram(
        self,
        n_bins: int = 10,
        width=600,
        height=400,
    ):
        """
        Plot the PIT histogram and its uniform reference frequency.

        Parameters
        ----------
        n_bins : int, default=10
            Number of equal-width bins over ``[0, 1]``. Must be at least 2.
        width : int, default=600
            Figure width in pixels.
        height : int, default=400
            Figure height in pixels.

        Returns
        -------
        plotly.graph_objects.Figure
            Bar chart of PIT counts. The dashed horizontal line shows the
            expected count per bin under a uniform distribution.

        Raises
        ------
        ValueError
            If ``n_bins`` is not an integer greater than or equal to 2.

        Notes
        -----
        A U-shaped histogram suggests underdispersion, while concentration near
        0.5 suggests overdispersion. Asymmetry can indicate forecast bias.
        """
        import plotly.graph_objects as go

        self._validate_bins(n_bins)
        counts, edges = np.histogram(self.pit_, bins=n_bins, range=(0.0, 1.0))
        fig = go.Figure(
            go.Bar(
                x=(edges[:-1] + edges[1:]) / 2.0,
                y=counts,
                width=np.diff(edges),
                name="PIT",
                marker_color="#8ecae6",
                hovertemplate="PIT: %{x:.2f}<br>Count: %{y}<extra></extra>",
            )
        )
        fig.add_hline(y=len(self.pit_) / n_bins, line_dash="dash", line_color="gray")
        fig.update_layout(
            title="PIT Histogram",
            xaxis_title="PIT",
            yaxis_title="Frequency",
            width=width,
            height=height,
            bargap=0.03,
        )
        return fig

    @requires_extra("plot")
    def pit_acf(
        self,
        max_lag: int = 30,
        width=600,
        height=400,
    ):
        """
        Plot the pooled within-series autocorrelation of the PIT.

        Parameters
        ----------
        max_lag : int, default=30
            Largest within-series lag to display. Must be positive.
        width : int, default=600
            Figure width in pixels.
        height : int, default=400
            Figure height in pixels.

        Returns
        -------
        plotly.graph_objects.Figure
            PIT autocorrelation bars with approximate 95% reference bounds.

        Raises
        ------
        ValueError
            If ``max_lag`` is not a positive integer.

        Notes
        -----
        Lagged pairs are formed independently within each series and then
        pooled. A lag is shown as missing when fewer than two valid pairs exist
        or either side of the pooled pair has zero variance. The reference
        bounds are the approximation ``+/- 1.96 / sqrt(n)``.
        """
        import plotly.graph_objects as go

        lags, correlations, pair_counts = self._acf(max_lag)
        confidence = 1.96 / np.sqrt(len(self.pit_))
        fig = go.Figure(
            go.Bar(
                x=lags,
                y=correlations,
                name="PIT ACF",
                marker_color="#ffb703",
                customdata=pair_counts,
                hovertemplate="Lag: %{x}<br>ACF: %{y:.3f}<br>Pairs: %{customdata}<extra></extra>",
            )
        )
        fig.add_hline(y=confidence, line_dash="dash", line_color="#219ebc")
        fig.add_hline(y=-confidence, line_dash="dash", line_color="#219ebc")
        fig.update_layout(
            title="ACF of PIT",
            xaxis_title="Lag",
            yaxis_title="ACF",
            width=width,
            height=height,
            showlegend=False,
        )
        return fig

    @requires_extra("plot")
    def calibration_curve(
        self,
        threshold: float,
        n_bins: int = 10,
        width=600,
        height=400,
    ):
        """
        Plot the calibration curve for the event ``Y > threshold``.

        Parameters
        ----------
        threshold : float
            Finite threshold defining the strict exceedance event. Forecast
            probabilities are obtained from the predictive survival function.
        n_bins : int, default=10
            Number of equal-width probability bins over ``[0, 1]``. Empty bins
            are omitted. Must be at least 2.
        width : int, default=600
            Figure width in pixels.
        height : int, default=400
            Figure height in pixels.

        Returns
        -------
        plotly.graph_objects.Figure
            Reliability curve comparing mean forecast probability with the
            observed exceedance frequency in each populated bin.

        Raises
        ------
        ValueError
            If ``threshold`` is not finite or ``n_bins`` is invalid.

        Notes
        -----
        The diagonal represents perfect event calibration. Vertical error bars
        use the normal approximation to a 95% binomial confidence interval.
        For discrete targets, the event remains strictly ``Y > threshold``.
        """
        import plotly.graph_objects as go

        calibration = self._calibration_data(threshold, n_bins)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=[0.0, 1.0],
                y=[0.0, 1.0],
                mode="lines",
                name="Perfect calibration",
                line={"color": "gray", "dash": "dash"},
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=calibration["forecast_probability"],
                y=calibration["observed_frequency"],
                error_y={
                    "type": "data",
                    "array": calibration["ci95"],
                    "visible": True,
                },
                customdata=calibration["count"],
                mode="lines+markers",
                name=f"Y > {threshold:g}",
                hovertemplate=(
                    "Forecast: %{x:.3f}<br>Observed: %{y:.3f}"
                    "<br>Count: %{customdata}<extra></extra>"
                ),
            )
        )
        fig.update_layout(
            title=f"Calibration Curve: P(Y > {threshold:g})",
            xaxis_title="Forecast probability",
            yaxis_title="Observed relative frequency",
            xaxis_range=[0, 1],
            yaxis_range=[0, 1],
            width=width,
            height=height,
        )
        return fig

    @requires_extra("plot")
    def summary(
        self,
        threshold: float,
        n_bins: int = 10,
        max_lag: int = 30,
        width=1200,
        height=400,
    ):
        """
        Plot all probabilistic calibration diagnostics in one figure.

        Parameters
        ----------
        threshold : float
            Finite threshold defining the strict event ``Y > threshold`` in
            the calibration-curve panel.
        n_bins : int, default=10
            Number of bins used by the PIT histogram and calibration curve.
        max_lag : int, default=30
            Largest within-series lag displayed by the PIT ACF.
        width : int, default=1200
            Combined figure width in pixels.
        height : int, default=400
            Combined figure height in pixels.

        Returns
        -------
        plotly.graph_objects.Figure
            Three-panel figure containing the PIT histogram, PIT ACF, and
            exceedance calibration curve.

        Raises
        ------
        ValueError
            If ``threshold``, ``n_bins``, or ``max_lag`` is invalid.

        Examples
        --------
        >>> diagnostics = ProbabilisticCalibrationPlot(forecast, evaluation_df)
        >>> figure = diagnostics.summary(
        ...     threshold=15.0,
        ...     n_bins=10,
        ...     max_lag=20,
        ... )
        >>> figure.show()
        """
        from plotly.subplots import make_subplots

        histogram = self.pit_histogram(n_bins=n_bins)
        acf = self.pit_acf(max_lag=max_lag)
        curve = self.calibration_curve(threshold=threshold, n_bins=n_bins)
        fig = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=("PIT Histogram", "ACF of PIT", "Calibration Curve"),
        )
        for column, source in enumerate((histogram, acf, curve), start=1):
            for trace in source.data:
                fig.add_trace(trace, row=1, col=column)
            for shape in source.layout.shapes or ():
                fig.add_shape(shape, row=1, col=column)
        fig.update_xaxes(title_text="PIT", range=[0, 1], row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        fig.update_xaxes(title_text="Lag", row=1, col=2)
        fig.update_yaxes(title_text="ACF", row=1, col=2)
        fig.update_xaxes(title_text="Forecast probability", range=[0, 1], row=1, col=3)
        fig.update_yaxes(
            title_text="Observed relative frequency", range=[0, 1], row=1, col=3
        )
        fig.update_layout(width=width, height=height, title="Probabilistic Calibration")
        return fig
