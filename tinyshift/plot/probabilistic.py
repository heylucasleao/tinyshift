"""Calibration plots for panel probabilistic forecasts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tinyshift.forecasting.probabilistic.distribution import (
    DiscretePredictiveDistribution,
)
from tinyshift.utils.imports import requires_extra

__all__ = ["ProbabilisticCalibrationPlot"]


class ProbabilisticCalibrationPlot:
    """Visual diagnostics for a row-aligned probabilistic forecast.

    The PIT histogram assesses distributional calibration, while the PIT ACF
    checks whether calibration errors retain temporal dependence.  The event
    calibration curve compares forecast probabilities of ``Y > threshold``
    with observed exceedance frequencies.

    Discrete distributions use the randomized PIT; continuous distributions
    use ``F(y)`` directly.
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
        result = np.asarray(
            self.forecast.distribution.cdf(np.asarray(values)[:, None]), dtype=float
        )
        return result.reshape(-1)

    def _compute_pit(
        self,
    ) -> np.ndarray:
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
        if isinstance(n_bins, bool) or not isinstance(n_bins, int) or n_bins < 2:
            raise ValueError("n_bins must be an integer greater than or equal to 2.")

    def _acf(
        self,
        max_lag: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        """Return a PIT histogram with the uniform reference frequency."""
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
        """Return the pooled within-series autocorrelation of the PIT."""
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
        """Return the reliability curve for the event ``Y > threshold``."""
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
        """Return PIT histogram, PIT ACF, and event calibration in one figure."""
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
