"""Panel-aligned facades over TSF predictive distributions."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from .distribution import PredictiveDistribution

__all__ = ["DiscretePanelPredictiveForecast", "PanelPredictiveForecast"]


class PanelPredictiveForecast:
    """Panel-aligned facade over a batch of TSF predictive distributions.

    Instances are returned by
    :meth:`TwoStageForecasterWrapper.predict_distribution`; users normally do
    not construct this class directly. Every distributional method preserves
    the rows and point-forecast columns returned by :meth:`to_frame`.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        distribution: PredictiveDistribution,
        model: str,
        id_col: str,
        time_col: str,
    ):
        self._frame = frame.copy()
        self._distribution = distribution
        self.model = model
        self.id_col = id_col
        self.time_col = time_col

    def __len__(self) -> int:
        return len(self._distribution)

    def to_frame(self) -> pd.DataFrame:
        """Return the point-forecast panel without distributional columns.

        Returns
        -------
        pandas.DataFrame
            A copy of the forecast panel containing the series identifier,
            timestamp, ``lambda_t`` conditional mean, and the calibrated
            family parameter. Changing it does not mutate this forecast.
        """
        return self._frame.copy()

    @staticmethod
    def _label(value) -> str:
        return np.format_float_positional(float(value), precision=12, trim="-")

    def _apply(self, method: str, inputs, prefix: str) -> pd.DataFrame:
        inputs_array = np.asarray(inputs)
        values = np.asarray(getattr(self._distribution, method)(inputs))
        result = self.to_frame()
        if values.ndim == 1:
            if inputs_array.ndim == 0:
                column = f"{self.model}-{prefix}-{self._label(inputs_array)}"
            else:
                column = f"{self.model}-{prefix}"
            result[column] = values
            return result

        labels = np.ravel(inputs_array)
        if labels.size != values.shape[1]:
            labels = np.arange(values.shape[1])
        for index, label in enumerate(labels):
            result[f"{self.model}-{prefix}-{self._label(label)}"] = values[:, index]
        return result

    def cdf(self, values: ArrayLike) -> pd.DataFrame:
        """Evaluate the cumulative distribution function on the panel.

        Parameters
        ----------
        values : float or array-like of float
            Target values at which to evaluate each predictive CDF. A scalar
            is applied to every forecast row. A one-dimensional array defines
            a common evaluation grid. A two-dimensional array with
            ``len(self)`` rows is evaluated row-wise.

        Returns
        -------
        pandas.DataFrame
            The point-forecast panel plus the evaluated probabilities. Scalar
            and common-grid columns are named ``lambda_t-cdf-<value>``.
            Row-wise input produces one column per input column.

        Raises
        ------
        ValueError
            If a value is non-finite or the input shape is unsupported.

        Notes
        -----
        CDF values lie in ``[0, 1]`` and remain positionally aligned with
        :meth:`to_frame`.
        """
        return self._apply("cdf", values, "cdf")

    def ppf(self, quantiles: ArrayLike) -> pd.DataFrame:
        """Evaluate predictive quantiles on the forecast panel.

        Parameters
        ----------
        quantiles : float or array-like of float
            Probabilities in ``[0, 1]``. A scalar is applied to every forecast
            row. A one-dimensional array defines a common quantile grid. A
            two-dimensional array with ``len(self)`` rows supplies row-wise
            quantile levels.

        Returns
        -------
        pandas.DataFrame
            The point-forecast panel plus requested quantiles. Scalar and
            common-grid columns use percentage names such as
            ``lambda_t-q-90`` for quantile ``0.9``. Discrete forecasts return
            integer quantiles.

        Raises
        ------
        ValueError
            If a quantile is non-finite, outside ``[0, 1]``, or the input
            shape is unsupported.

        Examples
        --------
        ``forecast.ppf([0.1, 0.5, 0.9])`` returns the 10th percentile, median,
        and 90th percentile for every forecast row.
        """
        quantiles_array = np.asarray(quantiles, dtype=float)
        result = self._apply("ppf", quantiles, "q")
        if quantiles_array.ndim == 0:
            return result.rename(
                columns={
                    f"{self.model}-q-{self._label(quantiles_array)}":
                    f"{self.model}-q-{self._label(100.0 * quantiles_array)}"
                }
            )
        if quantiles_array.ndim == 1:
            return result.rename(
                columns={
                    f"{self.model}-q-{self._label(q)}":
                    f"{self.model}-q-{self._label(100.0 * q)}"
                    for q in quantiles_array
                }
            )
        return result

    def interval(self, coverage: float = 0.95) -> pd.DataFrame:
        """Return an equal-tailed central predictive interval.

        Parameters
        ----------
        coverage : float, default=0.95
            Central probability covered by the interval. It must be strictly
            between 0 and 1. For example, ``0.9`` requests a 90% interval with
            5% probability in each tail.

        Returns
        -------
        pandas.DataFrame
            The point-forecast panel plus lower and upper bounds named
            ``lambda_t-lo-<percentage>`` and ``lambda_t-hi-<percentage>``.

        Raises
        ------
        ValueError
            If ``coverage`` is not strictly between 0 and 1.
        TypeError
            If ``coverage`` is not numeric.
        """
        bounds = np.asarray(self._distribution.interval(coverage))
        level = self._label(100.0 * float(coverage))
        result = self.to_frame()
        result[f"{self.model}-lo-{level}"] = bounds[:, 0]
        result[f"{self.model}-hi-{level}"] = bounds[:, 1]
        return result


class DiscretePanelPredictiveForecast(PanelPredictiveForecast):
    """Panel forecast for integer targets, additionally exposing a PMF."""

    def pmf(self, values: ArrayLike) -> pd.DataFrame:
        """Evaluate probability masses on the discrete forecast panel.

        Parameters
        ----------
        values : int or array-like of int
            Integer support values at which to evaluate each predictive PMF.
            A scalar is applied to every forecast row. A one-dimensional array
            defines a common support grid. A two-dimensional array with
            ``len(self)`` rows is evaluated row-wise.

        Returns
        -------
        pandas.DataFrame
            The point-forecast panel plus probability masses. Scalar and
            common-grid columns are named ``lambda_t-pmf-<value>``.

        Raises
        ------
        ValueError
            If a value is non-finite or non-integer, or the input shape is
            unsupported.

        Notes
        -----
        This method exists only for discrete-family forecasts. Each mass is
        computed as ``CDF(k) - CDF(k - 1)``.
        """
        return self._apply("pmf", values, "pmf")
