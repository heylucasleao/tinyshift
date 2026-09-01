"""Panel-aligned facades over TSF predictive distributions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .distribution import PredictiveDistribution


class _PanelPredictiveForecast:
    """Self-contained, row-aligned probabilistic forecast."""

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
        """Return the point-forecast panel without distributional columns."""
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

    def cdf(self, values) -> pd.DataFrame:
        """Evaluate CDF values on the forecast panel."""
        return self._apply("cdf", values, "cdf")

    def ppf(self, quantiles) -> pd.DataFrame:
        """Evaluate predictive quantiles on the forecast panel."""
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
        """Return a central predictive interval on the forecast panel."""
        bounds = np.asarray(self._distribution.interval(coverage))
        level = self._label(100.0 * float(coverage))
        result = self.to_frame()
        result[f"{self.model}-lo-{level}"] = bounds[:, 0]
        result[f"{self.model}-hi-{level}"] = bounds[:, 1]
        return result


class _DiscretePanelPredictiveForecast(_PanelPredictiveForecast):
    """Panel forecast that additionally exposes probability masses."""

    def pmf(self, values) -> pd.DataFrame:
        """Evaluate PMF values on the forecast panel."""
        return self._apply("pmf", values, "pmf")
