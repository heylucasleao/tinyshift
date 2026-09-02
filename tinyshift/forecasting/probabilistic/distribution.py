"""Predictive distribution objects used by two-stage forecasters."""

from abc import ABC, abstractmethod

import numpy as np
from scipy.special import gammaln
from scipy.stats import gamma, lognorm, nbinom, weibull_min


class PredictiveDistribution(ABC):
    """A row-aligned batch of one-dimensional predictive distributions."""

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of distributions in the batch."""

    @abstractmethod
    def cdf(self, values):
        """Evaluate the cumulative distribution function."""

    @abstractmethod
    def ppf(self, quantiles):
        """Evaluate the generalized inverse CDF."""

    def interval(self, coverage: float = 0.95) -> np.ndarray:
        if not np.isfinite(coverage) or not 0.0 < coverage < 1.0:
            raise ValueError("coverage must be finite and strictly between 0 and 1.")
        alpha = 1.0 - float(coverage)
        levels = np.broadcast_to(
            np.array([alpha / 2.0, 1.0 - alpha / 2.0]), (len(self), 2)
        )
        return np.asarray(self.ppf(levels))


class DiscretePredictiveDistribution(PredictiveDistribution):
    """Predictive distribution with ordered integer support."""

    def pmf(self, values) -> np.ndarray:
        values = np.asarray(values)
        if not np.all(np.isfinite(values)) or np.any(values != np.floor(values)):
            raise ValueError("pmf values must be finite integers.")
        return np.asarray(self.cdf(values)) - np.asarray(self.cdf(values - 1))


class _ParametricDistribution(PredictiveDistribution):
    def __init__(self, means, dispersions):
        self.means = np.asarray(means, dtype=float)
        self.dispersions = np.asarray(dispersions, dtype=float)
        if self.means.ndim != 1 or self.dispersions.shape != self.means.shape:
            raise ValueError(
                "means and dispersions must be aligned one-dimensional arrays."
            )
        if not np.all(np.isfinite(self.means)) or not np.all(
            np.isfinite(self.dispersions)
        ):
            raise ValueError("distribution parameters must be finite.")
        if np.any(self.means <= 0.0) or np.any(self.dispersions <= 0.0):
            raise ValueError("distribution parameters must be strictly positive.")

    def __len__(self) -> int:
        return self.means.size

    @staticmethod
    def _validate_quantiles(quantiles) -> None:
        """Require probability levels in the closed unit interval."""
        if np.any((quantiles < 0.0) | (quantiles > 1.0)):
            raise ValueError("quantiles must lie in [0, 1].")

    @staticmethod
    def _finalize(result, squeeze: bool):
        """Collapse a single row-wise output column when requested."""
        return result[:, 0] if squeeze and result.ndim == 2 else result

    def _align(self, values, name: str):
        array = np.asarray(values, dtype=float)
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite values.")
        if array.ndim == 0:
            return array, self.means, self.dispersions, True
        if array.ndim == 1:
            return (
                array[None, :],
                self.means[:, None],
                self.dispersions[:, None],
                False,
            )
        if array.ndim == 2 and array.shape[0] == len(self):
            return (
                array,
                self.means[:, None],
                self.dispersions[:, None],
                array.shape[1] == 1,
            )
        raise ValueError(
            f"{name} must be a scalar, a one-dimensional grid, or a matrix "
            f"with {len(self)} rows."
        )


class NegativeBinomialPredictiveDistribution(
    _ParametricDistribution, DiscretePredictiveDistribution
):
    """Negative Binomial batches parameterized by conditional mean and size."""

    def cdf(self, values):
        values, means, sizes, squeeze = self._align(values, "values")
        probabilities = sizes / (sizes + means)
        result = nbinom.cdf(np.floor(values), sizes, probabilities)
        return self._finalize(result, squeeze)

    def ppf(self, quantiles):
        quantiles, means, sizes, squeeze = self._align(quantiles, "quantiles")
        self._validate_quantiles(quantiles)
        probabilities = sizes / (sizes + means)
        projected = np.ceil(nbinom.ppf(quantiles, sizes, probabilities))
        projected = np.where(quantiles == 0.0, 0.0, projected)
        if np.all(np.isfinite(projected)):
            projected = projected.astype(int)
        return self._finalize(projected, squeeze)


class GammaPredictiveDistribution(_ParametricDistribution):
    """Gamma batches parameterized by conditional mean and shape."""

    def cdf(self, values):
        values, means, shapes, squeeze = self._align(values, "values")
        result = gamma.cdf(values, a=shapes, scale=means / shapes)
        return self._finalize(result, squeeze)

    def ppf(self, quantiles):
        quantiles, means, shapes, squeeze = self._align(quantiles, "quantiles")
        self._validate_quantiles(quantiles)
        result = gamma.ppf(quantiles, a=shapes, scale=means / shapes)
        return self._finalize(result, squeeze)


class LogNormalPredictiveDistribution(_ParametricDistribution):
    """Lognormal batches parameterized by conditional mean and log-scale."""

    @staticmethod
    def _scales(means, sigmas):
        return means * np.exp(-0.5 * sigmas**2)

    def cdf(self, values):
        values, means, sigmas, squeeze = self._align(values, "values")
        result = lognorm.cdf(values, s=sigmas, scale=self._scales(means, sigmas))
        return self._finalize(result, squeeze)

    def ppf(self, quantiles):
        quantiles, means, sigmas, squeeze = self._align(quantiles, "quantiles")
        self._validate_quantiles(quantiles)
        result = lognorm.ppf(quantiles, s=sigmas, scale=self._scales(means, sigmas))
        return self._finalize(result, squeeze)


class WeibullPredictiveDistribution(_ParametricDistribution):
    """Weibull batches parameterized by conditional mean and shape."""

    @staticmethod
    def _scales(means, shapes):
        return means / np.exp(gammaln(1.0 + 1.0 / shapes))

    def cdf(self, values):
        values, means, shapes, squeeze = self._align(values, "values")
        result = weibull_min.cdf(values, c=shapes, scale=self._scales(means, shapes))
        return self._finalize(result, squeeze)

    def ppf(self, quantiles):
        quantiles, means, shapes, squeeze = self._align(quantiles, "quantiles")
        self._validate_quantiles(quantiles)
        result = weibull_min.ppf(quantiles, c=shapes, scale=self._scales(means, shapes))
        return self._finalize(result, squeeze)
