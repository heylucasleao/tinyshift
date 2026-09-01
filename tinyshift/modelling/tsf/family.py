"""Distribution-family strategies for two-stage forecasting."""

from abc import ABC, abstractmethod

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import gamma, nbinom
from sklearn.base import BaseEstimator

from .distribution import (
    GammaPredictiveDistribution,
    NegativeBinomialPredictiveDistribution,
    PredictiveDistribution,
)


class DistributionFamily(BaseEstimator, ABC):
    """Calibrate one dispersion parameter around predicted conditional means."""

    is_discrete = False
    parameter_column = "dispersion"

    @abstractmethod
    def validate_target(self, y: np.ndarray) -> None:
        """Validate whether observations belong to this family's support."""

    @abstractmethod
    def negative_log_likelihood(
        self, dispersion: float, y: np.ndarray, means: np.ndarray
    ) -> float:
        """Return the negative log likelihood for one dispersion value."""

    @abstractmethod
    def distribution(self, means, dispersions) -> PredictiveDistribution:
        """Construct a batch distribution."""

    def _calibration_data(
        self, y: np.ndarray, means: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Validate and return aligned dispersion-calibration arrays."""
        y = np.asarray(y, dtype=float)
        means = np.asarray(means, dtype=float)
        if y.shape != means.shape:
            raise ValueError(
                "observations and predicted means must have the same shape."
            )
        if not np.all(np.isfinite(means)):
            raise ValueError("Predicted mean values must be finite.")
        self.validate_target(y)
        return y, means

    @staticmethod
    def _fitted_dispersion(result) -> float:
        """Validate an optimizer result and return its fitted value."""
        if (
            not result.success
            or not np.isfinite(result.fun)
            or not np.isfinite(result.x)
        ):
            raise RuntimeError(f"Dispersion optimization failed: {result.message}")
        return float(result.x)

    def fit_dispersion(self, y: np.ndarray, means: np.ndarray) -> float:
        """Estimate dispersion by bounded maximum likelihood."""
        y, means = self._calibration_data(y, means)
        result = minimize_scalar(
            lambda value: self.negative_log_likelihood(value, y, means),
            bounds=self.dispersion_bounds,
            method="bounded",
        )
        return self._fitted_dispersion(result)


class NegativeBinomialFamily(DistributionFamily):
    """Negative Binomial family for non-negative integer counts."""

    is_discrete = True
    parameter_column = "r_dispersion"

    def __init__(self, min_size: float = 1e-3, max_size: float = 50.0):
        self.min_size = min_size
        self.max_size = max_size

    @property
    def dispersion_bounds(self):
        return (self.min_size, self.max_size)

    def validate_target(self, y: np.ndarray) -> None:
        if not np.all(np.isfinite(y)):
            raise ValueError("Target values must be finite.")
        if np.any(y < 0):
            raise ValueError("Target values must be non-negative.")
        if np.any(y != np.floor(y)):
            raise ValueError(
                "Target values must be integer counts for the Negative Binomial model."
            )

    def negative_log_likelihood(self, dispersion, y, means) -> float:
        if not np.isfinite(dispersion) or dispersion <= 0:
            return 1e10
        means = np.maximum(means, 1e-6)
        probability = dispersion / (dispersion + means)
        log_probability = nbinom.logpmf(y, dispersion, probability)
        log_probability = np.where(np.isneginf(log_probability), -1e2, log_probability)
        return float(-np.sum(log_probability))

    def distribution(self, means, dispersions):
        return NegativeBinomialPredictiveDistribution(means, dispersions)


class GammaFamily(DistributionFamily):
    """Gamma family for strictly positive continuous targets."""

    parameter_column = "shape_dispersion"

    def __init__(self, min_shape: float = 1e-3, max_shape: float = 1e4):
        self.min_shape = min_shape
        self.max_shape = max_shape

    @property
    def dispersion_bounds(self):
        return (self.min_shape, self.max_shape)

    def validate_target(self, y: np.ndarray) -> None:
        if not np.all(np.isfinite(y)):
            raise ValueError("Target values must be finite.")
        if np.any(y <= 0.0):
            raise ValueError(
                "Target values must be strictly positive for the Gamma model."
            )

    def negative_log_likelihood(self, dispersion, y, means) -> float:
        if not np.isfinite(dispersion) or dispersion <= 0:
            return 1e10
        means = np.maximum(means, 1e-6)
        log_density = gamma.logpdf(y, a=dispersion, scale=means / dispersion)
        if not np.all(np.isfinite(log_density)):
            return 1e10
        return float(-np.sum(log_density))

    def distribution(self, means, dispersions):
        return GammaPredictiveDistribution(means, dispersions)
