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

    @property
    @abstractmethod
    def dispersion_bounds(self) -> tuple[float, float]:
        """Return the lower and upper bounds for dispersion optimization."""

    @abstractmethod
    def validate_target(self, y: np.ndarray) -> None:
        """Validate whether observations belong to this family's support."""

    @abstractmethod
    def negative_log_likelihood(
        self, dispersion: float, y: np.ndarray, conditional_means: np.ndarray
    ) -> float:
        """Return the negative log likelihood for one dispersion value."""

    @abstractmethod
    def distribution(self, conditional_means, dispersions) -> PredictiveDistribution:
        """Construct a batch distribution."""

    def _validate_calibration_data(
        self, y: np.ndarray, conditional_means: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Validate and return aligned dispersion-calibration arrays."""
        y = np.asarray(y, dtype=float)
        conditional_means = np.asarray(conditional_means, dtype=float)
        if y.shape != conditional_means.shape:
            raise ValueError(
                "observations and predicted means must have the same shape."
            )
        if not np.all(np.isfinite(conditional_means)):
            raise ValueError("Predicted mean values must be finite.")
        self.validate_target(y)
        return y, conditional_means

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

    def fit_dispersion(self, y: np.ndarray, conditional_means: np.ndarray) -> float:
        """Estimate dispersion by bounded maximum likelihood."""
        y, conditional_means = self._validate_calibration_data(y, conditional_means)
        result = minimize_scalar(
            lambda value: self.negative_log_likelihood(value, y, conditional_means),
            bounds=self.dispersion_bounds,
            method="bounded",
        )
        return self._fitted_dispersion(result)

    def fit_log_dispersion(
        self, y: np.ndarray, conditional_means: np.ndarray, epsilon: float = 0.05
    ) -> tuple[float, float, float]:
        """Estimate dispersion and the local variance of its logarithm."""
        y, conditional_means = self._validate_calibration_data(y, conditional_means)
        dispersion = self.fit_dispersion(y, conditional_means)
        log_dispersion = float(np.log(dispersion))

        def objective(theta: float) -> float:
            return self.negative_log_likelihood(np.exp(theta), y, conditional_means)

        curvature = (
            objective(log_dispersion + epsilon)
            - 2.0 * objective(log_dispersion)
            + objective(log_dispersion - epsilon)
        ) / epsilon**2
        variance = (
            1.0 / curvature if np.isfinite(curvature) and curvature > 1e-8 else np.inf
        )
        return dispersion, log_dispersion, float(variance)


class NegativeBinomialFamily(DistributionFamily):
    """Negative Binomial family for non-negative integer counts."""

    is_discrete = True
    parameter_column = "r_dispersion"

    def __init__(self, min_size: float = 1e-3, max_size: float = 50.0):
        self.min_size = min_size
        self.max_size = max_size

    @property
    def dispersion_bounds(self) -> tuple[float, float]:
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

    def negative_log_likelihood(self, dispersion, y, conditional_means) -> float:
        if not np.isfinite(dispersion) or dispersion <= 0:
            return 1e10
        conditional_means = np.maximum(conditional_means, 1e-6)
        probability = dispersion / (dispersion + conditional_means)
        log_probability = nbinom.logpmf(y, dispersion, probability)
        log_probability = np.where(np.isneginf(log_probability), -1e2, log_probability)
        return float(-np.sum(log_probability))

    def distribution(self, conditional_means, dispersions):
        return NegativeBinomialPredictiveDistribution(conditional_means, dispersions)


class GammaFamily(DistributionFamily):
    """Gamma family for strictly positive continuous targets."""

    parameter_column = "shape_dispersion"

    def __init__(self, min_shape: float = 1e-3, max_shape: float = 1e4):
        self.min_shape = min_shape
        self.max_shape = max_shape

    @property
    def dispersion_bounds(self) -> tuple[float, float]:
        return (self.min_shape, self.max_shape)

    def validate_target(self, y: np.ndarray) -> None:
        if not np.all(np.isfinite(y)):
            raise ValueError("Target values must be finite.")
        if np.any(y <= 0.0):
            raise ValueError(
                "Target values must be strictly positive for the Gamma model."
            )

    def negative_log_likelihood(self, dispersion, y, conditional_means) -> float:
        if not np.isfinite(dispersion) or dispersion <= 0:
            return 1e10
        conditional_means = np.maximum(conditional_means, 1e-6)
        log_density = gamma.logpdf(
            y, a=dispersion, scale=conditional_means / dispersion
        )
        if not np.all(np.isfinite(log_density)):
            return 1e10
        return float(-np.sum(log_density))

    def distribution(self, conditional_means, dispersions):
        return GammaPredictiveDistribution(conditional_means, dispersions)
