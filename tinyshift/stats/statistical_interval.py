# Copyright (c) 2024-2026 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License


import numpy as np
from scipy.stats import kurtosis, skew
from typing import Callable, Tuple, Union


class StatisticalInterval:
    """Utility class to compute different types of statistical interval."""

    @staticmethod
    def custom_interval(data: np.ndarray, custom_func: Callable) -> Tuple[float, float]:
        """Calculate interval using a custom function."""
        return custom_func(data)

    @staticmethod
    def calculate_interval(
        X: np.ndarray,
        center: Callable,
        spread: Callable,
        factor: float = 3,
    ) -> Tuple[float, float]:
        """Calculate interval using a central tendency and spread function."""

        if not callable(center) or not callable(spread):
            raise ValueError("center and spread must be callable functions")
        if not np.isfinite(factor) or factor <= 0:
            raise ValueError("factor must be a positive finite number")

        center_value = center(X)
        spread_value = spread(X)
        lower_bound = center_value - factor * spread_value
        upper_bound = center_value + factor * spread_value
        return lower_bound, upper_bound

    @staticmethod
    def iqr_interval(X: np.ndarray) -> Tuple[float, float]:
        """Calculates interval using IQR and median with a default factor of 1.5."""

        def iqr(x):
            q75, q25 = np.percentile(x, [75, 25])
            return q75 - q25

        q75, q25 = np.percentile(X, [75, 25])
        spread = iqr(X)
        return q25 - 1.5 * spread, q75 + 1.5 * spread

    @staticmethod
    def stddev_interval(X: np.ndarray) -> Tuple[float, float]:
        """Calculates interval using mean and standard deviation."""
        return StatisticalInterval.calculate_interval(X, np.mean, np.std)

    @staticmethod
    def mad_interval(X: np.ndarray) -> Tuple[float, float]:
        """Calculates interval using Median Absolute Deviation (MAD)."""
        mad = lambda x: np.median(np.abs(x - np.median(x)))
        return StatisticalInterval.calculate_interval(X, np.median, mad)

    @staticmethod
    def quantile_interval(
        X: np.ndarray, lower: float, upper: float
    ) -> Tuple[float, float]:
        """Calculates interval using quantiles."""
        for name, value in (("lower", lower), ("upper", upper)):
            if value is not None and not 0 <= value <= 1:
                raise ValueError(f"{name} quantile must be between 0 and 1")
        if lower is not None and upper is not None and lower > upper:
            raise ValueError("lower quantile cannot exceed upper quantile")

        lower_bound = (
            np.quantile(X, lower, method="lower") if lower is not None else np.nan
        )
        upper_bound = (
            np.quantile(X, upper, method="higher") if upper is not None else np.nan
        )
        return (lower_bound, upper_bound)

    @staticmethod
    def auto_select_method(X: np.ndarray) -> Union[str, Tuple[str, float, float]]:
        """Automatically selects the statistical interval method for outlier detection
        based on data distribution and dispersion (using Coefficient of Variation, CV).
        """

        skewness = abs(skew(X))
        kurt = abs(kurtosis(X))
        is_normal = (skewness <= 0.5) and (kurt <= 1)
        coeff_var = np.std(X) / abs(np.mean(X)) if np.mean(X) != 0 else np.inf

        if is_normal:
            if coeff_var < 0.15:
                return "stddev"
            elif coeff_var < 0.30:
                return "iqr"
            else:
                return "mad"

        if coeff_var > 1.5 or skewness > 0.5:
            return "mad"

        return "iqr"

    @staticmethod
    def compute_interval(
        X: np.ndarray,
        method: Union[str, Callable, Tuple[float]],
    ) -> Tuple[float, float]:
        """
        Determines the lower and upper bounds on the specified method.

        Args:
            data: Input data for threshold calculation.
            method: Method to compute interval. Can be:
                - "auto" (automatically selects a method)
                - "stddev" (mean ± 3σ)
                - "mad" (median ± 3*MAD)
                - "iqr" (median ± 1.5*IQR)
                - A custom function (returns lower, upper bounds)
                - A pre-defined tuple (lower, upper)

        Returns:
            Tuple[float, float]: Lower and upper bounds.
        """
        X = np.asarray(X)
        if X.ndim != 1 or X.size == 0:
            raise ValueError("X must be a non-empty one-dimensional array")
        if not np.issubdtype(X.dtype, np.number) or not np.isfinite(X).all():
            raise ValueError("X must contain only finite numeric values")

        def _none_to_nan(value):
            return np.nan if value is None else value

        if method == "auto":
            method = StatisticalInterval.auto_select_method(X)

        if isinstance(method, str):
            if method == "stddev":
                lower_bound, upper_bound = StatisticalInterval.stddev_interval(X)
            elif method == "mad":
                lower_bound, upper_bound = StatisticalInterval.mad_interval(X)
            elif method == "iqr":
                lower_bound, upper_bound = StatisticalInterval.iqr_interval(X)
            else:
                raise ValueError(f"Unsupported method: {method}")
        elif callable(method):
            lower_bound, upper_bound = StatisticalInterval.custom_interval(X, method)
            lower_bound = _none_to_nan(lower_bound)
            upper_bound = _none_to_nan(upper_bound)
        elif isinstance(method, tuple) and len(method) == 2:
            lower_bound, upper_bound = method
            lower_bound = _none_to_nan(lower_bound)
            upper_bound = _none_to_nan(upper_bound)
        elif isinstance(method, tuple) and len(method) == 3 and method[0] == "quantile":
            lower, upper = method[1], method[2]
            lower_bound, upper_bound = StatisticalInterval.quantile_interval(
                X, lower, upper
            )
        else:
            raise ValueError("Invalid method specification.")

        return lower_bound, upper_bound
