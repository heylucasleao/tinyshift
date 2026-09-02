# Copyright (c) 2024-2026 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License


import numpy as np
from typing import Callable, Union, Tuple
from scipy.stats import norm


class BootstrapBCA:
    def __init__(self, random_state: int = 42):
        """
        Initialize the BootstrapBCA class.

        Parameters:
        - random_state (int): Random seed for reproducibility. Default is 42.
        """
        self.random_state = random_state

    def _jackknife_acceleration(self, data: np.ndarray, statistic: Callable) -> float:
        """Calculate the acceleration parameter using jackknife resampling."""
        n = len(data)
        jackknife = np.array([statistic(np.delete(data, i)) for i in range(n)])
        jackknife_mean = jackknife.mean()
        jackknife_deviation = jackknife_mean - jackknife
        skew = jackknife_deviation**3
        variance = jackknife_deviation**2
        denominator = 6.0 * (np.sum(variance) ** 1.5)
        acceleration = 0.0 if denominator == 0 else np.sum(skew) / denominator
        return acceleration

    def _bootstrap_statistics(
        self, data: np.ndarray, statistic: Callable, n_resamples: int
    ) -> np.ndarray:
        """Perform bootstrap resampling and calculate statistics."""
        rng = np.random.RandomState(self.random_state)
        return np.array(
            [
                statistic(rng.choice(data, size=len(data), replace=True))
                for _ in range(n_resamples)
            ]
        )

    @classmethod
    def compute_interval(
        cls,
        data: Union[np.ndarray, list],
        confidence_level: float,
        statistic: Callable,
        n_resamples: int,
        random_state: int = 42,
    ) -> Tuple[float, float]:
        """
        Calculates the bias-corrected and accelerated (BCa) bootstrap confidence interval for the given data.

        Parameters:
        - data (list or numpy array): Sample data.
        - confidence_level (float): Desired confidence level (e.g., 0.95 for 95%).
        - statistic (function): Statistical function to apply to the data. Default is np.mean.
        - n_resamples (int): Number of bootstrap resamples to perform. Default is 1000.

        Returns:
        - tuple: A tuple containing the lower and upper bounds of the BCa confidence interval.
        """
        instance = cls(random_state=random_state)
        data = np.asarray(data, dtype=np.float64)
        if data.ndim != 1 or data.size < 2:
            raise ValueError(
                "data must be a one-dimensional array with at least two values"
            )
        if not np.isfinite(data).all():
            raise ValueError("data must contain only finite values")
        if not callable(statistic):
            raise TypeError("statistic must be callable")
        if not 0 < confidence_level < 1:
            raise ValueError("confidence_level must be between 0 and 1")
        if (
            isinstance(n_resamples, (bool, np.bool_))
            or not isinstance(n_resamples, (int, np.integer))
            or n_resamples < 2
        ):
            raise ValueError("n_resamples must be an integer >= 2")

        # Bootstrap resampling
        sample_statistics = instance._bootstrap_statistics(data, statistic, n_resamples)

        # Jackknife resampling for acceleration
        acceleration = instance._jackknife_acceleration(data, statistic)

        # Bias correction
        observed_stat = statistic(data)
        if not np.isfinite(sample_statistics).all() or not np.isfinite(observed_stat):
            raise ValueError("statistic must return finite scalar values")
        bias = np.mean(sample_statistics < observed_stat)
        bias = np.clip(bias, 0.5 / n_resamples, 1 - 0.5 / n_resamples)
        z0 = norm.ppf(bias)

        # Adjusting percentiles
        alpha = 1 - confidence_level
        z_alpha_lower = norm.ppf(alpha / 2)
        z_alpha_upper = norm.ppf(1 - alpha / 2)

        denominator_lower = 1 - acceleration * (z0 + z_alpha_lower)
        denominator_upper = 1 - acceleration * (z0 + z_alpha_upper)

        z_lower_bound = z0 + (z0 + z_alpha_lower) / denominator_lower
        z_upper_bound = z0 + (z0 + z_alpha_upper) / denominator_upper

        alpha_lower = norm.cdf(z_lower_bound)
        alpha_upper = norm.cdf(z_upper_bound)

        # Calculate lower and upper bounds from the percentiles
        lower_bound = np.quantile(sample_statistics, alpha_lower)
        upper_bound = np.quantile(sample_statistics, alpha_upper)

        return lower_bound, upper_bound
