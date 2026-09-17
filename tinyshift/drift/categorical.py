# Copyright (c) 2024-2026 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License

"""Categorical reference-to-current drift detection."""

from collections.abc import Hashable
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

from .base import BaseDrift


def chebyshev(a: np.ndarray, b: np.ndarray) -> float:
    """Return the largest absolute probability difference."""
    return float(np.max(np.abs(a - b)))


def psi(observed: np.ndarray, expected: np.ndarray, epsilon: float = 1e-4) -> float:
    """Return the population stability index between two distributions."""
    observed = np.clip(observed, epsilon, 1)
    expected = np.clip(expected, epsilon, 1)
    return float(np.sum((observed - expected) * np.log(observed / expected)))


class CatDrift(BaseDrift):
    """Compare categorical samples with a fitted reference distribution."""

    def __init__(
        self,
        metric: str = "jensen_shannon",
        alpha: float = 0.05,
        n_resamples: int = 500,
        random_state: int | None = None,
        min_reference_size: int = 2,
        min_current_size: int = 2,
    ) -> None:
        self.metric = metric
        super().__init__(
            alpha,
            n_resamples,
            random_state,
            min_reference_size,
            min_current_size,
        )

    def _validate_params(self) -> None:
        super()._validate_params()
        if self.metric not in {"chebyshev", "jensen_shannon", "psi"}:
            raise ValueError("metric must be 'chebyshev', 'jensen_shannon', or 'psi'.")

    def _validate_sample(self, values: Any, name: str) -> np.ndarray:
        if isinstance(values, pd.Series):
            values = values.to_numpy(dtype=object)
        array = np.asarray(values, dtype=object)
        if array.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional.")
        if array.size == 0:
            raise ValueError(f"{name} cannot be empty.")
        if pd.isna(array).any():
            raise ValueError(f"{name} must not contain missing values.")
        if not all(isinstance(value, Hashable) for value in array):
            raise ValueError(f"{name} must contain hashable category values.")
        return array

    @staticmethod
    def _probabilities(
        reference: np.ndarray, current: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        reference_counts = pd.Series(reference, dtype=object).value_counts(sort=False)
        current_counts = pd.Series(current, dtype=object).value_counts(sort=False)
        categories = reference_counts.index.union(current_counts.index, sort=False)
        reference_prob = (
            reference_counts.reindex(categories, fill_value=0).to_numpy(dtype=float)
            / reference.size
        )
        current_prob = (
            current_counts.reindex(categories, fill_value=0).to_numpy(dtype=float)
            / current.size
        )
        return reference_prob, current_prob

    def _distance(self, reference: np.ndarray, current: np.ndarray) -> float:
        reference_prob, current_prob = self._probabilities(reference, current)
        if self.metric == "chebyshev":
            return chebyshev(reference_prob, current_prob)
        if self.metric == "psi":
            return psi(current_prob, reference_prob)
        return float(jensenshannon(reference_prob, current_prob))
