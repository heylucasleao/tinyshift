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


class CatDrift(BaseDrift):
    """Compare categorical samples with a fitted reference distribution."""

    def __init__(
        self,
        alpha: float = 0.05,
        n_resamples: int = 500,
        random_state: int | None = None,
        min_reference_size: int = 2,
        min_current_size: int = 2,
    ) -> None:
        super().__init__(
            alpha,
            n_resamples,
            random_state,
            min_reference_size,
            min_current_size,
        )

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
        return float(jensenshannon(reference_prob, current_prob, base=2))
