# Copyright (c) 2024-2026 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License

"""Continuous reference-to-current drift detection."""

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from .base import BaseDrift


class ConDrift(BaseDrift):
    """Compare continuous samples with a fitted reference distribution."""

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
            values = values.to_numpy()
        array = np.asarray(values)
        if array.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional.")
        if array.size == 0:
            raise ValueError(f"{name} cannot be empty.")
        if not np.issubdtype(array.dtype, np.number):
            raise ValueError(f"{name} must contain numeric values.")
        array = array.astype(float, copy=False)
        if not np.isfinite(array).all():
            raise ValueError(f"{name} must contain finite values.")
        return array

    def fit(self, reference: Any) -> "ConDrift":
        super().fit(reference)
        self.scale_ = max(float(np.std(self.reference_)), np.finfo(float).eps)
        return self

    def _distance(self, reference: np.ndarray, current: np.ndarray) -> float:
        distance = float(wasserstein_distance(reference, current))
        return distance / self.scale_

    def _inference_distance(self, reference: np.ndarray, current: np.ndarray) -> float:
        distance = float(wasserstein_distance(reference, current))
        scale = max(float(np.std(reference)), np.finfo(float).eps)
        return distance / scale
