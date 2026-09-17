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
    """Detect drift between continuous reference and current samples.

    The detector uses the first Wasserstein distance divided by the reference
    standard deviation. It classifies drift with the two-sample permutation
    test implemented by :class:`BaseDrift`.

    Parameters
    ----------
    alpha : float, default=0.05
        Significance level used to classify drift.
    n_resamples : int, default=500
        Number of random label permutations used to approximate the null
        distribution.
    random_state : int or None, default=None
        Seed used for reproducible permutations.
    min_reference_size : int, default=2
        Minimum number of reference observations accepted by :meth:`fit`.
    min_current_size : int, default=2
        Minimum number of current observations accepted by :meth:`predict`.

    Attributes
    ----------
    reference_ : numpy.ndarray
        Validated continuous reference observations.
    reference_size_ : int
        Number of fitted reference observations.
    scale_ : float
        Reference population standard deviation used to standardize the
        observed Wasserstein distance.

    Notes
    -----
    The internal distance is expressed in reference-standard-deviation units.
    A value of one means that the Wasserstein distance equals one reference
    standard deviation. It is unbounded above.

    During permutation inference, the scale is recomputed from each permuted
    reference group. A machine-precision floor keeps the statistic finite for
    constant references.

    Examples
    --------
    >>> detector = ConDrift(n_resamples=999, random_state=42)
    >>> result = detector.fit(reference).predict(current)
    >>> result.p_value, result.drift
    (0.001, True)
    """

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
        """Return a one-dimensional array of finite floating-point values."""
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
        """Fit the continuous reference distribution and its scale.

        Parameters
        ----------
        reference : array-like of shape (n_observations,)
            Finite numeric baseline observations.

        Returns
        -------
        ConDrift
            Fitted detector.

        Raises
        ------
        ValueError
            If the input is empty, nonnumeric, nonfinite, not one-dimensional,
            or smaller than ``min_reference_size``.
        """
        super().fit(reference)
        self.scale_ = max(float(np.std(self.reference_)), np.finfo(float).eps)
        return self

    def _distance(self, reference: np.ndarray, current: np.ndarray) -> float:
        """Return Wasserstein distance standardized by the fitted scale."""
        distance = float(wasserstein_distance(reference, current))
        return distance / self.scale_

    def _inference_distance(self, reference: np.ndarray, current: np.ndarray) -> float:
        """Return standardized Wasserstein distance for a permuted split."""
        distance = float(wasserstein_distance(reference, current))
        scale = max(float(np.std(reference)), np.finfo(float).eps)
        return distance / scale
