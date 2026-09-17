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
    """Detect drift between categorical reference and current samples.

    The detector aligns the union of observed categories and computes
    Jensen--Shannon distance with logarithm base 2. It classifies drift with
    the two-sample permutation test implemented by :class:`BaseDrift`.

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
        Validated categorical reference observations.
    reference_size_ : int
        Number of fitted reference observations.

    Notes
    -----
    SciPy's ``jensenshannon`` function returns the square root of the
    Jensen--Shannon divergence. With logarithm base 2, the distance lies in
    ``[0, 1]``: zero represents identical empirical distributions and one
    represents disjoint supports.

    Categories appearing only in the current sample are included in the
    calculation. Missing and unhashable category values are rejected.

    Examples
    --------
    >>> detector = CatDrift(n_resamples=999, random_state=42)
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
        """Return a one-dimensional array of nonmissing hashable categories."""
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
        """Return probability vectors aligned over the union of categories.

        Parameters
        ----------
        reference : numpy.ndarray of shape (n_reference,)
            Validated categorical reference observations.
        current : numpy.ndarray of shape (n_current,)
            Validated categorical current observations.

        Returns
        -------
        reference_prob : numpy.ndarray
            Reference probabilities in the shared category order.
        current_prob : numpy.ndarray
            Current probabilities in the shared category order.
        """
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
        """Return base-2 Jensen--Shannon distance between empirical samples."""
        reference_prob, current_prob = self._probabilities(reference, current)
        return float(jensenshannon(reference_prob, current_prob, base=2))
