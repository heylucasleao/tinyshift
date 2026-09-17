# Copyright (c) 2024-2026 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License

"""Shared primitives for reference-to-current drift detection."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted


@dataclass(frozen=True)
class DriftResult:
    """Result of comparing one current sample with a fitted reference."""

    score: float
    threshold: float | None
    p_value: float | None
    drift: bool | None
    reference_size: int
    current_size: int


class BaseDrift(BaseEstimator, ABC):
    """Lifecycle shared by vector-based drift detectors."""

    def __init__(
        self,
        alpha: float,
        n_resamples: int,
        random_state: int | None,
        min_reference_size: int,
        min_current_size: int,
    ) -> None:
        self.alpha = alpha
        self.n_resamples = n_resamples
        self.random_state = random_state
        self.min_reference_size = min_reference_size
        self.min_current_size = min_current_size

    def _validate_params(self) -> None:
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be between 0 and 1.")
        if not isinstance(self.n_resamples, int) or self.n_resamples < 1:
            raise ValueError("n_resamples must be a positive integer.")
        for name, value in (
            ("min_reference_size", self.min_reference_size),
            ("min_current_size", self.min_current_size),
        ):
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")

    @abstractmethod
    def _validate_sample(self, values: Any, name: str) -> np.ndarray:
        """Return a validated one-dimensional sample."""

    @abstractmethod
    def _distance(self, reference: np.ndarray, current: np.ndarray) -> float:
        """Compute the configured distance between two validated samples."""

    def fit(self, reference: Any) -> "BaseDrift":
        """Store the reference sample used by subsequent comparisons."""
        self._validate_params()
        values = self._validate_sample(reference, "reference")
        if values.size < self.min_reference_size:
            raise ValueError(
                f"reference must contain at least {self.min_reference_size} observations."
            )
        self.reference_ = values.copy()
        self.reference_size_ = int(values.size)
        return self

    def _inference_distance(self, reference: np.ndarray, current: np.ndarray) -> float:
        """Statistic recomputed under a resampled reference/current assignment."""
        return self._distance(reference, current)

    def _permutation_scores(self, current: np.ndarray) -> np.ndarray:
        pooled = np.concatenate((self.reference_, current))
        rng = np.random.default_rng(self.random_state)
        scores = np.empty(self.n_resamples, dtype=float)
        for index in range(self.n_resamples):
            permuted = pooled[rng.permutation(pooled.size)]
            reference = permuted[: self.reference_size_]
            comparison = permuted[self.reference_size_ :]
            scores[index] = self._inference_distance(reference, comparison)
        return scores

    def _calibrate(
        self, current: np.ndarray, score: float
    ) -> tuple[float, float, bool]:
        null_scores = self._permutation_scores(current)
        threshold = float(np.quantile(null_scores, 1 - self.alpha, method="higher"))
        p_value = float(
            (1 + np.count_nonzero(null_scores >= score)) / (self.n_resamples + 1)
        )
        return threshold, p_value, p_value <= self.alpha

    def score(self, current: Any) -> float:
        """Return only the distance between the reference and current samples."""
        check_is_fitted(self, "reference_")
        values = self._validate_sample(current, "current")
        if values.size < self.min_current_size:
            raise ValueError(
                f"current must contain at least {self.min_current_size} observations."
            )
        return self._distance(self.reference_, values)

    def predict(self, current: Any) -> DriftResult:
        """Score and classify a current sample when a threshold is configured."""
        check_is_fitted(self, "reference_")
        values = self._validate_sample(current, "current")
        if values.size < self.min_current_size:
            raise ValueError(
                f"current must contain at least {self.min_current_size} observations."
            )
        score = self._distance(self.reference_, values)
        threshold, p_value, drift = self._calibrate(values, score)
        return DriftResult(
            score=score,
            threshold=threshold,
            p_value=p_value,
            drift=drift,
            reference_size=self.reference_size_,
            current_size=int(values.size),
        )
