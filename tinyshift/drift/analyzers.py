# Copyright (c) 2024-2026 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License

"""Panel adapters for vector-based drift detectors."""

from typing import Generic, TypeVar

import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.utils.validation import check_is_fitted

from .base import BaseDrift
from .categorical import CatDrift
from .continuous import ConDrift

DetectorT = TypeVar("DetectorT", bound=BaseDrift)


class BaseDriftAnalyzer(BaseEstimator, Generic[DetectorT]):
    """Coordinate independent reference-to-current tests across panel IDs.

    The analyzer clones one detector for every ID in a reference DataFrame.
    Prediction groups the current DataFrame by the same identifier and delegates
    each comparison to the corresponding fitted detector.

    Parameters
    ----------
    detector : BaseDrift
        Unfitted detector template. Its concrete type must match
        :attr:`detector_type` in the analyzer subclass.

    Attributes
    ----------
    detectors_ : dict
        Fitted detector for every reference ID. Created by :meth:`fit`.
    id_col_ : str
        Identifier column used during fitting.
    target_col_ : str
        Target column used during fitting.
    results_ : pandas.DataFrame
        Most recent prediction result. Created by :meth:`predict`.

    Notes
    -----
    A time column is intentionally not required. The caller defines which rows
    belong to the reference and current populations before passing each frame
    to the analyzer. Observations are pooled within ID and row order has no
    effect on the distribution comparison.

    Every ID appearing in the current frame must have a fitted reference.
    Reference IDs absent from a current frame are omitted from that prediction.
    """

    detector_type: type[BaseDrift]

    def __init__(self, detector: DetectorT) -> None:
        if not isinstance(detector, self.detector_type):
            raise TypeError(
                f"detector must be an instance of {self.detector_type.__name__}."
            )
        self.detector = detector

    @staticmethod
    def _validate_frame(df: pd.DataFrame, id_col: str, target_col: str) -> None:
        """Validate the structure shared by reference and current frames."""
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame.")
        missing = [column for column in (id_col, target_col) if column not in df]
        if missing:
            raise ValueError(f"DataFrame is missing required columns: {missing}.")
        if df.empty:
            raise ValueError("Panel input cannot be empty.")
        if df[id_col].isna().any():
            raise ValueError("ID values must not be missing.")

    def fit(
        self, df: pd.DataFrame, id_col: str = "unique_id", target_col: str = "y"
    ) -> "BaseDriftAnalyzer[DetectorT]":
        """Fit one cloned detector per reference ID.

        Parameters
        ----------
        df : pandas.DataFrame
            Reference observations in long format. Multiple rows may share an
            ID; each row contributes one target observation.
        id_col : str, default="unique_id"
            Column identifying independent reference populations.
        target_col : str, default="y"
            Column containing values passed to each detector.

        Returns
        -------
        BaseDriftAnalyzer
            Fitted analyzer.

        Raises
        ------
        TypeError
            If ``df`` is not a pandas DataFrame.
        ValueError
            If the frame is empty, required columns are missing, IDs contain
            missing values, or a grouped target is rejected by the detector.

        Notes
        -----
        The detector template is cloned with ``sklearn.base.clone``. Fitted
        state is therefore independent across IDs and the template itself is
        not modified.
        """
        self._validate_frame(df, id_col, target_col)
        self.id_col_ = id_col
        self.target_col_ = target_col
        self.detectors_: dict[object, DetectorT] = {}
        for unique_id, group in df.groupby(id_col, sort=False, observed=True):
            self.detectors_[unique_id] = clone(self.detector).fit(group[target_col])
        return self

    def predict(
        self, df: pd.DataFrame, id_col: str | None = None, target_col: str | None = None
    ) -> pd.DataFrame:
        """Test every current ID against its fitted reference distribution.

        Parameters
        ----------
        df : pandas.DataFrame
            Current observations in long format.
        id_col : str or None, default=None
            Identifier column in ``df``. By default, reuse the column supplied
            to :meth:`fit`.
        target_col : str or None, default=None
            Target column in ``df``. By default, reuse the column supplied to
            :meth:`fit`.

        Returns
        -------
        pandas.DataFrame
            One row per current ID, in first-appearance order, with columns for
            ID, internal ``score``, empirical ``threshold``, ``p_value``,
            ``drift``, ``reference_size``, and ``current_size``.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If :meth:`fit` has not been called.
        TypeError
            If ``df`` is not a pandas DataFrame.
        ValueError
            If the frame is invalid, contains an ID without a fitted reference,
            or a grouped target is rejected by its detector.

        Notes
        -----
        Prediction runs a separate permutation test for every current ID.
        Statistical decisions should use ``p_value`` and ``drift``; ``score``
        and ``threshold`` are retained as diagnostics.
        """
        check_is_fitted(self, "detectors_")
        id_col = self.id_col_ if id_col is None else id_col
        target_col = self.target_col_ if target_col is None else target_col
        self._validate_frame(df, id_col, target_col)
        unknown = [
            value for value in pd.unique(df[id_col]) if value not in self.detectors_
        ]
        if unknown:
            raise ValueError(f"No reference distribution for IDs: {unknown!r}.")

        rows = []
        for unique_id, group in df.groupby(id_col, sort=False, observed=True):
            result = self.detectors_[unique_id].predict(group[target_col])
            rows.append(
                {
                    id_col: unique_id,
                    "score": result.score,
                    "threshold": result.threshold,
                    "p_value": result.p_value,
                    "drift": result.drift,
                    "reference_size": result.reference_size,
                    "current_size": result.current_size,
                }
            )
        self.results_ = pd.DataFrame(rows)
        return self.results_.copy()

    def summary(self) -> pd.DataFrame:
        """Return a copy of the most recent prediction result.

        Returns
        -------
        pandas.DataFrame
            Same columns and row order returned by the latest :meth:`predict`.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If :meth:`predict` has not been called.
        """
        check_is_fitted(self, "results_")
        return self.results_.copy()


class ContinuousDriftAnalyzer(BaseDriftAnalyzer[ConDrift]):
    """Apply :class:`ConDrift` independently to every panel ID.

    Parameters
    ----------
    detector : ConDrift or None, default=None
        Detector template cloned for each reference ID. By default, use
        ``ConDrift()``.

    Examples
    --------
    >>> analyzer = ContinuousDriftAnalyzer(ConDrift(random_state=42))
    >>> result = analyzer.fit(reference_df).predict(current_df)
    >>> result[["unique_id", "p_value", "drift"]]
      unique_id  p_value  drift
    0         A     ...    ...
    """

    detector_type = ConDrift

    def __init__(self, detector: ConDrift | None = None) -> None:
        super().__init__(ConDrift() if detector is None else detector)


class CategoricalDriftAnalyzer(BaseDriftAnalyzer[CatDrift]):
    """Apply :class:`CatDrift` independently to every panel ID.

    Parameters
    ----------
    detector : CatDrift or None, default=None
        Detector template cloned for each reference ID. By default, use
        ``CatDrift()``.

    Examples
    --------
    >>> analyzer = CategoricalDriftAnalyzer(CatDrift(random_state=42))
    >>> result = analyzer.fit(reference_df, target_col="segment").predict(current_df)
    >>> result[["unique_id", "p_value", "drift"]]
      unique_id  p_value  drift
    0         A     ...    ...
    """

    detector_type = CatDrift

    def __init__(self, detector: CatDrift | None = None) -> None:
        super().__init__(CatDrift() if detector is None else detector)
