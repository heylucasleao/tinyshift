from numbers import Real
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

ArrayLike = Union[
    np.ndarray,
    List[float],
    pd.Series,
]

IntermittencyInput = Union[
    ArrayLike,
    pd.DataFrame,
]


class IntermittencyAnalyzer:
    """
    Analyze intermittent-demand characteristics for one or more time series.

    The analyzer summarizes intermittent demand using complementary diagnostics
    related to demand occurrence, demand magnitude, zero frequency, and the
    regularity of spacing between positive-demand observations.

    It supports both a single univariate series and panel data. For panel data,
    each series is identified by ``unique_id_col`` and analyzed independently.

    Observations are assumed to be provided in temporal order. For panel data,
    rows within each series are assumed to already follow their temporal
    ordering. The analyzer does not sort observations by a timestamp column.

    Parameters
    ----------
    unique_id_col : str, default="unique_id"
        Column identifying individual time series when ``X`` is a DataFrame.

    target_col : str, optional
        Column containing non-negative demand values when ``X`` is a DataFrame.

        If omitted, a column named ``"y"`` is used when present. Otherwise,
        the target is inferred when exactly one numeric column exists besides
        ``unique_id_col``.

    adi_threshold : float, default=1.32
        Threshold separating frequent from intermittent demand occurrence.

    cv2_threshold : float, default=0.49
        Threshold separating low from high variability in positive demand
        magnitudes.

    Attributes
    ----------
    adi_ : float or dict
        Average Demand Interval.

    cv2_ : float or dict
        Squared coefficient of variation of positive demand values.

    zero_proportion_ : float or dict
        Proportion of observations with zero demand.

    intervals_ : numpy.ndarray or dict
        Distances, in numbers of observations, between consecutive
        positive-demand occurrences.

    interval_cv_ : float or dict
        Coefficient of variation of inter-demand intervals.

    classification_ : str, None, or dict
        Intermittency classification.

        Possible values are:

        - ``"smooth"``
        - ``"intermittent"``
        - ``"erratic"``
        - ``"lumpy"``

        ``None`` is used when the classification is undefined, such as when
        the series contains no positive demand.

    Notes
    -----
    The Average Demand Interval is defined as

    .. math::

        ADI = \\frac{N}{N_{+}},

    where :math:`N` is the total number of observations and :math:`N_{+}` is
    the number of observations with positive demand.

    The squared coefficient of variation is calculated only from positive
    demand values:

    .. math::

        CV^2 =
        \\left(
            \\frac{\\sigma_{+}}{\\mu_{+}}
        \\right)^2.

    ADI measures how sparse demand occurrences are, while CV² measures how
    variable demand magnitudes are when demand occurs.

    ``interval_cv_`` captures a different property: the irregularity of the
    spacing between positive-demand occurrences.

    Inter-demand intervals are expressed in numbers of observations rather
    than physical time units.

    Examples
    --------
    Analyze a single demand series:

    >>> analyzer = IntermittencyAnalyzer()
    >>> analyzer.fit([0, 0, 4, 0, 2, 0, 0, 6])
    IntermittencyAnalyzer(...)

    >>> analyzer.adi_
    2.6666666666666665

    >>> analyzer.classification_
    'intermittent'

    Retrieve a compact profile:

    >>> analyzer.profile()
    {
        'adi': ...,
        'cv2': ...,
        'zero_proportion': ...,
        'interval_cv': ...,
        'classification': ...
    }

    Analyze panel data:

    >>> analyzer = IntermittencyAnalyzer(
    ...     unique_id_col="unique_id",
    ...     target_col="y",
    ... )
    >>> analyzer.fit(data)
    IntermittencyAnalyzer(...)

    >>> analyzer.profile()
               adi   cv2  zero_proportion  interval_cv classification
    unique_id
    item_a     ...
    item_b     ...
    """

    def __init__(
        self,
        unique_id_col: str = "unique_id",
        target_col: Optional[str] = None,
        adi_threshold: float = 1.32,
        cv2_threshold: float = 0.49,
    ) -> None:
        self.unique_id_col = unique_id_col
        self.target_col = target_col
        self.adi_threshold = adi_threshold
        self.cv2_threshold = cv2_threshold

        self._validate_params()

    def __repr__(self) -> str:
        return (
            "IntermittencyAnalyzer("
            f"unique_id_col={self.unique_id_col!r}, "
            f"target_col={self.target_col!r}, "
            f"adi_threshold={self.adi_threshold}, "
            f"cv2_threshold={self.cv2_threshold}"
            ")"
        )

    def _validate_params(self) -> None:
        """Validate analyzer configuration."""
        if (
            isinstance(self.adi_threshold, bool)
            or not isinstance(self.adi_threshold, Real)
            or not np.isfinite(self.adi_threshold)
            or self.adi_threshold <= 0
        ):
            raise ValueError(
                f"'adi_threshold' must be positive, got {self.adi_threshold}."
            )

        if (
            isinstance(self.cv2_threshold, bool)
            or not isinstance(self.cv2_threshold, Real)
            or not np.isfinite(self.cv2_threshold)
            or self.cv2_threshold <= 0
        ):
            raise ValueError(
                f"'cv2_threshold' must be positive, got {self.cv2_threshold}."
            )

    @staticmethod
    def _prepare_demand(
        X: ArrayLike,
    ) -> np.ndarray:
        """
        Validate and prepare a univariate demand series.
        """
        X = np.asarray(
            X,
            dtype=np.float64,
        )

        if X.ndim != 1:
            raise ValueError("Input data must be 1-dimensional.")

        if X.size < 2:
            raise ValueError("Input data must contain at least two observations.")

        if not np.isfinite(X).all():
            raise ValueError("Input data must contain only finite values.")

        if np.any(X < 0):
            raise ValueError("Demand values must be non-negative.")

        return X

    def _resolve_target_column(
        self,
        data: pd.DataFrame,
    ) -> str:
        """
        Resolve the target column used for panel input.
        """
        if self.unique_id_col not in data.columns:
            raise ValueError(
                f"DataFrame must contain the unique ID column {self.unique_id_col!r}."
            )

        if self.target_col is not None:
            if self.target_col not in data.columns:
                raise ValueError(
                    f"DataFrame does not contain target column {self.target_col!r}."
                )

            return self.target_col

        if "y" in data.columns:
            return "y"

        numeric_columns = [
            column
            for column in data.columns
            if (
                column != self.unique_id_col
                and pd.api.types.is_numeric_dtype(data[column])
            )
        ]

        if len(numeric_columns) != 1:
            raise ValueError(
                "Could not infer the target column. "
                "Provide `target_col`, include a column named 'y', "
                "or include exactly one numeric column besides "
                f"{self.unique_id_col!r}."
            )

        return numeric_columns[0]

    @staticmethod
    def _inter_demand_intervals(
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate distances between consecutive positive-demand occurrences.
        """
        demand_indices = np.flatnonzero(X > 0)

        if demand_indices.size < 2:
            return np.array(
                [],
                dtype=int,
            )

        return np.diff(demand_indices)

    @staticmethod
    def _average_demand_interval(
        X: np.ndarray,
    ) -> float:
        """
        Calculate the Average Demand Interval (ADI).
        """
        n_positive = np.count_nonzero(X > 0)

        if n_positive == 0:
            return float("inf")

        return float(X.size / n_positive)

    @staticmethod
    def _squared_coefficient_of_variation(
        X: np.ndarray,
    ) -> float:
        """
        Calculate CV² from strictly positive demand observations.
        """
        positive_demand = X[X > 0]

        if positive_demand.size == 0:
            return float("nan")

        mean_demand = np.mean(positive_demand)

        if mean_demand == 0:
            return float("nan")

        cv = (
            np.std(
                positive_demand,
                ddof=0,
            )
            / mean_demand
        )

        return float(cv**2)

    @staticmethod
    def _zero_proportion(
        X: np.ndarray,
    ) -> float:
        """
        Calculate the proportion of zero-demand observations.
        """
        return float(np.mean(X == 0))

    @staticmethod
    def _inter_demand_interval_cv(
        intervals: np.ndarray,
    ) -> float:
        """
        Calculate the coefficient of variation of inter-demand intervals.
        """
        if intervals.size < 2:
            return float("nan")

        mean_interval = np.mean(intervals)

        if mean_interval <= 0:
            return float("nan")

        return float(
            np.std(
                intervals,
                ddof=0,
            )
            / mean_interval
        )

    def _classify(
        self,
        adi: float,
        cv2: float,
    ) -> Optional[str]:
        """
        Classify demand according to the ADI-CV² framework.
        """
        if not np.isfinite(adi) or np.isnan(cv2):
            return None

        if adi < self.adi_threshold:
            if cv2 < self.cv2_threshold:
                return "smooth"

            return "erratic"

        if cv2 < self.cv2_threshold:
            return "intermittent"

        return "lumpy"

    def _fit_single(
        self,
        X: ArrayLike,
    ) -> Dict[str, Any]:
        """
        Calculate all intermittency diagnostics for a single series.
        """
        demand = self._prepare_demand(X)

        intervals = self._inter_demand_intervals(demand)

        adi = self._average_demand_interval(demand)

        cv2 = self._squared_coefficient_of_variation(demand)

        zero_proportion = self._zero_proportion(demand)

        interval_cv = self._inter_demand_interval_cv(intervals)

        classification = self._classify(
            adi=adi,
            cv2=cv2,
        )

        return {
            "adi": adi,
            "cv2": cv2,
            "zero_proportion": zero_proportion,
            "interval_cv": interval_cv,
            "intervals": intervals,
            "classification": classification,
        }

    def fit(
        self,
        X: IntermittencyInput,
    ) -> "IntermittencyAnalyzer":
        """
        Fit the intermittency analyzer.

        Parameters
        ----------
        X : numpy.ndarray, list of float, pandas.Series, or pandas.DataFrame
            Input demand data.

            A one-dimensional input is interpreted as a single time series.

            A DataFrame is interpreted as panel data. Each unique value of
            ``unique_id_col`` is analyzed independently.

            For all inputs, observations are assumed to already be in temporal
            order.

        Returns
        -------
        IntermittencyAnalyzer
            The fitted analyzer instance.

        Notes
        -----
        Calling ``fit`` updates the fitted attributes:

        - ``adi_``
        - ``cv2_``
        - ``zero_proportion_``
        - ``interval_cv_``
        - ``intervals_``
        - ``classification_``
        """
        if not isinstance(
            X,
            pd.DataFrame,
        ):
            result = self._fit_single(X)

            self.adi_ = result["adi"]

            self.cv2_ = result["cv2"]

            self.zero_proportion_ = result["zero_proportion"]

            self.interval_cv_ = result["interval_cv"]

            self.intervals_ = result["intervals"]

            self.classification_ = result["classification"]

            self._is_panel_ = False

            return self

        target_col = self._resolve_target_column(X)

        if X.empty:
            raise ValueError("Panel input must contain at least one series.")

        if X[self.unique_id_col].isna().any():
            raise ValueError("Unique ID values must not be missing.")

        results = {
            unique_id: self._fit_single(group[target_col])
            for unique_id, group in X.groupby(
                self.unique_id_col,
                sort=False,
            )
        }

        self.adi_ = {unique_id: result["adi"] for unique_id, result in results.items()}

        self.cv2_ = {unique_id: result["cv2"] for unique_id, result in results.items()}

        self.zero_proportion_ = {
            unique_id: result["zero_proportion"]
            for unique_id, result in results.items()
        }

        self.interval_cv_ = {
            unique_id: result["interval_cv"] for unique_id, result in results.items()
        }

        self.intervals_ = {
            unique_id: result["intervals"] for unique_id, result in results.items()
        }

        self.classification_ = {
            unique_id: result["classification"] for unique_id, result in results.items()
        }

        self._is_panel_ = True

        return self

    def profile(
        self,
    ) -> Union[
        Dict[str, Any],
        pd.DataFrame,
    ]:
        """
        Return a compact intermittency profile.

        Returns
        -------
        dict or pandas.DataFrame
            For a single time series, returns a dictionary containing the
            scalar intermittency diagnostics.

            For panel data, returns a DataFrame with one row per series.

        Raises
        ------
        RuntimeError
            If the analyzer has not been fitted.

        Notes
        -----
        ``intervals_`` is not included in the profile because it contains a
        variable-length array rather than a scalar summary statistic.
        """
        if not hasattr(
            self,
            "_is_panel_",
        ):
            raise RuntimeError(
                "The analyzer must be fitted before calling `profile()`."
            )

        if not self._is_panel_:
            return {
                "adi": self.adi_,
                "cv2": self.cv2_,
                "zero_proportion": (self.zero_proportion_),
                "interval_cv": (self.interval_cv_),
                "classification": (self.classification_),
            }

        profile = pd.DataFrame(
            {
                "adi": self.adi_,
                "cv2": self.cv2_,
                "zero_proportion": (self.zero_proportion_),
                "interval_cv": (self.interval_cv_),
                "classification": (self.classification_),
            }
        )

        profile.index.name = self.unique_id_col

        return profile

    def analyze(
        self,
        X: IntermittencyInput,
    ) -> Union[
        Dict[str, Any],
        pd.DataFrame,
    ]:
        """
        Fit the analyzer and return the intermittency profile directly.

        This is a convenience method equivalent to calling ``fit(X)`` followed
        by ``profile()``.

        Parameters
        ----------
        X : numpy.ndarray, list of float, pandas.Series, or pandas.DataFrame
            Input demand data.

        Returns
        -------
        dict or pandas.DataFrame
            Intermittency profile.
        """
        self.fit(X)

        return self.profile()
