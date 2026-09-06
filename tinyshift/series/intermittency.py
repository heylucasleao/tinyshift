from numbers import Real
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

ArrayLike = Union[
    np.ndarray,
    List[float],
    pd.Series,
]


class IntermittencyAnalyzer:
    """
    Analyze intermittent-demand characteristics for one or more time series.

    The analyzer summarizes intermittent demand using complementary diagnostics
    related to demand occurrence, demand magnitude, zero frequency, and the
    regularity of spacing between positive-demand observations.

    Input follows the panel convention: one identifier column, one time column,
    and one numeric target column. Each series is sorted by time and analyzed
    independently.

    Parameters
    ----------
    adi_threshold : float, default=1.32
        Threshold separating frequent from intermittent demand occurrence.

    cv2_threshold : float, default=0.49
        Threshold separating low from high variability in positive demand
        magnitudes.

    Attributes
    ----------
    results_ : dict
        Mapping from each unique ID to its intermittency diagnostics.

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
    Analyze panel demand data:

    >>> analyzer = IntermittencyAnalyzer()
    >>> analyzer.fit(data, id_col="unique_id", time_col="ds", target_col="y")
    IntermittencyAnalyzer(...)

    >>> analyzer.summary()
      unique_id  adi   cv2  zero_proportion  interval_cv classification
    0     item_a  ...   ...              ...          ...            ...
    1     item_b  ...   ...              ...          ...            ...
    """

    def __init__(
        self,
        adi_threshold: float = 1.32,
        cv2_threshold: float = 0.49,
    ) -> None:
        self.adi_threshold = adi_threshold
        self.cv2_threshold = cv2_threshold

        self._validate_params()

    def __repr__(self) -> str:
        return (
            "IntermittencyAnalyzer("
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
        required = [self.id_col_, self.time_col_, self.target_col_]
        missing = [column for column in required if column not in data.columns]
        if missing:
            raise ValueError(f"DataFrame is missing required columns: {missing}.")
        return self.target_col_

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
        df: pd.DataFrame,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
    ) -> "IntermittencyAnalyzer":
        """
        Fit the intermittency analyzer.

        Parameters
        ----------
        df : pandas.DataFrame
            Panel data containing the configured ID, time, and target columns.
        id_col : str, default="unique_id"
            Column identifying individual time series.
        time_col : str, default="ds"
            Column defining temporal order within each series.
        target_col : str, default="y"
            Column containing non-negative demand values.

        Returns
        -------
        IntermittencyAnalyzer
            The fitted analyzer instance.

        Notes
        -----
        Calling ``fit`` updates the fitted attribute ``results_``.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame in panel format.")

        self.id_col_ = id_col
        self.time_col_ = time_col
        self.target_col_ = target_col

        target_col = self._resolve_target_column(df)

        if df.empty:
            raise ValueError("Panel input must contain at least one series.")

        if df[[self.id_col_, self.time_col_]].isna().any().any():
            raise ValueError("ID and time values must not be missing.")
        if df.duplicated([self.id_col_, self.time_col_]).any():
            raise ValueError("Panel contains duplicate ID-time observations.")

        data = df.sort_values([self.id_col_, self.time_col_])

        results = {
            unique_id: self._fit_single(group[target_col])
            for unique_id, group in data.groupby(
                self.id_col_,
                sort=False,
            )
        }
        self.results_ = results

        return self

    def summary(
        self,
    ) -> pd.DataFrame:
        """
        Return a compact intermittency profile.

        Returns
        -------
        pandas.DataFrame
            One row per series, with the ID and scalar intermittency diagnostics.

        Raises
        ------
        RuntimeError
            If the analyzer has not been fitted.

        Notes
        -----
        The ``intervals`` entry stored in each ``results_[unique_id]`` mapping
        is not included because it contains a variable-length array rather
        than a scalar summary statistic.
        """
        if not hasattr(self, "results_"):
            raise RuntimeError(
                "The analyzer must be fitted before calling `summary()`."
            )

        columns = [
            "adi",
            "cv2",
            "zero_proportion",
            "interval_cv",
            "classification",
        ]
        rows = [
            {
                self.id_col_: unique_id,
                **{column: result[column] for column in columns},
            }
            for unique_id, result in self.results_.items()
        ]
        return pd.DataFrame(rows, columns=[self.id_col_, *columns])
