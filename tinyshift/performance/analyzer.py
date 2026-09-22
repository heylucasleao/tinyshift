# Copyright (c) 2024-2026 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License

"""Panel adapter for direct loss estimation."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils.validation import check_is_fitted

from .dle import DirectLossEstimator


class DirectLossAnalyzer(BaseEstimator):
    """Estimate model loss independently for each panel ID.

    The analyzer clones one :class:`DirectLossEstimator` per reference ID. The
    first ``1 - fraction`` of each ID fits its loss model; the remaining rows
    form a held-out baseline. :meth:`predict` estimates loss for current rows
    without observed targets.

    Parameters
    ----------
    estimator : DirectLossEstimator or None, default=None
        Estimator template cloned for each ID. By default, use a DLE with a
        random forest regressor as its learner.
    fraction : float, default=0.25
        Fraction of each reference ID reserved for the held-out baseline.
        Must be strictly between zero and one.

    Attributes
    ----------
    estimators_ : dict
        Fitted DLE for each reference ID.
    baselines_ : dict
        Held-out estimated loss, realized loss, and sample size for each ID.
    results_ : dict
        Most recent prediction result keyed by ID. Created by :meth:`predict`.

    Notes
    -----
    Reference rows retain their input order. For time series, order rows
    chronologically within each ID before fitting. ``degradation`` indicates
    a positive change in estimated loss; it is not a significance test or
    confirmation of realized degradation.

    Examples
    --------
    >>> analyzer = DirectLossAnalyzer(fraction=0.25)
    >>> analyzer.fit(reference_df, feature_cols=["feature_a", "feature_b"])
    >>> result = analyzer.predict(current_df)
    """

    def __init__(
        self,
        estimator: DirectLossEstimator | None = None,
        fraction: float = 0.25,
    ) -> None:
        if estimator is not None and not isinstance(estimator, DirectLossEstimator):
            raise TypeError("estimator must be a DirectLossEstimator.")
        self.estimator = (
            DirectLossEstimator(
                learner=RandomForestRegressor(
                    n_estimators=100, min_samples_leaf=3, random_state=42
                )
            )
            if estimator is None
            else estimator
        )
        self.fraction = fraction

    @staticmethod
    def _validate_frame(df, required, id_col):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame.")
        if df.empty:
            raise ValueError("Panel input cannot be empty.")
        missing = [column for column in required if column not in df.columns]
        if missing:
            raise ValueError(f"DataFrame is missing required columns: {missing}.")
        if df[id_col].isna().any():
            raise ValueError("ID values must not be missing.")

    def fit(
        self,
        reference: pd.DataFrame,
        feature_cols: list[str],
        id_col: str = "unique_id",
        target_col: str = "y",
        prediction_col: str = "y_pred",
    ) -> "DirectLossAnalyzer":
        """Fit one loss estimator and held-out baseline per reference ID.

        Parameters
        ----------
        reference : pandas.DataFrame
            Labeled reference panel in long format.
        feature_cols : list of str
            Distinct numeric feature columns supplied to the loss model.
        id_col : str, default="unique_id"
            Column identifying independent panel groups.
        target_col : str, default="y"
            Column with observed targets.
        prediction_col : str, default="y_pred"
            Column with monitored model predictions or class-1 probabilities.

        Returns
        -------
        DirectLossAnalyzer
            Fitted analyzer.

        Raises
        ------
        TypeError
            If ``reference`` is not a pandas DataFrame.
        ValueError
            If the panel, column choices, or ``fraction`` are invalid, or an ID
            lacks two training rows and one held-out row.
        """
        if not 0 < self.fraction < 1:
            raise ValueError("fraction must lie strictly between 0 and 1.")
        if not feature_cols or len(feature_cols) != len(set(feature_cols)):
            raise ValueError("feature_cols must contain distinct feature names.")
        if set(feature_cols) & {id_col, target_col, prediction_col}:
            raise ValueError("feature_cols must exclude ID, target, and prediction columns.")
        self._validate_frame(
            reference, [id_col, target_col, prediction_col, *feature_cols], id_col
        )
        estimators = {}
        baselines = {}
        for unique_id, group in reference.groupby(id_col, sort=False, observed=True):
            estimators[unique_id], baselines[unique_id] = self._fit_single(
                unique_id, group, feature_cols, target_col, prediction_col
            )

        self.estimators_ = estimators
        self.baselines_ = baselines
        self.feature_cols_ = list(feature_cols)
        self.id_col_ = id_col
        self.target_col_ = target_col
        self.prediction_col_ = prediction_col
        if hasattr(self, "results_"):
            del self.results_
        return self

    def _fit_single(self, unique_id, group, feature_cols, target_col, prediction_col):
        """Fit one ID's loss model and calculate its held-out baseline."""
        split = int(np.floor(len(group) * (1 - self.fraction)))
        if split < 2 or split == len(group):
            raise ValueError(
                f"ID {unique_id!r} needs at least two fitting rows and one held-out row."
            )
        train, holdout = group.iloc[:split], group.iloc[split:]
        fitted = clone(self.estimator).fit(
            train[feature_cols], train[target_col], train[prediction_col]
        )
        baseline = {
            "reference_estimated": fitted.estimate(
                holdout[feature_cols], holdout[prediction_col]
            ),
            "reference_realized": fitted.aggregate(
                fitted.observed_loss(holdout[target_col], holdout[prediction_col])
            ),
            "reference_size": len(holdout),
        }
        return fitted, baseline

    def _predict_single(self, unique_id, group, prediction_col):
        """Estimate one current ID's loss relative to its reference baseline."""
        baseline = self.baselines_[unique_id]
        estimated = self.estimators_[unique_id].estimate(
            group[self.feature_cols_], group[prediction_col]
        )
        delta = estimated - baseline["reference_estimated"]
        return {
            **baseline,
            "current_estimated": estimated,
            "estimated_delta": delta,
            "degradation": bool(delta > 0),
            "current_size": len(group),
        }

    def predict(
        self,
        current: pd.DataFrame,
        id_col: str | None = None,
        prediction_col: str | None = None,
    ) -> pd.DataFrame:
        """Estimate current loss and its change from the held-out baseline.

        Parameters
        ----------
        current : pandas.DataFrame
            Unlabeled current panel containing the fitted feature columns.
        id_col : str or None, default=None
            Current identifier column; defaults to the reference column.
        prediction_col : str or None, default=None
            Current prediction column; defaults to the reference column.

        Returns
        -------
        pandas.DataFrame
            One row per current ID in first-appearance order. Includes the
            reference estimated and realized loss, current estimated loss,
            estimated delta, degradation flag, and sample sizes.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If :meth:`fit` has not been called.
        ValueError
            If current data are invalid or contain an ID without a reference.
        """
        check_is_fitted(self, "estimators_")
        id_col = self.id_col_ if id_col is None else id_col
        prediction_col = self.prediction_col_ if prediction_col is None else prediction_col
        self._validate_frame(
            current, [id_col, prediction_col, *self.feature_cols_], id_col
        )
        unknown = [
            value for value in pd.unique(current[id_col])
            if value not in self.estimators_
        ]
        if unknown:
            raise ValueError(f"No reference performance for IDs: {unknown!r}.")

        self.results_ = {
            unique_id: self._predict_single(unique_id, group, prediction_col)
            for unique_id, group in current.groupby(id_col, sort=False, observed=True)
        }
        self.result_id_col_ = id_col
        return self.summary()

    def summary(self) -> pd.DataFrame:
        """Return the most recent per-ID prediction as a table.

        Returns
        -------
        pandas.DataFrame
            Same rows and columns as the latest :meth:`predict` result.

        Columns
        -------
        **reference_estimated**, **reference_realized** : ``float``
            Estimated and observed loss on held-out reference rows.
        **current_estimated**, **estimated_delta** : ``float``
            Estimated current loss and its difference from the estimated
            reference baseline.
        **degradation** : ``bool``
            Whether the estimated delta is positive.
        **reference_size**, **current_size** : ``int``
            Number of held-out reference and current rows.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If :meth:`predict` has not been called since the latest fit.
        """
        check_is_fitted(self, "results_")
        rows = [
            {self.result_id_col_: unique_id, **result}
            for unique_id, result in self.results_.items()
        ]
        return pd.DataFrame(rows)
