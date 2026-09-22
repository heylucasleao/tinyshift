# Copyright (c) 2024-2026 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License

"""Panel adapter for direct loss estimation."""

from dataclasses import asdict

import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils.validation import check_is_fitted

from .dle import DirectLossEstimator


class DirectLossAnalyzer(BaseEstimator):
    """Estimate model loss independently for each panel ID.

    The analyzer clones one :class:`DirectLossEstimator` per reference ID and
    delegates its reference split, baseline, and current comparison to that
    estimator. Current targets are not required.

    Parameters
    ----------
    estimator : DirectLossEstimator or None, default=None
        Estimator template cloned for each ID. By default, use a DLE with a
        random forest regressor as its learner.
    fraction : float or None, default=None
        Optional override for the estimator's held-out reference fraction.
        If omitted, use the fraction configured on ``estimator``.

    Attributes
    ----------
    estimators_ : dict
        Fitted DLE for each reference ID.
    results_ : dict
        Most recent :class:`DirectLossResult` for each ID.

    Notes
    -----
    Reference rows retain their input order. For time series, order rows
    chronologically within each ID before fitting. ``degradation`` is based on
    a one-sided permutation test of increased mean estimated loss; it does not
    confirm realized degradation.

    Examples
    --------
    >>> analyzer = DirectLossAnalyzer(fraction=0.25)
    >>> analyzer.fit(reference_df, feature_cols=["feature_a", "feature_b"])
    >>> result = analyzer.predict(current_df)
    """

    def __init__(
        self,
        estimator: DirectLossEstimator | None = None,
        fraction: float | None = None,
    ) -> None:
        if estimator is not None and not isinstance(estimator, DirectLossEstimator):
            raise TypeError("estimator must be a DirectLossEstimator.")
        self.estimator = (
            DirectLossEstimator(
                learner=RandomForestRegressor(
                    n_estimators=100, min_samples_leaf=3, random_state=42
                ),
                fraction=0.25,
                random_state=42,
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
            lacks two training rows and two held-out rows.
        """
        if not feature_cols or len(feature_cols) != len(set(feature_cols)):
            raise ValueError("feature_cols must contain distinct feature names.")
        if set(feature_cols) & {id_col, target_col, prediction_col}:
            raise ValueError(
                "feature_cols must exclude ID, target, and prediction columns."
            )
        self._validate_frame(
            reference, [id_col, target_col, prediction_col, *feature_cols], id_col
        )
        estimators = {}
        for unique_id, group in reference.groupby(id_col, sort=False, observed=True):
            estimators[unique_id] = self._fit_single(
                unique_id, group, feature_cols, target_col, prediction_col
            )

        self.estimators_ = estimators
        self.feature_cols_ = list(feature_cols)
        self.id_col_ = id_col
        self.target_col_ = target_col
        self.prediction_col_ = prediction_col
        if hasattr(self, "results_"):
            del self.results_
        return self

    def _fit_single(self, unique_id, group, feature_cols, target_col, prediction_col):
        """Fit an independent DLE for one reference ID."""
        fitted = clone(self.estimator)
        if self.fraction is not None:
            fitted.set_params(fraction=self.fraction)
        try:
            return fitted.fit(
                group[feature_cols], group[target_col], group[prediction_col]
            )
        except ValueError as error:
            if "Reference needs at least two fitting rows" in str(error):
                raise ValueError(f"ID {unique_id!r}: {error}") from error
            raise

    def _predict_single(self, unique_id, group, prediction_col, degradation_margin):
        """Delegate one current ID's comparison to its fitted DLE."""
        return self.estimators_[unique_id].predict(
            group[self.feature_cols_],
            group[prediction_col],
            degradation_margin=degradation_margin,
        )

    def predict(
        self,
        current: pd.DataFrame,
        id_col: str | None = None,
        prediction_col: str | None = None,
        degradation_margin: float = 0.0,
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
        degradation_margin : float, default=0.0
            Relative increase tested independently for each current ID.

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
            If current data are invalid, contain an ID without a reference,
            or contain fewer than two observations for any current ID.
        """
        check_is_fitted(self, "estimators_")
        id_col = self.id_col_ if id_col is None else id_col
        prediction_col = (
            self.prediction_col_ if prediction_col is None else prediction_col
        )
        self._validate_frame(
            current, [id_col, prediction_col, *self.feature_cols_], id_col
        )
        unknown = [
            value
            for value in pd.unique(current[id_col])
            if value not in self.estimators_
        ]
        if unknown:
            raise ValueError(f"No reference performance for IDs: {unknown!r}.")

        self.results_ = {
            unique_id: self._predict_single(
                unique_id, group, prediction_col, degradation_margin
            )
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
        **relative_delta** : ``float``
            Estimated change relative to reference estimated loss.
        **degradation_margin** : ``float``
            Relative increase tested by the permutation test.
        **p_value** : ``float``
            One-sided Monte Carlo p-value for exceeding the margin.
        **degradation** : ``bool``
            Whether relative delta exceeds the margin and p-value <= alpha.
        **reference_size**, **current_size** : ``int``
            Number of held-out reference and current rows.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If :meth:`predict` has not been called since the latest fit.
        """
        check_is_fitted(self, "results_")
        rows = [
            {self.result_id_col_: unique_id, **asdict(result)}
            for unique_id, result in self.results_.items()
        ]
        return pd.DataFrame(rows)
