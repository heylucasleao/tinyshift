# Copyright (c) 2024-2025 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License

import numpy as np
import pandas as pd
from numpy.polynomial.legendre import leggauss


def _require_columns(
    frame: pd.DataFrame, columns: list | tuple, frame_name: str
) -> None:
    """Raise a consistent error when required dataframe columns are absent."""
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"Columns not found in {frame_name}: {missing}")


class FirstStageForecasterEvaluator:
    r"""Evaluator utility for the first stage conditional expectation (lambda_t).

    Notes on Metrics & Interpretation
    -------------------------------
    - **WAPE**: Total absolute error divided by total observed demand. Lower is better.
    - **Score**: Composite operational loss defined as WAPE + |PBias|. Lower is better.
    - **Forecast Instability**: Relative change between consecutive forecasts. Lower is better.
        - **PBias (Bias)**: Measures the fractional global volume deviation ($\frac{\sum \hat{\lambda} - \sum y}{\sum y}$).
            * *Interpretation*: Should be close to 0. A negative bias indicates overall under-forecasting (risk of stockouts),
        while a positive bias indicates over-forecasting (excess holding costs).
    - **False Demand on Zero-Days (Avg Pred)**: Average predicted $\lambda_t$ specifically on days where true demand is strictly zero ($y = 0$).
      * *Interpretation*: Measures the model's tendency to "smear" or leak intermittent demand into non-active periods,
        creating false expectations of activity.
    - **Peak Demand Deviation**: Fractional error of predicted values relative to true values restricted to periods of positive/peak demand ($y > 0$).
      * *Interpretation*: Tracks the model's smoothing bias on positive-demand days. Negative values indicate that the model
        under-forecasts realized peaks. Since this conditions on the observed target, it is an operational diagnostic rather
        than a direct test of conditional-mean calibration.
    """

    @classmethod
    def evaluate(
        cls,
        df_res: pd.DataFrame,
        target_col: str = "y",
        lambda_col: str = "lambda_t",
        id_col: str = "unique_id",
        time_col: str = "ds",
    ) -> pd.DataFrame:
        """Evaluate the operational quality of out-of-sample mean forecasts.

        Returns
        -------
        pandas.DataFrame
            One-row operational evaluation summary.

        Columns
        -------
        **wape** : ``float``
            Total absolute error divided by total observed demand.
        **pbias** : ``float``
            Aggregate predicted volume minus observed volume, divided by
            observed volume.
        **score** : ``float``
            Composite operational loss computed as ``wape + abs(pbias)``.
        **forecast_instability** : ``float``
            Relative revisions between adjacent forecasts within each series.
        **false_demand_on_zero_days_avg_pred** : ``float``
            Mean prediction on observations whose target is zero.
        **peak_demand_deviation** : ``float``
            Relative difference between mean predicted and observed demand on
            positive-target observations.

        Notes
        -----
        Input predictions should come from temporal cross-validation or a held-
        out period. Evaluating fitted values would give optimistic results.
        Rows must already be ordered chronologically within each series; the
        evaluator does not reorder them.
        """
        required = [target_col, lambda_col, id_col, time_col]
        _require_columns(df_res, required, "the input DataFrame")

        valid = df_res[required].dropna().copy()
        if valid.empty:
            raise ValueError("No valid target/prediction pairs were found.")

        y_true = valid[target_col].to_numpy(dtype=float)
        y_pred = valid[lambda_col].to_numpy(dtype=float)
        cls._validate_mean_inputs(y_true, y_pred, lambda_col)

        total_true = np.sum(y_true)
        total_pred = np.sum(y_pred)
        total_abs_error = np.sum(np.abs(y_pred - y_true))
        if total_true > 0:
            wape = total_abs_error / total_true
            pbias = (total_pred - total_true) / total_true
        else:
            wape = 0.0 if total_abs_error == 0 else np.nan
            pbias = 0.0 if total_pred == 0 else np.nan

        zero_mask = y_true == 0
        pos_mask = y_true > 0

        false_alarm_zeros = np.mean(y_pred[zero_mask]) if np.sum(zero_mask) > 0 else 0.0
        peak_underestimation = (
            (np.mean(y_pred[pos_mask]) - np.mean(y_true[pos_mask]))
            / np.mean(y_true[pos_mask])
            if np.sum(pos_mask) > 0
            else 0.0
        )

        return pd.DataFrame(
            {
                "wape": [round(wape, 4)],
                "pbias": [round(pbias, 4)],
                "score": [round(wape + abs(pbias), 4)],
                "forecast_instability": [
                    round(
                        cls._forecast_instability(
                            valid,
                            lambda_col=lambda_col,
                            id_col=id_col,
                        ),
                        4,
                    )
                ],
                "false_demand_on_zero_days_avg_pred": [round(false_alarm_zeros, 4)],
                "peak_demand_deviation": [round(peak_underestimation, 4)],
            }
        )

    @staticmethod
    def _validate_mean_inputs(
        y_true: np.ndarray, y_pred: np.ndarray, prediction_name: str
    ) -> None:
        if not np.all(np.isfinite(y_true)) or not np.all(np.isfinite(y_pred)):
            raise ValueError("Target and prediction values must be finite.")
        if np.any(y_true < 0):
            raise ValueError("Target values must be non-negative.")
        if np.any(y_pred <= 0):
            raise ValueError(
                f"Conditional mean column '{prediction_name}' must be strictly positive."
            )

    @staticmethod
    def _forecast_instability(
        df_res: pd.DataFrame,
        lambda_col: str,
        id_col: str,
    ) -> float:
        previous = df_res.groupby(id_col, observed=True)[lambda_col].shift(1)
        current = df_res[lambda_col]
        paired = previous.notna()
        if not paired.any():
            return np.nan

        prev_values = previous[paired].to_numpy(dtype=float)
        curr_values = current[paired].to_numpy(dtype=float)
        average_volume = 0.5 * (prev_values.sum() + curr_values.sum())
        if average_volume == 0:
            return 0.0
        revisions = prev_values - curr_values
        return float((np.abs(revisions).sum() + abs(revisions.sum())) / average_volume)

    @classmethod
    def calibration_table(
        cls,
        df_res: pd.DataFrame,
        target_col: str = "y",
        lambda_col: str = "lambda_t",
        n_bins: int = 10,
    ) -> pd.DataFrame:
        """Compare observed and predicted means across quantile-based bins.

        Returns
        -------
        pandas.DataFrame
            Calibration summary with one row per realized prediction bin.

        Columns
        -------
        **calibration_bin** : ``object``
            Quantile interval of the predictions, or ``"all"`` when every
            prediction is identical.
        **count** : ``int``
            Number of valid target-prediction pairs in the bin.
        **mean_prediction** : ``float``
            Mean conditional prediction in the bin.
        **mean_observed** : ``float``
            Mean observed target in the bin.
        **mean_residual** : ``float``
            Mean observed target minus mean prediction.
        """
        if not isinstance(n_bins, int) or n_bins < 2:
            raise ValueError("n_bins must be an integer greater than or equal to 2.")
        _require_columns(df_res, (target_col, lambda_col), "the input DataFrame")

        valid = df_res[[target_col, lambda_col]].dropna().copy()
        if valid.empty:
            raise ValueError("No valid target/prediction pairs were found.")
        cls._validate_mean_inputs(
            valid[target_col].to_numpy(dtype=float),
            valid[lambda_col].to_numpy(dtype=float),
            lambda_col,
        )
        if valid[lambda_col].nunique() == 1:
            valid["calibration_bin"] = "all"
        else:
            valid["calibration_bin"] = pd.qcut(
                valid[lambda_col], q=n_bins, duplicates="drop"
            )
        result = (
            valid.groupby("calibration_bin", observed=True)
            .agg(
                count=(target_col, "size"),
                mean_prediction=(lambda_col, "mean"),
                mean_observed=(target_col, "mean"),
            )
            .reset_index()
        )
        result["mean_residual"] = result["mean_observed"] - result["mean_prediction"]
        return result


class TwoStageForecasterEvaluator:
    r"""Evaluator utility for complete probabilistic forecasts.

    A pair of symmetric forecast quantiles, such as ``Q(0.05)`` and
    ``Q(0.95)``,
    defines a central interval with coverage $1 - \alpha$. Evaluation reports
    its empirical coverage, mean width, and mean Winkler interval score
    (MWIS). Lower MWIS values indicate sharper, better-calibrated intervals.
    Full predictive distributions can also be evaluated with CRPS and nCRPS.
    """

    @staticmethod
    def mwis(
        y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float
    ) -> float:
        """Compute the mean Winkler interval score for a central interval."""
        y_true = np.asarray(y_true, dtype=float)
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)
        valid = ~(np.isnan(y_true) | np.isnan(lower) | np.isnan(upper))
        if not valid.any():
            return np.nan

        y_true, lower, upper = y_true[valid], lower[valid], upper[valid]
        width = upper - lower
        penalty_lower = (2.0 / alpha) * (lower - y_true) * (y_true < lower)
        penalty_upper = (2.0 / alpha) * (y_true - upper) * (y_true > upper)
        return float(np.mean(width + penalty_lower + penalty_upper))

    @staticmethod
    def _align_targets(
        y_true: pd.DataFrame,
        forecast_frame: pd.DataFrame,
        id_col: str,
        time_col: str,
        target_col: str,
    ) -> pd.DataFrame:
        """Align observed targets to forecast order using panel keys."""
        if not isinstance(y_true, pd.DataFrame):
            raise TypeError("y_true must be a pandas DataFrame.")

        keys = [id_col, time_col]
        required_true = [*keys, target_col]
        for frame, required, name in (
            (y_true, required_true, "y_true"),
            (forecast_frame, keys, "forecast"),
        ):
            missing = [column for column in required if column not in frame.columns]
            if missing:
                raise KeyError(f"Columns not found in {name}: {missing}")
            if frame.duplicated(keys).any():
                raise ValueError(f"{name} contains duplicate identifier/time rows.")

        aligned = forecast_frame.merge(
            y_true[required_true], on=keys, how="left", validate="one_to_one"
        )
        if aligned[target_col].isna().any():
            raise ValueError("y_true must contain a target for every forecast row.")
        try:
            observed = aligned[target_col].to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("y_true must contain numeric target values.") from exc
        if not np.all(np.isfinite(observed)):
            raise ValueError("y_true must contain only finite target values.")
        return aligned

    @classmethod
    def _evaluate_interval_bounds(
        cls,
        observed: np.ndarray,
        bounds: np.ndarray,
        coverage: float,
    ) -> dict:
        """Calculate summary metrics for one central prediction interval."""
        lower, upper = bounds[:, 0], bounds[:, 1]
        return {
            "coverage": coverage,
            "coverage_rate": round(
                float(((observed >= lower) & (observed <= upper)).mean()), 4
            ),
            "interval_width_mean": round(float(np.mean(upper - lower)), 4),
            "mwis": round(cls.mwis(observed, lower, upper, 1.0 - coverage), 4),
            "n_obs": len(observed),
        }

    @staticmethod
    def _crps(distribution, y_true: np.ndarray) -> np.ndarray:
        """Approximate row-wise CRPS from the predictive quantile function."""
        nodes, weights = leggauss(100)
        probabilities = 0.5 * (nodes + 1.0)
        weights = 0.5 * weights
        quantiles = np.asarray(distribution.ppf(probabilities), dtype=float)
        expected_shape = (len(y_true), len(probabilities))
        if quantiles.shape != expected_shape:
            raise ValueError(
                "The predictive distribution is not aligned with evaluation_df."
            )

        errors = y_true[:, None] - quantiles
        quantile_loss = np.where(
            errors >= 0.0,
            probabilities * errors,
            (probabilities - 1.0) * errors,
        )
        return 2.0 * np.sum(quantile_loss * weights, axis=1)

    @staticmethod
    def _numeric_target(
        frame: pd.DataFrame,
        target_col: str,
        frame_name: str,
        require_finite: bool = False,
    ) -> np.ndarray:
        """Extract a numeric target and optionally require finite values."""
        try:
            target = frame[target_col].to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Target values in {frame_name} must be numeric.") from exc
        if require_finite and not np.all(np.isfinite(target)):
            raise ValueError(f"Target values in {frame_name} must be finite.")
        return target

    @staticmethod
    def _validate_distribution_alignment(
        forecast_frame: pd.DataFrame,
        evaluation_df: pd.DataFrame,
        id_col: str,
        time_col: str,
    ) -> None:
        """Require the evaluation rows to match the forecast panel exactly."""
        if len(evaluation_df) != len(forecast_frame):
            raise ValueError(
                "evaluation_df and forecast must contain the same number of rows."
            )

        forecast_keys = forecast_frame[[id_col, time_col]].reset_index(drop=True)
        evaluation_keys = evaluation_df[[id_col, time_col]].reset_index(drop=True)
        if not forecast_keys.equals(evaluation_keys):
            raise ValueError(
                "evaluation_df series and timestamps must be aligned with forecast."
            )

    @classmethod
    def _target_scales(
        cls,
        train_df: pd.DataFrame,
        target_col: str,
        id_col: str,
    ) -> pd.Series:
        """Calculate the sample target standard deviation for each series."""
        targets = cls._numeric_target(train_df, target_col, "train_df")
        scale_frame = pd.DataFrame(
            {id_col: train_df[id_col].to_numpy(), target_col: targets}
        )
        return scale_frame.groupby(id_col, observed=True)[target_col].std()

    @staticmethod
    def _distribution_row_scores(
        evaluation_df: pd.DataFrame,
        row_crps: np.ndarray,
        train_scales: pd.Series,
        id_col: str,
        time_col: str,
    ) -> pd.DataFrame:
        """Build unaggregated CRPS and nCRPS rows with inferred horizons."""
        row_scores = evaluation_df[[id_col, time_col]].copy()
        row_scores["horizon"] = (
            row_scores.groupby(id_col, observed=True, sort=False).cumcount() + 1
        )
        row_scores["crps"] = row_crps
        row_scores["target_std"] = row_scores[id_col].map(train_scales)
        valid_scale = np.isfinite(row_scores["target_std"]) & (
            row_scores["target_std"] > 0.0
        )
        row_scores["ncrps"] = np.where(
            valid_scale,
            row_scores["crps"] / row_scores["target_std"],
            np.nan,
        )
        return row_scores

    @staticmethod
    def _aggregate_distribution_scores(
        row_scores: pd.DataFrame,
        agg: str | None,
        id_col: str,
    ) -> pd.DataFrame:
        """Aggregate row-level distribution scores at the requested level."""
        if agg is None:
            return row_scores.reset_index(drop=True)

        group_columns = {
            "series": [id_col],
            "horizon": ["horizon"],
            "overall": [],
        }
        if agg not in group_columns:
            raise ValueError(
                "agg must be one of: 'series', 'horizon', 'overall', or None."
            )

        columns = group_columns[agg]
        if columns:
            result = (
                row_scores.groupby(columns, observed=True, sort=False)
                .agg(
                    crps=("crps", "mean"),
                    ncrps=("ncrps", "mean"),
                    n_obs=("crps", "size"),
                )
                .reset_index()
            )
        else:
            result = pd.DataFrame(
                {
                    "crps": [row_scores["crps"].mean()],
                    "ncrps": [row_scores["ncrps"].mean()],
                    "n_obs": [len(row_scores)],
                }
            )

        if agg == "series":
            scales = row_scores.groupby(id_col, observed=True, sort=False)[
                "target_std"
            ].first()
            result.insert(2, "target_std", result[id_col].map(scales))
            return result[[id_col, "crps", "target_std", "ncrps", "n_obs"]]
        return result

    @classmethod
    def evaluate_distribution(
        cls,
        forecast,
        evaluation_df: pd.DataFrame,
        train_df: pd.DataFrame,
        target_col: str = "y",
        id_col: str = "unique_id",
        time_col: str = "ds",
        agg: str | None = "series",
    ) -> pd.DataFrame:
        """Evaluate a predictive distribution with CRPS and nCRPS.

        Parameters
        ----------
        forecast : PanelPredictiveForecast
            Row-aligned predictive forecast to evaluate.
        evaluation_df : pandas.DataFrame
            Realized targets. Its series and timestamps must have the same row
            order as the forecast.
        train_df : pandas.DataFrame
            Training observations used to calculate each series' target
            standard deviation without using evaluation data.
        target_col : str, default='y'
            Target column present in both dataframes.
        id_col : str, default='unique_id'
            Series identifier column.
        time_col : str, default='ds'
            Timestamp column used to validate positional alignment.
        agg : {'series', 'horizon', 'overall', None}, default='series'
            Aggregation level. ``'series'`` preserves the original one-row-per-
            series result. ``'horizon'`` aggregates series at each inferred
            forecast step, ``'overall'`` returns one row, and ``None`` returns
            one row per forecast observation. Horizons are inferred from row
            order within each series, starting at one.
        Returns
        -------
        pandas.DataFrame
            CRPS and nCRPS at the requested aggregation level. nCRPS is
            undefined for observations whose series is absent from training or
            has a zero or non-finite training standard deviation.

        Columns
        -------
        **id_col** : ``object``
            Series identifier using the resolved ``id_col`` name.
        **crps** : ``float``
            Mean Continuous Ranked Probability Score; lower is better.
        **target_std** : ``float``
            Sample standard deviation of the series in ``train_df``.
        **ncrps** : ``float``
            CRPS divided by ``target_std``; undefined for a non-positive or
            non-finite scale.
        **n_obs** : ``int``
            Number of evaluated forecast-target pairs for the series.
        """
        if agg not in {"series", "horizon", "overall", None}:
            raise ValueError(
                "agg must be one of: 'series', 'horizon', 'overall', or None."
            )
        _require_columns(evaluation_df, (id_col, time_col, target_col), "evaluation_df")
        _require_columns(train_df, (id_col, target_col), "train_df")
        forecast_frame = forecast.to_frame()
        _require_columns(forecast_frame, (id_col, time_col), "forecast")
        cls._validate_distribution_alignment(
            forecast_frame, evaluation_df, id_col, time_col
        )
        y_true = cls._numeric_target(
            evaluation_df, target_col, "evaluation_df", require_finite=True
        )
        row_crps = cls._crps(forecast.distribution, y_true)
        train_scales = cls._target_scales(train_df, target_col, id_col)
        row_scores = cls._distribution_row_scores(
            evaluation_df, row_crps, train_scales, id_col, time_col
        )
        return cls._aggregate_distribution_scores(row_scores, agg, id_col)

    @classmethod
    def evaluate_interval(
        cls,
        y_true: pd.DataFrame,
        forecast,
        coverages=(0.5, 0.8, 0.9, 0.95),
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
    ) -> pd.DataFrame:
        """Evaluate distribution-derived central intervals over a panel.

        Results are calculated independently per series when ``id_col`` is
        present. If it is absent, a single panel-wide result is returned for
        backward compatibility.

        Parameters
        ----------
        y_true : pandas.DataFrame
            Observed panel containing identifier, time, and target columns.
        forecast : PanelPredictiveForecast
            Panel forecast exposing ``distribution`` and ``to_frame``.
        coverages : iterable of float, default=(0.5, 0.8, 0.9, 0.95)
            Nominal coverage levels used to derive central intervals.
        id_col : str, default="unique_id"
            Series identifier column.
        time_col : str, default="ds"
            Time column.
        target_col : str, default="y"
            Target column in ``y_true``.

        Returns
        -------
        pandas.DataFrame
            One evaluation row per requested coverage.

        Columns
        -------
        **model** : ``object``
            Identifier carried by the predictive forecast.
        **coverage** : ``float``
            Requested nominal interval coverage.
        **coverage_rate** : ``float``
            Fraction of aligned targets inside their interval bounds.
        **interval_width_mean** : ``float``
            Mean upper-minus-lower interval width.
        **mwis** : ``float``
            Mean Winkler interval score; lower values are better.
        **n_obs** : ``int``
            Number of aligned panel observations.

        Notes
        -----
        Targets are aligned to forecast order by ``id_col`` and ``time_col``;
        both inputs must contain unique panel keys. Every forecast row must
        have one finite numeric target. Central equal-tailed bounds are derived
        directly from ``forecast.distribution``, so precomputed interval
        columns and interval specifications are not accepted.

        Coverage includes targets equal to either boundary. Metrics are pooled
        over all aligned panel rows and rounded to four decimal places.
        """
        if not hasattr(forecast, "distribution") or not hasattr(forecast, "to_frame"):
            raise TypeError("forecast must be a panel predictive forecast.")

        aligned = cls._align_targets(
            y_true, forecast.to_frame(), id_col, time_col, target_col
        )
        observed = aligned[target_col].to_numpy(dtype=float)
        if len(forecast.distribution) != len(observed):
            raise ValueError(
                "y_true and the predictive distribution must have equal length."
            )

        records = []
        for coverage in coverages:
            bounds = np.asarray(forecast.distribution.interval(coverage), dtype=float)
            metrics = cls._evaluate_interval_bounds(observed, bounds, coverage)
            records.append({"model": forecast.model, **metrics})
        return pd.DataFrame.from_records(
            records,
            columns=[
                "model",
                "coverage",
                "coverage_rate",
                "interval_width_mean",
                "mwis",
                "n_obs",
            ],
        )
