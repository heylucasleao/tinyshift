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
    def _quantile_column(quantile: float) -> str:
        """Return the column name emitted by ``PanelPredictiveForecast.ppf``."""
        label = np.format_float_positional(float(quantile), precision=12, trim="-")
        return f"Q({label})"

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

    @staticmethod
    def _quantile_coverage(distribution, n_observations: int) -> tuple[np.ndarray, np.ndarray]:
        """Return validated forecast quantiles and their attainable coverage."""
        levels = np.arange(1, 20, dtype=float) / 20.0
        forecast_quantiles = np.asarray(distribution.ppf(levels), dtype=float)
        expected_shape = (n_observations, len(levels))
        if forecast_quantiles.shape != expected_shape or not np.all(
            np.isfinite(forecast_quantiles)
        ):
            raise ValueError(
                "The predictive distribution returned invalid or misaligned quantiles."
            )

        attainable_coverage = np.asarray(
            distribution.cdf(forecast_quantiles), dtype=float
        )
        if (
            attainable_coverage.shape != expected_shape
            or not np.all(np.isfinite(attainable_coverage))
            or np.any((attainable_coverage < 0.0) | (attainable_coverage > 1.0))
        ):
            raise ValueError(
                "The predictive distribution returned invalid quantile coverage."
            )
        return forecast_quantiles, attainable_coverage

    @staticmethod
    def _calibration_errors(
        y_true: np.ndarray,
        series_ids: np.ndarray,
        forecast_quantiles: np.ndarray,
        attainable_coverage: np.ndarray,
    ) -> dict:
        """Calculate mean absolute quantile-calibration error per series."""
        errors = {}
        for unique_id in pd.unique(series_ids):
            positions = np.flatnonzero(series_ids == unique_id)
            observed_coverage = np.mean(
                y_true[positions, None] <= forecast_quantiles[positions], axis=0
            )
            expected_coverage = np.mean(attainable_coverage[positions], axis=0)
            errors[unique_id] = float(
                np.mean(np.abs(observed_coverage - expected_coverage))
            )
        return errors

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
    def _aggregate_distribution_scores(
        series_ids,
        row_crps: np.ndarray,
        train_scales: pd.Series,
        id_col: str,
    ) -> pd.DataFrame:
        """Aggregate row CRPS and normalize each series by its training scale."""
        row_scores = pd.DataFrame({id_col: np.asarray(series_ids), "crps": row_crps})
        per_series = (
            row_scores.groupby(id_col, observed=True, sort=False)["crps"]
            .agg(crps="mean", n_observations="size")
            .reset_index()
        )
        per_series["target_std"] = per_series[id_col].map(train_scales)
        valid_scale = np.isfinite(per_series["target_std"]) & (
            per_series["target_std"] > 0.0
        )
        per_series["ncrps"] = np.where(
            valid_scale,
            per_series["crps"] / per_series["target_std"],
            np.nan,
        )
        return per_series[[id_col, "crps", "target_std", "ncrps", "n_observations"]]

    @classmethod
    def evaluate_distribution(
        cls,
        forecast,
        evaluation_df: pd.DataFrame,
        train_df: pd.DataFrame,
        target_col: str = "y",
        id_col: str = "unique_id",
        time_col: str = "ds",
    ) -> pd.DataFrame:
        """Evaluate a predictive distribution with CRPS and per-series nCRPS.

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
        Returns
        -------
        pandas.DataFrame
            One row per evaluated series with mean CRPS, training-target
            standard deviation, nCRPS, mean absolute calibration error, and
            number of evaluated observations.
            nCRPS is undefined when the series is absent from training or its
            training standard deviation is zero or non-finite.

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
        **calibration_error** : ``float``
            Mean absolute difference between observed and attainable quantile
            coverage over the internal 5%-to-95% grid.
        **n_observations** : ``int``
            Number of evaluated forecast-target pairs for the series.
        """
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
        scores = cls._aggregate_distribution_scores(
            evaluation_df[id_col], row_crps, train_scales, id_col
        )

        distribution = forecast.distribution
        forecast_quantiles, attainable_coverage = cls._quantile_coverage(
            distribution, len(y_true)
        )
        series_ids = evaluation_df[id_col].to_numpy()
        calibration_errors = cls._calibration_errors(
            y_true,
            series_ids,
            forecast_quantiles,
            attainable_coverage,
        )
        scores["calibration_error"] = scores[id_col].map(calibration_errors)
        return scores[
            [
                id_col,
                "crps",
                "target_std",
                "ncrps",
                "calibration_error",
                "n_observations",
            ]
        ]

    @classmethod
    def evaluate_interval(
        cls,
        df_res: pd.DataFrame,
        target_col: str = "y",
        quantiles: tuple = (0.05, 0.50, 0.95),
        id_col: str = "unique_id",
    ) -> pd.DataFrame:
        """Evaluate central intervals over out-of-sample backtest predictions.

        Results are calculated independently per series when ``id_col`` is
        present. If it is absent, a single panel-wide result is returned for
        backward compatibility.

        Parameters
        ----------
        df_res : pandas.DataFrame
            DataFrame containing real ground truth targets and forecasted
            quantile columns named as ``Q(<probability>)``.
        target_col : str, default='y'
            Name of the column containing real observed values.
        quantiles : list of float, default=[0.05, 0.50, 0.95]
            Quantile levels used to construct symmetric central intervals.
        id_col : str, default="unique_id"
            Series identifier. Evaluation is panel-wide when this column is
            absent from ``df_res``.

        Returns
        -------
        pandas.DataFrame
            Summary containing empirical coverage, lower and upper miss rates,
            mean interval width, MWIS, and observation count for every
            available interval, independently per series when possible.

        Columns
        -------
        **id_col** : ``object``
            Series identifier when ``id_col`` is present in ``df_res``; omitted
            for a panel-wide evaluation.
        **level** : ``float``
            Nominal central interval coverage.
        **coverage_rate** : ``float``
            Fraction of valid observations inside the interval, including its
            boundaries.
        **lower_miss_rate** : ``float``
            Fraction of valid observations below the lower interval bound.
        **upper_miss_rate** : ``float``
            Fraction of valid observations above the upper interval bound.
        **interval_width_mean** : ``float``
            Mean upper bound minus lower bound.
        **mwis** : ``float``
            Mean Winkler interval score at the reported level; lower is better.
        **n_observations** : ``int``
            Number of valid target-lower-upper triples used in the row.
        """
        results = []

        if target_col not in df_res.columns:
            raise KeyError(
                f"Target column '{target_col}' not found in the input DataFrame."
            )

        quantiles = tuple(sorted(quantiles))
        for q in quantiles:
            if not np.isfinite(q) or not 0 < q < 1:
                raise ValueError(
                    "Quantiles must be finite and strictly between 0 and 1."
                )

        for lower_quantile in quantiles:
            if lower_quantile >= 0.5:
                continue
            upper_quantile = next(
                (
                    quantile
                    for quantile in quantiles
                    if np.isclose(quantile, 1.0 - lower_quantile)
                ),
                None,
            )
            if upper_quantile is None:
                continue

            lower_col = cls._quantile_column(lower_quantile)
            upper_col = cls._quantile_column(upper_quantile)
            if lower_col not in df_res.columns or upper_col not in df_res.columns:
                continue

            alpha = 2.0 * lower_quantile
            target_coverage = 1.0 - alpha

            if id_col in df_res.columns:
                groups = df_res.groupby(id_col, observed=True, sort=False)
            else:
                groups = [(None, df_res)]

            for unique_id, group in groups:
                valid = group[[target_col, lower_col, upper_col]].dropna()
                if valid.empty:
                    empirical_coverage = np.nan
                    lower_miss_rate = np.nan
                    upper_miss_rate = np.nan
                    interval_width = np.nan
                else:
                    lower_misses = valid[target_col] < valid[lower_col]
                    upper_misses = valid[target_col] > valid[upper_col]
                    empirical_coverage = float((~lower_misses & ~upper_misses).mean())
                    lower_miss_rate = float(lower_misses.mean())
                    upper_miss_rate = float(upper_misses.mean())
                    interval_width = float((valid[upper_col] - valid[lower_col]).mean())

                result = {
                    "level": target_coverage,
                    "coverage_rate": round(empirical_coverage, 4),
                    "lower_miss_rate": round(lower_miss_rate, 4),
                    "upper_miss_rate": round(upper_miss_rate, 4),
                    "interval_width_mean": round(interval_width, 4),
                    "mwis": round(
                        cls.mwis(
                            group[target_col].values,
                            group[lower_col].values,
                            group[upper_col].values,
                            alpha,
                        ),
                        4,
                    ),
                    "n_observations": len(valid),
                }
                if unique_id is not None:
                    result = {id_col: unique_id, **result}
                results.append(result)

        columns = [
            "level",
            "coverage_rate",
            "lower_miss_rate",
            "upper_miss_rate",
            "interval_width_mean",
            "mwis",
            "n_observations",
        ]
        if id_col in df_res.columns:
            columns.insert(0, id_col)
        return pd.DataFrame(results, columns=columns)
