# Copyright (c) 2024-2025 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License

import numpy as np
import pandas as pd


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

        Notes
        -----
        Input predictions should come from temporal cross-validation or a held-
        out period. Evaluating fitted values would give optimistic results.
        """
        required = [target_col, lambda_col, id_col, time_col]
        missing = [col for col in required if col not in df_res.columns]
        if missing:
            raise KeyError(f"Columns not found in the input DataFrame: {missing}")

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
                            time_col=time_col,
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
        time_col: str,
    ) -> float:
        ordered = df_res.sort_values([id_col, time_col])
        previous = ordered.groupby(id_col, observed=True)[lambda_col].shift(1)
        current = ordered[lambda_col]
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
        """Compare observed and predicted means across quantile-based bins."""
        if not isinstance(n_bins, int) or n_bins < 2:
            raise ValueError("n_bins must be an integer greater than or equal to 2.")
        missing = [c for c in (target_col, lambda_col) if c not in df_res.columns]
        if missing:
            raise KeyError(f"Columns not found in the input DataFrame: {missing}")

        valid = df_res[[target_col, lambda_col]].dropna().copy()
        if valid.empty:
            raise ValueError("No valid target/prediction pairs were found.")
        cls._validate_mean_inputs(
            valid[target_col].to_numpy(dtype=float),
            valid[lambda_col].to_numpy(dtype=float),
            lambda_col,
        )
        if valid[lambda_col].nunique() == 1:
            valid["Calibration Bin"] = "all"
        else:
            valid["Calibration Bin"] = pd.qcut(
                valid[lambda_col], q=n_bins, duplicates="drop"
            )
        result = (
            valid.groupby("Calibration Bin", observed=True)
            .agg(
                Count=(target_col, "size"),
                Mean_Prediction=(lambda_col, "mean"),
                Mean_Observed=(target_col, "mean"),
            )
            .reset_index()
        )
        result["Mean_Residual"] = result["Mean_Observed"] - result["Mean_Prediction"]
        return result


class TwoStageForecasterEvaluator:
    r"""Evaluator utility for probabilistic central prediction intervals.

    A pair of symmetric forecast quantiles, such as ``q_05`` and ``q_95``,
    defines a central interval with coverage $1 - \alpha$. Evaluation reports
    its empirical coverage, mean width, and mean Winkler interval score
    (MWIS). Lower MWIS values indicate sharper, better-calibrated intervals.
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

    @classmethod
    def evaluate(
        cls,
        df_res: pd.DataFrame,
        target_col: str = "y",
        quantiles: tuple = (0.05, 0.50, 0.95),
    ) -> pd.DataFrame:
        """Evaluate central intervals over out-of-sample backtest predictions.

        Parameters
        ----------
        df_res : pandas.DataFrame
            DataFrame containing real ground truth targets and forecasted quantile columns (``q_*``).
        target_col : str, default='y'
            Name of the column containing real observed values.
        quantiles : list of float, default=[0.05, 0.50, 0.95]
            Quantile levels used to construct symmetric central intervals.

        Returns
        -------
        pandas.DataFrame
            Summary dataframe containing level, empirical coverage, mean
            interval width, and MWIS for each available interval.
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

            lower_col = f"q_{round(lower_quantile * 100)}"
            upper_col = f"q_{round(upper_quantile * 100)}"
            if lower_col not in df_res.columns or upper_col not in df_res.columns:
                continue

            valid = df_res[[target_col, lower_col, upper_col]].dropna()
            alpha = 2.0 * lower_quantile
            target_coverage = 1.0 - alpha
            if valid.empty:
                empirical_coverage = np.nan
                interval_width = np.nan
            else:
                empirical_coverage = float(
                    (
                        (valid[target_col] >= valid[lower_col])
                        & (valid[target_col] <= valid[upper_col])
                    ).mean()
                )
                interval_width = float((valid[upper_col] - valid[lower_col]).mean())

            results.append(
                {
                    "level": target_coverage,
                    "coverage_rate": round(empirical_coverage, 4),
                    "interval_width_mean": round(interval_width, 4),
                    "mwis": round(
                        cls.mwis(
                            df_res[target_col].values,
                            df_res[lower_col].values,
                            df_res[upper_col].values,
                            alpha,
                        ),
                        4,
                    ),
                }
            )

        return pd.DataFrame(
            results, columns=["level", "coverage_rate", "interval_width_mean", "mwis"]
        )
