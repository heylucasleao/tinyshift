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
    - **PBias (Percentage Bias)**: Measures the percentage global volume deviation ($\frac{\sum \hat{\lambda} - \sum y}{\sum y} \times 100$).
      * *Interpretation*: Should be close to 0%. A negative bias indicates overall under-forecasting (risk of stockouts),
        while a positive bias indicates over-forecasting (excess holding costs).
    - **False Demand on Zero-Days (Avg Pred)**: Average predicted $\lambda_t$ specifically on days where true demand is strictly zero ($y = 0$).
      * *Interpretation*: Measures the model's tendency to "smear" or leak intermittent demand into non-active periods,
        creating false expectations of activity.
    - **Peak Demand Deviation (%)**: Percentage error of predicted values relative to true values restricted to periods of positive/peak demand ($y > 0$).
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
            wape = total_abs_error / total_true * 100
            pbias = (total_pred - total_true) / total_true * 100
        else:
            wape = 0.0 if total_abs_error == 0 else np.nan
            pbias = 0.0 if total_pred == 0 else np.nan

        zero_mask = y_true == 0
        pos_mask = y_true > 0

        false_alarm_zeros = np.mean(y_pred[zero_mask]) if np.sum(zero_mask) > 0 else 0.0
        peak_underestimation = (
            (np.mean(y_pred[pos_mask]) - np.mean(y_true[pos_mask]))
            / np.mean(y_true[pos_mask])
            * 100
            if np.sum(pos_mask) > 0
            else 0.0
        )

        report = {
            "WAPE": wape,
            "PBias": pbias,
            "Score": wape + abs(pbias),
            "Forecast Instability": cls._forecast_instability(
                valid, lambda_col=lambda_col, id_col=id_col, time_col=time_col
            ),
            "False Demand on Zero-Days (Avg Pred)": false_alarm_zeros,
            "Peak Demand Deviation (%)": peak_underestimation,
        }

        precision = {
            "PBias": 2,
            "Peak Demand Deviation (%)": 2,
        }
        return pd.DataFrame(
            {
                "Metrics": {
                    name: round(value, precision.get(name, 4))
                    for name, value in report.items()
                }
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
        return float(
            (np.abs(revisions).sum() + abs(revisions.sum())) / average_volume * 100
        )

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
    r"""Evaluator utility for probabilistic quantile forecasts.

    Notes on Metrics & Interpretation
    -------------------------------
    - **Pinball Loss (Quantile Loss)**: Asymmetrically penalizes over- and under-predictions based on the target quantile.
      * Lower values indicate sharper and more accurate quantile forecasts. For higher quantiles (e.g., 0.95, 0.99),
        it heavily penalizes under-forecasting (stockouts).
    - **Target Coverage**: The nominal probability level requested (e.g., 0.95 for the 95th percentile).
      * The theoretical frequency with which actual demand should fall at or below the predicted quantile.
    - **Empirical Coverage**: The actual observed frequency of $y_{true} \le q_{\tau}$ in the evaluation set.
      * Shows how well-calibrated the probabilistic tail behavior is in practice.
    - **Coverage Gap**: The arithmetic difference between empirical coverage and target coverage ($\text{Empirical} - \text{Target}$).
      * Ideally close to 0. A negative gap indicates under-coverage (higher stockout risk than planned),
        while a positive gap indicates over-coverage (safer inventory bounds, but potentially excess holding costs).
    """

    @staticmethod
    def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
        """Computes the Pinball Loss (Quantile Loss) for a given target quantile."""
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        # Remove NaNs to prevent metric corruption
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        if y_true_clean.size == 0:
            return np.nan

        err = y_true_clean - y_pred_clean
        return float(np.maximum(quantile * err, (quantile - 1) * err).mean())

    @classmethod
    def evaluate(
        cls,
        df_res: pd.DataFrame,
        target_col: str = "y",
        quantiles: tuple = (0.50, 0.67, 0.95, 0.99),
    ) -> pd.DataFrame:
        """Evaluates empirical coverage and quantile loss over out-of-sample backtest predictions.

        Parameters
        ----------
        df_res : pandas.DataFrame
            DataFrame containing real ground truth targets and forecasted quantile columns (`q_*`).
        target_col : str, default='y'
            Name of the column containing real observed values.
        quantiles : list of float, default=[0.50, 0.67, 0.95, 0.99]
            List of target quantiles evaluated.

        Returns
        -------
        pandas.DataFrame
            Summary dataframe containing Pinball Loss, Target Coverage, Empirical Coverage and Coverage Gap.
        """
        results = {}

        if target_col not in df_res.columns:
            raise KeyError(
                f"Target column '{target_col}' not found in the input DataFrame."
            )

        for q in sorted(quantiles):
            if not np.isfinite(q) or not 0 < q < 1:
                raise ValueError(
                    "Quantiles must be finite and strictly between 0 and 1."
                )

            q_int = round(q * 100)
            col_q = f"q_{q_int}"

            if col_q not in df_res.columns:
                continue

            loss = cls.pinball_loss(
                y_true=df_res[target_col].values,
                y_pred=df_res[col_q].values,
                quantile=q,
            )

            valid = df_res[[target_col, col_q]].dropna()
            empirical_coverage = float((valid[target_col] <= valid[col_q]).mean())
            coverage_gap = empirical_coverage - q

            results[col_q] = {
                "Pinball Loss": round(loss, 4),
                "Target Coverage": q,
                "Empirical Coverage": round(empirical_coverage, 4),
                "Coverage Gap": round(coverage_gap, 4),
            }

        return pd.DataFrame(results).T
