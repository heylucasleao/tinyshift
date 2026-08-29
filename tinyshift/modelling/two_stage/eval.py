# Copyright (c) 2024-2025 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License

import numpy as np
import pandas as pd


class FirstStageForecasterEvaluator:
    r"""Evaluator utility for the first stage conditional expectation (lambda_t).

    Notes on Metrics & Interpretation
    -------------------------------
    - **PBias (Percentage Bias)**: Measures the percentage global volume deviation ($\frac{\sum \hat{\lambda} - \sum y}{\sum y} \times 100$).
      * *Interpretation*: Should be close to 0%. A negative bias indicates overall under-forecasting (risk of stockouts),
        while a positive bias indicates over-forecasting (excess holding costs).
    - **False Demand on Zero-Days (Avg Pred)**: Average predicted $\lambda_t$ specifically on days where true demand is strictly zero ($y = 0$).
      * *Interpretation*: Measures the model's tendency to "smear" or leak intermittent demand into non-active periods,
        creating false expectations of activity.
    - **Peak Demand Deviation (%)**: Percentage error of predicted values relative to true values restricted to periods of positive/peak demand ($y > 0$).
      * *Interpretation*: Tracks the model's smoothing bias on high-demand days. Negative values indicate that the model
        under-forecasts peaks, which directly starves the upper tail of the second-stage quantile distributions.
    """

    @classmethod
    def evaluate(
        cls,
        df_res: pd.DataFrame,
        target_col: str = "y",
        lambda_col: str = "lambda_t",
    ) -> pd.DataFrame:
        """Gera métricas de calibração amigáveis e interpretáveis para o usuário final."""
        y_true = df_res[target_col].values
        y_pred = df_res[lambda_col].values

        total_true = np.sum(y_true)
        total_pred = np.sum(y_pred)

        pbias = (
            ((total_pred - total_true) / total_true) * 100 if total_true > 0 else 0.0
        )

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
            "Metrics": {
                "PBias": round(pbias, 2),
                "False Demand on Zero-Days (Avg Pred)": round(false_alarm_zeros, 4),
                "Peak Demand Deviation (%)": round(peak_underestimation, 2),
            }
        }
        return pd.DataFrame(report)


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
                raise ValueError("Quantiles must be finite and strictly between 0 and 1.")

            q_int = int(round(q * 100))
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
