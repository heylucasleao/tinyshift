# Copyright (c) 2024-2025 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License


import numpy as np
import pandas as pd


class ISSMForecastEvaluator:
    """Evaluator utility for probabilistic quantile forecasts."""

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
        quantiles: list = [0.50, 0.67, 0.95, 0.99],
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
            Summary dataframe containing Pinball Loss, Target Coverage, and Empirical Coverage.
        """
        results = {}

        if target_col not in df_res.columns:
            raise KeyError(
                f"Target column '{target_col}' not found in the input DataFrame."
            )

        for q in sorted(quantiles):
            col_q = f"q_{int(q * 100)}"
            if col_q not in df_res.columns:
                continue

            loss = cls.pinball_loss(
                y_true=df_res[target_col].values,
                y_pred=df_res[col_q].values,
                quantile=q,
            )

            empirical_coverage = (df_res[target_col] <= df_res[col_q]).mean()

            results[col_q] = {
                "Pinball Loss": round(loss, 4),
                "Target Coverage": f"{int(q * 100)}%",
                "Empirical Coverage": f"{empirical_coverage * 100:.1f}%",
            }

        return pd.DataFrame(results).T
