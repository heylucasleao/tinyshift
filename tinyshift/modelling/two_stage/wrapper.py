# Copyright (c) 2024-2025 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License


from typing import Dict, Union, Any, Tuple, Optional, List

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import nbinom
from sklearn.base import BaseEstimator, RegressorMixin

from tinyshift.utils.imports import requires_extra


class TwoStageForecasterWrapper(BaseEstimator, RegressorMixin):
    """Two-stage probabilistic forecasting wrapper using MLForecast and Negative Binomial distribution.

    This model decouples conditional expectation from dispersion:
        1. Employs `MLForecast` regressors (e.g., LightGBM) with optional exponential time-decay weighting to forecast conditional expectation (lambda_t).
        2. Calibrates a per-series dispersion parameter (r) via Maximum Likelihood Estimation (MLE) on out-of-sample temporal cross-validation residuals.
        3. Projects discrete inventory quantiles, exact probability mass functions (PMF), and Newsvendor optimal stock levels via the inverse CDF (PPF) of the Negative Binomial distribution.

    Demand Regime Applicability
    ---------------------------
    - **Intermittent Demand (Zero-Inflated)**: Highly recommended. Handles frequent zeros efficiently.
    - **Erratic / Lumpy Demand**: Highly recommended. Captures high variance (variance > mean) via 'r'.
    - **Smooth / Low-Variance Demand**: Supported. As r increases, it naturally converges to a Poisson regime.
    - **Continuous / High-Volume Demand**: Not recommended. High-volume series with high r values will automatically fall back
      or converge toward a Poisson regime (bounded by r = 50).

    Architecture & Key Features
    ---------------------------
    - **Decoupled Two-Stage Design**: Separates point expectation forecasting from variance and tail modeling.
    - **Out-of-Sample Dispersion Calibration**: Optimizes per-series dispersion (r) using cross-validation residuals, backed by a robust global median fallback for cold-start series.
    - **Time-Decay Recency Weighting**: Supports exponential time-decay scaling (`gamma`) to prioritize recent historical dynamics during base model fitting.
    - **Inventory Optimization**: Built-in native support for Newsvendor critical fractile calculations and discrete marginal cost evaluation.

    Parameters
    ----------
    fcst : MLForecast
        An un-fitted or fitted MLForecast instance configured with a single underlying regressor.
    """

    def __init__(self, fcst: Any):
        self.fcst = fcst

    @property
    def model_name(self) -> str:
        """Extracts the name/key of the underlying model from MLForecast."""
        models_dict = getattr(self.fcst, "models_", None)
        if not models_dict:
            models_dict = getattr(self.fcst, "models", None)

        if not models_dict:
            raise ValueError("The MLForecast object has no models configured.")

        if len(models_dict) > 1:
            raise ValueError(
                f"TwoStageForecasterWrapper supports exactly 1 model, but found: {list(models_dict.keys())}"
            )

        return next(iter(models_dict.keys()))

    @property
    def model(self):
        """Extracts and validates the underlying trained estimator from the MLForecast container."""
        models_dict = getattr(self.fcst, "models_", None)
        if not models_dict:
            raise ValueError("The MLForecast object has not been fitted yet.")

        if len(models_dict) > 1:
            raise ValueError(
                f"TwoStageForecasterWrapper supports exactly 1 model, but found: {list(models_dict.keys())}"
            )

        return next(iter(models_dict.values()))

    def _nbinom_log_likelihood(
        self,
        params: list,
        y: np.ndarray,
        lambda_t: np.ndarray,
    ) -> float:
        """Computes the negative log-likelihood of the Negative Binomial distribution.

        Process
        -------
        1. Ensures parameter r remains positive.
        2. Converts (lambda_t, r) to the success probability parameter `p = r / (r + lambda_t)`.
        3. Computes the log probability mass function (logpmf) for discrete counts.
        4. Replaces NaNs and negative infinities using a lower bound floor (-1e2).

        Parameters
        ----------
        params : list of float
            Parameter vector containing [r] (dispersion/size parameter of the Negative Binomial).
        y : numpy.ndarray
            Observed historical demand values.
        lambda_t : numpy.ndarray
            In-sample predicted conditional expectation (lambda).

        Returns
        -------
        float
            Negative log-likelihood value to be minimized.
        """
        r = params[0]
        if r <= 0:
            return 1e10

        lambda_t = np.maximum(lambda_t, 1e-6)
        p = r / (r + lambda_t)

        log_pdf = nbinom.logpmf(y, r, p)
        log_pdf = np.nan_to_num(log_pdf, neginf=-1e2)

        return -np.sum(log_pdf)

    def _estimate_r(
        self,
        y_obs: np.ndarray,
        lambdas: np.ndarray,
    ) -> float:
        """Internal helper to estimate the dispersion parameter (r) via Maximum Likelihood Estimation.

        Process
        -------
        Uses L-BFGS-B optimization to minimize `_nbinom_log_likelihood` over the observed target
        series and in-sample predictions. Parameter r is bounded between 1e-3 and 50.0 to prevent
        collapse into thin-tailed distributions (Poisson) and ensure tail coverage.

        Parameters
        ----------
        y_obs : numpy.ndarray
            Target demand values for a specific series.
        lambdas : numpy.ndarray
            Predicted conditional expectation values (lambda_t) for the same series.

        Returns
        -------
        float
            Optimized per-series dispersion parameter r.
        """
        res = minimize(
            self._nbinom_log_likelihood,
            x0=[1.0],
            method="L-BFGS-B",
            args=(y_obs, lambdas),
            bounds=[(1e-3, 50.0)],
        )
        if (
            not res.success
            or not np.isfinite(res.fun)
            or len(res.x) != 1
            or not np.isfinite(res.x[0])
        ):
            raise RuntimeError(f"Dispersion optimization failed: {res.message}")
        return float(res.x[0])

    def _compute_time_decay_weights(
        self,
        df: pd.DataFrame,
        time_col: str = "ds",
        gamma: float = 0.5,
    ) -> np.ndarray:
        """Generates an exponential time-decay weight vector based on date recency.

        The gamma scale is ANNUAL (e.g., gamma=0.5 reduces weight to ~60% after 1 year),
        regardless of series frequency (daily, weekly, monthly, or hourly).

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing the timestamp column.
        time_col : str, default='ds'
            Column name containing timestamps.
        gamma : float, default=0.5
            Exponential decay parameter for recency weighting (scale is annual).
            Applies larger weights to recent historical samples.

        Returns
        -------
        numpy.ndarray
            Vector of calculated exponential decay weights.
        """

        dates = pd.to_datetime(df[time_col])
        max_date = dates.max()

        seconds_in_year = 365.25 * 86400.0
        delta_years = (max_date - dates).dt.total_seconds() / seconds_in_year

        weights = np.exp(-gamma * delta_years).values
        return weights

    @staticmethod
    def _compute_quantile(
        df: pd.DataFrame,
        target_q: float,
    ) -> np.ndarray:
        """Computes discrete integer quantile projections for a target probability.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing 'r_dispersion' and 'lambda_t' columns.
        target_q : float
            Target quantile probability (e.g., 0.95 for 95th percentile).

        Returns
        -------
        numpy.ndarray
            Integer vector representing the discrete inventory quantiles.
        """
        target_q_array = np.asarray(target_q)
        if np.any(~np.isfinite(target_q_array)) or np.any(
            (target_q_array <= 0) | (target_q_array >= 1)
        ):
            raise ValueError("Quantiles must be finite and strictly between 0 and 1.")

        p_param = df["r_dispersion"] / (df["r_dispersion"] + df["lambda_t"])
        return np.ceil(nbinom.ppf(target_q, df["r_dispersion"], p_param)).astype(int)

    def _compute_critical_quantile(
        self,
        cu: np.ndarray,
        co: np.ndarray,
    ) -> np.ndarray:
        """Computes the critical quantile q_star = c_u / (c_u + c_o) in-place safely."""
        if np.any(~np.isfinite(cu)) or np.any(~np.isfinite(co)):
            raise ValueError("Costs must be finite.")
        if np.any(cu < 0) or np.any(co < 0):
            raise ValueError("Costs must be non-negative.")

        denom = cu + co
        out = np.empty_like(denom)
        zero_mask = denom == 0
        out[zero_mask] = 0.5
        np.divide(cu, denom, out=out, where=~zero_mask)
        return out

    def _extract_cost_array(
        self,
        df: pd.DataFrame,
        cost_input: Union[str, float, int, Dict[Union[str, Tuple[str, Any]], float]],
        id_col: str,
        time_col: str,
        n_rows: int,
    ) -> np.ndarray:
        """Extracts cost structures into flat NumPy arrays with optimized dict lookup."""
        if isinstance(cost_input, (int, float)):
            return np.full(n_rows, float(cost_input), dtype=float)
        elif isinstance(cost_input, str):
            return df[cost_input].to_numpy(dtype=float)
        elif isinstance(cost_input, dict):
            if not cost_input:
                raise ValueError("Cost dictionary cannot be empty.")

            first_key = next(iter(cost_input.keys()))
            if isinstance(first_key, tuple):
                tuples = zip(df[id_col].to_numpy(), df[time_col].to_numpy())
                return np.fromiter(
                    (cost_input.get(k, np.nan) for k in tuples),
                    dtype=float,
                    count=n_rows,
                )
            else:
                return df[id_col].map(cost_input).to_numpy(dtype=float)
        else:
            raise TypeError(
                f"Cost input must be a column name (str), numeric scalar (float/int), "
                f"or dict mapping IDs or (ID, Time) tuples to values. Received: {type(cost_input)}"
            )

    def _calibrate_dispersion_cv(
        self,
        df: pd.DataFrame,
        h: int,
        n_windows: int,
        step_size: Optional[int],
        refit: Union[bool, int],
        fit_kwargs: Dict[str, Any],
    ) -> Tuple[Dict[Any, float], float]:
        """Perform temporal cross-validation and calibrate the dispersion parameter (r) via OOF residuals.

        Returns
        -------
        Tuple[Dict[Any, float], float]
            Dictionary mapping each series ID to its optimized r, and the global r_fallback.
        """
        cv_df = self.fcst.cross_validation(
            df=df,
            h=h,
            n_windows=n_windows,
            step_size=step_size,
            refit=refit,
            id_col=self.id_col,
            time_col=self.time_col,
            target_col=self.target_col,
            static_features=self.static_features,
            **fit_kwargs,
        )

        r_dict = {}
        for uid, group in cv_df.groupby(self.id_col):
            y_obs = group[self.target_col].to_numpy()
            lambdas_oof = np.maximum(group[self.model_name].to_numpy(), 1e-6)
            r_dict[uid] = self._estimate_r(y_obs, lambdas_oof)

        valid_r = [v for v in r_dict.values() if np.isfinite(v)]
        r_fallback = float(np.median(valid_r)) if valid_r else 1.0

        return r_dict, r_fallback

    def _fit_base_forecaster(
        self,
        df: pd.DataFrame,
        fit_kwargs: Dict[str, Any],
    ) -> List[str]:
        """Fit the main estimator and extract the generated temporal features.

        Returns
        -------
        List[str]
            Ordered list of exogenous features/lags used by MLForecast.
        """
        self.fcst.fit(
            df=df,
            id_col=self.id_col,
            time_col=self.time_col,
            target_col=self.target_col,
            static_features=self.static_features,
            **fit_kwargs,
        )

        return self.fcst.ts.features_order_

    @requires_extra("series")
    def fit(
        self,
        df_train: pd.DataFrame,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
        static_features: list = None,
        gamma: float = None,
        h: int = 14,
        n_windows: int = 10,
        step_size: Optional[int] = None,
        refit: Union[bool, int] = True,
    ) -> "TwoStageForecasterWrapper":
        """Fits the underlying MLForecast model and optimizes per-series dispersion parameters.

        Process
        -------
        1. Fits the MLForecast pipeline on the training dataset (with optional time-decay weights).
        2. Preprocesses the data to generate feature matrices.
        3. Generates in-sample point predictions (lambda_t).
        4. Iterates over each unique ID and executes `_estimate_r` via MLE.
        5. Computes `r_fallback` as the global median of estimated 'r' values for cold-start prediction.

        Parameters
        ----------
        df_train : pandas.DataFrame
            Training data containing series IDs, timestamps, target values, and features.
        id_col : str, default='unique_id'
            Column name identifying individual time series.
        time_col : str, default='ds'
            Column name containing timestamps.
        target_col : str, default='y'
            Column name for target demand variable.
        static_features : list of str, optional
            List of static feature names to preserve.
        gamma : float, optional
            Exponential decay rate for recency weighting during fitting.

        Returns
        -------
        self : TwoStageForecasterWrapper
            Fitted instance of the TwoStageForecasterWrapper class.
        """

        self.id_col = id_col
        self.time_col = time_col
        self.target_col = target_col
        self.static_features = static_features or []

        df_fit = df_train.copy()

        fit_kwargs = {}
        if gamma is not None:
            weights_array = self._compute_time_decay_weights(
                df_fit, time_col=time_col, gamma=gamma
            )
            df_fit["_temp_weight"] = weights_array
            fit_kwargs["weight_col"] = "_temp_weight"

        self.r_dict_, self.r_fallback_ = self._calibrate_dispersion_cv(
            df_fit, h, n_windows, step_size, refit, fit_kwargs
        )

        self.exog_cols_ = self._fit_base_forecaster(
            df=df_fit,
            fit_kwargs=fit_kwargs,
        )

        return self

    @requires_extra("series")
    def predict(
        self,
        h: int,
        X_df: pd.DataFrame = None,
        quantiles: tuple = (0.05, 0.50, 0.95),
    ) -> pd.DataFrame:
        """Generates out-of-sample probabilistic forecast quantiles using the Negative Binomial CDF.

        Process
        -------
        1. Executes recursive forecasting via MLForecast to obtain future lambda_t values.
        2. Maps historical `r` dispersion parameters (or applies `r_fallback` for new series).
        3. Computes exact integer quantiles using the Negative Binomial Percent Point Function (PPF/Inverse CDF)
           and applies `np.ceil` to ensure discrete inventory units.

        Parameters
        ----------
        h : int
            Forecast horizon.
        X_df : pandas.DataFrame, optional
            Exogenous features for the forecast horizon.
        quantiles : list of float, default=[0.05, 0.50, 0.95]
            Target quantile values.

        Returns
        -------
        df_pred : pandas.DataFrame
            DataFrame containing time series identifiers, predicted lambda_t, dispersion parameters,
            and discrete stock quantile estimates (`q_*`).
        """

        df_pred = self.fcst.predict(h=h, X_df=X_df)

        df_pred = df_pred.rename(columns={self.model_name: "lambda_t"})
        df_pred["r_dispersion"] = (
            df_pred[self.id_col].map(self.r_dict_).fillna(self.r_fallback_)
        )

        for q in sorted(quantiles):
            col_name = f"q_{int(q * 100)}"
            df_pred[col_name] = self._compute_quantile(df_pred, target_q=q)

        return df_pred

    @requires_extra("series")
    def optimize(
        self,
        h: int,
        underage_cost: Union[
            str, float, int, Dict[Union[str, Tuple[str, Any]], float]
        ] = "cu",
        overage_cost: Union[
            str, float, int, Dict[Union[str, Tuple[str, Any]], float]
        ] = "co",
        X_df: pd.DataFrame = None,
        ratio_col: str = "critical_ratio",
        output_col: str = "y_optimal",
    ) -> pd.DataFrame:
        """Generates forecasts for horizon h and injects the optimal reorder quantity via Critical Fractile.

        Parameters
        ----------
        h : int
            Forecast horizon.
        underage_cost : str, float, int, or dict
            Unit cost of unfulfilled demand (shortage/stockout cost). Can be a scalar,
            column name, or dictionary mapping IDs or (ID, Time) tuples.
        overage_cost : str, float, int, or dict
            Unit cost of holding excess inventory (holding cost). Accepts same formats as underage_cost.
        X_df : pd.DataFrame, optional
            Exogenous features for the forecast horizon.
        ratio_col : str, default="critical_ratio"
            Column name to store the computed critical ratio/fractile values.
        output_col : str, default="y_optimal"
            Column name to store the optimal reorder quantity/quantile forecast.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the predictions, critical ratio, and optimal reorder quantity.
        """
        excluded = {
            cost for cost in (underage_cost, overage_cost) if isinstance(cost, str)
        }
        pred_cols = (
            [c for c in X_df.columns if c not in excluded] if X_df is not None else None
        )
        pred_x_df = X_df[pred_cols] if X_df is not None else None
        df_out = self.predict(h=h, X_df=pred_x_df, quantiles=[])

        cost_df = X_df if X_df is not None else df_out
        n_rows = len(df_out)
        if len(cost_df) != n_rows:
            raise ValueError("Cost inputs must have one row per forecast row.")
        cu_arr = self._extract_cost_array(
            cost_df, underage_cost, self.id_col, self.time_col, n_rows
        )
        co_arr = self._extract_cost_array(
            cost_df, overage_cost, self.id_col, self.time_col, n_rows
        )
        critical_fractile = self._compute_critical_quantile(cu=cu_arr, co=co_arr)

        df_out[ratio_col] = critical_fractile
        df_out[output_col] = self._compute_quantile(df_out, target_q=critical_fractile)

        return df_out

    @requires_extra("series")
    def pmf(
        self,
        h: int,
        max_k: int = 10,
        X_df: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """Generates exact discrete probabilities P(Y = k) for k = 0, 1, ..., max_k over horizon h.

        Parameters
        ----------
        h : int
            Forecast horizon.
        max_k : int, default=10
            Maximum number of units to evaluate individual probabilities for.
        X_df : pandas.DataFrame, optional
            Exogenous features for the forecast horizon.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing id_col, time_col, lambda_t, r_dispersion, P(Y=0)...P(Y=max_k), and P(Y>max_k).
        """

        if not isinstance(max_k, (int, np.integer)) or max_k < 0:
            raise ValueError("max_k must be a non-negative integer.")

        df_out = self.predict(h=h, X_df=X_df, quantiles=[])

        p_param = df_out["r_dispersion"] / (df_out["r_dispersion"] + df_out["lambda_t"])
        r_param = df_out["r_dispersion"].values

        k_range = np.arange(0, max_k + 1)

        pmf_matrix = nbinom.pmf(
            k_range[None, :], r_param[:, None], p_param.values[:, None]
        )

        for k in k_range:
            df_out[f"P(Y={k})"] = pmf_matrix[:, k]

        df_out[f"P(Y>{max_k})"] = 1.0 - pmf_matrix.sum(axis=1)

        return df_out

    @requires_extra("series")
    def marginal_cost(
        self,
        h: int,
        underage_cost: Union[
            str, float, int, Dict[Union[str, Tuple[str, Any]], float]
        ] = "cu",
        overage_cost: Union[
            str, float, int, Dict[Union[str, Tuple[str, Any]], float]
        ] = "co",
        max_k: int = 10,
        X_df: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """Calculates the expected marginal cost for each additional inventory unit k (from 0 to max_k).

        Process
        -------
        1. Extracts underage and overage cost arrays using the internal cost extraction utility.
        2. Leverages the internal `pmf` method to obtain exact discrete probabilities P(Y = k).
        3. Computes the cumulative distribution function (CDF) to derive P(Y < k) and P(Y >= k).
        4. Computes the marginal net benefit of stocking unit k:
           MC(k) = c_u * P(Y >= k) - c_o * P(Y < k)

        Parameters
        ----------
        h : int
            Forecast horizon.
        underage_cost : str, float, int, or dict
            Unit cost of unfulfilled demand (shortage/stockout cost). Can be a scalar,
            column name, or dictionary mapping IDs or (ID, Time) tuples.
        overage_cost : str, float, int, or dict
            Unit cost of holding excess inventory (holding cost). Accepts same formats as underage_cost.
        max_k : int, default=10
            Maximum number of units to evaluate individual marginal costs for.
        X_df : pandas.DataFrame, optional
            Exogenous features for the forecast horizon.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing id_col, time_col, lambda_t, r_dispersion, and expected marginal costs `MC(k=0)` through `MC(k=max_k)`.

        Notes
        -----
        - Positive values represent an expected net monetary gain (the benefit of avoiding a stockout outweighs the holding risk).
        - Negative values represent an expected net monetary loss or cost increase (the risk of overstocking outweighs the shortage benefit).
        """
        excluded = {
            cost for cost in (underage_cost, overage_cost) if isinstance(cost, str)
        }
        pred_cols = (
            [c for c in X_df.columns if c not in excluded] if X_df is not None else None
        )
        pred_x_df = X_df[pred_cols] if X_df is not None else None
        df_pmf = self.pmf(h=h, max_k=max_k, X_df=pred_x_df)

        cost_df = X_df if X_df is not None else df_pmf
        n_rows = len(df_pmf)
        if len(cost_df) != n_rows:
            raise ValueError("Cost inputs must have one row per forecast row.")
        cu_arr = self._extract_cost_array(
            cost_df,
            underage_cost,
            self.id_col,
            self.time_col,
            n_rows,
        )
        co_arr = self._extract_cost_array(
            cost_df,
            overage_cost,
            self.id_col,
            self.time_col,
            n_rows,
        )
        k_range = np.arange(0, max_k + 1)
        pmf_cols = [f"P(Y={k})" for k in k_range]
        pmf_matrix = df_pmf[pmf_cols].to_numpy()
        cdf_matrix = np.cumsum(pmf_matrix, axis=1)

        p_less_matrix = np.hstack([np.zeros((n_rows, 1)), cdf_matrix[:, :-1]])
        p_greater_equal_matrix = 1.0 - p_less_matrix

        mc_matrix = (
            cu_arr[:, None] * p_greater_equal_matrix - co_arr[:, None] * p_less_matrix
        )

        df_out = df_pmf[[self.id_col, self.time_col, "lambda_t", "r_dispersion"]].copy()

        for i, k in enumerate(k_range):
            df_out[f"MC(k={k})"] = mc_matrix[:, i]

        return df_out
