# Copyright (c) 2024-2025 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License


import numpy as np
import pandas as pd
from scipy.stats import nbinom
from scipy.optimize import minimize
from mlforecast import MLForecast
from typing import Dict, Union, Any, Tuple


class ISSMForecastWrapper:
    """Intermittent State-Space Model (ISSM) wrapper using MLForecast and Negative Binomial distribution.

    This model isolates the conditional expectation (lambda_t) using standard MLForecast regressors
    and fits a per-series dispersion parameter (r) via Maximum Likelihood Estimation (MLE).
    It projects inventory quantiles via the inverse CDF (PPF) of the Negative Binomial distribution.

    Demand Regime Applicability
    ---------------------------
    - **Intermittent Demand (Zero-Inflated)**: Highly recommended. Handles frequent zeros efficiently.
    - **Erratic / Lumpy Demand**: Highly recommended. Captures high variance (variance > mean) via 'r'.
    - **Smooth / Low-Variance Demand**: Supported. As r increases, it naturally converges to a Poisson regime.
    - **Continuous / High-Volume Demand**: Not recommended.

    References & Architecture
    -------------------------
    - **Theoretical Framework**: Inspired by Lokad's white-boxed ISSM approach for intermittent
      demand and zero-inflated sales (M5 Forecasting Competition, 2020).
      Paper reference: "A white-boxed ISSM approach to estimate uncertainty distributions of Walmart sales".
    - **Implementation Variant**: Unlike the pure state-space formulation (which uses ETS(A,N,M) state updates
      and Monte Carlo simulation), this wrapper decouples expectation from dispersion:
        1. Employs `MLForecast` regressors (e.g., LightGBM) to forecast conditional expectation (lambda_t).
        2. Fits a per-series dispersion parameter (r) via Negative Binomial MLE on in-sample residuals.
        3. Derives discrete inventory quantiles via inverse CDF (PPF) projection.

    Parameters
    ----------
    fcst : MLForecast
        An un-fitted or fitted MLForecast instance configured with a single underlying regressor.
    """

    def __init__(self, fcst: MLForecast):
        self.fcst = fcst
        self.r_dict = {}
        self.r_fallback = 1.0

    @property
    def model(self):
        """Extracts and validates the underlying trained estimator from the MLForecast container.

        Returns
        -------
        object
            The single regression estimator stored inside MLForecast.

        Raises
        ------
        ValueError
            If MLForecast has not been fitted or contains multiple estimators.
        """
        if not hasattr(self.fcst, "models_") or not self.fcst.models_:
            raise ValueError("The MLForecast object has not been fitted yet.")

        if len(self.fcst.models_) > 1:
            raise ValueError(
                f"ISSMForecast supports exactly 1 model, but found: {list(self.fcst.models_.keys())}"
            )

        return next(iter(self.fcst.models_.values()))

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
        p_param = df["r_dispersion"] / (df["r_dispersion"] + df["lambda_t"])
        return np.ceil(nbinom.ppf(target_q, df["r_dispersion"], p_param)).astype(int)

    def _compute_critical_quantile(
        self,
        cu: np.ndarray,
        co: np.ndarray,
    ) -> np.ndarray:
        """Computes the critical quantile q_star = c_u / (c_u + c_o) in-place safely."""
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

    def fit(
        self,
        df_train: pd.DataFrame,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
        static_features: list = None,
        gamma: float = None,
    ) -> "ISSMForecastWrapper":
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
        self : ISSMForecastWrapper
            Fitted instance of the ISSMForecastWrapper class.
        """
        self.id_col = id_col
        self.time_col = time_col
        self.target_col = target_col
        static_features = static_features or []

        df_fit = df_train.copy()

        fit_kwargs = {}
        if gamma is not None:
            weights_array = self._compute_time_decay_weights(
                df_fit, time_col=time_col, gamma=gamma
            )
            df_fit["_temp_weight"] = weights_array
            fit_kwargs["weight_col"] = "_temp_weight"

        self.fcst.fit(
            df_fit,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
            static_features=static_features,
            **fit_kwargs,
        )

        df_prep = self.fcst.preprocess(
            df_train,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
            static_features=static_features,
        )

        self.exog_cols_ = self.fcst.ts.features_order_
        X_train = df_prep[self.exog_cols_]
        df_prep["lambda_t"] = self.model.predict(X_train)

        # Fit dispersion parameter per series via MLE
        self.r_dict = {}
        for uid, group in df_prep.groupby(id_col):
            y_obs = group[target_col].values
            lambdas = group["lambda_t"].values
            self.r_dict[uid] = self._estimate_r(y_obs, lambdas)

        # Fallback value for unseen series during inference
        self.r_fallback = float(np.median(list(self.r_dict.values())))

        return self

    def predict(
        self,
        h: int,
        X_df: pd.DataFrame = None,
        quantiles: list = [0.50, 0.67, 0.95, 0.99],
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
        quantiles : list of float, default=[0.50, 0.67, 0.95, 0.99]
            Target quantile values.

        Returns
        -------
        df_pred : pandas.DataFrame
            DataFrame containing time series identifiers, predicted lambda_t, dispersion parameters,
            and discrete stock quantile estimates (`q_*`).
        """

        df_pred = self.fcst.predict(h=h, X_df=X_df)

        model_key = next(iter(self.fcst.models_.keys()))
        df_pred = df_pred.rename(columns={model_key: "lambda_t"})
        df_pred["r_dispersion"] = (
            df_pred[self.id_col].map(self.r_dict).fillna(self.r_fallback)
        )

        for q in sorted(quantiles):
            col_name = f"q_{int(q * 100)}"
            df_pred[col_name] = self._compute_quantile(df_pred, target_q=q)

        return df_pred

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
        excluded = {underage_cost, overage_cost}
        pred_cols = [c for c in X_df.columns if c not in excluded]
        n_rows = len(X_df)
        cu_arr = self._extract_cost_array(
            X_df, underage_cost, self.id_col, self.time_col, n_rows
        )
        co_arr = self._extract_cost_array(
            X_df, overage_cost, self.id_col, self.time_col, n_rows
        )
        critical_fractile = self._compute_critical_quantile(cu=cu_arr, co=co_arr)

        df_out = self.predict(h=h, X_df=X_df[pred_cols], quantiles=[])

        df_out[ratio_col] = critical_fractile
        df_out[output_col] = self._compute_quantile(df_out, target_q=critical_fractile)

        return df_out

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
        4. Applies the discrete Newsvendor marginal cost derivative:
           MC(k) = c_o * P(Y >= k) - c_u * P(Y < k)

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
        n_rows = len(X_df) if X_df is not None else len(self.fcst.predict(h=h))
        cu_arr = self._extract_cost_array(
            X_df,
            underage_cost,
            self.id_col,
            self.time_col,
            n_rows,
        )
        co_arr = self._extract_cost_array(
            X_df,
            overage_cost,
            self.id_col,
            self.time_col,
            n_rows,
        )

        df_pmf = self.pmf(h=h, max_k=max_k, X_df=X_df)

        k_range = np.arange(0, max_k + 1)
        pmf_cols = [f"P(Y={k})" for k in k_range]
        pmf_matrix = df_pmf[pmf_cols].to_numpy()
        cdf_matrix = np.cumsum(pmf_matrix, axis=1)

        p_less_matrix = np.hstack([np.zeros((n_rows, 1)), cdf_matrix[:, :-1]])
        p_greater_equal_matrix = 1.0 - p_less_matrix

        mc_matrix = (
            co_arr[:, None] * p_greater_equal_matrix - cu_arr[:, None] * p_less_matrix
        )

        df_out = df_pmf[[self.id_col, self.time_col, "lambda_t", "r_dispersion"]].copy()

        for i, k in enumerate(k_range):
            df_out[f"MC(k={k})"] = mc_matrix[:, i]

        return df_out
