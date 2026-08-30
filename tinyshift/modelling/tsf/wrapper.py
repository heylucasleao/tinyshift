# Copyright (c) 2024-2025 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License


from collections.abc import Iterable
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import nbinom
from sklearn.base import BaseEstimator, RegressorMixin

from tinyshift.utils.imports import requires_extra

from .distribution import DiscretePredictiveDistribution, PredictiveDistribution
from .family import DistributionFamily, NegativeBinomialFamily


class TwoStageForecasterWrapper(BaseEstimator, RegressorMixin):
    """Two-stage probabilistic forecasting wrapper using MLForecast.

    This model decouples conditional expectation from dispersion:
        1. Employs `MLForecast` regressors (e.g., LightGBM) with optional exponential time-decay weighting to forecast conditional expectation (lambda_t).
        2. Calibrates a per-series distribution parameter via maximum likelihood
           on out-of-sample temporal cross-validation predictions.
        3. Exposes a row-aligned predictive distribution and projects arbitrary
           quantiles or Newsvendor-optimal levels through its inverse CDF.

    Default Negative Binomial Applicability
    ---------------------------------------
    - **Intermittent Demand (Zero-Inflated)**: Highly recommended. Handles frequent zeros efficiently.
    - **Erratic / Lumpy Demand**: Highly recommended. Captures high variance (variance > mean) via 'r'.
    - **Smooth / Low-Variance Demand**: Supported. As r increases, it naturally converges to a Poisson regime.
    - **Continuous / High-Volume Demand**: Select an explicit continuous family,
      such as :class:`GammaFamily`, instead of the default.

    Architecture & Key Features
    ---------------------------
    - **Decoupled Two-Stage Design**: Separates point expectation forecasting from variance and tail modeling.
    - **Out-of-Sample Dispersion Calibration**: Optimizes a family-specific
      parameter per series, backed by a global median fallback for cold starts.
    - **Time-Decay Recency Weighting**: Supports exponential time-decay scaling (`gamma`) to prioritize recent historical dynamics during base model fitting.
    - **Inventory Optimization**: Built-in native support for Newsvendor critical fractile calculations and discrete marginal benefit evaluation.

    Parameters
    ----------
    fcst : MLForecast
        An un-fitted or fitted MLForecast instance configured with a single underlying regressor.
    distribution : DistributionFamily, optional
        Conditional distribution family for the second stage. Defaults to
        :class:`NegativeBinomialFamily`; use :class:`GammaFamily` for strictly
        positive continuous targets.
    """

    def __init__(
        self,
        fcst: Any,
        distribution: Optional[DistributionFamily] = None,
    ):
        self.fcst = fcst
        self.distribution = distribution

    @staticmethod
    def _validate_fit_parameters(
        h: int,
        n_windows: int,
        step_size: Optional[int],
        gamma: Optional[float],
    ) -> None:
        """Validate temporal calibration and recency-weighting parameters."""
        if (
            isinstance(h, (bool, np.bool_))
            or not isinstance(h, (int, np.integer))
            or h < 1
        ):
            raise ValueError("h must be a positive integer.")
        if (
            isinstance(n_windows, (bool, np.bool_))
            or not isinstance(n_windows, (int, np.integer))
            or n_windows < 1
        ):
            raise ValueError("n_windows must be a positive integer.")
        if step_size is not None and (
            isinstance(step_size, (bool, np.bool_))
            or not isinstance(step_size, (int, np.integer))
            or step_size < 1
        ):
            raise ValueError("step_size must be None or a positive integer.")
        if gamma is not None and (
            isinstance(gamma, (bool, np.bool_)) or not np.isfinite(gamma) or gamma < 0
        ):
            raise ValueError("gamma must be a finite, non-negative number.")

    @staticmethod
    def _validate_training_target(
        df: pd.DataFrame,
        target_col: str,
        distribution_family: DistributionFamily,
        numeric_label: str,
    ) -> np.ndarray:
        """Extract and validate the target against the selected family."""
        if target_col not in df.columns:
            raise ValueError(f"Target column {target_col!r} was not found.")

        try:
            target = df[target_col].to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Target values must be {numeric_label}.") from exc

        if target.size == 0:
            raise ValueError("Training data cannot be empty.")
        distribution_family.validate_target(target)
        return target

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
        4. Replaces negative infinities using a lower-bound floor (-1e2).

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
        if not np.isfinite(r) or r <= 0:
            return 1e10

        lambda_t = np.maximum(lambda_t, 1e-6)
        p = r / (r + lambda_t)

        log_pdf = nbinom.logpmf(y, r, p)
        log_pdf = np.where(np.isneginf(log_pdf), -1e2, log_pdf)

        return -np.sum(log_pdf)

    def _estimate_r(
        self,
        y_obs: np.ndarray,
        lambdas: np.ndarray,
    ) -> float:
        """Internal helper to estimate the dispersion parameter (r) via Maximum Likelihood Estimation.

        Process
        -------
        Uses bounded scalar optimization to minimize `_nbinom_log_likelihood` over the observed
        target series and in-sample predictions. Parameter r is bounded between 1e-3 and 50.0 to
        prevent collapse into thin-tailed distributions (Poisson) and ensure tail coverage.

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
        if not np.all(np.isfinite(lambdas)):
            raise ValueError("Predicted lambda values must be finite.")

        res = minimize_scalar(
            lambda r: self._nbinom_log_likelihood([r], y_obs, lambdas),
            bounds=(1e-3, 50.0),
            method="bounded",
        )
        if not res.success or not np.isfinite(res.fun) or not np.isfinite(res.x):
            raise RuntimeError(f"Dispersion optimization failed: {res.message}")
        return float(res.x)

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

    @staticmethod
    def _resolve_marginal_units(
        max_k: Optional[int], units: Optional[Iterable[int]]
    ) -> np.ndarray:
        """Validate the marginal-benefit unit selection and return its grid."""
        if max_k is not None and units is not None:
            raise ValueError("Provide either max_k or units, not both.")
        if units is None:
            max_k = 10 if max_k is None else max_k
            if (
                isinstance(max_k, (bool, np.bool_))
                or not isinstance(max_k, (int, np.integer))
                or max_k < 0
            ):
                raise ValueError("max_k must be a non-negative integer.")
            return np.arange(0, max_k + 1)

        if isinstance(units, (str, bytes)):
            raise ValueError("units must be a non-empty iterable of integers.")
        try:
            unit_values = list(units)
        except TypeError as exc:
            raise ValueError(
                "units must be a non-empty iterable of integers."
            ) from exc
        if not unit_values or any(
            isinstance(unit, (bool, np.bool_))
            or not isinstance(unit, (int, np.integer))
            or unit < 0
            for unit in unit_values
        ):
            raise ValueError(
                "units must be a non-empty iterable of non-negative integers."
            )
        if len(set(unit_values)) != len(unit_values):
            raise ValueError("units must not contain duplicates.")
        return np.asarray(unit_values, dtype=int)

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

    def _align_cost_frame(
        self,
        forecast_df: pd.DataFrame,
        X_df: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        """Align decision inputs with forecast rows by key or positional order.

        When ID and time columns are available, a one-to-one merge makes the
        alignment explicit and independent of input row order. A frame without
        those keys is treated as already row-aligned.
        """
        if X_df is None:
            return forecast_df
        if not {self.id_col, self.time_col}.issubset(X_df.columns):
            return X_df
        return forecast_df[[self.id_col, self.time_col]].merge(
            X_df,
            on=[self.id_col, self.time_col],
            how="left",
            sort=False,
            validate="one_to_one",
        )

    def _prepare_prediction_frame(
        self,
        X_df: Optional[pd.DataFrame],
        underage_cost: Union[
            str, float, int, Dict[Union[str, Tuple[str, Any]], float]
        ],
        overage_cost: Union[
            str, float, int, Dict[Union[str, Tuple[str, Any]], float]
        ],
    ) -> Optional[pd.DataFrame]:
        """Remove decision-only cost columns from the prediction frame."""
        if X_df is None:
            return None

        cost_columns = {
            cost for cost in (underage_cost, overage_cost) if isinstance(cost, str)
        }
        missing_cost_columns = cost_columns - set(X_df.columns)
        if missing_cost_columns:
            raise ValueError(
                f"Cost columns not found in X_df: {sorted(missing_cost_columns)}"
            )

        prediction_columns = [
            column for column in X_df.columns if column not in cost_columns
        ]
        exogenous_columns = set(prediction_columns) - {
            self.id_col,
            self.time_col,
        }
        if not exogenous_columns:
            return None
        return X_df[prediction_columns]

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

        dispersion_dict = {}
        for uid, group in cv_df.groupby(self.id_col):
            y_obs = group[self.target_col].to_numpy()
            lambdas_oof = group[self.model_name].to_numpy()
            dispersion_dict[uid] = self.distribution_family_.fit_dispersion(
                y_obs, lambdas_oof
            )

        if not dispersion_dict:
            raise RuntimeError("Dispersion calibration produced no series.")
        dispersion_fallback = float(np.median(list(dispersion_dict.values())))

        return dispersion_dict, dispersion_fallback

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
        1. Runs temporal cross-validation to generate out-of-fold point predictions (lambda_t).
        2. Iterates over each unique ID and executes `_estimate_r` via MLE.
        3. Computes `r_fallback` as the global median of estimated 'r' values for cold-start prediction.
        4. Fits the MLForecast pipeline on the complete training dataset, optionally using
           time-decay weights.

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

        self._validate_fit_parameters(h, n_windows, step_size, gamma)

        self.distribution_family_ = (
            NegativeBinomialFamily() if self.distribution is None else self.distribution
        )
        if not isinstance(self.distribution_family_, DistributionFamily):
            raise TypeError(
                "distribution must be a DistributionFamily instance or None."
            )
        numeric_label = "numeric counts" if self.distribution is None else "numeric"
        self._validate_training_target(
            df_train,
            target_col,
            self.distribution_family_,
            numeric_label,
        )
        df_fit = df_train.copy()

        fit_kwargs = {}
        if gamma is not None:
            weights_array = self._compute_time_decay_weights(
                df_fit, time_col=time_col, gamma=gamma
            )
            df_fit["_temp_weight"] = weights_array
            fit_kwargs["weight_col"] = "_temp_weight"

        self.dispersion_dict_, self.dispersion_fallback_ = (
            self._calibrate_dispersion_cv(
                df_fit, h, n_windows, step_size, refit, fit_kwargs
            )
        )
        # Backwards-compatible fitted attributes for the default family.
        if isinstance(self.distribution_family_, NegativeBinomialFamily):
            self.r_dict_ = self.dispersion_dict_
            self.r_fallback_ = self.dispersion_fallback_

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
        """Generate out-of-sample quantiles from the configured distribution.

        Process
        -------
        1. Executes recursive forecasting via MLForecast to obtain future lambda_t values.
        2. Maps calibrated per-series distribution parameters, using the global
           median fallback for unseen series.
        3. Evaluates the configured distribution's PPF. Results are integer-valued
           for discrete families and real-valued for continuous families.

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
            DataFrame containing identifiers, conditional means, distribution
            parameters, and quantile estimates (`q_*`).
        """

        df_pred, distribution = self.predict_distribution(h=h, X_df=X_df)

        quantile_columns = {}
        for q in sorted(quantiles):
            if not np.isfinite(q) or not 0.0 < q < 1.0:
                raise ValueError(
                    "Quantiles must be finite and strictly between 0 and 1."
                )
            col_name = f"q_{int(q * 100)}"
            if col_name in quantile_columns:
                raise ValueError(
                    f"Quantiles {quantile_columns[col_name]} and {q} map to the "
                    f"same output column {col_name!r}."
                )
            quantile_columns[col_name] = q
            df_pred[col_name] = distribution.ppf(q)

        return df_pred

    @requires_extra("series")
    def predict_distribution(
        self,
        h: int,
        X_df: pd.DataFrame = None,
    ) -> Tuple[pd.DataFrame, PredictiveDistribution]:
        """Return the forecast frame and its aligned predictive distributions."""
        df_pred = self.fcst.predict(h=h, X_df=X_df)
        df_pred = df_pred.rename(columns={self.model_name: "lambda_t"})
        means = df_pred["lambda_t"].to_numpy(dtype=float)
        if not np.all(np.isfinite(means)):
            raise ValueError("Predicted mean values must be finite.")
        means = np.maximum(means, 1e-6)
        df_pred["lambda_t"] = means
        parameter_column = self.distribution_family_.parameter_column
        df_pred[parameter_column] = (
            df_pred[self.id_col]
            .map(self.dispersion_dict_)
            .fillna(self.dispersion_fallback_)
        )
        predictive = self.distribution_family_.distribution(
            means, df_pred[parameter_column].to_numpy(dtype=float)
        )
        return df_pred, predictive

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
        pred_x_df = self._prepare_prediction_frame(
            X_df=X_df,
            underage_cost=underage_cost,
            overage_cost=overage_cost,
        )
        df_out, distribution = self.predict_distribution(h=h, X_df=pred_x_df)

        cost_df = self._align_cost_frame(df_out, X_df)
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
        df_out[output_col] = distribution.ppf(critical_fractile)

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

        df_out, distribution = self.predict_distribution(h=h, X_df=X_df)
        if not isinstance(distribution, DiscretePredictiveDistribution):
            raise TypeError("pmf is available only for discrete distribution families.")

        k_range = np.arange(0, max_k + 1)

        pmf_matrix = distribution.pmf(k_range)

        for k in k_range:
            df_out[f"P(Y={k})"] = pmf_matrix[:, k]

        df_out[f"P(Y>{max_k})"] = 1.0 - pmf_matrix.sum(axis=1)

        return df_out

    @requires_extra("series")
    def marginal_benefit(
        self,
        h: int,
        underage_cost: Union[
            str, float, int, Dict[Union[str, Tuple[str, Any]], float]
        ] = "cu",
        overage_cost: Union[
            str, float, int, Dict[Union[str, Tuple[str, Any]], float]
        ] = "co",
        max_k: Optional[int] = None,
        X_df: pd.DataFrame = None,
        units: Optional[Iterable[int]] = None,
    ) -> pd.DataFrame:
        """Calculates the expected marginal net benefit of each additional inventory unit k.

        Process
        -------
        1. Extracts underage and overage cost arrays using the internal cost extraction utility.
        2. Evaluates the predictive CDF directly to obtain P(Y < k) = F(k - 1).
        3. Computes the marginal net benefit of stocking unit k:
           MB(k) = c_u * P(Y >= k) - c_o * P(Y < k)

        Parameters
        ----------
        h : int
            Forecast horizon.
        underage_cost : str, float, int, or dict
            Unit cost of unfulfilled demand (shortage/stockout cost). Can be a scalar,
            column name, or dictionary mapping IDs or (ID, Time) tuples.
        overage_cost : str, float, int, or dict
            Unit cost of holding excess inventory (holding cost). Accepts same formats as underage_cost.
        max_k : int, optional
            Maximum unit to evaluate, producing all units from 0 through `max_k`.
            Defaults to 10 when `units` is not provided. Cannot be combined with
            `units`.
        X_df : pandas.DataFrame, optional
            Exogenous features for the forecast horizon.
        units : iterable of int, optional
            Exact non-negative units to evaluate, in the supplied order. Useful for
            sparse or stepped grids such as `[5, 10, 20]` or `range(0, 101, 5)`.
            Cannot be combined with `max_k`.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing id_col, time_col, lambda_t, r_dispersion, and one
            expected marginal-benefit column for each evaluated unit.

        Notes
        -----
        - Positive values represent an expected net monetary gain (the benefit of avoiding a stockout outweighs the holding risk).
        - Negative values represent an expected net monetary loss or cost increase (the risk of overstocking outweighs the shortage benefit).
        """
        units_array = self._resolve_marginal_units(max_k=max_k, units=units)

        pred_x_df = self._prepare_prediction_frame(
            X_df=X_df,
            underage_cost=underage_cost,
            overage_cost=overage_cost,
        )
        df_pred, distribution = self.predict_distribution(h=h, X_df=pred_x_df)
        if not isinstance(distribution, DiscretePredictiveDistribution):
            raise TypeError(
                "marginal_benefit is available only for discrete distribution families."
            )

        cost_df = self._align_cost_frame(df_pred, X_df)
        n_rows = len(df_pred)
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
        p_less_matrix = np.asarray(distribution.cdf(units_array - 1))
        p_greater_equal_matrix = 1.0 - p_less_matrix

        mb_matrix = (
            cu_arr[:, None] * p_greater_equal_matrix - co_arr[:, None] * p_less_matrix
        )

        parameter_column = self.distribution_family_.parameter_column
        df_out = df_pred[
            [self.id_col, self.time_col, "lambda_t", parameter_column]
        ].copy()

        for i, k in enumerate(units_array):
            df_out[f"MB(k={k})"] = mb_matrix[:, i]

        return df_out
