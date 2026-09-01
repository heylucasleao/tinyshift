# Copyright (c) 2024-2025 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License


from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import nbinom
from sklearn.base import BaseEstimator, RegressorMixin

from tinyshift.utils.imports import requires_extra

from .family import DistributionFamily, NegativeBinomialFamily
from .forecast import DiscretePanelPredictiveForecast, PanelPredictiveForecast


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
    - **Distribution-first API**: Returns row-aligned predictive distributions
      that can be consumed by forecasting and decision utilities.

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
        distribution: DistributionFamily | None = None,
    ):
        self.fcst = fcst
        self.distribution = distribution

    @staticmethod
    def _validate_fit_parameters(
        h: int,
        n_windows: int,
        step_size: int | None,
        gamma: float | None,
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

    def _calibrate_dispersion_cv(
        self,
        df: pd.DataFrame,
        h: int,
        n_windows: int,
        step_size: int | None,
        refit: bool | int,
        fit_kwargs: dict[str, Any],
    ) -> tuple[dict[Any, float], float]:
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
        fit_kwargs: dict[str, Any],
    ) -> list[str]:
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
        static_features: list | None = None,
        gamma: float | None = None,
        h: int = 14,
        n_windows: int = 10,
        step_size: int | None = None,
        refit: bool | int = True,
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
    def predict_distribution(
        self,
        h: int,
        X_df: pd.DataFrame = None,
    ) -> PanelPredictiveForecast | DiscretePanelPredictiveForecast:
        """Return predictive distributions aligned to the forecast panel.

        Parameters
        ----------
        h : int
            Number of future steps to forecast for each series.
        X_df : pandas.DataFrame or None, default=None
            Future exogenous features in MLForecast long format. Pass ``None``
            when the fitted forecaster does not require future features.

        Returns
        -------
        PanelPredictiveForecast or DiscretePanelPredictiveForecast
            One self-contained distributional forecast per series-step pair.
            Call :meth:`to_frame` for point forecasts, or call :meth:`cdf`,
            :meth:`ppf`, and :meth:`interval` for probabilistic results. A
            discrete family such as :class:`NegativeBinomialFamily` returns a
            :class:`DiscretePanelPredictiveForecast`, which also exposes
            :meth:`pmf`. Every method returns a DataFrame on the same row grid.

        Raises
        ------
        ValueError
            If predicted means are non-finite or the fitted forecast state is
            invalid.

        Examples
        --------
        ``forecast = model.predict_distribution(h=7)``

        ``forecast.ppf([0.5, 0.9, 0.95])``

        ``forecast.interval(0.9)``
        """
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
        forecast_type = (
            DiscretePanelPredictiveForecast
            if self.distribution_family_.is_discrete
            else PanelPredictiveForecast
        )
        return forecast_type(
            df_pred,
            predictive,
            model="lambda_t",
            id_col=self.id_col,
            time_col=self.time_col,
        )
