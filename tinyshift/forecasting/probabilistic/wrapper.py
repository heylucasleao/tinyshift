# Copyright (c) 2024-2025 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License


from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

from tinyshift.utils.imports import requires_extra

from .calibration import Calibration, Calibrator
from .family import DistributionFamily, NegativeBinomialFamily
from .forecast import DiscretePanelPredictiveForecast, PanelPredictiveForecast


class TwoStageForecasterWrapper(BaseEstimator, RegressorMixin):
    """Two-stage probabilistic forecasting wrapper using MLForecast.

    This model decouples conditional expectation from dispersion:
        1. Employs an `MLForecast` regressor to forecast the conditional
           expectation (lambda_t).
        2. Calibrates global, per-series, and per-horizon distribution parameters
           with empirical-Bayes shrinkage on temporal cross-validation predictions.
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
    - **Out-of-Sample Dispersion Calibration**: Shrinks family-specific
      series×horizon parameters toward per-series and global estimates.
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
    ) -> None:
        """Validate temporal calibration parameters."""
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

    @staticmethod
    def _single_model(models_dict, missing_message: str):
        """Return the only configured model or reject invalid containers."""
        if not models_dict:
            raise ValueError(missing_message)
        if len(models_dict) > 1:
            raise ValueError(
                "TwoStageForecasterWrapper supports exactly 1 model, but found: "
                f"{list(models_dict.keys())}"
            )
        return next(iter(models_dict.items()))

    @property
    def model_name(self) -> str:
        """Extracts the name/key of the underlying model from MLForecast."""
        models_dict = getattr(self.fcst, "models_", None)
        if not models_dict:
            models_dict = getattr(self.fcst, "models", None)
        name, _ = self._single_model(
            models_dict, "The MLForecast object has no models configured."
        )
        return name

    @property
    def model(self):
        """Extracts and validates the underlying trained estimator from the MLForecast container."""
        models_dict = getattr(self.fcst, "models_", None)
        _, model = self._single_model(
            models_dict, "The MLForecast object has not been fitted yet."
        )
        return model

    def _calibrate_dispersion_cv(
        self,
        df: pd.DataFrame,
        h: int,
        n_windows: int,
        step_size: int | None,
        refit: bool | int,
    ) -> Calibration:
        """Generate OOF predictions and fit hierarchical dispersion."""
        cv_df = self._dispersion_cv_predictions(
            df, h, n_windows, step_size, refit
        )
        calibrator = Calibrator(
            family=self.distribution_family_,
            id_col=self.id_col,
            target_col=self.target_col,
            prediction_col=self.model_name,
        )
        return calibrator.fit(cv_df)

    def _dispersion_cv_predictions(
        self,
        df: pd.DataFrame,
        h: int,
        n_windows: int,
        step_size: int | None,
        refit: bool | int,
    ) -> pd.DataFrame:
        """Generate OOF means and identify their forecast horizons."""
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
        )

        if cv_df.empty:
            raise RuntimeError("Dispersion calibration produced no series.")
        cv_df = cv_df.copy()
        cv_df["_horizon"] = (
            cv_df.groupby([self.id_col, "cutoff"], sort=False).cumcount() + 1
        )
        return cv_df

    def _fit_base_forecaster(
        self,
        df: pd.DataFrame,
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
        )

        return self.fcst.ts.features_order_

    def _set_fit_state(
        self,
        id_col: str,
        time_col: str,
        target_col: str,
        static_features: list | None,
    ) -> None:
        """Store the panel schema used by fitting and prediction."""
        self.id_col = id_col
        self.time_col = time_col
        self.target_col = target_col
        self.static_features = static_features or []

    def _resolve_distribution_family(self) -> DistributionFamily:
        """Return the configured family, defaulting to Negative Binomial."""
        family = (
            NegativeBinomialFamily() if self.distribution is None else self.distribution
        )
        if not isinstance(family, DistributionFamily):
            raise TypeError(
                "distribution must be a DistributionFamily instance or None."
            )
        return family

    @requires_extra("series")
    def fit(
        self,
        df_train: pd.DataFrame,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
        static_features: list | None = None,
        h: int = 14,
        n_windows: int = 10,
        step_size: int | None = None,
        refit: bool | int = True,
    ) -> "TwoStageForecasterWrapper":
        """Fit the point model and hierarchical shrinkage dispersions.

        Process
        -------
        Runs temporal cross-validation, calibrates the selected distribution
        family per series, and then fits MLForecast on all training rows.

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

        Returns
        -------
        self : TwoStageForecasterWrapper
            Fitted instance of the TwoStageForecasterWrapper class.
        """

        self._set_fit_state(id_col, time_col, target_col, static_features)
        self._validate_fit_parameters(h, n_windows, step_size)
        self.distribution_family_ = self._resolve_distribution_family()
        numeric_label = "numeric counts" if self.distribution is None else "numeric"
        self._validate_training_target(
            df_train,
            target_col,
            self.distribution_family_,
            numeric_label,
        )
        self.calibration_ = self._calibrate_dispersion_cv(
            df_train, h, n_windows, step_size, refit
        )
        self.exog_cols_ = self._fit_base_forecaster(df_train)

        return self

    def _prediction_frame(self, h: int, X_df: pd.DataFrame | None) -> pd.DataFrame:
        """Predict and normalize conditional means on the panel grid."""
        frame = self.fcst.predict(h=h, X_df=X_df)
        frame = frame.rename(columns={self.model_name: "lambda_t"})
        means = frame["lambda_t"].to_numpy(dtype=float)
        if not np.all(np.isfinite(means)):
            raise ValueError("Predicted mean values must be finite.")
        frame["lambda_t"] = np.maximum(means, 1e-6)
        return frame

    def _prediction_dispersions(self, frame: pd.DataFrame) -> np.ndarray:
        """Resolve and attach dispersions for all forecast rows."""
        horizons = frame.groupby(self.id_col, sort=False).cumcount() + 1
        dispersions = np.fromiter(
            (
                self.calibration_.resolve(uid, int(horizon))
                for uid, horizon in zip(frame[self.id_col], horizons)
            ),
            dtype=float,
            count=len(frame),
        )
        frame[self.distribution_family_.parameter_column] = dispersions
        return dispersions

    def _build_forecast(self, frame: pd.DataFrame):
        """Construct the predictive distribution and its panel facade."""
        means = frame["lambda_t"].to_numpy(dtype=float)
        dispersions = self._prediction_dispersions(frame)
        predictive = self.distribution_family_.distribution(means, dispersions)
        forecast_type = (
            DiscretePanelPredictiveForecast
            if self.distribution_family_.is_discrete
            else PanelPredictiveForecast
        )
        return forecast_type(
            frame,
            predictive,
            model="lambda_t",
            id_col=self.id_col,
            time_col=self.time_col,
        )

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
        return self._build_forecast(self._prediction_frame(h, X_df))
