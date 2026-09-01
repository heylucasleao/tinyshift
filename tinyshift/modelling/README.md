# Legacy Modelling Namespace (`tinyshift.modelling`)

This namespace is retained for backward compatibility. New code should use:

- `tinyshift.preprocessing` for scaling, residualization and VIF filtering;
- `tinyshift.features` for feature engineering;
- `tinyshift.forecasting` for DTL, DMSTL and probabilistic forecasting.

Existing imports continue to resolve, including historical submodule paths such
as `tinyshift.modelling.tsf`. The documentation for each implementation now
lives in the README of its canonical package.

---

# Previous combined documentation

The `modelling` module provides sklearn-compatible preprocessing, feature engineering, and decomposed forecasting utilities for robust machine learning workflows. It includes multicollinearity reduction, feature residualization, robust scaling, LOWESS and MSTL decomposition with forecasting, and time-series feature generation.

## Features

### 1. Multicollinearity Detection (`multicollinearity.py`)

#### **`filter_features_by_vif`** - Variance Inflation Factor Feature Selection
Iteratively removes features with high VIF values to reduce multicollinearity and improve model stability.

```python
from tinyshift.modelling import filter_features_by_vif
import numpy as np

X = np.random.randn(1000, 10)
X[:, 5] = X[:, 0] + X[:, 1] + np.random.randn(1000) * 0.1

feature_mask = filter_features_by_vif(
    X,
    threshold=5.0,
    verbose=True,
    n_jobs=-1,
)

X_filtered = X[:, feature_mask]
print(f"Kept {feature_mask.sum()} out of {len(feature_mask)} features")
```

**When to use:**
- Before linear regression or interpretable modeling
- When feature correlations are high
- To improve numerical stability and generalization

---

### 2. Feature Residualization (`residualizer.py`)

#### **`FeatureResidualizer`** - Residualize Correlated Predictors
Sklearn-compatible transformer that removes linear dependencies by replacing correlated features with their residuals from regression on the remaining set.

```python
from tinyshift.modelling import FeatureResidualizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

pipeline = Pipeline([
    ('residualizer', FeatureResidualizer()),
    ('regressor', LinearRegression()),
])

pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

**When to use:**
- When you want to keep all original features while reducing collinearity
- For preprocessing before regression or tree-based models
- To improve model interpretability without dropping variables

---

### 3. Robust Scaling (`scaler.py`)

#### **`RobustGaussianScaler`** - Winsorization + Power Transform + Standard Scaling
Robust scaler that reduces the influence of outliers, normalizes skewed distributions, and standardizes feature scale.

```python
from tinyshift.modelling import RobustGaussianScaler
import numpy as np

X = np.exp(np.random.randn(1000, 5))
X[0, :] = 1000

scaler = RobustGaussianScaler()
X_scaled = scaler.fit_transform(
    X,
    winsorize_method="mad",
    power_method="yeo-johnson",
)
```

**When to use:**
- For non-Gaussian data with outliers
- Prior to scale-sensitive algorithms like SVM or PCA
- When simple standard scaling is not enough

---

### 4. Decomposed Trend-Local Wrappers (`dtl/`)

#### **`DTLWrapper`** - LOWESS Trend + Residual ML Forecasting
Facade for non-seasonal panel forecasting. It extracts a robust LOWESS trend
for each `unique_id`, forecasts that component with StatsForecast, and selects
the residual strategy with `mode`:

- `mode="local"`: one residual `MLForecast` model per `unique_id`. The factory and lag configuration may be global or per series.
- `mode="global"`: one residual `MLForecast` model for the complete residual panel. The factory must be a single callable and receives the sorted union of the lags resolved per series.

```python
from tinyshift.modelling import DTLWrapper
from mlforecast import MLForecast
from sklearn.ensemble import RandomForestRegressor

def residual_model_callable(nlags, freq):
    return MLForecast(
        models=[RandomForestRegressor(random_state=42)],
        lags=nlags,
        freq=freq,
    )

wrapper = DTLWrapper(
    mode="global",
    residual_model_callable=residual_model_callable,
    freq="D",
    nlags="auto",
    pami_params={"max_tau": 48, "m": 3, "delay": 1},
    trend_frac=0.2,
    robust=True,
)

wrapper.fit(df, id_col='unique_id', time_col='ds', target_col='y')
predictions = wrapper.predict(h=14)
```

**When to use:**
- For non-seasonal panel forecasting
- When a smooth robust trend should be forecast separately from residual dynamics
- To combine LOWESS decomposition with machine-learning residual forecasts

Only `DTLWrapper` is part of the public DTL API. Select the local or global
residual strategy with `mode` when creating the wrapper.

### 5. Decomposed Forecasting Wrappers (`dmstl/`)

#### **`DMSTLWrapper`** - MSTL-Based Trend/Seasonality + Residual ML Modeling
Facade that decomposes each panel series with MSTL, fits statistical models for trend and seasonality, and delegates residual modeling to one of two strategies selected with `mode`.

- `mode="local"`: one residual `MLForecast` model per `unique_id`. A residual factory and lag configuration may be global or specific to each series.
- `mode="global"`: one residual `MLForecast` model for the complete residual panel. The factory must be a single callable, and the model receives the sorted union of the lags resolved for each series.

```python
from tinyshift.modelling import DMSTLWrapper
from mlforecast import MLForecast
from sklearn.ensemble import RandomForestRegressor

def residual_model_callable(nlags, freq):
    return MLForecast(
        models=[RandomForestRegressor(random_state=42)],
        lags=nlags,
        freq=freq,
    )

wrapper = DMSTLWrapper(
    mode="global",
    residual_model_callable=residual_model_callable,
    freq="D",
    season_length="auto",
    seasonal_detection_params={"top_k": 2, "noise_threshold_factor": 1.5},
    nlags="auto",
    pami_params={"max_tau": 48, "m": 3, "delay": 1},
)

wrapper.fit(df, id_col='unique_id', time_col='ds', target_col='y')
predictions = wrapper.predict(df_future)
```

Only `DMSTLWrapper` is part of the public DMSTL API. Select the residual
strategy with `mode="local"` or `mode="global"` when creating the wrapper.

**When to use:**
- For multi-seasonal panel forecasting
- When you want separate models for trend, seasonality, and residuals
- To combine statistical decomposition with ML residual forecasting

---

### 6. Two-Stage Probabilistic Forecasting (`tsf/`)

#### **`TwoStageForecasterWrapper`** - Probabilistic Demand Forecasting

Wraps a single-model `MLForecast` instance in a two-stage probabilistic workflow:

1. The base regressor forecasts the conditional mean demand (`lambda_t`).
2. Temporal cross-validation calibrates the selected family's dispersion parameter
   for each series, with a global median fallback for new series. The default
   Negative Binomial family uses `r_dispersion`; Gamma uses `shape_dispersion`.

The default Negative Binomial family produces discrete demand quantiles, exact
probability masses, Newsvendor-optimal stock levels, and the marginal benefit of
each additional inventory unit. Distribution construction is separated from
forecasting: every family exposes `cdf`, `ppf`, and `interval`, while
only discrete families expose `pmf`.

```python
import pandas as pd
from mlforecast import MLForecast
from sklearn.ensemble import RandomForestRegressor
from tinyshift.modelling import (
    FirstStageForecasterEvaluator,
    NewsvendorOptimizer,
    TwoStageForecasterEvaluator,
    TwoStageForecasterWrapper,
)

fcst = MLForecast(
    models=[RandomForestRegressor(random_state=42)],
    freq="D",
    lags=[1, 7, 14],
)

model = TwoStageForecasterWrapper(fcst)
model.fit(
    df_train,
    id_col="unique_id",
    time_col="ds",
    target_col="y",
    h=14,
    n_windows=5,
)

# Evaluate held-out or rolling-origin first-stage predictions
first_stage_summary = FirstStageForecasterEvaluator.evaluate(
    backtest_df,
    id_col="unique_id",
    time_col="ds",
)
calibration = FirstStageForecasterEvaluator.calibration_table(
    backtest_df, n_bins=10
)
probabilistic_summary = TwoStageForecasterEvaluator.evaluate(
    backtest_df, quantiles=(0.05, 0.50, 0.95)
)

# Forecast once and derive quantiles from the row-aligned distribution
forecast, distribution = model.predict_distribution(h=14)
forecast["q_50"] = distribution.ppf(0.50)
forecast["q_95"] = distribution.ppf(0.95)

# Newsvendor inventory level for fixed shortage and holding costs
stock_plan = NewsvendorOptimizer.optimize(
    forecast, distribution, underage_cost=10.0, overage_cost=2.0
)

# Exact probabilities from P(Y=0) through P(Y=10), plus P(Y>10)
probabilities = distribution.pmf(range(11))

# Expected marginal value of each additional discrete inventory unit
marginal_value = NewsvendorOptimizer.marginal_benefit(
    forecast,
    distribution,
    underage_cost=10.0,
    overage_cost=2.0,
    max_k=10,
)

probability_below_five = distribution.cdf(5)
median = distribution.ppf(0.50)
```

Cost columns used only by the Newsvendor decision do not need to be model
features. `X_df` may contain only row-aligned cost columns when the underlying
forecaster does not require future exogenous features:

```python
costs = pd.DataFrame(
    {"cu": [10.0] * len(forecast), "co": [2.0] * len(forecast)}
)
stock_plan = NewsvendorOptimizer.optimize(
    forecast,
    distribution,
    underage_cost="cu",
    overage_cost="co",
    cost_df=costs,
)
```

For strictly positive continuous targets, select the Gamma family explicitly:

```python
from tinyshift.modelling import GammaFamily, TwoStageForecasterWrapper

continuous_model = TwoStageForecasterWrapper(
    fcst,
    distribution=GammaFamily(),
)
continuous_model.fit(df_train, h=14, n_windows=5)

forecast, distribution = continuous_model.predict_distribution(h=14)
quantiles = distribution.ppf([0.1, 0.5, 0.9])
```

Use `FirstStageForecasterEvaluator` to inspect bias, false demand on zero-demand
days, and peak-demand deviation. Use `TwoStageForecasterEvaluator` to evaluate
quantile pinball loss and empirical coverage on out-of-sample predictions.

Persist the fitted wrapper—not only its base regressor—to retain the selected
family and calibrated per-series dispersion parameters:

```python
import joblib

joblib.dump(model, "two_stage_forecaster.joblib")
restored_model = joblib.load("two_stage_forecaster.joblib")
```

Only load joblib files from trusted sources.

**When to use:**
- For intermittent, erratic, or lumpy count demand
- When forecast uncertainty must be converted into inventory decisions
- When demand variance exceeds its mean and a Poisson model is too restrictive
- For strictly positive continuous targets when `GammaFamily` is selected

---

### 7. Time Series Feature Engineering (`ts_features.py`)

#### **`relative_strength_index`**
Computes the Relative Strength Index (RSI) for a univariate series.

#### **`standardize_returns`**
Computes log or simple returns and optionally standardizes them.

#### **`fourier_seasonality`**
Adds Fourier sine/cosine seasonal features for daily, weekly, monthly, quarterly, or yearly cycles.

#### **`estimate_history_length`**
Returns a heuristic history window size based on seasonal period and forecast horizon.

```python
from tinyshift.modelling import (
    relative_strength_index,
    standardize_returns,
    fourier_seasonality,
    estimate_history_length,
)

rsi = relative_strength_index(series, rolling_window=14)
returns = standardize_returns(series, log=True)
df = fourier_seasonality(df, time_col='ds', seasonality=['weekly', 'yearly'])
lag = estimate_history_length(seasonal_period=7, horizon=14)
```

**When to use:**
- To add momentum or return-based features
- To encode cyclic seasonality with Fourier terms
- To choose history length for lag-based forecasting models

---

## Exported API

The `tinyshift.modelling` package exports:

- `filter_features_by_vif`
- `FeatureResidualizer`
- `RobustGaussianScaler`
- `DTLWrapper`
- `DMSTLWrapper`
- `TwoStageForecasterWrapper`
- `DistributionFamily`, `NegativeBinomialFamily`, `GammaFamily`
- `PredictiveDistribution`, `DiscretePredictiveDistribution`
- `FirstStageForecasterEvaluator`
- `TwoStageForecasterEvaluator`
- `relative_strength_index`
- `standardize_returns`
- `fourier_seasonality`
- `estimate_history_length`

---

## Notes

- `DTLWrapper`, `DMSTLWrapper`, and `TwoStageForecasterWrapper` require the `series` extra.
- `TwoStageForecasterWrapper` defaults to Negative Binomial count targets; use
  `GammaFamily` for strictly positive continuous targets.
- `fourier_seasonality` expects a pandas DataFrame with a datetime column.
- `ts_features` functions are lightweight feature engineering helpers for time-series forecasting.
