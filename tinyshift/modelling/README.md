# ML Modeling Utilities Module (`modelling`)

The `modelling` module provides sklearn-compatible preprocessing, feature engineering, and decomposed forecasting utilities for robust machine learning workflows. It includes multicollinearity reduction, feature residualization, robust scaling, MSTL decomposition with forecasting, and time-series feature generation.

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

### 4. Decomposed Forecasting Wrapper (`dmstl.py`)

#### **`DMSTLWrapper`** - MSTL-Based Trend/Seasonality + Residual ML Modeling
Wrapper that decomposes a panel time series using MSTL, fits statistical models for trend and seasonality, and models residual structure with `MLForecast`.

```python
from tinyshift.modelling import DMSTLWrapper
from mlforecast import MLForecast
from sklearn.ensemble import RandomForestRegressor

mf_resid = MLForecast(models=[RandomForestRegressor()], freq='D')
wrapper = DMSTLWrapper(
    mf_resid=mf_resid,
    season_length=[7, 365],
)

wrapper.fit(df, id_col='unique_id', time_col='ds', target_col='y')
predictions = wrapper.predict(df_future)
```

**When to use:**
- For multi-seasonal panel forecasting
- When you want separate models for trend, seasonality, and residuals
- To combine statistical decomposition with ML residual forecasting

---

### 5. Time Series Feature Engineering (`ts_features.py`)

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
- `DMSTLWrapper`
- `relative_strength_index`
- `standardize_returns`
- `fourier_seasonality`
- `estimate_history_length`

---

## Notes

- `DMSTLWrapper` depends on `statsmodels`, `statsforecast`, and `mlforecast`.
- `fourier_seasonality` expects a pandas DataFrame with a datetime column.
- `ts_features` functions are lightweight feature engineering helpers for time-series forecasting.
