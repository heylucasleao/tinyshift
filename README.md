# TinyShift
<p align="center">
  <img src="https://github.com/user-attachments/assets/34668d33-459d-4dc3-b598-342130bf7db3" alt="tinyshift_full_logo" width="400" height="400">
</p>

**TinyShift** is a lightweight, sklearn-compatible Python library designed for **data drift detection**, **outlier identification**, and **MLOps monitoring** in production machine learning systems. The library provides modular, easy-to-use tools for detecting when data distributions or model performance change over time, with comprehensive visualization capabilities.

For enterprise-grade solutions, consider [Nannyml](https://github.com/NannyML/nannyml).

## Features

- **Data Drift Detection**: Categorical and continuous data drift monitoring with multiple distance metrics
- **Outlier Detection**: **HBOS**, **PCA-based** and **SPAD** outlier detection algorithms  
- **Classification Model Evaluation**: Calibration curves, confusion matrices, score distributions, and production confidence analysis
- **Time Series Analysis**: Seasonality decomposition, trend analysis, forecasting diagnostics, and forecast stabilization
- **Decomposed Forecasting**: DTL-based non-seasonal and DMSTL-based multi-seasonal forecasting for panel and long-horizon series
- **Probabilistic Demand Forecasting**: Two-stage discrete or continuous forecasts, calibrated distributions, forecast evaluation, and inventory optimization
- **Forecast Stability**: Metrics and interpolation methods for stable forecasting

## Technologies Used

- **Python 3.10+** 
- **Scikit-learn 1.3.0+**
- **Pandas 2.3.0+** 
- **NumPy**
- **SciPy**
- **Statsmodels 0.14.5+**
- **Plotly 5.22.0+** (optional, for plotting)

## 📦 Installation

Install the core package with pip:

```bash
pip install tinyshift
```

Or with uv:

```bash
uv add tinyshift
```

### Optional module extras

TinyShift now separates optional capabilities into extras so you can install only what you need. Some functions also use lazy importing so optional dependencies are loaded only when they are actually used, helping keep the import surface lightweight and avoiding unnecessary dependency overhead.

- `series`: forecasting and series-specific dependencies

```bash
pip install "tinyshift[series]"
# or
uv add "tinyshift[series]"
```

- `plot`: interactive plotting and export support

```bash
pip install "tinyshift[plot]"
# or
uv add "tinyshift[plot]"
```

- `notebook`: notebook support

```bash
pip install "tinyshift[notebook]"
# or
uv add "tinyshift[notebook]"
```

- `all`: install all optional extras

```bash
pip install "tinyshift[all]"
# or
uv add "tinyshift[all]"
```

### Development installation

Clone the repository and install from source:

```bash
git clone https://github.com/HeyLucasLeao/tinyshift.git
cd tinyshift
pip install -e ".[dev]"
```

With uv:

```bash
uv sync --extra dev
```

## 📖 Quick Start

### 1. Categorical Data Drift Detection

TinyShift provides sklearn-compatible drift detectors that follow the familiar `fit()` and `score()` pattern:

```python
import pandas as pd
from tinyshift.drift import CatDrift

# Load your data
df = pd.read_csv("data.csv")
reference_data = df[df["date"] < '2024-07-01']
analysis_data = df[df["date"] >= '2024-07-01'] 

# Initialize and fit the drift detector
detector = CatDrift(
    freq="D",                    # Daily frequency
    func="chebyshev",           # Distance metric
    drift_limit="auto",         # Automatic threshold detection
    method="expanding"          # Comparison method
)

# Fit on reference data
detector.fit(reference_data)

# Score new data for drift
drift_scores = detector.predict(analysis_data)
print(drift_scores)
```

Available distance metrics for **categorical** data:
- `"chebyshev"`: Maximum absolute difference between distributions
- `"jensenshannon"`: Jensen-Shannon divergence  
- `"psi"`: Population Stability Index

### 2. Continuous Data Drift Detection

For numerical features, use the continuous drift detector:

```python
from tinyshift.drift import ConDrift

# Initialize continuous drift detector
detector = ConDrift(
    freq="W",                   # Weekly frequency  
    func="ws",                  # Wasserstein distance
    drift_limit="auto",
    method="expanding"
)

# Fit and score
detector.fit(reference_data)
drift_predicts = detector.predict(analysis_data)
```

### 3. Outlier Detection

TinyShift includes sklearn-compatible outlier detection algorithms:

```python
from tinyshift.outlier import SPAD, HBOS, PCAReconstructionError

# SPAD (Simple Probabilistic Anomaly Detector)
spad = SPAD(plus=True)
spad.fit(X_train)

outlier_scores = spad.decision_function(X_test)
outlier_labels = spad.predict(X_test)

# HBOS (Histogram-Based Outlier Score)
hbos = HBOS(dynamic_bins=True)
hbos.fit(X_train, nbins="fd")
scores = hbos.predict(X_test)

# PCA-based outlier detection
pca_detector = PCAReconstructionError()
pca_detector.fit(X_train)
pca_scores = pca_detector.predict(X_test)
```
### 4. Binary Classification Model Evaluation

Evaluate and visualize classification model performance for production deployment:

```python
from tinyshift.plot import (
    reliability_curve,
    score_distribution, 
    confusion_matrix,
    efficiency_curve,
    beta_confidence_analysis
)

# Model calibration assessment
reliability_curve(
    clf=classifier,
    X=X_test,
    y=y_test,
    model_name="RandomForestClassifier",
    n_bins=15
)

# Analyze prediction confidence patterns
score_distribution(clf, X_test, nbins=20)

# Performance evaluation with interactive confusion matrix
confusion_matrix(clf, X_test, y_test, percentage_by_class=True)

# Conformal prediction analysis
efficiency_curve(conformal_classifier, X_test)

# Production deployment confidence analysis
beta_confidence_analysis(
    alpha=95, 
    beta_param=5, 
    fig_type=None
)
```
### 5. Time Series Analysis and Diagnostics

TinyShift provides comprehensive time series analysis capabilities:

```python
from tinyshift.plot import seasonal_decompose
from tinyshift.series import (
    trend_significance, 
    foreca, 
    sample_entropy,
    permutation_entropy,
    theoretical_limit,
    hurst_exponent,
    hampel_filter,
    bollinger_bands
)

seasonal_decompose(
    time_series, 
    periods=[7, 365],  # Weekly and yearly patterns
    width=1200, 
    height=800
)

# Test for significant trends
r_squared, p_value = trend_significance(time_series)

# Assess forecastability
forecastability = foreca(time_series)
print(f"Forecastability (Omega): {forecastability}")

# Measure complexity and regularity
complexity = sample_entropy(time_series, m=2, tolerance=0.2)
print(f"Sample Entropy: {complexity}")

# Measure ordinal complexity
perm_entropy = permutation_entropy(time_series, m=3, delay=1, normalize=True)
print(f"Permutation Entropy: {perm_entropy}")

# Calculate theoretical predictability limit
theo_limit = theoretical_limit(time_series, m=3, delay=1)
print(f"Theoretical Limit (Πmax): {theo_limit}")

# Detect long-term memory
hurst, p_value = hurst_exponent(time_series)
print(f"Hurst Exponent: {hurst}, P-value: {p_value}")

# Outlier detection in time series
outliers = hampel_filter(time_series, window_size=5)
outliers = bollinger_bands(time_series, window_size=20)

# Plot lag analysis with PAMI (Permutation Auto-Mutual Information)
from tinyshift.plot import pami
pami(time_series, nlags=20, m=3, delay=1, normalize=False)
```

### 6. Forecast Accuracy Metrics

TinyShift also includes forecast evaluation utilities in the series metrics module, implemented in [tinyshift/series/metric.py](tinyshift/series/metric.py). This module provides functions such as `wape`, `pbias`, `score`, `rmae`, and `fva_rmae` to compare forecasting models using aggregate error, bias, and baseline-relative performance:

```python
import pandas as pd
from tinyshift.series import wape, pbias, score, rmae, fva_rmae

# Example evaluation dataframe
# df must contain actual values in the 'y' column and model predictions as columns

wape_df = wape(df, models=["model_a", "model_b"], id_col="unique_id", target_col="y")
pbias_df = pbias(df, models=["model_a", "model_b"], id_col="unique_id", target_col="y")
score_df = score(df, models=["model_a", "model_b"], id_col="unique_id", target_col="y")
rmae_df = rmae(df, models=["model_a", "model_b"], baseline_col="naive", id_col="unique_id", target_col="y")

# Single-series Forecast Value Added (FVA) analysis
fva = fva_rmae(y_true, y_pred, nlags=1, baseline_type="naive")
print(f"FVA RMAE: {fva}")
```

These utilities cover:
- `wape`: weighted absolute percentage error for overall accuracy
- `pbias`: percent bias to detect over- or under-forecasting
- `score`: composite score combining WAPE and absolute bias
- `economic_loss`: financial loss from understock and overstock costs
- `rmae`: relative mean absolute error versus a baseline model
- `fva_rmae`: lead-time-aware RMAE for Forecast Value Added analysis

### 7. Forecast Stability and Interpolation

TinyShift includes forecast stability metrics and interpolation methods:

```python
from tinyshift.series import (
    forecast_instability,          # Period-over-period forecast instability
    macv, mach,           # Mean Absolute Change metrics
    mascv, masch,         # Mean Absolute Scaled Change metrics
    rmsscv, rmssch,       # Root Mean Squared Scaled Change metrics
    vi, hpi, hfi          # Interpolation methods
)

# Calculate forecast stability metrics
vertical_stability = macv(y_hat, y_hat_t_minus_1)
horizontal_stability = mach(y_hat) 

# Calculate period-over-period forecast variability (instability)
# `df` should contain `unique_id`, `ds` (ordered dates) and model forecast columns.
# Example: `variability(df, models=["model_a", "model_b"], ds_col="ds")`
instability_scores = forecast_instability(df, models=["model_a"], ds_col="ds")

# Scaled stability metrics
scaled_v_stability = mascv(y_train, y_hat, y_hat_t_minus_1, seasonality=12)
scaled_h_stability = masch(y_train, y_hat, seasonality=12)

# Apply forecast stabilization techniques
# Vertical Interpolation
stable_forecast = vi(y_hat, anchor, w_s=0.3)

# Horizontal Partial Interpolation
smooth_forecast = hpi(y_hat, w_s=0.4)

# Horizontal Full Interpolation
fully_stable_forecast = hfi(y_hat, w_s=0.5)
```

### 8. Modelling Utilities and Time-Series Feature Tools

The `tinyshift.modelling` package contains preprocessing and forecasting wrappers for machine learning workflows:

- `filter_features_by_vif` — remove highly correlated features using VIF filtering
- `FeatureResidualizer` — residualize correlated predictors while preserving information
- `RobustGaussianScaler` — robust scaling with winsorization and power transforms
- `DTLWrapper` — decomposed LOWESS trend plus ML residual forecasting for non-seasonal data
- `DMSTLWrapper` — decomposed MSTL forecasting wrapper for panel/multi-seasonal data
- `TwoStageForecasterWrapper` — configurable Negative Binomial or Gamma predictive distributions and inventory optimization on top of `MLForecast`
- `relative_strength_index`, `standardize_returns`, `fourier_seasonality`, `estimate_history_length` — feature engineering helpers for time-series models

Use `tinyshift.modelling` when you need preprocessors and decomposition-aware forecasting tools that complement the core `tinyshift.series` diagnostics and stability metrics.

### 9. Advanced Modeling Tools

```python
from tinyshift.modelling import filter_features_by_vif
from tinyshift.stats import bootstrap_bca_interval

#Residualizer
residualizer = FeatureResidualizer()
residualizer.fit(X_train[preprocess_columns], corrcoef=0.70)

#Train
X_train = X_train.astype({x: float for x in preprocess_columns})
X_train.loc[:, preprocess_columns] = residualizer.transform(X_train[preprocess_columns])

# Detect multicollinearity
mask = filter_features_by_vif(X_train, threshold=5, verbose=True)
X_train.columns = X_train.columns[mask]
X_test.columns = X_test.columns[mask]

#Test
X_test = X_test.astype({x: float for x in preprocess_columns})
X_test.loc[:, preprocess_columns] = residualizer.transform(X_test[preprocess_columns])

# Bootstrap confidence intervals
confidence_interval = bootstrap_bca_interval(
    data, 
    statistic=np.mean, 
    alpha=0.05, 
    n_bootstrap=1000
)
```

### 10. Decomposed Forecasting with DTL and DMSTL

TinyShift includes decomposed forecasting wrappers for non-seasonal and multi-seasonal panel data. `DTLWrapper` extracts a robust LOWESS trend and models residuals with `MLForecast`:

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

model = DTLWrapper(
    residual_model_callable=residual_model_callable,
    freq="D",
    nlags="auto",
    pami_params={"max_tau": 48, "m": 3, "delay": 1},
    trend_frac=0.2,
    robust=True,
)
model.fit(df, id_col="unique_id", time_col="ds", target_col="y")
preds = model.predict(h=14, stabilization_method="hfi", w_s=0.2)
```

For multiple seasonalities, use `DMSTLWrapper`:

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

model = DMSTLWrapper(
    residual_model_callable=residual_model_callable,
    freq="D",
    season_length="auto",
    seasonal_detection_params={"top_k": 2, "noise_threshold_factor": 1.5},
    nlags="auto",
    pami_params={"max_tau": 48, "m": 3, "delay": 1},
    log_transform=True,
)

model.fit(df, id_col="unique_id", time_col="ds", target_col="y")

preds = model.predict(h=14, stabilization_method="hfi", w_s=0.2)
print(preds.head())
```

### 11. Two-Stage Probabilistic Demand Forecasting

`TwoStageForecasterWrapper` separates the point forecast from uncertainty
calibration. An `MLForecast` model estimates the conditional mean (`lambda_t`),
while temporal cross-validation fits a per-series distribution parameter. The
default Negative Binomial family supports discrete demand and inventory
decisions; `GammaFamily` supports strictly positive continuous targets.

```python
import pandas as pd
from mlforecast import MLForecast
from sklearn.ensemble import RandomForestRegressor
from tinyshift.modelling import (
    FirstStageForecasterEvaluator,
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

# Discrete median and upper-tail demand forecasts
preds = model.predict(h=14, quantiles=(0.50, 0.95))

# Newsvendor-optimal inventory using shortage and holding costs
stock_plan = model.optimize(h=14, underage_cost=10.0, overage_cost=2.0)

# Exact probabilities P(Y=k), including the remaining upper tail
probabilities = model.pmf(h=14, max_k=10)

# Expected value of stocking each additional discrete inventory unit
marginal_value = model.marginal_benefit(
    h=14,
    underage_cost=10.0,
    overage_cost=2.0,
    max_k=10,
)

# Direct access to the aligned distributions
forecast, distribution = model.predict_distribution(h=14)
probability_below_five = distribution.cdf(5)
median = distribution.ppf(0.50)

# Cost columns may be supplied without exogenous forecast features.
costs = pd.DataFrame({"cu": [10.0] * len(forecast), "co": [2.0] * len(forecast)})
stock_plan = model.optimize(
    h=14,
    underage_cost="cu",
    overage_cost="co",
    X_df=costs,
)

# Continuous alternative; cdf/ppf/interval/sample share the same interface.
from tinyshift.modelling import GammaFamily

continuous_model = TwoStageForecasterWrapper(fcst, distribution=GammaFamily())
```

Evaluate only held-out or rolling-origin predictions after joining their actual
targets. The first-stage evaluator covers conditional-mean diagnostics and its
calibration table; the two-stage evaluator covers pinball loss and empirical
quantile coverage:

```python
mean_metrics = FirstStageForecasterEvaluator.evaluate(backtest_df)
calibration = FirstStageForecasterEvaluator.calibration_table(
    backtest_df, n_bins=10
)
probabilistic_metrics = TwoStageForecasterEvaluator.evaluate(
    backtest_df, quantiles=(0.05, 0.50, 0.95)
)
```

Persist the fitted wrapper so its base forecaster, selected family, and calibrated
per-series dispersion parameters remain together:

```python
import joblib

joblib.dump(model, "two_stage_forecaster.joblib")
restored_model = joblib.load("two_stage_forecaster.joblib")
```

Only load joblib files from trusted sources.

Negative Binomial targets must contain non-negative integer counts; Gamma targets
must be strictly positive. Install the `series` extra to use this wrapper:
`pip install "tinyshift[series]"`.

## 📁 Project Structure

```
tinyshift/
├── association_mining/          # Market basket analysis tools
│   ├── README.md                # Module documentation
│   ├── __init__.py              # Package exports
│   ├── analyzer.py              # Transaction pattern analysis
│   └── encoder.py               # Data encoder
├── drift/                       # Data drift detection
│   ├── README.md                # Module documentation
│   ├── __init__.py              # Package exports
│   ├── base.py                  # Base drift detection classes
│   ├── categorical.py           # CatDrift for categorical features
│   └── continuous.py            # ConDrift for numerical features
├── examples/                    # Jupyter notebook examples
│   ├── decomp_mstl_ml.ipynb     # MSTL decomposition and ML examples
│   ├── drift.ipynb              # Drift detection examples
│   ├── outlier.ipynb            # Outlier detection demos
│   ├── series.ipynb             # Time series analysis
│   ├── transaction_analyzer.ipynb  # Transaction analysis examples
│   └── ts_diagnostics.ipynb     # Time series diagnostics
├── modelling/                   # ML modeling utilities
│   ├── README.md                # Module documentation
│   ├── __init__.py              # Package exports
│   ├── dtl.py                   # DTL LOWESS trend/residual forecasting wrapper
│   ├── dmstl.py                 # DMSTL decomposed forecasting wrapper
│   ├── multicollinearity.py     # VIF-based multicollinearity detection
│   ├── residualizer.py          # Residualizer feature
│   ├── scaler.py                # Custom scaling transformations
│   └── ts_features.py           # Time-series feature engineering
├── outlier/                     # Outlier detection algorithms
│   ├── README.md                # Module documentation
│   ├── __init__.py              # Package exports
│   ├── base.py                  # Base outlier detection classes
│   ├── hbos.py                  # Histogram-Based Outlier Score
│   ├── pca.py                   # PCA-based outlier detection
│   └── spad.py                  # Simple Probabilistic Anomaly Detector
├── plot/                        # Visualization capabilities
│   ├── README.md                # Module documentation
│   ├── __init__.py              # Package exports
│   ├── calibration.py           # Binary classification model evaluation
│   ├── correlation.py           # Correlation analysis plots
│   ├── diagnostic.py            # Time series diagnostics plots
│   └── power.py                 # Power analysis and related plots
├── series/                      # Time series analysis tools
│   ├── README.md                # Module documentation
│   ├── __init__.py              # Package exports
│   ├── diagnostic.py            # Time series diagnostics and decomposition
│   ├── forecastability.py       # Forecast quality and complexity metrics
│   ├── interpolation.py         # Forecast stabilization methods
│   ├── metric.py                # Forecast accuracy and stability metrics
│   ├── outlier.py               # Time series outlier detection
│   └── stability.py             # Forecast stability metrics
└── stats/                       # Statistical utilities
    ├── __init__.py              # Package exports
    ├── bootstrap_bca.py         # Bootstrap confidence intervals
    ├── statistical_interval.py  # Statistical interval estimation
    └── utils.py                 # General statistical utilities
```

### Development Setup

```bash
git clone https://github.com/HeyLucasLeao/tinyshift.git
cd tinyshift
pip install -e ".[all]"
```

## 📋 Requirements

- **Python**: 3.10+
- **Core Dependencies**: 
  - pandas (>2.3.0)
  - scikit-learn (>1.3.0) 
  - statsmodels (>=0.14.5)
- **Optional Dependencies**:
  - plotly (>5.22.0) - for visualization
  - kaleido (<=0.2.1) - for static plot export
  - nbformat (>=5.10.4) - for notebook support

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by [Nannyml](https://github.com/NannyML/nannyml)
