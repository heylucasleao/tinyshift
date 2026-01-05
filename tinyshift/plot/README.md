# Visualization Module (`plot`)

The `plot` module provides comprehensive visualization tools for exploratory data analysis, correlation analysis, time series diagnostics, and classification model evaluation. Built on Plotly for interactive, publication-ready visualizations that support both statistical analysis and MLOps monitoring workflows.

## Features

### 1. Classification Model Evaluation (`calibration.py`)

#### **Model Calibration & Reliability**

#### **`reliability_curve`**
Generates a reliability curve (calibration curve) for binary classifiers, plotting true probability vs predicted probability.

```python
from tinyshift.plot import reliability_curve

reliability_curve(
    clf=classifier,
    X=X_test,
    y=y_test,
    model_name="RandomForest",
    n_bins=15
)
```

**Parameters:**
- `clf`: Trained classifier with predict_proba method
- `X`: Input feature data for evaluation
- `y`: True binary labels (0 or 1)
- `model_name`: Name to display in legend (default: "Model")
- `n_bins`: Number of bins for the curve (default: 15)
- `fig_type`: Display renderer (default: None)

**When to use:** 
- Assess model calibration quality
- Identify over/under-confident predictions
- Compare calibration across different models

---

#### **`score_distribution`**
Displays histogram of predicted probability scores to understand model confidence patterns.

```python
from tinyshift.plot import score_distribution

score_distribution(
    clf=classifier,
    X=X_test,
    nbins=20
)
```

**Parameters:**
- `clf`: Trained classifier with predict_proba method
- `X`: Input feature data
- `nbins`: Number of histogram bins (default: 15)
- `fig_type`: Display renderer (default: None)

**When to use:**
- Analyze distribution of model confidence
- Identify calibration issues (e.g., overconfidence)
- Understand prediction patterns

---

#### **Classification Performance**

#### **`confusion_matrix`**
Interactive confusion matrix heatmap with percentage annotations for binary classification.

```python
from tinyshift.plot import confusion_matrix

confusion_matrix(
    clf=classifier,
    X=X_test,
    y=y_test,
    percentage_by_class=True
)
```

**Parameters:**
- `clf`: Trained classifier with predict method
- `X`: Input feature data
- `y`: True binary labels
- `fig_type`: Display renderer (default: None)
- `percentage_by_class`: Show percentages by class vs overall (default: True)

**When to use:**
- Evaluate classification performance
- Identify class-specific errors
- Compare FP/FN trade-offs

---

#### **Conformal Prediction**

#### **`efficiency_curve`**
Visualizes efficiency and validity trade-off for conformal prediction classifiers across different error rates.

```python
from tinyshift.plot import efficiency_curve

efficiency_curve(
    clf=conformal_classifier,
    X=X_test,
    width=800,
    height=400
)
```

**Parameters:**
- `clf`: Conformal classifier with predict_set method
- `X`: Input feature data
- `fig_type`: Display renderer (default: None)
- `width`: Figure width in pixels (default: 800)
- `height`: Figure height in pixels (default: 400)

**When to use:**
- Assess conformal predictor calibration
- Optimize efficiency vs validity trade-off
- Validate coverage guarantees

---

#### **Statistical Distributions**

#### **`beta_confidence_analysis`**
Plots Beta distribution PDF with filled area, useful for Bayesian analysis and calibration studies.

```python
from tinyshift.plot import beta_confidence_analysis

beta_confidence_analysis(
    alpha=2,
    beta_param=5,
    fig_type=None
)

# Low confidence model (few successes, many failures)
beta_confidence_analysis(alpha=15, beta_param=85)
```

**Parameters:**
- `alpha`: Model successes/correct predictions (must be positive)
- `beta_param`: Model failures/incorrect predictions (must be positive)
- `fig_type`: Display renderer (default: None)

**When to use:**
- Assess model readiness for production deployment
- Evaluate deployment confidence based on success/failure ratio
- Visualize risk assessment for MLOps decision making
- Compare model reliability across different validation periods

---

### 2. Statistical Power Analysis (`power.py`)

#### **A/B Testing & Experimental Design**

#### **`power_curve`**
Generates an interactive power analysis plot showing the relationship between sample size and statistical power for two-sample t-tests.

```python
from tinyshift.plot import power_curve

power_curve(
    effect_size=0.5,
    alpha=0.05,
    power_target=0.80,
    height=500,
    width=600
)
```

**Parameters:**
- `effect_size`: Cohen's d effect size to be detected (must be positive)
- `alpha`: Significance level (Type I error probability) (default: 0.05)
- `power_target`: Target statistical power level (default: 0.80)
- `height`, `width`: Figure dimensions in pixels (default: 500x600)
- `fig_type`: Display renderer (default: None)

**When to use:**
- Plan sample sizes for A/B tests and experiments
- Understand power-sample size trade-offs
- Validate experimental design before data collection
- Assess detectability of effect sizes

---

#### **`power_vs_allocation`**
Visualizes how treatment allocation proportion affects statistical power while keeping total sample size fixed.

```python
from tinyshift.plot import power_vs_allocation

power_vs_allocation(
    effect_size=0.3,
    sample_size=1000,
    alpha=0.05,
    height=500,
    width=600
)
```

**Parameters:**
- `effect_size`: Cohen's d effect size to be detected (must be positive)
- `sample_size`: Total sample size (control + treatment combined)
- `alpha`: Significance level (Type I error probability) (default: 0.05)
- `height`, `width`: Figure dimensions in pixels (default: 500x600)
- `fig_type`: Display renderer (default: None)

**When to use:**
- Optimize treatment allocation in A/B tests
- Understand why balanced allocation (50/50) maximizes power
- Assess impact of unbalanced experimental designs
- Plan resource allocation in experiments

---

### 3. Correlation Analysis (`correlation.py`)

#### **`corr_heatmap`**
Generates an interactive correlation heatmap with diverging color scale and automatic feature handling.

```python
from tinyshift.plot import corr_heatmap
import numpy as np

# Basic usage
X = np.random.randn(100, 5)
corr_heatmap(X, width=800, height=600)
```

**Parameters:**
- `X`: numpy array or pandas DataFrame with numeric features
- `width`, `height`: Figure dimensions in pixels (default: 1600x1600)  
- `fig_type`: Display type ('notebook' for Jupyter, None for default)

**When to use:** 
- Identify multicollinearity in features
- Detect feature relationships before modeling
- Visual correlation matrix exploration

---

### 4. Time Series Diagnostics (`diagnostic.py`)

#### **`seasonal_decompose`**
Performs MSTL (Multiple Seasonal-Trend decomposition using Loess) with trend significance testing and residual analysis.

```python
from tinyshift.plot import seasonal_decompose
import pandas as pd

# Multiple seasonality decomposition
seasonal_decompose(
    X=time_series,
    periods=[7, 365],  # Weekly and yearly patterns
    nlags=10,
    width=1300,
    height=1200
)
```

**Parameters:**
- `X`: Time series data (numpy array, list, or pandas Series)
- `periods`: Single period (int) or multiple periods (list) for seasonal components
- `nlags`: Number of lags for Ljung-Box residual test (default: 10)
- `width`, `height`: Figure dimensions (default: 1300x1200)

**Output Components:**
- **Trend**: Long-term directional movement
- **Seasonal**: Regular periodic patterns 
- **Residuals**: Remaining unexplained variation
- **Statistics Panel**: Trend significance (R²/p-value) and Ljung-Box test

**When to use:**
- Decompose complex time series with multiple seasonalities
- Validate seasonal patterns in demand forecasting
- Assess model residuals for autocorrelation

---

#### **`stationarity_analysis`**
Comprehensive stationarity testing using Augmented Dickey-Fuller test with visual rolling statistics.

```python
from tinyshift.plot import stationarity_analysis

stationarity_analysis(
    X=time_series,
    window=30,  # Rolling window size
    width=1200,
    height=800
)
```

**Parameters:**
- `X`: Time series data
- `window`: Rolling window size for statistics (default: 30)
- `width`, `height`: Figure dimensions (default: 1200x800)

**Output:**
- Original time series plot
- Rolling mean and standard deviation
- ADF test results (statistic, p-value, critical values)

**When to use:**
- Test stationarity assumptions before ARIMA modeling
- Identify trend and variance changes over time
- Validate differencing transformations

---

#### **`residual_analysis`**
Comprehensive residual diagnostics for model validation and assumption testing.

```python
from tinyshift.plot import residual_analysis

residual_analysis(
    residuals=model_residuals,
    nlags=20,
    width=1400,
    height=1000
)
```

**Parameters:**
- `residuals`: Model residual values
- `nlags`: Number of lags for autocorrelation analysis (default: 20)
- `width`, `height`: Figure dimensions (default: 1400x1000)

**Output Panels:**
1. **Residuals vs Time**: Temporal patterns and heteroscedasticity
2. **Q-Q Plot**: Normality assessment
3. **ACF/PACF**: Autocorrelation structure
4. **Histogram**: Distribution shape
5. **Statistics**: Ljung-Box test, ARCH test, normality tests

**When to use:**
- Validate regression model assumptions
- Diagnose time series model adequacy
- Identify remaining patterns in residuals

---

#### **`pami` (Permutation Auto Mutual Information)**
Visualizes nonlinear autocorrelation using permutation-based mutual information across multiple lags.

```python
from tinyshift.plot import pami

pami(
    X=time_series,
    max_lag=24,
    width=1000,
    height=600
)
```

**Parameters:**
- `X`: Time series data
- `max_lag`: Maximum lag to compute mutual information (default: 24)
- `width`, `height`: Figure dimensions (default: 1000x600)

**When to use:**
- Detect nonlinear autocorrelation patterns
- Identify optimal lag structure for nonlinear models
- Complement traditional ACF/PACF analysis

---

#### **`forest_plot`**
Creates a forest-style plot of group means with their confidence intervals.

```python
from tinyshift.plot import forest_plot

fig = forest_plot(
    df=df,
    feature='outcome',
    group_col='group',
    confidence=0.95
)
```

**Parameters:**
- `df`: pandas DataFrame containing the data.
- `feature`: Name of the numeric column to summarize.
- `group_col`: Name of the categorical column used for grouping.
- `confidence`: Confidence level between 0 and 1 (default 0.95).
- `fig_type`, `height`, `width`: Plot rendering options.

**When to use:**
- Compare group means with confidence intervals across categories.
- Present effect estimates in a compact forest-style plot.

## Function Comparison Matrix

### Statistical Power Analysis

| Function | Purpose | Input Type | Key Output | Best Use Case |
|----------|---------|------------|------------|---------------|
| **`power_curve`** | Sample size planning | Effect size + parameters | Power vs sample size curve | A/B test planning, experimental design |
| **`power_vs_allocation`** | Allocation optimization | Effect size + total sample | Power vs allocation curve | Treatment allocation planning |

### Binary Classification Model Evaluation

| Function | Purpose | Input Type | Key Output | Best Use Case |
|----------|---------|------------|------------|---------------|
| **`reliability_curve`** | Model calibration assessment | Classifier + test data | Calibration curve | Evaluate prediction confidence accuracy |
| **`score_distribution`** | Confidence pattern analysis | Classifier + features | Score histogram | Identify over/underconfident predictions |
| **`confusion_matrix`** | Classification performance | Classifier + test data | Interactive heatmap | Analyze class-specific errors |
| **`efficiency_curve`** | Conformal prediction trade-offs | Conformal classifier | Efficiency vs validity | Optimize prediction set performance |
| **`beta_confidence_analysis`** | Production confidence assessment | Alpha/beta parameters | PDF plot | Evaluate model deployment readiness |

### Time Series & Correlation Analysis

| Function | Purpose | Input Type | Key Output | Best Use Case |
|----------|---------|------------|------------|---------------|
| **`corr_heatmap`** | Correlation visualization | Tabular data | Interactive heatmap | Feature selection, multicollinearity detection |
| **`seasonal_decompose`** | Time series decomposition | Time series | Trend/seasonal/residual components | Seasonal pattern analysis, forecasting prep |
| **`stationarity_analysis`** | Stationarity testing | Time series | ADF test + rolling stats | ARIMA modeling prep, trend detection |
| **`residual_analysis`** | Model diagnostics | Residuals | Multiple diagnostic plots | Model validation, assumption testing |
| **`pami`** | Nonlinear correlation | Time series | Mutual information by lag | Nonlinear dependency detection |

---

## Integration with TinyShift Workflow

### **Experimental Design & A/B Testing**
```python
# 1. Plan sample sizes for experiments
power_curve(effect_size=0.3, alpha=0.05, power_target=0.80)

# 2. Optimize treatment allocation
power_vs_allocation(effect_size=0.3, sample_size=1000)
```

### **Classification Model Validation**
```python
# 3. Model calibration assessment
reliability_curve(clf, X_test, y_test, model_name="XGBoost")

# 4. Prediction confidence analysis
score_distribution(clf, X_test)

# 5. Performance evaluation
confusion_matrix(clf, X_test, y_test)

# 4. Production deployment confidence
beta_confidence_analysis(alpha=successes, beta_param=failures)
```

### **Conformal Prediction Optimization**
```python
# 6. Efficiency-validity trade-off analysis
efficiency_curve(conformal_clf, X_test)
```

### **Data Quality Assessment**
```python
# 7. Correlation analysis for feature engineering
corr_heatmap(X_features)

# 8. Stationarity check before drift detection
stationarity_analysis(target_series)
```

### **Model Validation**
```python
# 9. Residual diagnostics after model fitting
residual_analysis(model.residuals_)

# 10. Seasonal validation for time series models
seasonal_decompose(y_true - y_pred, periods=[7, 30])
```

### **Advanced Pattern Detection**
```python
# 11. Nonlinear autocorrelation analysis
pami(feature_series, max_lag=48)
```

---

## Summary: Function Quick Reference

### Statistical Power Analysis
| Metric/Function | Input Required | Output | Question You Want to Answer |
|----------------|----------------|--------|----------------------------|
| **`power_curve`** | Effect size + parameters | Power vs sample size curve | "How many samples do I need to detect this effect?" |
| **`power_vs_allocation`** | Effect size + total sample | Power vs allocation curve | "How should I split my sample between groups?" |

### Model Calibration & Performance
| Metric/Function | Input Required | Output | Question You Want to Answer |
|----------------|----------------|--------|----------------------------|
| **`reliability_curve`** | Classifier + X + y | Calibration curve | "Are my model's confidence scores accurate?" |
| **`score_distribution`** | Classifier + X | Score histogram | "How confident is my model in its predictions?" |
| **`confusion_matrix`** | Classifier + X + y | Performance heatmap | "What types of errors is my model making?" |
| **`efficiency_curve`** | Conformal classifier + X | Efficiency vs validity | "How efficient are my prediction sets?" |
| **`beta_confidence_analysis`** | Alpha + beta parameters | PDF visualization | "How confident can I be putting this model in production?" |