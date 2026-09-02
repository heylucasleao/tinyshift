# Data Drift Detection Module (`drift`)

The `drift` module provides data drift detection tools for monitoring distribution changes in categorical and continuous features over time. Built for production MLOps workflows with automatic threshold detection and comprehensive statistical analysis.

## Features

### 1. Categorical Drift Detection (`CatDrift`)

#### **`CatDrift`** - Categorical Feature Drift Monitoring
Detects drift in categorical data distributions using probability-based distance metrics with time-series grouping capabilities.

```python
from tinyshift.drift import CatDrift
import pandas as pd

# Initialize detector
detector = CatDrift(
    freq="D",                    # Daily time grouping
    func="chebyshev",           # Distance metric
    drift_limit="auto",         # Automatic threshold
    method="expanding"          # Comparison strategy
)

# Fit on reference data
detector.fit(reference_df)

# Score new data for drift
drift_scores = detector.predict(analysis_df)
```

**Available Distance Metrics:**
- **`"chebyshev"`**: Maximum absolute difference between category probabilities
  - **Use case**: Robust to outlier categories, focuses on worst-case divergence
  - **Range**: [0, 1], where 1 = complete distribution change
- **`"jensenshannon"`**: Jensen-Shannon distance (the square root of the divergence; symmetric and bounded)
  - **Use case**: Balanced sensitivity, probabilistically interpretable
  - **Range**: [0, 1], where 0 = identical distributions
- **`"psi"`**: Population Stability Index (banking/credit risk standard)
  - **Use case**: Highly sensitive to small changes, industry standard
  - **Range**: [0, ∞], where PSI > 0.25 typically indicates significant drift

**Comparison Methods:**
- **`"expanding"`**: Each point compared against all accumulated historical data
  - **Use case**: Cumulative drift detection, stable baselines
- **`"jackknife"`**: Leave-one-out comparison against all other time points
  - **Use case**: Peer comparison, anomaly detection in time series

---

### 2. Continuous Drift Detection (`ConDrift`)

#### **`ConDrift`** - Continuous Feature Drift Monitoring
Monitors numerical feature distributions using optimal transport and statistical distance measures.

```python
from tinyshift.drift import ConDrift

# Initialize detector
detector = ConDrift(
    freq="W",                   # Weekly time grouping
    func="ws",                  # Wasserstein distance
    drift_limit="auto",         # Automatic threshold
    method="expanding"          # Comparison strategy
)

# Fit and score
detector.fit(reference_df)
drift_scores = detector.predict(analysis_df)
```

**Available Distance Metrics:**
- **`"ws"`**: Wasserstein distance (Earth Mover's Distance)
  - **Use case**: Captures shape, location, and scale changes
  - **Range**: [0, ∞], interpretable as "cost to transform distributions"
  - **Advantages**: Preserves metric properties and compares empirical distributions directly
  - **Caution**: Extreme values can materially increase the distance


---

## Automatic Threshold Detection

Both detectors support automatic drift threshold determination:

```python
# Automatic threshold methods
detector = CatDrift(freq="D", drift_limit="auto")   # Statistical interval estimation
detector = CatDrift(freq="D", drift_limit="mad")    # Median Absolute Deviation
detector = CatDrift(freq="D", drift_limit="stddev")

# Manual threshold specification
detector = CatDrift(freq="D", drift_limit=(None, 0.95))
```

**Threshold Methods:**
- **`"auto"`**: Statistical interval estimation using reference distribution
- **`"mad"`**: Median Absolute Deviation (robust to outliers)
- **`"stddev"`**: Standard deviation-based bounds (assumes normality)
- **`(lower, upper)`**: Custom threshold tuple

Thresholds are calibrated independently for each series. Both expanding and
jackknife calibration require at least two reference periods per series. A
period is classified as drift only when its metric is strictly greater than the
upper threshold.

Categorical scoring aligns the union of reference and current categories, so a
previously unseen category contributes to the distance instead of being
discarded. Prediction currently requires every series ID to have appeared in
the reference data; unknown IDs raise a clear error rather than borrowing an
unrelated threshold.

---

## Time Series Grouping & Frequency

All detectors support pandas frequency strings for temporal aggregation:

```python
# Common frequency patterns
CatDrift(freq="D")     # Daily aggregation
CatDrift(freq="W")     # Weekly aggregation  
CatDrift(freq="M")     # Monthly aggregation
CatDrift(freq="H")     # Hourly aggregation
CatDrift(freq="15T")   # 15-minute intervals
CatDrift(freq="QS")    # Quarter start
```

**Required DataFrame Structure:**
```python
# Expected column structure
reference_df = pd.DataFrame({
    'unique_id': ['entity_1', 'entity_1', 'entity_2', ...],
    'ds': ['2024-01-01', '2024-01-02', '2024-01-03', ...],  # datetime
    'y': ['category_A', 'category_B', 'category_A', ...]     # target feature
})
```

---

## Detector Comparison Matrix

| Feature | CatDrift | ConDrift |
|---------|----------|----------|
| **Data Type** | Categorical/discrete | Numerical/continuous |
| **Distance Metrics** | Chebyshev, Jensen-Shannon, PSI | Wasserstein |
| **Interpretation** | Probability distribution shifts | Shape/location/scale changes |
| **Sensitivity** | High (especially PSI) | Moderate, robust |
| **Computational Cost** | Low (histogram-based) | Moderate (optimal transport) |
| **Outlier Robustness** | Medium | Low to medium; extreme values affect Wasserstein distance |
| **Best Use Cases** | Categorical features, fraud detection | Numerical features, sensor data |

---

## Statistical Foundations

### **Distance Metric Properties**

| Metric | Symmetry | Bounded | Triangle Inequality | Interpretability |
|--------|----------|---------|-------------------|------------------|
| **Chebyshev** | ✅ | ✅ [0,1] | ❌ | Maximum single-category change |
| **Jensen-Shannon** | ✅ | ✅ [0,1] | ✅ | Square root of an information-theoretic divergence |
| **PSI** | ❌ | ❌ [0,∞) | ❌ | Banking industry standard |
| **Wasserstein** | ✅ | ❌ [0,∞) | ✅ | Transportation cost |
