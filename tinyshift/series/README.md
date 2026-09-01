# Time Series Module (`series`)

The `series` module of tinyshift provides quantitative tools for time series analysis, focusing on key features for MLOps, forecasting, and pattern detection. It covers metrics and transformations for volatility, intermittency, seasonality strength, trend, entropy, complexity, and forecast stability.

## Features

### 1. Outlier Detection & Volatility

- **`hampel_filter`**
  Detects outliers using a robust Hampel filter over a rolling window.
  **When to use:** To identify localized anomalies in time series data.

- **`bollinger_bands`**
  Computes Bollinger Bands and flags values outside the expected volatility envelope.
  **When to use:** To detect volatility breakouts and extreme deviations.

### 2. Forecastability, Entropy, Intermittency & Complexity

- **`foreca`**
  Measures the forecastability (ForeCA omega index) of a series from its spectral density.
  **When to use:** To assess whether a series is structured or noise-like.

- **`adi_cv`**
  Computes Average Demand Interval (ADI) and Coefficient of Variation (CV).
  **When to use:** To classify series as intermittent, erratic, or smooth.

- **`sample_entropy`**
  Calculates Sample Entropy for complexity and irregularity.
  **When to use:** To quantify unpredictability or regularity in time series.

- **`regularity_index`**
  Computes a regularity score from Sample Entropy.
  **When to use:** To quantify the temporal consistency of a series.

- **`permutation_entropy`**
  Computes ordinal complexity using Permutation Entropy.
  **When to use:** To assess how random the ordering of values is.

- **`theoretical_limit`**
  Computes the theoretical predictability ceiling (Πmax) based on normalized permutation entropy.
  **When to use:** To benchmark the maximum ordinal predictability of a series.

- **`permutation_auto_mutual_information`**
  Computes PAMI between a series and its lagged ordinal patterns.
  **When to use:** To detect non-linear temporal dependencies in ordinal structure.

### 3. Forecast Accuracy Metrics

- **`wape`**
  Calculates Weighted Absolute Percentage Error for one or more models.
  **When to use:** To compare volume accuracy across forecasts, even with zero demand.

- **`pbias`**
  Calculates Percent Bias for directional forecast drift.
  **When to use:** To detect systematic over- or under-forecasting.

- **`score`**
  Computes `WAPE + |PBias|` as a combined performance metric.
  **When to use:** To summarize accuracy and bias in one value.

- **`economic_loss`**
  Calculates the total financial loss from understock and overstock for one or
  more forecasting models using Newsvendor-style costs.
  **When to use:** To evaluate forecasts using business costs instead of only
  statistical error.

- **`rmae`**
  Computes Relative Mean Absolute Error against a baseline forecast.
  **When to use:** To evaluate whether a model adds value over a benchmark.

- **`fva_rmae`**
  Computes Forecast Value Added RMAE using naive or moving average baselines.
  **When to use:** To measure whether a model outperforms operational baselines.

- **`forecast_instability`**
  Measures instability across consecutive forecast origins.
  **When to use:** To quantify forecast nervousness and revision magnitude.

Example with per-row costs:

```python
from tinyshift.series import economic_loss

loss = economic_loss(
    df,
    models=["forecast"],
    id_col="unique_id",
    target_col="y",
    underage_cost="cu",
    overage_cost="co",
)
```

`economic_loss` calculates:

```text
understock = max(y - forecast, 0)
overstock = max(forecast - y, 0)
loss = understock * underage_cost + overstock * overage_cost
```

The result is aggregated by `unique_id` and returns the `economic_loss` metric
label. Costs can also be provided as fixed scalar values, such as
`underage_cost=3.0` and `overage_cost=1.0`.

### 4. Diagnostics & Decomposition

- **`detect_seasonal_periods`**
  Identifies candidate seasonal periods via FFT and spectral peak detection.
  **When to use:** To discover seasonality for decomposition or modeling.

- **`hurst_exponent`**
  Estimates the Hurst exponent and a p-value for the random walk hypothesis.
  **When to use:** To assess long-term memory and trend persistence.

- **`trend_significance`**
  Tests linear trend significance using R² and slope p-value.
  **When to use:** To determine whether a series has a meaningful linear trend.

- **`seasonal_significance`**
  Computes seasonal strength and performs an F-test on a seasonal component.
  **When to use:** To evaluate whether extracted seasonality is statistically significant.

- **`extract_mstl_components`**
  Converts `statsmodels` MSTL results into a tidy DataFrame of decomposed components.
  **When to use:** To work with MSTL decomposition output in tabular form.

### 5. Forecast Stability Metrics

- **`macv`**
  Mean Absolute Change Vertical (MAC(V)) for stability across forecast origins.
  **When to use:** To measure revision magnitude between consecutive forecast updates.

- **`mach`**
  Mean Absolute Change Horizontal (MAC(H)) for within-horizon smoothness.
  **When to use:** To assess the smoothness of a forecast window.

- **`mascv`**
  Mean Absolute Scaled Change Vertical (MASC(V)).
  **When to use:** To compare vertical stability normalized by seasonality.

- **`masch`**
  Mean Absolute Scaled Change Horizontal (MASC(H)).
  **When to use:** To compare horizontal stability normalized by seasonality.

- **`rmsscv`**
  Root Mean Squared Scaled Change Vertical (RMSSC(V)).
  **When to use:** To penalize larger vertical revisions more heavily.

- **`rmssch`**
  Root Mean Squared Scaled Change Horizontal (RMSSC(H)).
  **When to use:** To penalize larger horizontal revisions more heavily.

### 6. Forecast Stabilization

- **`vi`**
  Vertical Interpolation for stabilized forecasts using previous-origin anchors.
  **When to use:** To stabilize individual forecast points vertically.

- **`hpi`**
  Horizontal Partial Interpolation for smoother horizon transitions.
  **When to use:** To reduce jumpiness between adjacent horizons.

- **`hfi`**
  Horizontal Full Interpolation for maximum horizon smoothness.
  **When to use:** To generate smoother forecast curves using prior stabilized values.

## Notes

- The `series` module exports functions from `outlier`, `forecastability`, `stability`, `interpolation`, `metric`, and `diagnostic`.
- For decomposed forecasting wrappers such as `DMSTLWrapper`, see `tinyshift.forecasting`.

## Summary: Function Quick Reference

### Forecastability & Complexity
| Metric/Function                        | Range         | Interpretation                                             | Question You Want to Answer                                         |
|----------------------------------------|---------------|------------------------------------------------------------|--------------------------------------------------------------------|
| **foreca**                             | 0 → 1         | Forecastability (1 = highly predictable, 0 = noise)       | "How predictable is this time series?"                             |
| **ADI / CV**                           | ADI: 1 → ∞    | Intermittency and variability                              | "Is this series intermittent or erratic?"                          |
| **Sample Entropy**                     | 0 → ∞         | Complexity/irregularity                                     | "How complex or irregular is this time series?"                    |
| **Permutation Entropy**                | 0 → 1         | Ordinal complexity/randomness                               | "How random is the order of this time series?"                     |
| **Regularity Index**                   | 0 → 1         | Temporal regularity                                         | "How consistent are the values over time?"                        |
| **Theoretical Limit**                  | 0 → 1         | Predictability ceiling from ordinal structure               | "What is the maximum ordinal predictability?"                     |
| **PAMI**                               | 0 → ∞         | Lagged ordinal dependency                                   | "How much does recent behavior inform the future?"               |

### Forecast Accuracy Metrics
| Metric/Function                        | Range         | Interpretation                                             | Question You Want to Answer                                         |
|----------------------------------------|---------------|------------------------------------------------------------|--------------------------------------------------------------------|
| **WAPE**                               | 0 → ∞         | Volume-weighted accuracy                                     | "How far off are my forecasts in aggregate volume terms?"          |
| **PBias**                              | -∞ → ∞       | Directional bias                                             | "Am I systematically over- or under-forecasting?"                  |
| **Score**                              | 0 → ∞         | Accuracy + bias composite                                    | "How do accuracy and bias trade off in one metric?"               |
| **Economic Loss**                      | 0 → ∞         | Financial cost of understock and overstock                    | "What is the business cost of this forecast?"                    |
| **RMAE**                               | 0 → ∞         | Value against a baseline                                      | "Does this model outperform a benchmark?"                         |
| **FVA RMAE**                           | 0 → ∞         | Forecast Value Added                                          | "Does this model add operational value?"                          |
| **Forecast Instability**               | 0 → ∞         | Revision instability                                           | "How much do forecasts change between origins?"                   |

### Diagnostics & Decomposition
| Metric/Function                        | Range         | Interpretation                                             | Question You Want to Answer                                         |
|----------------------------------------|---------------|------------------------------------------------------------|--------------------------------------------------------------------|
| **detect_seasonal_periods**            | N/A           | Candidate season length detection                            | "What are the dominant season lengths?"                            |
| **Hurst Exponent**                     | 0 → 1         | Trend persistence / long memory                              | "Does the series have long-term memory?"                          |
| **Trend Significance**                 | 0 → 1         | Linear trend strength and significance                        | "Is there a meaningful linear trend?"                             |
| **Seasonal Significance**              | 0 → 1         | Seasonal strength and significance                            | "Is seasonality statistically significant?"                      |
| **extract_mstl_components**            | N/A           | MSTL decomposition extraction                                 | "How do I convert MSTL output into a DataFrame?"                 |

### Forecast Stability Metrics
| Metric/Function                        | Range         | Interpretation                                             | Question You Want to Answer                                         |
|----------------------------------------|---------------|------------------------------------------------------------|--------------------------------------------------------------------|
| **MAC(V)**                             | 0 → ∞         | Vertical stability across origins                             | "How much do forecasts change between updates?"                   |
| **MAC(H)**                             | 0 → ∞         | Horizontal smoothness in one window                           | "How smooth is the forecast curve?"                               |
| **MASC(V)**                            | 0 → ∞         | Scaled vertical stability                                      | "How stable are forecasts relative to seasonality?"               |
| **MASC(H)**                            | 0 → ∞         | Scaled horizontal stability                                    | "How smooth are forecasts relative to seasonality?"              |
| **RMSSC(V)**                           | 0 → ∞         | RMS scaled vertical stability                                  | "How stable are large vertical revisions?"                       |
| **RMSSC(H)**                           | 0 → ∞         | RMS scaled horizontal stability                                | "How stable are large horizontal revisions?"                     |

### Forecast Stabilization
| Metric/Function                        | Range         | Interpretation                                             | Question You Want to Answer                                         |
|----------------------------------------|---------------|------------------------------------------------------------|--------------------------------------------------------------------|
| **VI**                                 | Depends on data | Vertical forecast stabilization                               | "How can I stabilize individual forecast points?"                 |
| **HPI**                                | Depends on data | Partial horizontal smoothing                                   | "How can I smooth jumps between adjacent horizons?"              |
| **HFI**                                | Depends on data | Full horizontal smoothing                                      | "How can I create a maximally smooth forecast trajectory?"       |

### Outlier Detection & Stats
| Metric/Function                        | Range         | Interpretation                                             | Question You Want to Answer                                         |
|----------------------------------------|---------------|------------------------------------------------------------|--------------------------------------------------------------------|
| **Hampel Filter**                      | 0 or 1        | Local outlier detection                                       | "Is this point an outlier relative to its local window?"          |
| **Bollinger Bands**                    | 0 or 1        | Volatility breakout detection                                 | "Is the value outside the expected volatility range?"            |
