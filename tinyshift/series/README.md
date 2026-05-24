# Time Series Module (`series`)

The `series` module of tinyshift provides quantitative tools for time series analysis, focusing on key features for MLOps, forecasting, and pattern detection. It covers metrics and transformations for volatility, intermittency, seasonality strength, trend, entropy, complexity, and forecast stability.

## Features

### 1. Outlier Detection & Volatility

- **`hampel_filter`**  
  Robustly detects outliers using the median and median absolute deviation (MAD) in a moving window.  
  **When to use:** To identify outliers in time series, sensors, or any data sensitive to extreme noise.

- **`bollinger_bands`**  
  Calculates Bollinger Bands, indicating periods of high or low volatility.  
  **When to use:** To detect volatility regime changes, breakouts, and overbought/oversold zones.

### 2. Forecastability, Entropy, Intermittency & Complexity

- **`foreca`**  
  Measures the forecastability (omega index) of a series, based on spectral entropy.  
  **When to use:** To assess the predictability potential of a series. Values close to 1 indicate strong structure (low entropy), values near 0 indicate noise (high entropy).

- **`adi_cv`**  
  Calculates Average Days of Inventory (ADI) and Coefficient of Variation (CV), useful for classifying series by intermittency and variability.  
  **When to use:**  
    - **High ADI:** intermittent series (e.g., sporadic sales)
    - **High CV:** erratic/lumpy series (high variability)
    - **Low ADI & CV:** smooth and predictable series

- **`sample_entropy` (SampEn)**  
  Computes Sample Entropy, a robust measure of complexity and irregularity in time series. Low values indicate more regularity (repetitive patterns), high values indicate greater complexity (less repetition, more "randomness").  
  **When to use:** To quantify the complexity/irregularity of a series, especially in physiological, financial, or industrial contexts. Useful for comparing variability patterns between series or periods.

- **`permutation_entropy`**  
  Measures permutation entropy, quantifying complexity based on the order of values.  
  **When to use:** To assess the randomness and ordinal complexity of a series, especially useful for pattern analysis and regime change detection.

- **`regularity_index`**  
  Measures temporal regularity based on Sample Entropy. Values close to 1 indicate high regularity/stability, values near 0 indicate high variability/complexity.  
  **When to use:** To quantify the consistency and regularity of values over time, useful in biomedical, industrial, and financial applications where magnitude stability matters.


- **`theoretical_limit`**  
  Calculates the theoretical upper limit of predictability (Πmax) based on ordinal patterns, using normalized permutation entropy. Values close to 1 indicate highly regular ordinal patterns, values near 0 indicate random ordinal structure.  
  **When to use:** To estimate the theoretical predictability ceiling based on directional patterns only, serving as a benchmark for forecasting performance regardless of magnitude.

- **`permutation_auto_mutual_information` (PAMI)**  
  Measures the information dependency between a time series and itself delayed by a lag, using ordinal patterns. It quantifies how much information the ordinal patterns at time t provide about the ordinal patterns at time t+tau.  
  **When to use:** To detect non-linear predictive relationships and temporal dependencies in time series. Higher values indicate stronger temporal dependencies between ordinal patterns; values near zero suggest independence.

- **`relative_absolute_error` (RAE)**  
  Calculates the Relative Absolute Error between a model prediction and a baseline. Defined as MAE_model / MAE_baseline, providing a normalized error measure.  
  **When to use:** To assess how much better a forecasting model performs compared to a simple baseline. RAE < 1 indicates model improvement over baseline; RAE > 1 indicates worse performance. Useful for cross-series comparison with different scales.

### 3. Trend & Memory

- **`hurst_exponent`**  
  Estimates the Hurst exponent and p-value for the random walk hypothesis. The Hurst exponent measures trend persistence or long-term memory in a time series.  
  **When to use:**  
    - H < 0.5: anti-persistent series (tends to mean-revert)
    - H ≈ 0.5: random walk
    - H > 0.5: persistent series (long-term trend)  
  The p-value indicates whether the random walk hypothesis can be rejected.

- **`relative_strength_index` (RSI)**  
  Momentum oscillator, useful for identifying overbought/oversold zones and trend strength.

### 4. Forecast Stability Metrics

- **`macv`**  
  Calculates Mean Absolute Change Vertical (MAC(V)): measures forecast stability between consecutive forecast origins.  
  **When to use:** To quantify how much forecasts change between consecutive updates for the same time periods.

- **`mach`**  
  Calculates Mean Absolute Change Horizontal (MAC(H)): measures the smoothness of forecasts within a single prediction horizon.  
  **When to use:** To assess the horizontal stability and smoothness of forecast curves.

- **`mascv`**  
  Calculates Mean Absolute Scaled Change Vertical (MASC(V)): scaled version of MAC(V) normalized by seasonal variation.  
  **When to use:** To compare forecast stability across different series or scales, accounting for natural seasonal variation.

- **`masch`**  
  Calculates Mean Absolute Scaled Change Horizontal (MASC(H)): scaled version of MAC(H) normalized by seasonal variation.  
  **When to use:** To assess horizontal forecast stability in a scale-independent manner.

- **`rmsscv`**  
  Calculates Root Mean Squared Scaled Change Vertical (RMSSC(V)): RMS version of scaled vertical stability.  
  **When to use:** When you want to penalize larger forecast revisions more heavily than smaller ones.

- **`rmssch`**  
  Calculates Root Mean Squared Scaled Change Horizontal (RMSSC(H)): RMS version of scaled horizontal stability.  
  **When to use:** For horizontal stability assessment with heavier penalization of large variations.

### 5. Forecast Stabilization

- **`vi` (Vertical Interpolation)**  
  Calculates stable forecasts for specific target time points by linearly combining the latest original forecast with anchor values from the previous origin.  
  **When to use:** To stabilize forecasts vertically across different forecast origins for the same target time. Supports both Partial VI (PVI) using original forecasts as anchors, and Full VI (FVI) using stabilized forecasts as anchors.

- **`hpi` (Horizontal Partial Interpolation)**  
  Stabilizes forecasts by combining the original forecast of the current horizon with the original forecast of the previous horizon.  
  **When to use:** To reduce variability between consecutive forecast horizons and create smoother forecast trajectories while preserving original forecast dynamics.

- **`hfi` (Horizontal Full Interpolation)**  
  Blends the stable forecast from the previous horizon with the original forecast of the current horizon to create smooth forecast curves.  
  **When to use:** To achieve maximum smoothness in forecast trajectories by using previously stabilized values, creating more conservative and stable forecasts across horizons.


## Summary: Function Quick Reference

### Forecastability & Complexity
| Metric/Function                        | Range         | Interpretation                                             | Question You Want to Answer                                         |
|----------------------------------------|---------------|------------------------------------------------------------|--------------------------------------------------------------------|
| **Forecastability (ForeCA)**           | 0 → 1         | Forecastability (1 = highly predictable, 0 = noise)       | "How predictable is this time series?"                             |
| **ADI / CV**                           | ADI: 1 → ∞    | ADI: Intermittency; CV: Variability                        | "Is this series intermittent or erratic?"                          |
| **Sample Entropy (SampEn)**            | 0 → ∞         | Complexity/regularity (low = more regular, high = more complex) | "How complex or irregular is this time series?"                |
| **Permutation Entropy**                | 0 → ∞         | Ordinal complexity/randomness (low = more regular, high = more complex) | "How random or complex is the order of this time series?"     |
| **Regularity Index**                   | 0 → 1         | Temporal regularity (1 = highly regular, 0 = high variability) | "How consistent and regular are the values over time?"            |
| **Theoretical Limit**                  | 0 → 1         | Theoretical predictability ceiling based on ordinal patterns | "What is the maximum predictability based on directional patterns?" |
| **Permutation Auto Mutual Information (PAMI)** | 0 → 1     | Measures how much the series' recent behavior helps predict what comes next  | "How much does the series' recent pattern tell us about upcoming values?" |
| **Relative Absolute Error (RAE)**       | 0 → ∞         | Normalized forecast error vs baseline (RAE < 1 = improvement) | "How much better is the model compared to a simple baseline?" |

### Trend & Memory
| Metric/Function                        | Range         | Interpretation                                             | Question You Want to Answer                                         |
|----------------------------------------|---------------|------------------------------------------------------------|--------------------------------------------------------------------|
| **Hurst Exponent**                     | 0 → 1         | Trend persistence/long-term memory                         | "Does the series have persistent trend or mean-revert?"            |
| **Relative Strength Index (RSI)**      | 0 → 100       | Relative strength/momentum index                           | "Is the series overbought or oversold?"                            |

### Forecast Stability Metrics
| Metric/Function                        | Range         | Interpretation                                             | Question You Want to Answer                                         |
|----------------------------------------|---------------|------------------------------------------------------------|--------------------------------------------------------------------|
| **MAC(V)**                             | 0 → ∞         | Vertical forecast stability (lower = more stable)          | "How much do my forecasts change between updates?"                  |
| **MAC(H)**                             | 0 → ∞         | Horizontal forecast stability (lower = smoother)           | "How smooth are my forecast curves?"                                |
| **MASC(V)**                            | 0 → ∞         | Scaled vertical stability (lower = more stable)            | "How stable are forecasts relative to natural variation?"           |
| **MASC(H)**                            | 0 → ∞         | Scaled horizontal stability (lower = smoother)             | "How smooth are forecasts relative to seasonal patterns?"           |
| **RMSSC(V)**                           | 0 → ∞         | RMS scaled vertical stability (penalizes large changes)    | "How stable are forecasts with emphasis on large revisions?"        |
| **RMSSC(H)**                           | 0 → ∞         | RMS scaled horizontal stability (penalizes large changes)  | "How smooth are forecasts with emphasis on large variations?"       |


### Forecast Stabilization
| Metric/Function                        | Range         | Interpretation                                             | Question You Want to Answer                                         |
|----------------------------------------|---------------|------------------------------------------------------------|--------------------------------------------------------------------|
| **Vertical Interpolation (VI)**        | Depends on data | Stabilized forecast for target time points                 | "How can I stabilize forecasts across different origins?"          |
| **Horizontal Partial Interpolation (HPI)** | Depends on data | Smoothed forecasts using original previous horizons        | "How can I smooth forecasts while preserving dynamics?"            |
| **Horizontal Full Interpolation (HFI)** | Depends on data | Fully stabilized forecasts using stable previous horizons | "How can I create maximally smooth and stable forecast curves?"    |


### Outlier Detection & Stats
| Metric/Function                        | Range         | Interpretation                                             | Question You Want to Answer                                         |
|----------------------------------------|---------------|------------------------------------------------------------|--------------------------------------------------------------------|
| **Hampel Filter**                      | 0 or 1        | Outlier presence (per point)                               | "Is this point an outlier relative to its local window?"           |
| **Bollinger Bands**                    | 0 or 1        | Signals breakouts and volatility regimes                   | "Is the value outside the expected volatility range?"              |