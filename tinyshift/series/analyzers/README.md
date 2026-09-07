# Time Series Analyzers

The `tinyshift.series.analyzers` package contains panel-oriented analyzers for
profiling multiple time series with a shared `fit()` and `summary()` lifecycle.

## Input Contract

Analyzers expect a long-format pandas DataFrame with one row per observation:

| Column | Meaning | Default |
|---|---|---|
| `unique_id` | Series identifier | `unique_id` |
| `ds` | Observation time or ordering key | `ds` |
| `y` | Numeric target value | `y` |

The base class validates the panel, rejects missing identifiers or timestamps,
rejects duplicate ID-time pairs, sorts each series by time, and fits each ID
independently. Custom column names are supported through `fit()`.

```python
from tinyshift.series import SeasonalityAnalyzer

analyzer = SeasonalityAnalyzer(top_k=2)
analyzer.fit(
    df,
    id_col="unique_id",
    time_col="ds",
    target_col="y",
)
summary = analyzer.summary()
```

Every analyzer returns `self` from `fit()`. Results are retained in
`results_`, while `summary()` returns a compact DataFrame for downstream joins
or inspection. The exact result structure is analyzer-specific.

## Available Analyzers

### `IntermittencyAnalyzer`

Classifies demand profiles as smooth, intermittent, erratic, or lumpy. The
summary includes average demand interval, positive-demand variability, zero
proportion, and interval irregularity.

```python
from tinyshift.series import IntermittencyAnalyzer

summary = IntermittencyAnalyzer().fit(df).summary()
```

### `PredictabilityAnalyzer`

Reports complementary structure measures: ForeCA forecastability, the ordinal
predictability limit, and normalized spectral concentration. These describe
structure in the observed data; they are not out-of-sample forecast scores.

```python
from tinyshift.series import PredictabilityAnalyzer

summary = PredictabilityAnalyzer(detrend="linear").fit(df).summary()
```

### `PAMIAnalyzer`

Computes permutation auto-mutual information for each series and records local
minima as candidate nonlinear-dependence lags. Use `lags()` to convert the
minima into DTL/DMSTL-compatible lag dictionaries.

```python
from tinyshift.series import PAMIAnalyzer

analyzer = PAMIAnalyzer(max_tau=60).fit(df)
summary = analyzer.summary()
lags = analyzer.lags(mode="short", fallback=1, short=3)
```

### `SeasonalityAnalyzer`

Detects candidate seasonal periods from spectral peaks and tests their harmonic
significance. It retains spectral details and candidate/significant periods in
`results_`.

```python
from tinyshift.series import SeasonalityAnalyzer

summary = SeasonalityAnalyzer(top_k=2).fit(df).summary()
```

### `TrendAnalyzer`

Fits a linear trend independently for every series and reports slope, R²,
p-value, and a significance flag.

```python
from tinyshift.series import TrendAnalyzer

summary = TrendAnalyzer(significance_level=0.05).fit(df).summary()
```

### `VarianceRatioAnalyzer`

Evaluates persistence and mean reversion over one or more aggregation
horizons. Unlike the other summaries, it can return multiple rows per series,
one for each tested horizon.

```python
from tinyshift.series import VarianceRatioAnalyzer

summary = VarianceRatioAnalyzer(horizons=[2, 4, 8]).fit(df).summary()
```

## Combining Profiles

The one-row-per-series analyzers can be combined with validated one-to-one
merges. Keep `VarianceRatioAnalyzer` separate or aggregate it first because it
may return multiple rows per `unique_id`.

```python
from tinyshift.series import (
    IntermittencyAnalyzer,
    PredictabilityAnalyzer,
    SeasonalityAnalyzer,
    TrendAnalyzer,
)

analyzers = [
    IntermittencyAnalyzer(),
    PredictabilityAnalyzer(),
    SeasonalityAnalyzer(top_k=2),
    TrendAnalyzer(),
]

summaries = [analyzer.fit(df).summary() for analyzer in analyzers]
profile = summaries[0]
for section in summaries[1:]:
    profile = profile.merge(section, on="unique_id", validate="one_to_one")
```

For single-vector MSTL decomposition diagnostics, use
[`MSTLDiagnostics`](../../plot/mstl.py). It is separate from these panel
analyzers and does not process `unique_id` columns.
