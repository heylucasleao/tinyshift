# Feature Engineering (`tinyshift.features`)

Lightweight feature-engineering functions that do not own a model lifecycle.
The current API focuses on time-series inputs.

## Public API

- `relative_strength_index` computes momentum for a univariate series.
- `standardize_returns` computes log or simple returns and optionally
  standardizes them.
- `fourier_seasonality` adds sine/cosine encodings for calendar cycles.
- `estimate_history_length` estimates a lag history from seasonality and the
  forecast horizon.

```python
from tinyshift.features import (
    estimate_history_length,
    fourier_seasonality,
    relative_strength_index,
    standardize_returns,
)

rsi = relative_strength_index(series, rolling_window=14)
returns = standardize_returns(series, log=True)
df = fourier_seasonality(df, time_col="ds", seasonality=["weekly", "yearly"])
history = estimate_history_length(seasonal_period=7, horizon=14)
```
