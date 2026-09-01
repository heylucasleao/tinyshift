# Forecasting (`tinyshift.forecasting`)

Forecasting estimators, predictive distributions, evaluation helpers and
forecast-driven decision policies.

## Packages

- `dtl` separates a LOWESS trend from residual dynamics for non-seasonal panel
  forecasting.
- `dmstl` separates trend, multiple seasonal components and residual dynamics.
- `probabilistic` calibrates parametric predictive distributions around an
  `MLForecast` point forecast and includes Newsvendor decisions.

```python
from tinyshift.forecasting import (
    DMSTLWrapper,
    DTLWrapper,
    GammaFamily,
    NewsvendorOptimizer,
    TwoStageForecasterWrapper,
)
```

Each subpackage has its own README with its fitting flow, extension rules and
examples. The wrappers require the `series` optional dependency set.
