# Forecasting (`tinyshift.forecasting`)

Forecasting estimators, predictive distributions, evaluation helpers and
forecast-driven decision policies.

## Packages

- `dtl` separates a LOWESS trend from residual dynamics for non-seasonal panel
  forecasting.
- `dmstl` separates trend, multiple seasonal components and residual dynamics.
- `probabilistic` calibrates parametric predictive distributions around an
  `MLForecast` point forecast and includes Newsvendor decisions.
- `metrics` evaluates forecasts through accuracy, bias, stability, economic
  loss, and tail-risk measures.
- `stabilization` provides vertical and horizontal forecast interpolation.

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

## Forecast Metrics

The public API includes `wape`, `pbias`, `score`, `rmae`,
`forecast_instability`, `economic_loss`, and `tail_risk`:

```python
from tinyshift.forecasting import economic_loss, tail_risk, wape

accuracy = wape(df, models=["forecast"])
loss = economic_loss(
    df,
    models=["forecast"],
    underage_cost="cu",
    overage_cost="co",
)
risk = tail_risk(
    df,
    models=["forecast"],
    underage_cost="cu",
    overage_cost="co",
)
```

## Forecast Stabilization

`vi` combines forecasts for the same target from consecutive origins. `hpi`
and `hfi` smooth consecutive horizons using original or recursively stabilized
anchors, respectively:

```python
from tinyshift.forecasting import hfi, hpi, vi

vertical = vi(y_hat, anchor, w_s=0.3)
partial = hpi(y_hat, w_s=0.4)
full = hfi(y_hat, w_s=0.5)
```
