"""Compatibility facade for the former :mod:`tinyshift.modelling` API.

New code should import from ``tinyshift.preprocessing``, ``tinyshift.features``
or ``tinyshift.forecasting``.
"""

import importlib
import sys

from tinyshift import features, forecasting, preprocessing
from tinyshift.features import *
from tinyshift.forecasting import *
from tinyshift.preprocessing import *

multicollinearity = preprocessing.multicollinearity
residualizer = preprocessing.residualizer
scaler = preprocessing.scaler
ts_features = features.time_series
dtl = forecasting.dtl
dmstl = forecasting.dmstl
tsf = forecasting.probabilistic

_ALIASES = {
    "multicollinearity": multicollinearity,
    "residualizer": residualizer,
    "scaler": scaler,
    "ts_features": ts_features,
    "dtl": dtl,
    "dmstl": dmstl,
    "tsf": tsf,
}
for _name, _module in _ALIASES.items():
    sys.modules[f"{__name__}.{_name}"] = _module

for _old, _new, _modules in (
    ("dtl", "tinyshift.forecasting.dtl", ("base", "global_", "local_", "wrapper")),
    ("dmstl", "tinyshift.forecasting.dmstl", ("base", "global_", "local_", "wrapper")),
    ("tsf", "tinyshift.forecasting.probabilistic", ("calibration", "decision", "distribution", "eval", "family", "forecast", "wrapper")),
):
    for _module_name in _modules:
        _module = importlib.import_module(f"{_new}.{_module_name}")
        sys.modules[f"{__name__}.{_old}.{_module_name}"] = _module

__all__ = preprocessing.__all__ + features.__all__ + forecasting.__all__
