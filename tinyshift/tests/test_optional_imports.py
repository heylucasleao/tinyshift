# Copyright (c) 2024-2026 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License


from unittest.mock import patch

import pandas as pd
import pytest

from tinyshift.forecasting.dmstl import DMSTLWrapper
from tinyshift.plot.calibration import efficiency_curve
from tinyshift.plot.correlation import corr_heatmap
from tinyshift.plot.diagnostic import seasonal_decompose, stationarity_analysis
from tinyshift.plot.power import power_curve


def test_optional_dependency_guards_raise_the_expected_error():
    with patch("tinyshift.utils.imports.importlib.util.find_spec", return_value=None):
        for func, args in [
            (efficiency_curve, (None, [[0]], [0])),
            (corr_heatmap, ([[0, 1], [1, 0]],)),
            (seasonal_decompose, (pd.Series([1, 2, 3, 4]), [2])),
            (stationarity_analysis, (pd.Series([1, 2, 3, 4]),)),
            (power_curve, (0.5,)),
        ]:
            with pytest.raises(ImportError, match=r"tinyshift\[plot\]"):
                func(*args)


def test_series_optional_dependency_guard_uses_the_series_extra():
    with patch("tinyshift.utils.imports.importlib.util.find_spec", return_value=None):
        with pytest.raises(ImportError, match=r"tinyshift\[series\]"):
            DMSTLWrapper.fit(None, None)
