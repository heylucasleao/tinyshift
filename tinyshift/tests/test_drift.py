# Copyright (c) 2024-2026 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License


import numpy as np
import pandas as pd
import pytest

from tinyshift.drift.categorical import CatDrift, chebyshev, psi
from tinyshift.drift.continuous import ConDrift


class TestCatDrift:
    def test_chebyshev_distance(self):
        assert chebyshev(np.array([0.2, 0.8]), np.array([0.3, 0.7])) == pytest.approx(
            0.1
        )

    def test_psi(self):
        assert psi(np.array([0.5, 0.5]), np.array([0.4, 0.6])) > 0

    def test_fit_and_predict(self):
        df = pd.DataFrame(
            {
                "unique_id": ["A"] * 6 + ["B"] * 6,
                "ds": pd.date_range("2024-01-01", periods=12, freq="D"),
                "y": ["x", "x", "y", "y", "x", "x", "y", "y", "x", "x", "y", "y"],
            }
        )

        model = CatDrift(
            freq="D", func="chebyshev", drift_limit="auto", method="expanding"
        )
        fitted = model.fit(df)

        assert fitted.reference_distribution is not None
        assert fitted.thresholds

        predictions = model.predict(df)
        assert "drift" in predictions.columns
        assert predictions["drift"].dtype == bool

    @pytest.mark.parametrize("func_name", ["chebyshev", "jensenshannon", "psi"])
    def test_fit_and_predict_supports_other_categorical_metrics(self, func_name):
        df = pd.DataFrame(
            {
                "unique_id": ["A"] * 6 + ["B"] * 6,
                "ds": pd.date_range("2024-01-01", periods=12, freq="D"),
                "y": ["x", "x", "y", "y", "x", "x", "y", "y", "x", "x", "y", "y"],
            }
        )

        model = CatDrift(
            freq="D",
            func=func_name,
            drift_limit="auto",
            method="expanding",
        )
        fitted = model.fit(df)

        predictions = model.predict(df)

        assert fitted.reference_distribution is not None
        assert np.isfinite(predictions["metric"]).all()
        assert "drift" in predictions.columns
        assert predictions["drift"].dtype == bool

    def test_jackknife_method_works(self):
        df = pd.DataFrame(
            {
                "unique_id": ["A"] * 6 + ["B"] * 6,
                "ds": pd.date_range("2024-01-01", periods=12, freq="D"),
                "y": ["x", "x", "y", "y", "x", "x", "y", "y", "x", "x", "y", "y"],
            }
        )

        model = CatDrift(
            freq="D",
            func="chebyshev",
            drift_limit="auto",
            method="jackknife",
        )
        fitted = model.fit(df)
        predictions = model.predict(df)

        assert fitted.reference_distribution is not None
        assert np.isfinite(predictions["metric"]).all()
        assert predictions["drift"].dtype == bool

    def test_invalid_method_and_function_raise(self):
        with pytest.raises(ValueError):
            CatDrift(freq="D", method="invalid")

        with pytest.raises(ValueError):
            CatDrift(freq="D", func="invalid")


class TestConDrift:
    def test_fit_and_predict(self):
        df = pd.DataFrame(
            {
                "unique_id": ["A"] * 6 + ["B"] * 6,
                "ds": pd.date_range("2024-01-01", periods=12, freq="D"),
                "y": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5],
            }
        )

        model = ConDrift(freq="D", func="ws", drift_limit="auto", method="expanding")
        fitted = model.fit(df)

        assert fitted.reference_distribution is not None
        assert fitted.thresholds

        predictions = model.predict(df)
        assert "drift" in predictions.columns
        assert predictions["drift"].dtype == bool

    def test_jackknife_method_works(self):
        df = pd.DataFrame(
            {
                "unique_id": ["A"] * 6 + ["B"] * 6,
                "ds": pd.date_range("2024-01-01", periods=12, freq="D"),
                "y": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5],
            }
        )

        model = ConDrift(freq="D", func="ws", drift_limit="auto", method="jackknife")
        fitted = model.fit(df)
        predictions = model.predict(df)

        assert fitted.reference_distribution is not None
        assert np.isfinite(predictions["metric"]).all()
        assert predictions["drift"].dtype == bool
