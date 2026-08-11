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
