# Copyright (c) 2024-2026 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License


import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from tinyshift.drift.categorical import CatDrift, chebyshev, psi
from tinyshift.drift.continuous import ConDrift


def _categorical_panel(series_values):
    rows = []
    for unique_id, periods in series_values.items():
        for day, values in enumerate(periods):
            for value in values:
                rows.append(
                    (
                        unique_id,
                        pd.Timestamp("2024-01-01") + pd.DateOffset(days=day),
                        value,
                    )
                )
    return pd.DataFrame(rows, columns=["unique_id", "ds", "y"])


def _continuous_panel(series_values):
    rows = []
    for unique_id, periods in series_values.items():
        for day, values in enumerate(periods):
            for value in values:
                rows.append(
                    (
                        unique_id,
                        pd.Timestamp("2024-01-01") + pd.DateOffset(days=day),
                        value,
                    )
                )
    return pd.DataFrame(rows, columns=["unique_id", "ds", "y"])


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

    def test_reference_distributions_are_normalized_per_series(self):
        df = _categorical_panel(
            {"A": [["x", "x"], ["x", "y"]], "B": [["z", "z"], ["z", "y"]]}
        )
        model = CatDrift(freq="D", drift_limit=(None, 1.0)).fit(df)

        assert sum(model.reference_distribution["A"].values()) == pytest.approx(1.0)
        assert sum(model.reference_distribution["B"].values()) == pytest.approx(1.0)

    def test_threshold_calibration_is_isolated_by_series(self):
        a_periods = [["x", "x"], ["x", "y"], ["x", "x"], ["y", "y"]]
        b_periods = [["z", "z"], ["w", "w"], ["z", "w"], ["w", "w"]]
        only_a = CatDrift(freq="D", drift_limit="mad").fit(
            _categorical_panel({"A": a_periods})
        )
        with_b = CatDrift(freq="D", drift_limit="mad").fit(
            _categorical_panel({"A": a_periods, "B": b_periods})
        )

        assert with_b.thresholds["A"] == pytest.approx(only_a.thresholds["A"])

    def test_unseen_category_is_included_in_distance(self):
        reference = _categorical_panel({"A": [["x"], ["x"], ["x"]]})
        analysis = _categorical_panel({"A": [["new"]]})
        model = CatDrift(freq="D", func="chebyshev", drift_limit=(None, 0.5)).fit(
            reference
        )

        result = model.predict(analysis)

        assert result.loc[0, "metric"] == pytest.approx(1.0)
        assert bool(result.loc[0, "drift"])

    def test_metric_equal_to_threshold_is_not_drift(self):
        reference = _categorical_panel({"A": [["x"], ["x"], ["x"]]})
        model = CatDrift(freq="D", drift_limit=(None, 0.0)).fit(reference)

        result = model.predict(_categorical_panel({"A": [["x"]]}))

        assert result.loc[0, "metric"] == 0.0
        assert not bool(result.loc[0, "drift"])

    def test_unknown_series_and_score_before_fit_are_rejected(self):
        reference = _categorical_panel({"A": [["x"], ["x"]]})
        unknown = _categorical_panel({"B": [["x"]]})
        model = CatDrift(freq="D", drift_limit=(None, 1.0))

        with pytest.raises(ValueError, match="fitted"):
            model.score(reference)
        model.fit(reference)
        with pytest.raises(ValueError, match="No reference distribution"):
            model.score(unknown)

    def test_estimator_clone_preserves_function_name(self):
        model = CatDrift(freq="D", func="psi")

        assert clone(model).func == "psi"


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

    def test_threshold_calibration_is_isolated_by_series(self):
        a_periods = [[0.0, 0.1], [0.2, 0.3], [0.0, 0.2], [0.4, 0.5]]
        b_periods = [[100.0], [-100.0], [200.0], [-200.0]]
        only_a = ConDrift(freq="D", drift_limit="mad").fit(
            _continuous_panel({"A": a_periods})
        )
        with_b = ConDrift(freq="D", drift_limit="mad").fit(
            _continuous_panel({"A": a_periods, "B": b_periods})
        )

        assert with_b.thresholds["A"] == pytest.approx(only_a.thresholds["A"])

    @pytest.mark.parametrize("values", [["bad", "data"], [1.0, np.inf]])
    def test_non_finite_or_non_numeric_target_is_rejected(self, values):
        df = pd.DataFrame(
            {
                "unique_id": ["A", "A"],
                "ds": pd.date_range("2024-01-01", periods=2, freq="D"),
                "y": values,
            }
        )

        with pytest.raises(ValueError, match="numeric|finite"):
            ConDrift(freq="D").fit(df)

    @pytest.mark.parametrize("method", ["expanding", "jackknife"])
    def test_single_reference_period_is_rejected(self, method):
        df = _continuous_panel({"A": [[1.0, 2.0]]})

        with pytest.raises(ValueError, match="at least two reference periods"):
            ConDrift(freq="D", method=method).fit(df)

    def test_estimator_clone_preserves_function_name(self):
        model = ConDrift(freq="D", func="ws")

        assert clone(model).func == "ws"
