import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from tinyshift.drift import (
    CatDrift,
    CategoricalDriftAnalyzer,
    ConDrift,
    ContinuousDriftAnalyzer,
    DriftResult,
    chebyshev,
    psi,
)


class TestConDrift:
    def test_manual_threshold_and_structured_result(self):
        detector = ConDrift(threshold=0.5, normalize=False).fit([0.0, 0.0, 1.0, 1.0])
        result = detector.predict([10.0, 11.0])
        assert isinstance(result, DriftResult)
        assert result.score > result.threshold
        assert result.drift is True
        assert (result.reference_size, result.current_size) == (4, 2)
        assert detector.score([10.0, 11.0]) == pytest.approx(result.score)

    def test_no_threshold_returns_score_without_classification(self):
        result = ConDrift(threshold=None).fit([0.0, 1.0]).predict([0.0, 1.0])
        assert result.score == pytest.approx(0.0)
        assert result.threshold is None
        assert result.drift is None

    def test_normalization_is_scale_independent(self):
        small = ConDrift(threshold=None).fit([0.0, 1.0, 2.0])
        large = ConDrift(threshold=None).fit([0.0, 10.0, 20.0])
        assert small.score([1.0, 2.0, 3.0]) == pytest.approx(
            large.score([10.0, 20.0, 30.0])
        )

    def test_bootstrap_is_reproducible_and_cached_by_current_size(self):
        detector = ConDrift(n_resamples=30, random_state=7, min_current_size=1).fit(
            np.arange(10.0)
        )
        first = detector.predict([1.0, 2.0, 3.0])
        second = detector.predict([7.0, 8.0, 9.0])
        other_size = detector.predict([1.0, 2.0])
        assert first.threshold == second.threshold == detector.thresholds_[3]
        assert other_size.threshold == detector.thresholds_[2]
        assert set(detector.thresholds_) == {2, 3}

    @pytest.mark.parametrize("values", [["bad", "data"], [1.0, np.inf]])
    def test_invalid_continuous_values_are_rejected(self, values):
        with pytest.raises(ValueError, match="numeric|finite"):
            ConDrift().fit(values)

    def test_lifecycle_and_minimum_sizes_are_validated(self):
        with pytest.raises(NotFittedError):
            ConDrift().score([1.0, 2.0])
        with pytest.raises(ValueError, match="at least 2"):
            ConDrift().fit([1.0])
        with pytest.raises(ValueError, match="at least 2"):
            ConDrift().fit([1.0, 2.0]).score([1.0])

    def test_estimator_clone_preserves_configuration(self):
        detector = ConDrift(
            metric="wasserstein", threshold=0.3, normalize=False, random_state=4
        )
        assert clone(detector).get_params() == detector.get_params()


class TestCatDrift:
    def test_metric_helpers(self):
        assert chebyshev(np.array([0.2, 0.8]), np.array([0.3, 0.7])) == pytest.approx(
            0.1
        )
        assert psi(np.array([0.5, 0.5]), np.array([0.4, 0.6])) > 0

    @pytest.mark.parametrize("metric", ["chebyshev", "jensen_shannon", "psi"])
    def test_supported_metrics(self, metric):
        detector = CatDrift(metric=metric, threshold=None).fit(["a", "a", "b", "b"])
        assert detector.score(["a", "b", "b", "b"]) >= 0

    def test_unseen_categories_contribute_to_distance(self):
        detector = CatDrift(metric="chebyshev", threshold=0.5, min_current_size=1).fit(
            ["known", "known"]
        )
        result = detector.predict(["new"])
        assert result.score == pytest.approx(1.0)
        assert result.drift is True

    def test_missing_values_are_rejected(self):
        with pytest.raises(ValueError, match="missing"):
            CatDrift().fit(["a", None])

    def test_bootstrap_supports_string_categories(self):
        detector = CatDrift(n_resamples=20, random_state=1).fit(
            ["a", "a", "b", "b", "c"]
        )
        result = detector.predict(["a", "b"])
        assert np.isfinite(result.threshold)
        assert isinstance(result.drift, bool)


def _panel(a_values, b_values):
    return pd.DataFrame(
        {
            "entity": ["A"] * len(a_values) + ["B"] * len(b_values),
            "value": list(a_values) + list(b_values),
        }
    )


class TestDriftAnalyzers:
    def test_continuous_analyzer_fits_independent_detectors(self):
        reference = _panel([0.0, 0.1, 0.2], [100.0, 101.0, 102.0])
        current = _panel([10.0, 11.0], [100.0, 101.0])
        analyzer = ContinuousDriftAnalyzer(ConDrift(threshold=2.0, normalize=True)).fit(
            reference, id_col="entity", target_col="value"
        )
        result = analyzer.predict(current)
        assert list(result.columns) == [
            "entity",
            "score",
            "threshold",
            "drift",
            "reference_size",
            "current_size",
        ]
        assert list(result["entity"]) == ["A", "B"]
        assert bool(result.loc[result["entity"] == "A", "drift"].iloc[0])
        assert not bool(result.loc[result["entity"] == "B", "drift"].iloc[0])
        pd.testing.assert_frame_equal(analyzer.summary(), result)
        assert analyzer.detectors_["A"] is not analyzer.detectors_["B"]

    def test_categorical_analyzer_uses_one_result_per_id(self):
        analyzer = CategoricalDriftAnalyzer(
            CatDrift(metric="chebyshev", threshold=0.4)
        ).fit(_panel(["x", "x"], ["z", "z"]), "entity", "value")
        result = analyzer.predict(_panel(["y", "y"], ["z", "z"]))
        assert len(result) == 2
        assert result.set_index("entity").loc["A", "drift"]
        assert not result.set_index("entity").loc["B", "drift"]

    def test_unknown_current_id_is_rejected(self):
        analyzer = ContinuousDriftAnalyzer(ConDrift(threshold=1.0)).fit(
            _panel([1.0, 2.0], [3.0, 4.0]), "entity", "value"
        )
        unknown = pd.DataFrame({"entity": ["C", "C"], "value": [1.0, 2.0]})
        with pytest.raises(ValueError, match="No reference distribution"):
            analyzer.predict(unknown)

    def test_missing_reference_ids_do_not_appear_in_current_result(self):
        analyzer = ContinuousDriftAnalyzer(ConDrift(threshold=1.0)).fit(
            _panel([1.0, 2.0], [3.0, 4.0]), "entity", "value"
        )
        current = pd.DataFrame({"entity": ["A", "A"], "value": [1.0, 2.0]})
        assert analyzer.predict(current)["entity"].tolist() == ["A"]
