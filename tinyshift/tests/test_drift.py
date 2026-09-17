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
)


class TestConDrift:
    def test_permutation_and_structured_result(self):
        detector = ConDrift(n_resamples=99, random_state=7).fit(np.linspace(0, 1, 40))
        result = detector.predict(np.linspace(10, 11, 20))
        assert isinstance(result, DriftResult)
        assert result.score > result.threshold
        assert result.p_value == pytest.approx(0.01)
        assert result.drift is True
        assert (result.reference_size, result.current_size) == (40, 20)
        assert detector.score(np.linspace(10, 11, 20)) == pytest.approx(result.score)

    def test_normalization_is_scale_independent(self):
        small = ConDrift().fit([0.0, 1.0, 2.0])
        large = ConDrift().fit([0.0, 10.0, 20.0])
        assert small.score([1.0, 2.0, 3.0]) == pytest.approx(
            large.score([10.0, 20.0, 30.0])
        )

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
        detector = ConDrift(random_state=4)
        assert clone(detector).get_params() == detector.get_params()


class TestCatDrift:
    def test_jensen_shannon_distance(self):
        detector = CatDrift().fit(["a", "a", "b", "b"])
        assert detector.score(["a", "b", "b", "b"]) > 0

    def test_unseen_categories_contribute_to_distance(self):
        detector = CatDrift(min_current_size=1).fit(["known", "known"])
        assert detector.score(["new"]) == pytest.approx(1.0)

    def test_missing_values_are_rejected(self):
        with pytest.raises(ValueError, match="missing"):
            CatDrift().fit(["a", None])

    def test_permutation_supports_string_categories(self):
        detector = CatDrift(n_resamples=20, random_state=1).fit(
            ["a", "a", "b", "b", "c"]
        )
        result = detector.predict(["a", "b"])
        assert np.isfinite(result.threshold)
        assert 0 < result.p_value <= 1
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
        reference = _panel(np.linspace(0, 1, 40), np.linspace(100, 101, 40))
        current = _panel(np.linspace(10, 11, 20), np.linspace(100, 101, 20))
        analyzer = ContinuousDriftAnalyzer(
            ConDrift(n_resamples=99, random_state=7)
        ).fit(reference, id_col="entity", target_col="value")
        result = analyzer.predict(current)
        assert list(result.columns) == [
            "entity",
            "score",
            "threshold",
            "p_value",
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
            CatDrift(n_resamples=99, random_state=7)
        ).fit(_panel(["x"] * 40, ["z"] * 40), "entity", "value")
        result = analyzer.predict(_panel(["y"] * 20, ["z"] * 20))
        assert len(result) == 2
        assert result.set_index("entity").loc["A", "drift"]
        assert not result.set_index("entity").loc["B", "drift"]

    def test_unknown_current_id_is_rejected(self):
        analyzer = ContinuousDriftAnalyzer(ConDrift()).fit(
            _panel([1.0, 2.0], [3.0, 4.0]), "entity", "value"
        )
        unknown = pd.DataFrame({"entity": ["C", "C"], "value": [1.0, 2.0]})
        with pytest.raises(ValueError, match="No reference distribution"):
            analyzer.predict(unknown)

    def test_missing_reference_ids_do_not_appear_in_current_result(self):
        analyzer = ContinuousDriftAnalyzer(ConDrift()).fit(
            _panel([1.0, 2.0], [3.0, 4.0]), "entity", "value"
        )
        current = pd.DataFrame({"entity": ["A", "A"], "value": [1.0, 2.0]})
        assert analyzer.predict(current)["entity"].tolist() == ["A"]
