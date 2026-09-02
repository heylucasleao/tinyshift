# Copyright (c) 2024-2026 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License


import pickle

import pytest
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from tinyshift.association_mining.analyzer import TransactionAnalyzer
from tinyshift.association_mining.encoder import TransactionEncoder


class TestTransactionEncoder:
    def test_fit_transform_creates_expected_encoding(self):
        transactions = [["apple", "beer"], ["apple", "milk"], ["beer", "milk"]]

        encoder = TransactionEncoder()
        encoded = encoder.fit_transform(transactions)

        assert encoded.shape == (3, 3)
        assert encoded.dtype == bool
        assert encoded[0, 0] and encoded[0, 1]

    def test_inverse_transform_round_trips(self):
        transactions = [["apple", "beer"], ["apple", "milk"]]

        encoder = TransactionEncoder()
        encoded = encoder.fit_transform(transactions)
        recovered = encoder.inverse_transform(encoded)

        assert recovered[0] == ["apple", "beer"]
        assert recovered[1] == ["apple", "milk"]


class TestTransactionAnalyzer:
    def test_fit_transform_and_metrics(self):
        transactions = [
            ["apple", "beer"],
            ["apple", "milk"],
            ["beer", "milk"],
            ["apple", "beer", "milk"],
        ]

        analyzer = TransactionAnalyzer()
        analyzer.fit(transactions)

        assert analyzer.transactions_ is not None
        assert analyzer.columns_ is not None

        assert analyzer.lift("apple", "beer") >= 0
        assert 0 <= analyzer.confidence("apple", "beer") <= 1
        assert 0 <= analyzer.kulczynski("apple", "beer") <= 1
        assert 0 <= analyzer.sorensen_dice("apple", "beer") <= 1
        assert -1 <= analyzer.yules_q("apple", "beer") <= 1
        assert analyzer.hypergeom("apple", "beer") >= 0

    def test_correlation_matrix(self):
        transactions = [
            ["apple", "beer"],
            ["apple", "milk"],
            ["beer", "milk"],
            ["apple", "beer", "milk"],
        ]

        analyzer = TransactionAnalyzer()
        analyzer.fit(transactions)

        matrix = analyzer.correlation_matrix(
            ["apple"], ["beer", "milk"], metric="confidence"
        )

        assert matrix.shape == (1, 2)
        assert matrix.loc["apple", "beer"] >= 0
        assert matrix.loc["apple", "milk"] >= 0

    def test_unknown_metric_raises(self):
        analyzer = TransactionAnalyzer()
        analyzer.fit([["apple", "beer"]])

        with pytest.raises(ValueError):
            analyzer.correlation_matrix(["apple"], ["beer"], metric="unknown")

    def test_transform_before_fit_raises(self):
        analyzer = TransactionAnalyzer()

        with pytest.raises(ValueError, match="fitted"):
            analyzer.transform([["apple"]])

    def test_missing_item_raises_key_error(self):
        analyzer = TransactionAnalyzer()
        analyzer.fit([["apple", "beer"]])

        with pytest.raises(KeyError):
            analyzer.lift("apple", "milk")

    def test_metrics_match_known_contingency_table(self):
        analyzer = TransactionAnalyzer().fit(
            [["a", "b"], ["a", "b"], ["a"], ["x"]]
        )

        assert analyzer.lift("a", "b") == pytest.approx(4 / 3)
        assert analyzer.confidence("a", "b") == pytest.approx(2 / 3)
        assert analyzer.kulczynski("a", "b") == pytest.approx(5 / 6)
        assert analyzer.sorensen_dice("a", "b") == pytest.approx(0.8)
        assert analyzer.zhang_metric("a", "b") == pytest.approx(1.0)
        assert analyzer.hypergeom("a", "b") == pytest.approx(0.5)
        assert analyzer.yules_q("a", "b") == pytest.approx(1.0)

    def test_yules_q_handles_perfect_negative_and_degenerate_tables(self):
        negative = TransactionAnalyzer().fit([["a"], ["b"], ["x"]])
        degenerate = TransactionAnalyzer().fit([["a", "b"], ["a", "b"]])

        assert negative.yules_q("a", "b") == pytest.approx(-1.0)
        assert degenerate.yules_q("a", "b") == 0.0

    def test_unfitted_state_is_visible_to_sklearn(self):
        analyzer = TransactionAnalyzer()

        with pytest.raises(NotFittedError):
            check_is_fitted(analyzer)
        analyzer.fit([["apple"]])
        check_is_fitted(analyzer)

    @pytest.mark.parametrize(
        ("rows", "columns", "message"),
        [
            ([], ["beer"], "non-empty"),
            (["apple"], [], "non-empty"),
            (["apple", "apple"], ["beer"], "row_items"),
            (["apple"], ["beer", "beer"], "column_items"),
        ],
    )
    def test_correlation_matrix_rejects_invalid_axes(self, rows, columns, message):
        analyzer = TransactionAnalyzer().fit([["apple", "beer"]])

        with pytest.raises(ValueError, match=message):
            analyzer.correlation_matrix(rows, columns)

    def test_correlation_matrix_supports_non_string_items(self):
        analyzer = TransactionAnalyzer().fit([[1, 2], [1], [2]])

        result = analyzer.correlation_matrix([1], [2], metric="confidence")

        assert result.loc[1, 2] == pytest.approx(0.5)

    def test_save_and_load_require_an_analyzer(self, tmp_path):
        path = tmp_path / "analyzer.pkl"
        analyzer = TransactionAnalyzer().fit([["apple"], ["beer"]])
        analyzer.save(path)

        restored = TransactionAnalyzer.load(path)

        assert restored.columns_ == analyzer.columns_

        with (tmp_path / "wrong.pkl").open("wb") as stream:
            pickle.dump({"not": "an analyzer"}, stream)
        with pytest.raises(TypeError, match="TransactionAnalyzer"):
            TransactionAnalyzer.load(tmp_path / "wrong.pkl")

    def test_save_requires_fit(self, tmp_path):
        with pytest.raises(NotFittedError):
            TransactionAnalyzer().save(tmp_path / "analyzer.pkl")
