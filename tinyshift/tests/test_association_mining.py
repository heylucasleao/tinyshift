import numpy as np
import pandas as pd
import pytest

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
