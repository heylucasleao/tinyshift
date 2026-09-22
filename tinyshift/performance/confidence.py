# Copyright (c) 2024-2026 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License

"""Classification performance estimates from calibrated class probabilities."""

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, check_is_fitted


class ConfidenceBasedPerformanceEstimator(BaseEstimator):
    """Estimate classification performance from class probabilities.

    Probabilities must be calibrated and supplied for every class, in the same
    order during fitting and estimation. Each row must sum to one. The method
    estimates a confusion matrix by assigning each row to its most probable
    class and distributing its expected true class according to its probability
    vector. No current targets are needed.

    Parameters
    ----------
    metric : {"accuracy", "precision", "recall", "f1"}, default="accuracy"
        Metric to estimate from the expected confusion matrix.
    average : {"auto", "binary", "macro"}, default="auto"
        For non-accuracy metrics, ``auto`` uses the positive class in binary
        classification and macro averaging in multiclass classification.
    positive_label : object or None, default=None
        Positive class for binary averaging. Defaults to the last class in
        ``classes`` passed to :meth:`fit`.

    Notes
    -----
    The reference predictions should be out-of-sample predictions from the
    monitored classifier. Estimates can be inaccurate if probabilities are
    miscalibrated, particularly when calibration changes in the current batch.
    """

    def __init__(
        self, metric: str = "accuracy", average: str = "auto", positive_label=None
    ) -> None:
        self.metric = metric
        self.average = average
        self.positive_label = positive_label

    @staticmethod
    def _probabilities(probabilities, n_classes):
        values = check_array(probabilities, ensure_2d=True, dtype=float)
        if values.shape[1] != n_classes:
            raise ValueError(f"Expected {n_classes} probability columns.")
        if np.any((values < 0) | (values > 1)):
            raise ValueError("Probabilities must lie in [0, 1].")
        if not np.allclose(values.sum(axis=1), 1.0, rtol=0, atol=1e-6):
            raise ValueError("Probability rows must sum to one.")
        return values

    @staticmethod
    def _classes(classes):
        labels = list(classes)
        if len(labels) < 2 or len(set(labels)) != len(labels):
            raise ValueError("classes must contain at least two distinct labels.")
        return labels

    def fit(self, y_true, probabilities, classes):
        """Validate labeled reference predictions and store baseline metrics."""
        if self.metric not in {"accuracy", "precision", "recall", "f1"}:
            raise ValueError("metric must be accuracy, precision, recall, or f1.")
        if self.average not in {"auto", "binary", "macro"}:
            raise ValueError("average must be auto, binary, or macro.")
        labels = self._classes(classes)
        if self.average == "binary" and len(labels) != 2:
            raise ValueError("binary averaging requires exactly two classes.")
        positive = labels[-1] if self.positive_label is None else self.positive_label
        if positive not in labels:
            raise ValueError("positive_label must be one of classes.")
        values = self._probabilities(probabilities, len(labels))
        observed = np.asarray(y_true)
        if observed.ndim != 1 or len(observed) != len(values):
            raise ValueError("y_true must be a one-dimensional vector matching probabilities.")
        if any(label not in labels for label in observed):
            raise ValueError("y_true contains a label absent from classes.")

        self.classes_ = labels
        self.positive_label_ = positive
        self.reference_size_ = len(values)
        self.reference_estimated_ = self.estimate(values)
        predicted = values.argmax(axis=1)
        true_indices = np.array([labels.index(label) for label in observed])
        confusion = np.zeros((len(labels), len(labels)), dtype=float)
        np.add.at(confusion, (predicted, true_indices), 1.0)
        self.reference_realized_ = self._metric_from_confusion(confusion)
        return self

    def expected_confusion_matrix(self, probabilities) -> np.ndarray:
        """Return expected counts with predicted classes in rows, true classes in columns."""
        check_is_fitted(self, "classes_")
        values = self._probabilities(probabilities, len(self.classes_))
        predicted = values.argmax(axis=1)
        confusion = np.zeros((len(self.classes_), len(self.classes_)), dtype=float)
        np.add.at(confusion, predicted, values)
        return confusion

    def _metric_from_confusion(self, confusion):
        total = confusion.sum()
        true_positive = np.diag(confusion)
        if self.metric == "accuracy":
            return float(true_positive.sum() / total)

        predicted_count = confusion.sum(axis=1)
        true_count = confusion.sum(axis=0)
        precision = np.divide(
            true_positive, predicted_count, out=np.zeros_like(true_positive), where=predicted_count > 0
        )
        recall = np.divide(
            true_positive, true_count, out=np.zeros_like(true_positive), where=true_count > 0
        )
        f1 = np.divide(
            2 * precision * recall,
            precision + recall,
            out=np.zeros_like(true_positive),
            where=(precision + recall) > 0,
        )
        values = {"precision": precision, "recall": recall, "f1": f1}[self.metric]
        average = "binary" if self.average == "auto" and len(self.classes_) == 2 else self.average
        if average == "binary":
            return float(values[self.classes_.index(self.positive_label_)])
        return float(values.mean())

    def estimate(self, probabilities) -> float:
        """Estimate the configured metric for an unlabeled batch."""
        return self._metric_from_confusion(self.expected_confusion_matrix(probabilities))
