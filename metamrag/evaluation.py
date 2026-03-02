from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence


@dataclass
class ClassificationMetrics:
    accuracy: float
    brier_score: float
    ece: float


def accuracy_score(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    if not y_true:
        return 0.0
    correct = sum(int(t == p) for t, p in zip(y_true, y_pred))
    return correct / len(y_true)


def multiclass_brier_score(y_true: Sequence[int], probas: Sequence[Sequence[float]]) -> float:
    if not y_true:
        return 0.0
    total = 0.0
    k = len(probas[0]) if probas else 0
    for label, probs in zip(y_true, probas):
        total += sum((probs[c] - (1.0 if c == label else 0.0)) ** 2 for c in range(k))
    return total / len(y_true)


def expected_calibration_error(
    y_true: Sequence[int], probas: Sequence[Sequence[float]], n_bins: int = 10
) -> float:
    if not y_true:
        return 0.0

    confidences: List[float] = []
    correctness: List[int] = []
    for label, probs in zip(y_true, probas):
        pred = max(range(len(probs)), key=lambda i: probs[i])
        confidences.append(probs[pred])
        correctness.append(int(pred == label))

    ece = 0.0
    for idx in range(n_bins):
        lo = idx / n_bins
        hi = (idx + 1) / n_bins
        bucket = [i for i, conf in enumerate(confidences) if lo <= conf < hi or (idx == n_bins - 1 and conf == 1.0)]
        if not bucket:
            continue
        avg_conf = sum(confidences[i] for i in bucket) / len(bucket)
        avg_acc = sum(correctness[i] for i in bucket) / len(bucket)
        ece += (len(bucket) / len(y_true)) * abs(avg_acc - avg_conf)
    return ece


def build_metrics(y_true: Sequence[int], y_pred: Sequence[int], probas: Sequence[Sequence[float]]) -> ClassificationMetrics:
    return ClassificationMetrics(
        accuracy=accuracy_score(y_true, y_pred),
        brier_score=multiclass_brier_score(y_true, probas),
        ece=expected_calibration_error(y_true, probas),
    )


def coverage_at_threshold(confidences: Iterable[float], threshold: float) -> float:
    conf_list = list(confidences)
    if not conf_list:
        return 0.0
    kept = sum(1 for c in conf_list if c >= threshold)
    return kept / len(conf_list)
