from __future__ import annotations

from statistics import mean
from typing import List

from .experts import ExpertOutput


def _safe_mean(values: List[float]) -> float:
    return mean(values) if values else 0.0


def build_feature_vector(output: ExpertOutput) -> List[float]:
    """Construct confidence features for one expert output.

    Features:
    - avg_logprob (higher better)
    - entropy (lower better)
    - top retrieval score
    - mean retrieval score
    - retrieval spread (max - min)
    - answer length
    """

    scores = output.retrieval_scores
    top_score = max(scores) if scores else 0.0
    mean_score = _safe_mean(scores)
    spread = (max(scores) - min(scores)) if scores else 0.0
    answer_length = float(len(output.answer.split()))

    return [
        output.avg_logprob,
        output.entropy,
        top_score,
        mean_score,
        spread,
        answer_length,
    ]
