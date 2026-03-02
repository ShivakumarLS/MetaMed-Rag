from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol


@dataclass
class ExpertOutput:
    """Single expert prediction output."""

    name: str
    answer: str
    avg_logprob: float
    entropy: float
    retrieval_scores: List[float]


class Expert(Protocol):
    """Protocol each base expert adapter should satisfy."""

    name: str

    def predict(self, query: str) -> ExpertOutput:
        ...


class MockExpert:
    """Simple deterministic expert for local testing and demos."""

    def __init__(self, name: str, bias: float = 0.0) -> None:
        self.name = name
        self.bias = bias

    def predict(self, query: str) -> ExpertOutput:
        length_signal = min(len(query) / 120.0, 1.0)
        avg_logprob = -1.5 + self.bias + 0.5 * length_signal
        entropy = 1.0 - 0.25 * length_signal - self.bias * 0.1
        retrieval_scores = [0.8 - 0.1 * i + self.bias * 0.05 for i in range(3)]

        if "treatment" in query.lower():
            answer = f"[{self.name}] likely treatment recommendation"
        elif "diagnosis" in query.lower():
            answer = f"[{self.name}] likely differential diagnosis"
        else:
            answer = f"[{self.name}] medical answer"

        return ExpertOutput(
            name=self.name,
            answer=answer,
            avg_logprob=avg_logprob,
            entropy=max(entropy, 0.01),
            retrieval_scores=retrieval_scores,
        )
