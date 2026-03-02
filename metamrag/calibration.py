from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence


def _softmax(logits: Sequence[float]) -> List[float]:
    m = max(logits)
    exps = [math.exp(v - m) for v in logits]
    total = sum(exps)
    return [v / total for v in exps]


@dataclass
class TemperatureScaler:
    """Simple temperature scaling calibrated on validation logits."""

    temperature: float = 1.0
    min_temp: float = 0.05
    max_temp: float = 10.0
    candidates: int = 60

    @staticmethod
    def _nll(logits: Sequence[Sequence[float]], labels: Sequence[int], temperature: float) -> float:
        loss = 0.0
        for row, y in zip(logits, labels):
            probs = _softmax([v / temperature for v in row])
            loss -= math.log(max(probs[y], 1e-12))
        return loss / max(1, len(labels))

    def fit(self, logits: Sequence[Sequence[float]], labels: Sequence[int]) -> float:
        if not logits:
            self.temperature = 1.0
            return self.temperature

        best_t = 1.0
        best_loss = float("inf")
        for i in range(self.candidates):
            ratio = i / max(1, self.candidates - 1)
            t = self.min_temp + ratio * (self.max_temp - self.min_temp)
            loss = self._nll(logits, labels, t)
            if loss < best_loss:
                best_loss = loss
                best_t = t

        self.temperature = best_t
        return self.temperature

    def transform_logits(self, logits: Sequence[float]) -> List[float]:
        return [v / self.temperature for v in logits]
