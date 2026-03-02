from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from typing import List, Sequence


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _softmax(logits: Sequence[float]) -> List[float]:
    m = max(logits)
    exps = [math.exp(v - m) for v in logits]
    total = sum(exps)
    return [v / total for v in exps]


@dataclass
class StandardScaler:
    means: List[float] = field(default_factory=list)
    stds: List[float] = field(default_factory=list)

    def fit(self, rows: Sequence[Sequence[float]]) -> None:
        n_features = len(rows[0])
        self.means = []
        self.stds = []
        for j in range(n_features):
            values = [r[j] for r in rows]
            mean = sum(values) / len(values)
            var = sum((v - mean) ** 2 for v in values) / len(values)
            std = math.sqrt(max(var, 1e-12))
            self.means.append(mean)
            self.stds.append(std)

    def transform(self, row: Sequence[float]) -> List[float]:
        return [(v - m) / s for v, m, s in zip(row, self.means, self.stds)]


@dataclass
class LinearMetaModel:
    """Multinomial linear stacker with feature normalization and L2 regularization."""

    n_features: int
    expert_names: List[str]
    learning_rate: float = 0.03
    epochs: int = 400
    l2: float = 1e-3
    seed: int = 42
    weights: List[List[float]] = field(default_factory=list)
    bias: List[float] = field(default_factory=list)
    scaler: StandardScaler = field(default_factory=StandardScaler)

    def __post_init__(self) -> None:
        if not self.weights:
            self.weights = [[0.0 for _ in range(self.n_features)] for _ in self.expert_names]
        if not self.bias:
            self.bias = [0.0 for _ in self.expert_names]

    def fit(self, x_rows: List[List[float]], y_labels: List[int]) -> None:
        if not x_rows:
            raise ValueError("Training data is empty")

        self.scaler.fit(x_rows)
        normalized = [self.scaler.transform(x) for x in x_rows]

        n_classes = len(self.expert_names)
        rng = random.Random(self.seed)
        indices = list(range(len(normalized)))

        for _ in range(self.epochs):
            rng.shuffle(indices)
            for idx in indices:
                x = normalized[idx]
                y = y_labels[idx]

                logits = self.raw_logits(x, already_scaled=True)
                probs = _softmax(logits)

                for k in range(n_classes):
                    target = 1.0 if k == y else 0.0
                    grad = probs[k] - target
                    for j in range(self.n_features):
                        self.weights[k][j] -= self.learning_rate * (grad * x[j] + self.l2 * self.weights[k][j])
                    self.bias[k] -= self.learning_rate * grad

    def raw_logits(self, x: Sequence[float], already_scaled: bool = False) -> List[float]:
        x_in = list(x) if already_scaled else self.scaler.transform(x)
        return [_dot(w, x_in) + b for w, b in zip(self.weights, self.bias)]

    def predict_proba(self, x: Sequence[float]) -> List[float]:
        return _softmax(self.raw_logits(x))

    def predict(self, x: Sequence[float]) -> int:
        probs = self.predict_proba(x)
        return max(range(len(probs)), key=lambda i: probs[i])

    def save(self, path: str) -> None:
        payload = {
            "n_features": self.n_features,
            "expert_names": self.expert_names,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "l2": self.l2,
            "seed": self.seed,
            "weights": self.weights,
            "bias": self.bias,
            "scaler": {"means": self.scaler.means, "stds": self.scaler.stds},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "LinearMetaModel":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        model = cls(
            n_features=payload["n_features"],
            expert_names=payload["expert_names"],
            learning_rate=payload["learning_rate"],
            epochs=payload["epochs"],
            l2=payload.get("l2", 1e-3),
            seed=payload.get("seed", 42),
            weights=payload["weights"],
            bias=payload["bias"],
        )
        model.scaler = StandardScaler(
            means=payload.get("scaler", {}).get("means", [0.0] * model.n_features),
            stds=payload.get("scaler", {}).get("stds", [1.0] * model.n_features),
        )
        return model
