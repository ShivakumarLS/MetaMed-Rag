from __future__ import annotations

from dataclasses import dataclass
import json
import math
from difflib import SequenceMatcher
from typing import Dict, List, Sequence, Tuple

from .calibration import TemperatureScaler
from .confidence import build_feature_vector
from .data import TrainingSample, load_training_samples
from .evaluation import build_metrics, coverage_at_threshold
from .experts import Expert, ExpertOutput
from .meta_model import LinearMetaModel


@dataclass
class MetaPrediction:
    answer: str
    selected_expert: str
    confidence: float
    abstained: bool
    per_expert_confidence: Dict[str, float]


@dataclass
class TrainingReport:
    train_accuracy: float
    train_brier: float
    train_ece: float
    abstention_coverage: float
    samples: int
    calibrated_temperature: float


class MetaRAGPipeline:
    def __init__(self, experts: List[Expert], abstain_threshold: float = 0.6) -> None:
        if not experts:
            raise ValueError("At least one expert is required")
        self.experts = experts
        self.model: LinearMetaModel | None = None
        self.abstain_threshold = abstain_threshold
        self.temperature_scaler = TemperatureScaler()

    @staticmethod
    def _correctness_score(pred: str, gold: str) -> float:
        return SequenceMatcher(None, pred.lower().strip(), gold.lower().strip()).ratio()

    @staticmethod
    def _aggregate_features(outputs: List[ExpertOutput]) -> Tuple[List[float], List[List[float]]]:
        per_expert = [build_feature_vector(o) for o in outputs]
        flat = [value for expert_feats in per_expert for value in expert_feats]
        return flat, per_expert

    def _build_supervised_rows(self, samples: Sequence[TrainingSample]) -> Tuple[List[List[float]], List[int]]:
        x_rows: List[List[float]] = []
        y_labels: List[int] = []
        for sample in samples:
            flat, _ = self._aggregate_features(sample.experts)
            scores = [self._correctness_score(o.answer, sample.gold_answer) for o in sample.experts]
            best_idx = max(range(len(scores)), key=lambda i: scores[i])
            x_rows.append(flat)
            y_labels.append(best_idx)
        return x_rows, y_labels

    def train_from_jsonl(self, train_file: str, model_out: str) -> Dict[str, float]:
        samples = load_training_samples(train_file)
        expert_names = [e.name for e in self.experts]

        x_rows, y_labels = self._build_supervised_rows(samples)

        n_features = len(x_rows[0])
        self.model = LinearMetaModel(n_features=n_features, expert_names=expert_names)
        self.model.fit(x_rows, y_labels)

        logits = [self.model.raw_logits(x) for x in x_rows]
        calibrated_t = self.temperature_scaler.fit(logits, y_labels)

        calibrated_probas = [
            self._probas_from_logits(self.temperature_scaler.transform_logits(l)) for l in logits
        ]
        preds = [max(range(len(p)), key=lambda i: p[i]) for p in calibrated_probas]

        metrics = build_metrics(y_labels, preds, calibrated_probas)
        confidences = [max(p) for p in calibrated_probas]
        coverage = coverage_at_threshold(confidences, self.abstain_threshold)

        self.model.save(model_out)
        self._persist_temperature(model_out)
        return TrainingReport(
            train_accuracy=metrics.accuracy,
            train_brier=metrics.brier_score,
            train_ece=metrics.ece,
            abstention_coverage=coverage,
            samples=len(y_labels),
            calibrated_temperature=calibrated_t,
        ).__dict__

    @staticmethod
    def _probas_from_logits(logits: Sequence[float]) -> List[float]:
        max_v = max(logits)
        exps = [math.exp(v - max_v) for v in logits]
        total = sum(exps)
        return [e / total for e in exps]


    def _persist_temperature(self, model_file: str) -> None:
        with open(model_file, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        payload["temperature"] = self.temperature_scaler.temperature
        with open(model_file, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def load_model(self, model_file: str) -> None:
        self.model = LinearMetaModel.load(model_file)
        with open(model_file, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        self.temperature_scaler.temperature = payload.get("temperature", 1.0)

    def predict(self, query: str) -> MetaPrediction:
        if self.model is None:
            raise RuntimeError("Meta-model not loaded. Train or load model first.")

        outputs = [e.predict(query) for e in self.experts]
        flat, _ = self._aggregate_features(outputs)

        logits = self.model.raw_logits(flat)
        calibrated = self.temperature_scaler.transform_logits(logits)
        probs = self._probas_from_logits(calibrated)

        winner = max(range(len(probs)), key=lambda i: probs[i])
        confidence = probs[winner]
        abstained = confidence < self.abstain_threshold

        selected = outputs[winner]
        return MetaPrediction(
            answer=selected.answer if not abstained else "ABSTAIN: low confidence; requires clinician review",
            selected_expert=selected.name,
            confidence=confidence,
            abstained=abstained,
            per_expert_confidence={name: p for name, p in zip(self.model.expert_names, probs)},
        )
