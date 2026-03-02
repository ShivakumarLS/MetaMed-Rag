from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from .experts import ExpertOutput


@dataclass
class TrainingSample:
    query: str
    gold_answer: str
    experts: List[ExpertOutput]


class DatasetFormatError(ValueError):
    """Raised when a JSONL record does not satisfy expected schema."""


def _validate_record(record: dict, line_no: int) -> None:
    required = {"query", "gold_answer", "experts"}
    missing = required - set(record)
    if missing:
        raise DatasetFormatError(f"Line {line_no}: missing keys {sorted(missing)}")
    if not isinstance(record["experts"], list) or not record["experts"]:
        raise DatasetFormatError(f"Line {line_no}: experts must be a non-empty list")


def load_training_samples(path: str | Path) -> List[TrainingSample]:
    samples: List[TrainingSample] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            _validate_record(payload, line_no)
            expert_outputs = [ExpertOutput(**item) for item in payload["experts"]]
            samples.append(
                TrainingSample(
                    query=payload["query"],
                    gold_answer=payload["gold_answer"],
                    experts=expert_outputs,
                )
            )
    if not samples:
        raise DatasetFormatError("Dataset is empty")
    return samples


def iter_jsonl(path: str | Path) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)
