from __future__ import annotations

import argparse
import json

from .experts import MockExpert
from .pipeline import MetaRAGPipeline


def build_default_pipeline(abstain_threshold: float) -> MetaRAGPipeline:
    experts = [
        MockExpert("medrag", bias=0.25),
        MockExpert("clinicalbert", bias=0.05),
        MockExpert("pubmedbert", bias=-0.05),
    ]
    return MetaRAGPipeline(experts=experts, abstain_threshold=abstain_threshold)


def cmd_train(args: argparse.Namespace) -> None:
    pipeline = build_default_pipeline(args.abstain_threshold)
    metrics = pipeline.train_from_jsonl(args.train_file, args.model_out)
    print(json.dumps(metrics, indent=2))


def cmd_predict(args: argparse.Namespace) -> None:
    pipeline = build_default_pipeline(args.abstain_threshold)
    pipeline.load_model(args.model_file)
    prediction = pipeline.predict(args.query)
    print(
        json.dumps(
            {
                "answer": prediction.answer,
                "selected_expert": prediction.selected_expert,
                "confidence": prediction.confidence,
                "abstained": prediction.abstained,
                "per_expert_confidence": prediction.per_expert_confidence,
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="MetaMed-RAG CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train meta-model")
    train_parser.add_argument("--train-file", required=True, help="Training JSONL file")
    train_parser.add_argument("--model-out", required=True, help="Output model JSON path")
    train_parser.add_argument(
        "--abstain-threshold",
        type=float,
        default=0.6,
        help="Abstain when max confidence is below this threshold",
    )
    train_parser.set_defaults(func=cmd_train)

    predict_parser = subparsers.add_parser("predict", help="Run inference")
    predict_parser.add_argument("--model-file", required=True, help="Saved model JSON path")
    predict_parser.add_argument("--query", required=True, help="Input medical query")
    predict_parser.add_argument(
        "--abstain-threshold",
        type=float,
        default=0.6,
        help="Abstain when max confidence is below this threshold",
    )
    predict_parser.set_defaults(func=cmd_predict)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
