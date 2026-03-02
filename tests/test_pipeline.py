import json
import tempfile
import unittest
from pathlib import Path

from metamrag.experts import MockExpert
from metamrag.pipeline import MetaRAGPipeline


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = MetaRAGPipeline(
            experts=[
                MockExpert("medrag", bias=0.2),
                MockExpert("clinicalbert", bias=0.0),
                MockExpert("pubmedbert", bias=-0.1),
            ]
        )

    def test_train_predict_and_temperature_persistence(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "model.json"
            metrics = self.pipeline.train_from_jsonl("examples/train.jsonl", str(out))
            self.assertGreaterEqual(metrics["train_accuracy"], 0.0)
            self.assertIn("train_ece", metrics)
            self.assertIn("calibrated_temperature", metrics)

            payload = json.loads(out.read_text())
            self.assertIn("temperature", payload)

            self.pipeline.load_model(str(out))
            pred = self.pipeline.predict("What treatment is used for uncomplicated UTI?")
            self.assertIn(pred.selected_expert, {"medrag", "clinicalbert", "pubmedbert"})
            self.assertTrue(0.0 <= pred.confidence <= 1.0)


if __name__ == "__main__":
    unittest.main()
