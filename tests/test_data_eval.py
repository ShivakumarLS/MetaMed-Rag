import tempfile
import unittest
from pathlib import Path

from metamrag.data import DatasetFormatError, load_training_samples
from metamrag.evaluation import expected_calibration_error


class DataAndEvaluationTests(unittest.TestCase):
    def test_data_schema_validation(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "bad.jsonl"
            path.write_text('{"query": "x"}\n', encoding="utf-8")
            with self.assertRaises(DatasetFormatError):
                load_training_samples(path)

    def test_ece_range(self):
        y_true = [0, 1, 0, 1]
        probs = [
            [0.9, 0.1],
            [0.2, 0.8],
            [0.6, 0.4],
            [0.3, 0.7],
        ]
        ece = expected_calibration_error(y_true, probs)
        self.assertTrue(0.0 <= ece <= 1.0)


if __name__ == "__main__":
    unittest.main()
