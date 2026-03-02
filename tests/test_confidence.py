import unittest

from metamrag.confidence import build_feature_vector
from metamrag.experts import ExpertOutput


class ConfidenceTests(unittest.TestCase):
    def test_feature_vector_shape_and_values(self):
        output = ExpertOutput(
            name="medrag",
            answer="test answer",
            avg_logprob=-1.0,
            entropy=0.7,
            retrieval_scores=[0.9, 0.8, 0.7],
        )
        feats = build_feature_vector(output)
        self.assertEqual(len(feats), 6)
        self.assertAlmostEqual(feats[2], 0.9)
        self.assertAlmostEqual(feats[3], 0.8)


if __name__ == "__main__":
    unittest.main()
