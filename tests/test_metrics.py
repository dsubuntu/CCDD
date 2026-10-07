from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ccdd.metrics import answer_matches, faithfulness_metrics, normalize_answer


class MetricsTests(unittest.TestCase):
    def test_normalization(self):
        self.assertEqual(normalize_answer("  Lorne-Balfe! "), "lorne balfe")

    def test_alias_matching(self):
        self.assertTrue(answer_matches("Lorne Balfe", ["Other", "lorne balfe"]))

    def test_paper_metrics(self):
        metrics = faithfulness_metrics(
            ["a", "wrong"],
            ["a", "b"],
            ["x", "wrong"],
            ["x", "y"],
            parametric_predictions=["a", "z"],
        )
        self.assertEqual(metrics.ca, 0.5)
        self.assertEqual(metrics.cca, 0.5)
        self.assertEqual(metrics.nca, 0.5)
        self.assertEqual(metrics.pa, 0.5)

    def test_length_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            faithfulness_metrics(["a"], [], ["b"], ["b"])


if __name__ == "__main__":
    unittest.main()

