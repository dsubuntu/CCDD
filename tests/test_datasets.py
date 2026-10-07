from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ccdd.datasets import load_benchmark


class DatasetTests(unittest.TestCase):
    def test_confiqa_adapter(self):
        examples = load_benchmark(
            str(ROOT / "data" / "examples" / "confiqa_toy.json"), "confiqa"
        )
        self.assertEqual(len(examples), 2)
        self.assertIn("Lorne Balfe", examples[0].factual_answers)
        self.assertIsNone(examples[0].choices)

    def test_bbq_adapter(self):
        examples = load_benchmark(
            str(ROOT / "data" / "examples" / "bbq_age_toy.json"), "bbq_age"
        )
        self.assertEqual(len(examples), 2)
        self.assertEqual(examples[0].counterfactual_answers[0], "1")
        self.assertEqual(examples[0].choices["2"], "Cannot be determined")


if __name__ == "__main__":
    unittest.main()

