from pathlib import Path
import math
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ccdd.core import decode, divergence_scores, log_softmax, select_token


class CoreTests(unittest.TestCase):
    def test_log_softmax_normalizes(self):
        values = log_softmax([1.0, 2.0, 3.0])
        self.assertAlmostEqual(sum(math.exp(value) for value in values), 1.0)

    def test_equation_6_absolute_selection(self):
        factual = [math.log(value) for value in (0.70, 0.10, 0.10, 0.10)]
        counterfactual = [math.log(value) for value in (0.10, 0.30, 0.30, 0.30)]
        decision = select_token(factual, counterfactual)
        self.assertEqual(decision.token_id, 0)
        self.assertAlmostEqual(decision.divergence, math.log(7.0))

    def test_absolute_objective_is_symmetric(self):
        factual = [3.0, 2.0, 1.0]
        counterfactual = [1.0, 2.5, 2.0]
        self.assertEqual(
            divergence_scores(factual, counterfactual),
            divergence_scores(counterfactual, factual),
        )

    def test_signed_objective_is_oriented(self):
        factual = [4.0, 0.0, 0.0]
        counterfactual = [0.0, 2.0, 2.0]
        self.assertEqual(select_token(factual, counterfactual, objective="signed").token_id, 0)
        self.assertGreater(
            select_token(factual, counterfactual, objective="signed").signed_divergence,
            0.0,
        )

    def test_candidate_subset(self):
        decision = select_token([5.0, 1.0, 0.0], [0.0, 1.0, 5.0], candidate_token_ids=[1, 2])
        self.assertEqual(decision.token_id, 2)

    def test_decode_stops_at_eos(self):
        steps = [
            (
                [math.log(0.7), math.log(0.2), math.log(0.1)],
                [math.log(0.1), math.log(0.45), math.log(0.45)],
            ),
            ([0.0, 0.0, 3.0], [0.0, 0.0, 3.0]),
        ]

        def provider(generated):
            return steps[min(len(generated), 1)]

        result = decode(provider, max_new_tokens=5, eos_token_id=2)
        self.assertEqual(result.token_ids, (0, 2))
        self.assertTrue(result.stopped_on_eos)

    def test_invalid_temperature(self):
        with self.assertRaises(ValueError):
            log_softmax([1.0], temperature=0.0)


if __name__ == "__main__":
    unittest.main()

