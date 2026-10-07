from pathlib import Path
import importlib.util
import math
from types import SimpleNamespace
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "optional torch dependency is not installed")
class HuggingFaceAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        cls.torch = torch

    def test_decoder_only_generation_with_fake_model(self):
        from ccdd.hf import generate_ccdd

        torch = self.torch

        class Tokenizer:
            eos_token_id = 3
            pad_token_id = 3
            bos_token_id = None

            def __call__(self, text, return_tensors):
                token = 10 if text == "factual" else 20
                return {"input_ids": torch.tensor([[token]], dtype=torch.long)}

            def decode(self, token_ids, skip_special_tokens):
                return " ".join(str(token) for token in token_ids if token != 3)

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(1))
                self.config = SimpleNamespace(is_encoder_decoder=False, eos_token_id=3)

            def forward(self, input_ids):
                if input_ids.shape[1] > 1:
                    probabilities = (0.10, 0.10, 0.10, 0.70)
                elif int(input_ids[0, 0]) == 10:
                    probabilities = (0.70, 0.10, 0.10, 0.10)
                else:
                    probabilities = (0.10, 0.30, 0.30, 0.30)
                logits = torch.tensor(
                    [[[math.log(value) for value in probabilities]]], dtype=torch.float32
                )
                return SimpleNamespace(logits=logits)

        result = generate_ccdd(Model(), Tokenizer(), "factual", "counterfactual")
        self.assertEqual(result.token_ids, (0, 3))
        self.assertTrue(result.stopped_on_eos)

    def test_encoder_decoder_generation_with_fake_model(self):
        from ccdd.hf import generate_ccdd

        torch = self.torch

        class Tokenizer:
            eos_token_id = 3
            pad_token_id = 3
            bos_token_id = None

            def __call__(self, text, return_tensors):
                token = 10 if text == "factual" else 20
                return {
                    "input_ids": torch.tensor([[token]], dtype=torch.long),
                    "attention_mask": torch.ones((1, 1), dtype=torch.long),
                }

            def decode(self, token_ids, skip_special_tokens):
                return " ".join(str(token) for token in token_ids if token != 3)

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(1))
                self.config = SimpleNamespace(
                    is_encoder_decoder=True,
                    eos_token_id=3,
                    decoder_start_token_id=3,
                )

            def forward(self, input_ids, attention_mask, decoder_input_ids):
                if decoder_input_ids.shape[1] > 1:
                    probabilities = (0.10, 0.10, 0.10, 0.70)
                elif int(input_ids[0, 0]) == 10:
                    probabilities = (0.70, 0.10, 0.10, 0.10)
                else:
                    probabilities = (0.10, 0.30, 0.30, 0.30)
                logits = torch.tensor(
                    [[[math.log(value) for value in probabilities]]], dtype=torch.float32
                )
                return SimpleNamespace(logits=logits)

        result = generate_ccdd(Model(), Tokenizer(), "factual", "counterfactual")
        self.assertEqual(result.token_ids, (0, 3))
        self.assertTrue(result.stopped_on_eos)


if __name__ == "__main__":
    unittest.main()

