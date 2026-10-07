# CCDD: Counterfactual Context Divergence Decoding

This repository is the cleaned companion implementation for the paper
**Causal Decoding for Context-Faithful Generation in Large Language Models**.
CCDD compares next-token distributions under a factual context and a
counterfactual context, then selects a token according to their divergence.

The release is intentionally separated from model checkpoints, full datasets,
cluster logs, and duplicated experiment folders. A reviewer can validate the
central decoding equation, the proposed metrics, and the two dataset adapters
without a GPU, network connection, or third-party Python package.

## Reviewer quick check

From the repository root:

    python scripts/reviewer_smoke_test.py
    python -m unittest discover -s tests -v
    python scripts/evaluate_predictions.py data/example_predictions.jsonl

Expected result: the smoke test prints a JSON object with status PASS and the
unit test command reports all tests passing. These commands do not download a
model or dataset.

## What is included

- A dependency-free implementation of paper Eq. (6), Eqs. (7)-(8), and the
  CA, PA, CCA, and NCA metrics.
- A Hugging Face adapter for decoder-only and encoder-decoder models.
- Schema adapters for ConFiQA-QA and BBQ-Age.
- Synthetic, redistributable reviewer fixtures.
- A small local-model benchmark runner and an offline metric calculator.
- GitHub Actions checks on Linux and Windows.

Full model checkpoints and benchmark datasets are not committed. They are
large and may have their own distribution terms. See docs/REPRODUCIBILITY.md.

## Installation

The reviewer check needs Python 3.9 or newer and nothing else.

Install the package itself:

    python -m pip install -e .
    ccdd demo

Install optional Hugging Face support:

    python -m pip install -e ".[hf]"

## Run CCDD with a local model

The following command accepts either a local model directory or a Hugging Face
model identifier:

    ccdd generate \
      --model /path/to/model \
      --context "The current context states that the answer is Paris." \
      --counterfactual-context "The counterfactual context states that the answer is Lyon." \
      --question "What is the answer?" \
      --objective absolute \
      --trace

The default objective, absolute, is the literal implementation of Eq. (6).
The signed objective is also available for experiments that treat the first
context as the explicitly preferred condition:

    ccdd generate ... --objective signed

The distinction is important and is documented in
docs/IMPLEMENTATION_NOTES.md.

## Run a small benchmark

With a local Hugging Face-compatible model and a locally obtained dataset:

    python scripts/run_benchmark.py \
      --model /path/to/model \
      --data /path/to/ConFiQA-QA.json \
      --benchmark confiqa \
      --method ccdd \
      --objective absolute \
      --limit 10 \
      --output outputs/confiqa_smoke.json

For BBQ-Age, use benchmark value bbq_age. To run the deterministic greedy
baseline, use method value standard.

## Repository layout

    src/ccdd/core.py          Paper equation and model-independent decoder
    src/ccdd/hf.py            Hugging Face generation adapter
    src/ccdd/metrics.py       CA, PA, CCA, and NCA
    src/ccdd/datasets.py      ConFiQA-QA and BBQ-Age adapters
    scripts/                  Reviewer, metric, and benchmark entry points
    data/examples/            Synthetic schema examples
    tests/                    Offline unit tests
    docs/                     Reproducibility and implementation notes

## Citation

Citation metadata is provided in CITATION.cff. Update it with the final venue,
DOI, and publication URL after acceptance.

## License

A software license has deliberately not been selected in this draft because
license choice belongs to the authors and their institutions. Select and add
the intended license before making the GitHub repository public.

