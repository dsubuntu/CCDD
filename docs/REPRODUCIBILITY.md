# Reproducibility guide

## Level 1: offline reviewer validation

Requirements: Python 3.9 or newer.

    python scripts/reviewer_smoke_test.py
    python -m unittest discover -s tests -v

This validates:

- the vocabulary-wide absolute log-probability divergence in Eq. (6);
- autoregressive use of one shared output prefix;
- stopping at an EOS token;
- the symmetry of the absolute objective;
- the optional signed objective;
- CA, PA, CCA, and NCA;
- ConFiQA-QA and BBQ-Age schema adapters.

It does not claim to reproduce a paper table. It is a fast correctness test for
the released implementation.

## Level 2: local-model smoke run

Install optional dependencies:

    python -m pip install -e ".[hf]"

Run one paired-context example:

    ccdd generate \
      --model /path/to/a/huggingface-compatible-model \
      --context "Factual context." \
      --counterfactual-context "Counterfactual context." \
      --question "Question?" \
      --max-new-tokens 10 \
      --objective absolute \
      --trace

Use a small model for this interface check. No particular output accuracy is
expected from a tiny model.

## Level 3: benchmark slice

Obtain the dataset from its original source and preserve its license and
citation. The adapters expect the schemas documented in DATA_FORMAT.md.

    python scripts/run_benchmark.py \
      --model /path/to/model \
      --data /path/to/dataset.json \
      --benchmark confiqa \
      --method ccdd \
      --objective absolute \
      --limit 10 \
      --output outputs/slice.json

Run standard decoding by changing method to standard.

## Level 4: paper-scale experiments

The paper reports:

- DeepSeek-R1-Distill-Qwen-7B
- Mistral-7B
- Gemma2-2B
- Qwen2.5 at 1.5B, 3B, and 7B
- FLAN-T5-XL for summarization
- ConFiQA, BBQ-Age, CNN/DailyMail, and RealTime QA

Before claiming table reproduction, record all of the following:

- exact model repository and immutable revision;
- exact dataset version, split, filters, and sample order;
- prompt template;
- counterfactual construction procedure;
- decoding objective and temperature;
- maximum input and output lengths;
- random seed when any sampling is enabled;
- library versions, dtype, device, and GPU model;
- exact-match normalization and accepted aliases.

CNN/DailyMail metrics also require ROUGE, FactKB, and BERTScore dependencies.
They are intentionally not part of the minimal reviewer environment.

## Data and model policy

Do not commit model weights, dataset archives, local cache directories, private
cluster paths, or generated logs. The repository gitignore excludes common
checkpoint formats and raw-data directories.

