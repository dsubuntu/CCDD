# Implementation notes and audit trail

## Paper definition

The paper defines standard decoding at step t as the highest-probability token
conditioned on context c, query q, and the shared generated prefix. It defines
CCDD in Eq. (6) as:

    argmax over v of
    absolute(log p(v | c, q, y before t) - log p(v | c prime, q, y before t))

The implementation in src/ccdd/core.py follows this equation literally:

1. Apply log-softmax to both vocabulary-logit vectors.
2. Subtract token-aligned log probabilities.
3. Take the absolute value for the paper objective.
4. Select the largest score.
5. Append the selected token to the one shared autoregressive history.

The implementation records both signed and absolute divergence for every
selected token. This provides an auditable trace and supports the average
selected divergence described by Eqs. (7)-(8).

## Absolute versus signed objectives

The printed absolute objective is symmetric: swapping c and c prime produces
the same divergence scores. It identifies tokens most sensitive to the context
intervention, regardless of which condition makes a token more likely.

The signed objective instead selects:

    argmax over v of
    log p(v | c, q, y before t) - log p(v | c prime, q, y before t)

This objective is directional and treats the first context as primary. The
command-line interface therefore requires the objective to be reported in
saved experiment configuration. The default remains absolute because that is
the published equation.

## Relationship to the archived scripts

The source folder contained several families of scripts:

- BBQ_evaluation.py and confiqa_evaluation.py sampled complete candidate
  answers under the primary context, scored each complete answer under two
  contexts, and selected the largest signed sequence-level difference.
- Model-specific copies changed paths and small loading details but duplicated
  most evaluation logic.
- evaluate_with_CAD contained a T5 loop that scaled one distribution; it did
  not compute a factual-counterfactual distribution difference.
- trendEva scripts scored answer choices or pairwise score gaps rather than the
  token-level Eq. (6) rule.
- Cluster shell files embedded private absolute paths and hardware node names.

Those files were not copied into the release package. The cleaned package
implements the stated method once, adds deterministic tests, and records the
objective explicitly. Consequently, this repository is a paper-faithful
reference implementation, not a bit-for-bit rerun of every historical script.

If previously reported tables must be reproduced exactly, archive the original
environment and identify the exact script revision, seed, checkpoint, prompt,
candidate count, and dataset slice used for each row.

## Determinism

- The pure-Python core is deterministic.
- Ties are resolved by higher factual probability and then lower token id.
- The standard baseline uses greedy generation.
- CCDD itself takes an argmax at every step.
- Reviewer fixtures contain explicit probability distributions.

GPU kernels and model implementations may still introduce small numerical
differences. Record the model revision, Transformers version, Torch version,
device, dtype, and objective in full experiments.

