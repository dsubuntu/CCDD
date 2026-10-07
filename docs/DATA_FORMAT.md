# Data formats

## ConFiQA-QA

The adapter accepts a JSON array or JSONL file. Required fields are:

    {
      "question": "...",
      "orig_context": "...",
      "cf_context": "...",
      "orig_answer": "...",
      "cf_answer": "..."
    }

Optional orig_alias and cf_alias arrays are treated as accepted answers.
See data/examples/confiqa_toy.json for a synthetic example.

## BBQ-Age

The adapter accepts a JSON array or JSONL file. Required fields are:

    {
      "question": "...",
      "orig_context": "...",
      "cf_context": "...",
      "orig_answer": 2,
      "cf_answer": 1,
      "orig_ans0": "...",
      "orig_ans1": "...",
      "orig_ans2": "..."
    }

The numeric option and its text are both accepted during metric calculation.
See data/examples/bbq_age_toy.json.

## Offline prediction metrics

scripts/evaluate_predictions.py reads JSONL records with:

    {
      "factual_prediction": "...",
      "factual_target": "... or an alias array",
      "counterfactual_prediction": "...",
      "counterfactual_target": "... or an alias array",
      "parametric_prediction": "optional"
    }

PA is emitted only when every record contains parametric_prediction.

