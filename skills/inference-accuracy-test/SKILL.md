---
name: inference-accuracy-test
description: Run deterministic accuracy evaluations for xllm, vllm-ascend, and sglang. Use when Codex needs to execute fixed eval cases such as CEval or MMLU, preserve raw predictions, parse scores, compare baselines, and produce repeatable accuracy reports.
---

# Inference Accuracy Test

## Workflow

1. Resolve framework, model, and accuracy case from `development.yaml`.
2. Read the Accuracy section in `frameworks/<framework>.md`.
3. Create `runs/accuracy/<run_id>/` with `python3 scripts/dev.py run accuracy --framework <framework> --case <case>` when possible.
4. Use fixed decoding parameters and seeds.
5. Preserve raw eval output and failed cases.
6. Write `metrics.json` with dataset, score, baseline, delta, and verdict.
7. Produce `report.md`.

## Failure Policy

If evaluation cannot complete, keep logs and mark `INCONCLUSIVE`. If score regresses beyond threshold, mark `FAIL`.
