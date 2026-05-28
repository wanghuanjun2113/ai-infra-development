---
name: inference-code-review
description: Review code changes for AI inference frameworks including xllm, vllm-ascend, and sglang. Use when Codex needs to inspect diffs for correctness, compatibility, performance risk, distributed/runtime behavior, tests, and missing validation.
---

# Inference Code Review

## Workflow

1. Identify the target framework and changed files.
2. Review behavior changes before style issues.
3. Prioritize correctness, runtime safety, performance regressions, compatibility, and missing tests.
4. Reference exact files and lines in findings.
5. Recommend validation using build, deploy, performance, or accuracy skills when appropriate.

## Review Focus

- Model loading and tokenizer compatibility
- Scheduler, batching, cache, and memory behavior
- Distributed and device binding logic
- API compatibility
- Error handling and observability
- Benchmark and accuracy coverage
