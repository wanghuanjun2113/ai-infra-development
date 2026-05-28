---
name: inference-bugfix
description: Diagnose and fix bugs in AI inference frameworks including xllm, vllm-ascend, and sglang. Use when Codex needs to reproduce failures, inspect logs, identify root cause, implement focused fixes, and validate with deterministic build, deploy, performance, or accuracy checks.
---

# Inference Bugfix

## Workflow

1. Capture the failure command, logs, framework, branch, commit, model, and devices.
2. Read the Known Issues section in `frameworks/<framework>.md` when symptoms match.
3. Reproduce the issue with the smallest deterministic command.
4. Identify root cause before editing.
5. Make a focused fix.
6. Validate with the narrowest reliable test, then broader checks if risk is high.
7. Summarize root cause, fix, and validation.

## Artifacts

Save reproduction logs and validation output under `runs/bugfix/` when the task needs durable evidence.
