---
name: inference-build-install
description: Build, compile, package, install, and validate inference frameworks including xllm, vllm-ascend, and sglang. Use when Codex needs to prepare source repos, run framework-specific build or install steps, validate imports, preserve logs, and create a deterministic build report.
---

# Inference Build Install

## Workflow

1. Locate `development.yaml` and resolve the target framework.
2. Inspect `code/<framework>` git branch, commit, and dirty state.
3. Read the Build section in `frameworks/<framework>.md`.
4. Prefer `python3 scripts/dev.py run build --framework <framework>` to create a run directory and manifest.
5. Save raw build, install, and validation logs under `runs/build/<run_id>/`.
6. Mark the result `PASS`, `FAIL`, or `INCONCLUSIVE`.

## Required Artifacts

- `manifest.json`
- `env.json`
- `build.log`
- `install.log`
- `validation.log`
- `report.md`

## Failure Policy

If prerequisites are missing, preserve the detected context and mark the run `INCONCLUSIVE`. If build or validation fails, mark the run `FAIL`.
