---
name: inference-framework-router
description: Route multi-framework AI inference infrastructure tasks across xllm, vllm-ascend, and sglang. Use when Codex needs to identify the target framework, locate code under code/, select framework adapter docs, and choose the correct build, deploy, benchmark, accuracy, review, bugfix, or optimization skill.
---

# Inference Framework Router

## Workflow

1. Locate the project root by finding `development.yaml`.
2. Read `development.yaml` for framework names, source paths, and adapter paths.
3. Identify the target framework from the user request, current path, or `current_framework`.
4. Identify the task type and route it to the matching skill.
5. Load `frameworks/<framework>.md` and use the relevant section.

## Task Routing

- Build or install: use `inference-build-install` and the Build section.
- Deploy or serve: use `inference-deploy-service` and the Deploy section.
- Performance or benchmark: use `inference-performance-test` and the Performance section.
- Accuracy or eval: use `inference-accuracy-test` and the Accuracy section.
- Review or PR: use `inference-code-review`.
- Bug or failure: use `inference-bugfix` and read `known-issues.md` if relevant.
- Profiling or optimization: use `inference-performance-optimization`.

## Boundaries

Do not run framework commands directly from this skill. Use this skill to select context and the next workflow. Prefer `scripts/dev.py` for simple project operations.
