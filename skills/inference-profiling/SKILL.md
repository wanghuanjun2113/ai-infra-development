---
name: inference-profiling
description: Capture profiling traces for AI inference services, especially xllm on Ascend NPU with msprof dynamic attach. Use when Codex needs to collect PROF artifacts, separate warmup and measured workload windows, preserve profiling evidence, or prepare bottleneck analysis before optimization.
---

# Inference Profiling

## Workflow

1. Resolve the framework, service URL, model, workload, and run directory from the user request and `development.yaml`.
2. For xllm, read the Profiling section in `frameworks/xllm.md`.
3. Start xllm only with `bash scripts/launch_xllm.sh`; do not start xllm from profiling scripts.
4. Confirm the service is healthy and identify the xllm parent PID from `runs/deploy/<run_id>/pids.txt` or `ps -ef`.
5. Run warmup outside the profiling window when the request is sensitive to model loading, graph warmup, or first-request compilation.
6. Capture the measured window with `bash scripts/capture_xllm_profile.sh --pid <pid> --workload-cmd '<command>'`.
7. Store raw PROF directories, exported MindStudio files, logs, manifest, and device snapshots under `profiling/<run_id>/`.
8. Summarize what was captured: framework commit, model, graph mode, workload shape, PID, PROF directory, and any missing artifacts.
9. If this task started xllm, stop it before finishing unless the user explicitly asked to keep it running.

## Rules

- Match the profiling workload to the slow benchmark case being investigated.
- Keep prefill-focused and decode-focused captures separate when their bottlenecks may differ.
- Do not mix warmup, model loading, service health checks, and measured requests in the same profiling window.
- Treat missing `PROF_*`, missing `mindstudio_profiler_output/`, or failed workload commands as inconclusive evidence.
