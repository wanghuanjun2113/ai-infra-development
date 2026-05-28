---
name: inference-performance-optimization
description: Optimize performance for AI inference frameworks including xllm, vllm-ascend, and sglang. Use when Codex needs to profile latency, throughput, memory, device utilization, scheduler behavior, or operator bottlenecks and validate improvements against baselines.
---

# Inference Performance Optimization

## Workflow

1. Establish a reproducible baseline with `inference-performance-test`.
2. Capture profiling data under `profiling/<run_id>/`.
3. Analyze bottlenecks in scheduling, memory, communication, kernels, and request handling.
4. Apply one focused optimization at a time.
5. Re-run the same benchmark case.
6. Compare metrics and report deltas.

## Output

Produce a short optimization report containing baseline, hypothesis, change, measurement, and residual risk.
