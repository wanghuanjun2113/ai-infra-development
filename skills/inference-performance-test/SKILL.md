---
name: inference-performance-test
description: Run deterministic performance benchmarks for xllm, vllm-ascend, and sglang inference services. Use when Codex needs to select devices, run standardized benchmark cases, capture latency/throughput/device metrics, compare baselines, and produce repeatable performance reports.
---

# Inference Performance Test

## Workflow

1. Resolve framework and benchmark case from `development.yaml`.
2. Read the Performance section in `frameworks/<framework>.md`.
3. Create `runs/perf/<run_id>/` with `python3 scripts/dev.py run perf --framework <framework> --case <case>` when possible.
4. Capture `npu-smi info` before and after the run.
5. Save raw benchmark output before parsing.
6. Write `metrics.json` with stable field names.
7. Compare against baseline when configured.
8. Produce `report.md` with `PASS`, `FAIL`, or `INCONCLUSIVE`.

## Metrics

Track throughput, request throughput, TTFT, TPOT, latency percentiles, HBM usage, and AICore utilization.
