---
name: inference-deploy-service
description: Deploy AI inference framework services for xllm, vllm-ascend, and sglang. Use when Codex needs to bind devices, start an inference server, run health checks, capture PID and logs, and produce a deterministic deployment report.
---

# Inference Deploy Service

## Workflow

1. Resolve framework, model, devices, host, and port from the request and `development.yaml`.
2. Read the Deploy section in `frameworks/<framework>.md`.
3. For xllm, let `scripts/launch_xllm.sh` inspect `npu-smi info`, choose idle physical devices when `visible_devices: auto`, and export `ASCEND_RT_VISIBLE_DEVICES`.
4. For xllm, use only `bash scripts/launch_xllm.sh`; do not route xllm startup through `scripts/dev.py`.
5. Let `scripts/launch_xllm.sh` create the deploy run directory and start the service.
6. Save `npu-smi.before.txt`, `visible_devices.txt`, `pids.txt`, and per-node logs in the deploy run directory.
7. Run the documented health check and save output to `healthcheck.log`.
8. Produce `manifest.json` and `report.md`.

## Failure Policy

If the service does not start or health check times out, keep logs and mark the run `INCONCLUSIVE` unless the root cause is confirmed.
