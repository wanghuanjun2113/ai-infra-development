#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - the repo already depends on PyYAML.
    yaml = None


ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = ROOT / "development.yaml"
PROMPT_TEMPLATE_TOKEN_RESERVE = 64


@dataclass(frozen=True)
class PerfCase:
    name: str
    input_len: int
    output_len: int
    concurrency: int
    num_prompts: int
    data_profile: str


DEFAULT_LEVELS: dict[str, dict[str, Any]] = {
    "simple": {
        "time_limit_seconds": 300,
        "cases": [
            PerfCase("in128_out128_c1", input_len=128, output_len=128, concurrency=1, num_prompts=1, data_profile="technical"),
            PerfCase("in128_out128_c16", input_len=128, output_len=128, concurrency=16, num_prompts=16, data_profile="qa"),
            PerfCase("in2k_out1k_c1", input_len=2048, output_len=1024, concurrency=1, num_prompts=1, data_profile="code"),
            PerfCase("in2k_out1k_c16", input_len=2048, output_len=1024, concurrency=16, num_prompts=16, data_profile="mixed"),
            PerfCase("in8k_out1k_c1", input_len=8192, output_len=1024, concurrency=1, num_prompts=1, data_profile="technical"),
            PerfCase("in32k_out1_c1", input_len=32768, output_len=1, concurrency=1, num_prompts=1, data_profile="qa"),
            PerfCase("in128k_out1k_c1", input_len=131072, output_len=1024, concurrency=1, num_prompts=1, data_profile="code"),
        ],
    },
    "complex": {
        "time_limit_seconds": 1800,
        "cases": [
            PerfCase("latency_128_64_c1", input_len=128, output_len=64, concurrency=1, num_prompts=24, data_profile="technical"),
            PerfCase("throughput_512_128_c8", input_len=512, output_len=128, concurrency=8, num_prompts=96, data_profile="qa"),
            PerfCase("long_2048_256_c4", input_len=2048, output_len=256, concurrency=4, num_prompts=48, data_profile="code"),
            PerfCase("stress_1024_128_c16", input_len=1024, output_len=128, concurrency=16, num_prompts=128, data_profile="mixed"),
        ],
    },
}

PROMPT_PROFILES = {
    "technical": (
        "inference",
        "runtime",
        "scheduler",
        "latency",
        "throughput",
        "request",
        "tokens",
        "memory",
        "batch",
        "prefill",
        "decode",
        "service",
        "model",
        "prompt",
        "response",
        "benchmark",
    ),
    "qa": (
        "question",
        "context",
        "evidence",
        "answer",
        "reason",
        "choice",
        "passage",
        "summary",
        "fact",
        "detail",
        "compare",
        "select",
        "explain",
        "support",
        "claim",
        "result",
    ),
    "code": (
        "function",
        "class",
        "variable",
        "return",
        "exception",
        "module",
        "import",
        "thread",
        "future",
        "payload",
        "request",
        "response",
        "parser",
        "metric",
        "assert",
        "report",
    ),
    "mixed": (
        "system",
        "analysis",
        "dataset",
        "operator",
        "cluster",
        "cache",
        "queue",
        "window",
        "profile",
        "sample",
        "policy",
        "engine",
        "route",
        "worker",
        "trace",
        "record",
    ),
}


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"config file not found: {path}")
    if yaml is None:
        raise SystemExit("PyYAML is required to read development.yaml")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def default_base_url(cfg: dict[str, Any]) -> str:
    port = cfg.get("deploy", {}).get("xllm", {}).get("start_port", 18150)
    return f"http://127.0.0.1:{port}"


def first_model_name(cfg: dict[str, Any]) -> str:
    benchmark_model = cfg.get("benchmark", {}).get("model")
    if benchmark_model:
        return str(benchmark_model)
    models = cfg.get("models", {})
    if models:
        return str(next(iter(models)))
    return "default"


def create_run_dir(level: str) -> Path:
    target = ROOT / "runs" / "perf" / f"{timestamp()}_{level}"
    target.mkdir(parents=True, exist_ok=True)
    return target


def request_json(url: str, payload: dict[str, Any] | None, timeout: int, method: str = "GET") -> dict[str, Any]:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
        method = "POST"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"failed to request {url}: {exc}") from exc
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"non-JSON response from {url}: {body[:500]}") from exc


def get_service_model(base_url: str, fallback: str, timeout: int) -> str:
    try:
        data = request_json(f"{base_url.rstrip('/')}/v1/models", None, timeout)
    except Exception:
        return fallback
    models = data.get("data")
    if isinstance(models, list):
        for item in models:
            if isinstance(item, dict) and item.get("id"):
                return str(item["id"])
    return fallback


def build_prompt(input_len: int, index: int, data_profile: str = "technical") -> str:
    profile = PROMPT_PROFILES.get(data_profile, PROMPT_PROFILES["technical"])
    body_len = max(1, input_len - PROMPT_TEMPLATE_TOKEN_RESERVE)
    words: list[str] = []
    for i in range(body_len):
        words.append(profile[(i + index) % len(profile)])
    body = " ".join(words)
    return (
        "Continue the following synthetic benchmark text with concise technical prose. "
        f"Data profile: {data_profile}. Do not solve a problem or explain the benchmark.\n\n"
        f"{body}\n\nContinuation:"
    )


def iter_sse_payloads(response: Any):
    for raw_line in response:
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line or not line.startswith("data:"):
            continue
        payload = line[len("data:") :].strip()
        if payload == "[DONE]":
            break
        yield payload


def stream_text_from_chunk(data: dict[str, Any]) -> str:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    choice = choices[0]
    if not isinstance(choice, dict):
        return ""
    text = choice.get("text")
    if text is None:
        delta = choice.get("delta")
        if isinstance(delta, dict):
            text = delta.get("content")
    return str(text or "")


def streaming_completion_request(
    base_url: str,
    model: str,
    prompt: str,
    output_len: int,
    timeout: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": output_len,
        "stream": True,
        "prompt": prompt,
    }
    started = time.monotonic()
    first_token_at: float | None = None
    output_token_events = 0
    output_chars = 0
    chunks = 0
    finish_reason = ""
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/completions",
        data=data,
        headers={"Accept": "text/event-stream", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for payload_text in iter_sse_payloads(response):
                chunks += 1
                try:
                    chunk = json.loads(payload_text)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices")
                if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                    finish_reason = str(choices[0].get("finish_reason") or finish_reason)
                text = stream_text_from_chunk(chunk)
                if text != "":
                    if first_token_at is None:
                        first_token_at = time.monotonic()
                    output_token_events += 1
                    output_chars += len(text)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from /v1/completions stream: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"failed to stream /v1/completions: {exc}") from exc

    ended = time.monotonic()
    e2e_latency = ended - started
    ttft = (first_token_at - started) if first_token_at is not None else 0.0
    if output_token_events > 1:
        tpot = (e2e_latency - ttft) / (output_token_events - 1)
    else:
        tpot = 0.0
    return {
        "chunks": chunks,
        "completion_tokens": output_token_events,
        "e2e_latency_seconds": e2e_latency,
        "finish_reason": finish_reason,
        "output_chars": output_chars,
        "tpot_seconds": tpot,
        "ttft_seconds": ttft,
    }


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[int(rank)]
    fraction = rank - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def token_usage(data: dict[str, Any]) -> tuple[int, int, int]:
    usage = data.get("usage")
    if not isinstance(usage, dict):
        return 0, 0, 0
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or (prompt_tokens + completion_tokens))
    return prompt_tokens, completion_tokens, total_tokens


def output_size(data: dict[str, Any]) -> int:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return 0
    choice = choices[0]
    if not isinstance(choice, dict):
        return len(str(choice))
    return len(str(choice.get("text") or ""))


def run_one_request(
    base_url: str,
    model: str,
    case: PerfCase,
    request_index: int,
    timeout: int,
) -> dict[str, Any]:
    prompt = build_prompt(case.input_len, request_index, case.data_profile)
    started = time.monotonic()
    row: dict[str, Any] = {
        "case": case.name,
        "request_index": request_index,
        "input_len": case.input_len,
        "output_len": case.output_len,
        "concurrency": case.concurrency,
        "data_profile": case.data_profile,
    }
    try:
        data = streaming_completion_request(base_url, model, prompt, case.output_len, timeout)
        latency = float(data["e2e_latency_seconds"])
        completion_tokens = int(data["completion_tokens"])
        stream_chunks = int(data["chunks"])
        output_chars = int(data["output_chars"])
        if stream_chunks == 0 or completion_tokens == 0:
            raise RuntimeError(
                "empty stream response: "
                f"stream_chunks={stream_chunks}, completion_tokens={completion_tokens}, "
                f"output_chars={output_chars}, finish_reason={data['finish_reason']!r}"
            )
        prompt_tokens = case.input_len
        total_tokens = prompt_tokens + completion_tokens
        row.update(
            {
                "status": "ok",
                "latency_seconds": round(latency, 6),
                "e2e_latency_seconds": round(latency, 6),
                "ttft_seconds": round(float(data["ttft_seconds"]), 6),
                "tpot_seconds": round(float(data["tpot_seconds"]), 6),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "output_chars": output_chars,
                "stream_chunks": stream_chunks,
                "finish_reason": str(data["finish_reason"]),
                "error": "",
            }
        )
    except Exception as exc:
        latency = time.monotonic() - started
        row.update(
            {
                "status": "error",
                "latency_seconds": round(latency, 6),
                "e2e_latency_seconds": round(latency, 6),
                "ttft_seconds": 0.0,
                "tpot_seconds": 0.0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "output_chars": 0,
                "stream_chunks": 0,
                "finish_reason": "",
                "error": str(exc),
            }
        )
    return row


def summarize_case(case: PerfCase, rows: list[dict[str, Any]], wall_seconds: float) -> dict[str, Any]:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    failed = len(rows) - len(ok_rows)
    e2e_latencies = [float(row.get("e2e_latency_seconds") or row["latency_seconds"]) for row in ok_rows]
    ttfts = [float(row.get("ttft_seconds") or 0.0) for row in ok_rows]
    tpots = [float(row.get("tpot_seconds") or 0.0) for row in ok_rows if float(row.get("tpot_seconds") or 0.0) > 0]
    prompt_tokens = sum(int(row.get("prompt_tokens") or 0) for row in ok_rows)
    completion_tokens = sum(int(row.get("completion_tokens") or 0) for row in ok_rows)
    total_tokens = sum(int(row.get("total_tokens") or 0) for row in ok_rows)
    wall = max(wall_seconds, 1e-9)
    return {
        "name": case.name,
        "input_len": case.input_len,
        "output_len": case.output_len,
        "concurrency": case.concurrency,
        "num_prompts": case.num_prompts,
        "data_profile": case.data_profile,
        "completed": len(ok_rows),
        "failed": failed,
        "wall_seconds": round(wall_seconds, 3),
        "request_per_second": round(len(ok_rows) / wall, 4),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "prompt_tokens_per_second": round(prompt_tokens / wall, 3),
        "completion_tokens_per_second": round(completion_tokens / wall, 3),
        "total_tokens_per_second": round(total_tokens / wall, 3),
        "throughput_requests_per_second": round(len(ok_rows) / wall, 4),
        "throughput_output_tokens_per_second": round(completion_tokens / wall, 3),
        "throughput_total_tokens_per_second": round(total_tokens / wall, 3),
        "e2e_latency_avg_seconds": round(sum(e2e_latencies) / len(e2e_latencies), 4) if e2e_latencies else 0.0,
        "e2e_latency_p50_seconds": round(percentile(e2e_latencies, 0.50), 4),
        "e2e_latency_p90_seconds": round(percentile(e2e_latencies, 0.90), 4),
        "e2e_latency_p95_seconds": round(percentile(e2e_latencies, 0.95), 4),
        "e2e_latency_p99_seconds": round(percentile(e2e_latencies, 0.99), 4),
        "ttft_avg_seconds": round(sum(ttfts) / len(ttfts), 4) if ttfts else 0.0,
        "ttft_p50_seconds": round(percentile(ttfts, 0.50), 4),
        "ttft_p90_seconds": round(percentile(ttfts, 0.90), 4),
        "ttft_p95_seconds": round(percentile(ttfts, 0.95), 4),
        "ttft_p99_seconds": round(percentile(ttfts, 0.99), 4),
        "tpot_sample_count": len(tpots),
        "tpot_avg_seconds": round(sum(tpots) / len(tpots), 4) if tpots else 0.0,
        "tpot_p50_seconds": round(percentile(tpots, 0.50), 4),
        "tpot_p90_seconds": round(percentile(tpots, 0.90), 4),
        "tpot_p95_seconds": round(percentile(tpots, 0.95), 4),
        "tpot_p99_seconds": round(percentile(tpots, 0.99), 4),
        "latency_avg_seconds": round(sum(e2e_latencies) / len(e2e_latencies), 4) if e2e_latencies else 0.0,
        "latency_p50_seconds": round(percentile(e2e_latencies, 0.50), 4),
        "latency_p90_seconds": round(percentile(e2e_latencies, 0.90), 4),
        "latency_p95_seconds": round(percentile(e2e_latencies, 0.95), 4),
        "latency_p99_seconds": round(percentile(e2e_latencies, 0.99), 4),
    }


def run_case(
    base_url: str,
    model: str,
    case: PerfCase,
    request_timeout: int,
    deadline: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if time.monotonic() >= deadline:
        return [], summarize_case(case, [], 0.0)

    rows: list[dict[str, Any]] = []
    started = time.monotonic()
    next_request = 0
    futures: set[concurrent.futures.Future[dict[str, Any]]] = set()

    with concurrent.futures.ThreadPoolExecutor(max_workers=case.concurrency) as executor:
        while next_request < case.num_prompts or futures:
            while (
                next_request < case.num_prompts
                and len(futures) < case.concurrency
                and time.monotonic() < deadline
            ):
                remaining_timeout = max(1, min(request_timeout, int(deadline - time.monotonic())))
                future = executor.submit(
                    run_one_request,
                    base_url,
                    model,
                    case,
                    next_request,
                    remaining_timeout,
                )
                futures.add(future)
                next_request += 1

            if not futures:
                break
            done, futures = concurrent.futures.wait(
                futures,
                timeout=1,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in done:
                rows.append(future.result())

            if time.monotonic() >= deadline and not futures:
                break

    wall_seconds = time.monotonic() - started
    return rows, summarize_case(case, rows, wall_seconds)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def snapshot_npu_smi(path: Path) -> None:
    try:
        result = subprocess.run(
            ["npu-smi", "info"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=30,
        )
        path.write_text(result.stdout, encoding="utf-8")
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        path.write_text(f"npu-smi info unavailable: {exc}\n", encoding="utf-8")


def level_cases(level: str) -> list[PerfCase]:
    return list(DEFAULT_LEVELS[level]["cases"])


def format_result_table(summaries: list[dict[str, Any]]) -> str:
    headers = [
        "Case",
        "In",
        "Out",
        "Conc",
        "Done",
        "Fail",
        "RPS",
        "OutTok/s",
        "TTFT avg",
        "TTFT P90",
        "TTFT P95",
        "TPOT avg",
        "TPOT P90",
        "TPOT P95",
        "E2E avg",
        "E2E P90",
        "E2E P95",
    ]
    rows = []
    for item in summaries:
        has_success = int(item["completed"]) > 0

        def metric(key: str, digits: int) -> str:
            if not has_success:
                return "NA"
            return f"{item[key]:.{digits}f}"

        def tpot_metric(key: str) -> str:
            if not has_success or int(item.get("tpot_sample_count") or 0) == 0:
                return "NA"
            return f"{item[key]:.4f}"

        rows.append(
            [
                str(item["name"]),
                str(item["input_len"]),
                str(item["output_len"]),
                str(item["concurrency"]),
                str(item["completed"]),
                str(item["failed"]),
                metric("throughput_requests_per_second", 4),
                metric("throughput_output_tokens_per_second", 3),
                metric("ttft_avg_seconds", 4),
                metric("ttft_p90_seconds", 4),
                metric("ttft_p95_seconds", 4),
                tpot_metric("tpot_avg_seconds"),
                tpot_metric("tpot_p90_seconds"),
                tpot_metric("tpot_p95_seconds"),
                metric("e2e_latency_avg_seconds", 4),
                metric("e2e_latency_p90_seconds", 4),
                metric("e2e_latency_p95_seconds", 4),
            ]
        )
    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row)]

    def line(left: str, fill: str, sep: str, right: str) -> str:
        return left + sep.join(fill * (width + 2) for width in widths) + right

    def render_row(cells: list[str]) -> str:
        return "| " + " | ".join(cell.ljust(width) for cell, width in zip(cells, widths)) + " |"

    output = [line("+", "-", "+", "+"), render_row(headers), line("+", "-", "+", "+")]
    output.extend(render_row(row) for row in rows)
    output.append(line("+", "-", "+", "+"))
    return "\n".join(output)


def write_report(run_dir: Path, manifest: dict[str, Any]) -> None:
    summaries = manifest.get("case_summaries", [])
    result_table = format_result_table(summaries)
    lines = [
        "# Performance Report",
        "",
        "## Verdict",
        "",
        str(manifest.get("status", "INCONCLUSIVE")),
        "",
        "## Context",
        "",
        f"- Level: {manifest.get('level')}",
        f"- Model: {manifest.get('model')}",
        f"- Base URL: {manifest.get('base_url')}",
        f"- Run directory: {manifest.get('run_dir')}",
        f"- Time limit seconds: {manifest.get('time_limit_seconds')}",
        f"- Wall seconds: {manifest.get('wall_seconds')}",
        f"- Planned requests: {manifest.get('total_planned')}",
        f"- Completed requests: {manifest.get('total_completed')}",
        f"- Failed requests: {manifest.get('total_failed')}",
        f"- Time limited: {manifest.get('time_limited')}",
        "",
        "## Summary",
        "",
        "| Case | Data | In | Out | Conc | Done | Fail | RPS | Out tok/s | TTFT avg | TTFT P90 | TTFT P95 | TPOT avg | TPOT P90 | TPOT P95 | E2E avg | E2E P90 | E2E P95 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        lines.append(
            "| {name} | {data_profile} | {input_len} | {output_len} | {concurrency} | {completed} | {failed} | "
            "{throughput_requests_per_second:.4f} | {throughput_output_tokens_per_second:.3f} | "
            "{ttft_avg_seconds:.4f} | {ttft_p90_seconds:.4f} | {ttft_p95_seconds:.4f} | "
            "{tpot_avg_seconds:.4f} | {tpot_p90_seconds:.4f} | {tpot_p95_seconds:.4f} | "
            "{e2e_latency_avg_seconds:.4f} | {e2e_latency_p90_seconds:.4f} | {e2e_latency_p95_seconds:.4f} |".format(
                **item
            )
        )
    lines.extend(
        [
            "",
            "## Terminal Table",
            "",
            "```text",
            result_table,
            "```",
            "",
            "## Artifacts",
            "",
            f"- Manifest: {run_dir / 'manifest.json'}",
            f"- Summary: {run_dir / 'summary.json'}",
            f"- Result table: {run_dir / 'result_table.txt'}",
            f"- Raw results: {run_dir / 'results.jsonl'}",
            f"- Failed requests: {run_dir / 'failed.jsonl'}",
            f"- NPU before: {run_dir / 'npu_smi_before.txt'}",
            f"- NPU after: {run_dir / 'npu_smi_after.txt'}",
            "",
        ]
    )
    (run_dir / "result_table.txt").write_text(result_table + "\n", encoding="utf-8")
    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run_perf(args: argparse.Namespace, cfg: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    model = get_service_model(args.base_url, args.model or first_model_name(cfg), args.request_timeout)
    cases = level_cases(args.level)
    if args.case:
        cases = [case for case in cases if case.name == args.case]
        if not cases:
            raise RuntimeError(f"unknown case for level {args.level}: {args.case}")
    time_limit = args.time_limit_seconds or int(DEFAULT_LEVELS[args.level]["time_limit_seconds"])
    deadline = time.monotonic() + time_limit
    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    planned_requests = sum(case.num_prompts for case in cases)

    snapshot_npu_smi(run_dir / "npu_smi_before.txt")
    started = time.monotonic()
    print(f"Level: {args.level}")
    print(f"Model: {model}")
    print(f"Run dir: {run_dir}")
    print(f"Time limit seconds: {time_limit}")
    print(f"Cases: {', '.join(case.name for case in cases)}")

    for case in cases:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            print(f"Time limit reached before case {case.name}; stopping.")
            break
        print(
            f"Running {case.name}: input_len={case.input_len} output_len={case.output_len} "
            f"concurrency={case.concurrency} num_prompts={case.num_prompts}"
        )
        rows, summary = run_case(args.base_url, model, case, args.request_timeout, deadline)
        all_rows.extend(rows)
        summaries.append(summary)
        print(
            f"Finished {case.name}: done={summary['completed']} failed={summary['failed']} "
            f"rps={summary['throughput_requests_per_second']} "
            f"out_tok/s={summary['throughput_output_tokens_per_second']} "
            f"ttft_p95={summary['ttft_p95_seconds']}s "
            f"tpot_p95={summary['tpot_p95_seconds']}s "
            f"e2e_p95={summary['e2e_latency_p95_seconds']}s"
        )

    wall_seconds = time.monotonic() - started
    snapshot_npu_smi(run_dir / "npu_smi_after.txt")
    failed_rows = [row for row in all_rows if row.get("status") != "ok"]
    write_jsonl(run_dir / "results.jsonl", all_rows)
    write_jsonl(run_dir / "failed.jsonl", failed_rows)
    summary_doc = {
        "level": args.level,
        "model": model,
        "time_limit_seconds": time_limit,
        "wall_seconds": round(wall_seconds, 3),
        "case_summaries": summaries,
        "total_planned": planned_requests,
        "total_completed": sum(int(item["completed"]) for item in summaries),
        "total_failed": sum(int(item["failed"]) for item in summaries),
    }
    observed_requests = summary_doc["total_completed"] + summary_doc["total_failed"]
    time_limited = observed_requests < planned_requests
    summary_doc["time_limited"] = time_limited
    (run_dir / "summary.json").write_text(json.dumps(summary_doc, ensure_ascii=False, indent=2), encoding="utf-8")

    status = "PASS" if summary_doc["total_completed"] > 0 and summary_doc["total_failed"] == 0 and not time_limited else "FAIL"
    print("")
    print("Performance result table:")
    print(format_result_table(summaries))
    return {
        "status": status,
        "level": args.level,
        "model": model,
        "base_url": args.base_url,
        "time_limit_seconds": time_limit,
        "wall_seconds": round(wall_seconds, 3),
        "case_summaries": summaries,
        "total_planned": planned_requests,
        "total_completed": summary_doc["total_completed"],
        "total_failed": summary_doc["total_failed"],
        "time_limited": time_limited,
        "results": str(run_dir / "results.jsonl"),
        "failed": str(run_dir / "failed.jsonl"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run xllm performance checks against an OpenAI-compatible endpoint.")
    parser.add_argument("--level", choices=["simple", "complex"], required=True)
    parser.add_argument("--config", default=str(CONFIG_FILE))
    parser.add_argument("--base-url", default=None, help="Default: deploy.xllm.start_port from development.yaml")
    parser.add_argument("--model", default=None, help="Default: first /v1/models id, then benchmark.model")
    parser.add_argument("--case", default=None, help="Run only one named case from the selected level")
    parser.add_argument("--request-timeout", type=int, default=180)
    parser.add_argument("--time-limit-seconds", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(Path(args.config).expanduser().resolve())
    args.base_url = args.base_url or default_base_url(cfg)
    run_dir = create_run_dir(args.level)

    try:
        manifest = run_perf(args, cfg, run_dir)
    except Exception as exc:
        manifest = {
            "status": "FAIL",
            "level": args.level,
            "error": str(exc),
            "run_dir": str(run_dir),
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        write_report(run_dir, manifest)
        print(f"ERROR: {exc}", file=sys.stderr)
        print(f"Run dir: {run_dir}", file=sys.stderr)
        return 1

    manifest["run_dir"] = str(run_dir)
    manifest["created_at"] = datetime.now(timezone.utc).isoformat()
    (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(run_dir, manifest)
    print(f"Report: {run_dir / 'report.md'}")
    return 0 if manifest.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
