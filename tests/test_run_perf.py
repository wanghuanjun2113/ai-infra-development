from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_perf import (
    DEFAULT_LEVELS,
    PerfCase,
    build_prompt,
    format_result_table,
    percentile,
    stream_text_from_chunk,
    summarize_case,
    write_report,
)
from unittest.mock import patch


def test_percentile_interpolates():
    values = [1.0, 2.0, 3.0, 4.0]
    assert percentile(values, 0.50) == 2.5
    assert round(percentile(values, 0.95), 2) == 3.85


def test_build_prompt_scales_with_input_len():
    short_prompt = build_prompt(8, 0)
    long_prompt = build_prompt(128, 0, "qa")
    assert "Continuation:" in short_prompt
    assert "Data profile: qa" in long_prompt
    assert len(long_prompt.split()) > len(short_prompt.split())


def test_build_prompt_reserves_template_space():
    prompt = build_prompt(128, 0)
    assert len(prompt.split()) < 128


def test_default_level_time_limits():
    assert DEFAULT_LEVELS["simple"]["time_limit_seconds"] == 300
    assert DEFAULT_LEVELS["complex"]["time_limit_seconds"] == 1800


def test_simple_level_cases_match_requested_shapes():
    shapes = [(case.input_len, case.output_len, case.concurrency) for case in DEFAULT_LEVELS["simple"]["cases"]]
    assert shapes == [
        (128, 128, 1),
        (128, 128, 16),
        (2048, 1024, 1),
        (2048, 1024, 16),
        (8192, 1024, 1),
        (32768, 1, 1),
        (131072, 1024, 1),
    ]


def test_summarize_case_metrics():
    case = PerfCase("demo", input_len=128, output_len=32, concurrency=2, num_prompts=2, data_profile="technical")
    rows = [
        {
            "status": "ok",
            "latency_seconds": 1.0,
            "e2e_latency_seconds": 1.0,
            "ttft_seconds": 0.2,
            "tpot_seconds": 0.025,
            "prompt_tokens": 128,
            "completion_tokens": 32,
            "total_tokens": 160,
        },
        {
            "status": "ok",
            "latency_seconds": 3.0,
            "e2e_latency_seconds": 3.0,
            "ttft_seconds": 0.4,
            "tpot_seconds": 0.05,
            "prompt_tokens": 128,
            "completion_tokens": 32,
            "total_tokens": 160,
        },
    ]
    summary = summarize_case(case, rows, wall_seconds=4.0)
    assert summary["completed"] == 2
    assert summary["data_profile"] == "technical"
    assert summary["failed"] == 0
    assert summary["request_per_second"] == 0.5
    assert summary["total_tokens_per_second"] == 80.0
    assert summary["latency_p50_seconds"] == 2.0
    assert summary["e2e_latency_avg_seconds"] == 2.0
    assert summary["e2e_latency_p90_seconds"] == 2.8
    assert summary["e2e_latency_p95_seconds"] == 2.9
    assert summary["ttft_avg_seconds"] == 0.3
    assert summary["ttft_p90_seconds"] == 0.38
    assert summary["ttft_p95_seconds"] == 0.39
    assert summary["tpot_sample_count"] == 2
    assert summary["tpot_avg_seconds"] == 0.0375
    assert summary["tpot_p90_seconds"] == 0.0475
    assert summary["tpot_p95_seconds"] == 0.0488
    assert summary["throughput_output_tokens_per_second"] == 16.0


def test_stream_text_from_completion_and_chat_chunks():
    assert stream_text_from_chunk({"choices": [{"text": "hello"}]}) == "hello"
    assert stream_text_from_chunk({"choices": [{"delta": {"content": "world"}}]}) == "world"


def test_empty_stream_response_is_failed():
    from run_perf import run_one_request

    case = PerfCase("empty", input_len=128, output_len=128, concurrency=1, num_prompts=1, data_profile="technical")
    with patch(
        "run_perf.streaming_completion_request",
        return_value={
            "chunks": 0,
            "completion_tokens": 0,
            "e2e_latency_seconds": 0.1,
            "finish_reason": "",
            "output_chars": 0,
            "tpot_seconds": 0.0,
            "ttft_seconds": 0.0,
        },
    ):
        row = run_one_request("http://127.0.0.1:18150", "model", case, 0, 1)
    assert row["status"] == "error"
    assert row["completion_tokens"] == 0
    assert "empty stream response" in row["error"]


def test_result_table_and_report_file(tmp_path):
    summary = {
        "name": "in128_out128_c1",
        "input_len": 128,
        "output_len": 128,
        "concurrency": 1,
        "data_profile": "technical",
        "completed": 1,
        "failed": 0,
        "throughput_requests_per_second": 0.5,
        "throughput_output_tokens_per_second": 64.0,
        "ttft_avg_seconds": 0.1,
        "ttft_p90_seconds": 0.2,
        "ttft_p95_seconds": 0.3,
        "tpot_avg_seconds": 0.01,
        "tpot_p90_seconds": 0.02,
        "tpot_p95_seconds": 0.03,
        "e2e_latency_avg_seconds": 2.0,
        "e2e_latency_p90_seconds": 2.5,
        "e2e_latency_p95_seconds": 2.8,
    }
    table = format_result_table([summary])
    assert "TTFT P95" in table
    assert "in128_out128_c1" in table
    write_report(tmp_path, {"case_summaries": [summary]})
    assert (tmp_path / "result_table.txt").read_text(encoding="utf-8") == table + "\n"
    assert "Terminal Table" in (tmp_path / "report.md").read_text(encoding="utf-8")


def test_result_table_uses_na_without_success():
    summary = {
        "name": "failed_case",
        "input_len": 32768,
        "output_len": 1,
        "concurrency": 1,
        "completed": 0,
        "failed": 1,
        "throughput_requests_per_second": 0.0,
        "throughput_output_tokens_per_second": 0.0,
        "ttft_avg_seconds": 0.0,
        "ttft_p90_seconds": 0.0,
        "ttft_p95_seconds": 0.0,
        "tpot_avg_seconds": 0.0,
        "tpot_p90_seconds": 0.0,
        "tpot_p95_seconds": 0.0,
        "e2e_latency_avg_seconds": 0.0,
        "e2e_latency_p90_seconds": 0.0,
        "e2e_latency_p95_seconds": 0.0,
    }
    table = format_result_table([summary])
    assert "| failed_case | 32768 | 1   | 1    | 0    | 1" in table
    assert " NA " in table


def test_result_table_uses_na_when_tpot_is_not_defined():
    summary = {
        "name": "single_token",
        "input_len": 32768,
        "output_len": 1,
        "concurrency": 1,
        "completed": 1,
        "failed": 0,
        "throughput_requests_per_second": 0.1,
        "throughput_output_tokens_per_second": 0.1,
        "ttft_avg_seconds": 1.0,
        "ttft_p90_seconds": 1.0,
        "ttft_p95_seconds": 1.0,
        "tpot_sample_count": 0,
        "tpot_avg_seconds": 0.0,
        "tpot_p90_seconds": 0.0,
        "tpot_p95_seconds": 0.0,
        "e2e_latency_avg_seconds": 1.0,
        "e2e_latency_p90_seconds": 1.0,
        "e2e_latency_p95_seconds": 1.0,
    }
    table = format_result_table([summary])
    assert "single_token" in table
    assert " NA " in table
