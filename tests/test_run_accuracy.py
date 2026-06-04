from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_accuracy
from run_accuracy import clean_short_answer, default_concurrency, parse_choice


def test_parse_choice_ignores_prose_letters():
    assert parse_choice("The user wants me to answer a multiple-choice question.") == ""


def test_parse_choice_accepts_answer_forms():
    assert parse_choice(" C. normal distribution.") == "C"
    assert parse_choice("Answer: B") == "B"
    assert parse_choice("The final answer is D.") == "D"
    assert parse_choice("(A) continuity") == "A"


def test_clean_short_answer_removes_thinking_tail():
    assert clean_short_answer("1+1 等于 2。\n\n<think>\nThinking Process:") == "1+1 等于 2。"


def test_default_concurrency_uses_one_when_missing():
    assert default_concurrency({}) == 1


def test_default_concurrency_reads_accuracy_config():
    assert default_concurrency({"accuracy": {"concurrency": "4"}}) == 4


def test_sanity_requests_use_configured_concurrency(monkeypatch):
    calls = []

    def fake_text_completion(base_url, model, prompt, max_tokens, timeout):
        calls.append((base_url, model, prompt, max_tokens, timeout))
        return " 2。"

    monkeypatch.setattr(run_accuracy, "text_completion", fake_text_completion)
    args = SimpleNamespace(
        base_url="http://127.0.0.1:18150",
        answer_max_tokens=16,
        request_timeout=60,
        question=None,
        concurrency=3,
    )

    results = run_accuracy.run_sanity_requests(args, "demo-model", "1+1 等于几？")

    assert len(results) == 3
    assert len(calls) == 3
    assert all(item["ok"] for item in results)
