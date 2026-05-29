from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_accuracy import clean_short_answer, parse_choice


def test_parse_choice_ignores_prose_letters():
    assert parse_choice("The user wants me to answer a multiple-choice question.") == ""


def test_parse_choice_accepts_answer_forms():
    assert parse_choice(" C. normal distribution.") == "C"
    assert parse_choice("Answer: B") == "B"
    assert parse_choice("The final answer is D.") == "D"
    assert parse_choice("(A) continuity") == "A"


def test_clean_short_answer_removes_thinking_tail():
    assert clean_short_answer("1+1 等于 2。\n\n<think>\nThinking Process:") == "1+1 等于 2。"
