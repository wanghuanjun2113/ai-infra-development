#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
import sys
import tarfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - the repo already depends on PyYAML.
    yaml = None


ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = ROOT / "development.yaml"
MMLU_URL = "https://people.eecs.berkeley.edu/~hendrycks/data.tar"
MMLU_TAR = "mmlu_data.tar"
DOWNLOAD_RETRIES = 8
CHOICES = ("A", "B", "C", "D")
QUICK_SUBJECTS = (
    "high_school_biology",
    "high_school_geography",
    "computer_security",
    "global_facts",
)
FULL_SUBJECTS = (
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "human_aging",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "nutrition",
    "philosophy",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
)


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"config file not found: {path}")
    if yaml is None:
        raise SystemExit("PyYAML is required to read development.yaml")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def resolve_path(value: str) -> Path:
    candidate = Path(value).expanduser()
    return candidate if candidate.is_absolute() else ROOT / candidate


def first_model_name(cfg: dict[str, Any]) -> str:
    accuracy_model = cfg.get("accuracy", {}).get("model")
    if accuracy_model:
        return str(accuracy_model)
    models = cfg.get("models", {})
    if models:
        return str(next(iter(models)))
    return "default"


def default_base_url(cfg: dict[str, Any]) -> str:
    port = cfg.get("deploy", {}).get("xllm", {}).get("start_port", 18150)
    return f"http://127.0.0.1:{port}"


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def create_run_dir(level: str) -> Path:
    target = ROOT / "runs" / "accuracy" / f"{timestamp()}_{level}"
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


def chat_completion(base_url: str, model: str, prompt: str, max_tokens: int, timeout: int) -> str:
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
        "messages": [{"role": "user", "content": prompt}],
    }
    data = request_json(f"{base_url.rstrip('/')}/v1/chat/completions", payload, timeout)
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return json.dumps(data, ensure_ascii=False)
    message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
    content = message.get("content") if isinstance(message, dict) else None
    if content is None and isinstance(choices[0], dict):
        content = choices[0].get("text")
    return str(content or "")


def text_completion(base_url: str, model: str, prompt: str, max_tokens: int, timeout: int) -> str:
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
        "prompt": prompt,
    }
    data = request_json(f"{base_url.rstrip('/')}/v1/completions", payload, timeout)
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return json.dumps(data, ensure_ascii=False)
    choice = choices[0]
    if not isinstance(choice, dict):
        return str(choice)
    return str(choice.get("text") or "")


def has_garbled_text(text: str) -> bool:
    if not text.strip():
        return True
    if "\ufffd" in text:
        return True
    control_count = sum(1 for char in text if ord(char) < 32 and char not in "\n\r\t")
    if control_count:
        return True
    printable_count = sum(1 for char in text if char.isprintable() or char in "\n\r\t")
    return printable_count / max(len(text), 1) < 0.98


def build_sanity_prompt(question: str) -> str:
    return f"{question.rstrip()} 请只回答完整短句，不要解释。\n答案："


def clean_short_answer(text: str) -> str:
    answer = text.strip()
    for marker in ("\n\n", "<think>", "Thinking Process:", "思考过程：", "思考过程:"):
        position = answer.find(marker)
        if position > 0:
            answer = answer[:position].strip()
    return answer


def run_sanity(args: argparse.Namespace, cfg: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    model = get_service_model(args.base_url, args.model or first_model_name(cfg), args.request_timeout)
    question = args.question or "1+1 等于几？"
    print(f"Question: {question}")
    raw_answer = text_completion(
        args.base_url,
        model,
        build_sanity_prompt(question),
        max_tokens=args.answer_max_tokens,
        timeout=args.request_timeout,
    )
    answer = clean_short_answer(raw_answer)
    print(f"Answer: {answer}")
    garbled = has_garbled_text(answer)
    (run_dir / "sanity.json").write_text(
        json.dumps(
            {
                "level": "sanity",
                "model": model,
                "question": question,
                "answer": answer,
                "raw_answer": raw_answer,
                "garbled": garbled,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "status": "PASS" if not garbled else "FAIL",
        "level": "sanity",
        "model": model,
        "question": question,
        "answer": answer,
        "raw_answer": raw_answer,
        "garbled": garbled,
    }


def dataset_dir(cfg: dict[str, Any], dataset: str) -> Path:
    values = cfg.get("datasets", {}).get(dataset)
    if isinstance(values, dict) and values.get("path"):
        return resolve_path(str(values["path"]))
    return ROOT / "datasets" / dataset


def validate_tar(archive: Path, target: Path) -> None:
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            destination = (target / member.name).resolve()
            if not str(destination).startswith(str(target.resolve())):
                raise RuntimeError(f"unsafe path in archive: {member.name}")


def response_total_size(response: Any, existing_size: int) -> int | None:
    content_range = response.headers.get("Content-Range")
    if content_range:
        match = re.search(r"/(\d+)$", content_range)
        if match:
            return int(match.group(1))
    content_length = response.headers.get("Content-Length")
    if content_length and content_length.isdigit():
        return existing_size + int(content_length)
    return None


def download_file(url: str, destination: Path) -> None:
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    tmp.unlink(missing_ok=True)
    last_error: Exception | None = None
    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        try:
            existing_size = tmp.stat().st_size if tmp.exists() else 0
            headers = {"User-Agent": "ai-infra-development/accuracy"}
            if existing_size > 0:
                headers["Range"] = f"bytes={existing_size}-"
            request = urllib.request.Request(url, headers=headers)
            print(
                f"Downloading MMLU dataset to {destination} "
                f"(attempt {attempt}/{DOWNLOAD_RETRIES}, existing={existing_size} bytes)",
                flush=True,
            )
            with urllib.request.urlopen(request, timeout=600) as response:
                if existing_size > 0 and response.status != 206:
                    existing_size = 0
                    tmp.unlink(missing_ok=True)
                total_size = response_total_size(response, existing_size)
                mode = "ab" if existing_size > 0 else "wb"
                with tmp.open(mode) as handle:
                    shutil.copyfileobj(response, handle)
            downloaded_size = tmp.stat().st_size
            if total_size is None or downloaded_size >= total_size:
                tmp.replace(destination)
                return
            last_error = RuntimeError(f"incomplete download: got {downloaded_size}, expected {total_size}")
            print(f"Download incomplete: got {downloaded_size}, expected {total_size}", flush=True)
        except Exception as exc:
            last_error = exc
            print(f"Download failed: {exc}", flush=True)
    raise RuntimeError(f"failed to download {url}: {last_error}")


def ensure_valid_archive(archive: Path, target: Path, force_download: bool) -> None:
    if force_download:
        archive.unlink(missing_ok=True)

    if archive.is_file():
        try:
            validate_tar(archive, target)
            return
        except (tarfile.TarError, EOFError, OSError) as exc:
            print(f"Existing MMLU archive is invalid and will be re-downloaded: {archive} ({exc})", flush=True)
            archive.unlink(missing_ok=True)

    download_file(MMLU_URL, archive)
    try:
        validate_tar(archive, target)
    except (tarfile.TarError, EOFError, OSError) as exc:
        archive.unlink(missing_ok=True)
        raise RuntimeError(f"downloaded MMLU archive is invalid: {archive} ({exc})") from exc


def ensure_mmlu_dataset(cfg: dict[str, Any], force_download: bool = False) -> Path:
    target = dataset_dir(cfg, "mmlu")
    test_dir = target / "test"
    if not force_download and test_dir.is_dir() and any(test_dir.glob("*_test.csv")):
        return target

    target.mkdir(parents=True, exist_ok=True)
    archive = target / MMLU_TAR
    ensure_valid_archive(archive, target, force_download)

    print(f"Extracting MMLU dataset under {target}")
    with tarfile.open(archive) as tar:
        tar.extractall(target)

    nested = target / "data"
    if nested.is_dir():
        for child in nested.iterdir():
            destination = target / child.name
            if destination.exists():
                continue
            child.rename(destination)
        try:
            nested.rmdir()
        except OSError:
            pass

    if not test_dir.is_dir() or not any(test_dir.glob("*_test.csv")):
        raise RuntimeError(f"MMLU test csv files not found after download: {target}")
    return target


def read_mmlu_subject(dataset_root: Path, subject: str) -> list[dict[str, str]]:
    path = dataset_root / "test" / f"{subject}_test.csv"
    if not path.is_file():
        return []
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for index, row in enumerate(reader):
            if len(row) < 6:
                continue
            rows.append(
                {
                    "id": f"{subject}:{index}",
                    "subject": subject,
                    "question": row[0],
                    "A": row[1],
                    "B": row[2],
                    "C": row[3],
                    "D": row[4],
                    "answer": row[5].strip().upper()[:1],
                }
            )
    return rows


def build_prompt(example: dict[str, str]) -> str:
    subject = example["subject"].replace("_", " ")
    return (
        "Answer the following multiple-choice question. "
        "Choose the single best answer.\n\n"
        f"Subject: {subject}\n"
        f"Question: {example['question']}\n"
        f"A. {example['A']}\n"
        f"B. {example['B']}\n"
        f"C. {example['C']}\n"
        f"D. {example['D']}\n"
        "Answer: The correct answer is"
    )


def parse_choice(text: str) -> str:
    upper = text.strip().upper()
    match = re.search(r"(?:CORRECT\s+ANSWER\s+IS|FINAL\s+ANSWER\s+IS)\s*[\(\[]?([ABCD])(?:[\)\].,;:\s]|$)", upper)
    if match:
        return match.group(1)
    match = re.search(r"(?:ANSWER|ANS|答案|OPTION|CHOICE)\s*[:：]\s*[\(\[]?([ABCD])(?:[\)\].,;:\s]|$)", upper)
    if match:
        return match.group(1)
    match = re.match(r"^[\s\"'`*_]*[\(\[]?([ABCD])(?:[\)\].,;:\s]|$)", upper)
    if match:
        return match.group(1)
    for line in reversed([item.strip() for item in upper.splitlines() if item.strip()]):
        match = re.match(r"^(?:FINAL\s+)?(?:ANSWER\s*[:：]\s*)?[\(\[]?([ABCD])(?:[\)\].,;:\s]|$)", line)
        if match:
            return match.group(1)
    return ""


def select_examples(
    dataset_root: Path,
    subjects: tuple[str, ...],
    max_questions: int,
    per_subject: int,
    seed: int,
) -> list[dict[str, str]]:
    rng = random.Random(seed)
    selected: list[dict[str, str]] = []
    for subject in subjects:
        rows = read_mmlu_subject(dataset_root, subject)
        if not rows:
            continue
        rng.shuffle(rows)
        selected.extend(rows[:per_subject])
    rng.shuffle(selected)
    return selected[:max_questions]


def summarize(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(predictions)
    correct = sum(1 for item in predictions if item["correct"])
    by_subject: dict[str, dict[str, Any]] = {}
    for item in predictions:
        subject = item["subject"]
        stats = by_subject.setdefault(subject, {"total": 0, "correct": 0, "accuracy": 0.0})
        stats["total"] += 1
        stats["correct"] += int(item["correct"])
    for stats in by_subject.values():
        stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] else 0.0
    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "by_subject": by_subject,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_mmlu_eval(args: argparse.Namespace, cfg: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    level = args.level
    dataset_root = ensure_mmlu_dataset(cfg, args.force_download)
    model = get_service_model(args.base_url, args.model or first_model_name(cfg), args.request_timeout)

    if level == "quick":
        default_limit = 24
        default_per_subject = 8
        default_time_limit = 120
        subjects = QUICK_SUBJECTS
    else:
        default_limit = 600
        default_per_subject = 16
        default_time_limit = 3600
        subjects = FULL_SUBJECTS

    max_questions = args.max_questions or default_limit
    per_subject = args.per_subject or default_per_subject
    time_limit = args.time_limit_seconds or default_time_limit
    examples = select_examples(dataset_root, subjects, max_questions, per_subject, args.seed)
    if not examples:
        raise RuntimeError(f"no MMLU examples found in {dataset_root}")

    predictions: list[dict[str, Any]] = []
    deadline = time.monotonic() + time_limit
    print(f"Dataset: MMLU")
    print(f"Level: {level}")
    print(f"Model: {model}")
    print(f"Run dir: {run_dir}")
    print(f"Planned questions: {len(examples)}")
    print(f"Time limit seconds: {time_limit}")

    for index, example in enumerate(examples, start=1):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            print(f"Time limit reached before question {index}; stopping early.")
            break
        prompt = build_prompt(example)
        started = time.monotonic()
        timeout = max(1, min(args.request_timeout, int(remaining)))
        raw = text_completion(args.base_url, model, prompt, max_tokens=args.answer_max_tokens, timeout=timeout)
        latency = time.monotonic() - started
        prediction = parse_choice(raw)
        correct = prediction == example["answer"]
        item = {
            "index": index,
            "id": example["id"],
            "dataset": "mmlu",
            "subject": example["subject"],
            "question": example["question"],
            "answer": example["answer"],
            "prediction": prediction,
            "raw_response": raw,
            "correct": correct,
            "latency_seconds": round(latency, 3),
        }
        predictions.append(item)
        print(
            f"[{index}/{len(examples)}] {example['subject']} "
            f"gold={example['answer']} pred={prediction or 'NA'} correct={int(correct)}"
        )

    summary = summarize(predictions)
    write_jsonl(run_dir / "predictions.jsonl", predictions)
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    failed = [item for item in predictions if not item["correct"]]
    write_jsonl(run_dir / "failed.jsonl", failed)

    print(f"Accuracy: {summary['accuracy']:.4f} ({summary['correct']}/{summary['total']})")
    print("Accuracy by subject:")
    for subject, stats in sorted(summary["by_subject"].items()):
        print(f"  {subject}: {stats['accuracy']:.4f} ({stats['correct']}/{stats['total']})")

    return {
        "status": "PASS" if predictions else "INCONCLUSIVE",
        "level": level,
        "dataset": "mmlu",
        "dataset_path": str(dataset_root),
        "model": model,
        "time_limit_seconds": time_limit,
        "planned_questions": len(examples),
        "answered_questions": len(predictions),
        "summary": summary,
        "predictions": str(run_dir / "predictions.jsonl"),
        "failed": str(run_dir / "failed.jsonl"),
    }


def write_report(run_dir: Path, manifest: dict[str, Any]) -> None:
    (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = manifest.get("summary", {})
    lines = [
        "# Accuracy Report",
        "",
        "## Verdict",
        "",
        str(manifest.get("status", "INCONCLUSIVE")),
        "",
        "## Context",
        "",
        f"- Level: {manifest.get('level')}",
        f"- Dataset: {manifest.get('dataset', 'none')}",
        f"- Dataset path: {manifest.get('dataset_path', 'none')}",
        f"- Model: {manifest.get('model')}",
        f"- Run directory: {manifest.get('run_dir')}",
    ]
    if isinstance(summary, dict) and summary:
        lines.extend(
            [
                "",
                "## Accuracy",
                "",
                f"- Total: {summary.get('total', 0)}",
                f"- Correct: {summary.get('correct', 0)}",
                f"- Accuracy: {summary.get('accuracy', 0.0):.4f}",
                "",
                "## By Subject",
                "",
            ]
        )
        for subject, stats in sorted(summary.get("by_subject", {}).items()):
            lines.append(f"- {subject}: {stats['accuracy']:.4f} ({stats['correct']}/{stats['total']})")
    if "question" in manifest:
        lines.extend(["", "## Sanity", "", f"- Question: {manifest['question']}", f"- Answer: {manifest['answer']}"])
    lines.append("")
    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run xllm accuracy checks against an OpenAI-compatible chat endpoint.")
    parser.add_argument("--level", choices=["sanity", "quick", "full"], required=True)
    parser.add_argument("--config", default=str(CONFIG_FILE))
    parser.add_argument("--base-url", default=None, help="Default: deploy.xllm.start_port from development.yaml")
    parser.add_argument("--model", default=None, help="Default: first /v1/models id, then accuracy.model")
    parser.add_argument("--question", default=None, help="Sanity question override")
    parser.add_argument("--request-timeout", type=int, default=60)
    parser.add_argument("--answer-max-tokens", type=int, default=16)
    parser.add_argument("--time-limit-seconds", type=int, default=None)
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--per-subject", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()

    cfg = load_config(Path(args.config).expanduser().resolve())
    args.base_url = args.base_url or default_base_url(cfg)
    run_dir = create_run_dir(args.level)

    try:
        if args.level == "sanity":
            manifest = run_sanity(args, cfg, run_dir)
        else:
            manifest = run_mmlu_eval(args, cfg, run_dir)
    except Exception as exc:
        manifest = {
            "status": "FAIL",
            "level": args.level,
            "error": str(exc),
            "run_dir": str(run_dir),
        }
        write_report(run_dir, manifest)
        print(f"ERROR: {exc}", file=sys.stderr)
        print(f"Run dir: {run_dir}", file=sys.stderr)
        return 1

    manifest["run_dir"] = str(run_dir)
    manifest["base_url"] = args.base_url
    manifest["created_at"] = datetime.now(timezone.utc).isoformat()
    write_report(run_dir, manifest)
    print(f"Report: {run_dir / 'report.md'}")
    return 0 if manifest.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
