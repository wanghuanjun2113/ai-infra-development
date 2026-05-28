from __future__ import annotations

import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

PATHS = {
    "code": "code",
    "runs": "runs",
    "profiling": "profiling",
    "logs": "logs",
}


def root() -> Path:
    current = Path.cwd().resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "development.yaml").is_file():
            return candidate
    raise FileNotFoundError("development.yaml not found")


def config(project_root: Path | None = None) -> dict[str, Any]:
    project_root = project_root or root()
    with (project_root / "development.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def path(project_root: Path, value: str) -> Path:
    candidate = Path(value).expanduser()
    return candidate if candidate.is_absolute() else project_root / candidate


def enabled_frameworks(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        name: values
        for name, values in cfg.get("frameworks", {}).items()
        if isinstance(values, dict) and values.get("enabled", True)
    }


def framework(cfg: dict[str, Any], name: str | None) -> tuple[str, dict[str, Any]]:
    selected = name or cfg.get("current_framework")
    if not selected:
        raise SystemExit("No framework specified and current_framework is missing")
    values = cfg.get("frameworks", {}).get(selected)
    if not values:
        raise SystemExit(f"Unknown framework: {selected}")
    return selected, values


def git_config(values: dict[str, Any], key: str, default: str | None = None) -> str | None:
    git = values.get("git", {})
    return git.get(key, values.get(key, default))


def model_path(cfg: dict[str, Any], value: str | None) -> str | None:
    if not value:
        return None
    model = cfg.get("models", {}).get(value)
    if isinstance(model, dict):
        return model.get("path")
    return value


def dataset_path(cfg: dict[str, Any], value: str | None) -> str | None:
    if not value:
        return None
    dataset = cfg.get("datasets", {}).get(value)
    if isinstance(dataset, dict):
        return dataset.get("path")
    return value


def draft_model_path(cfg: dict[str, Any], model_name: str | None, explicit: str | None = None) -> str | None:
    if explicit:
        return model_path(cfg, explicit)
    if not model_name:
        return None
    model = cfg.get("models", {}).get(model_name)
    if isinstance(model, dict):
        return model.get("draft_model_path")
    return None


def run(command: list[str], cwd: Path | None = None, dry_run: bool = False) -> int:
    print("$", " ".join(command))
    if dry_run:
        return 0
    return subprocess.call(command, cwd=cwd)


def git_value(repo: Path, args: list[str], default: str = "unknown") -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()
    except Exception:
        return default


def ensure_dirs(project_root: Path, cfg: dict[str, Any]) -> None:
    for key in ["code", "runs", "profiling", "logs"]:
        path(project_root, PATHS[key]).mkdir(parents=True, exist_ok=True)


def collect_env() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cwd": str(Path.cwd()),
        "env": {
            "ASCEND_RT_VISIBLE_DEVICES": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
            "PYTHONPATH": os.environ.get("PYTHONPATH"),
        },
    }


def run_dir(project_root: Path, cfg: dict[str, Any], task: str, name: str, repo: Path) -> Path:
    commit = git_value(repo, ["rev-parse", "--short", "HEAD"])
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{name}_{commit}"
    target = path(project_root, PATHS["runs"]) / task / run_id
    target.mkdir(parents=True, exist_ok=True)
    return target


def write_report(target: Path, title: str, manifest: dict[str, Any]) -> None:
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (target / "env.json").write_text(json.dumps(collect_env(), indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        f"# {title}",
        "",
        "## Verdict",
        "",
        str(manifest.get("status", "INCONCLUSIVE")),
        "",
        "## Context",
        "",
    ]
    for key, value in manifest.items():
        if key != "status":
            lines.append(f"- {key}: {value}")
    lines.append("")
    (target / "report.md").write_text("\n".join(lines), encoding="utf-8")
