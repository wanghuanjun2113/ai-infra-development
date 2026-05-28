#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "shared"))

import devkit


def do_check(args: argparse.Namespace) -> int:
    cfg = devkit.config(ROOT)
    if args.create_dirs:
        devkit.ensure_dirs(ROOT, cfg)
    for key in ["code", "runs", "profiling", "logs"]:
        target = devkit.path(ROOT, devkit.PATHS[key])
        print(f"{key}: {target} {'ok' if target.exists() else 'missing'}")
    for name, values in devkit.enabled_frameworks(cfg).items():
        repo = devkit.path(ROOT, values["path"])
        adapter = devkit.path(ROOT, values["adapter"])
        print(f"{name}: repo={repo} adapter={adapter}")
    for name, values in cfg.get("models", {}).items():
        target = Path(values["path"]).expanduser()
        print(f"model {name}: {target} {'ok' if target.exists() else 'missing'}")
        draft = values.get("draft_model_path")
        if draft:
            draft_path = Path(draft).expanduser()
            print(f"model {name}.draft: {draft_path} {'ok' if draft_path.exists() else 'missing'}")
    for name, values in cfg.get("datasets", {}).items():
        target = Path(values["path"]).expanduser()
        print(f"dataset {name}: {target} {'ok' if target.exists() else 'missing'}")
    return 0


def selected(cfg: dict, name: str) -> dict[str, dict]:
    frameworks = devkit.enabled_frameworks(cfg)
    if name == "all":
        return frameworks
    if name not in frameworks:
        raise SystemExit(f"Unknown or disabled framework: {name}")
    return {name: frameworks[name]}


def do_sync(args: argparse.Namespace) -> int:
    cfg = devkit.config(ROOT)
    for name, values in selected(cfg, args.framework).items():
        repo = devkit.path(ROOT, values["path"])
        if args.action == "status":
            if not repo.exists():
                print(f"{name}: missing ({repo})")
                continue
            branch = devkit.git_value(repo, ["rev-parse", "--abbrev-ref", "HEAD"])
            commit = devkit.git_value(repo, ["rev-parse", "--short", "HEAD"])
            dirty = devkit.git_value(repo, ["status", "--short"], "")
            print(f"{name}: {repo} branch={branch} commit={commit} state={'dirty' if dirty else 'clean'}")
            continue

        if args.action == "clone":
            if repo.exists():
                print(f"{name}: already exists at {repo}")
                continue
            repo.parent.mkdir(parents=True, exist_ok=True)
            origin = devkit.git_config(values, "origin")
            if not origin:
                raise SystemExit(f"{name}: missing git.origin")
            rc = devkit.run(["git", "clone", origin, str(repo)], dry_run=args.dry_run)
            if rc != 0:
                return rc

        if args.action == "pull":
            if not repo.exists():
                print(f"{name}: missing, clone first")
                continue
            rc = devkit.run(["git", "fetch", "--all", "--prune"], cwd=repo, dry_run=args.dry_run)
            if rc != 0:
                return rc

        branch = devkit.git_config(values, "branch")
        if branch and repo.exists():
            rc = devkit.run(["git", "checkout", branch], cwd=repo, dry_run=args.dry_run)
            if rc != 0:
                return rc
            if args.action == "pull":
                rc = devkit.run(["git", "pull", "--ff-only"], cwd=repo, dry_run=args.dry_run)
                if rc != 0:
                    return rc
    return 0


def find_case(cfg: dict, group: str, name: str) -> dict:
    for case in cfg.get(group, {}).get("cases", []):
        if case.get("name") == name:
            return case
    raise SystemExit(f"Unknown {group} case: {name}")


def do_run(args: argparse.Namespace) -> int:
    cfg = devkit.config(ROOT)
    name, values = devkit.framework(cfg, args.framework)
    if args.task == "deploy" and name == "xllm":
        raise SystemExit("Use single-purpose script instead: bash scripts/launch_xllm.sh")
    repo = devkit.path(ROOT, values["path"])
    target = devkit.run_dir(ROOT, cfg, args.task, name, repo)
    adapter = devkit.path(ROOT, values["adapter"])
    manifest = {
        "task": args.task,
        "framework": name,
        "repo": str(repo),
        "branch": devkit.git_value(repo, ["rev-parse", "--abbrev-ref", "HEAD"]),
        "commit": devkit.git_value(repo, ["rev-parse", "--short", "HEAD"]),
        "adapter": str(adapter),
        "run_dir": str(target),
        "status": "INCONCLUSIVE",
    }
    if args.model:
        manifest["model"] = args.model
    elif args.task == "perf":
        manifest["model"] = cfg.get("benchmark", {}).get("model")
    elif args.task == "accuracy":
        manifest["model"] = cfg.get("accuracy", {}).get("model")
    if args.case:
        group = "benchmark" if args.task == "perf" else "accuracy"
        manifest["case"] = find_case(cfg, group, args.case)
    devkit.write_report(target, f"{args.task.title()} Report", manifest)
    print(f"created {target}")
    print(f"read adapter: {adapter}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="ai-infra-development helper")
    sub = parser.add_subparsers(dest="cmd", required=True)

    check = sub.add_parser("check")
    check.add_argument("--create-dirs", action="store_true")
    check.set_defaults(func=do_check)

    sync = sub.add_parser("sync")
    sync.add_argument("action", choices=["clone", "pull", "status"])
    sync.add_argument("--framework", default="all")
    sync.add_argument("--dry-run", action="store_true")
    sync.set_defaults(func=do_sync)

    run = sub.add_parser("run")
    run.add_argument("task", choices=["build", "deploy", "perf", "accuracy"])
    run.add_argument("--framework")
    run.add_argument("--model")
    run.add_argument("--draft-model")
    run.add_argument("--case")
    run.set_defaults(func=do_run)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
