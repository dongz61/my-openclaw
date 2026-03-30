#!/usr/bin/env python3
"""
Run one SWE-bench Lite instance with OpenClaw, end-to-end.

What this script does:
1. Load one instance from SWE-bench Lite by instance_id
2. Clone repo / checkout base_commit into a workspace
3. Create or reuse a dedicated OpenClaw agent bound to that workspace
4. Write the problem statement to a prompt file
5. Invoke OpenClaw on the task
6. Export git diff as patch.diff
7. Build predictions.jsonl for SWE-bench
8. Optionally run SWE-bench evaluation

Designed around the manual workflow that was already verified by hand.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_DATASET = "SWE-bench/SWE-bench_Lite"
DEFAULT_SPLIT = "test"
DEFAULT_MODEL = "modelstudio/qwen3.5-plus"


class CommandError(RuntimeError):
    pass


def run_cmd(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    capture_output: bool = True,
    check: bool = True,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        input=input_text,
        text=True,
        capture_output=capture_output,
    )
    if check and result.returncode != 0:
        raise CommandError(
            "Command failed:\n"
            f"  {' '.join(cmd)}\n"
            f"  cwd={cwd}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return result


def print_step(message: str) -> None:
    print(f"\n=== {message} ===")


def load_instance(dataset_name: str, split: str, instance_id: str) -> dict[str, Any]:
    print_step(f"Loading instance {instance_id} from {dataset_name} [{split}]")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Python package 'datasets' is required. Install it in the current environment first."
        ) from exc

    ds = load_dataset(dataset_name, split=split)
    for item in ds:
        if item["instance_id"] == instance_id:
            return dict(item)
    raise RuntimeError(f"Instance not found: {instance_id}")


def repo_url_from_name(repo_name: str) -> str:
    return f"https://github.com/{repo_name}.git"


def ensure_clean_parent_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def clone_or_reuse_repo(workspace_dir: Path, repo_name: str, base_commit: str, reset: bool) -> None:
    repo_url = repo_url_from_name(repo_name)

    if workspace_dir.exists() and (workspace_dir / ".git").exists():
        print_step(f"Reusing existing workspace: {workspace_dir}")
        if reset:
            print("Reset requested: cleaning existing repository state")
            run_cmd(["git", "fetch", "--all", "--tags"], cwd=workspace_dir)
            run_cmd(["git", "reset", "--hard"], cwd=workspace_dir)
            run_cmd(["git", "clean", "-fdx"], cwd=workspace_dir)
    else:
        if workspace_dir.exists() and not (workspace_dir / ".git").exists():
            raise RuntimeError(
                f"Workspace path exists but is not a git repo: {workspace_dir}"
            )
        print_step(f"Cloning {repo_url} -> {workspace_dir}")
        ensure_clean_parent_dir(workspace_dir.parent)
        run_cmd(["git", "clone", repo_url, str(workspace_dir)])

    print_step(f"Checking out base commit {base_commit}")
    run_cmd(["git", "fetch", "origin"], cwd=workspace_dir)
    run_cmd(["git", "checkout", base_commit], cwd=workspace_dir)


def ensure_openclaw_agent(agent_name: str, workspace_dir: Path, model: str) -> None:
    print_step(f"Ensuring OpenClaw agent exists: {agent_name}")
    result = run_cmd(["openclaw", "agents", "list"], check=False)
    stdout = result.stdout or ""

    if f"- {agent_name}" in stdout or f"Agent: {agent_name}" in stdout:
        print(f"Agent already exists: {agent_name}")
        return

    run_cmd(
        [
            "openclaw",
            "agents",
            "add",
            agent_name,
            "--workspace",
            str(workspace_dir),
            "--model",
            model,
            "--non-interactive",
        ]
    )


def write_prompt_file(prompt_path: Path, problem_statement: str) -> None:
    print_step(f"Writing prompt file: {prompt_path}")
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(problem_statement, encoding="utf-8")


SYSTEM_TASK_PREFIX = """You are solving one SWE-bench Lite task.

Rules:
1. Work only inside the current repository workspace.
2. Read the issue carefully.
3. Inspect the codebase.
4. Make the minimal code changes needed to fix the issue.
5. Add or update tests when appropriate.
6. Run relevant tests if possible.
7. Do NOT commit.
8. At the end, leave the final changes in the working tree so git diff contains the patch.

Issue description:
"""


def invoke_openclaw(agent_name: str, workspace_dir: Path, problem_statement: str) -> str:
    print_step("Invoking OpenClaw")
    prompt = SYSTEM_TASK_PREFIX + problem_statement
    result = run_cmd(
        ["openclaw", "agent", "--agent", agent_name, "--message", prompt],
        cwd=workspace_dir,
    )
    return result.stdout


def export_patch(workspace_dir: Path, patch_path: Path) -> str:
    print_step(f"Exporting patch -> {patch_path}")
    result = run_cmd(["git", "diff"], cwd=workspace_dir)
    patch = result.stdout
    patch_path.parent.mkdir(parents=True, exist_ok=True)
    patch_path.write_text(patch, encoding="utf-8")
    return patch


def ensure_nonempty_patch(patch: str) -> None:
    if not patch.strip():
        raise RuntimeError(
            "git diff is empty. OpenClaw may not have modified the repository."
        )


def write_predictions(
    predictions_path: Path,
    instance_id: str,
    model_name: str,
    patch_text: str,
) -> None:
    print_step(f"Writing predictions file: {predictions_path}")
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "instance_id": instance_id,
        "model_name_or_path": f"openclaw+{model_name}",
        "model_patch": patch_text,
    }
    predictions_path.write_text(
        json.dumps(record, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def run_evaluation(
    dataset_name: str,
    split: str,
    instance_id: str,
    predictions_path: Path,
    run_id: str,
    report_dir: Path,
    max_workers: int,
) -> subprocess.CompletedProcess[str]:
    print_step("Running SWE-bench evaluation")
    report_dir.mkdir(parents=True, exist_ok=True)
    return run_cmd(
        [
            sys.executable,
            "-m",
            "swebench.harness.run_evaluation",
            "--dataset_name",
            dataset_name,
            "--split",
            split,
            "--instance_ids",
            instance_id,
            "--predictions_path",
            str(predictions_path),
            "--max_workers",
            str(max_workers),
            "--run_id",
            run_id,
            "--report_dir",
            str(report_dir),
        ],
        cwd=report_dir,
    )


def sanitize_name(name: str) -> str:
    return name.replace("/", "__")


def build_paths(base_dir: Path, instance_id: str) -> dict[str, Path]:
    workspace_dir = base_dir / "workspaces" / instance_id
    prompt_path = base_dir / "prompts" / f"{instance_id}_prompt.txt"
    patch_path = base_dir / "outputs" / "patches" / f"{instance_id}.patch.diff"
    predictions_path = (
        base_dir / "outputs" / "predictions" / f"{instance_id}.predictions.jsonl"
    )
    report_dir = base_dir / "outputs" / "reports"
    return {
        "workspace_dir": workspace_dir,
        "prompt_path": prompt_path,
        "patch_path": patch_path,
        "predictions_path": predictions_path,
        "report_dir": report_dir,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one SWE-bench Lite instance with OpenClaw")
    parser.add_argument("instance_id", help="SWE-bench instance id, e.g. astropy__astropy-12907")
    parser.add_argument(
        "--base-dir",
        default=str(Path(__file__).resolve().parents[1]),
        help="Base directory for experiment files. Defaults to the parent of scripts/.",
    )
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--agent-name",
        default=None,
        help="OpenClaw agent name. Default: swebench-<instance_id>",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="SWE-bench evaluation max_workers",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Prepare patch and predictions but do not run SWE-bench evaluation.",
    )
    parser.add_argument(
        "--reset-workspace",
        action="store_true",
        help="Hard reset and clean the workspace repo before running.",
    )
    parser.add_argument(
        "--allow-dirty-workspace",
        action="store_true",
        help="Do not fail if the workspace has uncommitted changes before running.",
    )
    return parser.parse_args()


def assert_workspace_clean(workspace_dir: Path) -> None:
    result = run_cmd(["git", "status", "--porcelain"], cwd=workspace_dir)
    if result.stdout.strip():
        raise RuntimeError(
            "Workspace is dirty before run. Use --reset-workspace or clean it manually, "
            "or pass --allow-dirty-workspace if you truly want to continue."
        )


def main() -> int:
    args = parse_args()

    base_dir = Path(args.base_dir).expanduser().resolve()
    instance_id = args.instance_id
    agent_name = args.agent_name or f"swebench-{instance_id}"
    run_id = f"openclaw_{sanitize_name(instance_id)}"

    paths = build_paths(base_dir, instance_id)
    workspace_dir = paths["workspace_dir"]
    prompt_path = paths["prompt_path"]
    patch_path = paths["patch_path"]
    predictions_path = paths["predictions_path"]
    report_dir = paths["report_dir"]

    print_step("Resolved directories")
    print(f"base_dir       : {base_dir}")
    print(f"workspace_dir  : {workspace_dir}")
    print(f"prompt_path    : {prompt_path}")
    print(f"patch_path     : {patch_path}")
    print(f"predictions    : {predictions_path}")
    print(f"report_dir     : {report_dir}")
    print(f"agent_name     : {agent_name}")
    print(f"model          : {args.model}")

    item = load_instance(args.dataset_name, args.split, instance_id)
    repo_name = item["repo"]
    base_commit = item["base_commit"]
    problem_statement = item["problem_statement"]

    print_step("Instance metadata")
    print(f"repo           : {repo_name}")
    print(f"base_commit    : {base_commit}")

    clone_or_reuse_repo(
        workspace_dir=workspace_dir,
        repo_name=repo_name,
        base_commit=base_commit,
        reset=args.reset_workspace,
    )

    if not args.allow_dirty_workspace:
        assert_workspace_clean(workspace_dir)

    ensure_openclaw_agent(agent_name, workspace_dir, args.model)
    write_prompt_file(prompt_path, problem_statement)

    openclaw_output = invoke_openclaw(agent_name, workspace_dir, problem_statement)
    print_step("OpenClaw output")
    print(openclaw_output)

    patch_text = export_patch(workspace_dir, patch_path)
    ensure_nonempty_patch(patch_text)
    write_predictions(predictions_path, instance_id, args.model, patch_text)

    print_step("Artifacts ready")
    print(f"Patch         : {patch_path}")
    print(f"Predictions   : {predictions_path}")

    if not args.skip_eval:
        result = run_evaluation(
            dataset_name=args.dataset_name,
            split=args.split,
            instance_id=instance_id,
            predictions_path=predictions_path,
            run_id=run_id,
            report_dir=report_dir,
            max_workers=args.max_workers,
        )
        print_step("Evaluation output")
        print(result.stdout)
    else:
        print_step("Evaluation skipped")

    print_step("Done")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
