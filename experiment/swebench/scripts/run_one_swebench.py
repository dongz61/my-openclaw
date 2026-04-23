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
from uuid import uuid4


DEFAULT_DATASET = "SWE-bench/SWE-bench_Lite"
DEFAULT_SPLIT = "test"
DEFAULT_MODEL = "modelstudio/qwen3.5-plus"
DEFAULT_EXPERIMENT_TAG = "baseline"
DEFAULT_EXTRA_SYSTEM_PROMPT_FILE = (
    Path(__file__).resolve().parents[1]
    / "prompts"
    / "system"
    / "swebench_code_repair_system_prompt.txt"
)


class CommandError(RuntimeError):
    pass


def run_cmd(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    capture_output: bool = True,
    check: bool = True,
    input_text: str | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        input=input_text,
        text=True,
        capture_output=capture_output,
        env=env,
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


def load_dataset_split(dataset_name: str, split: str):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Python package 'datasets' is required. Install it in the current environment first."
        ) from exc
    return load_dataset(dataset_name, split=split)


def load_instance(dataset_name: str, split: str, instance_id: str) -> dict[str, Any]:
    print_step(f"Loading instance {instance_id} from {dataset_name} [{split}]")
    ds = load_dataset_split(dataset_name, split)
    for item in ds:
        if item["instance_id"] == instance_id:
            return dict(item)
    raise RuntimeError(f"Instance not found: {instance_id}")


def load_instance_by_index(dataset_name: str, split: str, index: int) -> dict[str, Any]:
    print_step(f"Loading dataset index {index} from {dataset_name} [{split}]")
    ds = load_dataset_split(dataset_name, split)
    if index < 0 or index >= len(ds):
        raise RuntimeError(f"Index out of range: {index}. Dataset length is {len(ds)}")
    return dict(ds[index])


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


def write_prompt_file(prompt_path: Path, problem_statement: str) -> None:
    print_step(f"Writing prompt file: {prompt_path}")
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(SYSTEM_TASK_PREFIX + problem_statement, encoding="utf-8")


def write_text_file(path: Path, content: str) -> None:
    print_step(f"Writing prompt file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def invoke_openclaw(
    *,
    agent_name: str,
    session_id: str,
    session_key: str,
    workspace_dir: Path,
    prompt_path: Path,
    extra_system_prompt_file: Path | None,
    manifest_path: Path,
    events_path: Path,
    summary_path: Path,
    stdout_path: Path,
    final_path: Path,
    instance_id: str,
    dataset_name: str,
    split: str,
    run_protocol: str,
    system_prompt_file: Path | None = None,
    tool_allow: list[str] | None = None,
    tool_deny: list[str] | None = None,
    exec_env_guard: bool = False,
) -> str:
    print_step("Invoking OpenClaw")
    if extra_system_prompt_file and system_prompt_file:
        raise RuntimeError("Pass only one of extra_system_prompt_file or system_prompt_file.")
    script_path = Path(__file__).resolve().parent / "run_openclaw_trace.ts"
    env = None
    if exec_env_guard:
        env = os.environ.copy()
        env["OPENCLAW_SWEBENCH_EXEC_ENV_GUARD"] = "1"
    result = run_cmd(
        [
            "node",
            "--import",
            "tsx",
            str(script_path),
            "--agent-id",
            agent_name,
            "--session-id",
            session_id,
            "--session-key",
            session_key,
            "--workspace-dir",
            str(workspace_dir),
            "--prompt-file",
            str(prompt_path),
            *(
                [
                    "--extra-system-prompt-file",
                    str(extra_system_prompt_file),
                ]
                if extra_system_prompt_file
                else []
            ),
            *(
                [
                    "--system-prompt-file",
                    str(system_prompt_file),
                ]
                if system_prompt_file
                else []
            ),
            *(
                [
                    "--tool-allow",
                    ",".join(tool_allow),
                ]
                if tool_allow
                else []
            ),
            *(
                [
                    "--tool-deny",
                    ",".join(tool_deny),
                ]
                if tool_deny
                else []
            ),
            "--manifest-path",
            str(manifest_path),
            "--events-path",
            str(events_path),
            "--summary-path",
            str(summary_path),
            "--stdout-path",
            str(stdout_path),
            "--final-path",
            str(final_path),
            "--instance-id",
            instance_id,
            "--dataset-name",
            dataset_name,
            "--split",
            split,
            "--run-protocol",
            run_protocol,
        ],
        cwd=workspace_dir,
        env=env,
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
    cache_level: str = "instance",
) -> subprocess.CompletedProcess[str]:
    print_step("Running SWE-bench evaluation")
    report_dir.mkdir(parents=True, exist_ok=True)
    eval_log_dir = report_dir / "logs" / "run_evaluation" / run_id
    if eval_log_dir.exists():
        print(f"Removing stale evaluation logs: {eval_log_dir}")
        shutil.rmtree(eval_log_dir)
    for stale_report in report_dir.glob(f"*.{run_id}.json"):
        print(f"Removing stale evaluation report: {stale_report}")
        stale_report.unlink()
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
            "--cache_level",
            cache_level,
        ],
        cwd=report_dir,
    )


def sanitize_name(name: str) -> str:
    return name.replace("/", "__")


def sanitize_tag(value: str) -> str:
    normalized = "".join(
        char if char.isalnum() or char in {"-", "_", "."} else "-"
        for char in value.strip()
    ).strip(".-")
    if not normalized:
        raise RuntimeError("experiment_tag is empty after sanitization")
    return normalized


def resolve_experiment_dir(base_dir: Path, experiment_tag: str) -> Path:
    if experiment_tag == "baseline":
        return base_dir / "outputs_baseline"
    return base_dir / f"outputs_{experiment_tag}"


def build_run_session_values(
    *,
    agent_name: str,
    instance_id: str,
    experiment_tag: str,
    stage: str,
    run_slug: str,
) -> dict[str, str]:
    instance_slug = sanitize_name(instance_id)
    stage_slug = sanitize_tag(stage)
    return {
        "session_id": f"swebench-{instance_slug}-{experiment_tag}-{stage_slug}-{run_slug}",
        "session_key": (
            f"agent:{agent_name}:swebench:{experiment_tag}:{instance_slug}:"
            f"{stage_slug}:{run_slug}"
        ),
    }


def build_paths(base_dir: Path, instance_id: str, experiment_tag: str) -> dict[str, Path]:
    experiment_dir = resolve_experiment_dir(base_dir, experiment_tag)
    workspace_dir = base_dir / "workspaces" / instance_id
    prompt_path = base_dir / "prompts" / f"{instance_id}_prompt.txt"
    patch_path = experiment_dir / "patches" / f"{instance_id}.patch.diff"
    predictions_path = experiment_dir / "predictions" / f"{instance_id}.predictions.jsonl"
    report_dir = experiment_dir / "reports"
    run_dir = experiment_dir / "runs" / instance_id
    return {
        "experiment_dir": experiment_dir,
        "workspace_dir": workspace_dir,
        "prompt_path": prompt_path,
        "patch_path": patch_path,
        "predictions_path": predictions_path,
        "report_dir": report_dir,
        "run_dir": run_dir,
        "manifest_path": run_dir / "manifest.json",
        "events_path": run_dir / "events.jsonl",
        "summary_path": run_dir / "summary.json",
        "stdout_path": run_dir / "stdout.txt",
        "final_path": run_dir / "final.json",
        "timeline_path": run_dir / "timeline.json",
        "calls_path": run_dir / "calls.json",
        "verification_path": run_dir / "verification.json",
        "run_report_path": run_dir / "run_report.md",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one SWE-bench Lite instance with OpenClaw")
    parser.add_argument(
        "instance_id",
        nargs="?",
        help="SWE-bench instance id, e.g. astropy__astropy-12907",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="Dataset index to resolve into an instance_id, e.g. --index 0",
    )
    parser.add_argument(
        "--base-dir",
        default=str(Path(__file__).resolve().parents[1]),
        help="Base directory for experiment files. Defaults to the parent of scripts/.",
    )
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--experiment-tag",
        default=DEFAULT_EXPERIMENT_TAG,
        help=(
            "Output namespace under experiment/swebench/outputs/. "
            "Use a new tag for each experiment to avoid overwriting prior results."
        ),
    )
    parser.add_argument(
        "--extra-system-prompt-file",
        default=str(DEFAULT_EXTRA_SYSTEM_PROMPT_FILE),
        help=(
            "Optional extra system prompt file injected into OpenClaw for SWE-bench runs. "
            "Pass an empty string to disable."
        ),
    )
    parser.add_argument(
        "--system-prompt-file",
        default="",
        help=(
            "Optional complete system prompt replacement for SWE-bench runs. "
            "Disables the default extra system prompt unless --extra-system-prompt-file "
            "is explicitly provided, which remains invalid."
        ),
    )
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
        "--eval-cache-level",
        choices=["none", "base", "env", "instance"],
        default="instance",
        help="SWE-bench image cache level for evaluation cleanup. Defaults to instance.",
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

    if (args.instance_id is None) == (args.index is None):
        raise RuntimeError("Pass exactly one of: instance_id or --index")

    base_dir = Path(args.base_dir).expanduser().resolve()
    experiment_tag = sanitize_tag(args.experiment_tag)

    if args.index is not None:
        item = load_instance_by_index(args.dataset_name, args.split, args.index)
        instance_id = item["instance_id"]
        print_step("Resolved dataset index")
        print(f"index          : {args.index}")
        print(f"instance_id    : {instance_id}")
    else:
        instance_id = args.instance_id
        item = load_instance(args.dataset_name, args.split, instance_id)

    agent_name = args.agent_name or f"swebench-{instance_id}"
    run_slug = uuid4().hex
    eval_run_id = f"openclaw_{sanitize_name(instance_id)}"
    session_values = build_run_session_values(
        agent_name=agent_name,
        instance_id=instance_id,
        experiment_tag=experiment_tag,
        stage="single",
        run_slug=run_slug,
    )

    paths = build_paths(base_dir, instance_id, experiment_tag)
    experiment_dir = paths["experiment_dir"]
    workspace_dir = paths["workspace_dir"]
    prompt_path = paths["prompt_path"]
    patch_path = paths["patch_path"]
    predictions_path = paths["predictions_path"]
    report_dir = paths["report_dir"]
    manifest_path = paths["manifest_path"]
    events_path = paths["events_path"]
    summary_path = paths["summary_path"]
    stdout_path = paths["stdout_path"]
    final_path = paths["final_path"]
    timeline_path = paths["timeline_path"]
    calls_path = paths["calls_path"]
    verification_path = paths["verification_path"]
    run_report_path = paths["run_report_path"]
    extra_system_prompt_arg = args.extra_system_prompt_file
    if (
        args.system_prompt_file
        and "--extra-system-prompt-file" not in sys.argv
        and extra_system_prompt_arg == str(DEFAULT_EXTRA_SYSTEM_PROMPT_FILE)
    ):
        extra_system_prompt_arg = ""

    extra_system_prompt_file = (
        Path(extra_system_prompt_arg).expanduser().resolve()
        if extra_system_prompt_arg
        else None
    )
    system_prompt_file = (
        Path(args.system_prompt_file).expanduser().resolve()
        if args.system_prompt_file
        else None
    )
    if extra_system_prompt_file and system_prompt_file:
        raise RuntimeError("Pass only one of --extra-system-prompt-file or --system-prompt-file")

    print_step("Resolved directories")
    print(f"base_dir       : {base_dir}")
    print(f"experiment_tag : {experiment_tag}")
    print(f"experiment_dir : {experiment_dir}")
    print(f"workspace_dir  : {workspace_dir}")
    print(f"prompt_path    : {prompt_path}")
    print(f"patch_path     : {patch_path}")
    print(f"predictions    : {predictions_path}")
    print(f"report_dir     : {report_dir}")
    print(f"agent_name     : {agent_name}")
    print(f"run_slug       : {run_slug}")
    print(f"eval_run_id    : {eval_run_id}")
    print(f"session_id     : {session_values['session_id']}")
    print(f"session_key    : {session_values['session_key']}")
    print(f"model          : {args.model}")
    print(f"extra_system   : {extra_system_prompt_file or '(disabled)'}")
    print(f"system_prompt  : {system_prompt_file or '(generated)'}")

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

    openclaw_output = invoke_openclaw(
        agent_name=agent_name,
        session_id=session_values["session_id"],
        session_key=session_values["session_key"],
        workspace_dir=workspace_dir,
        prompt_path=prompt_path,
        extra_system_prompt_file=extra_system_prompt_file,
        system_prompt_file=system_prompt_file,
        manifest_path=manifest_path,
        events_path=events_path,
        summary_path=summary_path,
        stdout_path=stdout_path,
        final_path=final_path,
        instance_id=instance_id,
        dataset_name=args.dataset_name,
        split=args.split,
        run_protocol="single_pass_no_retry",
    )
    print_step("OpenClaw output")
    print(openclaw_output)

    patch_text = export_patch(workspace_dir, patch_path)
    ensure_nonempty_patch(patch_text)
    write_predictions(predictions_path, instance_id, args.model, patch_text)

    print_step("Artifacts ready")
    print(f"Patch         : {patch_path}")
    print(f"Predictions   : {predictions_path}")
    print(f"Manifest      : {manifest_path}")
    print(f"Events        : {events_path}")
    print(f"Summary       : {summary_path}")
    print(f"Final         : {final_path}")
    print(f"Timeline      : {timeline_path}")
    print(f"Calls         : {calls_path}")
    print(f"Verification  : {verification_path}")
    print(f"Run report    : {run_report_path}")

    if not args.skip_eval:
        result = run_evaluation(
            dataset_name=args.dataset_name,
            split=args.split,
            instance_id=instance_id,
            predictions_path=predictions_path,
            run_id=eval_run_id,
            report_dir=report_dir,
            max_workers=args.max_workers,
            cache_level=args.eval_cache_level,
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
