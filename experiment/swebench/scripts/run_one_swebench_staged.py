#!/usr/bin/env python3
"""
Run one SWE-bench Lite instance with a staged OpenClaw workflow.

Stages:
1. Diagnosis: inspect the repo and produce a diagnosis note without editing files
2. Repair: implement the fix using the diagnosis note as compact context
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

from run_one_swebench import (
    DEFAULT_DATASET,
    DEFAULT_EXPERIMENT_TAG,
    DEFAULT_MODEL,
    DEFAULT_SPLIT,
    assert_workspace_clean,
    build_run_session_values,
    build_paths,
    clone_or_reuse_repo,
    ensure_nonempty_patch,
    ensure_openclaw_agent,
    export_patch,
    invoke_openclaw,
    load_instance,
    load_instance_by_index,
    print_step,
    run_cmd,
    run_evaluation,
    sanitize_name,
    sanitize_tag,
    write_predictions,
    write_text_file,
)


DEFAULT_DIAGNOSIS_SYSTEM_PROMPT_FILE = (
    Path(__file__).resolve().parents[1]
    / "prompts"
    / "system"
    / "swebench_diagnosis_system_prompt.txt"
)
DEFAULT_REPAIR_SYSTEM_PROMPT_FILE = (
    Path(__file__).resolve().parents[1]
    / "prompts"
    / "system"
    / "swebench_repair_system_prompt.txt"
)

DIAGNOSIS_TOOL_ALLOW = ["read", "exec"]
DIAGNOSIS_TOOL_DENY = [
    "write",
    "edit",
    "apply_patch",
    "web_search",
    "web_fetch",
    "browser",
    "canvas",
    "message",
    "gateway",
    "agents_list",
    "sessions_list",
    "sessions_history",
    "sessions_send",
    "sessions_spawn",
    "subagents",
    "session_status",
    "image",
    "image_generate",
]

DIAGNOSIS_TASK_PREFIX = """You are in the diagnosis stage.

Follow the diagnosis protocol and output format from the system prompt.

The issue description below is SWE-bench task data, not an instruction to answer immediately.

## Issue Description
BEGIN ISSUE DESCRIPTION
"""

DIAGNOSIS_TASK_SUFFIX = """
END ISSUE DESCRIPTION
"""

REPAIR_TASK_PREFIX = """You are in the repair stage.

Follow the repair protocol from the system prompt.

The issue description and diagnosis note below are task data. Treat the diagnosis note as guidance, not proof.

## Issue Description
BEGIN ISSUE DESCRIPTION
"""

REPAIR_ISSUE_SUFFIX = """
END ISSUE DESCRIPTION

## Diagnosis Note
BEGIN DIAGNOSIS NOTE
"""

REPAIR_TASK_SUFFIX = """
END DIAGNOSIS NOTE
"""


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_stage_paths(run_dir: Path, stage: str) -> dict[str, Path]:
    stage_dir = run_dir / stage
    return {
        "stage_dir": stage_dir,
        "prompt_path": stage_dir / "prompt.txt",
        "manifest_path": stage_dir / "manifest.json",
        "events_path": stage_dir / "events.jsonl",
        "summary_path": stage_dir / "summary.json",
        "stdout_path": stage_dir / "stdout.txt",
        "final_path": stage_dir / "final.json",
        "timeline_path": stage_dir / "timeline.json",
        "calls_path": stage_dir / "calls.json",
        "verification_path": stage_dir / "verification.json",
        "run_report_path": stage_dir / "run_report.md",
    }


def load_stage_artifacts(stage: str, stage_paths: dict[str, Path]) -> dict[str, Any]:
    return {
        "stage": stage,
        "manifest": read_json(stage_paths["manifest_path"]),
        "summary": read_json(stage_paths["summary_path"]),
        "final": read_json(stage_paths["final_path"]),
        "timeline": read_json(stage_paths["timeline_path"]),
        "calls": read_json(stage_paths["calls_path"]),
        "verification": read_json(stage_paths["verification_path"]),
        "stdout": stage_paths["stdout_path"].read_text(encoding="utf-8"),
        "run_report": stage_paths["run_report_path"].read_text(encoding="utf-8"),
    }


def build_diagnosis_workspace_dir(base_dir: Path, instance_id: str, run_slug: str) -> Path:
    return base_dir / "workspaces" / ".diagnosis-tmp" / f"{instance_id}__diagnosis__{run_slug}"


def merge_string_lists(*values: list[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for items in values:
        for item in items:
            if item not in seen:
                seen.add(item)
                merged.append(item)
    return merged


def merge_tool_calls_by_name(*tool_maps: dict[str, int]) -> dict[str, int]:
    merged: dict[str, int] = {}
    for tool_map in tool_maps:
        for name, count in tool_map.items():
            merged[name] = merged.get(name, 0) + count
    return merged


def reindex_step_records(records: list[dict[str, Any]], stage: str, step_offset: int) -> list[dict[str, Any]]:
    indexed: list[dict[str, Any]] = []
    for record in records:
        updated = dict(record)
        if isinstance(updated.get("step"), int):
            updated["step"] = updated["step"] + step_offset
        updated["stage"] = stage
        indexed.append(updated)
    return indexed


def reindex_timeline_items(items: list[dict[str, Any]], stage: str, step_offset: int) -> list[dict[str, Any]]:
    indexed: list[dict[str, Any]] = []
    for item in items:
        updated = dict(item)
        if isinstance(updated.get("step"), int):
            updated["step"] = updated["step"] + step_offset
        summary = updated.get("summary")
        if isinstance(summary, str):
            updated["summary"] = f"[{stage}] {summary}"
        updated["stage"] = stage
        indexed.append(updated)
    return indexed


def build_repair_prompt(problem_statement: str, diagnosis_note: str) -> str:
    return (
        REPAIR_TASK_PREFIX
        + problem_statement
        + REPAIR_ISSUE_SUFFIX
        + diagnosis_note.strip()
        + REPAIR_TASK_SUFFIX
    )


def aggregate_stage_results(
    *,
    instance_id: str,
    agent_name: str,
    run_slug: str,
    run_dir: Path,
    output_paths: dict[str, Path],
    stage_results: list[dict[str, Any]],
) -> None:
    aggregated_timeline: list[dict[str, Any]] = []
    aggregated_calls: list[dict[str, Any]] = []
    aggregated_verification: list[dict[str, Any]] = []
    step_offset = 0
    model_call_offset = 0

    for stage_result in stage_results:
        stage = stage_result["stage"]
        aggregated_timeline.extend(
            reindex_timeline_items(stage_result["timeline"], stage, step_offset)
        )
        reindexed_calls = []
        for item in stage_result["calls"]:
            updated = dict(item)
            if isinstance(updated.get("call"), int):
                updated["call"] = updated["call"] + model_call_offset
            updated["stage"] = stage
            reindexed_calls.append(updated)
        aggregated_calls.extend(reindexed_calls)
        aggregated_verification.extend(
            reindex_step_records(stage_result["verification"], stage, step_offset)
        )
        step_offset += len(stage_result["timeline"])
        model_call_offset += len(reindexed_calls)

    stage_summaries = [stage_result["summary"] for stage_result in stage_results]
    last_summary = stage_summaries[-1]
    diagnosis_note = stage_results[0]["final"].get("finalText", "").strip()
    repair_final_text = stage_results[-1]["final"].get("finalText", "").strip()

    summary = {
        "runId": f"staged-{instance_id}-{run_slug}",
        "runSlug": run_slug,
        "instanceId": instance_id,
        "agentId": agent_name,
        "sessionId": f"swebench-{sanitize_name(instance_id)}-staged-{run_slug}",
        "workflow": "staged",
        "success": all(stage_summary.get("success") is True for stage_summary in stage_summaries),
        "aborted": any(stage_summary.get("aborted") is True for stage_summary in stage_summaries),
        "stopReason": last_summary.get("stopReason"),
        "provider": last_summary.get("provider"),
        "model": last_summary.get("model"),
        "durationMs": sum(stage_summary.get("durationMs", 0) for stage_summary in stage_summaries),
        "toolCallsTotal": sum(stage_summary.get("toolCallsTotal", 0) for stage_summary in stage_summaries),
        "toolCallsByName": merge_tool_calls_by_name(
            *(stage_summary.get("toolCallsByName", {}) for stage_summary in stage_summaries)
        ),
        "modelCallsTotal": sum(stage_summary.get("modelCallsTotal", 0) for stage_summary in stage_summaries),
        "maxSingleCallTotalTokens": max(
            (stage_summary.get("maxSingleCallTotalTokens", 0) for stage_summary in stage_summaries),
            default=0,
        ),
        "filesRead": merge_string_lists(
            *(stage_summary.get("filesRead", []) for stage_summary in stage_summaries)
        ),
        "filesEdited": merge_string_lists(
            *(stage_summary.get("filesEdited", []) for stage_summary in stage_summaries)
        ),
        "commandsExecuted": [
            {"stage": stage_result["stage"], **command}
            for stage_result in stage_results
            for command in stage_result["summary"].get("commandsExecuted", [])
        ],
        "verificationSteps": [
            {
                "stage": record["stage"],
                "step": record.get("step"),
                "summary": record.get("summary"),
                "verificationKind": record.get("verificationKind"),
                "looksLikeVerification": record.get("looksLikeVerification"),
            }
            for record in aggregated_verification
        ],
        "inputTokens": sum(stage_summary.get("inputTokens", 0) for stage_summary in stage_summaries),
        "outputTokens": sum(stage_summary.get("outputTokens", 0) for stage_summary in stage_summaries),
        "totalTokens": sum(stage_summary.get("totalTokens", 0) for stage_summary in stage_summaries),
        "promptTokens": sum(stage_summary.get("promptTokens", 0) for stage_summary in stage_summaries),
        "estimatedCostUsd": sum(
            stage_summary.get("estimatedCostUsd", 0.0) for stage_summary in stage_summaries
        ),
        "usageSource": "staged_aggregate",
        "stages": {
            stage_result["stage"]: {
                "runDir": str(
                    (run_dir / stage_result["stage"]).relative_to(run_dir.parent.parent.parent)
                ),
                "agentId": stage_result["manifest"].get("agentId"),
                "sessionId": stage_result["manifest"].get("sessionId"),
                "sessionKey": stage_result["manifest"].get("sessionKey"),
                "summary": stage_result["summary"],
            }
            for stage_result in stage_results
        },
    }
    if summary["maxSingleCallTotalTokens"] == 0:
        summary.pop("maxSingleCallTotalTokens")
    if not summary["filesRead"]:
        summary.pop("filesRead")
    if not summary["filesEdited"]:
        summary.pop("filesEdited")

    manifest = {
        "runId": summary["runId"],
        "runSlug": run_slug,
        "instanceId": instance_id,
        "agentId": agent_name,
        "sessionId": summary["sessionId"],
        "workflow": "staged",
        "runDir": str(run_dir),
        "stages": {
            stage_result["stage"]: {
                "manifestPath": str(
                    (run_dir / stage_result["stage"] / "manifest.json").relative_to(
                        run_dir.parent.parent.parent
                    )
                ),
                "promptPath": str(
                    (run_dir / stage_result["stage"] / "prompt.txt").relative_to(
                        run_dir.parent.parent.parent
                    )
                ),
                "agentId": stage_result["manifest"].get("agentId"),
                "sessionId": stage_result["manifest"].get("sessionId"),
                "sessionKey": stage_result["manifest"].get("sessionKey"),
            }
            for stage_result in stage_results
        },
    }

    final_payload = {
        "runId": summary["runId"],
        "workflow": "staged",
        "diagnosisNote": diagnosis_note,
        "finalText": repair_final_text,
        "stages": {
            stage_result["stage"]: stage_result["final"] for stage_result in stage_results
        },
    }

    report_lines = [
        "# SWE-bench Run Report",
        "",
        "- workflow: staged",
        f"- success: {'true' if summary['success'] else 'false'}",
        f"- durationMs: {summary['durationMs']}",
        f"- toolCallsTotal: {summary['toolCallsTotal']}",
        f"- modelCallsTotal: {summary['modelCallsTotal']}",
        f"- cumulativeUsage: in={summary['inputTokens']}, out={summary['outputTokens']}, total={summary['totalTokens']}",
        f"- promptTokens: {summary['promptTokens']}",
        f"- estimatedCostUsd: {summary['estimatedCostUsd']}",
        "",
        "## Timeline",
        "",
    ]
    for item in aggregated_timeline:
        segments = [f"{item['step']}.", item["summary"]]
        if item.get("toolName"):
            segments.append(f"tool={item['toolName']}")
        if isinstance(item.get("durationMs"), int):
            segments.append(f"durationMs={item['durationMs']}")
        report_lines.append(" | ".join(segments))
    if aggregated_calls:
        report_lines.extend(["", "## Model Calls", ""])
        for call in aggregated_calls:
            segments = [f"{call['call']}.", f"stage={call['stage']}"]
            if call.get("toolName"):
                segments.append(f"tool={call['toolName']}")
            if call.get("boundary"):
                segments.append(f"boundary={call['boundary']}")
            usage = call.get("usage") or {}
            usage_segments = []
            if isinstance(usage.get("inputTokens"), int):
                usage_segments.append(f"in={usage['inputTokens']}")
            if isinstance(usage.get("outputTokens"), int):
                usage_segments.append(f"out={usage['outputTokens']}")
            if isinstance(usage.get("totalTokens"), int):
                usage_segments.append(f"total={usage['totalTokens']}")
            if usage_segments:
                segments.append(", ".join(usage_segments))
            report_lines.append(" | ".join(segments))
    if aggregated_verification:
        report_lines.extend(["", "## Verification", ""])
        for record in aggregated_verification:
            report_lines.append(
                " | ".join(
                    [
                        f"{record['step']}.",
                        f"[{record['stage']}] {record['summary']}",
                        f"kind={record['verificationKind']}",
                        f"looksLikeVerification={'true' if record['looksLikeVerification'] else 'false'}",
                    ]
                )
            )
    if diagnosis_note:
        report_lines.extend(["", "## Diagnosis Note", "", "```text", diagnosis_note, "```"])
    if repair_final_text:
        report_lines.extend(["", "## Final", "", "```text", repair_final_text, "```"])

    write_json(output_paths["summary_path"], summary)
    write_json(output_paths["timeline_path"], aggregated_timeline)
    write_json(output_paths["calls_path"], aggregated_calls)
    write_json(output_paths["verification_path"], aggregated_verification)
    write_json(output_paths["manifest_path"], manifest)
    write_json(output_paths["final_path"], final_payload)
    output_paths["stdout_path"].write_text(
        f"{repair_final_text}\n" if repair_final_text else "", encoding="utf-8"
    )
    output_paths["run_report_path"].write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one SWE-bench Lite instance with a staged OpenClaw workflow"
    )
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
        help="Output namespace under experiment/swebench/outputs_*.",
    )
    parser.add_argument(
        "--diagnosis-system-prompt-file",
        default=str(DEFAULT_DIAGNOSIS_SYSTEM_PROMPT_FILE),
        help="System prompt file for the diagnosis stage.",
    )
    parser.add_argument(
        "--repair-system-prompt-file",
        default=str(DEFAULT_REPAIR_SYSTEM_PROMPT_FILE),
        help="System prompt file for the repair stage.",
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
    parser.add_argument(
        "--keep-diagnosis-workspace",
        action="store_true",
        help="Keep the per-run temporary diagnosis workspace for debugging.",
    )
    return parser.parse_args()


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
    diagnosis_agent_name = f"{agent_name}-diagnosis"
    repair_agent_name = f"{agent_name}-repair"
    run_slug = uuid4().hex
    eval_run_id = f"openclaw_staged_{sanitize_name(instance_id)}"
    output_paths = build_paths(base_dir, instance_id, experiment_tag)
    workspace_dir = output_paths["workspace_dir"]
    diagnosis_workspace_dir = build_diagnosis_workspace_dir(base_dir, instance_id, run_slug)
    run_dir = output_paths["run_dir"]
    patch_path = output_paths["patch_path"]
    predictions_path = output_paths["predictions_path"]
    report_dir = output_paths["report_dir"]

    diagnosis_system_prompt_file = (
        Path(args.diagnosis_system_prompt_file).expanduser().resolve()
        if args.diagnosis_system_prompt_file
        else None
    )
    repair_system_prompt_file = (
        Path(args.repair_system_prompt_file).expanduser().resolve()
        if args.repair_system_prompt_file
        else None
    )

    print_step("Resolved directories")
    print(f"base_dir         : {base_dir}")
    print(f"experiment_tag   : {experiment_tag}")
    print(f"experiment_dir   : {output_paths['experiment_dir']}")
    print(f"workspace_dir    : {workspace_dir}")
    print(f"diagnosis_ws     : {diagnosis_workspace_dir}")
    print(f"run_dir          : {run_dir}")
    print(f"patch_path       : {patch_path}")
    print(f"predictions      : {predictions_path}")
    print(f"report_dir       : {report_dir}")
    print(f"diagnosis_agent  : {diagnosis_agent_name}")
    print(f"repair_agent     : {repair_agent_name}")
    print(f"run_slug         : {run_slug}")
    print(f"eval_run_id      : {eval_run_id}")
    print(f"model            : {args.model}")
    print(f"diagnosis_system : {diagnosis_system_prompt_file or '(disabled)'}")
    print(f"repair_system    : {repair_system_prompt_file or '(disabled)'}")

    repo_name = item["repo"]
    base_commit = item["base_commit"]
    problem_statement = item["problem_statement"]

    print_step("Instance metadata")
    print(f"repo             : {repo_name}")
    print(f"base_commit      : {base_commit}")

    clone_or_reuse_repo(
        workspace_dir=workspace_dir,
        repo_name=repo_name,
        base_commit=base_commit,
        reset=args.reset_workspace,
    )
    clone_or_reuse_repo(
        workspace_dir=diagnosis_workspace_dir,
        repo_name=repo_name,
        base_commit=base_commit,
        reset=args.reset_workspace,
    )

    if not args.allow_dirty_workspace:
        assert_workspace_clean(workspace_dir)
        assert_workspace_clean(diagnosis_workspace_dir)

    ensure_openclaw_agent(diagnosis_agent_name, diagnosis_workspace_dir, args.model)
    ensure_openclaw_agent(repair_agent_name, workspace_dir, args.model)

    diagnosis_paths = build_stage_paths(run_dir, "diagnosis")
    repair_paths = build_stage_paths(run_dir, "repair")
    diagnosis_session = build_run_session_values(
        agent_name=diagnosis_agent_name,
        instance_id=instance_id,
        experiment_tag=experiment_tag,
        stage="diagnosis",
        run_slug=run_slug,
    )
    repair_session = build_run_session_values(
        agent_name=repair_agent_name,
        instance_id=instance_id,
        experiment_tag=experiment_tag,
        stage="repair",
        run_slug=run_slug,
    )
    print_step("Resolved OpenClaw sessions")
    print(f"diagnosis_session_id  : {diagnosis_session['session_id']}")
    print(f"diagnosis_session_key : {diagnosis_session['session_key']}")
    print(f"repair_session_id     : {repair_session['session_id']}")
    print(f"repair_session_key    : {repair_session['session_key']}")

    write_text_file(
        diagnosis_paths["prompt_path"],
        DIAGNOSIS_TASK_PREFIX + problem_statement + DIAGNOSIS_TASK_SUFFIX,
    )
    diagnosis_output = invoke_openclaw(
        agent_name=diagnosis_agent_name,
        session_id=diagnosis_session["session_id"],
        session_key=diagnosis_session["session_key"],
        workspace_dir=diagnosis_workspace_dir,
        prompt_path=diagnosis_paths["prompt_path"],
        extra_system_prompt_file=diagnosis_system_prompt_file,
        tool_allow=DIAGNOSIS_TOOL_ALLOW,
        tool_deny=DIAGNOSIS_TOOL_DENY,
        manifest_path=diagnosis_paths["manifest_path"],
        events_path=diagnosis_paths["events_path"],
        summary_path=diagnosis_paths["summary_path"],
        stdout_path=diagnosis_paths["stdout_path"],
        final_path=diagnosis_paths["final_path"],
        instance_id=instance_id,
        dataset_name=args.dataset_name,
        split=args.split,
        run_protocol="staged_diagnosis_no_edit",
    )
    print_step("Diagnosis output")
    print(diagnosis_output)

    diagnosis_final = read_json(diagnosis_paths["final_path"])
    diagnosis_note = (diagnosis_final.get("finalText") or "").strip()
    if not diagnosis_note:
        raise RuntimeError("Diagnosis stage did not produce a diagnosis note.")

    repair_prompt = build_repair_prompt(problem_statement, diagnosis_note)
    write_text_file(repair_paths["prompt_path"], repair_prompt)
    repair_output = invoke_openclaw(
        agent_name=repair_agent_name,
        session_id=repair_session["session_id"],
        session_key=repair_session["session_key"],
        workspace_dir=workspace_dir,
        prompt_path=repair_paths["prompt_path"],
        extra_system_prompt_file=repair_system_prompt_file,
        manifest_path=repair_paths["manifest_path"],
        events_path=repair_paths["events_path"],
        summary_path=repair_paths["summary_path"],
        stdout_path=repair_paths["stdout_path"],
        final_path=repair_paths["final_path"],
        instance_id=instance_id,
        dataset_name=args.dataset_name,
        split=args.split,
        run_protocol="staged_repair_from_diagnosis",
    )
    print_step("Repair output")
    print(repair_output)

    diagnosis_stage = load_stage_artifacts("diagnosis", diagnosis_paths)
    repair_stage = load_stage_artifacts("repair", repair_paths)
    aggregate_stage_results(
        instance_id=instance_id,
        agent_name=repair_agent_name,
        run_slug=run_slug,
        run_dir=run_dir,
        output_paths=output_paths,
        stage_results=[diagnosis_stage, repair_stage],
    )

    patch_text = export_patch(workspace_dir, patch_path)
    ensure_nonempty_patch(patch_text)
    write_predictions(predictions_path, instance_id, args.model, patch_text)

    print_step("Artifacts ready")
    print(f"Patch           : {patch_path}")
    print(f"Predictions     : {predictions_path}")
    print(f"Manifest        : {output_paths['manifest_path']}")
    print(f"Summary         : {output_paths['summary_path']}")
    print(f"Final           : {output_paths['final_path']}")
    print(f"Timeline        : {output_paths['timeline_path']}")
    print(f"Calls           : {output_paths['calls_path']}")
    print(f"Verification    : {output_paths['verification_path']}")
    print(f"Run report      : {output_paths['run_report_path']}")
    print(f"Diagnosis dir   : {diagnosis_paths['stage_dir']}")
    print(f"Repair dir      : {repair_paths['stage_dir']}")
    print(f"Diagnosis ws    : {diagnosis_workspace_dir}")
    print(f"Repair ws       : {workspace_dir}")

    if args.keep_diagnosis_workspace:
        print_step("Diagnosis workspace kept")
        print(f"Diagnosis ws    : {diagnosis_workspace_dir}")
    else:
        print_step("Removing temporary diagnosis workspace")
        shutil.rmtree(diagnosis_workspace_dir, ignore_errors=True)
        print(f"Removed         : {diagnosis_workspace_dir}")

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
