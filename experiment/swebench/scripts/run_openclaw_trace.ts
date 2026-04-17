#!/usr/bin/env -S node --import tsx

import { randomUUID } from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";
import { agentCommand } from "../../../src/agents/agent-command.js";
import { derivePromptTokens } from "../../../src/agents/usage.js";
import { loadConfig } from "../../../src/config/config.js";
import { createNonExitingRuntime } from "../../../src/runtime.js";
import { estimateUsageCost, resolveModelCostConfig } from "../../../src/utils/usage-format.js";
import { providerUsageCaptureLogs, providerUsageCaptures } from "./provider-usage-capture.ts";

type CliArgs = {
  agentId: string;
  sessionId: string;
  workspaceDir: string;
  promptFile: string;
  manifestPath: string;
  eventsPath: string;
  summaryPath: string;
  stdoutPath: string;
  finalPath: string;
  instanceId?: string;
  datasetName?: string;
  split?: string;
  runProtocol?: string;
};

type TraceEvent = {
  ts: string;
  tsMs: number;
  eventType: "runner" | "agent_event" | "tool_result" | "tool_usage";
  runId: string;
  payload: Record<string, unknown>;
};

type NormalizedUsageLike = {
  input?: number;
  output?: number;
  cacheRead?: number;
  cacheWrite?: number;
  total?: number;
};

type UsageTotals = {
  inputTokens?: number;
  outputTokens?: number;
  totalTokens?: number;
  cacheReadTokens?: number;
  cacheWriteTokens?: number;
};

type TimelineStep = {
  step: number;
  kind: "tool" | "assistant_final" | "runner";
  startedAt?: string;
  endedAt?: string;
  durationMs?: number;
  toolName?: string;
  toolCallId?: string;
  phase?: string;
  summary: string;
  status?: "ok" | "error";
  updates?: number;
  usageBefore?: UsageTotals;
  usageAfter?: UsageTotals;
  usageDelta?: UsageTotals;
  artifact?: Record<string, unknown>;
};

type ModelCallSummary = {
  call: number;
  ts: string;
  provider?: string;
  model?: string;
  phase?: string;
  toolName?: string;
  toolCallId?: string;
  boundary?: string;
  usage?: UsageTotals;
  cumulativeUsage?: UsageTotals;
  usageDeltaFromPrevious?: UsageTotals;
};

type ParsedToolMeta = {
  summary: string;
  artifact?: Record<string, unknown>;
};

type VerificationStep = {
  step: number;
  toolCallId?: string;
  summary: string;
  cwd?: string;
  cmd?: string;
  status?: "ok" | "error";
  durationMs?: number;
  looksLikeVerification: boolean;
  verificationKind: "test_command" | "python_verification" | "git_diff_check" | "other_exec";
  usageDelta?: UsageTotals;
};

function parseArgs(argv: string[]): CliArgs {
  const map = new Map<string, string>();
  for (let i = 0; i < argv.length; i += 1) {
    const token = argv[i];
    if (!token.startsWith("--")) {
      continue;
    }
    const value = argv[i + 1];
    if (!value || value.startsWith("--")) {
      throw new Error(`Missing value for ${token}`);
    }
    map.set(token.slice(2), value);
    i += 1;
  }

  const getRequired = (name: string): string => {
    const value = map.get(name)?.trim();
    if (!value) {
      throw new Error(`Missing required argument --${name}`);
    }
    return value;
  };

  return {
    agentId: getRequired("agent-id"),
    sessionId: getRequired("session-id"),
    workspaceDir: getRequired("workspace-dir"),
    promptFile: getRequired("prompt-file"),
    manifestPath: getRequired("manifest-path"),
    eventsPath: getRequired("events-path"),
    summaryPath: getRequired("summary-path"),
    stdoutPath: getRequired("stdout-path"),
    finalPath: getRequired("final-path"),
    instanceId: map.get("instance-id")?.trim() || undefined,
    datasetName: map.get("dataset-name")?.trim() || undefined,
    split: map.get("split")?.trim() || undefined,
    runProtocol: map.get("run-protocol")?.trim() || "single_pass_no_retry",
  };
}

async function ensureParent(filePath: string): Promise<void> {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
}

function createTraceEvent(
  runId: string,
  eventType: TraceEvent["eventType"],
  payload: Record<string, unknown>,
): TraceEvent {
  const tsMs = Date.now();
  return {
    ts: new Date(tsMs).toISOString(),
    tsMs,
    eventType,
    runId,
    payload,
  };
}

function normalizeUsage(
  usage:
    | {
        input?: number;
        output?: number;
        cacheRead?: number;
        cacheWrite?: number;
        total?: number;
      }
    | null
    | undefined,
): NormalizedUsageLike | undefined {
  if (!usage) {
    return undefined;
  }
  const input = usage?.input;
  const output = usage?.output;
  const cacheRead = usage?.cacheRead;
  const cacheWrite = usage?.cacheWrite;
  const derivedTotal = [input ?? 0, output ?? 0, cacheRead ?? 0, cacheWrite ?? 0].reduce(
    (sum, value) => sum + value,
    0,
  );
  const total =
    typeof usage.total === "number" &&
    Number.isFinite(usage.total) &&
    (derivedTotal === 0 || usage.total >= derivedTotal)
      ? usage.total
      : derivedTotal > 0
        ? derivedTotal
        : undefined;

  if (
    input === undefined &&
    output === undefined &&
    cacheRead === undefined &&
    cacheWrite === undefined &&
    total === undefined
  ) {
    return undefined;
  }

  return {
    ...(typeof input === "number" ? { input } : {}),
    ...(typeof output === "number" ? { output } : {}),
    ...(typeof total === "number" && Number.isFinite(total) ? { total } : {}),
    ...(typeof cacheRead === "number" ? { cacheRead } : {}),
    ...(typeof cacheWrite === "number" ? { cacheWrite } : {}),
  };
}

function resolveUsageTotals(usage: NormalizedUsageLike | null | undefined): UsageTotals {
  const normalized = normalizeUsage(usage);
  return {
    ...(typeof normalized?.input === "number" ? { inputTokens: normalized.input } : {}),
    ...(typeof normalized?.output === "number" ? { outputTokens: normalized.output } : {}),
    ...(typeof normalized?.total === "number" ? { totalTokens: normalized.total } : {}),
    ...(typeof normalized?.cacheRead === "number" ? { cacheReadTokens: normalized.cacheRead } : {}),
    ...(typeof normalized?.cacheWrite === "number"
      ? { cacheWriteTokens: normalized.cacheWrite }
      : {}),
  };
}

function resolveOptionalUsageTotals(
  usage:
    | {
        input?: number;
        output?: number;
        cacheRead?: number;
        cacheWrite?: number;
        total?: number;
      }
    | null
    | undefined,
): UsageTotals | undefined {
  const totals = resolveUsageTotals(normalizeUsage(usage));
  return Object.keys(totals).length > 0 ? totals : undefined;
}

function hasNonzeroUsage(usage: NormalizedUsageLike | null | undefined): boolean {
  if (!usage) {
    return false;
  }
  return [usage.input, usage.output, usage.cacheRead, usage.cacheWrite, usage.total].some(
    (value) => typeof value === "number" && Number.isFinite(value) && value > 0,
  );
}

function toFiniteNumber(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function diffUsageTotals(current?: UsageTotals, previous?: UsageTotals): UsageTotals | undefined {
  if (!current) {
    return undefined;
  }
  const inputTokens = (current.inputTokens ?? 0) - (previous?.inputTokens ?? 0);
  const outputTokens = (current.outputTokens ?? 0) - (previous?.outputTokens ?? 0);
  const totalTokens = (current.totalTokens ?? 0) - (previous?.totalTokens ?? 0);
  const cacheReadTokens = (current.cacheReadTokens ?? 0) - (previous?.cacheReadTokens ?? 0);
  const cacheWriteTokens = (current.cacheWriteTokens ?? 0) - (previous?.cacheWriteTokens ?? 0);
  const delta: UsageTotals = {};
  if (inputTokens !== 0) {
    delta.inputTokens = inputTokens;
  }
  if (outputTokens !== 0) {
    delta.outputTokens = outputTokens;
  }
  if (totalTokens !== 0) {
    delta.totalTokens = totalTokens;
  }
  if (cacheReadTokens !== 0) {
    delta.cacheReadTokens = cacheReadTokens;
  }
  if (cacheWriteTokens !== 0) {
    delta.cacheWriteTokens = cacheWriteTokens;
  }
  return Object.keys(delta).length > 0 ? delta : undefined;
}

function subtractUsageTotals(current?: UsageTotals, delta?: UsageTotals): UsageTotals | undefined {
  if (!current) {
    return undefined;
  }
  const value: UsageTotals = {};
  const fields: Array<keyof UsageTotals> = [
    "inputTokens",
    "outputTokens",
    "totalTokens",
    "cacheReadTokens",
    "cacheWriteTokens",
  ];
  for (const field of fields) {
    const currentValue = current[field];
    if (typeof currentValue !== "number") {
      continue;
    }
    const deltaValue = delta?.[field] ?? 0;
    value[field] = currentValue - deltaValue;
  }
  return Object.keys(value).length > 0 ? value : undefined;
}

function formatUsageTotals(usage?: UsageTotals): string | undefined {
  if (!usage) {
    return undefined;
  }
  const segments: string[] = [];
  if (typeof usage.inputTokens === "number") {
    segments.push(`in=${usage.inputTokens}`);
  }
  if (typeof usage.outputTokens === "number") {
    segments.push(`out=${usage.outputTokens}`);
  }
  if (typeof usage.totalTokens === "number") {
    segments.push(`total=${usage.totalTokens}`);
  }
  if (typeof usage.cacheReadTokens === "number" && usage.cacheReadTokens > 0) {
    segments.push(`cacheRead=${usage.cacheReadTokens}`);
  }
  if (typeof usage.cacheWriteTokens === "number" && usage.cacheWriteTokens > 0) {
    segments.push(`cacheWrite=${usage.cacheWriteTokens}`);
  }
  return segments.length > 0 ? segments.join(", ") : undefined;
}

function sanitizePath(rawPath: string): string {
  return rawPath.replace(/^~\/桌面\/my-openclaw\//, "");
}

function parseToolMeta(toolName: string, meta: unknown): ParsedToolMeta {
  const metaText = typeof meta === "string" ? meta.trim() : "";
  if (!metaText) {
    return { summary: toolName };
  }

  if (toolName === "read") {
    const match = /^lines (\d+)-(\d+) from (.+)$/.exec(metaText);
    if (match) {
      const startLine = Number(match[1]);
      const endLine = Number(match[2]);
      const filePath = sanitizePath(match[3]);
      return {
        summary: `read ${filePath}:${startLine}-${endLine}`,
        artifact: { path: filePath, startLine, endLine },
      };
    }
  }

  if (toolName === "edit") {
    const match = /^in (.+) \((\d+) chars\)$/.exec(metaText);
    if (match) {
      const filePath = sanitizePath(match[1]);
      const chars = Number(match[2]);
      return {
        summary: `edit ${filePath}`,
        artifact: { path: filePath, chars },
      };
    }
  }

  if (toolName === "exec") {
    const commandMarker = ", `";
    const markerIndex = metaText.lastIndexOf(commandMarker);
    if (markerIndex >= 0 && metaText.endsWith("`")) {
      const summaryPrefix = metaText.slice(0, markerIndex).trim();
      const command = metaText.slice(markerIndex + commandMarker.length, -1).trim();
      const cwdMatch = /\(in ([^)]+)\)$/.exec(summaryPrefix);
      const cwd = cwdMatch ? sanitizePath(cwdMatch[1]) : undefined;
      const summary = summaryPrefix.replace(/\s*\(in [^)]+\)$/, "").trim();
      return {
        summary: summary || "exec command",
        artifact: {
          ...(cwd ? { cwd } : {}),
          cmd: command,
        },
      };
    }
  }

  return { summary: metaText };
}

function buildModelCallSummaries(events: TraceEvent[]): ModelCallSummary[] {
  const calls = events.filter((event) => event.eventType === "tool_usage");
  const summaries: ModelCallSummary[] = [];
  let previousCumulative: UsageTotals | undefined;
  for (const [index, event] of calls.entries()) {
    const payload = event.payload;
    const usage = resolveOptionalUsageTotals(
      payload.usage as NormalizedUsageLike | null | undefined,
    );
    const cumulativeUsage = resolveOptionalUsageTotals(
      payload.cumulativeUsage as NormalizedUsageLike | null | undefined,
    );
    const toolContext = (payload.toolContext as Record<string, unknown> | undefined) ?? undefined;
    summaries.push({
      call: index + 1,
      ts: event.ts,
      provider: typeof payload.provider === "string" ? payload.provider : undefined,
      model: typeof payload.model === "string" ? payload.model : undefined,
      phase: typeof payload.phase === "string" ? payload.phase : undefined,
      toolName: typeof toolContext?.toolName === "string" ? toolContext.toolName : undefined,
      toolCallId: typeof toolContext?.toolCallId === "string" ? toolContext.toolCallId : undefined,
      boundary: typeof toolContext?.boundary === "string" ? toolContext.boundary : undefined,
      usage,
      cumulativeUsage,
      usageDeltaFromPrevious: diffUsageTotals(cumulativeUsage, previousCumulative),
    });
    if (cumulativeUsage) {
      previousCumulative = cumulativeUsage;
    }
  }
  return summaries;
}

function buildTimeline(
  events: TraceEvent[],
  finalText: string,
  modelCalls: ModelCallSummary[],
): TimelineStep[] {
  const steps: TimelineStep[] = [];
  let step = 1;
  const tools = new Map<
    string,
    {
      stepIndex: number;
      startedAt?: string;
      updates: number;
      toolName: string;
    }
  >();

  for (const event of events) {
    if (event.eventType === "tool_usage") {
      continue;
    }

    if (event.eventType !== "agent_event") {
      continue;
    }

    if (event.payload.stream !== "tool") {
      continue;
    }

    const data = (event.payload.data as Record<string, unknown> | undefined) ?? {};
    const phase = typeof data.phase === "string" ? data.phase : undefined;
    const toolName = typeof data.name === "string" ? data.name : undefined;
    const toolCallId = typeof data.toolCallId === "string" ? data.toolCallId : undefined;
    if (!toolName || !toolCallId || !phase) {
      continue;
    }

    if (phase === "start") {
      steps.push({
        step,
        kind: "tool",
        startedAt: event.ts,
        toolName,
        toolCallId,
        phase,
        summary: toolName,
        updates: 0,
      });
      tools.set(toolCallId, {
        stepIndex: steps.length - 1,
        startedAt: event.ts,
        updates: 0,
        toolName,
      });
      step += 1;
      continue;
    }

    const toolState = tools.get(toolCallId);
    if (!toolState) {
      continue;
    }

    if (phase === "update") {
      toolState.updates += 1;
      steps[toolState.stepIndex].updates = toolState.updates;
      continue;
    }

    if (phase === "result") {
      const parsedMeta = parseToolMeta(toolName, data.meta);
      const startedAtMs = Date.parse(toolState.startedAt ?? event.ts);
      const endedAtMs = Date.parse(event.ts);
      const relatedCalls = modelCalls.filter((call) => call.toolCallId === toolCallId);
      const usageAfter = relatedCalls.at(-1)?.cumulativeUsage;
      const totalUsageDelta = relatedCalls.reduce<UsageTotals | undefined>((acc, call) => {
        if (!call.usageDeltaFromPrevious) {
          return acc;
        }
        const merged: UsageTotals = {
          inputTokens: (acc?.inputTokens ?? 0) + (call.usageDeltaFromPrevious.inputTokens ?? 0),
          outputTokens: (acc?.outputTokens ?? 0) + (call.usageDeltaFromPrevious.outputTokens ?? 0),
          totalTokens: (acc?.totalTokens ?? 0) + (call.usageDeltaFromPrevious.totalTokens ?? 0),
          cacheReadTokens:
            (acc?.cacheReadTokens ?? 0) + (call.usageDeltaFromPrevious.cacheReadTokens ?? 0),
          cacheWriteTokens:
            (acc?.cacheWriteTokens ?? 0) + (call.usageDeltaFromPrevious.cacheWriteTokens ?? 0),
        };
        return Object.fromEntries(
          Object.entries(merged).filter(([, value]) => typeof value === "number" && value !== 0),
        ) as UsageTotals;
      }, undefined);
      const usageBefore = subtractUsageTotals(usageAfter, totalUsageDelta);
      steps[toolState.stepIndex] = {
        ...steps[toolState.stepIndex],
        endedAt: event.ts,
        durationMs:
          Number.isFinite(startedAtMs) && Number.isFinite(endedAtMs)
            ? Math.max(0, endedAtMs - startedAtMs)
            : undefined,
        phase,
        summary: parsedMeta.summary,
        status: data.isError === true ? "error" : "ok",
        usageBefore,
        usageAfter,
        usageDelta: totalUsageDelta,
        ...(parsedMeta.artifact ? { artifact: parsedMeta.artifact } : {}),
      };
      tools.delete(toolCallId);
    }
  }

  if (finalText.trim()) {
    steps.push({
      step,
      kind: "assistant_final",
      summary: finalText.trim().split("\n")[0] ?? finalText.trim(),
    });
    step += 1;
  }

  let lifecycleEnd: TraceEvent | undefined;
  for (let index = events.length - 1; index >= 0; index -= 1) {
    const event = events[index];
    if (
      event?.eventType === "agent_event" &&
      event.payload.stream === "lifecycle" &&
      (event.payload.data as { phase?: unknown } | undefined)?.phase === "end"
    ) {
      lifecycleEnd = event;
      break;
    }
  }
  if (lifecycleEnd) {
    steps.push({
      step,
      kind: "runner",
      endedAt: lifecycleEnd.ts,
      summary: "agent lifecycle end",
    });
  }

  return steps;
}

function classifyVerificationKind(params: {
  summary: string;
  cmd?: string;
}): VerificationStep["verificationKind"] {
  const haystack = `${params.summary}\n${params.cmd ?? ""}`.toLowerCase();
  if (haystack.includes("git diff")) {
    return "git_diff_check";
  }
  if (
    haystack.includes("pytest") ||
    haystack.includes("tox") ||
    haystack.includes("unittest") ||
    haystack.includes("pnpm test") ||
    haystack.includes("npm test")
  ) {
    return "test_command";
  }
  if (
    haystack.includes("python") ||
    haystack.includes("verify.py") ||
    haystack.includes("run separable")
  ) {
    return "python_verification";
  }
  return "other_exec";
}

function buildVerificationSteps(timeline: TimelineStep[]): VerificationStep[] {
  return timeline
    .filter((step) => step.kind === "tool" && step.toolName === "exec")
    .map((step) => {
      const artifact = step.artifact ?? {};
      const cmd = typeof artifact.cmd === "string" ? artifact.cmd : undefined;
      const cwd = typeof artifact.cwd === "string" ? artifact.cwd : undefined;
      const verificationKind = classifyVerificationKind({
        summary: step.summary,
        cmd,
      });
      return {
        step: step.step,
        toolCallId: step.toolCallId,
        summary: step.summary,
        ...(cwd ? { cwd } : {}),
        ...(cmd ? { cmd } : {}),
        status: step.status,
        durationMs: step.durationMs,
        looksLikeVerification: verificationKind !== "other_exec",
        verificationKind,
        usageDelta: step.usageDelta,
      };
    });
}

function buildRunReport(params: {
  summary: Record<string, unknown>;
  timeline: TimelineStep[];
  calls: ModelCallSummary[];
  verificationSteps: VerificationStep[];
  finalText: string;
}): string {
  const lines: string[] = [];
  const success = params.summary.success === true ? "true" : "false";
  lines.push("# SWE-bench Run Report");
  lines.push("");
  lines.push(`- success: ${success}`);
  if (typeof params.summary.stopReason === "string") {
    lines.push(`- stopReason: ${params.summary.stopReason}`);
  }
  if (typeof params.summary.durationMs === "number") {
    lines.push(`- durationMs: ${params.summary.durationMs}`);
  }
  if (typeof params.summary.toolCallsTotal === "number") {
    lines.push(`- toolCallsTotal: ${params.summary.toolCallsTotal}`);
  }
  if (typeof params.summary.modelCallsTotal === "number") {
    lines.push(`- modelCallsTotal: ${params.summary.modelCallsTotal}`);
  }
  const totalUsage = formatUsageTotals({
    inputTokens: toFiniteNumber(params.summary.inputTokens),
    outputTokens: toFiniteNumber(params.summary.outputTokens),
    totalTokens: toFiniteNumber(params.summary.totalTokens),
  });
  if (totalUsage) {
    lines.push(`- cumulativeUsage: ${totalUsage}`);
  }
  if (typeof params.summary.promptTokens === "number") {
    lines.push(`- promptTokens: ${params.summary.promptTokens}`);
  }
  if (typeof params.summary.estimatedCostUsd === "number") {
    lines.push(`- estimatedCostUsd: ${params.summary.estimatedCostUsd}`);
  }
  if (typeof params.summary.maxSingleCallTotalTokens === "number") {
    lines.push(`- maxSingleCallTotalTokens: ${params.summary.maxSingleCallTotalTokens}`);
  }
  lines.push("");
  lines.push("## Timeline");
  lines.push("");
  for (const item of params.timeline) {
    const segments = [`${item.step}.`, item.summary];
    if (item.kind === "tool" && item.toolName) {
      segments.push(`tool=${item.toolName}`);
    }
    if (typeof item.durationMs === "number") {
      segments.push(`durationMs=${item.durationMs}`);
    }
    const usageDelta = formatUsageTotals(item.usageDelta);
    if (usageDelta) {
      segments.push(`delta(${usageDelta})`);
    }
    lines.push(segments.join(" | "));
  }
  if (params.calls.length > 0) {
    lines.push("");
    lines.push("## Model Calls");
    lines.push("");
    for (const call of params.calls) {
      const segments = [`${call.call}.`];
      if (call.toolName) {
        segments.push(`tool=${call.toolName}`);
      }
      if (call.boundary) {
        segments.push(`boundary=${call.boundary}`);
      }
      const usage = formatUsageTotals(call.usage);
      if (usage) {
        segments.push(usage);
      }
      lines.push(segments.join(" | "));
    }
  }
  if (params.verificationSteps.length > 0) {
    lines.push("");
    lines.push("## Verification");
    lines.push("");
    for (const step of params.verificationSteps) {
      const segments = [
        `${step.step}.`,
        step.summary,
        `kind=${step.verificationKind}`,
        `looksLikeVerification=${step.looksLikeVerification ? "true" : "false"}`,
      ];
      const usage = formatUsageTotals(step.usageDelta);
      if (usage) {
        segments.push(`delta(${usage})`);
      }
      lines.push(segments.join(" | "));
    }
  }
  if (params.finalText.trim()) {
    lines.push("");
    lines.push("## Final");
    lines.push("");
    lines.push("```text");
    lines.push(params.finalText.trim());
    lines.push("```");
  }
  return `${lines.join("\n")}\n`;
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));
  const runId = randomUUID();
  const prompt = await fs.readFile(args.promptFile, "utf8");
  providerUsageCaptures.length = 0;
  providerUsageCaptureLogs.length = 0;
  const runtimeLogs: string[] = [];
  const runtimeErrors: string[] = [];
  const cfg = loadConfig();
  const runtime = createNonExitingRuntime();
  runtime.log = (...values: unknown[]) => {
    runtimeLogs.push(values.map((value) => String(value)).join(" "));
  };
  runtime.error = (...values: unknown[]) => {
    runtimeErrors.push(values.map((value) => String(value)).join(" "));
  };
  const events: TraceEvent[] = [];
  events.push(
    createTraceEvent(runId, "runner", {
      phase: "start",
      agentId: args.agentId,
      sessionId: args.sessionId,
      workspaceDir: args.workspaceDir,
    }),
  );

  const startedAt = Date.now();
  let exitCode = 0;
  let agentResult: Awaited<ReturnType<typeof agentCommand>> | undefined;
  let failureMessage: string | undefined;

  try {
    agentResult = await agentCommand(
      {
        agentId: args.agentId,
        sessionId: args.sessionId,
        workspaceDir: args.workspaceDir,
        message: prompt,
        runId,
        onAgentEvent: (evt) => {
          if (evt.stream === "usage") {
            events.push(
              createTraceEvent(runId, "tool_usage", {
                ...evt.data,
              }),
            );
          }
          events.push(
            createTraceEvent(runId, "agent_event", {
              stream: evt.stream,
              data: evt.data,
            }),
          );
        },
        onToolResult: (payload) => {
          events.push(
            createTraceEvent(runId, "tool_result", {
              text: payload.text,
              mediaUrls: payload.mediaUrls,
            }),
          );
        },
      },
      runtime,
    );
  } catch (error) {
    exitCode = 1;
    failureMessage = error instanceof Error ? error.message : String(error);
    events.push(
      createTraceEvent(runId, "runner", {
        phase: "error",
        message: failureMessage,
      }),
    );
  }

  const endedAt = Date.now();
  events.push(
    createTraceEvent(runId, "runner", {
      phase: "end",
      success: exitCode === 0,
      durationMs: endedAt - startedAt,
    }),
  );

  const payloads = agentResult?.payloads ?? [];
  const finalText = payloads
    .map((payload) => payload.text?.trim())
    .filter((value): value is string => Boolean(value))
    .join("\n\n");
  const agentMeta = agentResult?.meta.agentMeta;
  const usage = normalizeUsage(agentMeta?.usage);
  const provider = agentMeta?.provider;
  const model = agentMeta?.model;
  const promptTokens = agentMeta?.promptTokens;
  const providerUsageCapture =
    [...providerUsageCaptures].toReversed().find((capture) => hasNonzeroUsage(capture.usage)) ??
    [...providerUsageCaptures].toReversed().find((capture) => capture.rawUsage) ??
    [...providerUsageCaptures].at(-1);
  const providerUsage = normalizeUsage(providerUsageCapture?.usage);
  const effectiveUsage = hasNonzeroUsage(usage) ? usage : providerUsage;
  const effectivePromptTokens =
    typeof promptTokens === "number" && Number.isFinite(promptTokens) && promptTokens > 0
      ? promptTokens
      : derivePromptTokens(providerUsage);
  const estimatedCostUsd = estimateUsageCost({
    usage: effectiveUsage,
    cost: resolveModelCostConfig({
      provider,
      model,
      config: cfg,
    }),
  });

  const toolStartEvents = events.filter(
    (event) =>
      event.eventType === "agent_event" &&
      event.payload.stream === "tool" &&
      (event.payload.data as { phase?: unknown } | undefined)?.phase === "start",
  );
  const toolCallsByName: Record<string, number> = {};
  for (const event of toolStartEvents) {
    const name = (event.payload.data as { name?: unknown } | undefined)?.name;
    if (typeof name !== "string" || !name) {
      continue;
    }
    toolCallsByName[name] = (toolCallsByName[name] ?? 0) + 1;
  }
  const modelCalls = buildModelCallSummaries(events);
  const timeline = buildTimeline(events, finalText, modelCalls);
  const verificationSteps = buildVerificationSteps(timeline);
  const filesRead = [
    ...new Set(
      timeline
        .filter((step) => step.toolName === "read")
        .map((step) => step.artifact?.path)
        .filter((value): value is string => typeof value === "string" && value.length > 0),
    ),
  ];
  const filesEdited = [
    ...new Set(
      timeline
        .filter((step) => step.toolName === "edit")
        .map((step) => step.artifact?.path)
        .filter((value): value is string => typeof value === "string" && value.length > 0),
    ),
  ];
  const commandsExecuted = timeline
    .filter((step) => step.toolName === "exec")
    .map((step) => {
      const artifact = step.artifact ?? {};
      return {
        ...(typeof artifact.cwd === "string" ? { cwd: artifact.cwd } : {}),
        ...(typeof artifact.cmd === "string" ? { cmd: artifact.cmd } : {}),
        summary: step.summary,
      };
    });
  const maxSingleCallTotalTokens = modelCalls.reduce<number | undefined>((max, call) => {
    const total = call.usage?.totalTokens;
    if (!(typeof total === "number")) {
      return max;
    }
    if (!(typeof max === "number")) {
      return total;
    }
    return Math.max(max, total);
  }, undefined);

  const manifest = {
    runId,
    instanceId: args.instanceId,
    datasetName: args.datasetName,
    split: args.split,
    agentId: args.agentId,
    sessionId: args.sessionId,
    workspaceDir: args.workspaceDir,
    promptFile: args.promptFile,
    runProtocol: args.runProtocol,
    startedAt: new Date(startedAt).toISOString(),
    endedAt: new Date(endedAt).toISOString(),
  };

  const summary = {
    runId,
    instanceId: args.instanceId,
    agentId: args.agentId,
    sessionId: args.sessionId,
    provider,
    model,
    durationMs: endedAt - startedAt,
    success: exitCode === 0,
    error: failureMessage,
    aborted: agentResult?.meta.aborted ?? false,
    stopReason: agentResult?.meta.stopReason,
    toolCallsTotal: toolStartEvents.length,
    toolCallsByName,
    modelCallsTotal: modelCalls.length,
    ...(typeof maxSingleCallTotalTokens === "number" ? { maxSingleCallTotalTokens } : {}),
    ...(filesRead.length > 0 ? { filesRead } : {}),
    ...(filesEdited.length > 0 ? { filesEdited } : {}),
    ...(commandsExecuted.length > 0 ? { commandsExecuted } : {}),
    ...(verificationSteps.length > 0
      ? {
          verificationSteps: verificationSteps.map((step) => ({
            step: step.step,
            summary: step.summary,
            verificationKind: step.verificationKind,
            looksLikeVerification: step.looksLikeVerification,
          })),
        }
      : {}),
    ...resolveUsageTotals(effectiveUsage),
    ...(typeof effectivePromptTokens === "number" && Number.isFinite(effectivePromptTokens)
      ? { promptTokens: effectivePromptTokens }
      : {}),
    ...(typeof estimatedCostUsd === "number" ? { estimatedCostUsd } : {}),
    ...(providerUsageCapture
      ? {
          usageSource: hasNonzeroUsage(usage) ? "openclaw" : "provider_capture",
        }
      : {}),
  };

  const finalResult = {
    runId,
    payloads,
    meta: agentResult?.meta,
    usageDebug: {
      provider,
      model,
      usage: resolveOptionalUsageTotals(effectiveUsage),
      promptTokens:
        typeof promptTokens === "number" && Number.isFinite(promptTokens)
          ? promptTokens
          : undefined,
      providerCaptureSeen: providerUsageCaptures.length > 0,
      providerCaptureUsage: resolveOptionalUsageTotals(providerUsage),
      effectivePromptTokens:
        typeof effectivePromptTokens === "number" && Number.isFinite(effectivePromptTokens)
          ? effectivePromptTokens
          : undefined,
      usageSource: hasNonzeroUsage(usage)
        ? "openclaw"
        : providerUsageCapture
          ? "provider_capture"
          : "none",
    },
    finalText,
    runtimeLogs:
      providerUsageCapture && !hasNonzeroUsage(usage)
        ? [...runtimeLogs, ...providerUsageCaptureLogs]
        : runtimeLogs,
    runtimeErrors,
  };
  const runDir = path.dirname(args.manifestPath);
  const timelinePath = path.join(runDir, "timeline.json");
  const callsPath = path.join(runDir, "calls.json");
  const verificationPath = path.join(runDir, "verification.json");
  const runReportPath = path.join(runDir, "run_report.md");
  const runReport = buildRunReport({
    summary,
    timeline,
    calls: modelCalls,
    verificationSteps,
    finalText,
  });

  for (const filePath of [
    args.manifestPath,
    args.eventsPath,
    args.summaryPath,
    args.stdoutPath,
    args.finalPath,
    timelinePath,
    callsPath,
    verificationPath,
    runReportPath,
  ]) {
    await ensureParent(filePath);
  }

  await fs.writeFile(args.manifestPath, JSON.stringify(manifest, null, 2) + "\n", "utf8");
  await fs.writeFile(
    args.eventsPath,
    events.map((event) => JSON.stringify(event)).join("\n") + "\n",
    "utf8",
  );
  await fs.writeFile(args.summaryPath, JSON.stringify(summary, null, 2) + "\n", "utf8");
  await fs.writeFile(args.stdoutPath, finalText ? `${finalText}\n` : "", "utf8");
  await fs.writeFile(args.finalPath, JSON.stringify(finalResult, null, 2) + "\n", "utf8");
  await fs.writeFile(timelinePath, JSON.stringify(timeline, null, 2) + "\n", "utf8");
  await fs.writeFile(callsPath, JSON.stringify(modelCalls, null, 2) + "\n", "utf8");
  await fs.writeFile(verificationPath, JSON.stringify(verificationSteps, null, 2) + "\n", "utf8");
  await fs.writeFile(runReportPath, runReport, "utf8");

  if (finalText) {
    process.stdout.write(`${finalText}\n`);
  }
  if (failureMessage) {
    process.stderr.write(`${failureMessage}\n`);
  }
  process.exitCode = exitCode;
}

void main().catch((error) => {
  process.stderr.write(
    `${error instanceof Error ? (error.stack ?? error.message) : String(error)}\n`,
  );
  process.exitCode = 1;
});
