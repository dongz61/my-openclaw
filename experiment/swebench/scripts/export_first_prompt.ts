#!/usr/bin/env -S node --import tsx

import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import type { AgentTool } from "@mariozechner/pi-agent-core";
import { resolveSessionAgentIds } from "../../../src/agents/agent-scope.js";
import {
  analyzeBootstrapBudget,
  appendBootstrapPromptWarning,
  buildBootstrapInjectionStats,
  buildBootstrapPromptWarning,
} from "../../../src/agents/bootstrap-budget.js";
import {
  makeBootstrapWarn,
  resolveBootstrapContextForRun,
} from "../../../src/agents/bootstrap-files.js";
import { resolveOpenClawDocsPath } from "../../../src/agents/docs-path.js";
import { buildModelAliasLines } from "../../../src/agents/model-alias-lines.js";
import {
  normalizeProviderId,
  parseModelRef,
  resolveDefaultModelForAgent,
} from "../../../src/agents/model-selection.js";
import { resolveOwnerDisplaySetting } from "../../../src/agents/owner-display.js";
import {
  resolveBootstrapMaxChars,
  resolveBootstrapPromptTruncationWarningMode,
  resolveBootstrapTotalMaxChars,
} from "../../../src/agents/pi-embedded-helpers.js";
import { buildEmbeddedSystemPrompt } from "../../../src/agents/pi-embedded-runner/system-prompt.js";
import { detectRuntimeShell } from "../../../src/agents/shell-utils.js";
import { resolveSkillsPromptForRun } from "../../../src/agents/skills.js";
import { buildSystemPromptParams } from "../../../src/agents/system-prompt-params.js";
import { buildSystemPromptReport } from "../../../src/agents/system-prompt-report.js";
import { DEFAULT_BOOTSTRAP_FILENAME } from "../../../src/agents/workspace.js";
import { resolveHeartbeatPrompt } from "../../../src/auto-reply/heartbeat.js";
import { loadConfig } from "../../../src/config/config.js";
import { isCronSessionKey, isSubagentSessionKey } from "../../../src/routing/session-key.js";
import { buildTtsSystemPromptHint } from "../../../src/tts/tts.js";

type CliArgs = {
  baseDir?: string;
  experimentTag?: string;
  instanceId?: string;
  runDir?: string;
  workspaceDir?: string;
  promptFile?: string;
  outputDir?: string;
  agentId?: string;
  sessionId?: string;
  sessionKey?: string;
  modelRef?: string;
  trigger?: "user" | "manual" | "heartbeat" | "cron" | "memory" | "overflow";
};

type RunManifest = {
  instanceId?: string;
  agentId?: string;
  sessionId?: string;
  workspaceDir?: string;
  promptFile?: string;
};

function parseArgs(argv: string[]): CliArgs {
  const map = new Map<string, string>();
  for (let i = 0; i < argv.length; i += 1) {
    const token = argv[i];
    if (!token?.startsWith("--")) {
      continue;
    }
    const value = argv[i + 1];
    if (!value || value.startsWith("--")) {
      throw new Error(`Missing value for ${token}`);
    }
    map.set(token.slice(2), value);
    i += 1;
  }

  const triggerRaw = map.get("trigger")?.trim().toLowerCase();
  const trigger =
    triggerRaw === "user" ||
    triggerRaw === "manual" ||
    triggerRaw === "heartbeat" ||
    triggerRaw === "cron" ||
    triggerRaw === "memory" ||
    triggerRaw === "overflow"
      ? triggerRaw
      : undefined;

  return {
    baseDir: map.get("base-dir")?.trim() || undefined,
    experimentTag: map.get("experiment-tag")?.trim() || undefined,
    instanceId: map.get("instance-id")?.trim() || undefined,
    runDir: map.get("run-dir")?.trim() || undefined,
    workspaceDir: map.get("workspace-dir")?.trim() || undefined,
    promptFile: map.get("prompt-file")?.trim() || undefined,
    outputDir: map.get("output-dir")?.trim() || undefined,
    agentId: map.get("agent-id")?.trim() || undefined,
    sessionId: map.get("session-id")?.trim() || undefined,
    sessionKey: map.get("session-key")?.trim() || undefined,
    modelRef: map.get("model-ref")?.trim() || undefined,
    trigger,
  };
}

async function readJsonFile<T>(filePath: string): Promise<T | null> {
  try {
    return JSON.parse(await fs.readFile(filePath, "utf8")) as T;
  } catch {
    return null;
  }
}

function resolvePromptModeForSession(sessionKey?: string): "minimal" | "full" {
  if (!sessionKey) {
    return "full";
  }
  return isSubagentSessionKey(sessionKey) || isCronSessionKey(sessionKey) ? "minimal" : "full";
}

function shouldInjectHeartbeatPrompt(params: {
  isDefaultAgent: boolean;
  trigger?: CliArgs["trigger"];
}): boolean {
  if (!params.isDefaultAgent) {
    return false;
  }
  return params.trigger !== "cron";
}

function buildMarkdown(params: {
  systemPrompt: string;
  userPrompt: string;
  firstRequestView: string;
}): string {
  return [
    "# First Model Request",
    "",
    "## System Prompt",
    "",
    "```text",
    params.systemPrompt.trimEnd(),
    "```",
    "",
    "## User Prompt",
    "",
    "```text",
    params.userPrompt.trimEnd(),
    "```",
    "",
    "## Combined View",
    "",
    "```text",
    params.firstRequestView.trimEnd(),
    "```",
    "",
  ].join("\n");
}

function sanitizeTag(value: string): string {
  const normalized = Array.from(value.trim())
    .map((char) => (/[A-Za-z0-9._-]/.test(char) ? char : "-"))
    .join("")
    .replace(/^[.-]+|[.-]+$/g, "");
  if (!normalized) {
    throw new Error("experiment tag is empty after sanitization");
  }
  return normalized;
}

function resolveExperimentDir(baseDir: string, experimentTag: string): string {
  return experimentTag === "baseline"
    ? path.join(baseDir, "outputs_baseline")
    : path.join(baseDir, `outputs_${experimentTag}`);
}

function createLightweightPromptTools(): AgentTool[] {
  const names = [
    "read",
    "write",
    "edit",
    "apply_patch",
    "grep",
    "find",
    "ls",
    "exec",
    "process",
    "browser",
    "canvas",
    "nodes",
    "cron",
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
  ];
  return names.map(
    (name) =>
      ({
        name,
        description: `${name} tool`,
        parameters: { type: "object", properties: {} },
      }) as AgentTool,
  );
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));
  const cfg = loadConfig();

  let manifest: RunManifest | null = null;
  let reportSessionKey: string | undefined;
  let reportModelRef: string | undefined;

  if (args.runDir) {
    manifest = await readJsonFile<RunManifest>(path.join(args.runDir, "manifest.json"));
    const finalJson = await readJsonFile<{
      meta?: {
        systemPromptReport?: {
          sessionKey?: string;
          provider?: string;
          model?: string;
        };
      };
    }>(path.join(args.runDir, "final.json"));
    reportSessionKey = finalJson?.meta?.systemPromptReport?.sessionKey;
    const provider = finalJson?.meta?.systemPromptReport?.provider?.trim();
    const model = finalJson?.meta?.systemPromptReport?.model?.trim();
    reportModelRef = provider && model ? `${provider}/${model}` : undefined;
  }

  const workspaceDir = args.workspaceDir ?? manifest?.workspaceDir;
  const promptFile = args.promptFile ?? manifest?.promptFile;
  const instanceId =
    args.instanceId ??
    manifest?.instanceId ??
    (workspaceDir ? path.basename(workspaceDir) : undefined);
  const agentId = args.agentId ?? manifest?.agentId;
  const sessionId = args.sessionId ?? manifest?.sessionId ?? agentId;
  const sessionKey = args.sessionKey ?? reportSessionKey;
  const modelRefRaw = args.modelRef ?? reportModelRef ?? "modelstudio/qwen3.5-plus";
  const baseDir = path.resolve(
    args.baseDir ?? path.join(path.dirname(new URL(import.meta.url).pathname), ".."),
  );
  const experimentTag = sanitizeTag(args.experimentTag ?? "baseline");

  if (!workspaceDir) {
    throw new Error("Missing workspace directory. Pass --workspace-dir or --run-dir.");
  }
  if (!promptFile) {
    throw new Error("Missing prompt file. Pass --prompt-file or --run-dir.");
  }
  if (!sessionId) {
    throw new Error("Missing session id. Pass --session-id or provide a manifest via --run-dir.");
  }

  const parsedModelRef = parseModelRef(modelRefRaw, "modelstudio");
  if (!parsedModelRef) {
    throw new Error(`Invalid --model-ref: ${modelRefRaw}`);
  }
  const provider = normalizeProviderId(parsedModelRef.provider);
  const model = parsedModelRef.model;

  const resolvedWorkspaceDir = path.resolve(workspaceDir);
  const resolvedPromptFile = path.resolve(promptFile);
  const outputDir = path.resolve(
    args.outputDir ??
      (args.runDir
        ? path.join(args.runDir, "prompt-export")
        : instanceId
          ? path.join(
              resolveExperimentDir(baseDir, experimentTag),
              "runs",
              instanceId,
              "prompt-export",
            )
          : path.join(resolvedWorkspaceDir, ".prompt-export")),
  );

  const userPromptRaw = await fs.readFile(resolvedPromptFile, "utf8");
  const { bootstrapFiles, contextFiles } = await resolveBootstrapContextForRun({
    workspaceDir: resolvedWorkspaceDir,
    config: cfg,
    sessionKey,
    sessionId,
    warn: makeBootstrapWarn({
      sessionLabel: sessionKey ?? sessionId,
      warn: () => {},
    }),
  });

  const skillsPrompt =
    resolveSkillsPromptForRun({
      config: cfg,
      workspaceDir: resolvedWorkspaceDir,
    }) ?? "";

  const bootstrapStats = buildBootstrapInjectionStats({
    bootstrapFiles,
    injectedFiles: contextFiles,
  });
  const bootstrapMaxChars = resolveBootstrapMaxChars(cfg);
  const bootstrapTotalMaxChars = resolveBootstrapTotalMaxChars(cfg);
  const bootstrapAnalysis = analyzeBootstrapBudget({
    files: bootstrapStats,
    bootstrapMaxChars,
    bootstrapTotalMaxChars,
  });
  const bootstrapPromptWarning = buildBootstrapPromptWarning({
    analysis: bootstrapAnalysis,
    mode: resolveBootstrapPromptTruncationWarningMode(cfg),
    seenSignatures: [],
  });

  const { defaultAgentId, sessionAgentId } = resolveSessionAgentIds({
    sessionKey,
    config: cfg,
    agentId,
  });
  const isDefaultAgent = sessionAgentId === defaultAgentId;
  const heartbeatPrompt = shouldInjectHeartbeatPrompt({
    isDefaultAgent,
    trigger: args.trigger,
  })
    ? resolveHeartbeatPrompt(cfg?.agents?.defaults?.heartbeat?.prompt)
    : undefined;
  const promptMode = resolvePromptModeForSession(sessionKey);
  const workspaceNotes = bootstrapFiles.some(
    (file) => file.name === DEFAULT_BOOTSTRAP_FILENAME && !file.missing,
  )
    ? ["Reminder: commit your changes in this workspace after edits."]
    : undefined;

  // Keep this exporter lightweight: we only need a stable tool list for prompt reading,
  // not the full runtime tool initialization path.
  const tools = createLightweightPromptTools();

  const defaultModelRef = resolveDefaultModelForAgent({
    cfg: cfg ?? {},
    agentId: sessionAgentId,
  });
  const { runtimeInfo, userTimezone, userTime, userTimeFormat } = buildSystemPromptParams({
    config: cfg,
    agentId: sessionAgentId,
    workspaceDir: resolvedWorkspaceDir,
    cwd: resolvedWorkspaceDir,
    runtime: {
      host: os.hostname(),
      os: `${os.type()} ${os.release()}`,
      arch: os.arch(),
      node: process.version,
      model: `${provider}/${model}`,
      defaultModel: `${defaultModelRef.provider}/${defaultModelRef.model}`,
      shell: detectRuntimeShell(),
    },
  });
  const docsPath = await resolveOpenClawDocsPath({
    workspaceDir: resolvedWorkspaceDir,
    argv1: process.argv[1],
    cwd: resolvedWorkspaceDir,
    moduleUrl: import.meta.url,
  });
  const ownerDisplay = resolveOwnerDisplaySetting(cfg);
  const systemPrompt = buildEmbeddedSystemPrompt({
    workspaceDir: resolvedWorkspaceDir,
    defaultThinkLevel: undefined,
    reasoningLevel: "off",
    extraSystemPrompt: undefined,
    ownerNumbers: undefined,
    ownerDisplay: ownerDisplay.ownerDisplay,
    ownerDisplaySecret: ownerDisplay.ownerDisplaySecret,
    reasoningTagHint: false,
    heartbeatPrompt,
    skillsPrompt,
    docsPath: docsPath ?? undefined,
    ttsHint: cfg ? buildTtsSystemPromptHint(cfg) : undefined,
    workspaceNotes,
    promptMode,
    acpEnabled: cfg?.acp?.enabled !== false,
    runtimeInfo,
    messageToolHints: undefined,
    sandboxInfo: undefined,
    reactionGuidance: undefined,
    tools,
    modelAliasLines: buildModelAliasLines(cfg),
    userTimezone,
    userTime,
    userTimeFormat,
    contextFiles,
    memoryCitationsMode: cfg?.memory?.citations,
  });
  const userPrompt = appendBootstrapPromptWarning(userPromptRaw, bootstrapPromptWarning.lines, {
    preserveExactPrompt: heartbeatPrompt,
  });
  const firstRequestView = [
    "===== SYSTEM PROMPT =====",
    systemPrompt.trimEnd(),
    "",
    "===== USER PROMPT =====",
    userPrompt.trimEnd(),
    "",
  ].join("\n");

  const systemPromptReport = buildSystemPromptReport({
    source: "run",
    generatedAt: Date.now(),
    sessionId,
    sessionKey,
    provider,
    model,
    workspaceDir: resolvedWorkspaceDir,
    bootstrapMaxChars,
    bootstrapTotalMaxChars,
    systemPrompt,
    bootstrapFiles,
    injectedFiles: contextFiles,
    skillsPrompt,
    tools,
  });

  await fs.mkdir(outputDir, { recursive: true });
  await fs.writeFile(path.join(outputDir, "system_prompt.txt"), `${systemPrompt}\n`, "utf8");
  await fs.writeFile(path.join(outputDir, "user_prompt.txt"), `${userPrompt}\n`, "utf8");
  await fs.writeFile(path.join(outputDir, "first_request.txt"), firstRequestView, "utf8");
  await fs.writeFile(
    path.join(outputDir, "first_request.md"),
    buildMarkdown({ systemPrompt, userPrompt, firstRequestView }),
    "utf8",
  );
  await fs.writeFile(
    path.join(outputDir, "prompt_report.json"),
    `${JSON.stringify(
      {
        instanceId,
        experimentTag,
        sessionId,
        sessionKey,
        agentId: sessionAgentId,
        workspaceDir: resolvedWorkspaceDir,
        promptFile: resolvedPromptFile,
        modelRef: `${provider}/${model}`,
        bootstrapWarningLines: bootstrapPromptWarning.lines,
        systemPromptReport,
      },
      null,
      2,
    )}\n`,
    "utf8",
  );

  console.log(`Exported first prompt view to ${outputDir}`);
  process.exit(0);
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exitCode = 1;
});
