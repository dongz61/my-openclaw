type NormalizedUsageLike = {
  input?: number;
  output?: number;
  cacheRead?: number;
  cacheWrite?: number;
  total?: number;
};

export type ProviderUsageCapture = {
  url: string;
  status: number;
  mode: "stream" | "json";
  model?: string;
  usage?: NormalizedUsageLike;
  rawUsage?: Record<string, unknown>;
};

export const providerUsageCaptures: ProviderUsageCapture[] = [];
export const providerUsageCaptureLogs: string[] = [];

function isDashScopeUrl(input: string): boolean {
  try {
    const host = new URL(input).hostname.toLowerCase();
    return host === "dashscope.aliyuncs.com" || host === "dashscope-intl.aliyuncs.com";
  } catch {
    return false;
  }
}

function asUsageRecord(value: unknown): Record<string, unknown> | undefined {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : undefined;
}

function asFiniteNumber(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function normalizeProviderUsage(
  raw: Record<string, unknown> | undefined,
): NormalizedUsageLike | undefined {
  if (!raw) {
    return undefined;
  }
  const promptTokens = asFiniteNumber(raw.prompt_tokens);
  const completionTokens = asFiniteNumber(raw.completion_tokens);
  const totalTokens = asFiniteNumber(raw.total_tokens);
  const promptDetails = asUsageRecord(raw.prompt_tokens_details);
  const completionDetails = asUsageRecord(raw.completion_tokens_details);
  const cachedTokens = asFiniteNumber(promptDetails?.cached_tokens) ?? 0;
  const reasoningTokens = asFiniteNumber(completionDetails?.reasoning_tokens) ?? 0;
  const input =
    typeof promptTokens === "number" ? Math.max(0, promptTokens - cachedTokens) : undefined;
  const output =
    typeof completionTokens === "number"
      ? Math.max(0, completionTokens + reasoningTokens)
      : undefined;
  const total =
    totalTokens ??
    (typeof input === "number" || typeof output === "number" || cachedTokens > 0
      ? (input ?? 0) + (output ?? 0) + cachedTokens
      : undefined);

  if (input === undefined && output === undefined && cachedTokens === 0 && total === undefined) {
    return undefined;
  }

  return {
    ...(typeof input === "number" ? { input } : {}),
    ...(typeof output === "number" ? { output } : {}),
    ...(cachedTokens > 0 ? { cacheRead: cachedTokens } : {}),
    ...(typeof total === "number" ? { total } : {}),
  };
}

function extractStreamUsageFromText(text: string): Record<string, unknown> | undefined {
  for (const line of text.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed.startsWith("data:")) {
      continue;
    }
    const payload = trimmed.slice("data:".length).trim();
    if (!payload || payload === "[DONE]") {
      continue;
    }
    try {
      const parsed = JSON.parse(payload) as Record<string, unknown>;
      const usage = asUsageRecord(parsed.usage);
      if (usage) {
        return usage;
      }
    } catch {
      continue;
    }
  }
  return undefined;
}

const originalFetch = globalThis.fetch.bind(globalThis);

globalThis.fetch = async (input, init) => {
  const isRequestObject =
    typeof input !== "string" && !(input instanceof URL) && typeof input.clone === "function";
  const url =
    typeof input === "string" ? input : input instanceof URL ? input.toString() : input.url;
  const inputKind =
    typeof input === "string"
      ? "string"
      : input instanceof URL
        ? "URL"
        : input?.constructor?.name || typeof input;
  const requestMethod = init?.method ?? (isRequestObject ? input.method : undefined) ?? "GET";
  const requestHeaders = new Headers(
    init?.headers ?? (isRequestObject ? input.headers : undefined) ?? undefined,
  );
  const requestContentType = requestHeaders.get("content-type") ?? undefined;

  const response = await originalFetch(input, init);

  if (!isDashScopeUrl(url)) {
    return response;
  }

  void (async () => {
    try {
      const clone = response.clone();
      const contentType = clone.headers.get("content-type")?.toLowerCase() ?? "";
      const bodyText = await clone.text();
      if (!bodyText.trim()) {
        providerUsageCaptureLogs.push(
          `dashscope capture skipped(empty): ${url} inputKind=${inputKind} requestMethod=${requestMethod} contentType=${requestContentType ?? "undefined"}`,
        );
        return;
      }

      let rawUsage: Record<string, unknown> | undefined;
      let mode: ProviderUsageCapture["mode"] = "json";
      let model: string | undefined;

      if (contentType.includes("text/event-stream") || bodyText.includes("data:")) {
        mode = "stream";
        rawUsage = extractStreamUsageFromText(bodyText);
      } else {
        const parsed = JSON.parse(bodyText) as Record<string, unknown>;
        rawUsage = asUsageRecord(parsed.usage);
        model = typeof parsed.model === "string" ? parsed.model : undefined;
      }

      if (!rawUsage) {
        providerUsageCaptures.push({
          url,
          status: response.status,
          mode,
          model,
        });
        providerUsageCaptureLogs.push(
          `dashscope capture no-usage: ${url} inputKind=${inputKind} requestMethod=${requestMethod} contentType=${requestContentType ?? "undefined"}`,
        );
        return;
      }

      providerUsageCaptures.push({
        url,
        status: response.status,
        mode,
        model,
        usage: normalizeProviderUsage(rawUsage),
        rawUsage,
      });
      providerUsageCaptureLogs.push(`dashscope capture ok(${mode}): ${url}`);
    } catch (error) {
      providerUsageCaptureLogs.push(
        `dashscope capture error: ${url}: ${error instanceof Error ? error.message : String(error)}`,
      );
    }
  })();

  return response;
};
