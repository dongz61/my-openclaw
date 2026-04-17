import type {
  ModelDefinitionConfig,
  ModelProviderConfig,
} from "openclaw/plugin-sdk/provider-models";

export const MODELSTUDIO_BASE_URL = "https://coding-intl.dashscope.aliyuncs.com/v1";
export const MODELSTUDIO_DEFAULT_MODEL_ID = "qwen3.5-plus";

// Model Studio pricing is tiered for several models, but the current model
// catalog only supports one static price per model. Use the official first-tier
// real-time price as the default baseline.
const MODELSTUDIO_QWEN_35_PLUS_COST = {
  input: 0.8,
  output: 4.8,
  cacheRead: 0,
  cacheWrite: 0,
};
const MODELSTUDIO_QWEN_3_MAX_COST = {
  input: 2.5,
  output: 10,
  cacheRead: 0,
  cacheWrite: 0,
};
const MODELSTUDIO_QWEN_3_CODER_NEXT_COST = {
  input: 2.202,
  output: 11.009,
  cacheRead: 0,
  cacheWrite: 0,
};
const MODELSTUDIO_QWEN_3_CODER_PLUS_COST = {
  input: 4,
  output: 16,
  cacheRead: 0,
  cacheWrite: 0,
};
const MODELSTUDIO_MINIMAX_M25_COST = {
  input: 2.1,
  output: 8.4,
  cacheRead: 0,
  cacheWrite: 0,
};
const MODELSTUDIO_GLM_5_COST = {
  input: 4,
  output: 18,
  cacheRead: 0,
  cacheWrite: 0,
};
const MODELSTUDIO_GLM_47_COST = {
  input: 3,
  output: 14,
  cacheRead: 0,
  cacheWrite: 0,
};
const MODELSTUDIO_KIMI_K25_COST = {
  input: 4,
  output: 21,
  cacheRead: 0,
  cacheWrite: 0,
};

const MODELSTUDIO_MODEL_CATALOG: ReadonlyArray<ModelDefinitionConfig> = [
  {
    id: "qwen3.5-plus",
    name: "qwen3.5-plus",
    reasoning: false,
    input: ["text", "image"],
    cost: MODELSTUDIO_QWEN_35_PLUS_COST,
    contextWindow: 1_000_000,
    maxTokens: 65_536,
  },
  {
    id: "qwen3-max-2026-01-23",
    name: "qwen3-max-2026-01-23",
    reasoning: false,
    input: ["text"],
    cost: MODELSTUDIO_QWEN_3_MAX_COST,
    contextWindow: 262_144,
    maxTokens: 65_536,
  },
  {
    id: "qwen3-coder-next",
    name: "qwen3-coder-next",
    reasoning: false,
    input: ["text"],
    cost: MODELSTUDIO_QWEN_3_CODER_NEXT_COST,
    contextWindow: 262_144,
    maxTokens: 65_536,
  },
  {
    id: "qwen3-coder-plus",
    name: "qwen3-coder-plus",
    reasoning: false,
    input: ["text"],
    cost: MODELSTUDIO_QWEN_3_CODER_PLUS_COST,
    contextWindow: 1_000_000,
    maxTokens: 65_536,
  },
  {
    id: "MiniMax-M2.5",
    name: "MiniMax-M2.5",
    reasoning: true,
    input: ["text"],
    cost: MODELSTUDIO_MINIMAX_M25_COST,
    contextWindow: 1_000_000,
    maxTokens: 65_536,
  },
  {
    id: "glm-5",
    name: "glm-5",
    reasoning: false,
    input: ["text"],
    cost: MODELSTUDIO_GLM_5_COST,
    contextWindow: 202_752,
    maxTokens: 16_384,
  },
  {
    id: "glm-4.7",
    name: "glm-4.7",
    reasoning: false,
    input: ["text"],
    cost: MODELSTUDIO_GLM_47_COST,
    contextWindow: 202_752,
    maxTokens: 16_384,
  },
  {
    id: "kimi-k2.5",
    name: "kimi-k2.5",
    reasoning: false,
    input: ["text", "image"],
    cost: MODELSTUDIO_KIMI_K25_COST,
    contextWindow: 262_144,
    maxTokens: 32_768,
  },
];

export function buildModelStudioProvider(): ModelProviderConfig {
  return {
    baseUrl: MODELSTUDIO_BASE_URL,
    api: "openai-completions",
    models: MODELSTUDIO_MODEL_CATALOG.map((model) => ({ ...model })),
  };
}
