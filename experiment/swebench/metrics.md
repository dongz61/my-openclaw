# SWE-bench Trace And Usage Notes

这份说明服务于 `experiment/swebench/scripts/run_one_swebench.py` 和
`experiment/swebench/scripts/run_openclaw_trace.ts` 产出的运行记录。

目标只有两个：

- 明确每个输出文件该怎么读
- 明确 token / cost 统计的口径，避免把不同语义的 usage 混在一起

## 产物说明

一次 run 的核心产物位于 `experiment/swebench/outputs/runs/<instance_id>/`。

### `manifest.json`

记录 run 的基本元数据：

- `runId`
- `instanceId`
- `agentId`
- `sessionId`
- `workspaceDir`
- `promptFile`
- `runProtocol`
- `startedAt`
- `endedAt`

它回答的是“这次 run 是什么”。

### `events.jsonl`

按时间顺序记录完整事件流，用于重建执行过程。

当前主要包含：

- `runner`：脚本开始、结束、报错
- `agent_event`：agent lifecycle、assistant 文本流、tool 事件、usage 事件
- `tool_usage`：从 `agent_event.stream === "usage"` 单独抽出的 usage 事件，便于分析
- `tool_result`：工具结果文本

它回答的是“这次 run 是怎么走完的”。

### `summary.json`

面向实验统计的摘要，优先用于聚合和横向比较。

当前重点字段：

- `durationMs`
- `success`
- `aborted`
- `stopReason`
- `toolCallsTotal`
- `toolCallsByName`
- `inputTokens`
- `outputTokens`
- `totalTokens`
- `promptTokens`
- `estimatedCostUsd`
- `usageSource`

它回答的是“这次 run 的总体成本和结果是什么”。

### `final.json`

面向调试和复盘，保留较完整的 agent 最终返回结构。

当前重点字段：

- `payloads`
- `meta`
- `usageDebug`
- `finalText`
- `runtimeLogs`
- `runtimeErrors`

它回答的是“agent 最后返回了什么，以及底层记录了哪些调试信息”。

## Usage 口径

最容易混淆的是 `summary.json` 和 `final.json` 里的 usage 字段。

### 1. 整次 run 的累计 usage

这是“整次求解总共消耗了多少 token”。

适合用于：

- 成本统计
- baseline 对比
- 多 agent 方案前后对比

在当前实验记录里，优先看：

- `summary.json.inputTokens`
- `summary.json.outputTokens`
- `summary.json.totalTokens`
- `summary.json.estimatedCostUsd`

其中：

- `inputTokens` 和 `outputTokens` 是整次 run 内多次模型调用的累计值
- `totalTokens` 也应按整次 run 的累计语义理解
- `estimatedCostUsd` 应和累计 usage 对应

### 2. 最后一轮模型调用的 usage

这是“agent 最终完成回答时，最后一次模型调用本身用了多少 token”。

它更接近：

- 最终上下文窗口大小
- 最后一轮推理负担
- 当前上下文是否过长

在当前实验记录里，优先看：

- `final.json.meta.agentMeta.lastCallUsage`
- `final.json.meta.agentMeta.promptTokens`

不要把它直接当成整次 run 成本。

### 3. `agentMeta.usage` 的特殊性

`final.json.meta.agentMeta.usage` 是一个容易误读的字段。

在当前 OpenClaw 实现里，它不是一个完全“单一语义”的对象：

- `usage.input` / `usage.output` 倾向于反映整次 run 的累计 usage
- `usage.total` 可能会被修正为“最后一轮调用的 total”

因此可能出现这种情况：

- `usage.input` 很大，表示累计输入 token
- `usage.output` 很大，表示累计输出 token
- 但 `usage.total` 明显更小，因为它表示最后一轮 total

这不是数据损坏，而是字段语义混合。

结论很简单：

- 做实验统计时，不直接使用 `final.json.meta.agentMeta.usage.total`
- 做 run 成本分析时，优先使用 `summary.json`
- 做上下文长度分析时，优先使用 `lastCallUsage` 和 `promptTokens`

## 当前推荐读法

如果目标是回答“这次 run 花了多少”：

- 看 `summary.json.inputTokens`
- 看 `summary.json.outputTokens`
- 看 `summary.json.totalTokens`
- 看 `summary.json.estimatedCostUsd`

如果目标是回答“最后一轮上下文有多大”：

- 看 `final.json.meta.agentMeta.lastCallUsage`
- 看 `final.json.meta.agentMeta.promptTokens`

如果目标是回答“成本主要耗在哪些步骤”：

- 先看 `events.jsonl` 中的 `tool_usage`
- 按时间顺序看每次 `model_call_complete`
- 观察 tool 前后 usage 的增长

## 推荐实验指标

当前阶段建议把实验指标分成两组，不要混用。

### A. Run-level 指标

用于比较方案成本和效果：

- 是否成功产出 patch
- 是否通过 SWE-bench eval
- `durationMs`
- `inputTokens`
- `outputTokens`
- `totalTokens`
- `estimatedCostUsd`
- `toolCallsTotal`

### B. Context-level 指标

用于分析上下文压力和架构拆解方向：

- `promptTokens`
- `lastCallUsage.total`
- 最后一轮调用前后的工具序列

前者回答“方案贵不贵、值不值”。
后者回答“为什么贵、应该拆哪一段”。

## 一个实际例子

以 `astropy__astropy-12907` 为例：

- `summary.json.totalTokens` 应按整次 run 累计成本理解
- `final.json.meta.agentMeta.lastCallUsage.total` 应按最后一轮调用理解
- 如果两者差很多，说明这次 run 不是“单次大推理”，而是“多轮工具交互 + 多次模型调用”

这正是后续做 workflow 拆解时需要重点关注的信号。

## 当前结论

在现阶段：

- `summary.json` 是实验统计主入口
- `final.json` 是调试和复盘主入口
- `events.jsonl` 是过程分析主入口

不要把三者当成重复文件；它们服务的是不同层级的问题。
