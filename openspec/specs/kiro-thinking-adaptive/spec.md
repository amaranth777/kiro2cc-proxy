# Spec: 账号级 thinking adaptive 注入（kiro-thinking-adaptive）

## Purpose

提供账号粒度的 Kiro thinking 最终裁决开关，默认开启：路由到开关开启账号且模型支持原生 thinking 时，代理即使客户端未显式请求也向 Kiro 上游 `additionalModelRequestFields` 注入 `thinking: {"type": "adaptive"}`；开关关闭时剥离请求侧 thinking 控制并丢弃响应侧推理内容。模型目录仅展示普通模型 ID，`-thinking` / `-Thinking` 旧别名保留请求兼容但不单独展示。

## Requirements

### Requirement: 账号级 thinking adaptive 开关持久化

系统 SHALL 在 `KiroCredentials` 上持久化账号级开关 `thinkingAdaptive`（bool，serde default=true），经 Admin API `PUT /credentials/{id}` 的 `thinkingAdaptive` 字段（`Option<bool>`，未提供时不更新）变更，并随账号快照（`CredentialEntrySnapshot`）透出给前端。

#### Scenario: 旧配置文件向后兼容

- **GIVEN** credentials.json 由旧版本生成，某账号对象不含 `thinkingAdaptive` 字段
- **WHEN** 服务加载该 credentials.json
- **THEN** 该账号的 `thinking_adaptive` 反序列化为 `true`
- **AND** credentials.json 中不因加载而出现序列化噪音（bool 字段照常输出，与 `disabled` 字段行为一致）

#### Scenario: Admin API 部分更新开关

- **GIVEN** 存在账号 #N，当前 `thinkingAdaptive = false`
- **WHEN** Admin 调用 `PUT /credentials/N`，body 携带 `{"thinkingAdaptive": true}`
- **THEN** 账号 #N 的 `thinking_adaptive` 更新为 `true` 并持久化
- **AND** 同请求未提供的其他字段（refresh_token、proxy_url 等）不被修改
- **AND** 快照接口返回该账号 `thinkingAdaptive: true`

#### Scenario: 未提供该字段时不更新

- **WHEN** Admin 调用 `PUT /credentials/N`，body 不含 `thinkingAdaptive`
- **THEN** 账号 #N 的 `thinking_adaptive` 保持原值不变

### Requirement: 账号开关是 thinking 的最终裁决

Provider 层 SHALL 在确定账号（`CallContext`）后执行账号开关判定：`thinkingAdaptive == false` 时 SHALL 剥离原生 thinking 字段及 converter 注入的 thinking 文本标签，并在响应侧丢弃推理内容；`thinkingAdaptive == true` 且模型支持原生 thinking 时 SHALL 强制注入 `thinking: {"type": "adaptive"}`，即使客户端未请求。显式 `thinking: {"type": "disabled"}` 也服从账号最终裁决。`thinking_adaptive_requested` 标记继续保留用于兼容现有透传链。

#### Scenario: 开关开启 + 客户端 adaptive thinking → 注入保留

- **GIVEN** 账号 A 的 `thinkingAdaptive = true`，客户端请求携带 `thinking: {"type": "adaptive"}`，模型为 claude-sonnet-4.6
- **WHEN** 请求路由到账号 A 并发往 Kiro 上游
- **THEN** request body 的 `additionalModelRequestFields` 含 `"thinking": {"type": "adaptive"}`
- **AND** 其余字段（output_config、max_tokens）保持既有构建逻辑不变

#### Scenario: 开关开启 + 客户端未请求 thinking → 默认开启

- **GIVEN** 账号 A 的 `thinkingAdaptive = true`，客户端请求未携带 `thinking`，模型为 claude-sonnet-4.6
- **WHEN** 请求路由到账号 A 并发往 Kiro 上游
- **THEN** request body 的 `additionalModelRequestFields` 含 `"thinking": {"type": "adaptive"}`

#### Scenario: 开关关闭 → 强制关闭

- **GIVEN** 账号 B 的 `thinkingAdaptive = false`，客户端请求携带 adaptive thinking
- **WHEN** 请求路由到账号 B
- **THEN** request body 不含 `additionalModelRequestFields.thinking`
- **AND** converter 注入的 `<thinking_mode>` 文本标签也被剥离
- **AND** 响应侧不向客户端转发推理内容

#### Scenario: 故障转移后按新账号重新判定

- **GIVEN** 账号 A（开关开）与账号 B（开关关）均可用，客户端请求携带 thinking
- **WHEN** 首选账号 A 请求失败，故障转移到账号 B 重试
- **THEN** 发往账号 B 的 request body 不含 thinking（按账号 B 的开关值判定）
- **AND** 每次重试都以当次实际选中的账号状态为准

### Requirement: 仅对支持原生 thinking 的模型默认开启

系统 SHALL 使用 `native_thinking_supported` 判定模型是否接受 Kiro 原生 `thinking: {"type": "adaptive"}` 字段。支持的模型在账号开关开启时默认注入；GPT 系、4.5 代际及未证明支持该字段的第三方模型 SHALL NOT 被强制注入，以免上游以 400 拒绝。

#### Scenario: 客户端携带 adaptive thinking
- **WHEN** 客户端向 `/v1/messages` 发送 `thinking: {"type": "adaptive"}`，模型映射为 Claude 非 4.5 代际
- **THEN** converter 生成的 `additionalModelRequestFields` 包含 `thinking` 字段，且 `output_config.effort` 与 `max_tokens` 仍按现有规则存在

#### Scenario: 客户端携带 enabled thinking 不注入原生字段
- **WHEN** 客户端发送 `thinking: {"type": "enabled", "budget_tokens": 20000}`，模型映射为 Claude 非 4.5 代际
- **THEN** `additionalModelRequestFields` 不含 `thinking` 字段（enabled 仅走 history[0] `<thinking_mode>` 文本标签协议，两套控制信号不得叠加，否则客户端规则遵从性回退，v3.4.1 回归）

#### Scenario: 客户端未请求 thinking 且模型不支持原生字段
- **WHEN** 请求不携带 `thinking` 字段
- **AND** 模型不属于 `native_thinking_supported`
- **THEN** `additionalModelRequestFields` 不含 `thinking` 字段

#### Scenario: 4.5 代际跳过
- **WHEN** 模型映射为 4.5 代际 Claude（如 claude-4-5-sonnet）
- **THEN** `additionalModelRequestFields` 整体为 None（维持现状），不含 `thinking` 字段

#### Scenario: GPT 系与 luna 不注入
- **WHEN** 请求携带 thinking 但模型为 GPT 系或 luna
- **THEN** `additionalModelRequestFields` 不含 `thinking` 字段（GPT 系维持 `reasoning.effort` 现状，luna 维持现状）

### Requirement: 模型目录隐藏 thinking 变体

`/v1/models` 与 Admin 模型列表 SHALL 对大小写不敏感地隐藏 ID 尾缀为 `-thinking` 的模型，只保留普通版本。旧客户端仍可请求 thinking 变体，`map_model()` SHALL 将其归一化到普通 Kiro 模型 ID。

#### Scenario: 隐藏静态与动态 thinking 变体

- **WHEN** 模型目录包含 `claude-opus-4-6`、`claude-opus-4-6-thinking` 与 `future-model-Thinking`
- **THEN** 对外列表包含 `claude-opus-4-6`
- **AND** 不包含任一尾缀为 `-thinking` / `-Thinking` 的条目

### Requirement: 上游推理事件转发（现有通路回归约束）
当上游返回 `reasoningContentEvent` 且 `thinking_enabled` 为 true 时，系统 SHALL 通过现有 `process_native_reasoning` 通路产生 Anthropic SSE `thinking_delta` 与 `signature_delta` 事件；本变更 SHALL NOT 修改该通路的行为。

#### Scenario: 思考文案渲染前提
- **WHEN** 上游返回非空 `reasoningContentEvent`
- **THEN** SSE 流包含 `content_block_delta`（`thinking_delta`）事件，CC 据此渲染思考文案

## 接口说明

本变更不改变 `/v1/messages` 与 `/cc/v1/messages` 的请求/响应格式；差异体现在模型目录可见性，以及代理 → Kiro 上游 request body 中按账号最终裁决注入或剥离 thinking 控制。
