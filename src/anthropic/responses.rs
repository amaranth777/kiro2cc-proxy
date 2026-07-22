// Copyright (c) 2026 Harllan He. Licensed under MIT.
//! OpenAI Responses API 兼容层。
//!
//! 请求侧归一化为现有 Anthropic Messages 请求并复用 Kiro 转换链路；响应侧直接解析
//! Kiro 事件，避免经过 Anthropic SSE 时丢失 function_call 与 reasoning 语义。

use std::{
    collections::HashMap,
    convert::Infallible,
    sync::{Arc, Mutex, OnceLock},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use axum::{
    Extension,
    body::Body,
    extract::State,
    http::{StatusCode, header},
    response::{IntoResponse, Json, Response},
};
use base64::Engine as _;
use bytes::Bytes;
use futures::{StreamExt, stream};
use serde::Deserialize;
use serde_json::{Map, Value, json};
use tokio::time::Instant;
use uuid::Uuid;

use crate::{
    kiro::{
        model::{events::Event, requests::kiro::KiroRequest},
        parser::decoder::EventStreamDecoder,
    },
    model::usage::UsageTracker,
    token,
};

use super::{
    converter::convert_request,
    middleware::{ApiKeyContext, AppState},
    types::{Message, MessagesRequest, OutputConfig, OutputFormat, SystemMessage, Thinking, Tool},
};

#[derive(Debug, Deserialize)]
struct ResponsesRequest {
    model: String,
    input: Value,
    #[serde(default)]
    instructions: Option<String>,
    #[serde(default)]
    tools: Option<Vec<Value>>,
    #[serde(default)]
    stream: bool,
    #[serde(default, alias = "max_tokens", alias = "max_completion_tokens")]
    max_output_tokens: Option<i32>,
    #[serde(default)]
    reasoning: Option<Value>,
    #[serde(default)]
    text: Option<Value>,
    #[serde(default)]
    response_format: Option<Value>,
    #[serde(default)]
    previous_response_id: Option<String>,
    #[serde(default = "default_true")]
    store: bool,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(default)]
    stream: bool,
    #[serde(default, alias = "max_completion_tokens")]
    max_tokens: Option<i32>,
    #[serde(default)]
    tools: Option<Vec<Value>>,
    #[serde(default)]
    functions: Option<Vec<Value>>,
    #[serde(default)]
    response_format: Option<Value>,
    #[serde(default)]
    reasoning: Option<Value>,
}

#[derive(Debug, Deserialize)]
struct ChatMessage {
    role: String,
    #[serde(default)]
    content: Option<Value>,
    #[serde(default)]
    tool_call_id: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<ChatToolCall>>,
    #[serde(default)]
    function_call: Option<ChatFunctionCall>,
}

#[derive(Debug, Deserialize)]
struct ChatToolCall {
    id: String,
    #[serde(default)]
    function: ChatFunctionCall,
}

#[derive(Debug, Default, Deserialize)]
struct ChatFunctionCall {
    #[serde(default)]
    name: String,
    #[serde(default)]
    arguments: String,
}

#[derive(Debug, Deserialize)]
struct CompletionRequest {
    model: String,
    prompt: Value,
    #[serde(default)]
    stream: bool,
    #[serde(default, alias = "max_completion_tokens")]
    max_tokens: Option<i32>,
}

fn default_true() -> bool {
    true
}

#[derive(Clone, Debug)]
struct StoredOutputItem {
    id: String,
    item: Value,
}

#[derive(Clone, Debug, Default)]
struct RequestSnapshot {
    messages: Vec<Message>,
    system: Option<Vec<SystemMessage>>,
    tools: Option<Vec<Tool>>,
}

impl From<&MessagesRequest> for RequestSnapshot {
    fn from(value: &MessagesRequest) -> Self {
        Self {
            messages: value.messages.clone(),
            system: value.system.clone(),
            tools: value.tools.clone(),
        }
    }
}

#[derive(Clone, Debug, Default)]
struct ParsedInput {
    messages: Vec<Message>,
    system: Vec<SystemMessage>,
    tools: Vec<Value>,
    output_items: Vec<StoredOutputItem>,
}

#[derive(Clone)]
struct StoredConversation {
    snapshot: RequestSnapshot,
    output_items: Vec<StoredOutputItem>,
    expires_at: Instant,
}

static RESPONSES: OnceLock<Mutex<HashMap<String, StoredConversation>>> = OnceLock::new();

fn conversations() -> &'static Mutex<HashMap<String, StoredConversation>> {
    RESPONSES.get_or_init(|| Mutex::new(HashMap::new()))
}

fn error_response(status: StatusCode, message: impl Into<String>) -> Response {
    (
        status,
        Json(json!({
            "error": {
                "type": "invalid_request_error",
                "message": message.into()
            }
        })),
    )
        .into_response()
}

fn unsupported_response(endpoint: &str) -> Response {
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(json!({
            "error": {
                "type": "unsupported_endpoint",
                "code": "unsupported_endpoint",
                "message": format!("{endpoint} is not supported by the Kiro backend")
            }
        })),
    )
        .into_response()
}

pub async fn unsupported_embeddings() -> Response {
    unsupported_response("/v1/embeddings")
}

pub async fn unsupported_images() -> Response {
    unsupported_response("/v1/images")
}

pub async fn unsupported_audio() -> Response {
    unsupported_response("/v1/audio")
}

fn text_content(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Array(items) => items
            .iter()
            .filter_map(|item| item.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    }
}

fn message(role: &str, content: Value) -> Message {
    Message {
        role: role.to_string(),
        content,
    }
}

fn push_user_block(messages: &mut Vec<Message>, block: Value) {
    if let Some(Message { content, .. }) = messages.last_mut().filter(|m| m.role == "user")
        && let Value::Array(blocks) = content
    {
        blocks.push(block);
        return;
    }
    messages.push(message("user", json!([block])));
}

fn image_block(image_url: &Value) -> Result<Value, String> {
    let url = image_url
        .as_str()
        .or_else(|| image_url.get("url").and_then(Value::as_str))
        .ok_or("input_image.image_url is required")?;
    let (header, data) = url
        .split_once(',')
        .filter(|(prefix, _)| prefix.starts_with("data:image/"))
        .ok_or("input_image only supports data URLs")?;
    let media_type = header
        .strip_prefix("data:")
        .and_then(|value| value.strip_suffix(";base64"))
        .ok_or("input_image data URL must be base64")?;
    if !matches!(media_type, "image/jpeg" | "image/png" | "image/gif" | "image/webp") {
        return Err(format!("unsupported input_image media type: {media_type}"));
    }
    base64::engine::general_purpose::STANDARD
        .decode(data)
        .map_err(|_| "input_image contains invalid base64 data")?;
    Ok(json!({
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": media_type,
            "data": data
        }
    }))
}

fn item_to_context_text(kind: &str, item: &Value) -> String {
    format!("OpenAI {kind} item:\n{item}")
}

fn message_content(content: &Value, role: &str) -> Result<Value, String> {
    if let Some(text) = content.as_str() {
        return Ok(json!(text));
    }
    let items = content
        .as_array()
        .ok_or("message.content must be a string or array")?;
    let mut blocks = Vec::new();
    for item in items {
        let kind = item.get("type").and_then(Value::as_str).unwrap_or("text");
        match kind {
            "input_text" | "output_text" | "text" | "summary_text" => {
                blocks.push(json!({
                    "type": "text",
                    "text": item.get("text").and_then(Value::as_str).unwrap_or_default()
                }));
            }
            "input_image" | "image_url" => blocks.push(image_block(
                item.get("image_url")
                    .or_else(|| item.get("image"))
                    .or_else(|| item.get("url"))
                    .ok_or("input_image.image_url is required")?,
            )?),
            "function_call" => {
                let id = item
                    .get("call_id")
                    .or_else(|| item.get("id"))
                    .and_then(Value::as_str)
                    .ok_or("function_call.call_id is required")?;
                let name = item
                    .get("name")
                    .and_then(Value::as_str)
                    .ok_or("function_call.name is required")?;
                let arguments = item
                    .get("arguments")
                    .and_then(Value::as_str)
                    .ok_or("function_call.arguments is required")?;
                let input: Value = serde_json::from_str(arguments)
                    .map_err(|_| "function_call.arguments must be valid JSON")?;
                blocks.push(json!({
                    "type": "tool_use",
                    "id": id,
                    "name": name,
                    "input": input
                }));
            }
            "function_call_output" => {
                let id = item
                    .get("call_id")
                    .and_then(Value::as_str)
                    .ok_or("function_call_output.call_id is required")?;
                let output = item.get("output").cloned().unwrap_or(Value::Null);
                let mut block = json!({
                    "type": "tool_result",
                    "tool_use_id": id,
                    "content": output_text(&output)
                });
                if item.get("status").and_then(Value::as_str) == Some("failed") {
                    block["is_error"] = json!(true);
                }
                blocks.push(block);
            }
            _ => return Err(format!("unsupported message content type: {kind}")),
        }
    }
    if role == "assistant"
        && blocks
            .iter()
            .any(|block| block.get("type") == Some(&json!("image")))
    {
        return Err("assistant messages cannot contain input_image".into());
    }
    Ok(Value::Array(blocks))
}

fn output_text(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

fn input_messages(input: &Value) -> Result<ParsedInput, String> {
    if let Some(text) = input.as_str() {
        return Ok(ParsedInput {
            messages: vec![message("user", json!(text))],
            ..ParsedInput::default()
        });
    }
    let items = input.as_array().ok_or("input must be a string or array")?;
    let mut parsed = ParsedInput::default();
    for item in items {
        let kind = item.get("type").and_then(Value::as_str).unwrap_or("message");
        match kind {
            "message" => {
                let role = item
                    .get("role")
                    .and_then(Value::as_str)
                    .ok_or("message.role is required")?;
                match role {
                    "user" | "assistant" => parsed.messages.push(message(
                        role,
                        message_content(item.get("content").unwrap_or(&Value::Null), role)?,
                    )),
                    "developer" | "system" => {
                        let text = text_content(item.get("content").unwrap_or(&Value::Null));
                        if !text.is_empty() {
                            parsed.system.push(SystemMessage { text });
                        }
                    }
                    _ => return Err(format!("unsupported message role: {role}")),
                }
            }
            "input_text" => push_user_block(
                &mut parsed.messages,
                json!({
                    "type": "text",
                    "text": item.get("text").and_then(Value::as_str).unwrap_or_default()
                }),
            ),
            "input_image" | "image_url" => push_user_block(
                &mut parsed.messages,
                image_block(
                    item.get("image_url")
                        .or_else(|| item.get("image"))
                        .or_else(|| item.get("url"))
                        .ok_or("input_image.image_url is required")?,
                )?,
            ),
            "function_call" => {
                let id = item
                    .get("call_id")
                    .or_else(|| item.get("id"))
                    .and_then(Value::as_str)
                    .ok_or("function_call.call_id is required")?;
                let name = item
                    .get("name")
                    .and_then(Value::as_str)
                    .ok_or("function_call.name is required")?;
                if name.is_empty() {
                    return Err("function_call.name must not be empty".into());
                }
                let arguments = item
                    .get("arguments")
                    .and_then(Value::as_str)
                    .ok_or("function_call.arguments is required")?;
                let input: Value = serde_json::from_str(arguments)
                    .map_err(|_| "function_call.arguments must be valid JSON")?;
                parsed.messages.push(message(
                    "assistant",
                    json!([{"type": "tool_use", "id": id, "name": name, "input": input}]),
                ));
            }
            "function_call_output" => {
                let id = item
                    .get("call_id")
                    .and_then(Value::as_str)
                    .ok_or("function_call_output.call_id is required")?;
                let output = item.get("output").cloned().unwrap_or(Value::Null);
                let mut block = json!({
                    "type": "tool_result",
                    "tool_use_id": id,
                    "content": output_text(&output)
                });
                if item.get("status").and_then(Value::as_str) == Some("failed") {
                    block["is_error"] = json!(true);
                }
                push_user_block(&mut parsed.messages, block);
            }
            "reasoning" => {
                let summary = item
                    .get("summary")
                    .and_then(Value::as_array)
                    .map(|value| text_content(&Value::Array(value.clone())))
                    .unwrap_or_default();
                if !summary.is_empty() {
                    parsed.messages.push(message(
                        "assistant",
                        json!([{"type": "thinking", "thinking": summary}]),
                    ));
                }
            }
            "item_reference" => {
                let id = item
                    .get("id")
                    .and_then(Value::as_str)
                    .ok_or("item_reference.id is required")?;
                parsed.output_items.push(StoredOutputItem {
                    id: id.to_string(),
                    item: item.clone(),
                });
            }
            "additional_tools" => {
                if let Some(tools) = item.get("tools").and_then(Value::as_array) {
                    parsed.tools.extend(tools.iter().cloned());
                } else if let Some(tools) = item.get("items").and_then(Value::as_array) {
                    parsed.tools.extend(tools.iter().cloned());
                }
            }
            "input_file" => push_user_block(
                &mut parsed.messages,
                json!({"type": "text", "text": item_to_context_text(kind, item)}),
            ),
            "computer_call"
            | "local_shell_call"
            | "custom_tool_call"
            | "web_search_call"
            | "file_search_call"
            | "code_interpreter_call"
            | "image_generation_call"
            | "mcp_call"
            | "mcp_list_tools"
            | "mcp_approval_request" => parsed.messages.push(message(
                "assistant",
                json!([{"type": "text", "text": item_to_context_text(kind, item)}]),
            )),
            "computer_call_output"
            | "local_shell_call_output"
            | "custom_tool_call_output"
            | "mcp_approval_response" => push_user_block(
                &mut parsed.messages,
                json!({"type": "text", "text": item_to_context_text(kind, item)}),
            ),
            _ => return Err(format!("unsupported input item type: {kind}")),
        }
    }
    if parsed.messages.is_empty() && parsed.system.is_empty() && parsed.tools.is_empty() {
        return Err("input must not be empty".into());
    }
    Ok(parsed)
}

fn function_tools(tools: Option<&[Value]>) -> Result<Option<Vec<Tool>>, String> {
    let Some(tools) = tools else {
        return Ok(None);
    };
    let mut result = Vec::new();
    for tool in tools {
        let tool_type = tool.get("type").and_then(Value::as_str).unwrap_or("function");
        let function = tool.get("function").filter(|value| value.is_object());
        let raw_name = tool
            .get("name")
            .or_else(|| function.and_then(|value| value.get("name")))
            .and_then(Value::as_str)
            .unwrap_or_else(|| fallback_tool_name(tool_type));
        let name = sanitize_tool_name(raw_name);
        if name.is_empty() {
            return Err("function.name is required".into());
        }
        let mut description = tool
            .get("description")
            .or_else(|| function.and_then(|value| value.get("description")))
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        if description.is_empty() && tool_type != "function" {
            description = format!("{tool_type} tool");
        }
        let parameters = if raw_name == "exec" {
            json!({
                "type": "object",
                "properties": {
                    "input": {"type": "string"}
                },
                "required": ["input"],
                "additionalProperties": false
            })
        } else {
            tool
                .get("parameters")
                .or_else(|| function.and_then(|value| value.get("parameters")))
                .or_else(|| tool.get("input_schema"))
                .or_else(|| tool.get("schema"))
                .cloned()
                .map_or_else(|| default_tool_parameters(tool_type), |value| {
                    if value.is_object() {
                        value
                    } else if tool_type == "function" {
                        json!(null)
                    } else {
                        default_tool_parameters(tool_type)
                    }
                })
        };
        if parameters.is_null() {
            return Err("function.parameters must be an object".into());
        }
        let input_schema: Map<String, Value> = serde_json::from_value(parameters)
            .map_err(|_| "function.parameters must be an object")?;
        result.push(Tool {
            tool_type: None,
            name,
            description,
            input_schema: input_schema.into_iter().collect(),
            max_uses: None,
            defer_loading: None,
        });
    }
    Ok(Some(result))
}

fn fallback_tool_name(tool_type: &str) -> &str {
    match tool_type {
        "web_search_preview" | "web_search_preview_2025_03_11" | "web_search" => "web_search",
        "file_search" => "file_search",
        "computer_use_preview" | "computer" => "computer",
        "local_shell" => "local_shell",
        "custom" => "custom_tool",
        "mcp" => "mcp_tool",
        _ => tool_type,
    }
}

fn sanitize_tool_name(name: &str) -> String {
    name.chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
                ch
            } else {
                '_'
            }
        })
        .collect::<String>()
        .trim_matches('_')
        .to_string()
}

fn default_tool_parameters(tool_type: &str) -> Value {
    match tool_type {
        "local_shell" => json!({
            "type": "object",
            "properties": {
                "command": {"type": "string"}
            },
            "required": ["command"]
        }),
        "custom" => json!({
            "type": "object",
            "properties": {
                "input": {"type": "string"}
            },
            "required": ["input"]
        }),
        _ => json!({"type": "object", "properties": {}}),
    }
}

fn output_config_from_format(format: Option<&Value>) -> Result<Option<OutputConfig>, String> {
    let Some(format) = format else {
        return Ok(None);
    };
    let kind = format
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or("text");
    match kind {
        "text" => Ok(None),
        "json_object" => Ok(Some(OutputConfig {
            effort: "high".to_string(),
            format: Some(OutputFormat {
                format_type: "json_schema".to_string(),
                schema: json!({"type": "object"}),
            }),
        })),
        "json_schema" => {
            let schema = format
                .get("json_schema")
                .and_then(|value| value.get("schema"))
                .or_else(|| format.get("schema"))
                .cloned()
                .unwrap_or_else(|| json!({"type": "object"}));
            Ok(Some(OutputConfig {
                effort: "high".to_string(),
                format: Some(OutputFormat {
                    format_type: "json_schema".to_string(),
                    schema,
                }),
            }))
        }
        other => Err(format!("unsupported response format type: {other}")),
    }
}

fn responses_output_config(req: &ResponsesRequest) -> Result<Option<OutputConfig>, String> {
    let format = req
        .text
        .as_ref()
        .and_then(|text| text.get("format"))
        .or(req.response_format.as_ref());
    output_config_from_format(format)
}

fn referenced_messages(
    ids: &[StoredOutputItem],
    previous: Option<&StoredConversation>,
) -> Result<Vec<Message>, String> {
    let Some(previous) = previous else {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        return Err("item_reference requires previous_response_id".into());
    };

    let mut messages = Vec::new();
    for reference in ids {
        let Some(stored) = previous.output_items.iter().find(|item| {
            item.id == reference.id
                || item
                    .item
                    .get("call_id")
                    .and_then(Value::as_str)
                    .is_some_and(|call_id| call_id == reference.id)
        }) else {
            return Err(format!("unknown item_reference id: {}", reference.id));
        };
        if let Some(kind) = stored.item.get("type").and_then(Value::as_str) {
            match kind {
                "message" => {
                    let text = stored
                        .item
                        .get("content")
                        .map(text_content)
                        .unwrap_or_default();
                    messages.push(message("assistant", json!(text)));
                }
                "function_call" => {
                    let id = stored
                        .item
                        .get("call_id")
                        .and_then(Value::as_str)
                        .unwrap_or(&reference.id);
                    let name = stored
                        .item
                        .get("name")
                        .and_then(Value::as_str)
                        .unwrap_or_default();
                    let arguments = stored
                        .item
                        .get("arguments")
                        .and_then(Value::as_str)
                        .unwrap_or("{}");
                    let input = serde_json::from_str(arguments).unwrap_or_else(|_| json!({}));
                    messages.push(message(
                        "assistant",
                        json!([{
                            "type": "tool_use",
                            "id": id,
                            "name": name,
                            "input": input
                        }]),
                    ));
                }
                "reasoning" => {}
                _ => return Err(format!("unsupported item_reference type: {kind}")),
            }
        }
    }
    Ok(messages)
}

fn to_messages(
    req: &ResponsesRequest,
    previous: Option<StoredConversation>,
) -> Result<MessagesRequest, String> {
    let parsed = input_messages(&req.input)?;
    let mut tool_values = req.tools.clone().unwrap_or_default();
    tool_values.extend(parsed.tools.iter().cloned());
    let tools = if tool_values.is_empty() {
        previous
            .as_ref()
            .and_then(|stored| stored.snapshot.tools.clone())
    } else {
        function_tools(Some(&tool_values))?
    };

    let mut system = Vec::new();
    if let Some(text) = &req.instructions
        && !text.is_empty()
    {
        system.push(SystemMessage { text: text.clone() });
    }
    system.extend(parsed.system);
    if system.is_empty() {
        system = previous
            .as_ref()
            .and_then(|stored| stored.snapshot.system.clone())
            .unwrap_or_default();
    }

    let mut messages = previous
        .as_ref()
        .map(|stored| stored.snapshot.messages.clone())
        .unwrap_or_default();
    messages.extend(referenced_messages(&parsed.output_items, previous.as_ref())?);
    messages.extend(parsed.messages);
    if messages.is_empty() {
        return Err("input must contain a message or usable item".into());
    }

    let max_tokens = req.max_output_tokens.unwrap_or(4096).max(1);
    let thinking = if req.reasoning.is_some() || req.model.to_lowercase().contains("thinking") {
        Some(Thinking {
            thinking_type: "adaptive".into(),
            budget_tokens: max_tokens.min(24576),
        })
    } else {
        None
    };

    Ok(MessagesRequest {
        model: req.model.clone(),
        max_tokens,
        messages,
        stream: req.stream,
        system: (!system.is_empty()).then_some(system),
        tools,
        tool_choice: None,
        thinking,
        output_config: responses_output_config(req)?,
        metadata: None,
    })
}

struct ToolState {
    call_id: String,
    name: String,
    arguments: String,
    done: bool,
}

struct ResponseAccumulator {
    id: String,
    model: String,
    exec_command_requested: bool,
    text: String,
    reasoning: String,
    thinking: bool,
    thinking_seen: bool,
    pending: String,
    tools: Vec<ToolState>,
    input_tokens: i32,
    output_tokens: i32,
    cached_tokens: i32,
    cache_creation_tokens: i32,
    metering_usage: Option<f64>,
    failed: Option<String>,
}

impl ResponseAccumulator {
    fn new(
        id: String,
        model: String,
        input_tokens: i32,
        exec_command_requested: bool,
    ) -> Self {
        Self {
            id,
            model,
            exec_command_requested,
            text: String::new(),
            reasoning: String::new(),
            thinking: false,
            thinking_seen: false,
            pending: String::new(),
            tools: Vec::new(),
            input_tokens,
            output_tokens: 0,
            cached_tokens: 0,
            cache_creation_tokens: 0,
            metering_usage: None,
            failed: None,
        }
    }

    fn push_assistant(&mut self, content: &str) {
        self.pending.push_str(content);
        self.drain_text(false);
    }

    fn push_tool(&mut self, name: &str, call_id: &str, input: &str, stop: bool) {
        let index = if let Some(index) = self.tools.iter().position(|tool| tool.call_id == call_id)
        {
            index
        } else {
            let index = self.tools.len();
            self.tools.push(ToolState {
                call_id: call_id.to_string(),
                name: name.to_string(),
                arguments: String::new(),
                done: false,
            });
            index
        };
        let previous_name = self.tools[index].name.clone();
        self.tools[index].arguments.push_str(input);
        self.tools[index].name = normalize_tool_name(
            &self.tools[index].name,
            self.exec_command_requested,
            &self.tools[index].arguments,
        );
        if previous_name != self.tools[index].name || stop {
            tracing::info!(
                target: "kiro2cc_proxy::responses_diag",
                call_id = %call_id,
                source_name = %name,
                normalized_name = %self.tools[index].name,
                fragment_len = input.len(),
                arguments_len = self.tools[index].arguments.len(),
                argument_prefix = %argument_prefix(&self.tools[index].arguments),
                stop,
                "Responses ToolUse normalized"
            );
        } else {
            tracing::debug!(
                target: "kiro2cc_proxy::responses_diag",
                call_id = %call_id,
                source_name = %name,
                fragment_len = input.len(),
                arguments_len = self.tools[index].arguments.len(),
                argument_prefix = %argument_prefix(&self.tools[index].arguments),
                stop,
                "Responses ToolUse fragment"
            );
        }
        if stop {
            self.tools[index].done = true;
        }
    }

    fn char_boundary_at_or_before(value: &str, index: usize) -> usize {
        let mut boundary = index.min(value.len());
        while boundary > 0 && !value.is_char_boundary(boundary) {
            boundary -= 1;
        }
        boundary
    }

    fn drain_text(&mut self, final_chunk: bool) {
        loop {
            if !self.thinking_seen && !self.thinking {
                if let Some(pos) = self.pending.find("<thinking>") {
                    self.text.push_str(&self.pending[..pos]);
                    self.pending.drain(..pos + "<thinking>".len());
                    self.thinking = true;
                    continue;
                }
                let keep = "<thinking>".len().saturating_sub(1);
                let target = if final_chunk {
                    self.pending.len()
                } else {
                    self.pending.len().saturating_sub(keep)
                };
                let emit = Self::char_boundary_at_or_before(&self.pending, target);
                if emit > 0 {
                    self.text.push_str(&self.pending[..emit]);
                    self.pending.drain(..emit);
                }
                break;
            }
            if self.thinking {
                if let Some(pos) = self.pending.find("</thinking>") {
                    self.reasoning.push_str(&self.pending[..pos]);
                    self.pending.drain(..pos + "</thinking>".len());
                    self.thinking = false;
                    self.thinking_seen = true;
                    continue;
                }
                let keep = "</thinking>".len().saturating_sub(1);
                let target = if final_chunk {
                    self.pending.len()
                } else {
                    self.pending.len().saturating_sub(keep)
                };
                let emit = Self::char_boundary_at_or_before(&self.pending, target);
                if emit > 0 {
                    self.reasoning.push_str(&self.pending[..emit]);
                    self.pending.drain(..emit);
                }
                break;
            }
            self.text.push_str(&self.pending);
            self.pending.clear();
            break;
        }
    }

    fn finish(&mut self) {
        self.drain_text(true);
        self.output_tokens = token::estimate_output_tokens(&output_items(self));
    }
}

#[derive(Default)]
struct StreamMarkers {
    emitted_tool_args: HashMap<String, usize>,
    completed_tools: HashMap<String, bool>,
    tool_output_indices: HashMap<String, usize>,
    next_output_index: usize,
    emitted_reasoning_len: usize,
    reasoning_output_index: Option<usize>,
    reasoning_item_done: bool,
    emitted_text_len: usize,
    text_output_index: Option<usize>,
    text_item_done: bool,
    sequence_number: u64,
}

fn now_seconds() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs() as i64)
        .unwrap_or_default()
}

fn function_call_item(tool: &ToolState, status: &str) -> Value {
    json!({
        "type": "function_call",
        "id": format!("fc_{}", tool.call_id),
        "call_id": tool.call_id,
        "name": tool.name,
        "arguments": tool.arguments,
        "status": status
    })
}

fn custom_tool_input_complete(arguments: &str) -> Option<String> {
    serde_json::from_str::<Value>(arguments).ok().and_then(|value| {
        value
            .get("input")
            .and_then(Value::as_str)
            .map(str::to_string)
            .or_else(|| value.as_str().map(str::to_string))
    })
}

fn custom_tool_input(arguments: &str) -> String {
    custom_tool_input_complete(arguments).unwrap_or_else(|| arguments.to_string())
}

fn is_custom_tool(tool: &ToolState) -> bool {
    tool.name == "exec" && tool.arguments.trim_start().starts_with("{\"input\"")
}

fn custom_tool_call_item(tool: &ToolState, status: &str, input: &str) -> Value {
    json!({
        "type": "custom_tool_call",
        "id": format!("ctc_{}", tool.call_id),
        "call_id": tool.call_id,
        "name": tool.name,
        "input": input,
        "status": status
    })
}

fn output_items(acc: &ResponseAccumulator) -> Vec<Value> {
    let mut output = Vec::new();
    if !acc.reasoning.is_empty() {
        output.push(json!({
            "type": "reasoning",
            "id": format!("rs_{}", acc.id),
            "summary": [{"type": "summary_text", "text": acc.reasoning}],
            "status": "completed"
        }));
    }
    if !acc.text.is_empty() {
        output.push(json!({
            "type": "message",
            "id": format!("msg_{}", acc.id),
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": acc.text, "annotations": []}]
        }));
    }
    for tool in &acc.tools {
        output.push(if is_custom_tool(tool) {
            custom_tool_call_item(tool, "completed", &custom_tool_input(&tool.arguments))
        } else {
            function_call_item(tool, "completed")
        });
    }
    output
}

fn assistant_history_message(acc: &ResponseAccumulator) -> Option<Message> {
    let mut content = Vec::new();
    if !acc.text.is_empty() {
        content.push(json!({"type": "text", "text": acc.text}));
    }
    for tool in &acc.tools {
        let input = serde_json::from_str(&tool.arguments).unwrap_or_else(|_| json!({}));
        content.push(json!({
            "type": "tool_use",
            "id": tool.call_id,
            "name": tool.name,
            "input": input
        }));
    }
    (!content.is_empty()).then(|| message("assistant", Value::Array(content)))
}

fn stored_output_items(acc: &ResponseAccumulator) -> Vec<StoredOutputItem> {
    output_items(acc)
        .into_iter()
        .filter_map(|item| {
            let id = item.get("id")?.as_str()?.to_string();
            Some(StoredOutputItem { id, item })
        })
        .collect()
}

fn response_body(acc: &ResponseAccumulator) -> Value {
    let output = output_items(acc);
    let output_text = acc.text.clone();
    let status = if acc.failed.is_some() {
        "failed"
    } else {
        "completed"
    };
    let reasoning_tokens = token::estimate_output_tokens(&[json!(acc.reasoning)]);
    let mut body = json!({
        "id": acc.id,
        "object": "response",
        "created_at": now_seconds(),
        "status": status,
        "model": acc.model,
        "output": output,
        "output_text": output_text,
        "usage": {
            "input_tokens": acc.input_tokens.max(0),
            "input_tokens_details": {"cached_tokens": acc.cached_tokens.max(0)},
            "output_tokens": acc.output_tokens.max(0),
            "output_tokens_details": {"reasoning_tokens": reasoning_tokens.max(0)},
            "total_tokens": (acc.input_tokens + acc.output_tokens).max(0)
        }
    });
    if let Some(error) = &acc.failed {
        body["error"] = json!({"code": "upstream_error", "message": error});
    }
    body
}

fn sse(event: &str, data: Value) -> Bytes {
    Bytes::from(format!("event: {event}\ndata: {data}\n\n"))
}

fn numbered_sse(markers: &mut StreamMarkers, event: &str, mut data: Value) -> Bytes {
    if let Value::Object(object) = &mut data {
        object.insert("sequence_number".to_string(), json!(markers.sequence_number));
    }
    markers.sequence_number += 1;
    sse(event, data)
}

fn process_event(acc: &mut ResponseAccumulator, event: Event) {
    match event {
        Event::AssistantResponse(response) => acc.push_assistant(&response.content),
        Event::ToolUse(tool) => acc.push_tool(&tool.name, &tool.tool_use_id, &tool.input, tool.stop),
        Event::Metering(metering) => {
            acc.metering_usage = Some(metering.usage);
            acc.cached_tokens = metering.cache_read_input_tokens.unwrap_or(0);
            acc.cache_creation_tokens = metering.cache_creation_input_tokens.unwrap_or(0);
        }
        Event::ContextUsage(usage) if usage.context_usage_percentage >= 100.0 => {
            acc.failed = Some("model context window exceeded".to_string());
        }
        Event::Error {
            error_code,
            error_message,
        } => {
            acc.failed = Some(format!("{error_code}: {error_message}"));
        }
        Event::Exception {
            exception_type,
            message,
        } => {
            acc.failed = Some(format!("{exception_type}: {message}"));
        }
        _ => {}
    }
}

fn parse_events(bytes: &[u8], acc: &mut ResponseAccumulator) {
    let mut decoder = EventStreamDecoder::new();
    if let Err(error) = decoder.feed(bytes) {
        tracing::warn!(%error, "Responses 上游帧缓冲失败");
    }
    for result in decoder.decode_iter() {
        match result {
            Ok(frame) => match Event::from_frame(frame) {
                Ok(event) => process_event(acc, event),
                Err(error) => tracing::warn!(%error, "Responses 上游事件解析失败"),
            },
            Err(error) => tracing::warn!(%error, "Responses 上游帧解析失败"),
        }
    }
}

fn parse_request(body: &[u8]) -> Result<ResponsesRequest, Response> {
    serde_json::from_slice(body).map_err(|error| {
        error_response(
            StatusCode::BAD_REQUEST,
            format!("Request body could not be parsed: {error}"),
        )
    })
}

fn previous_conversation(id: Option<&str>) -> Result<Option<StoredConversation>, Response> {
    let Some(id) = id else {
        return Ok(None);
    };
    let mut store = conversations().lock().expect("responses store poisoned");
    let now = Instant::now();
    store.retain(|_, value| value.expires_at > now);
    store
        .get(id)
        .cloned()
        .ok_or_else(|| {
            error_response(
                StatusCode::BAD_REQUEST,
                "previous_response_id was not found or has expired",
            )
        })
        .map(Some)
}

fn save_conversation(id: String, snapshot: &RequestSnapshot, accumulator: &ResponseAccumulator) {
    let mut stored_snapshot = snapshot.clone();
    if let Some(assistant) = assistant_history_message(accumulator) {
        stored_snapshot.messages.push(assistant);
    }

    let mut store = conversations().lock().expect("responses store poisoned");
    if store.len() >= 1024
        && let Some(oldest) = store
            .iter()
            .min_by_key(|(_, value)| value.expires_at)
            .map(|(key, _)| key.clone())
    {
        store.remove(&oldest);
    }
    store.insert(
        id,
        StoredConversation {
            snapshot: stored_snapshot,
            output_items: stored_output_items(accumulator),
            expires_at: Instant::now() + Duration::from_secs(3600),
        },
    );
}

fn build_kiro_request(state: &AppState, messages: &MessagesRequest) -> Result<String, Response> {
    let conversion = convert_request(messages)
        .map_err(|error| error_response(StatusCode::BAD_REQUEST, error.to_string()))?;
    serde_json::to_string(&KiroRequest {
        conversation_state: conversion.conversation_state,
        profile_arn: state.profile_arn.clone(),
        additional_model_request_fields: conversion.additional_model_request_fields,
    })
    .map_err(|error| error_response(StatusCode::INTERNAL_SERVER_ERROR, error.to_string()))
}

fn input_tokens(messages: &MessagesRequest) -> i32 {
    token::count_all_tokens(
        messages.model.clone(),
        messages.system.clone(),
        messages.messages.clone(),
        messages.tools.clone(),
    ) as i32
}

fn argument_prefix(arguments: &str) -> String {
    arguments.chars().take(32).collect()
}

fn tool_summary(messages: &MessagesRequest) -> Vec<String> {
    messages
        .tools
        .as_deref()
        .unwrap_or_default()
        .iter()
        .map(|tool| {
            let mut keys = tool.input_schema.keys().cloned().collect::<Vec<_>>();
            keys.sort();
            format!("{}[{}]", tool.name, keys.join(","))
        })
        .collect()
}

fn has_exec_command_tool(messages: &MessagesRequest) -> bool {
    messages
        .tools
        .as_deref()
        .is_some_and(|tools| tools.iter().any(|tool| tool.name == "exec_command"))
}

fn normalize_tool_name(
    name: &str,
    exec_command_requested: bool,
    arguments: &str,
) -> String {
    if name == "exec"
        && (exec_command_requested
            || arguments.trim_start().starts_with("{\"cmd\"")
            || serde_json::from_str::<Value>(arguments)
                .ok()
                .and_then(|value| value.get("cmd").cloned())
                .is_some())
    {
        "exec_command".to_string()
    } else {
        name.to_string()
    }
}

fn record_usage(
    usage_tracker: &Option<Arc<UsageTracker>>,
    api_key_id: Option<u32>,
    credential_id: Option<u64>,
    accumulator: &ResponseAccumulator,
) {
    if let (Some(tracker), Some(key_id)) = (usage_tracker, api_key_id) {
        tracker.record(
            key_id,
            credential_id,
            accumulator.model.clone(),
            accumulator.input_tokens,
            accumulator.output_tokens,
            None,
            accumulator.metering_usage,
            Some(accumulator.cached_tokens),
            Some(accumulator.cache_creation_tokens),
        );
    }
}

async fn execute_non_stream(
    provider: Arc<crate::kiro::provider::KiroProvider>,
    request_body: &str,
    bound_ids: &[u64],
    response_id: String,
    model: String,
    input_tokens: i32,
    exec_command_requested: bool,
) -> Result<(ResponseAccumulator, u64), Response> {
    let (response, credential_id) = provider
        .call_api(request_body, bound_ids)
        .await
        .map_err(|error| error_response(StatusCode::BAD_GATEWAY, error.to_string()))?;
    let bytes = response
        .bytes()
        .await
        .map_err(|error| error_response(StatusCode::BAD_GATEWAY, error.to_string()))?;
    let mut accumulator = ResponseAccumulator::new(
        response_id,
        model,
        input_tokens,
        exec_command_requested,
    );
    parse_events(&bytes, &mut accumulator);
    accumulator.finish();
    Ok((accumulator, credential_id))
}

fn chat_message_content(content: Option<&Value>, role: &str) -> Result<Value, String> {
    match content {
        Some(value) => message_content(value, role),
        None => Ok(json!("")),
    }
}

fn chat_to_messages(req: &ChatCompletionRequest) -> Result<MessagesRequest, String> {
    let mut system = Vec::new();
    let mut messages = Vec::new();

    for item in &req.messages {
        match item.role.as_str() {
            "system" | "developer" => {
                let text = item
                    .content
                    .as_ref()
                    .map(text_content)
                    .unwrap_or_default();
                if !text.is_empty() {
                    system.push(SystemMessage { text });
                }
            }
            "user" => messages.push(message(
                "user",
                chat_message_content(item.content.as_ref(), "user")?,
            )),
            "assistant" => {
                let mut blocks = Vec::new();
                if let Some(content) = &item.content {
                    match chat_message_content(Some(content), "assistant")? {
                        Value::String(text) if !text.is_empty() => {
                            blocks.push(json!({"type": "text", "text": text}));
                        }
                        Value::Array(items) => blocks.extend(items),
                        _ => {}
                    }
                }
                if let Some(tool_calls) = &item.tool_calls {
                    for call in tool_calls {
                        if call.function.name.is_empty() {
                            return Err("assistant.tool_calls[].function.name is required".into());
                        }
                        let input = serde_json::from_str(&call.function.arguments)
                            .unwrap_or_else(|_| json!({}));
                        blocks.push(json!({
                            "type": "tool_use",
                            "id": call.id,
                            "name": call.function.name,
                            "input": input
                        }));
                    }
                }
                if let Some(call) = &item.function_call {
                    if call.name.is_empty() {
                        return Err("assistant.function_call.name is required".into());
                    }
                    let id = format!("call_{}", Uuid::new_v4().simple());
                    let input = serde_json::from_str(&call.arguments).unwrap_or_else(|_| json!({}));
                    blocks.push(json!({
                        "type": "tool_use",
                        "id": id,
                        "name": call.name,
                        "input": input
                    }));
                }
                messages.push(message("assistant", Value::Array(blocks)));
            }
            "tool" => {
                let id = item
                    .tool_call_id
                    .as_deref()
                    .ok_or("tool message requires tool_call_id")?;
                let content = item.content.clone().unwrap_or(Value::Null);
                push_user_block(
                    &mut messages,
                    json!({
                        "type": "tool_result",
                        "tool_use_id": id,
                        "content": output_text(&content)
                    }),
                );
            }
            "function" => {
                let content = item.content.clone().unwrap_or(Value::Null);
                push_user_block(
                    &mut messages,
                    json!({
                        "type": "text",
                        "text": format!("Function result:\n{}", output_text(&content))
                    }),
                );
            }
            role => return Err(format!("unsupported chat message role: {role}")),
        }
    }

    if messages.is_empty() {
        return Err("messages must contain at least one user or assistant message".into());
    }

    let mut tool_values = req.tools.clone().unwrap_or_default();
    if let Some(functions) = &req.functions {
        tool_values.extend(functions.iter().map(|function| {
            json!({
                "type": "function",
                "function": function
            })
        }));
    }

    let max_tokens = req.max_tokens.unwrap_or(4096).max(1);
    let thinking = if req.reasoning.is_some() || req.model.to_lowercase().contains("thinking") {
        Some(Thinking {
            thinking_type: "adaptive".into(),
            budget_tokens: max_tokens.min(24576),
        })
    } else {
        None
    };

    Ok(MessagesRequest {
        model: req.model.clone(),
        max_tokens,
        messages,
        stream: req.stream,
        system: (!system.is_empty()).then_some(system),
        tools: if tool_values.is_empty() {
            None
        } else {
            function_tools(Some(&tool_values))?
        },
        tool_choice: None,
        thinking,
        output_config: output_config_from_format(req.response_format.as_ref())?,
        metadata: None,
    })
}

fn completion_to_messages(req: &CompletionRequest) -> Result<MessagesRequest, String> {
    let prompt = match &req.prompt {
        Value::String(text) => text.clone(),
        Value::Array(items) => items
            .iter()
            .map(|item| match item {
                Value::String(text) => text.clone(),
                other => other.to_string(),
            })
            .collect::<Vec<_>>()
            .join("\n"),
        other => other.to_string(),
    };
    if prompt.trim().is_empty() {
        return Err("prompt must not be empty".into());
    }

    Ok(MessagesRequest {
        model: req.model.clone(),
        max_tokens: req.max_tokens.unwrap_or(4096).max(1),
        messages: vec![message("user", json!(prompt))],
        stream: req.stream,
        system: None,
        tools: None,
        tool_choice: None,
        thinking: None,
        output_config: None,
        metadata: None,
    })
}

fn chat_tool_calls(acc: &ResponseAccumulator) -> Vec<Value> {
    acc.tools
        .iter()
        .map(|tool| {
            json!({
                "id": tool.call_id,
                "type": "function",
                "function": {
                    "name": tool.name,
                    "arguments": tool.arguments
                }
            })
        })
        .collect()
}

fn chat_completion_body(acc: &ResponseAccumulator) -> Value {
    let mut message = json!({
        "role": "assistant",
        "content": if acc.text.is_empty() { Value::Null } else { json!(acc.text) }
    });
    let tool_calls = chat_tool_calls(acc);
    if !tool_calls.is_empty() {
        message["tool_calls"] = Value::Array(tool_calls);
    }
    let finish_reason = if !acc.tools.is_empty() {
        "tool_calls"
    } else {
        "stop"
    };
    json!({
        "id": acc.id,
        "object": "chat.completion",
        "created": now_seconds(),
        "model": acc.model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason
        }],
        "usage": {
            "prompt_tokens": acc.input_tokens.max(0),
            "completion_tokens": acc.output_tokens.max(0),
            "total_tokens": (acc.input_tokens + acc.output_tokens).max(0)
        }
    })
}

fn completion_body(acc: &ResponseAccumulator) -> Value {
    json!({
        "id": acc.id,
        "object": "text_completion",
        "created": now_seconds(),
        "model": acc.model,
        "choices": [{
            "text": acc.text,
            "index": 0,
            "logprobs": Value::Null,
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": acc.input_tokens.max(0),
            "completion_tokens": acc.output_tokens.max(0),
            "total_tokens": (acc.input_tokens + acc.output_tokens).max(0)
        }
    })
}

fn openai_data_sse(data: Value) -> Bytes {
    Bytes::from(format!("data: {data}\n\n"))
}

fn openai_done_sse() -> Bytes {
    Bytes::from("data: [DONE]\n\n")
}

fn chat_chunk(id: &str, model: &str, delta: Value, finish_reason: Option<&str>) -> Value {
    json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": now_seconds(),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish_reason
        }]
    })
}

fn completion_chunk(id: &str, model: &str, text: &str, finish_reason: Option<&str>) -> Value {
    json!({
        "id": id,
        "object": "text_completion",
        "created": now_seconds(),
        "model": model,
        "choices": [{
            "text": text,
            "index": 0,
            "logprobs": Value::Null,
            "finish_reason": finish_reason
        }]
    })
}

fn collect_chat_tool_stream_events(
    acc: &ResponseAccumulator,
    markers: &mut StreamMarkers,
) -> Vec<Result<Bytes, Infallible>> {
    let mut events: Vec<Result<Bytes, Infallible>> = Vec::new();
    for (index, tool) in acc.tools.iter().enumerate() {
        if !markers.emitted_tool_args.contains_key(&tool.call_id) {
            markers.emitted_tool_args.insert(tool.call_id.clone(), 0);
            events.push(Ok(openai_data_sse(chat_chunk(
                &acc.id,
                &acc.model,
                json!({
                    "tool_calls": [{
                        "index": index,
                        "id": tool.call_id,
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "arguments": ""
                        }
                    }]
                }),
                None,
            ))));
        }
        let emitted_len = markers
            .emitted_tool_args
            .get(&tool.call_id)
            .copied()
            .unwrap_or(0);
        if tool.arguments.len() > emitted_len {
            let delta = &tool.arguments[emitted_len..];
            markers
                .emitted_tool_args
                .insert(tool.call_id.clone(), tool.arguments.len());
            events.push(Ok(openai_data_sse(chat_chunk(
                &acc.id,
                &acc.model,
                json!({
                    "tool_calls": [{
                        "index": index,
                        "function": {
                            "arguments": delta
                        }
                    }]
                }),
                None,
            ))));
        }
    }
    events
}

fn collect_reasoning_stream_events(
    acc: &ResponseAccumulator,
    markers: &mut StreamMarkers,
    final_chunk: bool,
) -> Vec<Result<Bytes, Infallible>> {
    if acc.reasoning.is_empty() { return Vec::new(); }
    let output_index = match markers.reasoning_output_index {
        Some(index) => index,
        None => {
            let index = markers.next_output_index;
            markers.next_output_index += 1;
            markers.reasoning_output_index = Some(index);
            index
        }
    };
    let item_id = format!("rs_{}", acc.id);
    let mut events = Vec::new();
    if markers.emitted_reasoning_len == 0 {
        events.push(Ok(numbered_sse(markers,"response.output_item.added", json!({"type":"response.output_item.added","response_id":acc.id,"output_index":output_index,"item":{"type":"reasoning","id":item_id,"status":"in_progress","summary":[]}}))));
        events.push(Ok(numbered_sse(markers,"response.reasoning_summary_part.added", json!({"type":"response.reasoning_summary_part.added","response_id":acc.id,"item_id":item_id,"output_index":output_index,"summary_index":0,"part":{"type":"summary_text","text":""}}))));
    }
    if acc.reasoning.len() > markers.emitted_reasoning_len {
        let delta = &acc.reasoning[markers.emitted_reasoning_len..];
        markers.emitted_reasoning_len = acc.reasoning.len();
        events.push(Ok(numbered_sse(markers,"response.reasoning_summary_text.delta", json!({"type":"response.reasoning_summary_text.delta","response_id":acc.id,"item_id":item_id,"output_index":output_index,"summary_index":0,"delta":delta}))));
    }
    if final_chunk && !markers.reasoning_item_done {
        markers.reasoning_item_done = true;
        events.push(Ok(numbered_sse(markers,"response.reasoning_summary_text.done", json!({"type":"response.reasoning_summary_text.done","response_id":acc.id,"item_id":item_id,"output_index":output_index,"summary_index":0,"text":acc.reasoning}))));
        events.push(Ok(numbered_sse(markers,"response.reasoning_summary_part.done", json!({"type":"response.reasoning_summary_part.done","response_id":acc.id,"item_id":item_id,"output_index":output_index,"summary_index":0,"part":{"type":"summary_text","text":acc.reasoning}}))));
        events.push(Ok(numbered_sse(markers,"response.output_item.done", json!({"type":"response.output_item.done","response_id":acc.id,"output_index":output_index,"item":{"type":"reasoning","id":item_id,"status":"completed","summary":[{"type":"summary_text","text":acc.reasoning}]}}))));
    }
    events
}

fn tool_output_index(markers: &mut StreamMarkers, call_id: &str) -> usize {
    if let Some(index) = markers.tool_output_indices.get(call_id) {
        return *index;
    }
    let index = markers.next_output_index;
    markers.next_output_index += 1;
    markers.tool_output_indices.insert(call_id.to_string(), index);
    index
}

fn collect_text_stream_events(
    acc: &ResponseAccumulator,
    markers: &mut StreamMarkers,
    final_chunk: bool,
) -> Vec<Result<Bytes, Infallible>> {
    if acc.text.is_empty() { return Vec::new(); }
    let output_index = match markers.text_output_index {
        Some(index) => index,
        None => {
            let index = markers.next_output_index;
            markers.next_output_index += 1;
            markers.text_output_index = Some(index);
            index
        }
    };
    let item_id = format!("msg_{}", acc.id);
    let mut events = Vec::new();
    if markers.emitted_text_len == 0 {
        events.push(Ok(numbered_sse(markers,"response.output_item.added", json!({"type":"response.output_item.added","response_id":acc.id,"output_index":output_index,"item":{"type":"message","id":item_id,"status":"in_progress","role":"assistant","content":[]}}))));
        events.push(Ok(numbered_sse(markers,"response.content_part.added", json!({"type":"response.content_part.added","response_id":acc.id,"item_id":item_id,"output_index":output_index,"content_index":0,"part":{"type":"output_text","text":"","annotations":[]}}))));
    }
    if acc.text.len() > markers.emitted_text_len {
        let delta = &acc.text[markers.emitted_text_len..];
        markers.emitted_text_len = acc.text.len();
        events.push(Ok(numbered_sse(markers,"response.output_text.delta", json!({"type":"response.output_text.delta","response_id":acc.id,"item_id":item_id,"output_index":output_index,"content_index":0,"delta":delta}))));
    }
    if final_chunk && !markers.text_item_done {
        markers.text_item_done = true;
        events.push(Ok(numbered_sse(markers,"response.output_text.done", json!({"type":"response.output_text.done","response_id":acc.id,"item_id":item_id,"output_index":output_index,"content_index":0,"text":acc.text}))));
        events.push(Ok(numbered_sse(markers,"response.content_part.done", json!({"type":"response.content_part.done","response_id":acc.id,"item_id":item_id,"output_index":output_index,"content_index":0,"part":{"type":"output_text","text":acc.text,"annotations":[]}}))));
        events.push(Ok(numbered_sse(markers,"response.output_item.done", json!({"type":"response.output_item.done","response_id":acc.id,"output_index":output_index,"item":{"type":"message","id":item_id,"status":"completed","role":"assistant","content":[{"type":"output_text","text":acc.text,"annotations":[]}]}}))));
    }
    events
}

fn tool_stream_ready(tool: &ToolState) -> bool {
    if tool.name != "exec" || tool.done {
        return true;
    }
    let arguments = tool.arguments.trim_start();
    if arguments.starts_with("{\"input\"") {
        custom_tool_input_complete(&tool.arguments).is_some()
    } else {
        arguments.starts_with("{\"cmd\"")
    }
}

fn response_tool_summary(acc: &ResponseAccumulator) -> Vec<String> {
    acc.tools
        .iter()
        .map(|tool| {
            format!(
                "{}:{}[len={},done={}]",
                tool.call_id,
                tool.name,
                tool.arguments.len(),
                tool.done
            )
        })
        .collect()
}

fn collect_tool_stream_events(
    acc: &ResponseAccumulator,
    markers: &mut StreamMarkers,
) -> Vec<Result<Bytes, Infallible>> {
    let mut events: Vec<Result<Bytes, Infallible>> = Vec::new();
    for tool in &acc.tools {
        if !tool_stream_ready(tool) {
            tracing::info!(
                target: "kiro2cc_proxy::responses_diag",
                call_id = %tool.call_id,
                name = %tool.name,
                arguments_len = tool.arguments.len(),
                argument_prefix = %argument_prefix(&tool.arguments),
                "Responses tool SSE delayed"
            );
            continue;
        }
        let output_index = tool_output_index(markers, &tool.call_id);
        let custom = is_custom_tool(tool);
        let item_id = if custom {
            format!("ctc_{}", tool.call_id)
        } else {
            format!("fc_{}", tool.call_id)
        };
        if !markers.emitted_tool_args.contains_key(&tool.call_id) {
            markers.emitted_tool_args.insert(tool.call_id.clone(), 0);
            markers.completed_tools.insert(tool.call_id.clone(), false);
            tracing::info!(
                target: "kiro2cc_proxy::responses_diag",
                response_id = %acc.id,
                call_id = %tool.call_id,
                name = %tool.name,
                output_index,
                arguments_len = tool.arguments.len(),
                argument_prefix = %argument_prefix(&tool.arguments),
                custom,
                "Responses SSE output_item.added"
            );
            let item = if custom {
                custom_tool_call_item(tool, "in_progress", "")
            } else {
                json!({
                    "type": "function_call",
                    "id": item_id,
                    "call_id": tool.call_id,
                    "name": tool.name,
                    "arguments": "",
                    "status": "in_progress"
                })
            };
            events.push(Ok(numbered_sse(markers,
                "response.output_item.added",
                json!({
                    "type": "response.output_item.added",
                    "response_id": acc.id,
                    "output_index": output_index,
                    "item": item
                }),
            )));
        }

        let input = if custom {
            custom_tool_input_complete(&tool.arguments).unwrap_or_default()
        } else {
            tool.arguments.clone()
        };
        let emitted_len = markers
            .emitted_tool_args
            .get(&tool.call_id)
            .copied()
            .unwrap_or(0);
        if input.len() > emitted_len {
            let delta = &input[emitted_len..];
            markers
                .emitted_tool_args
                .insert(tool.call_id.clone(), input.len());
            let event_name = if custom {
                "response.custom_tool_call_input.delta"
            } else {
                "response.function_call_arguments.delta"
            };
            let mut data = json!({
                "type": event_name,
                "response_id": acc.id,
                "item_id": item_id,
                "output_index": output_index,
                "delta": delta
            });
            events.push(Ok(numbered_sse(markers, event_name, data.take())));
        }

        let already_done = markers
            .completed_tools
            .get(&tool.call_id)
            .copied()
            .unwrap_or(false);
        if tool.done && !already_done {
            markers.completed_tools.insert(tool.call_id.clone(), true);
            let done_event_name = if custom {
                "response.custom_tool_call_input.done"
            } else {
                "response.function_call_arguments.done"
            };
            let done_data = if custom {
                json!({
                    "type": done_event_name,
                    "response_id": acc.id,
                    "item_id": item_id,
                    "output_index": output_index,
                    "input": input
                })
            } else {
                json!({
                    "type": done_event_name,
                    "response_id": acc.id,
                    "item_id": item_id,
                    "output_index": output_index,
                    "arguments": input
                })
            };
            events.push(Ok(numbered_sse(markers, done_event_name, done_data)));
            tracing::info!(
                target: "kiro2cc_proxy::responses_diag",
                response_id = %acc.id,
                call_id = %tool.call_id,
                name = %tool.name,
                output_index,
                arguments_len = input.len(),
                argument_prefix = %argument_prefix(&input),
                custom,
                "Responses SSE output_item.done"
            );
            let item = if custom {
                custom_tool_call_item(tool, "completed", &input)
            } else {
                function_call_item(tool, "completed")
            };
            events.push(Ok(numbered_sse(markers,
                "response.output_item.done",
                json!({
                    "type": "response.output_item.done",
                    "response_id": acc.id,
                    "output_index": output_index,
                    "item": item
                }),
            )));
        }
    }
    events
}

/// POST /v1/responses
pub async fn post_responses(
    State(state): State<AppState>,
    identity: Option<Extension<ApiKeyContext>>,
    body: Bytes,
) -> Response {
    let request = match parse_request(&body) {
        Ok(request) => request,
        Err(response) => return response,
    };
    let previous = match previous_conversation(request.previous_response_id.as_deref()) {
        Ok(previous) => previous,
        Err(response) => return response,
    };
    let messages = match to_messages(&request, previous) {
        Ok(messages) => messages,
        Err(error) => return error_response(StatusCode::BAD_REQUEST, error),
    };
    let request_body = match build_kiro_request(&state, &messages) {
        Ok(body) => body,
        Err(response) => return response,
    };
    let Some(provider) = state.kiro_provider.clone() else {
        return error_response(StatusCode::SERVICE_UNAVAILABLE, "Kiro API provider not configured");
    };
    if let Some(rpm) = &state.rpm_tracker {
        rpm.record_request(identity.as_ref().map(|context| context.0.id));
    }
    let bound_ids = identity
        .as_ref()
        .and_then(|context| context.0.bound_credential_ids.clone())
        .unwrap_or_default();
    let api_key_id = identity.as_ref().map(|context| context.0.id);
    let usage_tracker = state.usage_tracker.clone();
    let response_id = format!("resp_{}", Uuid::new_v4().simple());
    let estimated_input_tokens = input_tokens(&messages);
    let exec_command_requested = has_exec_command_tool(&messages);
    tracing::info!(
        target: "kiro2cc_proxy::responses_diag",
        response_id = %response_id,
        model = %request.model,
        stream = request.stream,
        input_tokens = estimated_input_tokens,
        exec_command_requested,
        tools = ?tool_summary(&messages),
        "Responses request tool summary"
    );
    let snapshot = RequestSnapshot::from(&messages);

    if request.stream {
        return stream_response(
            provider,
            request_body,
            bound_ids,
            response_id,
            request.model,
            estimated_input_tokens,
            exec_command_requested,
            snapshot,
            request.store,
            usage_tracker,
            api_key_id,
        )
        .await;
    }

    let (response, credential_id) = match provider.call_api(&request_body, &bound_ids).await {
        Ok(response) => response,
        Err(error) => return error_response(StatusCode::BAD_GATEWAY, error.to_string()),
    };
    let bytes = match response.bytes().await {
        Ok(bytes) => bytes,
        Err(error) => return error_response(StatusCode::BAD_GATEWAY, error.to_string()),
    };
    let mut accumulator = ResponseAccumulator::new(
        response_id.clone(),
        request.model,
        estimated_input_tokens,
        exec_command_requested,
    );
    parse_events(&bytes, &mut accumulator);
    accumulator.finish();
    tracing::info!(
        target: "kiro2cc_proxy::responses_diag",
        response_id = %accumulator.id,
        model = %accumulator.model,
        failed = ?accumulator.failed,
        tools = ?response_tool_summary(&accumulator),
        "Responses non-stream completed"
    );
    if request.store && accumulator.failed.is_none() {
        save_conversation(response_id, &snapshot, &accumulator);
    }
    record_usage(
        &usage_tracker,
        api_key_id,
        Some(credential_id),
        &accumulator,
    );
    (StatusCode::OK, Json(response_body(&accumulator))).into_response()
}

pub async fn post_chat_completions(
    State(state): State<AppState>,
    identity: Option<Extension<ApiKeyContext>>,
    body: Bytes,
) -> Response {
    let request: ChatCompletionRequest = match serde_json::from_slice(&body) {
        Ok(request) => request,
        Err(error) => {
            return error_response(
                StatusCode::BAD_REQUEST,
                format!("Request body could not be parsed: {error}"),
            );
        }
    };
    let messages = match chat_to_messages(&request) {
        Ok(messages) => messages,
        Err(error) => return error_response(StatusCode::BAD_REQUEST, error),
    };
    let request_body = match build_kiro_request(&state, &messages) {
        Ok(body) => body,
        Err(response) => return response,
    };
    let Some(provider) = state.kiro_provider.clone() else {
        return error_response(StatusCode::SERVICE_UNAVAILABLE, "Kiro API provider not configured");
    };
    if let Some(rpm) = &state.rpm_tracker {
        rpm.record_request(identity.as_ref().map(|context| context.0.id));
    }
    let bound_ids = identity
        .as_ref()
        .and_then(|context| context.0.bound_credential_ids.clone())
        .unwrap_or_default();
    let api_key_id = identity.as_ref().map(|context| context.0.id);
    let usage_tracker = state.usage_tracker.clone();
    let response_id = format!("chatcmpl_{}", Uuid::new_v4().simple());
    let estimated_input_tokens = input_tokens(&messages);
    let exec_command_requested = has_exec_command_tool(&messages);

    if request.stream {
        return chat_stream_response(
            provider,
            request_body,
            bound_ids,
            response_id,
            request.model,
            estimated_input_tokens,
            exec_command_requested,
            usage_tracker,
            api_key_id,
        )
        .await;
    }

    let (accumulator, credential_id) = match execute_non_stream(
        provider,
        &request_body,
        &bound_ids,
        response_id,
        request.model,
        estimated_input_tokens,
        exec_command_requested,
    )
    .await
    {
        Ok(result) => result,
        Err(response) => return response,
    };
    if let Some(error) = &accumulator.failed {
        return error_response(StatusCode::BAD_GATEWAY, error.clone());
    }
    record_usage(
        &usage_tracker,
        api_key_id,
        Some(credential_id),
        &accumulator,
    );
    (StatusCode::OK, Json(chat_completion_body(&accumulator))).into_response()
}

pub async fn post_completions(
    State(state): State<AppState>,
    identity: Option<Extension<ApiKeyContext>>,
    body: Bytes,
) -> Response {
    let request: CompletionRequest = match serde_json::from_slice(&body) {
        Ok(request) => request,
        Err(error) => {
            return error_response(
                StatusCode::BAD_REQUEST,
                format!("Request body could not be parsed: {error}"),
            );
        }
    };
    let messages = match completion_to_messages(&request) {
        Ok(messages) => messages,
        Err(error) => return error_response(StatusCode::BAD_REQUEST, error),
    };
    let request_body = match build_kiro_request(&state, &messages) {
        Ok(body) => body,
        Err(response) => return response,
    };
    let Some(provider) = state.kiro_provider.clone() else {
        return error_response(StatusCode::SERVICE_UNAVAILABLE, "Kiro API provider not configured");
    };
    if let Some(rpm) = &state.rpm_tracker {
        rpm.record_request(identity.as_ref().map(|context| context.0.id));
    }
    let bound_ids = identity
        .as_ref()
        .and_then(|context| context.0.bound_credential_ids.clone())
        .unwrap_or_default();
    let api_key_id = identity.as_ref().map(|context| context.0.id);
    let usage_tracker = state.usage_tracker.clone();
    let response_id = format!("cmpl_{}", Uuid::new_v4().simple());
    let estimated_input_tokens = input_tokens(&messages);
    let exec_command_requested = false;

    if request.stream {
        return completion_stream_response(
            provider,
            request_body,
            bound_ids,
            response_id,
            request.model,
            estimated_input_tokens,
            exec_command_requested,
            usage_tracker,
            api_key_id,
        )
        .await;
    }

    let (accumulator, credential_id) = match execute_non_stream(
        provider,
        &request_body,
        &bound_ids,
        response_id,
        request.model,
        estimated_input_tokens,
        exec_command_requested,
    )
    .await
    {
        Ok(result) => result,
        Err(response) => return response,
    };
    if let Some(error) = &accumulator.failed {
        return error_response(StatusCode::BAD_GATEWAY, error.clone());
    }
    record_usage(
        &usage_tracker,
        api_key_id,
        Some(credential_id),
        &accumulator,
    );
    (StatusCode::OK, Json(completion_body(&accumulator))).into_response()
}

#[allow(clippy::too_many_arguments)]
async fn stream_response(
    provider: Arc<crate::kiro::provider::KiroProvider>,
    request_body: String,
    bound_ids: Vec<u64>,
    response_id: String,
    model: String,
    estimated_input_tokens: i32,
    exec_command_requested: bool,
    snapshot: RequestSnapshot,
    store_response: bool,
    usage_tracker: Option<Arc<UsageTracker>>,
    api_key_id: Option<u32>,
) -> Response {
    let (response, credential_id) = match provider.call_api_stream(&request_body, &bound_ids).await {
        Ok(response) => response,
        Err(error) => return error_response(StatusCode::BAD_GATEWAY, error.to_string()),
    };
    let initial = vec![
        Ok::<Bytes, Infallible>(sse(
            "response.created",
            json!({
                "type": "response.created",
                "sequence_number": 0,
                "response": {
                    "id": response_id,
                    "object": "response",
                    "status": "in_progress",
                    "model": model
                }
            }),
        )),
        Ok::<Bytes, Infallible>(sse(
            "response.in_progress",
            json!({
                "type": "response.in_progress",
                "sequence_number": 1,
                "response": {
                    "id": response_id,
                    "object": "response",
                    "status": "in_progress"
                }
            }),
        )),
    ];
    let body_stream = response.bytes_stream();
    let stream = futures::stream::unfold(
        (
            body_stream,
            EventStreamDecoder::new(),
            ResponseAccumulator::new(
                response_id,
                model,
                estimated_input_tokens,
                exec_command_requested,
            ),
            StreamMarkers {
                sequence_number: 2,
                ..StreamMarkers::default()
            },
            snapshot,
            store_response,
            usage_tracker,
            api_key_id,
            Some(credential_id),
            false,
        ),
        |(
            mut body_stream,
            mut decoder,
            mut accumulator,
            mut markers,
            snapshot,
            store_response,
            usage_tracker,
            api_key_id,
            credential_id,
            finished,
        )| async move {
            if finished {
                return None;
            }
            match body_stream.next().await {
                Some(Ok(chunk)) => {
                    if let Err(error) = decoder.feed(&chunk) {
                        tracing::warn!(%error, "Responses 流式帧缓冲失败");
                    }
                    let mut events: Vec<Result<Bytes, Infallible>> = Vec::new();
                    for result in decoder.decode_iter() {
                        match result {
                            Ok(frame) => match Event::from_frame(frame) {
                                Ok(event) => process_event(&mut accumulator, event),
                                Err(error) => tracing::warn!(%error, "Responses 流式事件解析失败"),
                            },
                            Err(error) => tracing::warn!(%error, "Responses 流式帧解析失败"),
                        }
                    }
                    events.extend(collect_reasoning_stream_events(&accumulator, &mut markers, false));
                    events.extend(collect_text_stream_events(&accumulator, &mut markers, false));
                    events.extend(collect_tool_stream_events(&accumulator, &mut markers));
                    Some((
                        stream::iter(events),
                        (
                            body_stream,
                            decoder,
                            accumulator,
                            markers,
                            snapshot,
                            store_response,
                            usage_tracker,
                            api_key_id,
                            credential_id,
                            false,
                        ),
                    ))
                }
                Some(Err(error)) => {
                    accumulator.failed = Some(error.to_string());
                    accumulator.finish();
                    tracing::warn!(
                        target: "kiro2cc_proxy::responses_diag",
                        response_id = %accumulator.id,
                        model = %accumulator.model,
                        failed = ?accumulator.failed,
                        tools = ?response_tool_summary(&accumulator),
                        "Responses stream failed"
                    );
                    record_usage(&usage_tracker, api_key_id, credential_id, &accumulator);
                    let body = response_body(&accumulator);
                    Some((
                        stream::iter(vec![Ok(numbered_sse(
                            &mut markers,
                            "response.failed",
                            json!({"type": "response.failed", "response": body}),
                        ))]),
                        (
                            body_stream,
                            decoder,
                            accumulator,
                            markers,
                            snapshot,
                            store_response,
                            usage_tracker,
                            api_key_id,
                            credential_id,
                            true,
                        ),
                    ))
                }
                None => {
                    accumulator.finish();
                    if store_response && accumulator.failed.is_none() {
                        save_conversation(accumulator.id.clone(), &snapshot, &accumulator);
                    }
                    record_usage(&usage_tracker, api_key_id, credential_id, &accumulator);
                    let mut events: Vec<Result<Bytes, Infallible>> = Vec::new();
                    events.extend(collect_reasoning_stream_events(&accumulator, &mut markers, true));
                    events.extend(collect_text_stream_events(&accumulator, &mut markers, true));
                    events.extend(collect_tool_stream_events(&accumulator, &mut markers));
                    let body = response_body(&accumulator);
                    let event_name = if accumulator.failed.is_some() {
                        "response.failed"
                    } else {
                        "response.completed"
                    };
                    events.push(Ok(numbered_sse(&mut markers,
                        event_name,
                        json!({"type": event_name, "response": body}),
                    )));
                    Some((
                        stream::iter(events),
                        (
                            body_stream,
                            decoder,
                            accumulator,
                            markers,
                            snapshot,
                            store_response,
                            usage_tracker,
                            api_key_id,
                            credential_id,
                            true,
                        ),
                    ))
                }
            }
        },
    )
    .flatten();
    let output = stream::iter(initial).chain(stream);
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .body(Body::from_stream(output))
        .unwrap()
}

#[allow(clippy::too_many_arguments)]
async fn chat_stream_response(
    provider: Arc<crate::kiro::provider::KiroProvider>,
    request_body: String,
    bound_ids: Vec<u64>,
    response_id: String,
    model: String,
    estimated_input_tokens: i32,
    exec_command_requested: bool,
    usage_tracker: Option<Arc<UsageTracker>>,
    api_key_id: Option<u32>,
) -> Response {
    let (response, credential_id) = match provider.call_api_stream(&request_body, &bound_ids).await {
        Ok(response) => response,
        Err(error) => return error_response(StatusCode::BAD_GATEWAY, error.to_string()),
    };
    let initial = vec![Ok::<Bytes, Infallible>(openai_data_sse(chat_chunk(
        &response_id,
        &model,
        json!({"role": "assistant"}),
        None,
    )))];
    let body_stream = response.bytes_stream();
    let stream = futures::stream::unfold(
        (
            body_stream,
            EventStreamDecoder::new(),
            ResponseAccumulator::new(
                response_id,
                model,
                estimated_input_tokens,
                exec_command_requested,
            ),
            StreamMarkers::default(),
            usage_tracker,
            api_key_id,
            Some(credential_id),
            false,
        ),
        |(
            mut body_stream,
            mut decoder,
            mut accumulator,
            mut markers,
            usage_tracker,
            api_key_id,
            credential_id,
            finished,
        )| async move {
            if finished {
                return None;
            }
            match body_stream.next().await {
                Some(Ok(chunk)) => {
                    if let Err(error) = decoder.feed(&chunk) {
                        tracing::warn!(%error, "Chat Completions 流式帧缓冲失败");
                    }
                    let old_text = accumulator.text.len();
                    let mut events: Vec<Result<Bytes, Infallible>> = Vec::new();
                    for result in decoder.decode_iter() {
                        match result {
                            Ok(frame) => match Event::from_frame(frame) {
                                Ok(event) => process_event(&mut accumulator, event),
                                Err(error) => {
                                    tracing::warn!(%error, "Chat Completions 流式事件解析失败")
                                }
                            },
                            Err(error) => tracing::warn!(%error, "Chat Completions 流式帧解析失败"),
                        }
                    }
                    if accumulator.text.len() > old_text {
                        events.push(Ok(openai_data_sse(chat_chunk(
                            &accumulator.id,
                            &accumulator.model,
                            json!({"content": &accumulator.text[old_text..]}),
                            None,
                        ))));
                    }
                    events.extend(collect_chat_tool_stream_events(&accumulator, &mut markers));
                    Some((
                        stream::iter(events),
                        (
                            body_stream,
                            decoder,
                            accumulator,
                            markers,
                            usage_tracker,
                            api_key_id,
                            credential_id,
                            false,
                        ),
                    ))
                }
                Some(Err(error)) => {
                    accumulator.failed = Some(error.to_string());
                    accumulator.finish();
                    tracing::warn!(
                        target: "kiro2cc_proxy::responses_diag",
                        response_id = %accumulator.id,
                        model = %accumulator.model,
                        failed = ?accumulator.failed,
                        tools = ?response_tool_summary(&accumulator),
                        "Responses stream failed"
                    );
                    record_usage(&usage_tracker, api_key_id, credential_id, &accumulator);
                    let events = vec![
                        Ok(openai_data_sse(json!({
                            "error": {
                                "type": "server_error",
                                "message": accumulator.failed.clone().unwrap_or_default()
                            }
                        }))),
                        Ok(openai_done_sse()),
                    ];
                    Some((
                        stream::iter(events),
                        (
                            body_stream,
                            decoder,
                            accumulator,
                            markers,
                            usage_tracker,
                            api_key_id,
                            credential_id,
                            true,
                        ),
                    ))
                }
                None => {
                    accumulator.finish();
                    record_usage(&usage_tracker, api_key_id, credential_id, &accumulator);
                    let finish_reason = if !accumulator.tools.is_empty() {
                        "tool_calls"
                    } else {
                        "stop"
                    };
                    let events = vec![
                        Ok(openai_data_sse(chat_chunk(
                            &accumulator.id,
                            &accumulator.model,
                            json!({}),
                            Some(finish_reason),
                        ))),
                        Ok(openai_done_sse()),
                    ];
                    Some((
                        stream::iter(events),
                        (
                            body_stream,
                            decoder,
                            accumulator,
                            markers,
                            usage_tracker,
                            api_key_id,
                            credential_id,
                            true,
                        ),
                    ))
                }
            }
        },
    )
    .flatten();
    let output = stream::iter(initial).chain(stream);
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .body(Body::from_stream(output))
        .unwrap()
}

#[allow(clippy::too_many_arguments)]
async fn completion_stream_response(
    provider: Arc<crate::kiro::provider::KiroProvider>,
    request_body: String,
    bound_ids: Vec<u64>,
    response_id: String,
    model: String,
    estimated_input_tokens: i32,
    exec_command_requested: bool,
    usage_tracker: Option<Arc<UsageTracker>>,
    api_key_id: Option<u32>,
) -> Response {
    let (response, credential_id) = match provider.call_api_stream(&request_body, &bound_ids).await {
        Ok(response) => response,
        Err(error) => return error_response(StatusCode::BAD_GATEWAY, error.to_string()),
    };
    let body_stream = response.bytes_stream();
    let stream = futures::stream::unfold(
        (
            body_stream,
            EventStreamDecoder::new(),
            ResponseAccumulator::new(
                response_id,
                model,
                estimated_input_tokens,
                exec_command_requested,
            ),
            usage_tracker,
            api_key_id,
            Some(credential_id),
            false,
        ),
        |(
            mut body_stream,
            mut decoder,
            mut accumulator,
            usage_tracker,
            api_key_id,
            credential_id,
            finished,
        )| async move {
            if finished {
                return None;
            }
            match body_stream.next().await {
                Some(Ok(chunk)) => {
                    if let Err(error) = decoder.feed(&chunk) {
                        tracing::warn!(%error, "Completions 流式帧缓冲失败");
                    }
                    let old_text = accumulator.text.len();
                    let mut events: Vec<Result<Bytes, Infallible>> = Vec::new();
                    for result in decoder.decode_iter() {
                        match result {
                            Ok(frame) => match Event::from_frame(frame) {
                                Ok(event) => process_event(&mut accumulator, event),
                                Err(error) => {
                                    tracing::warn!(%error, "Completions 流式事件解析失败")
                                }
                            },
                            Err(error) => tracing::warn!(%error, "Completions 流式帧解析失败"),
                        }
                    }
                    if accumulator.text.len() > old_text {
                        events.push(Ok(openai_data_sse(completion_chunk(
                            &accumulator.id,
                            &accumulator.model,
                            &accumulator.text[old_text..],
                            None,
                        ))));
                    }
                    Some((
                        stream::iter(events),
                        (
                            body_stream,
                            decoder,
                            accumulator,
                            usage_tracker,
                            api_key_id,
                            credential_id,
                            false,
                        ),
                    ))
                }
                Some(Err(error)) => {
                    accumulator.failed = Some(error.to_string());
                    accumulator.finish();
                    tracing::warn!(
                        target: "kiro2cc_proxy::responses_diag",
                        response_id = %accumulator.id,
                        model = %accumulator.model,
                        failed = ?accumulator.failed,
                        tools = ?response_tool_summary(&accumulator),
                        "Responses stream failed"
                    );
                    record_usage(&usage_tracker, api_key_id, credential_id, &accumulator);
                    let events = vec![
                        Ok(openai_data_sse(json!({
                            "error": {
                                "type": "server_error",
                                "message": accumulator.failed.clone().unwrap_or_default()
                            }
                        }))),
                        Ok(openai_done_sse()),
                    ];
                    Some((
                        stream::iter(events),
                        (
                            body_stream,
                            decoder,
                            accumulator,
                            usage_tracker,
                            api_key_id,
                            credential_id,
                            true,
                        ),
                    ))
                }
                None => {
                    accumulator.finish();
                    record_usage(&usage_tracker, api_key_id, credential_id, &accumulator);
                    let events = vec![
                        Ok(openai_data_sse(completion_chunk(
                            &accumulator.id,
                            &accumulator.model,
                            "",
                            Some("stop"),
                        ))),
                        Ok(openai_done_sse()),
                    ];
                    Some((
                        stream::iter(events),
                        (
                            body_stream,
                            decoder,
                            accumulator,
                            usage_tracker,
                            api_key_id,
                            credential_id,
                            true,
                        ),
                    ))
                }
            }
        },
    )
    .flatten();
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .body(Body::from_stream(stream))
        .unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exec_tool_uses_code_mode_input_schema() {
        let tools = function_tools(Some(&[json!({
            "type": "function",
            "name": "exec",
            "parameters": {
                "type": "object",
                "properties": {"cmd": {"type": "string"}}
            }
        })]))
        .unwrap()
        .unwrap();

        assert_eq!(tools[0].name, "exec");
        assert_eq!(tools[0].input_schema.get("required"), Some(&json!(["input"])));
        assert!(tools[0].input_schema.contains_key("properties"));
        assert_eq!(tools[0].input_schema["properties"]["input"]["type"], "string");
    }

    #[test]
    fn code_mode_exec_waits_for_argument_kind() {
        let mut tool = ToolState {
            call_id: "call_1".into(),
            name: "exec".into(),
            arguments: "{\"".into(),
            done: false,
        };
        assert!(!tool_stream_ready(&tool));

        tool.arguments.push_str("cmd\":\"pwd\"}");
        assert!(tool_stream_ready(&tool));

        tool.arguments = "{\"input\":\"pwd\"}".into();
        assert!(tool_stream_ready(&tool));
    }

    #[test]
    fn normalize_exec_only_when_exec_command_was_requested() {
        assert_eq!(normalize_tool_name("exec", true, "{}"), "exec_command");
        assert_eq!(normalize_tool_name("exec", false, "{}"), "exec");
        assert_eq!(normalize_tool_name("lookup", true, "{}"), "lookup");
        assert_eq!(normalize_tool_name("exec", false, "{\"cmd\":"), "exec_command");
    }

    #[test]
    fn push_tool_normalizes_and_merges_by_call_id() {
        let mut accumulator = ResponseAccumulator::new("resp".into(), "model".into(), 0, true);

        accumulator.push_tool("exec", "call_1", "{\"cmd\":", false);
        accumulator.push_tool("exec", "call_1", "\"pwd\"}", true);
        accumulator.push_tool("lookup", "call_2", "{\"q\":\"x\"}", true);

        assert_eq!(accumulator.tools.len(), 2);
        assert_eq!(accumulator.tools[0].name, "exec_command");
        assert_eq!(accumulator.tools[0].arguments, "{\"cmd\":\"pwd\"}");
        assert!(accumulator.tools[0].done);
        assert_eq!(accumulator.tools[1].name, "lookup");
        assert_eq!(function_call_item(&accumulator.tools[0], "completed")["name"], "exec_command");
    }

    #[test]
    fn ordinary_exec_function_is_unchanged_without_exec_command_tool() {
        let mut accumulator = ResponseAccumulator::new("resp".into(), "model".into(), 0, false);
        accumulator.push_tool("exec", "call_1", "{}", true);

        assert_eq!(accumulator.tools[0].name, "exec");
    }

    #[test]
    fn tool_added_event_does_not_duplicate_arguments_delta() {
        let mut accumulator = ResponseAccumulator::new("resp".into(), "model".into(), 0, false);
        accumulator.push_tool("exec", "call_1", "{\"input\":\"pwd\"}", true);
        let mut markers = StreamMarkers::default();
        let events = collect_tool_stream_events(&accumulator, &mut markers);
        let output = events
            .into_iter()
            .map(|event| String::from_utf8(event.unwrap().to_vec()).unwrap())
            .collect::<String>();

        assert!(output.contains("event: response.output_item.added\n"));
        assert!(output.contains("\"type\":\"custom_tool_call\""));
        assert!(output.contains("\"input\":\"\""));
        assert!(output.contains("event: response.custom_tool_call_input.delta\n"));
        assert!(output.contains("\"delta\":\"pwd\""));
        assert!(output.contains("event: response.custom_tool_call_input.done\n"));
        assert!(output.contains("\"input\":\"pwd\""));
        assert!(!output.contains("response.function_call_arguments"));
        assert!(!output.contains("{\\\"input\\\":\\\"pwd\\\"}"));
    }
}
