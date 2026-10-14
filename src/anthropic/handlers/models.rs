// Copyright (c) 2026 Harllan He. Licensed under MIT.
//! 模型列表 Handler：/v1/models 端点，静态列表与上游动态拉取相结合，带 TTL 缓存（CachedModels）。

use crate::anthropic::middleware::AppState;
use crate::anthropic::types::{ErrorResponse, Model, ModelsResponse};
use crate::kiro::model::available_models::AvailableModelInfo;

use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Json, Response},
};
use std::time::Duration;

/// 仅屏蔽显式列出的模型族；按 `map_model()` 归一化后判断，日期版和 thinking 别名一并隐藏。
/// 未列出的模型保持可见，便于新模型无需改代码即可发现。
const BLOCKED_MODEL_IDS: &[&str] = &[
    "auto",
    "claude-haiku-4.5",
    "claude-opus-4.5",
    "claude-opus-4.7",
    "claude-opus-5",
    "claude-sonnet-4",
    "claude-sonnet-4.5",
    "claude-sonnet-5",
    "deepseek-3.2",
    "glm-5",
    "minimax-m2.1",
    "minimax-m2.5",
    "qwen3-coder-next",
];

/// GET /v1/models
///
/// 返回可用的模型列表（动态来源，带 TTL 缓存 + 静态表回退）
pub async fn get_models(State(state): State<AppState>) -> impl IntoResponse {
    tracing::info!("Received GET /v1/models request");

    Json(ModelsResponse {
        object: "list".to_string(),
        data: fetch_models_dynamic(&state).await,
    })
}

/// 模型缓存别名，简化签名
pub(crate) type ModelCache =
    std::sync::Arc<parking_lot::RwLock<Option<super::super::middleware::CachedModels>>>;

pub(crate) fn is_blocked_model(id: &str) -> bool {
    let canonical_id = crate::anthropic::converter::map_model(id)
        .unwrap_or_else(|| id.trim().to_lowercase());
    BLOCKED_MODEL_IDS.contains(&canonical_id.as_str())
}

/// 判断模型是否为 thinking 变体。
///
/// 模型目录只展示普通版本；`-thinking` / `-Thinking` 等旧别名仍由
/// `map_model()` 归一化到普通模型，兼容已缓存模型列表的旧客户端。
pub(crate) fn is_thinking_variant(id: &str) -> bool {
    id.trim().to_ascii_lowercase().ends_with("-thinking")
}

/// 判断模型是否应从对外模型目录隐藏。
pub(crate) fn is_hidden_model(id: &str) -> bool {
    is_thinking_variant(id) || is_blocked_model(id)
}

fn visible_models(models: Vec<Model>) -> Vec<Model> {
    models
        .into_iter()
        .filter(|model| !is_hidden_model(&model.id))
        .collect()
}

/// 将 Kiro 动态元数据转换为对外模型目录。
///
/// 动态发现成功时只返回动态模型，不再把静态别名混入目录；动态发现不可用时
/// 传入 `use_dynamic_metadata=false`，回退到唯一的静态模型。
pub(crate) fn build_model_catalog(
    metadata: Option<&[AvailableModelInfo]>,
    use_dynamic_metadata: bool,
) -> Vec<Model> {
    if !use_dynamic_metadata {
        return build_model_list();
    }

    let models: Vec<_> = metadata
        .unwrap_or_default()
        .iter()
        .filter(|info| !info.model_id.trim().is_empty() && !is_hidden_model(&info.model_id))
        .map(available_model_to_model)
        .collect();
    if models.is_empty() {
        build_model_list()
    } else {
        models
    }
}

/// 动态获取模型列表：优先上游实时响应（带 TTL 缓存），失败按顺序回退。
///
/// 分支：
/// 1. 缓存存在且未过期 → 直接返回缓存，不打上游
/// 2. 共享动态元数据仍在 TTL 内 → 返回动态模型
/// 3. 未配置 `kiro_provider` → 直接回退静态表（无上游可查）
/// 4. 上游成功返回非空模型集 → 使用动态模型、写缓存、返回
/// 5. 上游失败 → 平滑回退静态表 `build_model_list()`（warn）
pub(crate) async fn fetch_models_dynamic(state: &AppState) -> Vec<Model> {
    // 未配置上游 provider，直接静态表
    let Some(provider) = state.kiro_provider.as_ref() else {
        return build_model_list();
    };

    let ttl = Duration::from_secs(provider.token_manager().config().model_cache_ttl_secs);

    // 分支 1：缓存命中且未过期
    if let Some(hit) = cached_if_fresh(&state.model_cache, ttl) {
        return visible_models(hit);
    }

    if let Some(metadata) = provider.token_manager().cached_model_catalog(ttl) {
        return build_model_catalog(Some(&metadata), true);
    }

    // 缓存缺失/过期，尝试刷新上游（仅此处涉及网络；结果归一化为 Option<Vec<Model>>）
    let refreshed: Option<Vec<Model>> = match provider.token_manager().list_available_models().await
    {
        Ok(resp) if !resp.models.is_empty() => {
            let models = build_model_catalog(Some(&resp.models), true);
            (!models.is_empty()).then_some(models)
        }
        Ok(_) => {
            tracing::warn!("上游模型列表为空，回退静态表");
            None
        }
        Err(e) => {
            tracing::warn!(error = %e, "刷新上游模型列表失败，回退静态表");
            None
        }
    };

    resolve_after_refresh(&state.model_cache, refreshed)
}

/// 缓存命中判定（纯逻辑，无网络）：存在且未超过 TTL 时返回克隆的模型列表。
pub(crate) fn cached_if_fresh(cache: &ModelCache, ttl: Duration) -> Option<Vec<Model>> {
    let guard = cache.read();
    guard
        .as_ref()
        .filter(|cached| cached.fetched_at.elapsed() < ttl)
        .map(|cached| visible_models(cached.models.clone()))
        .filter(|models| !models.is_empty())
}

/// 刷新结果落地（纯逻辑，无网络）：
/// - `Some(非空)` → 写缓存并返回（分支 2）
/// - `None` → 平滑回退静态表（分支 3/4）
pub(crate) fn resolve_after_refresh(
    cache: &ModelCache,
    refreshed: Option<Vec<Model>>,
) -> Vec<Model> {
    if let Some(models) = refreshed {
        let models = visible_models(models);
        if !models.is_empty() {
            *cache.write() = Some(super::super::middleware::CachedModels {
                models: models.clone(),
                fetched_at: std::time::Instant::now(),
            });
            return models;
        }
    }
    build_model_list()
}

/// 构建可用模型列表（供 get_models 和 get_model 共用）
pub(crate) fn build_model_list() -> Vec<Model> {
    // 静态表只保留最小回退项；正常情况下以 Kiro 动态目录为准。
    visible_models(vec![Model {
        id: "claude-opus-5.5".to_string(),
        object: "model".to_string(),
        created: 0,
        owned_by: "anthropic".to_string(),
        display_name: "Claude Opus 5.5".to_string(),
        model_type: "chat".to_string(),
        context_window: Some(1_000_000),
        max_tokens: 128_000,
    }])
}

/// 根据模型 ID 前缀推断提供方（ListAvailableModels 响应不含厂商归属字段）
///
/// 命名规则与模型 ID 的提供方前缀保持一致；未知前缀返回 `"unknown"`。
/// 供 `/v1/models` 动态映射与 Admin 端共享，避免两份实现漂移。
pub(crate) fn guess_owned_by(model_id: &str) -> &'static str {
    let id = model_id.to_lowercase();
    if id.contains("claude") {
        "anthropic"
    } else if id.contains("gpt") {
        "openai"
    } else if id == "auto" {
        "kiro"
    } else if id.contains("deepseek") {
        "deepseek"
    } else if id.contains("minimax") {
        "minimax"
    } else if id.contains("glm") {
        "glm"
    } else if id.contains("qwen") {
        "qwen"
    } else {
        "unknown"
    }
}

/// 将上游 `ListAvailableModels` 返回的单条模型映射为 Anthropic `Model`
///
/// 纯函数，不涉及网络调用，可直接用 fake `AvailableModelInfo` 单测。
pub(crate) fn available_model_to_model(
    info: &crate::kiro::model::available_models::AvailableModelInfo,
) -> Model {
    Model {
        id: info.model_id.clone(),
        object: "model".to_string(),
        created: 0,
        owned_by: guess_owned_by(&info.model_id).to_string(),
        display_name: if info.model_name.trim().is_empty() {
            info.model_id.clone()
        } else {
            info.model_name.clone()
        },
        model_type: "chat".to_string(),
        context_window: i32::try_from(info.token_limits.max_input_tokens)
            .ok()
            .filter(|value| *value > 0),
        max_tokens: i32::try_from(info.token_limits.max_output_tokens).unwrap_or(i32::MAX),
    }
}

/// GET /v1/models/:model_id
///
/// 返回指定模型的信息
pub async fn get_model(
    State(state): State<AppState>,
    axum::extract::Path(model_id): axum::extract::Path<String>,
) -> Response {
    tracing::info!(model_id = %model_id, "Received GET /v1/models/:model_id request");

    // 与 /v1/models 相同的动态来源，查找匹配的模型
    let models = fetch_models_dynamic(&state).await;
    if let Some(model) = models.into_iter().find(|m| m.id == model_id) {
        Json(model).into_response()
    } else {
        (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse::new(
                "not_found_error",
                format!("Model '{}' not found", model_id),
            )),
        )
            .into_response()
    }
}
