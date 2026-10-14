// Copyright (c) 2026 Harllan He. Licensed under MIT.
// 模型列表与缓存测试（自 handlers/tests.rs 拆出，纯代码搬移）
#[cfg(test)]
mod tests {

    use super::super::super::models::{
        ModelCache, available_model_to_model, build_model_catalog, build_model_list,
        cached_if_fresh, fetch_models_dynamic, get_model, guess_owned_by, resolve_after_refresh,
    };

    use crate::anthropic::middleware::AppState;

    use crate::anthropic::types::Model;

    use axum::{extract::State, http::StatusCode};

    use std::time::Duration;

    fn find_by_id(id: &str) -> Option<Model> {
        build_model_list().into_iter().find(|m| m.id == id)
    }

    #[test]
    fn test_guess_owned_by_known_and_unknown_prefixes() {
        assert_eq!(guess_owned_by("claude-sonnet-4.6"), "anthropic");
        assert_eq!(guess_owned_by("gpt-5.6-sol"), "openai");
        assert_eq!(guess_owned_by("auto"), "kiro");
        assert_eq!(guess_owned_by("deepseek-3.2"), "deepseek");
        assert_eq!(guess_owned_by("minimax-m2.5"), "minimax");
        assert_eq!(guess_owned_by("glm-5"), "glm");
        assert_eq!(guess_owned_by("qwen3-coder-next"), "qwen");
        assert_eq!(guess_owned_by("foo-model"), "unknown");
    }

    fn fake_model(id: &str) -> Model {
        Model {
            id: id.to_string(),
            object: "model".to_string(),
            created: 0,
            owned_by: "test".to_string(),
            display_name: id.to_string(),
            model_type: "chat".to_string(),
            context_window: None,
            max_tokens: 8192,
        }
    }

    fn new_cache(entry: Option<crate::anthropic::middleware::CachedModels>) -> ModelCache {
        std::sync::Arc::new(parking_lot::RwLock::new(entry))
    }

    // 分支 1：缓存命中且未过期 → 返回缓存，不触发刷新
    #[test]
    fn test_cached_if_fresh_hit() {
        let cache = new_cache(Some(crate::anthropic::middleware::CachedModels {
            models: vec![fake_model("cached-a")],
            fetched_at: std::time::Instant::now(),
        }));
        let hit = cached_if_fresh(&cache, Duration::from_secs(3600));
        assert!(hit.is_some());
        assert_eq!(hit.unwrap()[0].id, "cached-a");
    }

    #[test]
    fn test_cached_if_fresh_hides_blocked_models() {
        let cache = new_cache(Some(crate::anthropic::middleware::CachedModels {
            models: vec![fake_model("auto"), fake_model("future-model")],
            fetched_at: std::time::Instant::now(),
        }));
        let hit = cached_if_fresh(&cache, Duration::from_secs(3600)).unwrap();
        assert_eq!(hit.len(), 1);
        assert_eq!(hit[0].id, "future-model");
    }

    #[test]
    fn test_cached_if_fresh_rejects_cache_containing_only_blocked_models() {
        let cache = new_cache(Some(crate::anthropic::middleware::CachedModels {
            models: vec![fake_model("auto")],
            fetched_at: std::time::Instant::now(),
        }));
        assert!(cached_if_fresh(&cache, Duration::from_secs(3600)).is_none());
    }

    // 分支 1 反例：缓存过期 → 视为未命中
    #[test]
    fn test_cached_if_fresh_expired() {
        let cache = new_cache(Some(crate::anthropic::middleware::CachedModels {
            models: vec![fake_model("stale")],
            fetched_at: std::time::Instant::now() - Duration::from_secs(10),
        }));
        // TTL 5s，已过 10s
        assert!(cached_if_fresh(&cache, Duration::from_secs(5)).is_none());
        // 空缓存亦未命中
        assert!(cached_if_fresh(&new_cache(None), Duration::from_secs(3600)).is_none());
    }

    // 分支 2：刷新成功 → 写缓存并返回
    #[test]
    fn test_resolve_after_refresh_success_writes_cache() {
        let cache = new_cache(None);
        let out = resolve_after_refresh(&cache, Some(vec![fake_model("fresh")]));
        assert_eq!(out[0].id, "fresh");
        // 缓存已写入
        let guard = cache.read();
        assert_eq!(guard.as_ref().unwrap().models[0].id, "fresh");
    }

    #[test]
    fn test_resolve_after_refresh_hides_blocked_models() {
        let cache = new_cache(None);
        let out = resolve_after_refresh(
            &cache,
            Some(vec![fake_model("auto"), fake_model("future-model")]),
        );
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].id, "future-model");
        let guard = cache.read();
        assert_eq!(guard.as_ref().unwrap().models.len(), 1);
        assert_eq!(guard.as_ref().unwrap().models[0].id, "future-model");
    }

    #[test]
    fn test_resolve_after_refresh_falls_back_when_all_models_are_blocked() {
        let cache = new_cache(None);
        let out = resolve_after_refresh(&cache, Some(vec![fake_model("auto")]));
        assert_eq!(out.len(), build_model_list().len());
        assert!(out.iter().any(|model| model.id == "claude-opus-4-6"));
    }

    // 分支 3：刷新失败但有旧缓存 → 仍平滑回退静态表
    #[test]
    fn test_resolve_after_refresh_failure_falls_back_static_despite_old_cache() {
        let cache = new_cache(Some(crate::anthropic::middleware::CachedModels {
            models: vec![fake_model("old")],
            fetched_at: std::time::Instant::now(),
        }));
        let out = resolve_after_refresh(&cache, None);
        assert_eq!(out.len(), build_model_list().len());
        assert!(out.iter().any(|model| model.id == "claude-opus-4-6"));
        assert!(!out.iter().any(|model| model.id == "old"));
    }

    // 分支 4：刷新失败且无缓存 → 回退静态表
    #[test]
    fn test_resolve_after_refresh_failure_no_cache_falls_back_static() {
        let cache = new_cache(None);
        let out = resolve_after_refresh(&cache, None);
        assert_eq!(out.len(), build_model_list().len());
        assert!(out.iter().any(|m| m.id == "claude-opus-4-6"));
    }

    // 分支 5：无 provider → fetch_models_dynamic 直接回退静态表
    #[tokio::test]
    async fn test_fetch_models_dynamic_no_provider_static() {
        let state = AppState::new();
        assert!(state.kiro_provider.is_none());
        let out = fetch_models_dynamic(&state).await;
        assert_eq!(out.len(), build_model_list().len());
    }

    fn test_state() -> AppState {
        AppState::new()
    }

    // get_model 命中（无 provider → 静态表来源）
    #[tokio::test]
    async fn test_get_model_hit() {
        let resp = get_model(
            State(test_state()),
            axum::extract::Path("claude-opus-4-6".to_string()),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::OK);
    }

    // get_model 未命中 → 404
    #[tokio::test]
    async fn test_get_model_not_found() {
        let resp = get_model(
            State(test_state()),
            axum::extract::Path("no-such-model-xyz".to_string()),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_get_model_blocked_model_not_found() {
        let resp = get_model(
            State(test_state()),
            axum::extract::Path("auto".to_string()),
        )
        .await;
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[test]
    fn test_build_model_list_hides_blocked_models() {
        let list = build_model_list();
        for id in [
            "auto",
            "claude-haiku-4.5",
            "claude-haiku-4-5",
            "claude-haiku-4-5-20251001",
            "claude-haiku-4-5-20251001-thinking",
            "claude-opus-4.5",
            "claude-opus-4-5",
            "claude-opus-4-5-20251101",
            "claude-opus-4-5-20251101-thinking",
            "claude-opus-4.7",
            "claude-opus-4-7",
            "claude-opus-4-7-thinking",
            "claude-opus-5",
            "claude-opus-5-thinking",
            "claude-sonnet-4.5",
            "claude-sonnet-4-5",
            "claude-sonnet-4-5-20250929",
            "claude-sonnet-4-5-20250929-thinking",
            "claude-sonnet-4",
            "claude-sonnet-4-20250514",
            "claude-sonnet-5",
            "claude-sonnet-5-thinking",
            "deepseek-3.2",
            "glm-5",
            "minimax-m2.1",
            "minimax-m2.5",
            "qwen3-coder-next",
        ] {
            assert!(!list.iter().any(|model| model.id == id), "{id} 不应可见");
        }
        assert!(list.iter().any(|model| model.id == "claude-sonnet-4-6"));
        assert!(list.iter().any(|model| model.id == "claude-opus-4-6"));
        assert!(list.iter().any(|model| model.id == "claude-opus-4-8"));
    }

    #[test]
    fn test_available_model_to_model_maps_fields() {
        use crate::kiro::model::available_models::{AvailableModelInfo, TokenLimits};
        let info = AvailableModelInfo {
            model_id: "claude-sonnet-4.6".to_string(),
            model_name: "Claude Sonnet 4.6".to_string(),
            rate_multiplier: Some(1.3),
            token_limits: TokenLimits {
                max_input_tokens: 1_000_000,
                max_output_tokens: 128_000,
            },
            additional_model_request_fields_schema: None,
        };
        let m = available_model_to_model(&info);
        assert_eq!(m.id, "claude-sonnet-4.6");
        assert_eq!(m.display_name, "Claude Sonnet 4.6");
        assert_eq!(m.owned_by, "anthropic");
        assert_eq!(m.context_window, Some(1_000_000));
        assert_eq!(m.max_tokens, 128_000);
        assert_eq!(m.object, "model");
        assert_eq!(m.model_type, "chat");
    }

    #[test]
    fn test_build_model_catalog_enriches_static_and_appends_new_models() {
        use crate::kiro::model::available_models::{AvailableModelInfo, TokenLimits};

        let metadata = vec![
            AvailableModelInfo {
                model_id: "claude-sonnet-4-6".to_string(),
                model_name: "Live Sonnet".to_string(),
                rate_multiplier: Some(1.3),
                token_limits: TokenLimits {
                    max_input_tokens: 1_000_000,
                    max_output_tokens: 99_999,
                },
                additional_model_request_fields_schema: None,
            },
            AvailableModelInfo {
                model_id: "future-model".to_string(),
                model_name: "Future Model".to_string(),
                rate_multiplier: None,
                token_limits: TokenLimits::default(),
                additional_model_request_fields_schema: None,
            },
        ];

        let models = build_model_catalog(Some(&metadata), true);
        let sonnet = models
            .iter()
            .find(|model| model.id == "claude-sonnet-4-6")
            .unwrap();
        assert_eq!(sonnet.display_name, "Live Sonnet");
        assert_eq!(sonnet.max_tokens, 99_999);
        assert!(models.iter().any(|model| model.id == "future-model"));
        assert!(models.iter().any(|model| model.id == "claude-opus-4-8"));

        let fallback = build_model_catalog(Some(&metadata), false);
        assert!(fallback.iter().any(|model| model.id == "claude-opus-4-8"));
        assert!(!fallback.iter().any(|model| model.id == "future-model"));
        let fallback_sonnet = fallback
            .iter()
            .find(|model| model.id == "claude-sonnet-4-6")
            .unwrap();
        assert_eq!(fallback_sonnet.display_name, "Claude Sonnet 4.6");
        assert_eq!(fallback_sonnet.max_tokens, 64_000);
    }

    #[test]
    fn test_opus_4_6_max_tokens_is_128k() {
        let m = find_by_id("claude-opus-4-6").expect("claude-opus-4-6 缺失");
        assert_eq!(m.max_tokens, 128000);
        let mt = find_by_id("claude-opus-4-6-thinking").expect("claude-opus-4-6-thinking 缺失");
        assert_eq!(mt.max_tokens, 128000);
    }

    #[test]
    fn test_allowed_model_max_tokens_unchanged() {
        assert_eq!(find_by_id("claude-opus-4-8").unwrap().max_tokens, 128000);
        assert_eq!(
            find_by_id("claude-opus-4-8-thinking").unwrap().max_tokens,
            128000
        );
    }

    #[test]
    fn test_sonnet_4_6_max_tokens_unchanged() {
        // 回归
        assert_eq!(find_by_id("claude-sonnet-4-6").unwrap().max_tokens, 64000);
    }
}
