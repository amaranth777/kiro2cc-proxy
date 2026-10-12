// Copyright (c) 2026 Harllan He. Licensed under MIT.
//! Kiro 动态模型元数据的进程内共享目录。
//!
//! 这里保存 `ListAvailableModels` 的原始元数据，供 Anthropic 模型列表、Admin
//! 模型页及后续模型能力查询共享；动态发现失败时，调用方仍可回退静态模型表。

use crate::kiro::model::available_models::AvailableModelInfo;
use parking_lot::RwLock;
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
struct ModelCatalogSnapshot {
    models: Vec<AvailableModelInfo>,
    fetched_at: Instant,
}

#[derive(Debug, Default)]
pub struct ModelCatalogStore {
    snapshot: RwLock<Option<ModelCatalogSnapshot>>,
}

impl ModelCatalogStore {
    pub fn remember(&self, models: Vec<AvailableModelInfo>) {
        *self.snapshot.write() = Some(ModelCatalogSnapshot {
            models,
            fetched_at: Instant::now(),
        });
    }

    pub fn fresh(&self, ttl: Duration) -> Option<Vec<AvailableModelInfo>> {
        self.snapshot
            .read()
            .as_ref()
            .filter(|snapshot| snapshot.fetched_at.elapsed() < ttl)
            .map(|snapshot| snapshot.models.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kiro::model::available_models::TokenLimits;

    fn info(id: &str) -> AvailableModelInfo {
        AvailableModelInfo {
            model_id: id.to_string(),
            model_name: id.to_string(),
            rate_multiplier: Some(1.0),
            token_limits: TokenLimits::default(),
            additional_model_request_fields_schema: None,
        }
    }

    #[test]
    fn remembers_models_within_ttl() {
        let store = ModelCatalogStore::default();
        store.remember(vec![info("claude-sonnet-4-6")]);

        let models = store.fresh(Duration::from_secs(1)).unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].model_id, "claude-sonnet-4-6");
    }

    #[test]
    fn expires_models_after_ttl() {
        let store = ModelCatalogStore::default();
        store.remember(vec![info("future-model")]);

        assert!(store.fresh(Duration::ZERO).is_none());
    }
}
