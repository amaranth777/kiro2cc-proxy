// Copyright (c) 2026 Harllan He. Licensed under MIT.
use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

/// 单个 API Key
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ApiKey {
    pub id: u32,
    pub key: String,
    pub name: String,
    #[serde(default = "default_enabled")]
    pub enabled: bool,
    pub created_at: DateTime<Utc>,
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<DateTime<Utc>>,
    /// 额度限制数值，None 表示不限额（按日期模式）
    /// 单位由 `limit_unit` 决定：`"usd"`（默认，estimated_cost 累加）或 `"credits"`（真实 credits 累加）
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spending_limit: Option<f64>,
    /// 额度计量单位（"usd" | "credits"），默认 "usd" 保持向后兼容
    #[serde(default = "default_limit_unit")]
    pub limit_unit: String,
    /// 有效期天数（懒激活模式），首次使用后才开始计时
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration_days: Option<f64>,
    /// 首次使用激活时间
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub activated_at: Option<DateTime<Utc>>,
    /// 绑定的账号 ID 列表，None 或空列表表示不限制（使用全局策略）
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bound_credential_ids: Option<Vec<u64>>,
}

fn default_enabled() -> bool {
    true
}

fn default_limit_unit() -> String {
    "usd".to_string()
}

impl ApiKey {
    /// 生成新的 API Key
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: u32,
        name: String,
        expires_at: Option<DateTime<Utc>>,
        spending_limit: Option<f64>,
        limit_unit: String,
        duration_days: Option<f64>,
        bound_credential_ids: Option<Vec<u64>>,
    ) -> Self {
        Self {
            id,
            key: generate_api_key(),
            name,
            enabled: true,
            created_at: Utc::now(),
            expires_at,
            spending_limit,
            limit_unit,
            duration_days,
            activated_at: None,
            bound_credential_ids,
        }
    }

    /// 检查 key 是否有效（启用且未过期）
    #[allow(dead_code)]
    pub fn is_valid(&self) -> bool {
        if !self.enabled {
            return false;
        }
        if let Some(expires_at) = self.expires_at {
            return Utc::now() < expires_at;
        }
        true
    }

    /// 检查是否已过期
    /// 待激活状态（duration_days 有值但 activated_at 为 None）返回 false
    pub fn is_expired(&self) -> bool {
        if self.duration_days.is_some() && self.activated_at.is_none() {
            return false;
        }
        self.expires_at
            .map(|exp| Utc::now() >= exp)
            .unwrap_or(false)
    }

    /// 检查是否为活跃状态（已激活且未过期）
    pub fn is_active(&self) -> bool {
        self.activated_at.is_some() && !self.is_expired()
    }

    /// 激活 key：设置 activated_at 并计算 expires_at
    /// 幂等操作，已激活的 key 直接跳过
    pub fn activate(&mut self) -> bool {
        if self.activated_at.is_some() || self.duration_days.is_none() {
            return false;
        }
        let now = Utc::now();
        let days = self.duration_days.unwrap();
        let duration = chrono::Duration::milliseconds((days * 86_400_000.0) as i64);
        self.activated_at = Some(now);
        self.expires_at = Some(now + duration);
        true
    }
}
/// 生成 sk- 前缀的随机 API Key
fn generate_api_key() -> String {
    let id = uuid::Uuid::new_v4();
    format!("sk-{}", id.simple())
}

/// API Key 认证结果
pub enum ApiKeyAuthResult {
    /// 认证通过，携带 key ID 和名称
    Valid {
        id: u32,
        name: String,
        spending_limit: Option<f64>,
        limit_unit: String,
        bound_credential_ids: Option<Vec<u64>>,
    },
    /// Key 已被禁用
    Disabled,
    /// Key 已过期
    Expired,
    /// Key 不存在
    NotFound,
}

/// API Key 管理器（线程安全）
pub struct ApiKeyManager {
    keys: RwLock<Vec<ApiKey>>,
    file_path: PathBuf,
}

impl ApiKeyManager {
    /// 从文件加载，文件不存在则创建空列表
    pub fn load<P: AsRef<Path>>(path: P) -> anyhow::Result<Self> {
        let path = path.as_ref().to_path_buf();
        let keys = if path.exists() {
            let content = fs::read_to_string(&path)?;
            if content.trim().is_empty() {
                Vec::new()
            } else {
                serde_json::from_str(&content)?
            }
        } else {
            Vec::new()
        };
        Ok(Self {
            keys: RwLock::new(keys),
            file_path: path,
        })
    }

    /// 持久化到文件
    fn save(&self) -> anyhow::Result<()> {
        let keys = self.keys.read();
        let content = serde_json::to_string_pretty(&*keys)?;
        if let Some(parent) = self.file_path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&self.file_path, content)?;
        Ok(())
    }

    /// 验证请求中的 key
    pub fn authenticate(&self, key: &str) -> ApiKeyAuthResult {
        let keys = self.keys.read();
        match keys.iter().find(|k| k.key == key) {
            Some(api_key) => {
                if !api_key.enabled {
                    ApiKeyAuthResult::Disabled
                } else if api_key.is_expired() {
                    ApiKeyAuthResult::Expired
                } else {
                    ApiKeyAuthResult::Valid {
                        id: api_key.id,
                        name: api_key.name.clone(),
                        spending_limit: api_key.spending_limit,
                        limit_unit: api_key.limit_unit.clone(),
                        bound_credential_ids: api_key.bound_credential_ids.clone(),
                    }
                }
            }
            None => ApiKeyAuthResult::NotFound,
        }
    }

    /// 只读认证：只要 key 存在就放行（不检查过期/禁用/额度）
    /// 用于用户查询用量等只读场景
    pub fn authenticate_readonly(&self, key: &str) -> ApiKeyAuthResult {
        let keys = self.keys.read();
        match keys.iter().find(|k| k.key == key) {
            Some(api_key) => ApiKeyAuthResult::Valid {
                id: api_key.id,
                name: api_key.name.clone(),
                spending_limit: api_key.spending_limit,
                limit_unit: api_key.limit_unit.clone(),
                bound_credential_ids: api_key.bound_credential_ids.clone(),
            },
            None => ApiKeyAuthResult::NotFound,
        }
    }
    /// 获取所有 key（克隆）
    pub fn list(&self) -> Vec<ApiKey> {
        self.keys.read().clone()
    }

    /// 创建新 key
    #[allow(clippy::too_many_arguments)]
    pub fn create(
        &self,
        name: String,
        expires_at: Option<DateTime<Utc>>,
        spending_limit: Option<f64>,
        limit_unit: Option<String>,
        duration_days: Option<f64>,
        bound_credential_ids: Option<Vec<u64>>,
    ) -> anyhow::Result<ApiKey> {
        let mut keys = self.keys.write();
        let next_id = keys.iter().map(|k| k.id).max().unwrap_or(0) + 1;
        let api_key = ApiKey::new(
            next_id,
            name,
            expires_at,
            spending_limit,
            limit_unit.unwrap_or_else(default_limit_unit),
            duration_days,
            bound_credential_ids,
        );
        keys.push(api_key.clone());
        drop(keys);
        self.save()?;
        Ok(api_key)
    }

    /// 更新 key（name, enabled, expires_at, spending_limit, limit_unit, duration_days）
    #[allow(clippy::too_many_arguments)]
    pub fn update(
        &self,
        id: u32,
        name: Option<String>,
        enabled: Option<bool>,
        expires_at: Option<Option<DateTime<Utc>>>,
        spending_limit: Option<Option<f64>>,
        limit_unit: Option<String>,
        duration_days: Option<Option<f64>>,
        bound_credential_ids: Option<Option<Vec<u64>>>,
    ) -> anyhow::Result<Option<ApiKey>> {
        let mut keys = self.keys.write();
        let Some(api_key) = keys.iter_mut().find(|k| k.id == id) else {
            return Ok(None);
        };
        if let Some(name) = name {
            api_key.name = name;
        }
        if let Some(enabled) = enabled {
            api_key.enabled = enabled;
        }
        if let Some(expires_at) = expires_at {
            api_key.expires_at = expires_at;
        }
        if let Some(spending_limit) = spending_limit {
            api_key.spending_limit = spending_limit;
        }
        if let Some(limit_unit) = limit_unit {
            api_key.limit_unit = limit_unit;
        }
        if let Some(duration_days) = duration_days {
            match duration_days {
                Some(new_days) => {
                    if api_key.is_active() && api_key.expires_at.is_some() {
                        // 活跃 Key（有到期时间）：在当前到期时间上增量续期
                        let extension =
                            chrono::Duration::milliseconds((new_days * 86_400_000.0) as i64);
                        let new_expires = api_key.expires_at.unwrap() + extension;
                        api_key.expires_at = Some(new_expires);
                        // 重算 duration_days 为从激活到新到期的总天数
                        let total_ms =
                            (new_expires - api_key.activated_at.unwrap()).num_milliseconds();
                        api_key.duration_days = Some(total_ms as f64 / 86_400_000.0);
                    } else {
                        // 已过期或待激活：重置为待激活状态
                        api_key.duration_days = Some(new_days);
                        api_key.activated_at = None;
                        api_key.expires_at = None;
                    }
                }
                None => {
                    // 切换为"永不过期"模式
                    api_key.duration_days = None;
                    api_key.activated_at = None;
                }
            }
        }
        if let Some(ids) = bound_credential_ids {
            api_key.bound_credential_ids = ids;
        }
        let updated = api_key.clone();
        drop(keys);
        self.save()?;
        Ok(Some(updated))
    }

    /// 删除 key
    pub fn delete(&self, id: u32) -> anyhow::Result<bool> {
        let mut keys = self.keys.write();
        let len_before = keys.len();
        keys.retain(|k| k.id != id);
        let deleted = keys.len() < len_before;
        drop(keys);
        if deleted {
            self.save()?;
        }
        Ok(deleted)
    }

    /// 获取文件路径
    #[allow(dead_code)]
    pub fn file_path(&self) -> &Path {
        &self.file_path
    }

    /// 激活指定 key（幂等操作）
    /// 已激活或非懒激活模式的 key 直接跳过
    pub fn activate_key(&self, id: u32) -> anyhow::Result<()> {
        let mut keys = self.keys.write();
        let Some(api_key) = keys.iter_mut().find(|k| k.id == id) else {
            return Ok(());
        };
        if api_key.activate() {
            drop(keys);
            self.save()?;
        }
        Ok(())
    }

    /// 幂等确保存在一条固定值的内置 Key（无限额度、永不过期、立即激活）。
    ///
    /// 用于兼容客户端写死密钥的场景（如 mihaha 桌面端配置里的 `apiKey`）：
    /// 该 key 一旦存在（无论是否是本方法创建的）即跳过，不会重复插入或覆盖用户
    /// 在 Admin UI 里对同名 key 做的任何修改（启用状态、额度、绑定账号等）。
    pub fn ensure_fixed_key(&self, key: &str, name: &str) -> anyhow::Result<()> {
        {
            let keys = self.keys.read();
            if keys.iter().any(|k| k.key == key) {
                return Ok(());
            }
        }
        let mut keys = self.keys.write();
        // 双重检查：持锁期间可能已被其他调用插入
        if keys.iter().any(|k| k.key == key) {
            return Ok(());
        }
        let next_id = keys.iter().map(|k| k.id).max().unwrap_or(0) + 1;
        let mut api_key = ApiKey::new(
            next_id,
            name.to_string(),
            None,
            None,
            default_limit_unit(),
            None,
            None,
        );
        api_key.key = key.to_string();
        api_key.activated_at = Some(Utc::now());
        keys.push(api_key);
        drop(keys);
        self.save()
    }
    // APPEND_MARKER2
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 临时目录守卫：Drop 时自动清理，避免测试产物残留
    struct TempDirGuard(PathBuf);

    impl Drop for TempDirGuard {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn temp_manager() -> (TempDirGuard, ApiKeyManager) {
        let dir =
            std::env::temp_dir().join(format!("kiro2cc-api-key-test-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&dir).expect("创建临时目录失败");
        let path = dir.join("api_keys.json");
        let manager = ApiKeyManager::load(&path).expect("加载空 ApiKeyManager 失败");
        (TempDirGuard(dir), manager)
    }

    #[test]
    fn test_ensure_fixed_key_creates_active_unlimited_key() {
        let (_dir, manager) = temp_manager();
        manager
            .ensure_fixed_key("mihaha", "内置固定 Key")
            .expect("首次注册固定 Key 失败");

        let keys = manager.list();
        assert_eq!(keys.len(), 1);
        let key = &keys[0];
        assert_eq!(key.key, "mihaha");
        assert!(key.enabled);
        assert!(key.is_active());
        assert!(!key.is_expired());
        assert!(key.spending_limit.is_none());
        assert!(key.expires_at.is_none());

        match manager.authenticate("mihaha") {
            ApiKeyAuthResult::Valid { .. } => {}
            other => panic!("期望认证通过，实际: {:?}", std::mem::discriminant(&other)),
        }
    }

    #[test]
    fn test_ensure_fixed_key_is_idempotent() {
        let (_dir, manager) = temp_manager();
        manager
            .ensure_fixed_key("mihaha", "内置固定 Key")
            .expect("首次注册固定 Key 失败");
        manager
            .ensure_fixed_key("mihaha", "内置固定 Key")
            .expect("重复注册固定 Key 应为幂等操作");

        assert_eq!(manager.list().len(), 1);
    }

    #[test]
    fn test_ensure_fixed_key_preserves_user_edits() {
        let (_dir, manager) = temp_manager();
        manager
            .ensure_fixed_key("mihaha", "内置固定 Key")
            .expect("首次注册固定 Key 失败");

        let id = manager.list()[0].id;
        // 模拟用户在 Admin UI 里手动禁用/改名该 key
        manager
            .update(
                id,
                Some("用户改名".to_string()),
                Some(false),
                None,
                None,
                None,
                None,
                None,
            )
            .expect("更新固定 Key 失败");

        // 重新确保固定 Key 存在：不应覆盖用户已做的修改
        manager
            .ensure_fixed_key("mihaha", "内置固定 Key")
            .expect("重复注册固定 Key 应为幂等操作");

        let keys = manager.list();
        assert_eq!(keys.len(), 1);
        assert_eq!(keys[0].name, "用户改名");
        assert!(!keys[0].enabled);
    }
}
