// Copyright (c) 2026 Harllan He. Licensed under MIT.

//! 配置模型：config.json 反序列化结构（TlsBackend、缓存模拟、限流、超时等）与环境变量覆盖逻辑。
use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
#[derive(Default)]
pub enum TlsBackend {
    #[default]
    Rustls,
    NativeTls,
}

/// Prompt cache 模拟与指纹追踪配置
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CacheSimulationConfig {
    /// 是否启用指纹追踪（替代 from_ratio_config 末层兜底）
    #[serde(default = "default_fingerprint_enabled")]
    pub fingerprint_enabled: bool,

    /// 5m ephemeral TTL（秒）
    #[serde(default = "default_fingerprint_ttl_5m")]
    pub fingerprint_ttl_5m: u64,

    /// 1h ephemeral TTL（秒）
    #[serde(default = "default_fingerprint_ttl_1h")]
    pub fingerprint_ttl_1h: u64,

    /// 新建 cache_creation 中 1h tier 占比（0.0~1.0，默认 0.0 全部 5m）
    #[serde(default = "default_ephemeral_1h_ratio")]
    pub ephemeral_1h_ratio: f64,

    /// 单账号指纹断点上限（超出按 LRU 淘汰）
    #[serde(default = "default_fingerprint_max_breakpoints")]
    pub fingerprint_max_breakpoints_per_account: usize,
}

fn default_true() -> bool {
    true
}

fn default_fingerprint_enabled() -> bool {
    true
}
fn default_fingerprint_ttl_5m() -> u64 {
    5 * 60
}
fn default_fingerprint_ttl_1h() -> u64 {
    60 * 60
}
fn default_ephemeral_1h_ratio() -> f64 {
    0.0
}

const DEFAULT_FINGERPRINT_MAX_BREAKPOINTS: usize = 256;

fn default_fingerprint_max_breakpoints() -> usize {
    DEFAULT_FINGERPRINT_MAX_BREAKPOINTS
}

impl Default for CacheSimulationConfig {
    fn default() -> Self {
        Self {
            fingerprint_enabled: default_fingerprint_enabled(),
            fingerprint_ttl_5m: default_fingerprint_ttl_5m(),
            fingerprint_ttl_1h: default_fingerprint_ttl_1h(),
            ephemeral_1h_ratio: default_ephemeral_1h_ratio(),
            fingerprint_max_breakpoints_per_account: default_fingerprint_max_breakpoints(),
        }
    }
}

/// KNA 应用配置
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Config {
    #[serde(default = "default_host")]
    pub host: String,

    #[serde(default = "default_port")]
    pub port: u16,

    #[serde(default = "default_region")]
    pub region: String,

    /// Auth Region（用于 Token 刷新），未配置时回退到 region
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub auth_region: Option<String>,

    /// API Region（用于 API 请求），未配置时回退到 region
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_region: Option<String>,

    #[serde(default = "default_kiro_version")]
    pub kiro_version: String,

    #[serde(default)]
    pub machine_id: Option<String>,

    #[serde(default = "default_system_version")]
    pub system_version: String,

    #[serde(default = "default_node_version")]
    pub node_version: String,

    #[serde(default = "default_tls_backend")]
    pub tls_backend: TlsBackend,

    /// 额外信任的 CA 证书路径（PEM 格式），仅用于本进程出站 HTTPS 请求
    #[serde(default)]
    pub ca_cert_path: Option<String>,

    /// 外部 count_tokens API 地址（可选）
    #[serde(default)]
    pub count_tokens_api_url: Option<String>,

    /// count_tokens API 密钥（可选）
    #[serde(default)]
    pub count_tokens_api_key: Option<String>,

    /// count_tokens API 认证类型（可选，"x-api-key" 或 "bearer"，默认 "x-api-key"）
    #[serde(default = "default_count_tokens_auth_type")]
    pub count_tokens_auth_type: String,

    /// HTTP 代理地址（可选）
    /// 支持格式: http://host:port, https://host:port, socks5://host:port
    #[serde(default)]
    pub proxy_url: Option<String>,

    /// 代理认证用户名（可选）
    #[serde(default)]
    pub proxy_username: Option<String>,

    /// 代理认证密码（可选）
    #[serde(default)]
    pub proxy_password: Option<String>,

    /// Admin Password（可选，启用 Admin API 功能）
    #[serde(default, alias = "adminApiKey")]
    pub admin_psw: Option<String>,

    /// 负载均衡模式（"priority" 或 "balanced"）
    #[serde(default = "default_load_balancing_mode")]
    pub load_balancing_mode: String,

    /// 单账号每分钟最大请求数（超出时排队等待），0 表示不限制
    #[serde(default = "default_max_rpm_per_credential")]
    pub max_rpm_per_credential: u32,

    /// `/v1/models` 动态列表缓存 TTL（秒），默认 3600
    #[serde(default = "default_model_cache_ttl_secs")]
    pub model_cache_ttl_secs: u64,

    /// Prompt cache 模拟与指纹追踪配置
    #[serde(default)]
    pub cache_simulation: CacheSimulationConfig,

    /// 客户端 token 直通（true 时 usage 字段 1:1 上报真实值，不再按
    /// CLIENT_TOKEN_DISPLAY_SCALE 缩放）。默认 false 维持既有展示缩放。
    #[serde(default)]
    pub client_token_passthrough: bool,

    /// Suggestion Mode 输入建议请求放行开关
    ///
    /// Claude Code 每轮主对话后会自动发起输入建议请求（末条消息以 `[SUGGESTION MODE`
    /// 开头），携带全量上下文却只产出候选短语，转发上游按全价计费。
    /// 默认 false：代理拦截并返回空响应（不转发、不计费）；
    /// true：正常转发 Kiro 上游（保留客户端输入框建议短语功能）。
    #[serde(default)]
    pub forward_suggestion_mode: bool,

    /// 思考内容文本化展示（仅 Claude Code 客户端）
    ///
    /// Claude Code 默认隐藏/折叠 thinking，长推理期间界面像卡住。开启后，出站 SSE 的
    /// thinking 块被改写为 markdown 引用文本块，以 ANSI 变暗（灰色）样式逐行流式输出
    /// （类似 Kiro CLI 的展示；转义字符是否生效取决于客户端渲染）；
    /// 回传的历史里这些文本会被自动剥离，上游看不到。默认 true（文本化展示）。
    /// 副作用：客户端自己的上下文会包含这些文本，auto-compact 会更早触发。
    #[serde(default = "default_true")]
    pub thinking_as_text: bool,

    /// 分块写入约束的行数上限（issue #46）
    ///
    /// 请求带 Write / Edit / MultiEdit 工具时，代理在系统提示末尾追加“单次工具调用参数不超过
    /// N 行”的约束，避免上游对大块工具参数静默约 240s 后重置流。约束会让模型把大写入拆成
    /// 多轮，每轮重新思考，credits 随之上升。默认 300；调大更省 credits 但更容易撞 240s 重置；
    /// 0 表示不追加约束。可在 Admin 设置页热修改（仅影响之后的新请求）。
    #[serde(default = "default_write_chunking_max_lines")]
    pub write_chunking_max_lines: u32,

    /// 配置文件路径（运行时元数据，不写入 JSON）
    #[serde(skip)]
    config_path: Option<PathBuf>,
}

fn default_write_chunking_max_lines() -> u32 {
    crate::anthropic::DEFAULT_WRITE_CHUNKING_MAX_LINES
}

fn default_host() -> String {
    "127.0.0.1".to_string()
}

fn default_port() -> u16 {
    8080
}

fn default_region() -> String {
    "us-east-1".to_string()
}

fn default_kiro_version() -> String {
    "2.2.2".to_string()
}

// 0 = 关闭单账号 RPM 硬限（仅依赖上游 429 + throttle_delay 兜底）
fn default_max_rpm_per_credential() -> u32 {
    0
}

fn default_system_version() -> String {
    "darwin#24.6.0".to_string()
}

fn default_node_version() -> String {
    "22.21.1".to_string()
}

fn default_count_tokens_auth_type() -> String {
    "x-api-key".to_string()
}

fn default_tls_backend() -> TlsBackend {
    TlsBackend::Rustls
}

fn default_load_balancing_mode() -> String {
    "priority".to_string()
}

fn default_model_cache_ttl_secs() -> u64 {
    3600
}

impl Default for Config {
    fn default() -> Self {
        Self {
            host: default_host(),
            port: default_port(),
            region: default_region(),
            auth_region: None,
            api_region: None,
            kiro_version: default_kiro_version(),
            machine_id: None,
            system_version: default_system_version(),
            node_version: default_node_version(),
            tls_backend: default_tls_backend(),
            ca_cert_path: None,
            count_tokens_api_url: None,
            count_tokens_api_key: None,
            count_tokens_auth_type: default_count_tokens_auth_type(),
            proxy_url: None,
            proxy_username: None,
            proxy_password: None,
            admin_psw: None,
            load_balancing_mode: default_load_balancing_mode(),
            max_rpm_per_credential: default_max_rpm_per_credential(),
            model_cache_ttl_secs: default_model_cache_ttl_secs(),
            cache_simulation: CacheSimulationConfig::default(),
            client_token_passthrough: false,
            forward_suggestion_mode: false,
            thinking_as_text: true,
            write_chunking_max_lines: default_write_chunking_max_lines(),
            config_path: None,
        }
    }
}

impl Config {
    /// 获取默认配置文件路径
    pub fn default_config_path() -> &'static str {
        "config.json"
    }

    /// 获取有效的 Auth Region（用于 Token 刷新）
    /// 优先使用 auth_region，未配置时回退到 region
    pub fn effective_auth_region(&self) -> &str {
        self.auth_region.as_deref().unwrap_or(&self.region)
    }

    /// 获取有效的 API Region（用于 API 请求）
    /// 优先使用 api_region，未配置时回退到 region
    pub fn effective_api_region(&self) -> &str {
        self.api_region.as_deref().unwrap_or(&self.region)
    }

    /// 从文件加载配置
    pub fn load<P: AsRef<Path>>(path: P) -> anyhow::Result<Self> {
        let path = path.as_ref();
        if !path.exists() {
            return Ok(Self {
                config_path: Some(path.to_path_buf()),
                ..Self::default()
            });
        }

        let content = fs::read_to_string(path)?;
        let mut config: Config = serde_json::from_str(&content)?;
        config.config_path = Some(path.to_path_buf());
        Ok(config)
    }

    /// 获取配置文件路径（如果有）
    pub fn config_path(&self) -> Option<&Path> {
        self.config_path.as_deref()
    }

    /// 从环境变量覆盖配置项（用于容器化部署，如 Zeabur）
    ///
    /// 支持的环境变量:
    /// - `HOST`: 监听地址
    /// - `PORT`: 监听端口
    /// - `REGION`: AWS 区域
    /// - `AUTH_REGION`: Token 刷新区域
    /// - `API_REGION`: API 请求区域
    /// - `ADMIN_PSW`: Admin Password（未设置时回退读取 `ADMIN_API_KEY`）
    /// - `PROXY_URL`: HTTP 代理地址
    /// - `PROXY_USERNAME`: 代理用户名
    /// - `PROXY_PASSWORD`: 代理密码
    /// - `CA_CERT_PATH`: 额外信任的 CA 证书路径（PEM）
    /// - `LOAD_BALANCING_MODE`: 负载均衡模式
    /// - `MODEL_CACHE_TTL_SECS`: /v1/models 动态列表缓存 TTL（秒）
    /// - `CLIENT_TOKEN_PASSTHROUGH`: 客户端 token 直通开关
    /// - `FORWARD_SUGGESTION_MODE`: Suggestion Mode 输入建议请求放行开关
    pub fn apply_env_overrides(&mut self) {
        if let Ok(v) = env::var("HOST") {
            self.host = v;
        }
        if let Ok(v) = env::var("PORT")
            && let Ok(p) = v.parse::<u16>()
        {
            self.port = p;
        }
        if let Ok(v) = env::var("REGION") {
            self.region = v;
        }
        if let Ok(v) = env::var("AUTH_REGION") {
            self.auth_region = Some(v);
        }
        if let Ok(v) = env::var("API_REGION") {
            self.api_region = Some(v);
        }
        if let Ok(v) = env::var("ADMIN_PSW") {
            self.admin_psw = Some(v);
        } else if let Ok(v) = env::var("ADMIN_API_KEY") {
            self.admin_psw = Some(v);
        }
        if let Ok(v) = env::var("PROXY_URL") {
            self.proxy_url = Some(v);
        }
        if let Ok(v) = env::var("PROXY_USERNAME") {
            self.proxy_username = Some(v);
        }
        if let Ok(v) = env::var("PROXY_PASSWORD") {
            self.proxy_password = Some(v);
        }
        if let Ok(v) = env::var("CA_CERT_PATH") {
            self.ca_cert_path = Some(v);
        }
        if let Ok(v) = env::var("LOAD_BALANCING_MODE") {
            self.load_balancing_mode = v;
        }
        if let Ok(v) = env::var("MODEL_CACHE_TTL_SECS")
            && let Ok(n) = v.parse::<u64>()
        {
            self.model_cache_ttl_secs = n;
        }
        if let Ok(v) = env::var("CLIENT_TOKEN_PASSTHROUGH")
            && let Ok(b) = v.parse::<bool>()
        {
            self.client_token_passthrough = b;
        }
        if let Ok(v) = env::var("FORWARD_SUGGESTION_MODE")
            && let Ok(b) = v.parse::<bool>()
        {
            self.forward_suggestion_mode = b;
        }
        if let Ok(v) = env::var("THINKING_AS_TEXT")
            && let Ok(b) = v.parse::<bool>()
        {
            self.thinking_as_text = b;
        }
        if let Ok(v) = env::var("WRITE_CHUNKING_MAX_LINES")
            && let Ok(n) = v.parse::<u32>()
        {
            self.write_chunking_max_lines = n;
        }

        // CacheSimulationConfig 嵌套字段覆盖
        if let Ok(v) = env::var("CACHE_SIMULATION_FINGERPRINT_ENABLED")
            && let Ok(b) = v.parse::<bool>()
        {
            self.cache_simulation.fingerprint_enabled = b;
        }
        if let Ok(v) = env::var("CACHE_SIMULATION_FINGERPRINT_TTL_5M")
            && let Ok(n) = v.parse::<u64>()
        {
            self.cache_simulation.fingerprint_ttl_5m = n;
        }
        if let Ok(v) = env::var("CACHE_SIMULATION_FINGERPRINT_TTL_1H")
            && let Ok(n) = v.parse::<u64>()
        {
            self.cache_simulation.fingerprint_ttl_1h = n;
        }
        if let Ok(v) = env::var("CACHE_SIMULATION_EPHEMERAL_1H_RATIO")
            && let Ok(f) = v.parse::<f64>()
        {
            self.cache_simulation.ephemeral_1h_ratio = f.clamp(0.0, 1.0);
        }
        if let Ok(v) = env::var("CACHE_SIMULATION_FINGERPRINT_MAX_BREAKPOINTS")
            && let Ok(n) = v.parse::<usize>()
        {
            self.cache_simulation
                .fingerprint_max_breakpoints_per_account = n;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_cache_ttl_default() {
        let config = Config::default();
        assert_eq!(config.model_cache_ttl_secs, 3600);
    }

    #[test]
    fn test_model_cache_ttl_deserialize_default() {
        // 配置缺省该字段时应回退默认 3600
        let config: Config = serde_json::from_str("{}").unwrap();
        assert_eq!(config.model_cache_ttl_secs, 3600);
    }

    #[test]
    fn test_model_cache_ttl_deserialize_explicit() {
        let config: Config = serde_json::from_str(r#"{"modelCacheTtlSecs": 60}"#).unwrap();
        assert_eq!(config.model_cache_ttl_secs, 60);
    }

    #[test]
    fn test_client_token_passthrough_default_false() {
        // 缺省该字段时必须为 false（维持展示缩放，Issue #44 修复的默认零回归前提）
        let config: Config = serde_json::from_str("{}").unwrap();
        assert!(!config.client_token_passthrough);
    }

    #[test]
    fn test_write_chunking_max_lines_default_and_override() {
        assert_eq!(Config::default().write_chunking_max_lines, 300);
        let c: Config = serde_json::from_str("{}").unwrap();
        assert_eq!(c.write_chunking_max_lines, 300);
        let c: Config = serde_json::from_str(r#"{"writeChunkingMaxLines": 0}"#).unwrap();
        assert_eq!(c.write_chunking_max_lines, 0);
        let c: Config = serde_json::from_str(r#"{"writeChunkingMaxLines": 450}"#).unwrap();
        assert_eq!(c.write_chunking_max_lines, 450);
    }

    #[test]
    fn test_thinking_as_text_default_is_true() {
        let config = Config::default();
        assert!(config.thinking_as_text);
    }

    #[test]
    fn test_thinking_as_text_deserialize_default_is_true() {
        // 缺省该字段时必须为 true（新用户默认开启思考文本化展示）
        let config: Config = serde_json::from_str("{}").unwrap();
        assert!(config.thinking_as_text);
    }

    #[test]
    fn test_client_token_passthrough_deserialize_explicit() {
        let config: Config = serde_json::from_str(r#"{"clientTokenPassthrough": true}"#).unwrap();
        assert!(config.client_token_passthrough);
    }

    #[test]
    fn test_forward_suggestion_mode_default_false() {
        // 缺省该字段时必须为 false（拦截 Suggestion Mode，维持零计费默认行为）
        let config: Config = serde_json::from_str("{}").unwrap();
        assert!(!config.forward_suggestion_mode);
    }

    #[test]
    fn test_forward_suggestion_mode_deserialize_explicit() {
        let config: Config = serde_json::from_str(r#"{"forwardSuggestionMode": true}"#).unwrap();
        assert!(config.forward_suggestion_mode);
    }

    #[test]
    fn test_client_token_passthrough_env_override() {
        // 环境变量覆盖 config.json（容器化部署场景）。
        // 通过子进程注入环境变量验证：cargo test 默认并行运行其他测试，
        // 在测试进程内 set_var/remove_var（Rust 2024 unsafe）与系统库并发
        // 读取环境存在 UB 风险，子进程隔离可完全规避。
        const CHILD_MARKER: &str = "KIRO_CONFIG_ENV_TEST_CHILD";
        if env::var_os(CHILD_MARKER).is_some() {
            // 子进程分支：父进程已注入 CLIENT_TOKEN_PASSTHROUGH
            let mut config: Config = serde_json::from_str("{}").unwrap();
            config.apply_env_overrides();
            assert!(config.client_token_passthrough);
            return;
        }

        // 父进程分支：以子进程模式重启自身测试二进制
        let status = std::process::Command::new(env::current_exe().unwrap())
            .args([
                "--exact",
                "model::config::tests::test_client_token_passthrough_env_override",
                "--test-threads=1",
            ])
            .env(CHILD_MARKER, "1")
            .env("CLIENT_TOKEN_PASSTHROUGH", "true")
            .status()
            .unwrap();
        assert!(status.success());
    }

    #[test]
    fn test_forward_suggestion_mode_env_override() {
        // 环境变量覆盖 config.json（容器化部署场景），子进程隔离规避 set_var UB
        const CHILD_MARKER: &str = "KIRO_CONFIG_ENV_TEST_CHILD";
        if env::var_os(CHILD_MARKER).is_some() {
            let mut config: Config = serde_json::from_str("{}").unwrap();
            config.apply_env_overrides();
            assert!(config.forward_suggestion_mode);
            return;
        }

        let status = std::process::Command::new(env::current_exe().unwrap())
            .args([
                "--exact",
                "model::config::tests::test_forward_suggestion_mode_env_override",
                "--test-threads=1",
            ])
            .env(CHILD_MARKER, "1")
            .env("FORWARD_SUGGESTION_MODE", "true")
            .status()
            .unwrap();
        assert!(status.success());
    }
}
