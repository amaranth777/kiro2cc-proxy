// Copyright (c) 2026 Harllan He. Licensed under MIT.
//! HTTP Client 构建模块
//!
//! 提供统一的 HTTP Client 构建功能，支持代理配置

use reqwest::{Certificate, Client, Proxy};
use std::fs;
use std::time::Duration;

use crate::model::config::TlsBackend;

/// 代理配置
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct ProxyConfig {
    /// 代理地址，支持 http/https/socks5
    pub url: String,
    /// 代理认证用户名
    pub username: Option<String>,
    /// 代理认证密码
    pub password: Option<String>,
}

impl ProxyConfig {
    /// 从 url 创建代理配置
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            username: None,
            password: None,
        }
    }

    /// 设置认证信息
    pub fn with_auth(mut self, username: impl Into<String>, password: impl Into<String>) -> Self {
        self.username = Some(username.into());
        self.password = Some(password.into());
        self
    }
}

/// 构建 HTTP Client
///
/// # Arguments
/// * `proxy` - 可选的代理配置
/// * `timeout_secs` - 超时时间（秒）
///
/// # Returns
/// 配置好的 reqwest::Client
pub fn build_client(
    proxy: Option<&ProxyConfig>,
    timeout_secs: u64,
    tls_backend: TlsBackend,
    ca_cert_path: Option<&str>,
) -> anyhow::Result<Client> {
    let mut builder = Client::builder()
        .timeout(Duration::from_secs(timeout_secs))
        .pool_max_idle_per_host(20)
        .pool_idle_timeout(Duration::from_secs(90))
        .tcp_keepalive(Duration::from_secs(30));

    // Linux/macOS 使用 reqwest 默认的 webpki-roots。
    builder = builder.use_rustls_tls();
    let _ = tls_backend; // 保留参数以兼容配置结构

    #[cfg(windows)]
    {
        let native_roots = rustls_native_certs::load_native_certs();
        for error in native_roots.errors {
            tracing::warn!("加载 Windows 系统根证书失败: {error}");
        }
        for cert in native_roots.certs {
            builder = builder.add_root_certificate(Certificate::from_der(cert.as_ref())?);
        }
        tracing::info!("HTTP Client 已加载 Windows 系统根证书");
    }

    if let Some(path) = ca_cert_path.filter(|path| !path.trim().is_empty()) {
        let pem = fs::read(path)?;
        let cert = Certificate::from_pem(&pem)?;
        builder = builder.add_root_certificate(cert);
        tracing::info!("HTTP Client 已加载额外 CA 证书: {}", path);
    }

    if let Some(proxy_config) = proxy {
        let mut proxy = Proxy::all(&proxy_config.url)?;

        // 设置代理认证
        if let (Some(username), Some(password)) = (&proxy_config.username, &proxy_config.password) {
            proxy = proxy.basic_auth(username, password);
        }

        builder = builder.proxy(proxy);
        tracing::debug!("HTTP Client 使用代理: {}", proxy_config.url);
    }

    Ok(builder.build()?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_proxy_config_new() {
        let config = ProxyConfig::new("http://127.0.0.1:7890");
        assert_eq!(config.url, "http://127.0.0.1:7890");
        assert!(config.username.is_none());
        assert!(config.password.is_none());
    }

    #[test]
    fn test_proxy_config_with_auth() {
        let config = ProxyConfig::new("socks5://127.0.0.1:1080").with_auth("user", "pass");
        assert_eq!(config.url, "socks5://127.0.0.1:1080");
        assert_eq!(config.username, Some("user".to_string()));
        assert_eq!(config.password, Some("pass".to_string()));
    }

    #[test]
    fn test_build_client_without_proxy() {
        let client = build_client(None, 30, TlsBackend::Rustls, None);
        assert!(client.is_ok());
    }

    #[test]
    fn test_build_client_with_proxy() {
        let config = ProxyConfig::new("http://127.0.0.1:7890");
        let client = build_client(Some(&config), 30, TlsBackend::Rustls, None);
        assert!(client.is_ok());
    }
}
