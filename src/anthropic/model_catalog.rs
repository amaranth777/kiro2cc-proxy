//! Runtime discovery of the models exposed by the official Kiro CLI.
//!
//! The CLI is the authoritative source for currently available Kiro models.
//! Discovery is deliberately isolated from request handling and cached so a
//! slow or unavailable CLI cannot block every `/v1/models` request.

use serde::{Deserialize, Serialize};
use std::process::{Command, Stdio};
use std::sync::{Mutex, OnceLock};
use std::thread;
use std::time::{Duration, Instant};

const DEFAULT_TTL: Duration = Duration::from_secs(300);
const CLI_TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct UpstreamModel {
    pub id: String,
    pub context_length: i32,
    pub description: String,
    pub rate_multiplier: Option<f64>,
    pub thinking: bool,
}

#[derive(Debug, Deserialize)]
struct KiroModelList {
    models: Vec<KiroModel>,
}

#[derive(Debug, Deserialize)]
struct KiroModel {
    model_id: String,
    #[serde(default)]
    description: String,
    #[serde(default)]
    context_window_tokens: Option<i32>,
    #[serde(default)]
    rate_multiplier: Option<f64>,
}

#[derive(Debug)]
struct CacheEntry {
    loaded_at: Instant,
    models: Option<Vec<UpstreamModel>>,
}

static CACHE: OnceLock<Mutex<CacheEntry>> = OnceLock::new();

fn cache() -> &'static Mutex<CacheEntry> {
    CACHE.get_or_init(|| {
        Mutex::new(CacheEntry {
            loaded_at: Instant::now() - DEFAULT_TTL,
            models: None,
        })
    })
}

/// Discover the current upstream model list, using a short-lived process cache.
pub fn discover_models() -> Option<Vec<UpstreamModel>> {
    discover_models_with_ttl(DEFAULT_TTL)
}

pub(crate) fn metadata_for_id(id: &str) -> Option<UpstreamModel> {
    discover_models()?.into_iter().find(|model| model.id == id)
}

/// Return the best available context window for a model.
///
/// Discovered values win. The small fallback table is only used when the CLI
/// is unavailable, preserving compatibility with existing callers.
pub(crate) fn context_length_for_model(id: &str) -> i32 {
    discover_models()
        .and_then(|models| models.into_iter().find(|model| model.id == id))
        .map(|model| model.context_length)
        .or_else(|| fallback_context_length(id))
        .unwrap_or(750_000)
}

fn discover_models_with_ttl(ttl: Duration) -> Option<Vec<UpstreamModel>> {
    {
        let entry = cache().lock().ok()?;
        if entry.loaded_at.elapsed() < ttl {
            return entry.models.clone();
        }
    }

    // Do not hold the process-wide cache lock while starting an external CLI.
    let models = run_discovery_command().and_then(|stdout| parse_models(&stdout).ok());

    if let Ok(mut entry) = cache().lock() {
        entry.loaded_at = Instant::now();
        entry.models = models.clone();
    }
    models
}

fn run_discovery_command() -> Option<Vec<u8>> {
    let mut child = Command::new("kiro-cli")
        .args(["chat", "--list-models", "--format", "json"])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;

    let started = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) if status.success() => {
                return child.wait_with_output().ok().map(|output| output.stdout);
            }
            Ok(Some(_)) => return None,
            Ok(None) if started.elapsed() < CLI_TIMEOUT => thread::sleep(Duration::from_millis(25)),
            Ok(None) => {
                let _ = child.kill();
                let _ = child.wait();
                return None;
            }
            Err(_) => {
                let _ = child.kill();
                let _ = child.wait();
                return None;
            }
        }
    }
}

fn parse_models(bytes: &[u8]) -> Result<Vec<UpstreamModel>, serde_json::Error> {
    let payload: KiroModelList = serde_json::from_slice(bytes)?;
    Ok(payload
        .models
        .into_iter()
        .filter_map(|model| {
            let id = model.model_id.trim().to_string();
            if id.is_empty() {
                return None;
            }
            let context_length = model
                .context_window_tokens
                .filter(|value| *value > 0)
                .or_else(|| fallback_context_length(&id))?;
            Some(UpstreamModel {
                id,
                context_length,
                description: model.description,
                rate_multiplier: model.rate_multiplier,
                // The current upstream manifest does not expose thinking
                // capability. Never infer it for discovered models.
                thinking: false,
            })
        })
        .collect())
}

fn fallback_context_length(id: &str) -> Option<i32> {
    match id {
        "gpt-5.6-sol" | "gpt-5.6-terra" | "gpt-5.6-luna" => Some(272_000),
        "deepseek-3.2" => Some(164_000),
        "minimax-m2.5" | "minimax-m2.1" => Some(196_000),
        "glm-5" => Some(200_000),
        "qwen3-coder-next" => Some(256_000),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_upstream_models_and_defaults_thinking_to_false() {
        let models = parse_models(
            br#"{"models":[{"model_id":"glm-5","description":"GLM","context_window_tokens":200000,"rate_multiplier":0.5}]}"#,
        )
        .unwrap();
        assert_eq!(models[0].id, "glm-5");
        assert_eq!(models[0].context_length, 200000);
        assert!(!models[0].thinking);
    }

    #[test]
    fn uses_known_context_fallback_when_upstream_omits_it() {
        let models = parse_models(br#"{"models":[{"model_id":"qwen3-coder-next"}]}"#).unwrap();
        assert_eq!(models[0].context_length, 256000);
    }

    #[test]
    fn rejects_invalid_json() {
        assert!(parse_models(b"not-json").is_err());
    }

    #[test]
    fn serializes_metadata_fields() {
        let model = parse_models(br#"{"models":[{"model_id":"deepseek-3.2"}]}"#).unwrap();
        let value = serde_json::to_value(&model[0]).unwrap();
        assert_eq!(value["id"], "deepseek-3.2");
        assert_eq!(value["context_length"], 164000);
        assert_eq!(value["thinking"], false);
    }
}
