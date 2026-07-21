#!/usr/bin/env python3
"""切换 kiro2cc 候选版本，失败自动恢复旧二进制。

默认只做检查；真正切换必须显式传 --run。
脚本不把 API key 放进命令行、日志或输出，只从运行配置读取。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

SERVICE = "kiro2cc-proxy.service"
REPO = Path(__file__).resolve().parents[1]
PRODUCTION_BINARY = REPO / "target/release/kiro2cc-proxy"
CONFIG = REPO / "app/config/config.json"
LOG_DIR = Path.home() / ".hermes/logs"
BACKUP_ROOT = Path.home() / ".hermes/backups/kiro2cc-release-heartbeat"
DEFAULT_MODEL = "claude-sonnet-4-6"
PORT = 8787

_SECRET_RE = re.compile(r"(?i)(?:sk-|bearer\s+)[A-Za-z0-9._~+/=-]+")

def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def redact(text: str) -> str:
    return _SECRET_RE.sub("[REDACTED]", text)


def run(command: list[str], *, check: bool = True, timeout: int = 60) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, capture_output=True, check=check, timeout=timeout)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_candidate(candidate: Path) -> None:
    if not candidate.is_file() or not os.access(candidate, os.X_OK):
        raise RuntimeError(f"候选二进制不可执行: {candidate}")
    if not CONFIG.is_file():
        raise RuntimeError(f"生产配置不存在: {CONFIG}")
    config = json.loads(CONFIG.read_text())
    if config.get("port") != PORT:
        raise RuntimeError(f"生产配置端口不是 {PORT}: {config.get('port')!r}")


def service_is_active() -> bool:
    return run(["systemctl", "--user", "is-active", SERVICE], check=False).stdout.strip() == "active"


def wait_ready(timeout: int = 30) -> None:
    key = load_api_key()
    headers = {
        "x-api-key": key,
        "Authorization": f"Bearer {key}",
    }
    request = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/models",
        headers=headers,
    )
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if service_is_active():
            try:
                with urllib.request.urlopen(request, timeout=3) as response:
                    if response.status == 200:
                        return
            except Exception:
                pass
        time.sleep(1)
    raise RuntimeError("kiro2cc 服务未在限定时间内恢复并通过认证 /v1/models")


def load_api_key() -> str:
    value = json.loads(CONFIG.read_text()).get("apiKey")
    if not isinstance(value, str) or not value:
        raise RuntimeError("生产配置没有有效 apiKey")
    return value


def smoke_test(model: str) -> dict[str, object]:
    key = load_api_key()
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        os.environ.pop(name, None)
    os.environ["NO_PROXY"] = "*"
    headers = {
        "x-api-key": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
    }

    def get(path: str) -> tuple[int, bytes]:
        request = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read()

    def post(path: str, payload: dict[str, object]) -> tuple[int, bytes]:
        request = urllib.request.Request(
            f"http://127.0.0.1:{PORT}{path}",
            data=json.dumps(payload).encode(),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read()

    models_status, models_body = get("/v1/models")
    if models_status != 200:
        raise RuntimeError(f"/v1/models 返回 HTTP {models_status}: {redact(models_body.decode(errors='replace'))[:500]}")
    models = json.loads(models_body)
    ids = {item.get("id") for item in models.get("data", []) if isinstance(item, dict)}
    if model not in ids:
        raise RuntimeError(f"测试模型不在 /v1/models: {model}")

    status, body = post(
        "/v1/messages",
        {
            "model": model,
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Reply with exactly: kiro2cc heartbeat ok"}],
        },
    )
    if status != 200:
        raise RuntimeError(f"测试消息返回 HTTP {status}: {redact(body.decode(errors='replace'))[:800]}")
    response = json.loads(body)
    content = response.get("content")
    if not isinstance(content, list) or not content:
        raise RuntimeError("测试消息返回 200，但 content 为空")
    text = "".join(
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    )
    if "kiro2cc heartbeat ok" not in text.lower():
        raise RuntimeError(f"测试消息内容异常: {redact(text)[:500]}")
    return {
        "models": len(ids),
        "message_status": status,
        "content_blocks": len(content),
        "heartbeat_text_verified": True,
    }


def journal_since(iso_time: str) -> str:
    # journalctl 不接受带冒号时区偏移的 ISO 字符串（例如 +00:00）。
    # 转成本机时间的无时区格式，避免日志采集异常掩盖真实回滚结果。
    parsed = datetime.fromisoformat(iso_time)
    since = parsed.astimezone().strftime("%Y-%m-%d %H:%M:%S")
    result = run(["journalctl", "--user", "-u", SERVICE, "--since", since, "--no-pager"], check=False, timeout=30)
    return redact(result.stdout + result.stderr)


def write_failure_log(tag: str, content: str) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOG_DIR / f"kiro2cc-release-heartbeat-{tag}.log"
    path.write_text(content)
    return path


def atomic_install(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=f".{destination.name}.", delete=False) as tmp:
        temp_path = Path(tmp.name)
        with source.open("rb") as src:
            shutil.copyfileobj(src, tmp)
        tmp.flush()
        os.fsync(tmp.fileno())
    shutil.copymode(source, temp_path)
    os.replace(temp_path, destination)


def restart_and_test(model: str) -> dict[str, object]:
    run(["systemctl", "--user", "restart", SERVICE], timeout=60)
    wait_ready()
    return smoke_test(model)


def deploy(candidate: Path, model: str) -> int:
    validate_candidate(candidate)
    if not PRODUCTION_BINARY.is_file():
        raise RuntimeError(f"生产二进制不存在: {PRODUCTION_BINARY}")
    if not service_is_active():
        raise RuntimeError("生产服务当前不是 active，拒绝自动切换")

    tag = now_tag()
    backup_dir = BACKUP_ROOT / tag
    backup_dir.mkdir(parents=True, exist_ok=False)
    old_hash = sha256(PRODUCTION_BINARY)
    candidate_hash = sha256(candidate)
    old_backup = backup_dir / PRODUCTION_BINARY.name
    shutil.copy2(PRODUCTION_BINARY, old_backup)
    manifest = {
        "service": SERVICE,
        "production_binary": str(PRODUCTION_BINARY),
        "candidate_binary": str(candidate),
        "old_sha256": old_hash,
        "candidate_sha256": candidate_hash,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (backup_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    started = datetime.now(timezone.utc).isoformat()

    try:
        atomic_install(candidate, PRODUCTION_BINARY)
        result = restart_and_test(model)
        print(json.dumps({"status": "promoted", "backup": str(backup_dir), **result}, ensure_ascii=False))
        return 0
    except Exception as exc:
        failure = journal_since(started)
        failure_path = write_failure_log(tag, f"candidate_failure={redact(str(exc))}\n\n{failure}")
        try:
            atomic_install(old_backup, PRODUCTION_BINARY)
            rollback_result = restart_and_test(model)
            print(json.dumps({
                "status": "rolled_back",
                "reason": redact(str(exc)),
                "candidate_log": str(failure_path),
                "backup": str(backup_dir),
                "rollback_smoke_test": rollback_result,
            }, ensure_ascii=False), file=sys.stderr)
            return 2
        except Exception as rollback_exc:
            rollback_log = write_failure_log(f"{tag}-rollback", journal_since(started))
            print(json.dumps({
                "status": "rollback_failed",
                "reason": redact(str(exc)),
                "rollback_reason": redact(str(rollback_exc)),
                "candidate_log": str(failure_path),
                "rollback_log": str(rollback_log),
                "backup": str(backup_dir),
            }, ensure_ascii=False), file=sys.stderr)
            return 3


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True, help="已构建并验证的候选二进制")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--run", action="store_true", help="执行切换；不传时只做 dry-run")
    args = parser.parse_args()
    candidate = args.candidate.resolve()
    validate_candidate(candidate)
    info = {"mode": "run" if args.run else "dry-run", "candidate": str(candidate), "candidate_sha256": sha256(candidate)}
    if not args.run:
        info["production"] = str(PRODUCTION_BINARY)
        info["note"] = "未修改服务、二进制或配置"
        print(json.dumps(info, ensure_ascii=False))
        return 0
    return deploy(candidate, args.model)


if __name__ == "__main__":
    raise SystemExit(main())
