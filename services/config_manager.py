"""
集中式配置管理器 — 管理可在网页端自定义的模型配置。

数据流：
  config.json (用户自定义) > .env (出厂默认)
  ↓
  ConfigManager ──→ Agent / EmbeddingService 统一读取
  ↓
  GET/POST /api/settings ──→ 前端设置页

安全性：
  - config.json 在 data/ 目录下，已被 .gitignore 排除，不会提交到 GitHub
  - 前端 GET 时 API Key 自动掩码（仅显示首 4 + 尾 4 位）
"""
from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# config.json 文件路径
_CONFIG_DIR = Path(__file__).resolve().parent.parent / "data"
_CONFIG_FILE = _CONFIG_DIR / "config.json"
_CONFIG_LOCK = threading.RLock()
MODEL_PROVIDERS = {
    "GLM": "https://open.bigmodel.cn/api/paas/v4/",
    "DeepSeek": "https://api.deepseek.com/v1/",
    "Kimi": "https://api.moonshot.cn/v1/",
    "Gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "自定义": "",
}
_PROVIDER_KEYS = {"GLM": ("GLM_API_KEY", "ZHIPU_API_KEY"), "DeepSeek": ("DEEPSEEK_API_KEY",),
                  "Kimi": ("KIMI_API_KEY", "MOONSHOT_API_KEY"), "Gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY")}

def infer_provider(base_url: str) -> str:
    return next((name for name, url in MODEL_PROVIDERS.items()
                 if url and base_url.rstrip("/") == url.rstrip("/")), "自定义")

def validate_api_url(value: str) -> str:
    url = urlparse(value)
    if url.scheme not in ("http", "https") or not url.netloc or url.username or url.password or url.query or url.fragment:
        raise ValueError("必须是有效的 HTTP(S) API 地址，不能含凭证、查询参数或片段")
    return value.rstrip("/") + "/"

def _read_config_file() -> dict[str, Any]:
    """从 config.json 读取用户自定义配置，不存在则返回空字典。"""
    if not _CONFIG_FILE.exists():
        return {}
    try:
        with open(_CONFIG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or any(not isinstance(data.get(k, {}), dict) for k in ("llm", "embedding", "qdrant", "policy")):
            raise ValueError("配置必须是对象，llm / embedding / qdrant / policy 必须是对象")
        return data
    except (json.JSONDecodeError, OSError, ValueError) as e:
        logger.warning(f"[ConfigManager] config.json 读取失败，回退到 .env 默认值: {e}")
        return {}

def _write_config_file(data: dict[str, Any]) -> None:
    """将配置写入 config.json。"""
    _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    temporary = _CONFIG_FILE.with_suffix(".json.tmp")
    with open(temporary, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(temporary, _CONFIG_FILE)

def _mask_key(key: str) -> str:
    """遮蔽 API Key：仅显示前 4 位和后 4 位，中间替换为 ****。"""
    if not key or len(key) <= 8:
        return "****"
    return key[:4] + "****" + key[-4:]

class ConfigManager:
    """集中式配置管理（单例模式，所有 Agent 共享）。"""

    _instance: ConfigManager | None = None

    def __new__(cls) -> ConfigManager:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    # ============================================================
    # LLM 配置
    # ============================================================
    def get_llm_config(self) -> dict[str, str]:
        """
        返回 LLM 配置字典: {model, api_key, base_url}.
        优先 config.json，不存在则回退 .env 环境变量。
        """
        cfg = _read_config_file()
        llm = cfg.get("llm", {})
        return {
            "model": llm.get("model") or os.getenv("LLM_MODEL_ID", ""),
            "api_key": llm.get("api_key") or os.getenv("LLM_API_KEY", ""),
            "base_url": llm.get("base_url") or os.getenv("LLM_BASE_URL", ""),
        }

    # ============================================================
    # Embedding 配置
    # ============================================================
    def get_embedding_config(self) -> dict[str, str]:
        """
        返回 Embedding 配置字典: {model, api_key, base_url}.
        """
        cfg = _read_config_file()
        emb = cfg.get("embedding", {})
        return {
            "model": emb.get("model") or os.getenv("EMBEDDING_MODEL_ID", ""),
            "api_key": emb.get("api_key") or os.getenv("EMBEDDING_API_KEY", ""),
            "base_url": emb.get("base_url") or os.getenv("EMBEDDING_BASE_URL", ""),
        }

    def resolve_model_key(self, section: str, base_url: str, api_key: str = "") -> str:
        """只复用相同端点的密钥；切换提供商时不向新网站发送旧密钥。"""
        if section not in ("llm", "embedding"):
            raise ValueError("配置区域必须为 llm / embedding")
        if not isinstance(api_key, str):
            raise ValueError("API Key 必须是字符串")
        if api_key and "****" not in api_key:
            return api_key.strip()
        current = self.get_llm_config() if section == "llm" else self.get_embedding_config()
        if base_url.rstrip("/") == current["base_url"].rstrip("/"):
            return current["api_key"]
        prefix = "LLM" if section == "llm" else "EMBEDDING"
        if base_url.rstrip("/") == os.getenv(prefix + "_BASE_URL", "").rstrip("/"):
            return os.getenv(prefix + "_API_KEY", "")
        provider = infer_provider(base_url)
        return next((os.getenv(name, "") for name in _PROVIDER_KEYS.get(provider, ()) if os.getenv(name)), "")

    def get_qdrant_config(self) -> dict:
        cfg = _read_config_file().get("qdrant", {})
        return {"url": cfg.get("url") or os.getenv("QDRANT_CLUSTER_URL", ""),
                "api_key": cfg.get("api_key", os.getenv("QDRANT_API_KEY", "")),
                "vector_limit": cfg.get("vector_limit", 0)}

    def get_policy_config(self) -> dict:
        return _read_config_file().get("policy", {})

    def save_policy_config(self, data: dict) -> None:
        with _CONFIG_LOCK:
            existing = _read_config_file()
            existing.setdefault("policy", {}).update(data)
            _write_config_file(existing)

    # ============================================================
    # 全量读取（供前端 API 使用）
    # ============================================================
    def get_all_config(self) -> dict[str, Any]:
        """
        返回前端展示用的完整配置，API Key 已掩码。
        如果某个字段未配置，用 .env 的值填充（仅用于提示用户"当前默认值"）。
        """
        cfg = _read_config_file()

        llm_raw = cfg.get("llm", {})
        emb_raw = cfg.get("embedding", {})

        # 未自定义的字段用 .env 默认值填充
        llm_model = llm_raw.get("model") or os.getenv("LLM_MODEL_ID", "")
        llm_key = llm_raw.get("api_key") or os.getenv("LLM_API_KEY", "")
        llm_url = llm_raw.get("base_url") or os.getenv("LLM_BASE_URL", "")
        emb_model = emb_raw.get("model") or os.getenv("EMBEDDING_MODEL_ID", "")
        emb_key = emb_raw.get("api_key") or os.getenv("EMBEDDING_API_KEY", "")
        emb_url = emb_raw.get("base_url") or os.getenv("EMBEDDING_BASE_URL", "")

        return {
            "llm": {
                "model": llm_model,
                "api_key": _mask_key(llm_key),
                "api_key_set": bool(llm_key),  # 前端据此判断是否已配置
                "base_url": llm_url,
                "provider": llm_raw.get("provider") or infer_provider(llm_url),
            },
            "embedding": {
                "model": emb_model,
                "api_key": _mask_key(emb_key),
                "api_key_set": bool(emb_key),
                "base_url": emb_url,
                "provider": emb_raw.get("provider") or infer_provider(emb_url),
            },
            "qdrant": {**self.get_qdrant_config(), "api_key": _mask_key(self.get_qdrant_config()["api_key"]),
                       "api_key_set": bool(self.get_qdrant_config()["api_key"])},
        }

    # ============================================================
    # 保存配置
    # ============================================================
    def save_config(self, data: dict[str, Any]) -> None:
        """
        接收前端提交的配置，写入 config.json。

        规则：
        - llm/embedding 区块整体覆盖
        - api_key 为 "****" 或空字符串时，不覆盖已有 key（保留旧值）
        """
        with _CONFIG_LOCK:
            self._save_config_locked(data)

    def _save_config_locked(self, data: dict[str, Any], persist: bool = True) -> dict:
        existing = _read_config_file()

        existing.setdefault("llm", {})
        existing.setdefault("embedding", {})

        for section in ("llm", "embedding"):
            incoming = data.get(section)
            if not isinstance(incoming, dict):
                continue
            old = self.get_llm_config() if section == "llm" else self.get_embedding_config()
            new_url = incoming.get("base_url") or old["base_url"]
            if not isinstance(new_url, str):
                raise ValueError(f"{section}.base_url 必须是字符串")
            if new_url.strip().rstrip("/") != old["base_url"].rstrip("/"):
                if not isinstance(incoming.get("model"), str) or not incoming["model"].strip():
                    raise ValueError(f"切换 {section} 提供商时请填写模型 ID")
                key = self.resolve_model_key(section, new_url.strip(), incoming.get("api_key") or "")
                if not key:
                    raise ValueError(f"切换 {section} 提供商时请填写对应 API Key")
                existing[section]["api_key"] = key
            for field in ("model", "api_key", "base_url", "provider"):
                val = incoming.get(field)
                if val is None or val == "":
                    continue
                if not isinstance(val, str):
                    raise ValueError(f"{section}.{field} 必须是字符串")
                val = val.strip()
                if not val:
                    continue
                if field == "api_key" and "****" in val:
                    continue  # 用户没改，保留旧值
                if field == "provider" and val not in MODEL_PROVIDERS:
                    raise ValueError("不支持的提供商")
                if field == "base_url":
                    validate_api_url(val)
                existing[section][field] = val

        if isinstance(data.get("qdrant"), dict):
            incoming = data["qdrant"]
            old = self.get_qdrant_config()
            url = incoming.get("url") or old["url"]
            if not isinstance(url, str):
                raise ValueError("qdrant.url 必须是字符串")
            validate_api_url(url.strip())
            key = incoming.get("api_key")
            if key is not None and not isinstance(key, str):
                raise ValueError("qdrant.api_key 必须是字符串")
            changed = url.rstrip("/") != old["url"].rstrip("/")
            existing["qdrant"] = {**old, "url": url.strip(),
                "api_key": key.strip() if key and "****" not in key else "" if changed else old["api_key"]}
            limit = incoming.get("vector_limit", old["vector_limit"])
            if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
                raise ValueError("向量预警上限必须为非负整数；0 表示不设数量上限")
            existing["qdrant"]["vector_limit"] = limit

        if persist:
            _write_config_file(existing)
            logger.info("[ConfigManager] 配置已保存到 %s", _CONFIG_FILE)
        return existing
