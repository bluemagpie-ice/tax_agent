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
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# config.json 文件路径
_CONFIG_DIR = Path(__file__).resolve().parent.parent / "data"
_CONFIG_FILE = _CONFIG_DIR / "config.json"

def _read_config_file() -> dict[str, Any]:
    """从 config.json 读取用户自定义配置，不存在则返回空字典。"""
    if not _CONFIG_FILE.exists():
        return {}
    try:
        with open(_CONFIG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or any(not isinstance(data.get(k, {}), dict) for k in ("llm", "embedding")):
            raise ValueError("配置必须是对象，llm / embedding 必须是对象")
        return data
    except (json.JSONDecodeError, OSError, ValueError) as e:
        logger.warning(f"[ConfigManager] config.json 读取失败，回退到 .env 默认值: {e}")
        return {}

def _write_config_file(data: dict[str, Any]) -> None:
    """将配置写入 config.json。"""
    _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(_CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

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
            },
            "embedding": {
                "model": emb_model,
                "api_key": _mask_key(emb_key),
                "api_key_set": bool(emb_key),
                "base_url": emb_url,
            },
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
        existing = _read_config_file()

        existing.setdefault("llm", {})
        existing.setdefault("embedding", {})

        for section in ("llm", "embedding"):
            incoming = data.get(section)
            if not isinstance(incoming, dict):
                continue
            for field in ("model", "api_key", "base_url"):
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
                if field == "base_url":
                    url = urlparse(val)
                    if url.scheme not in ("http", "https") or not url.netloc or url.username or url.password:
                        raise ValueError(f"{section}.base_url 必须是有效的 HTTP(S) API 地址，不能含凭证")
                existing[section][field] = val

        _write_config_file(existing)
        logger.info("[ConfigManager] 配置已保存到 %s", _CONFIG_FILE)
