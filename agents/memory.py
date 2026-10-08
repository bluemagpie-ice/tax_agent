"""
记忆模块

实现三层记忆：
    1. 工作记忆（短期）：最近 N 轮对话，直接拼接到 prompt。
    2. 会话摘要（中期）：历史对话压缩后形成的可检索摘要。
    3. 用户画像（长期）：跨会话沉淀的税务偏好、关注税种、常用系统等。

存储：基于 SQLite（多租户物理隔离）+ JSON 落地（历史兼容）。

# todo：在compress方法中的滚动窗口的更好的实现。对于摘要截取部分是否存在更好的实现？

"""

from __future__ import annotations

import json
import logging
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# 不再硬依赖 config.PATHS，改为稳健的"项目根目录推导"：
# 本文件路径为 <project>/agents/memory.py，因此项目根为其 parent.parent。
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

from services.record_store import SQLiteRecordStore
from services.security import current_tenant_id, record_visible

class ConversationMemory:
    """
    会话记忆管理器（租户隔离、乐观锁、自动摘要压缩）。

    对外主要接口：
        - context_for(session_id, message)     获取供 LLM 使用的上下文包
        - record_turn(session_id, user, ast)  记录一轮对话并触发压缩
        - get_session(session_id)             读取会话
        - list_sessions / delete_session      会话管理
        - stats()                             记忆健康度概况
    """

    # 税务场景默认值（可在实例化时覆盖）
    DEFAULT_COMPRESS_AT = 18
    DEFAULT_RETAIN_RECENT = 8
    DEFAULT_CONTEXT_BUDGET = 2400  # tokens

    # 跨会话用户画像的存储命名空间（按租户隔离）
    USER_PROFILE_NAMESPACE = "user_profile"

    def __init__(
        self,
        base_dir: Optional[Path] = None,
        compress_at: int = DEFAULT_COMPRESS_AT,
        retain_recent: int = DEFAULT_RETAIN_RECENT,
        context_budget_tokens: int = DEFAULT_CONTEXT_BUDGET,
        title_llm=None,
    ) -> None:
        self.base_dir = Path(base_dir) if base_dir else _PROJECT_ROOT / "data" / "conversation_memory"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.compress_at = max(4, int(compress_at))
        self.retain_recent = max(2, int(retain_recent))
        self.context_budget_tokens = max(600, int(context_budget_tokens))
        self._lock = threading.RLock()
        self.store = SQLiteRecordStore(self.base_dir / ".records.sqlite3")
        self._title_llm = title_llm  # 可选：用于对话标题概括的 LLM

    # ------------------------------------------------------------------ 对外主入口
    def context_for(self, session_id: str, message: str) -> Dict[str, Any]:
        """组装供 LLM 使用的完整上下文包（含历史、画像、相关片段、最近对话）。"""
        session = self.get_session(session_id) or self._empty(session_id)
        related = self._related(session.get("messages", []), message)

        profile = self._load_user_profile() or session.get("profile", {})
        prompt_parts: List[str] = []
        if session.get("summary"):
            prompt_parts.append(f"[会话摘要]\n{session['summary']}")
        if profile:
            prompt_parts.append(f"[用户画像]\n{json.dumps(profile, ensure_ascii=False)}")
        if related:
            prompt_parts.append(
                "[相关历史]\n" + "\n".join(f"- {item['content'][:240]}" for item in related)
            )
        recent = session.get("messages", [])[-6:]
        if recent:
            prompt_parts.append(
                "[最近对话]\n"
                + "\n".join(f"{item['role']}: {item['content'][:360]}" for item in recent)
            )
        # token压缩
        prompt_text = self._fit_context(prompt_parts)
        estimated_tokens = self._estimate_tokens(prompt_text)
        return {
            "session_id": session_id,
            "summary": session.get("summary", ""),
            "profile": profile,
            "related_messages": related,
            "recent_messages": recent,
            "prompt_text": prompt_text, # 最终传给LLM的完整提示词字段
            "context_budget": { # Token预算统计
                "limit_tokens": self.context_budget_tokens, # 预算上限
                "estimated_tokens": estimated_tokens, # 预估实际使用的token
                "utilization": round(estimated_tokens / max(self.context_budget_tokens, 1), 3), # 利用率
                "policy": "即时检索 → 重点保留 → 自动压缩",
                "cache_stable_prefix": True,
            },
            "memory_layers": {
                "working": len(session.get("messages", [])), # 工作记忆（最近消息数）
                "episodic": len(session.get("episodes", [])),
                "profile_fields": len(session.get("profile", {})),
            },
        }

    def record_turn(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        记录一轮对话（用户 + 助手），自动压缩与画像更新。

        :raises PermissionError: 未绑定租户上下文时（由 RecordStore 底层抛出）
        """
        with self._lock:
            session = self.get_session(session_id) or self._empty(session_id)
            now = datetime.now().isoformat()
            session["messages"].extend(
                [
                    {"role": "user", "content": user_message, "at": now},
                    {"role": "assistant", "content": assistant_message, "at": now},
                ]
            )
            # 更新用户画像（会话级 + 跨会话用户级）
            session["profile"] = self._update_profile(
                session.get("profile", {}), user_message, metadata or {}
            )
            self._merge_user_profile(session["profile"])
            # 更新累计对话次数和时间戳
            session["turns"] = int(session.get("turns", 0)) + 1
            session["updated_at"] = now
            # 自动生成对话标题（基于第一条用户消息）
            if not session.get("title"):
                session["title"] = self._generate_title(user_message)
            # 自动压缩消息（18条）
            if len(session["messages"]) >= self.compress_at:
                self._compress(session)
            self._write(session)
            return self._summary(session)

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """优先从 SQLite 读取；不存在时回退到 JSON 文件（历史数据），并迁移入库。"""
        session = self.store.get("conversation_session", session_id)
        if session is not None:
            return session
        path = self._path(session_id)
        if not path.exists():
            return None
        try:
            session = json.loads(path.read_text(encoding="utf-8"))
            if not record_visible(session):
                return None
            session.setdefault("tenant_id", current_tenant_id())
            self._write(session)
            return session
        except (OSError, json.JSONDecodeError):
            return None

    def list_sessions(self, limit: int = 20) -> List[Dict[str, Any]]:
        """列出当前租户下的会话（自动触发历史 JSON → SQLite 迁移）。"""
        self._migrate_legacy_sessions()
        records = self.store.list("conversation_session", limit=max(limit, 1))
        return [self._summary(session) for session in records]

    def delete_session(self, session_id: str) -> bool:
        """删除会话；同时清理可能存在的 JSON 历史文件。"""
        path = self._path(session_id)
        if self.get_session(session_id) is None:
            return False
        removed = self.store.delete("conversation_session", session_id)
        if path.exists():
            path.unlink()
        return removed

    def rename_session(self, session_id: str, title: str) -> bool:
        """重命名会话标题；标题为空或会话不存在时返回 False。"""
        session = self.get_session(session_id)
        if session is None:
            return False
        title = (title or "").strip()
        if not title:
            return False
        session["title"] = title[:50]
        session["updated_at"] = datetime.now().isoformat()
        self._write(session)
        return True

    def stats(self) -> Dict[str, Any]:
        """返回当前租户的记忆概览（最多扫描 1000 条会话）。"""
        sessions = self.list_sessions(limit=1000)
        return {
            "sessions": len(sessions),
            "turns": sum(int(item.get("turns", 0)) for item in sessions),
            "working_messages": sum(int(item.get("working_messages", 0)) for item in sessions),
            "episodes": sum(int(item.get("episodes", 0)) for item in sessions),
        }

    # ------------------------------------------------------------------ 会话构造
    def _empty(self, session_id: str) -> Dict[str, Any]:
        now = datetime.now().isoformat()
        return {
            "session_id": session_id,
            "tenant_id": current_tenant_id(),
            "summary": "",
            "profile": {
                "taxpayer_types": [],  # 小规模/一般纳税人/个体户...
                "tax_years": [],      # 最近涉及的年度
                "topics": [],         # 关注话题
                "systems": [],        # 常用业务系统
                "last_intent": None,
                "last_agent": None,
            },
            "messages": [],
            "episodes": [],
            "turns": 0,
            "created_at": now,
            "updated_at": now,
        }

    # ------------------------------------------------------------------ 跨会话用户画像
    def _load_user_profile(self) -> Dict[str, Any]:
        """读取当前租户的跨会话用户画像（不存在返回空）。"""
        try:
            rec = self.store.get(self.USER_PROFILE_NAMESPACE, "profile")
            return rec.get("profile", {}) if rec else {}
        except Exception as e:
            logging.warning(f"读取跨会话画像失败: {e}")
            return {}

    def _save_user_profile(self, profile: Dict[str, Any]) -> None:
        """将跨会话用户画像写入 SQLite（按租户隔离）。"""
        self.store.put(self.USER_PROFILE_NAMESPACE, "profile", {"profile": profile})

    def _merge_user_profile(self, session_profile: Dict[str, Any]) -> None:
        """把当前会话的画像合并进跨会话用户画像（列表字段取并集）。"""
        try:
            user_profile = self._load_user_profile()
            merged = dict(user_profile)
            for key in ("taxpayer_types", "tax_years", "topics", "systems"):
                merged[key] = sorted(
                    set(user_profile.get(key, [])) | set(session_profile.get(key, []))
                )
            if session_profile.get("last_intent"):
                merged["last_intent"] = session_profile["last_intent"]
            if session_profile.get("last_agent"):
                merged["last_agent"] = session_profile["last_agent"]
            merged["updated_at"] = datetime.now().isoformat()
            self._save_user_profile(merged)
        except Exception as e:
            logging.warning(f"合并跨会话画像失败: {e}")

    def _compress(self, session: Dict[str, Any]) -> None:
        """将超出窗口的历史消息压缩成摘要，保留最近 N 条。"""
        archived = session["messages"][:-self.retain_recent]
        if not archived:
            return
        user_points = [item["content"] for item in archived if item.get("role") == "user"][-5:]
        assistant_points = [item["content"] for item in archived if item.get("role") == "assistant"][-3:]
        summary = "；".join([*user_points, *assistant_points])
        summary = re.sub(r"\s+", " ", summary)[:1400]
        previous = session.get("summary", "")
        session["summary"] = (f"{previous}；{summary}" if previous else summary)[-2200:]
        session.setdefault("episodes", []).append(
            {
                "episode_id": f"EP-{len(session.get('episodes', [])) + 1:04d}",
                "summary": summary,
                "message_count": len(archived),
                "confidence": 0.72,
                "status": "active",
                "valid_from": datetime.now().isoformat(),
                "valid_until": None,
                "privacy": "session",
                "helpful_count": 0,
                "harmful_count": 0,
                "created_at": datetime.now().isoformat(),
            }
        )
        session["episodes"] = session["episodes"][-20:]
        session["messages"] = session["messages"][-self.retain_recent:]

    # ------------------------------------------------------------------ 画像更新（税务场景）
    # 画像更新（税务场景）
    def _update_profile(
        self, profile: Dict[str, Any], message: str, metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        updated = dict(profile)

        # 1) 纳税人类型
        taxpayer_types = set(updated.get("taxpayer_types", []))
        for keyword, label in (
            ("小规模", "小规模纳税人"),
            ("一般纳税人", "一般纳税人"),
            ("个体工商户", "个体工商户"),
            ("个体户", "个体工商户"),
            ("合伙企业", "合伙企业"),
            ("个人独资", "个人独资企业"),
        ):
            if keyword in message:
                taxpayer_types.add(label)

        # 2) 年份识别（如 "2024 年"、"23年"）
        tax_years = set(updated.get("tax_years", []))
        for y in re.findall(r"(20\d{2})\s*年?|[0-9]{2}\s*年", message):
            tax_years.add(y)

        # 3) 税务话题（替换原 IT 运维关键词）
        topics = set(updated.get("topics", []))
        for topic in (
            "发票", "增值税", "企业所得税", "个人所得税", "印花税", "消费税",
            "汇算清缴", "留抵退税", "进项", "销项", "抵扣", "税负",
            "小规模", "一般纳税人", "小微企业", "加计扣除", "税前扣除",
            "专项附加扣除", "申报", "稽查", "筹划", "社保", "公积金",
        ):
            if topic in message:
                topics.add(topic)

        # 4) 业务系统识别
        systems = set(updated.get("systems", []))
        systems.update(
            re.findall(r"[\w\u4e00-\u9fff-]{1,24}(?:ERP|CRM|OA|系统|平台|财务软件|金蝶|用友|电子税局)", message, flags=re.IGNORECASE)
        )

        # 5) 意图与 Agent 记录
        last_intent = metadata.get("intent") or updated.get("last_intent")
        last_agent = metadata.get("agent") or updated.get("last_agent")

        updated.update(
            {
                "taxpayer_types": sorted(taxpayer_types),
                "tax_years": sorted(tax_years),
                "topics": sorted(topics),
                "systems": sorted(systems)[:12],
                "last_intent": last_intent,
                "last_agent": last_agent,
                "updated_at": datetime.now().isoformat(),
            }
        )
        return updated

    # ------------------------------------------------------------------ 相关历史 / Token / 上下文压缩
    def _related(self, messages: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        query_tokens = self._tokens(query)
        ranked = []
        for item in messages[:-2]:
            tokens = self._tokens(str(item.get("content", "")))
            if not query_tokens or not tokens:
                continue
            score = len(query_tokens & tokens) / max(len(query_tokens | tokens), 1)
            if score >= 0.08:
                ranked.append((score, item))
        ranked.sort(key=lambda pair: pair[0], reverse=True)
        return [{**item, "score": round(score, 3)} for score, item in ranked[:3]]

    def _tokens(self, text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]{1,4}", text.lower()))


    def _fit_context(self, parts: List[str]) -> str:
        """按 token 预算保留高信号段落，超出部分以占位符遮罩。"""
        if not parts:
            return ""
        selected: List[str] = []
        used_tokens = 0
        mask = "\n[其余内容已遮罩，可按需从会话记忆取回]"
        for part in parts:
            separator_tokens = self._estimate_tokens("\n\n") if selected else 0 #分隔符的token
            remaining = self.context_budget_tokens - used_tokens - separator_tokens
            if remaining <= 0:
                break
            part_tokens = self._estimate_tokens(part)
            if part_tokens <= remaining:
                selected.append(part)
                used_tokens += separator_tokens + part_tokens
                continue

            mask_tokens = self._estimate_tokens(mask)
            available = max(0, remaining - mask_tokens)
            low, high = 0, len(part)
            while low < high:
                middle = (low + high + 1) // 2
                if self._estimate_tokens(part[:middle]) <= available:
                    low = middle
                else:
                    high = middle - 1
            reference = f"{part[:low]}{mask}" if low else mask.strip()
            if self._estimate_tokens(reference) <= remaining:
                selected.append(reference)
            break
        return "\n\n".join(selected)

    def _estimate_tokens(self, text: str) -> int:
        if not text:
            return 0
        latin = len(re.findall(r"[A-Za-z0-9_]+", text))
        cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
        symbols = max(0, len(text) - cjk)
        return max(1, int(cjk * 0.9 + latin * 0.35 + symbols * 0.18))

    # ------------------------------------------------------------------ 内部工具
    def _summary(self, session: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "session_id": session.get("session_id"),
            "title": session.get("title", ""),
            "turns": session.get("turns", 0),
            "working_messages": len(session.get("messages", [])),
            "episodes": len(session.get("episodes", [])),
            "profile": session.get("profile", {}),
            "summary": session.get("summary", ""),
            "updated_at": session.get("updated_at"),
        }

    def set_title_llm(self, llm) -> None:
        """注入用于对话标题概括的 LLM（可选，未注入时标题回退为截断）。"""
        self._title_llm = llm

    def _generate_title(self, text: str) -> str:
        """生成对话标题：优先用 LLM 概括，失败或未配置时回退到前 20 字截断。"""
        cleaned = (text or "").strip().replace("\n", " ")[:200]
        if not cleaned:
            return "新对话"
        if self._title_llm is not None:
            try:
                title = self._summarize_title(cleaned)
                if title:
                    return title
            except Exception as e:
                logging.warning(f"标题 LLM 概括失败，回退截断: {e}")
        return cleaned[:20] + ("…" if len(cleaned) > 20 else "")

    def _summarize_title(self, text: str) -> str:
        """用 LLM 把首条用户消息概括成不超过 10 字的对话标题。"""
        from langchain_core.messages import SystemMessage, HumanMessage
        prompt = "请用不超过 10 个字概括下面这段用户问题的主题，只输出主题本身，不要标点、不要解释。"
        resp = self._title_llm.invoke([SystemMessage(content=prompt), HumanMessage(content=text)])
        content = getattr(resp, "content", "") if resp else ""
        title = (content or "").strip().strip("。，,.！!？?\"'“”‘’ ").replace("\n", " ")
        return title[:20]

    def _path(self, session_id: str) -> Path:
        safe_id = re.sub(r"[^a-zA-Z0-9_-]", "_", session_id)[:120] or "session"
        return self.base_dir / f"{safe_id}.json"

    def _write(self, session: Dict[str, Any]) -> None:
        session.setdefault("tenant_id", current_tenant_id())
        expected = session.get("_storage_version")
        version = self.store.put(
            "conversation_session",
            str(session["session_id"]),
            session,
            expected_version=int(expected) if expected is not None else None,
        )
        session["_storage_version"] = version

    def _migrate_legacy_sessions(self) -> None:
        """将 base_dir 下遗留的 JSON 会话文件迁移进 SQLite。"""
        for path in self.base_dir.glob("*.json"):
            try:
                session = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            session_id = str(session.get("session_id") or path.stem)
            if record_visible(session) and self.store.get("conversation_session", session_id) is None:
                session.setdefault("tenant_id", current_tenant_id())
                self.store.put("conversation_session", session_id, session)