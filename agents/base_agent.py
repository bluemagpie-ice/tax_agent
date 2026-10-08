"""
基础 Agent 类
供各类税务 Agent 继承使用

设计目的：
1. 提供统一的元信息（name / description / capabilities / context_schema）
2. 提供统一的工作目录（workdir）管理
3. 提供统一的对话接口（stream_messages / invoke_messages）
4. 子类只需实现 get_graph() 方法，返回编译好的 LangGraph
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional
import logging

from agents.base_context import BaseContext

# LangGraph 状态持久化（checkpointer）—— 使用 AsyncSqliteSaver 适配 FastAPI 异步模型
# AsyncSqliteSaver 接受 aiosqlite.Connection，兼容 graph.astream() 调用的 async 方法
# 文件路径：<项目根>/data/checkpoints.sqlite
# 进程重启后自动恢复，真正实现多轮对话状态持久化
# 依赖：pip install aiosqlite
try:
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    _CHECKPOINT_AVAILABLE = True
except ImportError:
    _CHECKPOINT_AVAILABLE = False
    AsyncSqliteSaver = None  # type: ignore[assignment]
    logging.warning(
        "langgraph.checkpoint.sqlite.aio 不可用，请运行 pip install aiosqlite。"
    )

# 项目根目录
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# SQLite checkpoint 文件路径（每个 Agent 实例首次调用 stream/invoke 时用 aiosqlite 连接）
_CHECKPOINT_DB_PATH = _PROJECT_ROOT / "data" / "checkpoints.sqlite"

# 长记忆模块（ConversationMemory）—— 懒加载单例，跨 Agent 共享
_conversation_memory: Optional["ConversationMemory"] = None
_memory_lock = None


def _get_shared_memory() -> "ConversationMemory":
    """获取全局共享的 ConversationMemory 单例（线程安全，懒初始化）。"""
    global _conversation_memory, _memory_lock
    if _conversation_memory is not None:
        return _conversation_memory
    import threading
    if _memory_lock is None:
        _memory_lock = threading.Lock()
    with _memory_lock:
        if _conversation_memory is not None:
            return _conversation_memory
        from agents.memory import ConversationMemory
        _conversation_memory = ConversationMemory()
        # 注入标题概括 LLM（可选；失败或未配置时 _generate_title 自动回退为截断）
        try:
            from services.config_manager import ConfigManager
            from langchain_openai import ChatOpenAI
            cfg = ConfigManager().get_llm_config()
            _conversation_memory.set_title_llm(ChatOpenAI(
                model=cfg["model"],
                api_key=cfg["api_key"],
                base_url=cfg["base_url"],
                temperature=0.2,
                timeout=30.0,
                max_retries=2,
            ))
        except Exception:
            pass
        return _conversation_memory

class BaseAgent(ABC):
    """
    所有税务 Agent 的基类。

    子类需要覆盖：
        name            智能体名称
        description     智能体描述
        capabilities    智能体能力列表
        context_schema  上下文 Schema 类型
        get_graph()     构建并返回编译后的 LangGraph
    """

    name: str = "base_agent"
    description: str = "base_agent"
    capabilities: list[str] = []
    context_schema: type[BaseContext] = BaseContext

    # 工作目录根路径（项目级配置，便于被覆盖）
    base_workdir: str = "agents"

    def __init__(self, workdir_root: str = ".", **kwargs):
        """
        :param workdir_root: 工作目录根路径，默认当前目录
        """
        self.graph = None  # 由 get_graph 构建并缓存
        self.workdir = (
                Path(workdir_root) / self.base_workdir / self.__class__.__name__
        )
        self.workdir.mkdir(parents=True, exist_ok=True)

        # ---- checkpointer 懒初始化（首次 stream/invoke 时通过 aiosqlite 连接） ----
        self._checkpointer: Any = None

    # ---- checkpointer 属性 + 懒初始化方法 ----

    @property
    def checkpointer(self):
        """供 get_graph() / has_checkpointer() 读取的 checkpointer 实例。

        首次调用 stream_messages / invoke_messages 之前尚未初始化，
        返回 None 不会导致编译失败（LangGraph builder.compile(checkpointer=None) 正常）。
        """
        return self._checkpointer

    async def _ensure_checkpointer(self) -> None:
        """确保 AsyncSqliteSaver 已创建（懒初始化，幂等）。

        首次调用时通过 aiosqlite.connect() 建立连接，后续调用直接返回。
        """
        if self._checkpointer is not None or not _CHECKPOINT_AVAILABLE:
            return
        try:
            import aiosqlite
            _CHECKPOINT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
            conn = await aiosqlite.connect(str(_CHECKPOINT_DB_PATH))
            self._checkpointer = AsyncSqliteSaver(conn)
            await self._checkpointer.setup()
            logging.info(
                f"[{self.__class__.__name__}] AsyncSqliteSaver 已初始化: {_CHECKPOINT_DB_PATH}"
            )
        except Exception as e:
            logging.warning(
                f"[{self.__class__.__name__}] AsyncSqliteSaver 初始化失败: {e}"
            )
            self._checkpointer = None

    # ------------------------------------------------------------
    # 元信息
    # ------------------------------------------------------------
    def get_info(self) -> dict:
        """返回智能体元信息，供前端展示/调试"""
        return {
            "name": self.name,
            "description": self.description,
            "capabilities": self.capabilities,
            "context_schema": self.context_schema.__name__,
        }

    # ------------------------------------------------------------
    # 子类必覆写
    # ------------------------------------------------------------
    @abstractmethod
    def get_graph(self, **kwargs):
        """
        构建并返回编译后的 LangGraph 实例。
        建议：在编译时设置 checkpointer，以支持多轮历史查询。
        """
        raise NotImplementedError

    # ------------------------------------------------------------
    # 主对话接口（同步封装为异步 generator）
    # ------------------------------------------------------------
    async def stream_messages(
        self,
        messages: list,
        input_context: dict | None = None,
        **kwargs: Any,
    ):
        """
        流式输出对话消息。

        :param messages: 本轮输入消息列表（LangChain 消息对象或字符串）
        :param input_context: 会话上下文（thread_id、uid、system_prompt 等）
        """
        context = self.context_schema()
        if input_context:
            for k, v in input_context.items():
                if hasattr(context, k):
                    setattr(context, k, v)

        await self._ensure_checkpointer()
        graph = await self.get_graph(**kwargs)

        config = {
            "configurable": {
                "thread_id": context.thread_id,
                "uid": context.uid,
            },
            "recursion_limit": kwargs.get("recursion_limit", 50),
        }

        # Langfuse / 回调可选参数
        for key in ("callbacks", "metadata", "tags"):
            if kwargs.get(key):
                config[key] = list(kwargs[key]) if isinstance(kwargs[key], (list, tuple)) else kwargs[key]

        async for event in graph.astream(
            {"messages": messages},
            stream_mode="messages",
            config=config,
        ):
            yield event


    async def invoke_messages(
        self,
        messages: list,
        input_context: dict | None = None,
        **kwargs: Any,
    ):
        """非流式调用，直接返回完整响应"""
        context = self.context_schema()
        if input_context:
            for k, v in input_context.items():
                if hasattr(context, k):
                    setattr(context, k, v)

        await self._ensure_checkpointer()
        graph = await self.get_graph(**kwargs)

        config = {
            "configurable": {
                "thread_id": context.thread_id,
                "uid": context.uid,
            },
            "recursion_limit": kwargs.get("recursion_limit", 50),
        }

        initial_state: dict = {"messages": messages}
        if hasattr(context, "source_types") and getattr(context, "source_types"):
            initial_state["source_types"] = getattr(context, "source_types")

        return await graph.ainvoke(initial_state, config=config)

    # ------------------------------------------------------------
    # 历史记录 / 状态检查
    # ------------------------------------------------------------
    async def get_history(self, thread_id: str) -> list[dict]:
        """
        获取指定会话的历史消息。
        要求：get_graph() 编译时必须挂上 checkpointer，否则返回空列表。
        """
        try:
            await self._ensure_checkpointer()
            graph = await self.get_graph()
            if not getattr(graph, "checkpointer", None):
                return []

            config = {"configurable": {"thread_id": thread_id}}
            state = await graph.aget_state(config)
            if not state:
                return []

            history = []
            for msg in state.values.get("messages", []):
                if hasattr(msg, "model_dump"):
                    history.append(msg.model_dump())
                elif hasattr(msg, "__dict__"):
                    history.append(vars(msg))
                else:
                    history.append({"content": str(msg)})
            return history
        except Exception:
            return []

    async def has_checkpointer(self) -> bool:
        """检查 graph 是否具备状态持久化能力"""
        graph = await self.get_graph()
        return getattr(graph, "checkpointer", None) is not None

    async def delete_thread(self, thread_id: str) -> bool:
        """删除指定会话的 LangGraph checkpointer 状态（含多轮历史）。

        :return: 是否实际执行了删除（checkpointer 不可用时返回 False）
        """
        await self._ensure_checkpointer()
        if self._checkpointer is None:
            return False
        try:
            await self._checkpointer.adelete_thread(thread_id)
            return True
        except Exception as e:
            logging.warning(f"[{self.__class__.__name__}] 删除 checkpointer 线程失败: {e}")
            return False

    # ------------------------------------------------------------
    # 长记忆方法（委托给全局 ConversationMemory 单例）
    # ------------------------------------------------------------
    def get_memory_context(self, session_id: str, message: str) -> dict:
        """
        获取记忆上下文包，可在调用 LLM 前注入到 system prompt 中。

        返回包含 summary、profile、related_messages、recent_messages、
        prompt_text（完整记忆提示词）等字段的字典。
        """
        return _get_shared_memory().context_for(session_id, message)

    def record_turn(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
        metadata: dict | None = None,
    ) -> dict:
        """
        记录一轮对话到长记忆，自动触发压缩和画像更新。

        :param session_id:        会话ID
        :param user_message:      用户消息
        :param assistant_message: 助手回复
        :param metadata:          可选元数据（intent、agent 等）
        :return:                  会话摘要
        """
        return _get_shared_memory().record_turn(
            session_id, user_message, assistant_message, metadata
        )

    def get_memory(self, session_id: str) -> dict | None:
        """获取指定会话的完整记忆数据。"""
        return _get_shared_memory().get_session(session_id)

    def list_memory_sessions(self, limit: int = 20) -> list[dict]:
        """列出当前租户下的所有记忆会话。"""
        return _get_shared_memory().list_sessions(limit=limit)

    def delete_memory_session(self, session_id: str) -> bool:
        """删除指定会话的记忆。"""
        return _get_shared_memory().delete_session(session_id)

    def memory_stats(self) -> dict:
        """返回当前租户的记忆健康度概况。"""
        return _get_shared_memory().stats()

    # ------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------
    def reload_graph(self):
        """清空 graph 缓存，下次 get_graph 时重新构建"""
        self.graph = None

    async def run(self, user_query, context):
        pass