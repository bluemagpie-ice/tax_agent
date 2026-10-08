"""
RAGSubAgent —— 单轮 RAG 子 Agent 基类
======================================

消除 PolicyAgent / DefinitionAgent 等子 Agent 的重复代码。
子类只需覆盖 5 个类属性 + 可选 1 个方法，其余全部复用。

子类需要提供：
    - NODE_NAME       : LangGraph 中节点的名字，如 "policy" / "definition"
    - STATE_CLASS     : TypedDict 状态类（所有子类字段完全一致，可直接复用 RAGState）
    - SYSTEM_PROMPT   : 专用的系统提示词，含 {retrieved_context} 占位符
    - RETRIEVE_PREFIX : 检索增强 query 前缀，如 "政策" / "术语解释"
    - REF_TAG         : 检索结果引用标签，如 "政策参考" / "术语参考"
    - FALLBACK_TEXT   : run() 无结果时的 fallback 文本
    - name / description / capabilities : 元信息

使用示例：
    class PolicyAgent(RAGSubAgent):
        NODE_NAME = "policy"
        SYSTEM_PROMPT = "...政策解读..."
        RETRIEVE_PREFIX = "政策"
        REF_TAG = "政策参考"
        FALLBACK_TEXT = "（政策解读未返回有效内容）"
        ...
"""
# 新增的修改内容

import logging
import os
from services.config_manager import ConfigManager
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import add_messages, StateGraph, START, END

from agents.base_agent import BaseAgent
from agents.base_context import BaseContext
from knowledge.embedding import EmbeddingService, PersistentDocumentStore

load_dotenv()

# ==================== 统一状态定义 ====================
class RAGState(TypedDict, total=False):
    """所有 RAG 子 Agent 共用的状态结构。"""
    messages: Annotated[list, add_messages]
    step_count: int
    retrieved_context: str
    source_types: list[str] | None
    query_date: str | None

# ==================== RAG 子 Agent 基类 ====================
class RAGSubAgent(BaseAgent):
    """
    单轮 RAG 子 Agent 基类（抽象类）。

    子类只需覆盖 NODE_NAME / SYSTEM_PROMPT / RETRIEVE_PREFIX / REF_TAG / FALLBACK_TEXT
    以及元信息（name / description / capabilities）。
    """

    # ---- 子类必须覆盖 ----
    NODE_NAME: str = "rag"                       # LangGraph 节点名
    STATE_CLASS: type = RAGState                 # 状态 TypedDict（默认 RAGState，子类可覆盖）
    SYSTEM_PROMPT: str = ""                      # 必须含 {retrieved_context}
    RETRIEVE_PREFIX: str = ""                    # 检索增强前缀
    REF_TAG: str = "参考"                        # 检索引用标签
    FALLBACK_TEXT: str = "（未返回有效内容）"      # run() 无结果时的 fallback

    # ---- 元信息（子类必须覆盖） ----
    name: str = "rag_sub_agent"
    description: str = ""
    capabilities: list[str] = []
    context_schema: type[BaseContext] = BaseContext

    # ---- 内部使用的统一日志前缀 ----
    _log_prefix: str = "RAGSubAgent"

    def __init__(self, workdir_root: str = ".", **kwargs):
        super().__init__(workdir_root, **kwargs)

        # LLM 配置
        llm_cfg = ConfigManager().get_llm_config()
        self.llm = ChatOpenAI(
            model=llm_cfg["model"],
            api_key=llm_cfg["api_key"],
            base_url=llm_cfg["base_url"],
            temperature=0.3,
            timeout=60.0,
            max_retries=3,
        )

        # RAG 服务初始化
        try:
            logging.info(f"[{self._log_prefix}] 正在加载知识库和 Embedding 服务...")
            self._store = PersistentDocumentStore()
            self._embedding_service = EmbeddingService(store=self._store, use_qdrant=True)
            logging.info(f"[{self._log_prefix}] RAG 服务加载成功")
        except Exception as e:
            logging.warning(f"[{self._log_prefix}] RAG 服务加载失败（后续检索将降级）: {e}")
            self._store = None
            self._embedding_service = None

    # ==================== RAG 检索 ====================
    def retrieve_context(
            self,
            query: str,
            limit: int = 3,
            filters: dict | None = None,
            source_types: list[str] | None = None,
            query_date: str | None = None,
    ) -> str:
        """从知识库检索相关文本块，拼接为上下文。

        :param source_types: 知识来源筛选
        :param query_date: 后端自动提取的查询日期（YYYY-MM-DD），
                           用于筛选该日期前生效的条例；None 表示不限日期
        """
        if self._store is None or self._embedding_service is None:
            return "（知识库检索服务未初始化，请检查 Qdrant 连接）"

        try:
            merged_filters = dict(filters) if filters else {}
            if source_types:
                merged_filters["source_type"] = source_types

            fetch_limit = max(limit * 3, 15)
            enhanced_query = f"{self.RETRIEVE_PREFIX} {query}" if self.RETRIEVE_PREFIX else query
            raw_results = self._embedding_service.search(
                enhanced_query, limit=fetch_limit,
                filters=merged_filters if merged_filters else None,
            )

            if not raw_results:
                return f"（知识库中暂未检索到相关{self.REF_TAG}内容）"

            if query_date:
                # 有具体日期：过滤 effective_date <= query_date
                filtered = []
                for item in raw_results:
                    ed = item.get("effective_date", "")
                    if ed and isinstance(ed, str) and ed <= query_date:
                        filtered.append(item)
                    elif not ed:
                        filtered.append(item)
                raw_results = filtered

            # 排序：现行有效优先 + effective_date 降序
            raw_results.sort(
                key=lambda x: (
                    0 if (x.get("effective_date") and x.get("is_current") is not False) else 1,
                    x.get("effective_date") or "",
                ),
                reverse=True,
            )

            results = raw_results[:limit]

            if not results:
                return f"（知识库中未找到{'适用于 ' + query_date if query_date else ''}的相关{self.REF_TAG}内容）"

            context_parts = []
            for i, item in enumerate(results, 1):
                # 组装引用溯源信息
                ref_lines = [f"[相似度:{item['score']:.2f}]"]
                if item.get("title"):
                    ref_lines.append(f"《{item['title']}》")
                if item.get("document_number"):
                    ref_lines.append(f"{item['document_number']}")
                if item.get("issuer"):
                    ref_lines.append(f"发布机关：{item['issuer']}")
                if item.get("effective_date"):
                    ref_lines.append(f"生效日期：{item['effective_date']}")
                if item.get("is_current") is False or item.get("superseded_by"):
                    superseded = item.get("superseded_by", "")
                    ref_lines.append(
                        f"注：此条例已被《{superseded}》替代，内容仅供参考"
                        if superseded else "注：此文件已废止/失效"
                    )
                if item.get("expiry_date"):
                    ref_lines.append(f"废止日期：{item['expiry_date']}")

                header = f"[{self.REF_TAG}{i}] {' | '.join(ref_lines)}"
                context_parts.append(f"{header}\n{item['content']}")
            return "\n\
".join(context_parts)

        except Exception as e:
            logging.error(f"[{self._log_prefix}] RAG 检索失败: {e}")
            return "（知识库检索暂时不可用）"

    # ==================== 节点方法（统一实现） ====================
    def _rag_node(self, state: dict) -> dict:
        """
        统一的 RAG 节点：取用户消息 → 检索 → 拼 SystemMessage → 调 LLM → 返回状态。
        """
        user_query = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                user_query = msg.content
                break

        source_types = state.get("source_types")
        query_date = state.get("query_date")
        retrieved_context = self.retrieve_context(
            user_query, source_types=source_types, query_date=query_date,
        )

        system_msg = SystemMessage(
            content=self.SYSTEM_PROMPT.format(retrieved_context=retrieved_context)
        )
        full_messages = [system_msg] + state["messages"]

        response = self.llm.invoke(full_messages)

        return {
            "messages": [response],
            "step_count": state.get("step_count", 0) + 1,
            "retrieved_context": retrieved_context,
        }

    # ==================== 条件边（默认直接结束） ====================
    def should_continue(self, state: dict) -> str:
        return END

    # ==================== 构建并编译图 ====================
    async def get_graph(self, **kwargs):
        if self.graph:
            return self.graph

        builder = StateGraph(self.STATE_CLASS)
        builder.add_node(self.NODE_NAME, self._rag_node)
        builder.add_edge(START, self.NODE_NAME)
        builder.add_conditional_edge(self.NODE_NAME, self.should_continue)

        self.graph = builder.compile(checkpointer=self.checkpointer)
        return self.graph

    # ==================== 子 Agent 统一入口 ====================
    async def run(self, query: str, context: dict | None = None) -> str:
        messages = [HumanMessage(content=query)]
        if context is None:
            context = {}

        result = await self.invoke_messages(messages, input_context=context)

        for msg in reversed(result["messages"]):
            if hasattr(msg, "content") and not isinstance(msg, HumanMessage):
                return msg.content

        return self.FALLBACK_TEXT