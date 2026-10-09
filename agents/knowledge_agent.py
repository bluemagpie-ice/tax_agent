"""
利用langgraph实现带RAG的税务智能体
功能：
1. 接收用户消息 → 检索知识库 → 调用LLM → 生成回答
2. 使用LangGraph的StateGraph编排

todo: 添加ReAct模式，增加工具调用
todo: 添加human-in-loop模式，增加敏感操作前的人工审批
"""
from typing import TypedDict, Annotated
import logging

# 修改：使用 langchain_openai 替代已弃用的 langchain_community
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import add_messages, StateGraph, START, END
import os
from services.config_manager import ConfigManager

from agents.sub_agents.definition_agent import DefinitionAgent
from agents.sub_agents.policy_agent import PolicyAgent
from knowledge.embedding import EmbeddingService, PersistentDocumentStore
from dotenv import load_dotenv

from agents.base_agent import BaseAgent
from agents.base_context import BaseContext
from agents.streaming import invoke_answer

# 加载 .env 环境变量
load_dotenv()
logger = logging.getLogger(__name__)

# ==================== 状态定义 ====================
class MyState(TypedDict, total=False):
    """
    对话状态定义。

    total=False 表示所有字段都是可选的（非必填），
    这样 BaseAgent 传入的 {"messages": messages} 就能直接作为初始状态。
    """
    messages: Annotated[list, add_messages]   # 对话历史
    user_name: str                             # 用户名
    step_count: int                            # 推理步数
    retrieved_context: str                     # RAG检索到的上下文
    source_types: list[str] | None             # 知识来源筛选
    query_date: str | None                      # 后端自动提取的查询日期

# ==================== SimpleAgent 类 ====================
class KnowledgeAgent(BaseAgent):
    """
    基于 RAG 的简单税务智能体。

    继承 BaseAgent，实现 get_graph() 方法。
    目前功能：知识检索 + LLM 生成回答。
    """

    # ---- 元信息（供前端展示/调试） ----
    name: str = "knowledge_agent"
    description: str = "基于知识库检索的税务问答智能体，可调用政策解读和术语解释子智能体"
    capabilities: list[str] = ["税务问答", "知识检索", "政策解读", "术语解释"]
    context_schema: type[BaseContext] = BaseContext

    # ---- 系统提示词 ----
    SYSTEM_PROMPT = """你是一个专业的税务AI助手。请基于以下检索到的税务知识回答用户问题。
如果检索到的知识不足以回答问题，请如实告知用户。

【检索到的税务知识】：
{retrieved_context}

请根据以上知识，用清晰、准确、易懂的语言回答用户的问题。"""

    def __init__(self, workdir_root: str = ".", **kwargs):
        """
        初始化：创建 LLM 实例和 RAG 服务实例。

        :param workdir_root: 工作目录根路径，默认当前目录
        """
        super().__init__(workdir_root, **kwargs)
        # ---- 子智能体挂载 ----
        self.policy_agent = PolicyAgent(workdir_root=workdir_root)
        self.definition_agent = DefinitionAgent(workdir_root=workdir_root)

        # ---- LLM 配置 ----
        import httpx
        llm_cfg = ConfigManager().get_llm_config()
        self.llm = ChatOpenAI(
            model=llm_cfg["model"],
            api_key=llm_cfg["api_key"],
            base_url=llm_cfg["base_url"],
            temperature=0.7,
            timeout=httpx.Timeout(connect=10.0, read=45.0, write=10.0, pool=5.0),
            max_retries=2,
        )

        # ---- RAG 服务初始化（skip_rebuild=True：Qdrant 已有数据，Agent 只做检索） ----
        try:
            logger.info("正在加载知识库和 Embedding 服务（复用已有索引）...")
            self._store = PersistentDocumentStore()
            self._embedding_service = EmbeddingService(store=self._store, use_qdrant=True, skip_rebuild=True)
            logger.info("RAG 服务加载成功")
        except Exception as e:
            logger.warning(f"RAG 服务加载失败（后续检索将降级）: {e}")
            self._store = None
            self._embedding_service = None


    # ==================== RAG 检索方法 ====================
    def retrieve_context(
        self,
        query: str,
        limit: int = 3,
        source_types: list[str] | None = None,
        query_date: str | None = None,
    ) -> str:
        if self._store is None or self._embedding_service is None:
            return "（知识库检索服务未初始化，请检查 Qdrant 连接）"

        try:
            filters = None
            if source_types:
                filters = {"source_type": source_types}
            fetch_limit = max(limit * 3, 15)
            raw_results = self._embedding_service.search(query, limit=fetch_limit, filters=filters)

            if not raw_results:
                return "（知识库中暂未检索到相关内容）"

            if query_date:
                filtered = []
                for item in raw_results:
                    ed = item.get("effective_date", "")
                    if ed and isinstance(ed, str) and ed <= query_date:
                        filtered.append(item)
                    elif not ed:
                        filtered.append(item)
                raw_results = filtered

            raw_results.sort(
                key=lambda x: (
                    0 if (x.get("is_current") is False or x.get("superseded_by")) else 1,
                    x.get("effective_date") or "",
                ),
                reverse=True,
            )
            results = raw_results[:limit]

            if not results:
                return "（知识库中暂未检索到相关内容）"

            # 把检索到的文本块拼接成上下文
            context_parts = []
            for i, item in enumerate(results, 1):
                extra = ""
                if item.get("is_current") is False or item.get("superseded_by"):
                    sup = item.get("superseded_by", "")
                    extra = (
                        f" [注：此条例已被《{sup}》替代，仅供参考]"
                        if sup else " [注：此文件已废止/失效]"
                    )
                context_parts.append(
                    f"[参考{i}] 相似度:{item['score']:.2f}{extra}\n{item['content']}"
                )

            return "\n\n".join(context_parts)

        except Exception as e:
            logger.error(f"[RAG检索] 失败: {e}")
            return "（知识库检索暂时不可用）"


    # ==================== 节点方法 ====================
    def agent_node(self, state: MyState) -> dict:
        """
        Agent 节点：先检索知识库，再调用 LLM 推理。

        :param state: 当前对话状态
        :return: 更新后的状态
        """
        # 1. 获取用户的最新消息
        user_query = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                user_query = msg.content
                break

        # 2. RAG 检索
        source_types = state.get("source_types")
        query_date = state.get("query_date")
        retrieved_context = self.retrieve_context(
            user_query, source_types=source_types, query_date=query_date,
        )

        # 3. 拼接完整的消息列表
        system_msg = SystemMessage(
            content=self.SYSTEM_PROMPT.format(retrieved_context=retrieved_context)
        )

        # 构建完整的消息列表：系统提示 + 历史消息
        full_messages = [system_msg] + state["messages"]

        # 4. 调用 LLM
        import time as _time
        t0 = _time.time()
        response = invoke_answer(self.llm, full_messages)
        logger.info(f"[KnowledgeAgent] LLM 调用完成，耗时 {_time.time() - t0:.1f}s")

        # 5. 返回更新后的状态
        return {
            "messages": [response],
            "step_count": state.get("step_count", 0) + 1,
            "retrieved_context": retrieved_context,
        }


    # ==================== 条件边判断 ====================
    #todo：条件边判断需要更新
    def should_continue(self, state: MyState) -> str:
        """
        判断是否继续推理（目前直接结束，后续可扩展）。

        :param state: 当前对话状态
        :return: END 或其他节点名
        """
        if state.get("step_count", 0) >= 15:
            return END
        return END


    # ==================== 构建并编译图 ====================
    async def get_graph(self, **kwargs):
        """
        构建并编译 LangGraph。
        首次调用时构建图并缓存到 self.graph，后续调用直接返回缓存。

        子类必须实现此方法（BaseAgent 抽象方法）。
        """
        # 缓存机制：避免重复构建
        if self.graph:
            return self.graph

        builder = StateGraph(MyState)

        # 添加节点
        builder.add_node("agent", self.agent_node)

        # 连接边
        builder.add_edge(START, "agent")
        builder.add_conditional_edges("agent", self.should_continue)

        # 编译并缓存（挂上 checkpointer 以支持多轮历史持久化）
        self.graph = builder.compile(checkpointer=self.checkpointer)
        return self.graph

    # ==================== 判断使用哪个子智能体 ====================
    async def handle(self, query: str, context: dict | None = None) -> str:
        """
        统一入口：根据查询内容路由到对应的子智能体或自身。

        :param query: 用户查询字符串
        :param context: 可选上下文
        :return: Agent 回答字符串
        """
        # 判断是否是政策解读类问题
        if self._is_policy_query(query):
            return await self.policy_agent.run(query, context or {})

        # 判断是否是术语定义类问题
        if self._is_definition_query(query):
            return await self.definition_agent.run(query, context or {})

        # 其他情况：通用检索 + LLM 生成（调用自身）
        return await self._general_qa(query, context or {})

    async def _general_qa(self, query: str, context: dict) -> str:
        """
        通用问答：使用 RAG 检索 + LLM 生成回答（即调用自身的 invoke_messages）。

        :param query: 用户查询
        :param context: 上下文
        :return: 回答字符串
        """
        from langchain_core.messages import HumanMessage
        messages = [HumanMessage(content=query)]
        result = await self.invoke_messages(messages, input_context=context)
        # 提取最后一条 AI 回复
        for msg in reversed(result["messages"]):
            if hasattr(msg, "content") and not isinstance(msg, HumanMessage):
                return msg.content
        return "（未能生成有效回答）"

    def _is_policy_query(self, query: str) -> bool:
        """判断是否为政策解读类问题"""
        keywords = (
            # —— 政策文件类 ——
            "政策", "规定", "条例", "办法", "通知", "公告", "文件",
            # —— 依据类 ——
            "依据", "根据", "按照", "依照",
            # —— 合规判断类 ——
            "是否违反", "符合", "合法", "合规", "违规",
            # —— 税务处理 / 优惠类 ——
            "优惠", "减免", "抵扣", "退税", "税率", "税额",
        )
        return any(kw in query for kw in keywords)

    def _is_definition_query(self, query: str) -> bool:
        """判断是否为术语定义类问题"""
        keywords = ("什么是", "是什么", "定义", "概念", "解释", "含义")
        return any(kw in query for kw in keywords)

    # ==================== 子 Agent 挂载统一入口（可选，自身也可被调用） ====================
    async def run(self, query: str, context: dict | None = None) -> str:
        """
        统一调用入口（与子 Agent 接口一致），直接转发给 handle。

        :param query: 用户查询字符串
        :param context: 可选上下文
        :return: Agent 回答字符串
        """
        return await self.handle(query, context)


# ==================== 测试入口 ====================
if __name__ == "__main__":
    import asyncio

    async def test():
        # 1. 创建智能体实例
        agent = KnowledgeAgent()

        # 2. 准备输入消息
        from langchain_core.messages import HumanMessage
        messages = [HumanMessage(content="增值税的基本税率是多少？")]

        # 3. 调用 BaseAgent 的统一接口
        input_context = {
            "thread_id": "session_001",
            "uid": "user_001",
        }

        result = await agent.invoke_messages(messages, input_context=input_context)

        # 4. 打印结果
        print("\n" + "=" * 50)
        print("测试结果为：")
        for msg in result["messages"]:
            if hasattr(msg, 'content'):
                print(msg.content)
        print(f"\n检索到的上下文：\n{result.get('retrieved_context', '无')}")
        print(f"\n总步数：{result.get('step_count')}")

        # 5. 测试 handle() 方法
        print("\n" + "=" * 50)
        print("测试 handle() 方法（政策解读）：")
        response = await agent.handle("增值税留抵退税政策", context={"thread_id": "handle_test"})
        print(response)

        print("\n" + "=" * 50)
        print("测试 handle() 方法（术语解释）：")
        response = await agent.handle("什么是进项税？", context={"thread_id": "handle_test2"})
        print(response)

        print("\n" + "=" * 50)
        print("测试 handle() 方法（通用问答）：")
        response = await agent.handle("一般纳税人认定标准", context={"thread_id": "handle_test3"})
        print(response)

    asyncio.run(test())
