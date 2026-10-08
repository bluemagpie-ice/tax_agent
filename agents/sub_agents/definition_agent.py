"""
术语解释智能体（DefinitionAgent）
----------------------------
继承 RAGSubAgent 基类，仅提供差异配置。
"""

from agents.sub_agents.rag_sub_agent import RAGSubAgent

class DefinitionAgent(RAGSubAgent):
    """税务术语解释智能体。"""

    _log_prefix = "DefinitionAgent"
    NODE_NAME = "definition"
    RETRIEVE_PREFIX = "术语解释"
    REF_TAG = "术语参考"
    FALLBACK_TEXT = "（术语解释未返回有效内容）"

    name = "definition_agent"
    description = "税务术语解释智能体，专注术语定义、概念说明与要点解读"
    capabilities = ["术语解释", "概念定义", "要点说明"]

    SYSTEM_PROMPT = """你是一个专业的税务术语解释专家。请基于以下检索到的税务术语相关内容，
为用户提供清晰、准确、易懂的术语解释。

【检索到的术语内容】：
{retrieved_context}

请按以下结构输出（如有对应内容）：
1. 【术语定义】：用简洁语言给出术语的核心定义
2. 【要点说明】：解释术语的关键要点、适用范围
3. 【常见示例】：列举典型场景帮助理解
4. 【参考依据】：标注相关条文或出处

如果检索到的内容不足以回答问题，请如实告知用户。"""

# ==================== 测试入口 ====================
if __name__ == "__main__":
    import asyncio
    from langchain_core.messages import HumanMessage

    async def test():
        agent = DefinitionAgent()
        messages = [HumanMessage(content="什么是留抵退税？")]

        input_context = {"thread_id": "definition_test_001", "uid": "user_001"}
        result = await agent.invoke_messages(messages, input_context=input_context)

        print("\n" + "=" * 50)
        print("术语解释测试结果：")
        for msg in result["messages"]:
            if hasattr(msg, "content"):
                print(msg.content)
        print(f"\n检索到的术语上下文：\n{result.get('retrieved_context', '无')}")

        print("\n" + "=" * 50)
        print("测试 run() 方法：")
        response = await agent.run("什么是进项税？", context={"thread_id": "run_test_002"})
        print(response)

    asyncio.run(test())