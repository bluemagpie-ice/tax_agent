"""
政策解读智能体（PolicyAgent）
----------------------------
继承 RAGSubAgent 基类，仅提供差异配置。
"""

from agents.sub_agents.rag_sub_agent import RAGSubAgent

class PolicyAgent(RAGSubAgent):
    """税务政策解读智能体。"""

    _log_prefix = "PolicyAgent"
    NODE_NAME = "policy"
    RETRIEVE_PREFIX = "政策"
    REF_TAG = "政策参考"
    FALLBACK_TEXT = "（政策解读未返回有效内容）"

    name = "policy_agent"
    description = "税务政策解读智能体，专注政策条文、适用条件与要点解读"
    capabilities = ["政策解读", "政策查询", "适用条件说明"]

    SYSTEM_PROMPT = """你是一个专业的税务政策解读专家。请基于以下检索到的税务政策内容，
为用户提供清晰、准确、结构化的政策解读。

【检索到的税务政策】：
{retrieved_context}

请按以下结构输出（如有对应内容）：
1. 【政策要点】：提炼核心要点，用简洁语言概括
2. 【适用条件】：说明政策适用的前提条件和范围
3. 【注意事项】：指出执行时需要关注的细节和风险点
4. 【参考依据】：标注政策条文出处

如果检索到的内容不足以回答问题，请如实告知用户。"""

# ==================== 测试入口 ====================
if __name__ == "__main__":
    import asyncio
    from langchain_core.messages import HumanMessage

    async def test():
        agent = PolicyAgent()
        messages = [HumanMessage(content="增值税留抵退税政策")]

        input_context = {"thread_id": "policy_test_001", "uid": "user_001"}
        result = await agent.invoke_messages(messages, input_context=input_context)

        print("\n" + "=" * 50)
        print("政策解读测试结果：")
        for msg in result["messages"]:
            if hasattr(msg, "content"):
                print(msg.content)
        print(f"\n检索到的政策上下文：\n{result.get('retrieved_context', '无')}")

        print("\n" + "=" * 50)
        print("测试 run() 方法：")
        response = await agent.run("小微企业税收优惠有哪些？", context={"thread_id": "run_test_001"})
        print(response)

    asyncio.run(test())