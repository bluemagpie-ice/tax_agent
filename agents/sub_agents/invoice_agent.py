"""
发票处理智能体（InvoiceAgent）
----------------------------
继承 RAGSubAgent 基类，仅提供差异配置。
处理发票开具、抬头、专票/普票、红字发票、抵扣与报销规范等问题。
"""

from agents.sub_agents.rag_sub_agent import RAGSubAgent

class InvoiceAgent(RAGSubAgent):
    """税务发票处理智能体。"""

    _log_prefix = "InvoiceAgent"
    NODE_NAME = "invoice"
    RETRIEVE_PREFIX = "发票"
    REF_TAG = "发票参考"
    FALLBACK_TEXT = "（发票处理未返回有效内容）"

    name = "invoice_agent"
    description = "发票处理智能体，专注发票开具、抬头、专票/普票、红字发票、抵扣与报销规范"
    capabilities = ["发票开具指导", "发票抵扣判断", "红字发票流程", "报销规范说明"]

    SYSTEM_PROMPT = """你是一个专业的发票处理专家。请基于以下检索到的发票相关规定，
为用户提供清晰、准确、可操作的发票处理建议。

【检索到的发票规定】：
{retrieved_context}

请按以下结构输出（如有对应内容）：
1. 【处理结论】：先给出明确结论（能否开具/能否抵扣/如何处理）
2. 【操作步骤】：说明具体操作流程
3. 【注意事项】：指出容易出错或需要留意的细节
4. 【参考依据】：标注相关法规或公告出处

如果检索到的内容不足以回答问题，请如实告知用户，不要臆造发票规则。"""


if __name__ == "__main__":
    import asyncio

    async def test():
        agent = InvoiceAgent()
        response = await agent.run(
            "餐饮发票能开专票抵扣吗？", context={"thread_id": "invoice_test_001"}
        )
        print(response)

    asyncio.run(test())
