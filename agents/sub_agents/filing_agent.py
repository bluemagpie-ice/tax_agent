"""
纳税申报指导智能体（FilingAgent）
------------------------------
继承 RAGSubAgent 基类，仅提供差异配置。
专注申报流程、汇算清缴、申报期限与补退税办理。
"""

from agents.sub_agents.rag_sub_agent import RAGSubAgent

class FilingAgent(RAGSubAgent):
    """纳税申报指导智能体。"""

    _log_prefix = "FilingAgent"
    NODE_NAME = "filing"
    RETRIEVE_PREFIX = "申报"
    REF_TAG = "申报参考"
    FALLBACK_TEXT = "（申报指导未返回有效内容）"

    name = "filing_agent"
    description = "纳税申报指导智能体，专注申报流程、汇算清缴、申报期限与补退税办理"
    capabilities = ["申报流程指导", "汇算清缴", "申报期限提醒", "补退税办理"]

    SYSTEM_PROMPT = """你是一个专业的纳税申报指导专家。请基于以下检索到的申报相关规定，
为用户提供清晰、准确、可操作的申报指导。

【检索到的申报规定】：
{retrieved_context}

请按以下结构输出（如有对应内容）：
1. 【申报结论】：先给出明确结论（如何申报/截止时间/是否需补退税）
2. 【申报流程】：说明具体申报步骤
3. 【注意事项】：指出申报期限、所需材料等关键细节
4. 【参考依据】：标注相关法规或公告出处

如果检索到的内容不足以回答问题，请如实告知用户，不要臆造申报流程或期限。"""


if __name__ == "__main__":
    import asyncio

    async def test():
        agent = FilingAgent()
        response = await agent.run(
            "年度企业所得税汇算清缴怎么申报？", context={"thread_id": "filing_test_001"}
        )
        print(response)

    asyncio.run(test())
