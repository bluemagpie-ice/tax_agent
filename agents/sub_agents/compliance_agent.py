"""
税务合规检查智能体（ComplianceAgent）
----------------------------------
继承 RAGSubAgent 基类，仅提供差异配置。
专注税务风险识别、合规性自查与稽查应对。
"""

from agents.sub_agents.rag_sub_agent import RAGSubAgent

class ComplianceAgent(RAGSubAgent):
    """税务合规检查智能体。"""

    _log_prefix = "ComplianceAgent"
    NODE_NAME = "compliance"
    RETRIEVE_PREFIX = "合规"
    REF_TAG = "合规参考"
    FALLBACK_TEXT = "（合规检查未返回有效内容）"

    name = "compliance_agent"
    description = "税务合规检查智能体，专注税务风险识别、合规性自查与稽查应对"
    capabilities = ["合规风险识别", "税务自查", "稽查应对", "合规性判断"]

    SYSTEM_PROMPT = """你是一个专业的税务合规检查专家。请基于以下检索到的合规相关规定，
为用户提供清晰、准确、可操作的合规判断与风险提示。

【检索到的合规规定】：
{retrieved_context}

请按以下结构输出（如有对应内容）：
1. 【合规结论】：先给出明确判断（合规/存在风险/需进一步核实）
2. 【风险点】：列出可能存在的税务风险
3. 【整改建议】：给出可操作的整改或规避建议
4. 【参考依据】：标注相关法规或公告出处

重要：你的职责是识别风险、给出合规建议，不是教用户规避监管。涉及重大风险的，请建议用户咨询专业税务顾问。如果检索到的内容不足以判断，请如实告知。"""


if __name__ == "__main__":
    import asyncio

    async def test():
        agent = ComplianceAgent()
        response = await agent.run(
            "我们公司这样做账是否有税务风险？", context={"thread_id": "compliance_test_001"}
        )
        print(response)

    asyncio.run(test())
