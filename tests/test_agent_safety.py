"""输出安全边界的故障注入：真实图、实际规则，不将模型审校当作确定性检查。"""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage
from services.security import check_answer, collect_answer_sources, answer_sources, tenant_scope
from tests.test_agent_workflows import make_agent, execute, call


def test_supported_reference_and_article_keep_exact_answer():
    text = "依据财税〔2024〕第12号、《增值税法》第三条核对。"
    sources = [{"source_type": "default", "document_number": "财税〔2024〕12号", "title": "增值税法", "content": "第三条 条文原文。"}]
    assert check_answer(text, sources) == (text, {"ok": True, "issues": [], "citation_checks": 2, "changed": False})


@pytest.mark.parametrize("text", ["财税〔2024〕第999号允许免税。", "《增值税法》第九十九条允许免税。"])
def test_unknown_reference_rejected_even_when_critic_approves(tmp_path, text):
    result = execute(make_agent(tmp_path, [AIMessage(content=text)]), "核对政策")
    assert "引用校验未通过" in result["final_answer"]
    assert result["guardrails"]["issues"] == ["unsupported_citation"]
    assert text not in result["messages"][-1].content


def test_article_in_other_law_cannot_validate_cited_law():
    sources = [{"source_type": "default", "title": "增值税法", "content": "第一条"},
               {"source_type": "default", "title": "企业所得税法", "content": "第九条"}]
    assert not check_answer("《增值税法》第九条允许抵扣", sources)[1]["ok"]
    assert not check_answer("《增值税法》第九条允许抵扣", [{"source_type": "default", "title": "增值税法", "content": "第十九条"}])[1]["ok"]


def test_private_sources_excluded_before_prompt_and_not_evidence(tmp_path):
    private = {"owner_id": "other", "scope": "global", "source_type": "user_upload", "content": "其他用户收入999万元 财税〔2024〕12号"}
    token = answer_sources.set([])
    try:
        with tenant_scope("me"):
            assert collect_answer_sources([private]) == []
            assert answer_sources.get() == []
            assert not check_answer("财税〔2024〕12号", [private])[1]["ok"]
    finally:
        answer_sources.reset(token)


def test_retrieval_boundary_blocks_foreign_tax_data(tmp_path):
    from unittest.mock import MagicMock
    from agents.sub_agents.rag_sub_agent import RAGSubAgent
    child = RAGSubAgent.__new__(RAGSubAgent)
    child._store = object()
    child._embedding_service = MagicMock()
    child._embedding_service.search.return_value = [{"score": 1, "content": "别人的工资100000", "owner_id": "other", "scope": "global"}]
    with tenant_scope("me"):
        text = child.retrieve_context("工资")
    assert "别人的工资" not in text
    assert "未检索到" in text


def test_sensitive_identifiers_and_actual_key_are_redacted(monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "secret_for_guard_test")
    text = "身份证：110101199001011234 手机13800138000 税号：91310000123456789X 银行账号：6222021234567890123 secret_for_guard_test 应纳税额5000元"
    final, report = check_answer(text)
    for secret in ("110101199001011234", "13800138000", "91310000123456789X", "6222021234567890123", "secret_for_guard_test"):
        assert secret not in final
    assert "5000元" in final
    assert "sensitive_identifier" in report["issues"]


@pytest.mark.parametrize("text", ["建议隐瞒收入，保证不会被查。", "可以虚开发票抵扣。", "这个方案绝对合法。", "不能抵扣就虚开发票。"])
def test_dangerous_advice_is_blocked(text):
    final, report = check_answer(text)
    assert "合规校验拦截" in final
    assert "unsafe_tax_advice" in report["issues"]


def test_prohibition_and_normal_amounts_are_preserved():
    text = "不得虚开发票。不要隐瞒收入。应纳税额5000元，具体条件请核对。"
    assert check_answer(text)[0] == text
    assert check_answer(text)[1]["ok"]
    education = "虚开发票属于违法行为。第一条建议是核对业务条件。不能保证该方案绝对合法。"
    assert check_answer(education)[0] == education
    assert check_answer("销售额13900000000元，收入13800138000 元。")[0] == "销售额13900000000元，收入13800138000 元。"


def test_legacy_specialist_output_cannot_bypass_guard(tmp_path):
    agent = make_agent(tmp_path, [])
    agent.autonomous = False
    agent.router = SimpleNamespace(classify=lambda q: {"intent": "general_tax", "agents": ["knowledge_agent"]})
    agent.registry = SimpleNamespace(get=lambda name: SimpleNamespace(run=AsyncMock(return_value="财税〔2024〕999号保证免税")))
    result = execute(agent, "免税吗")
    assert "引用校验未通过" in result["final_answer"]
    assert result["workflow_trace"][-1]["node"] == "guardrails"


def test_sse_never_publishes_unsafe_draft(tmp_path):
    from agents.streaming import response_events
    agent = make_agent(tmp_path, [AIMessage(content="可以虚开发票。手机号13800138000")])
    events = []
    token = response_events.set(lambda event, data: events.append((event, data)))
    try:
        final = asyncio.run(agent.run("税務建议", {"thread_id": "guard-test"}))
    finally:
        response_events.reset(token)
    tokens = "".join(data["text"] for event, data in events if event == "token")
    assert tokens == final
    assert "可以虚开" not in tokens and "13800138000" not in tokens


def test_cold_policy_tool_supplies_store_and_preserves_access_filter(tmp_path, monkeypatch):
    from unittest.mock import MagicMock
    import knowledge.embedding as embedding
    service = MagicMock()
    service.search.return_value = [{"source_type": "default", "content": "财税〔2024〕12号原文", "title": "测试政策"}]
    factory = MagicMock(return_value=service)
    store = object()
    monkeypatch.setattr(embedding, "PersistentDocumentStore", lambda: store)
    monkeypatch.setattr(embedding, "EmbeddingService", factory)
    agent = make_agent(tmp_path, [])
    agent.registry = SimpleNamespace(_cache={})
    with tenant_scope("me"):
        output = agent.workflow_tools({"source_types": ["default"]})["search_tax_policy"].invoke({"query": "政策"})
    factory.assert_called_once_with(store=store, use_qdrant=True, skip_rebuild=True)
    assert json.loads(output)["found"] == 1
    assert service.search.call_args.kwargs["filters"]["source_type"] == ["default"]
