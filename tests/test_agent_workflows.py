"""真实 LangGraph + 临时 SQLite；可控模型验收分支、边界及恢复，不替代真实模型实测。"""
from tests.auth_helpers import USER_BEARER
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from agents.base_agent import BaseAgent
from agents.tax_agent import TaxAgent


class QueryInput(BaseModel):
    query: str


class ScriptedModel:
    model_name = "test-model"

    def __init__(self, responses, reviews=None):
        self.responses = iter(responses)
        self.reviews = iter(reviews) if reviews is not None else None
        self.histories = []
        self.review_histories = []

    def bind_tools(self, tools, **kwargs):
        self.bound_names = {t.name for t in tools}
        return self

    def invoke(self, history, **kwargs):
        if history[0].content.startswith("你是独立税务审校员"):
            self.review_histories.append(history)
            verdict = next(self.reviews) if self.reviews is not None else {"ok": True, "issues": [], "corrected_answer": ""}
            if isinstance(verdict, Exception):
                raise verdict
            return AIMessage(content=json.dumps(verdict, ensure_ascii=False))
        self.histories.append(history)
        response = next(self.responses)
        return response(history) if callable(response) else response


def call(name, args, identity="call-1"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": identity}])


def make_agent(tmp_path, responses):
    agent = TaxAgent.__new__(TaxAgent)
    BaseAgent.__init__(agent, str(tmp_path))
    agent.autonomous = True
    agent.max_tool_rounds = 2
    agent.llm = ScriptedModel(responses)
    agent._ensure_checkpointer = AsyncMock()
    agent.get_memory_context = MagicMock(return_value={})
    agent.record_turn = MagicMock()
    return agent


def execute(agent, query, **context):
    return asyncio.run(agent.invoke_messages([HumanMessage(content=query)],
                       {"thread_id": "workflow", "uid": "u", **context}))


def test_agent_search_observe_read_then_answer(tmp_path, caplog):
    calls = []
    agent = make_agent(tmp_path, [call("search", {"query": "policy"}),
        call("read", {"query": "https://example.com/policy"}, "call-2"), AIMessage(content="Verified policy")])
    def tool(name):
        def run(query):
            calls.append((name, query))
            return "actual policy" if name == "read" else "https://example.com/policy"
        return StructuredTool.from_function(func=run, name=name, description=name, args_schema=QueryInput)
    agent.workflow_tools = lambda state: {name: tool(name) for name in ("search", "read")}
    with caplog.at_level("INFO"):
        result = execute(agent, "Search and read policy", web_search=True)
    assert result["final_answer"] == "Verified policy"
    assert [c[0] for c in calls] == ["search", "read"]
    assert "actual policy" in agent.llm.histories[-1][-1].content
    assert isinstance(agent.llm.histories[-1][-1], ToolMessage)
    assert '"tool": "read"' in caplog.text
    assert result["decision_round"] == 3


def test_no_search_question_has_no_tool_calls(tmp_path, caplog):
    agent = make_agent(tmp_path, [AIMessage(content="你好")])
    with caplog.at_level("INFO"):
        result = execute(agent, "你好", web_search=True)
    assert result["final_answer"] == "你好"
    assert result["tool_evidence"] == []
    assert '"tools": []' in caplog.text


@pytest.mark.parametrize("expected", ["disabled", "success", "no_results", "read_failed", "request_failed"])
def test_chat_stream_executes_selected_live_search_even_if_model_skips_tools(tmp_path, monkeypatch, expected):
    from fastapi.testclient import TestClient
    from requests import Response, Timeout
    from services.tools import kb_tools
    import web.main as main

    url = "https://fgk.chinatax.gov.cn/zcfgk/c100012/c12345/content.html"
    policy = "本公告规定了测试用政策正文，适用条件以公开文件列明的范围为准。"
    calls = []

    def get(url_requested, **kwargs):
        calls.append(url_requested)
        if expected == "request_failed":
            raise Timeout()
        response = Response()
        response.status_code = 200
        response.encoding = "utf-8"
        response._content_consumed = True
        if url_requested.endswith("/search/s"):
            rows = [] if expected == "no_results" else [{"url": url, "title": "测试政策公告"}]
            response._content = json.dumps({"searchResultAll": {"searchTotal": rows}}).encode()
        elif url_requested == url:
            response.headers["Content-Type"] = "text/html"
            body = f'<div class="arc_cont">{policy}</div>' if expected == "success" else "<body>只有网站导航</body>"
            response._content = ('<meta name="PubDate" content="2026-04-07">' + body).encode()
        elif kwargs["params"].get("format") == "rss":
            response._content = b"<rss><channel></channel></rss>"
        else:
            response._content = b'<ul id="b_results"><li class="b_no">No results</li></ul>'
        return response

    agent = make_agent(tmp_path, [AIMessage(content="根据已取得的资料回答。")])
    monkeypatch.setattr(main, "tax_agent", agent)
    monkeypatch.setattr(main, "conversation_memory", None)
    monkeypatch.setattr(kb_tools.requests, "get", get)
    query = "小型微利企业有什么企业所得税优惠？"
    response = TestClient(main.app).post("/api/chat/stream", headers={"Authorization": USER_BEARER},
        json={"user_id": "web_user", "query": query, "session_id": "live-policy", "web_search": expected != "disabled"})
    assert response.status_code == 200, response.text
    assert f'"web_search_status": "{expected}"' in response.text
    assert bool(calls) == (expected != "disabled")
    assert (url in calls) == (expected in ("success", "read_failed"))
    model_input = " ".join(message.content for message in agent.llm.histories[0])
    assert agent.llm.histories[0][-1].content == query
    assert (policy in model_input) == (expected == "success")
    assert "只有网站导航" not in model_input
    if expected == "success":
        assert "2026-04-07" in response.text and url in response.text
    elif expected != "disabled":
        assert "联网检索未成功，本次回答仅参考本地知识库。" in response.text


def test_long_live_policy_keeps_applicability_and_isolated_multi_turn_context(tmp_path, monkeypatch, caplog):
    from agents import streaming
    from agents.policy_context import WEB_POLICY
    from services.tools import kb_tools
    from langchain_core.utils.function_calling import convert_to_openai_tool

    monkeypatch.setenv("LLM_CONTEXT_WINDOW", "8192")
    monkeypatch.setenv("LLM_MAX_OUTPUT_TOKENS", "1536")
    monkeypatch.setattr(streaming, "count_tokens", lambda text, name="": (len(str(text).encode()), "utf8_test_estimate"))
    query = "小型微利企业有什么企业所得税优惠？"
    url = "https://fgk.chinatax.gov.cn/zcfgk/c100012/c12345/content.html"
    clause = "小型微利企业所得税优惠须同时符合公告所列条件。"
    condition = "同时符合从业人数、资产总额、年度应纳税所得额限制。"
    secret_history = "PRIVATE_TEST_HISTORY"
    agent = make_agent(tmp_path, [AIMessage(content=f"已取得测试原文，适用条件见 {url}")] * 3)
    agent.get_memory_context.return_value = {"prompt_text": secret_history * 1000}
    model = agent.llm
    model.bind_tools = lambda tools, **kwargs: SimpleNamespace(
        bound=model, kwargs={"tools": [convert_to_openai_tool(t) for t in tools]}, invoke=model.invoke)
    # Keep this test's provider-independent estimate small enough for realistic evidence space.
    tool = StructuredTool.from_function(func=lambda query: "{}", name="search_tax_policy",
                                        description="查找政策", args_schema=QueryInput)
    agent.workflow_tools = lambda state: {tool.name: tool}
    search = MagicMock(return_value=("success", [{"url": url, "snippet": "SEARCH_SUMMARY_ONLY"}]))
    read = MagicMock(return_value={"status": "success", "url": url, "title": "小型微利企业所得税优惠",
                                  "document_number": "测试文号", "issuer": "测试机关",
                                  "publish_date": "2026-01-01", "effective_date": "2026年2月1日",
                                  "content": "无关背景资料。\n" * 140 + clause + "\n" + condition + "\n" + "其他内容。\n" * 200,
                                  "truncated": True})
    monkeypatch.setattr(kb_tools, "search_live_policy_candidates", search)
    monkeypatch.setattr(kb_tools, "read_web_page", read)

    async def chat():
        return [await agent.run(query, {"thread_id": "workflow", "web_search": enabled})
                for enabled in (True, True, False)]

    with caplog.at_level("INFO"):
        results = asyncio.run(chat())
    assert search.call_count == read.call_count == 2
    assert url in results[0] and "部分官网正文因上下文预算" in results[0]
    assert "自我校验未完成" not in results[0]
    assert secret_history not in caplog.text and clause not in caplog.text and "SEARCH_SUMMARY_ONLY" not in caplog.text
    for history in model.histories[:2]:
        records = [json.loads(m.content) for m in history if m.additional_kwargs.get("context_kind") == WEB_POLICY]
        assert len(records) == 1
        record = records[0]
        assert record["url"] == url and record["effective_date"] == "2026-02-01"
        assert record["document_number"] == "测试文号" and record["issuer"] == "测试机关"
        assert record["publish_date"] == "2026-01-01" and "证据不完整" in record["context_notice"]
        assert clause in record["content"] and condition in record["content"]
        assert history[-1].content == query
    assert not any(m.additional_kwargs.get("context_kind") == WEB_POLICY for m in model.histories[-1])
    assert any(clause in m.content for m in model.review_histories[0])
    assert agent.record_turn.call_args.kwargs["user_message"] == query


def test_deepseek_official_chat_sends_reserved_output_as_max_tokens(tmp_path):
    import httpx
    from langchain_openai import ChatOpenAI

    requests = []
    def reply(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"id": "test", "model": "deepseek-flash",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "答复"}, "finish_reason": "stop"}]})

    agent = make_agent(tmp_path, [])
    with httpx.Client(transport=httpx.MockTransport(reply)) as client:
        agent.llm = ChatOpenAI(model="deepseek-flash", api_key="test-only",
                               base_url="https://api.deepseek.com/v1", http_client=client)
        response = agent._invoke_chat(agent.llm, [HumanMessage(content="短问题")], 256)
    assert response.content == "答复"
    assert requests[0]["max_tokens"] == 256
    assert "max_completion_tokens" not in requests[0]


def test_web_reflection_gets_citation_diagnostics_before_publishing(tmp_path):
    url = "https://fgk.chinatax.gov.cn/zcfgk/c100012/c12345/content.html"
    bad_draft = "《测试公告》第二条列明了优惠范围。"
    corrected = f"测试公告的原文编号为二、，请核对官网正文 {url}"
    agent = make_agent(tmp_path, [AIMessage(content=bad_draft)])
    agent.llm.reviews = iter([{"ok": False, "issues": ["条号格式与原文不符"], "corrected_answer": corrected}])
    result = execute(agent, "查询测试公告", web_search=True, web_policy={"status": "success", "sources": [{
        "page_status": "success", "title": "测试公告", "url": url, "content": "二、优惠范围以已列明的条件为准。",
    }]})
    review = json.loads(agent.llm.review_histories[0][-1].content)
    assert review["unmatched_citations"] == ["第二条"]
    assert result["final_answer"] == corrected and result["guardrails"]["ok"]


@pytest.mark.parametrize("schema_size", [3550, 4350])
def test_policy_receipt_fits_budget_and_duplicate_search_reuses_result(tmp_path, monkeypatch, schema_size):
    from agents import streaming
    monkeypatch.setattr(streaming, "count_tokens", lambda text, name="": (len(str(text).encode()), "test"))
    monkeypatch.setenv("LLM_CONTEXT_WINDOW", "8192")
    monkeypatch.setenv("LLM_MAX_OUTPUT_TOKENS", "1536")
    names = []
    def search(query):
        names.append(query)
        return json.dumps({"found": 5, "results": [
            {"title": f"政策 {i}", "url": f"https://example.com/{i}", "content": "政策原文" * 700,
             "articles": ["长条文索引"] * 500} for i in range(5)]}, ensure_ascii=False)
    tool = StructuredTool.from_function(func=search, name="search_tax_policy", description="政策检索", args_schema=QueryInput)
    responses = [call(tool.name, {"query": "输入税额"})]
    if schema_size == 3550:
        responses.append(call(tool.name, {"query": "  输入税额 "}, "second"))
    responses.append(AIMessage(content="已核对资料，仍需核对具体适用条件。"))
    agent = make_agent(tmp_path, responses)
    agent.max_tool_rounds = 3
    agent.workflow_tools = lambda state: {tool.name: tool}
    model = agent.llm
    model.bind_tools = lambda tools, **kwargs: SimpleNamespace(
        bound=model, kwargs={"tools": [{"description": "x" * schema_size}]}, invoke=model.invoke)
    result = execute(agent, "核对输入税额")
    assert names == ["输入税额"]
    assert result["final_answer"].startswith("已核对资料")
    assert len(result["tool_evidence"]) == 1
    reply = model.histories[1][-1]
    assert isinstance(reply, ToolMessage) and reply.tool_call_id == "call-1"
    observed = json.loads(reply.content)
    assert observed["context_notice"] and observed["results"][0]["url"] == "https://example.com/0"


def test_chat_sse_budget_error_is_terminal_and_actionable(monkeypatch):
    from fastapi.testclient import TestClient
    from agents.streaming import ContextBudgetError
    import web.main as main

    mock = AsyncMock(side_effect=ContextBudgetError("请缩小问题或分批查询。"))
    monkeypatch.setattr(main, "tax_agent", SimpleNamespace(run=mock))
    monkeypatch.setattr(main, "conversation_memory", None)
    response = TestClient(main.app).post("/api/chat/stream", headers={"Authorization": USER_BEARER},
                                         json={"user_id": "test", "query": "请核对进项税额", "session_id": "budget"})
    assert response.status_code == 200, response.text
    assert "event: error" in response.text and "event: done" not in response.text
    assert "context_budget" in response.text and "请缩小问题" in response.text


def test_round_limit_stops_even_if_model_keeps_calling(tmp_path):
    tool = StructuredTool.from_function(func=lambda query: "result", name="repeat", description="repeat", args_schema=QueryInput)
    agent = make_agent(tmp_path, [call("repeat", {"query": "x"}, str(i)) for i in range(3)])
    agent.workflow_tools = lambda state: {"repeat": tool}
    result = execute(agent, "repeat forever")
    assert len(result["tool_evidence"]) == 2
    assert "轮数上限" in result["final_answer"]
    assert any(e.get("limited") is True for e in result["workflow_trace"])


def test_tool_errors_are_structured_observations(tmp_path):
    agent = make_agent(tmp_path, [call("not_allowed", {"query": "x"}), AIMessage(content="工具不可用")])
    result = execute(agent, "x")
    evidence = result["tool_evidence"][0]
    assert evidence["status"] == "error"
    assert json.loads(evidence["output"])["error_type"] == "ValueError"
    assert "工具不可用" == result["final_answer"]


def test_llm_can_select_original_specialist_path(tmp_path, caplog):
    from agents.intent_router import HybridIntentRouter
    agent = make_agent(tmp_path, [call("consult_specialists", {"reason": "需要申报指导"})])
    agent.router = HybridIntentRouter()
    child = SimpleNamespace(run=AsyncMock(return_value="原申报路径仍可回答"))
    agent.registry = SimpleNamespace(get=lambda name: child)
    with caplog.at_level("INFO"):
        result = execute(agent, "增值税怎么申报")
    assert result["final_answer"] == "原申报路径仍可回答"
    child.run.assert_awaited()
    assert result["collaboration_chain"] == ["filing_agent"]
    assert '"branch": "specialists"' in caplog.text
    assert any(e["node"] == "route" for e in result["workflow_trace"])


def test_web_checkbox_controls_tool_permissions(tmp_path):
    agent = make_agent(tmp_path, [])
    assert "read_web_page" not in agent.workflow_tools({"web_search": False})
    assert "read_web_page" in agent.workflow_tools({"web_search": True})


@pytest.mark.parametrize("approved", [True, False])
def test_approval_pause_restart_resume_and_no_replay(tmp_path, approved):
    from agents.base_agent import AsyncSqliteSaver
    from services.security import tenant_scope
    async def verify():
        path = str(tmp_path / "approval.sqlite")
        with tenant_scope("u"):
            async with AsyncSqliteSaver.from_conn_string(path) as saver:
                await saver.setup()
                agent = make_agent(tmp_path, [call("generate_filing_file", {"title": "测试申报辅助", "content": "应纳税额5000元"})])
                agent._checkpointer = saver
                result = await agent.invoke_messages([HumanMessage(content="导出申报文件")], {"thread_id": "approval", "uid": "u"})
                assert result["__interrupt__"][0].value["kind"] == "approval"
                snapshot = await agent.workflow_state("approval")
                assert not snapshot.values.get("artifacts")
                pending = await agent.pending_workflow("approval")
            # 新连接、新Agent，恢复依靠持久化状态，而不是进程内变量。
            async with AsyncSqliteSaver.from_conn_string(path) as saver:
                restarted = make_agent(tmp_path, [])
                restarted._checkpointer = saver
                assert await restarted.pending_workflow("approval") == pending
                for invalid in ({"id": "stale", "value": True}, {"id": pending["id"], "value": "yes"}):
                    with pytest.raises(ValueError):
                        await restarted.invoke_messages([HumanMessage(content="approve")], {"thread_id": "approval", "resume": invalid})
                result = await restarted.invoke_messages([HumanMessage(content="approve")],
                         {"thread_id": "approval", "resume": {"id": pending["id"], "value": approved}})
                assert not await restarted.pending_workflow("approval")
                assert bool(result.get("artifacts")) == approved
                assert result["workflow_trace"][-1]["node"] == ("export" if approved else "approval")
                if not approved:
                    assert "拒绝" in result["final_answer"]
                with pytest.raises(ValueError, match="失效"):
                    await restarted.invoke_messages([HumanMessage(content="approve")],
                        {"thread_id": "approval", "resume": {"id": pending["id"], "value": True}})
        with tenant_scope("another-user"):
            async with AsyncSqliteSaver.from_conn_string(path) as saver:
                restarted = make_agent(tmp_path, [])
                restarted._checkpointer = saver
                assert not await restarted.pending_workflow("approval")
                snapshot = await restarted.workflow_state("approval")
                assert not snapshot.values
    asyncio.run(verify())


def test_approval_without_durable_store_is_visible_error(tmp_path):
    agent = make_agent(tmp_path, [call("generate_filing_file", {"title": "draft", "content": "data"}), AIMessage(content="持久化不可用，文件未生成")])
    result = execute(agent, "export")
    assert not result.get("artifacts")
    assert json.loads(result["tool_evidence"][0]["output"])["error_type"] == "RuntimeError"


def test_approval_api_requires_real_boolean_and_owned_session(tmp_path, monkeypatch):
    import web.main as main
    from fastapi.testclient import TestClient
    monkeypatch.setenv("AUTH_USER_TOKEN", "test_token_123")
    memory = MagicMock()
    memory.get_session.return_value = None
    monkeypatch.setattr(main, "conversation_memory", memory)
    client = TestClient(main.app)
    headers = {"Authorization": USER_BEARER}
    assert client.post("/api/conversations/private/resume", headers=headers,
                       json={"id": "pause", "value": "yes"}).status_code == 422
    assert client.post("/api/conversations/private/resume", headers=headers,
                       json={"id": "pause", "value": True}).status_code == 404
    assert client.get("/api/conversations/private/artifacts/x", headers=headers).status_code == 404


@pytest.mark.parametrize("critic_catches", [True, False])
def test_reflection_corrects_bad_vat_even_if_critic_misses(tmp_path, critic_catches):
    args = {"taxpayer_type": "general", "sales_amount": 100000, "sales_is_inclusive": False,
            "input_tax": 8000, "vat_rate": 0.13}
    agent = make_agent(tmp_path, [call("calculate_vat", args), AIMessage(content="应纳税额为6000元")])
    agent.llm.reviews = iter([{"ok": not critic_catches, "issues": ["计算错误"] if critic_catches else [],
        "corrected_answer": "应纳税额为5000元；100000×13%-8000=5000。" if critic_catches else ""}])
    result = execute(agent, "100000×13%-8000等于6000吗？请算应纳税额")
    assert "5000" in result["final_answer"]
    assert "6000元" not in result["final_answer"]
    assert result["reflection"]["changed"]
    assert result["reflection"]["calculator_checks"] == 1
    assert result["messages"][0].content.startswith("100000")
    assert result["messages"][-1].content == result["final_answer"]
    assert not any(isinstance(m, AIMessage) and "6000元" in m.content for m in result["messages"])


def test_reflection_normal_answer_keeps_exact_text(tmp_path):
    text = "应纳税额为5000元。"
    agent = make_agent(tmp_path, [call("calculate_vat", {"sales_amount": 100000, "input_tax": 8000, "vat_rate": 0.13}), AIMessage(content=text)])
    result = execute(agent, "一般纳税人，不含税销售100000，进项8000，税率13%，算增值税")
    assert result["final_answer"] == text
    assert result["reflection"]["changed"] is False


def test_reflection_rejects_hallucinated_policy_and_invalid_correction(tmp_path):
    agent = make_agent(tmp_path, [AIMessage(content="虚构的第9999号法规允许全部免税。")])
    agent.llm.reviews = iter([{"ok": False, "issues": ["不存在的法规依据"], "corrected_answer": "未检索到该法规，不能据此确定免税，请提供政策原文核验。"}])
    result = execute(agent, "第9999号法规是不是让我全部免税？")
    assert "不能据此确定" in result["final_answer"]
    assert result["reflection"]["changed"]
    agent = make_agent(tmp_path, [AIMessage(content="税務草稿")])
    agent.llm.reviews = iter([{"ok": False, "issues": ["缺少来源"], "corrected_answer": "依据 https://invented.invalid/policy 免税。"}])
    result = execute(agent, "免税吗？")
    assert "未通过核验" in result["final_answer"]
    assert "invented.invalid" not in result["final_answer"]


def test_reflection_network_failure_does_not_claim_passed(tmp_path):
    agent = make_agent(tmp_path, [AIMessage(content="未经核验的税务结论")])
    agent.llm.reviews = iter([ConnectionError("simulated critic outage")])
    result = execute(agent, "税务问题")
    assert "未通过核验" in result["final_answer"]
    assert result["reflection"]["ok"] is False


def test_clarification_interrupt_restart_then_calculation_keeps_state(tmp_path):
    from agents.base_agent import AsyncSqliteSaver
    from services.security import tenant_scope
    async def verify():
        path = str(tmp_path / "clarification.sqlite")
        with tenant_scope("u"):
            async with AsyncSqliteSaver.from_conn_string(path) as saver:
                await saver.setup()
                agent = make_agent(tmp_path, [call("ask_clarification", {"question": "请提供税种、收入、税率和扣除？"})])
                agent._checkpointer = saver
                first = await agent.invoke_messages([HumanMessage(content="帮我算税")], {"thread_id": "clarify", "uid": "u"})
                assert first["__interrupt__"][0].value["kind"] == "clarification"
                assert not first.get("final_answer")
                assert first["original_query"] == "帮我算税"
                assert first["decision_round"] == 1
            async with AsyncSqliteSaver.from_conn_string(path) as saver:
                agent = make_agent(tmp_path, [call("calculate_vat", {"sales_amount": 100000, "input_tax": 8000, "vat_rate": 0.13}, "vat"), AIMessage(content="应纳税额为5000元。")])
                agent._checkpointer = saver
                supplement = "2025年一般纳税人增值税，不含税销售100000元，税率13%，进项8000元"
                result = await agent.invoke_messages([HumanMessage(content=supplement)], {"thread_id": "clarify", "uid": "u", "web_search": True, "source_types": ["default"]})
                assert result["final_answer"] == "应纳税额为5000元。"
                assert result["original_query"] == "帮我算税\n用户补充：" + supplement
                assert result["decision_round"] == 3
                assert result["web_search"] is True
                assert result["source_types"] == ["default"]
                assert result["query_date"].startswith("2025")
                assert result["messages"][0].content == "帮我算税"
                assert result["messages"][1].content == supplement
                assert any(e["node"] == "clarification" and e["resumed"] for e in result["workflow_trace"])
                assert not await agent.pending_workflow("clarify")
                assert agent.llm.histories[0][1].content == "帮我算税"
    asyncio.run(verify())


def test_complex_plan_executes_in_order_and_uses_previous_result(tmp_path):
    vat = {"sales_amount": 100000, "input_tax": 8000, "vat_rate": 0.13}
    plan = {"steps": [{"goal": "算增值税", "tool": "calculate_vat"},
                      {"goal": "以增值税实际缴纳额计算市区附加税", "tool": "calculate_surcharges"}]}
    def dependent(history):
        completed = json.loads(history[-1].content)["completed"]
        assert json.loads(completed[0]["output"])["tax_amount"] == 5000
        return call("calculate_surcharges", {"vat_paid": 5000, "city_type": "urban"}, "surcharge")
    agent = make_agent(tmp_path, [call("plan_and_execute", {"objective": "增值税然后附加税"}),
        AIMessage(content=json.dumps(plan, ensure_ascii=False)), call("calculate_vat", vat, "vat"), dependent,
        AIMessage(content="增值税5000元，附加税600元。")])
    result = execute(agent, "算增值税，再依据增值税计算市区附加税；一般纳税人不含税销售100000，进项8000，税率13%")
    assert result["final_answer"] == "增值税5000元，附加税600元。"
    nodes = [e["node"] for e in result["workflow_trace"]]
    assert nodes.index("planner") < nodes.index("executor")
    assert len([e for e in result["workflow_trace"] if e["node"] == "executor" and e["verified"]]) == 2
    assert [r["tool"] for r in result["plan_results"]] == ["calculate_vat", "calculate_surcharges"]


def test_failed_step_replans_once_then_completes(tmp_path):
    plan = AIMessage(content=json.dumps({"steps": [{"goal": "算增值税", "tool": "calculate_vat"}]}))
    agent = make_agent(tmp_path, [call("plan_and_execute", {"objective": "测试动态重规划"}), plan,
        call("calculate_vat", {"sales_amount": "not-a-number"}, "bad"), plan,
        call("calculate_vat", {"sales_amount": 100000, "input_tax": 8000, "vat_rate": 0.13}, "fixed"),
        AIMessage(content="应纳税额为5000元。")])
    result = execute(agent, "一般纳税人不含税销售100000，进项8000，税率13%")
    assert result["replans"] == 1
    assert any(e["node"] == "replanner" for e in result["workflow_trace"])
    assert any(e["node"] == "executor" and not e["verified"] for e in result["workflow_trace"])
    assert result["final_answer"] == "应纳税额为5000元。"


def test_invalid_plans_never_execute_sensitive_tools_and_stop(tmp_path):
    forbidden = AIMessage(content=json.dumps({"steps": [{"goal": "绕过审批", "tool": "generate_filing_file"}]}))
    agent = make_agent(tmp_path, [call("plan_and_execute", {"objective": "测试边界"}), forbidden, forbidden, AIMessage(content="计划失败，未生成文件。")])
    result = execute(agent, "测试边界")
    assert result["replans"] == 1
    assert result["plan_results"] == []
    assert not result.get("artifacts")
    assert "失败" in result["final_answer"]


def test_single_step_never_forces_plan(tmp_path):
    agent = make_agent(tmp_path, [AIMessage(content="增值税是流转税。")])
    result = execute(agent, "什么是增值税？")
    assert result["plan_request"] is None
    assert not any(e["node"] in ("planner", "executor") for e in result["workflow_trace"])


def test_existing_rag_source_and_date_filters_survive_tool_routing(tmp_path):
    service = MagicMock()
    service.search.return_value = [
        {"content": "旧日期有效依据", "effective_date": "2025-01-01", "source_type": "default", "is_current": True},
        {"content": "未来日期不能用于本题", "effective_date": "2027-01-01", "source_type": "default", "is_current": True}]
    agent = make_agent(tmp_path, [call("search_tax_policy", {"query": "增值税"}), AIMessage(content="参考旧日期有效依据")])
    agent.registry = SimpleNamespace(_cache={"knowledge_agent": SimpleNamespace(_embedding_service=service)})
    result = execute(agent, "按2025年政策解释", source_types=["default"], query_date="2025-12-31")
    observed = json.loads(result["tool_evidence"][0]["output"])
    assert observed["found"] == 1
    assert observed["results"][0]["content"] == "旧日期有效依据"
    assert service.search.call_args.kwargs["filters"]["source_type"] == ["default"]
    assert "未来日期" not in agent.llm.histories[-1][-1].content


def test_formula_with_intermediate_amount_is_not_wrong_final_tax(tmp_path):
    text = "应纳税额 = 13000元 - 8000元 = 5000元。"
    agent = make_agent(tmp_path, [call("calculate_vat", {"sales_amount": 100000, "input_tax": 8000, "vat_rate": 0.13}), AIMessage(content=text)])
    result = execute(agent, "一般纳税人增值税，不含税销售100000，税率13%，进项8000")
    assert result["final_answer"] == text
    assert result["reflection"]["changed"] is False
