"""发请求前的实际预算、重要历史保留、工具完整性与告警。"""
from types import SimpleNamespace
from unittest.mock import MagicMock
import json
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from agents.streaming import prepare_context, bounded_invoke, context_reports, response_events


@pytest.fixture
def budget(monkeypatch):
    monkeypatch.setenv("LLM_CONTEXT_WINDOW", "1400")
    monkeypatch.setenv("LLM_MAX_OUTPUT_TOKENS", "256")
    monkeypatch.setenv("LLM_CONTEXT_WARN_RATIO", "0.8")
    reports = []
    token = context_reports.set(reports)
    yield reports
    context_reports.reset(token)


def test_overflow_trims_before_model_call_and_warns(budget):
    seen = []
    model = SimpleNamespace(model_name="test", max_tokens=None, invoke=lambda messages, **kw: seen.append((messages, kw)))
    messages = [SystemMessage(content="必须遵循用户给定条件")]
    messages += [HumanMessage(content="无关背景" * 100), AIMessage(content="旧聊天" * 100)] * 8
    messages += [HumanMessage(content="增值税税率13%，计算销售100000元")]
    events = []
    token = response_events.set(lambda event, data: events.append((event, data)))
    try:
        bounded_invoke(model, messages)
    finally:
        response_events.reset(token)
    assert len(seen) == 1
    assert seen[0][0][0].content == messages[0].content and seen[0][0][-1].content == messages[-1].content
    assert seen[0][1]["max_tokens"] == 256
    assert budget[0]["trimmed_groups"] > 0 and budget[0]["after_tokens"] <= budget[0]["input_budget"]
    assert events and "预算预警" in events[0][1]["text"]


def test_old_important_fact_retained_over_new_unrelated_message(budget):
    important = HumanMessage(content="一般纳税人销售100000元，税率13%，进项8000元，必须以此计算")
    irrelevant = HumanMessage(content="聊一下天气吧。" * 160)
    messages = [important, irrelevant, AIMessage(content="天气相关内容。" * 160), HumanMessage(content="按前面税率和进项计算增值税")]
    selected, _ = prepare_context(SimpleNamespace(), messages)
    assert important in selected
    assert len(selected) < len(messages)


def test_tool_call_group_kept_or_removed_as_whole(budget):
    call = AIMessage(content="", tool_calls=[{"name": "old_tool", "args": {}, "id": "old"}])
    reply = ToolMessage(content="旧工具资料" * 400, tool_call_id="old")
    selected, _ = prepare_context(SimpleNamespace(), [HumanMessage(content="旧问题"), call, reply,
                                                    HumanMessage(content="当前问题")])
    assert (call in selected) == (reply in selected)
    assert call not in selected


def test_latest_tool_result_and_current_question_retained(budget):
    call = AIMessage(content="", tool_calls=[{"name": "vat", "args": {}, "id": "vat"}])
    reply = ToolMessage(content='{"tax_amount":5000}', tool_call_id="vat")
    query = HumanMessage(content="算增值税")
    selected, _ = prepare_context(SimpleNamespace(), [query, call, reply])
    assert selected == [query, call, reply]


@pytest.mark.parametrize("messages", [
    [ToolMessage(content="oops", tool_call_id="missing")],
    [AIMessage(content="", tool_calls=[{"name": "x", "args": {}, "id": "missing"}])],
])
def test_invalid_tool_history_never_sent(budget, messages):
    model = SimpleNamespace(invoke=MagicMock())
    with pytest.raises(ValueError, match="回执"):
        bounded_invoke(model, messages)
    model.invoke.assert_not_called()


def test_oversized_pinned_question_visible_error_and_no_request(budget):
    model = SimpleNamespace(invoke=MagicMock())
    with pytest.raises(ValueError, match="未发送"):
        bounded_invoke(model, [SystemMessage(content="安全约束"), HumanMessage(content="超长当前问题" * 5000)])
    model.invoke.assert_not_called()


def test_memory_background_can_be_trimmed_without_losing_current_question(budget):
    enriched = HumanMessage(content="[历史对话背景]\n" + "历史摘要" * 5000 + "\n[用户当前问题]\n销售100000税率13%进项8000，算税")
    selected, _ = prepare_context(SimpleNamespace(), [SystemMessage(content="不得编造条件"), enriched])
    assert selected[-1].content == "销售100000税率13%进项8000，算税"
    assert budget[-1]["trimmed_groups"] + budget[-1]["clipped_observations"] == 1
    assert budget[-1]["after_tokens"] < budget[-1]["before_tokens"] and budget[-1]["after_tokens"] <= budget[-1]["input_budget"]


def test_large_latest_search_result_clips_body_but_preserves_source_and_pairing(budget):
    payload = {"results": [{"content": "很长的政策原文" * 5000, "title": "增值税", "url": "https://example.com/policy", "document_number": "财税〔2024〕12号"}], "found": 1}
    call = AIMessage(content="", tool_calls=[{"name": "search", "args": {}, "id": "search"}])
    reply = ToolMessage(content=json.dumps(payload, ensure_ascii=False), tool_call_id="search")
    selected, _ = prepare_context(SimpleNamespace(), [HumanMessage(content="核对增值税"), call, reply])
    observed = json.loads(selected[-1].content)
    assert selected[-2] == call and selected[-1].tool_call_id == "search"
    assert observed["results"][0]["url"] == payload["results"][0]["url"]
    assert observed["results"][0]["document_number"] == payload["results"][0]["document_number"]
    assert observed["results"][0]["context_truncated"] and observed["context_notice"]
    assert json.loads(reply.content) == payload
    assert budget[-1]["after_tokens"] <= budget[-1]["input_budget"]


def test_optional_structured_evidence_is_never_cut_into_invalid_json(budget):
    current = HumanMessage(content="核对增值税")
    evidence = HumanMessage(content=json.dumps({"evidence": "增值税" * 2000}, ensure_ascii=False),
                            additional_kwargs={"context_optional": True, "context_atomic": True})
    selected, _ = prepare_context(SimpleNamespace(), [evidence, current])
    assert selected == [current]
    assert budget[-1]["trimmed_groups"] == 1 and budget[-1]["clipped_observations"] == 0


def test_optional_document_keeps_matching_tax_condition_and_current_question(budget):
    evidence = SystemMessage(content="【文件】vat.md\n" + "无关背景\n" * 2000 + "增值税率13%，进项8000元。\n",
                             additional_kwargs={"context_optional": True})
    current = HumanMessage(content="按增值税率13%和进项8000元计算")
    selected, _ = prepare_context(SimpleNamespace(), [SystemMessage(content="不得编造条件"), evidence, current])
    assert selected[-1] == current
    assert any("进项8000元" in str(m.content) and "节选" in str(m.content) for m in selected[:-1])
    assert budget[-1]["clipped_observations"] == 1


def test_tool_schema_consumes_input_budget(budget):
    model = SimpleNamespace(kwargs={"tools": [{"description": "说明" * 30}]})
    prepare_context(model, [HumanMessage(content="你好")])
    assert budget[-1]["tool_schema_tokens"] > 20
    assert budget[-1]["input_budget"] + budget[-1]["output_reserve"] + budget[-1]["tool_schema_tokens"] + 128 == 1400


def test_count_is_labeled_estimate_and_normal_history_unchanged(budget):
    messages = [SystemMessage(content="规则"), HumanMessage(content="你好")]
    selected, _ = prepare_context(SimpleNamespace(model_name="unknown-custom-model"), messages)
    assert selected == messages and budget[-1]["trimmed_groups"] == 0
    assert "estimate" in budget[-1]["count_method"]


def test_long_critic_evidence_trimmed_without_losing_draft_or_recalculation(budget):
    payload = {"query": "核对增值税", "draft": "应纳税额5000元。", "recalculated": [{"tax_amount": 5000}],
               "evidence": [{"tool": "search_tax_policy", "output": json.dumps({"results": [{"title": "增值税法", "content": "增值税原文" * 5000}]}, ensure_ascii=False)}],
               "specialist_responses": []}
    selected, _ = prepare_context(SimpleNamespace(), [SystemMessage(content="审校系统约束不可裁剪"), HumanMessage(content=json.dumps(payload, ensure_ascii=False))])
    required = json.loads(selected[-1].content)
    assert required["draft"] == payload["draft"] and required["recalculated"] == payload["recalculated"]
    assert required["query"] == payload["query"] and required["evidence_notice"]
    assert budget[-1]["after_tokens"] <= budget[-1]["input_budget"] and budget[-1]["warning"]
    assert any("实际工具资料" in str(message.content) for message in selected[:-1])
