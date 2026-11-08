"""Research regressions with isolated SQLite and scripted transport only."""
import asyncio
import json
import threading
from types import SimpleNamespace

import httpx
import pytest
import requests
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from sqlalchemy import create_engine

from agents.research_agent import ARRAY_OUTPUT_FIELDS, ResearchAgent, Understanding
from agents.research_output import InvalidOutput, parse_output
from services.research_evidence import normalize_evidence, official_url
from services.research_store import ResearchStore
from services.security import tenant_scope
from services.tools import kb_tools
from tests.test_research import POLICY, ScriptedModel, assess_covered, covered, report, understanding


QUERY = "我们企业买了一套二手房，想问一下，该如何计提折旧？"
REPLY = "二手房购入后是自用办公"
QUESTIONS = ["二手房折旧如何处理", "不同使用方式的影响", "折旧何时起算"]
PENDING = ["用于办公、生产经营、出租还是闲置？", "何时购入并投入使用？", "入账原值是多少？"]


def make_store(tmp_path):
    return ResearchStore(create_engine(f"sqlite:///{tmp_path / 'research.db'}"))


@pytest.fixture(autouse=True)
def offline_tokenizer(monkeypatch):
    def count(text, name=""):
        return len(str(text)), "test"
    monkeypatch.setattr("agents.streaming.count_tokens", count)
    monkeypatch.setattr("agents.research_agent.count_tokens", count)


@pytest.mark.parametrize("wrap", [
    lambda body: AIMessage(content=body),
    lambda body: AIMessage(content="\ufeff```json\n" + body + "\n```"),
    lambda body: AIMessage(content="提取结果如下。\n" + body + "\n以上。"),
    lambda body: AIMessage(content=[{"type": "reasoning", "reasoning": '{"untrusted":true}'},
                                    {"type": "text", "text": body[:5]},
                                    {"type": "text", "text": body[5:]}]),
    lambda body: {"content": [{"type": "output_text", "text": body}]},
    lambda body: {"content": json.loads(body)},
    lambda body: AIMessage(content='{"type":"object","properties":' + body + '}'),
])
def test_model_response_instances_and_compatible_wrappers(wrap):
    value = {"questions": ["核对折旧"], "facts": ["购入二手房"]}
    assert parse_output(wrap(json.dumps(value, ensure_ascii=False)), Understanding)["facts"] == value["facts"]


@pytest.mark.parametrize("body,finish,code", [
    ("{malformed}", None, "output_json_invalid"),
    ('{"questions":["核对折旧"],}', None, "output_json_invalid"),
    ('{"questions":["A"]} {"questions":["B"]}', None, "output_json_invalid"),
    ('{"questions":["A"],"questions":["B"]}', None, "output_json_invalid"),
    ('[{"questions":["A"]}]', None, "output_json_invalid"),
    ('{"facts":["原话"]}', None, "output_schema_invalid"),
    ('{"questions":[1]}', None, "output_schema_invalid"),
    ('{"questions":["核对折旧"],"facts":"原话","unknown_kinds":{"购入日期？":"日期"}}', None, "output_schema_invalid"),
    ('{"questions":["核对折旧"],"unknowns":"购入日期？"}', None, "output_schema_invalid"),
    ('{"properties":{"questions":["核对折旧"],"facts":{"type":"array","items":["原话"]}}}',
     None, "output_schema_invalid"),
    ("   ", "stop", "output_empty"),
    ('{"questions":["A"]}', "length", "output_truncated"),
    ('{"questions":["A"]}', "content_filter", "output_refused"),
])
def test_output_rejects_invalid_instances_without_filling_fields(body, finish, code):
    with pytest.raises(InvalidOutput) as exc:
        parse_output(AIMessage(content=body), Understanding, finish_reason=finish)
    assert exc.value.code == code


@pytest.mark.parametrize("wrapper", [
    {"type": "array", "items": {"type": "string"}},
    {"type": "array", "items": ["原话"], "default": ["另一个值"]},
    {"type": "array", "items": ["原话", {"value": "推测"}]},
    {"type": "array", "items": ["原话"], "maxItems": 0},
])
def test_fact_repair_rejects_ambiguous_arrays(wrapper):
    body = json.dumps({"facts": wrapper, "questions": ["核对折旧"]})
    with pytest.raises(InvalidOutput, match="output_schema_invalid"):
        parse_output(body, Understanding, array_fields=ARRAY_OUTPUT_FIELDS[Understanding])


def test_long_assessment_checks_numbered_passages_and_period_source(tmp_path, monkeypatch):
    monkeypatch.setenv("RESEARCH_CONTEXT_WINDOW", "16000")
    long = "其他业务办理步骤，仅供测试。" * 200
    rule = "本企业所得税业务满足确认条件时应在发出商品时确认收入。"
    first = normalize_evidence({**POLICY, "content": long + rule})
    dated = normalize_evidence({**POLICY, "content": "其他规则自2025年1月1日起施行。"})
    questions = {f"q{i}": f"企业所得税业务发出商品确认收入子问题{i}" for i in range(1, 5)}

    def assess(messages):
        groups = [json.loads(m.content)["evidence"] for m in messages if m.content.startswith('{"evidence":')]
        assert all("content" not in group and sum(map(len, group["passages"].values())) <= 1000 for group in groups)
        pid = next(key for group in groups for key, value in group["passages"].items() if value == rule)
        wrong_period = next(key for group in groups if group["evidence_id"] == dated["evidence_id"]
                            for key in group["passages"])
        return {"coverage": [{"question_id": qid, "passage_ids": [pid, "p999"] if qid == "q4" else [pid],
                              "period_ids": [wrong_period], "sufficient": True} for qid in questions]}

    agent = ResearchAgent(make_store(tmp_path), llm=ScriptedModel([assess]), workdir_root=tmp_path)
    data = asyncio.run(agent.assess({"query": rule, "facts": [], "questions": list(questions.values()),
        "candidates": [first, dated], "research_rounds": 1, "tax_period": "2025年"}))
    coverage = data["assessment"]["coverage"]
    assert len(coverage) == 4 and all(not row["sufficient"] for row in coverage)
    assert all(row["excerpts"] == {first["evidence_id"]: rule} for row in coverage)
    assert all(row["period_excerpts"] == {} for row in coverage)
    assert len(data["metrics"]["llm_calls"]) == 1


def test_retrieval_deduplicates_query_and_page_without_losing_question_links(tmp_path):
    url = POLICY["source_url"]
    requested = []

    def read_page(request_url, **kwargs):
        requested.append(request_url)
        return {**POLICY, "url": url, "status": "success"}

    agent = ResearchAgent(make_store(tmp_path), llm=ScriptedModel([]), search=lambda *a, **k: [],
        web_search=lambda *a, **k: [{"url": url + "#one"}, {"url": url + "#two"}],
        read_page=read_page, workdir_root=tmp_path)
    data = asyncio.run(agent.retrieve({"query": "模拟业务", "web_search": True, "questions": ["甲", "乙"],
        "plan": [{"question": "甲", "search_query": "ＡＢ　业务"}, {"question": "乙", "search_query": "ab  业务"}]}))
    assert requested == [url]
    assert len(data["metrics"]["tool_calls"]) == 3
    assert data["candidates"][0]["question_ids"] == ["q1", "q2"]


@pytest.mark.parametrize("failure, expected", [(None, "reference_analysis"), ("error", "failed")])
def test_reference_analysis_separate_from_report_and_technical_failure(tmp_path, failure, expected):
    def search(*args, **kwargs):
        if failure:
            raise ConnectionError("mock")
        return []

    async def verify():
        store = make_store(tmp_path)
        with tenant_scope("test"):
            task = store.create("模拟业务", allow_reference_analysis=True, budget={"clarifications": 0})
            replies = [understanding(), {"analysis": "如业务用于经营，仍需核对相应的适用条件。"}]
            async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
                agent = ResearchAgent(store, llm=ScriptedModel(replies), search=search, workdir_root=tmp_path)
                agent._checkpointer = saver
                result = await agent.execute(task["task_id"], task["revision"])
                assert result["status"] == expected and not result.get("report")
                assert store.history(task["task_id"]) == []
                if failure:
                    assert result["failure"]["code"] == "retrieval_failed"
                    assert len(result["metrics"]["llm_calls"]) == 1
                else:
                    assert result["reference_analysis"]["notice"] == "未经政策原文充分核验，仅供参考"
                    assert result["reference_analysis"]["unknown_facts"] == ["地区未知"]
                    assert result["reference_analysis"]["policy_gaps"]
    asyncio.run(verify())


@pytest.mark.parametrize("finish,first_content,reason", [
    ("stop", " ", "output_empty"),
    ("length", '{"questions":[', "output_truncated"),
])
def test_official_deepseek_transport_options_and_regeneration(tmp_path, finish, first_content, reason):
    sent = []
    def respond(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"model": "deepseek-flash", "choices": [
            {"index": 0, "finish_reason": finish if len(sent) == 1 else "stop", "message": {
                "role": "assistant", "content": first_content if len(sent) == 1 else '{"questions":["核对折旧"]}'}}]})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        llm = ChatOpenAI(model="deepseek-flash", base_url="https://api.deepseek.com/v1",
                         api_key="unit-model-key", http_client=client, max_retries=0)
        agent = ResearchAgent(make_store(tmp_path), llm=llm, workdir_root=tmp_path)
        data = asyncio.run(agent.understand({"query": QUERY}))
    assert len(sent) == 2
    assert sent[0]["response_format"] == {"type": "json_object"}
    assert sent[0]["thinking"] == {"type": "disabled"}
    assert sent[0]["max_tokens"] <= 768 and "max_completion_tokens" not in sent[0]
    first, retry = data["metrics"]["llm_calls"]
    assert first["outcome"] == reason and retry["retry_kind"] == reason
    if finish == "length":
        assert sent[1]["max_tokens"] > sent[0]["max_tokens"]
    assert "candidate_output" not in sent[1]["messages"][-1]["content"]
    assert all("response" not in call for call in (first, retry))


def test_partial_clarification_resume_keeps_goal_and_excludes_user_wait(tmp_path, monkeypatch):
    clock = 0
    def perf_counter():
        return clock
    monkeypatch.setattr("agents.research_agent.time", SimpleNamespace(perf_counter=perf_counter))

    def response(value, seconds):
        def scripted(_messages):
            nonlocal clock
            clock += seconds
            return value
        return scripted

    async def verify():
        nonlocal clock
        initial = {"facts": ["我们企业买了一套二手房"], "questions": QUESTIONS, "unknowns": PENDING}
        mixed_schema = {"properties": {**initial, **{
            key: {"type": "array", "items": initial.get(key, []), "description": "数组", "title": key, "maxItems": 30}
            for key in ("facts", "unknowns", "tax_types")}},
                        "required": ["questions"], "title": "Understanding", "type": "object"}
        update = {"facts": [REPLY, "模型猜测金额"], "answered": ["f1"], "kept_unknown": [],
                  "needs_reanalysis": False, "questions": ["改成计算房产税", "自用办公的适用条件", QUESTIONS[2]]}
        model = ScriptedModel([response(mixed_schema, 12), response({}, 23), response({}, 8), response(update, 2)])
        store = make_store(tmp_path)
        statuses = []
        async def emit(kind, value):
            assert kind == "status"
            statuses.append(value)
        with tenant_scope("test"):
            task = store.create(QUERY, web_search=True)
            async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "checkpoint.db")) as saver:
                agent = ResearchAgent(store, llm=model, workdir_root=tmp_path)
                agent._checkpointer = saver
                first = await agent.execute(task["task_id"], task["revision"], emit=emit)
                assert first["status"] == "awaiting_facts" and first["elapsed_ms"] == 12000
                first_call, = first["metrics"]["llm_calls"]
                assert {repair.get("field") for repair in first_call["format_repairs"]} == {
                    None, "facts", "unknowns", "tax_types"}
                assert not any(event["assessment_started"] for event in statuses)
                clock += 120
                failed = await agent.execute(task["task_id"], first["revision"],
                                             action={"action": "clarify", "reply": REPLY}, emit=emit)
                assert failed["status"] == "failed" and failed["resume_stage"] == "update_facts"
                assert failed["failure"]["details"]["output_error"] == "output_schema_invalid"
                assert failed["elapsed_ms"] == 43000 and failed["run_elapsed_ms"] == 31000
                assert len(failed["metrics"]["llm_calls"]) == 3 and not failed["metrics"]["tool_calls"]
                clock += 60
                saved = await agent.execute(task["task_id"], failed["revision"], emit=emit)
        assert saved["status"] == "awaiting_facts"
        assert saved["original_query"] == QUERY and saved["query"].count(REPLY) == 1
        assert saved["questions"][0] == QUESTIONS[0]
        assert saved["facts"] == [*initial["facts"], REPLY] and saved["unknowns"] == PENDING[1:]
        assert saved["question_ids"] == ["q1", "q4", "q3"]
        assert saved["elapsed_ms"] == 45000 and saved["run_elapsed_ms"] == 2000
        attempts = [item for item in saved["trace"] if item["stage"] == "update_facts"]
        assert [item["elapsed_ms"] for item in attempts] == [31000, 2000]
        assert len({item["step_id"] for item in saved["trace"]}) == len(saved["trace"])
        assert attempts[0]["run_id"] != attempts[1]["run_id"]
        assert statuses[-1]["elapsed_ms"] == saved["elapsed_ms"] and statuses[-1]["run_elapsed_ms"] == saved["run_elapsed_ms"]
    asyncio.run(verify())


def test_legacy_failed_clarification_resumes_incremental_node(tmp_path):
    async def verify():
        store = make_store(tmp_path)
        with tenant_scope("test"):
            task = store.create(QUERY, web_search=True)
            lease = store.claim(task["task_id"], task["revision"])
            saved = store.save({**task, "status": "failed", "resume_stage": "understand",
                "original_query": QUERY, "query": QUERY + "\n用户补充：" + REPLY,
                "clarification_has_reply": True, "clarification_rounds": 1, "questions": QUESTIONS, "unknowns": PENDING}, lease)
            store.release(task["task_id"], lease)
            update = {"facts": [REPLY], "answered": ["f1"], "kept_unknown": [], "needs_reanalysis": False}
            async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "checkpoint.db")) as saver:
                agent = ResearchAgent(store, llm=ScriptedModel([update]), workdir_root=tmp_path)
                agent._checkpointer = saver
                resumed = await agent.execute(saved["task_id"], saved["revision"])
        assert resumed["status"] == "awaiting_facts" and resumed["unknowns"] == PENDING[1:]
        assert [call["purpose"] for call in resumed["metrics"]["llm_calls"]] == ["ClarificationUpdate"]
    asyncio.run(verify())


def test_failed_understanding_resumes_with_optional_label_mismatch(tmp_path):
    fact_question = PENDING[0]
    policy_question = "适用税率是多少？"
    other_policy_question = "应查哪一条法规？"
    response = {"facts": ["我们企业买了一套二手房"], "questions": [QUESTIONS[0]],
                "unknowns": [fact_question, policy_question, other_policy_question],
                "unknown_kinds": {fact_question: "业务用途", policy_question: "政策规则", other_policy_question: "policy"}}

    def model_reply(messages):
        assert '值只能用英文fact（业务事实）或policy（政策规则）' in messages[0].content
        return response

    async def verify():
        store = make_store(tmp_path)
        with tenant_scope("test"):
            task = store.create(QUERY, web_search=True)
            lease = store.claim(task["task_id"], task["revision"])
            failed = store.save({**task, "status": "failed", "stage": "understand", "resume_stage": "understand",
                "failure": {"code": "structured_output_invalid",
                            "details": {"validation_issues": [{"field": "unknown_kinds", "type": "literal_error"}]}}}, lease)
            store.release(task["task_id"], lease)
            async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "checkpoint.db")) as saver:
                agent = ResearchAgent(store, llm=ScriptedModel([model_reply]), workdir_root=tmp_path)
                agent._checkpointer = saver
                resumed = await agent.execute(failed["task_id"], failed["revision"])
        assert resumed["status"] == "awaiting_facts"
        assert resumed["facts"] == response["facts"]
        assert resumed["unknowns"] == [fact_question]
        assert resumed["questions"] == [QUESTIONS[0], policy_question, other_policy_question]
        assert [(call["purpose"], call["outcome"]) for call in resumed["metrics"]["llm_calls"]] == [
            ("Understanding", "success")]
    asyncio.run(verify())


def test_understanding_retry_reports_pending_fact_limit(tmp_path):
    def repair(messages):
        assert "unknowns最多3项" in messages[0].content
        payload = json.loads(messages[-1].content)
        assert payload["structure_errors"] == [{"field": "unknowns", "type": "too_long", "max_length": 3}]
        return {"questions": QUESTIONS, "unknowns": PENDING}

    agent = ResearchAgent(make_store(tmp_path), llm=ScriptedModel([
        {"questions": QUESTIONS, "unknowns": [*PENDING, "企业性质是什么？"]}, repair]), workdir_root=tmp_path)
    data = asyncio.run(agent.understand({"query": QUERY}))
    assert data["unknowns"] == PENDING and data["questions"] == QUESTIONS
    assert [call["outcome"] for call in data["metrics"]["llm_calls"]] == ["output_schema_invalid", "success"]


def test_complex_correction_can_reanalyze_without_retaining_retracted_fact(tmp_path):
    corrected = "更正，是新房"
    initial = {"facts": ["二手房"], "questions": QUESTIONS, "unknowns": PENDING}
    update = {"facts": [], "answered": [], "kept_unknown": [], "needs_reanalysis": True}
    reanalyzed = {"facts": ["是新房"], "questions": ["新房折旧如何处理", QUESTIONS[2]], "unknowns": PENDING[1:]}
    async def verify():
        store = make_store(tmp_path)
        with tenant_scope("test"):
            task = store.create(QUERY)
            async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "checkpoint.db")) as saver:
                agent = ResearchAgent(store, llm=ScriptedModel([initial, update, reanalyzed]), workdir_root=tmp_path)
                agent._checkpointer = saver
                first = await agent.execute(task["task_id"], task["revision"])
                second = await agent.execute(task["task_id"], first["revision"], action={"action": "clarify", "reply": corrected})
        assert second["status"] == "awaiting_facts" and second["original_query"] == QUERY
        assert second["facts"] == ["是新房"] and second["question_ids"] == ["q4", "q3"]
        assert [call["purpose"] for call in second["metrics"]["llm_calls"]] == ["Understanding", "ClarificationUpdate", "Understanding"]
    asyncio.run(verify())


def test_cancelled_stage_is_timed_and_replayed_with_a_new_attempt(tmp_path):
    async def verify():
        store = make_store(tmp_path)
        started = asyncio.Event()
        with tenant_scope("test"):
            task = store.create("模拟业务")
            async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "checkpoint.db")) as saver:
                agent = ResearchAgent(store, llm=ScriptedModel([understanding()]), workdir_root=tmp_path)
                agent._checkpointer = saver
                understand = agent.understand
                async def blocked(data):
                    started.set()
                    await asyncio.Future()
                agent.understand = blocked
                run = asyncio.create_task(agent.execute(task["task_id"], task["revision"]))
                await asyncio.wait_for(started.wait(), 3)
                run.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await run
                saved = store.get(task["task_id"])
                assert saved["status"] == "interrupted" and saved["trace"][-1]["outcome"] == "interrupted"
                agent.understand = understand
                resumed = await agent.execute(saved["task_id"], saved["revision"])
        assert resumed["status"] == "awaiting_facts"
        attempts = [item for item in resumed["trace"] if item["stage"] == "understand"]
        assert [item["outcome"] for item in attempts] == ["interrupted", "completed"]
        assert resumed["elapsed_ms"] == saved["elapsed_ms"] + resumed["run_elapsed_ms"]
    asyncio.run(verify())


def test_scope_changes_invalidate_assessment_and_keep_raw_sources(tmp_path):
    agent = ResearchAgent(make_store(tmp_path), workdir_root=tmp_path)
    source = {**normalize_evidence(POLICY), "question_ids": ["q1", "q2", "q3"], "search_steps": ["1:s1"]}
    data = {"questions": QUESTIONS, "facts": [], "plan": [{"question_ids": ["q2"]}],
            "assessment": covered(), "completed_questions": 1, "evidence": [source], "research_rounds": 1}
    agent.update_scope(data, {"questions": QUESTIONS})
    assert data["assessment"] and data["plan"] and data["evidence"]
    changed = {"questions": ["办公用房折旧", QUESTIONS[2]]}
    agent.update_scope(data, changed)
    data.update(changed)
    assert agent.question_map(data) == {"q4": "办公用房折旧", "q3": QUESTIONS[2]}
    assert not data["evidence"] and "plan" not in data and "assessment" not in data and "completed_questions" not in data
    assert data["candidates"][0]["question_ids"] == ["q3"] and not data["candidates"][0]["search_steps"]


def test_unrelated_fact_retains_questions_and_plan_and_skip_needs_no_model(tmp_path):
    initial = {"query": QUERY + "\n用户补充：" + REPLY, "original_query": QUERY, "questions": QUESTIONS,
               "unknowns": PENDING, "facts": [], "plan": [{"question_ids": ["q1"]}],
               "clarification_reply": REPLY, "skip_unknowns": True}
    answer = {"facts": [REPLY], "answered": ["f1"], "kept_unknown": [], "needs_reanalysis": False}
    agent = ResearchAgent(make_store(tmp_path), llm=ScriptedModel([answer]), workdir_root=tmp_path)
    saved = asyncio.run(agent.update_facts(initial))
    assert saved["questions"] == QUESTIONS and saved["plan"] == [{"question_ids": ["q1"]}]
    assert saved["kept_unknowns"] == PENDING[1:] and not saved["unknowns"]
    assert len(saved["metrics"]["llm_calls"]) == 1
    asyncio.run(agent.update_facts(saved))
    assert len(saved["metrics"]["llm_calls"]) == 1


def http_response(payload, *, html=False, status=200):
    response = requests.Response()
    response.status_code = status
    response._content = payload.encode() if html else json.dumps(payload, ensure_ascii=False).encode()
    response._content_consumed = True
    response.headers["Content-Type"] = "text/html" if html else "application/json"
    return response


@pytest.mark.parametrize("case", ["empty", "http", "invalid", "success"])
def test_official_search_status_and_rank(monkeypatch, case):
    official = "https://fgk.chinatax.gov.cn/zcfgk/c100010/c5194417/content.html"
    rows = [] if case == "empty" else [{"url": "https://www.chinatax.gov.cn/example", "title": "政策新闻"},
        {"url": "https://chinatax.gov.cn.example.org/fake", "title": "二手房折旧"},
        {"url": official, "title": "二手房如何计提<em>折旧</em>"}]
    def get(url, **options):
        assert url == "https://www.chinatax.gov.cn/search5/search/s" and options["allow_redirects"] is False
        assert options["params"]["searchSiteName"] == "GSFFK" and options["params"]["pageSize"] == 10
        return http_response({"searchResultAll": {"searchTotal": None if case == "invalid" else rows}},
                             status=503 if case == "http" else 200)
    monkeypatch.setattr(kb_tools.requests, "get", get)
    if case in ("http", "invalid"):
        with pytest.raises((requests.HTTPError, ValueError)):
            kb_tools.search_official_research("二手房 折旧", rank_titles=True)
    else:
        result = kb_tools.search_official_research("二手房 折旧", rank_titles=True)
        assert [item["url"] for item in result] == ([] if case == "empty" else [official, rows[0]["url"]])
    assert not official_url("https://www.chinatax.gov.cn:8000/policy")


def test_empty_local_policy_library_can_read_and_relink_official_original(tmp_path, monkeypatch):
    source = "https://fgk.chinatax.gov.cn/zcfgk/c100010/c5194417/content.html"
    body = "模拟政策原文仅供测试，包括业务条件和政策资料引用。这里故意提供足够长度。"
    simultaneous = threading.Barrier(2, timeout=3)
    def local(*args, **kwargs):
        assert kwargs["research"] is True
        simultaneous.wait()
        return []
    def get(url, **kwargs):
        if "search/s" in url:
            simultaneous.wait()
            return http_response({"searchResultAll": {"searchTotal": [{"url": source, "title": "模拟政策"}]}})
        assert url == source and kwargs["allow_redirects"] is False
        return http_response(f'<html><meta charset="utf-8"><title>模拟政策</title><div class="arc_cont">{body}</div></html>', html=True)
    monkeypatch.setattr(kb_tools, "search_tax_policy", local)
    monkeypatch.setattr(kb_tools.requests, "get", get)
    async def verify():
        agent = ResearchAgent(make_store(tmp_path), workdir_root=tmp_path)
        data = {"query": QUERY, "web_search": True, "questions": QUESTIONS[:2],
                "plan": [{"step_id": "s1", "question": QUESTIONS[0], "search_query": "模拟政策", "question_ids": ["q1"]},
                         {"step_id": "s2", "question": QUESTIONS[1], "search_query": "模拟政策", "question_ids": ["q2"]}]}
        with tenant_scope("test"):
            found = await agent.retrieve(data)
        assert [call["status"] for call in found["metrics"]["tool_calls"]] == ["empty", "success", "success"]
        item, = found["candidates"]
        assert item["content"] == body and item["question_ids"] == ["q1", "q2"]
        assert item["validity_check"] == "unknown" and item["applicability_check"] == "unknown"
    asyncio.run(verify())


def test_search_failure_resumes_only_failed_lookup(tmp_path):
    calls = []
    def local(*args, **kwargs):
        calls.append("local")
        return []
    def web(*args, **kwargs):
        calls.append("web")
        if calls.count("web") == 1:
            raise requests.Timeout
        return [{"url": POLICY["source_url"]}]
    def read(url, **kwargs):
        return {**POLICY, "status": "success", "url": url}
    async def verify():
        store = make_store(tmp_path)
        with tenant_scope("test"):
            task = store.create("模拟业务", web_search=True, budget={"clarifications": 0, "research_rounds": 1})
            async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "checkpoint.db")) as saver:
                agent = ResearchAgent(store, llm=ScriptedModel([understanding(), assess_covered, report()]),
                                      search=local, web_search=web, read_page=read, workdir_root=tmp_path)
                agent._checkpointer = saver
                failed = await agent.execute(task["task_id"], task["revision"])
                assert failed["failure"]["code"] == "retrieval_failed" and failed["resume_stage"] == "retrieve"
                assert [item["status"] for item in failed["metrics"]["tool_calls"]] == ["empty", "error"]
                resumed = await agent.execute(task["task_id"], failed["revision"])
        assert resumed["status"] == "awaiting_review" and calls == ["local", "web", "web"]
        assert resumed["evidence"][0]["question_ids"] == ["q1"]
        assert len([call for call in resumed["metrics"]["llm_calls"] if call["purpose"] == "Understanding"]) == 1
    asyncio.run(verify())


def test_report_resumes_citation_correction_without_regenerating_draft(tmp_path):
    def timeout(_messages):
        raise TimeoutError("mocked timeout")
    async def verify():
        store = make_store(tmp_path)
        draft = report()
        draft["analysis"][0]["evidence_ids"] = ["invented"]
        with tenant_scope("test"):
            task = store.create("模拟业务", budget={"clarifications": 0})
            async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "checkpoint.db")) as saver:
                agent = ResearchAgent(store, llm=ScriptedModel([understanding(), assess_covered, draft, timeout, report()]),
                                      search=lambda *a, **k: [POLICY], workdir_root=tmp_path)
                agent._checkpointer = saver
                failed = await agent.execute(task["task_id"], task["revision"])
                assert failed["resume_stage"] == "report" and failed["failure"]["code"] == "model_call_failed"
                assert not failed.get("report") and failed["report_version"] == 0
                saved = await agent.execute(task["task_id"], failed["revision"])
        assert saved["status"] == "awaiting_review" and saved["report_version"] == 1
        assert not saved.get("pending_report")
        assert [call["retry_kind"] for call in saved["metrics"]["llm_calls"] if call["purpose"] == "ResearchReport"] == [None, "citation", "citation"]
        assert len(saved["metrics"]["tool_calls"]) == 1
    asyncio.run(verify())


def test_official_challenge_html_is_not_policy_original(monkeypatch):
    monkeypatch.setattr(kb_tools.requests, "get",
                        lambda *a, **k: http_response("<html><body>请完成验证后再访问</body></html>", html=True))
    result = kb_tools.read_web_page("https://fgk.chinatax.gov.cn/mock", official_only=True, require_article=True)
    assert result["status"] == "error" and result["error_code"] == "no_original" and not result["content"]
