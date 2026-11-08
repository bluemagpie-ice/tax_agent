"""Isolated SQLite + scripted model/tools; never calls a provider or cloud KB."""
import asyncio
import json

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from sqlalchemy import create_engine

from agents.research_agent import ResearchAgent
from services.research_evidence import normalize_evidence
from services.research_store import ResearchStore, ResearchConflict
from services.security import tenant_scope


POLICY = {"id": "public-test", "source_type": "default", "title": "模拟政策资料（非税法）",
          "content": "模拟资料只用于验证证据读取和引用，不提供真实税务判断。", "source_url": "https://www.chinatax.gov.cn/test"}


@pytest.fixture(autouse=True)
def offline_tokenizer(monkeypatch):
    monkeypatch.setattr("agents.streaming.count_tokens", lambda text, name="": (len(str(text)), "test_estimate"))


class ScriptedModel:
    def __init__(self, responses):
        self.responses = iter(responses)

    def invoke(self, messages, **kwargs):
        value = next(self.responses)
        if callable(value):
            value = value(messages)
        return AIMessage(content=json.dumps(value, ensure_ascii=False))


def understanding():
    return {"facts": ["模拟业务", "模型猜测事实"], "questions": ["核对模拟业务政策"], "unknowns": ["地区未知"]}


def report():
    return {"scope": "模拟研究", "analysis": [{"question": "核对模拟业务政策", "finding": "仍需核实", "evidence_ids": [normalize_evidence(POLICY)["evidence_id"]]}],
            "uncertainties": ["政策效力未知"], "follow_up_questions": ["业务所在地？"], "answer_draft": "资料不足，暂不能确定适用。"}


def covered():
    evidence_id = normalize_evidence(POLICY)["evidence_id"]
    return {"coverage": [{"question":"核对模拟业务政策", "sufficient":True,
                          "evidence_ids":[evidence_id], "excerpts":{evidence_id:POLICY["content"]}}]}


def test_research_end_to_end_restart_review_and_isolation(tmp_path):
    async def verify():
        store = ResearchStore(create_engine(f"sqlite:///{tmp_path / 'business.sqlite'}"))
        with tenant_scope("alice"):
            task = store.create("模拟业务需要研究", web_search=False, budget={"clarifications":0})
            async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "graph.sqlite")) as saver:
                agent = ResearchAgent(store, llm=ScriptedModel([understanding(), covered(), report()]),
                                      search=lambda *a, **k: [POLICY], workdir_root=tmp_path)
                agent._checkpointer = saver
                result = await agent.execute(task["task_id"], task["revision"])
                assert result["status"] == "awaiting_review"
                assert result["facts"] == ["模拟业务"]
                assert result["evidence"][0]["validity_check"] == "unknown"
                assert len(store.history(task["task_id"])) == 1
            async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "graph.sqlite")) as saver:
                agent = ResearchAgent(store, llm=ScriptedModel([]), workdir_root=tmp_path)
                agent._checkpointer = saver
                with pytest.raises(ResearchConflict):
                    await agent.execute(task["task_id"], result["revision"], action={"action":"confirm", "report_version":99})
                result = await agent.execute(task["task_id"], result["revision"], action={"action":"confirm", "report_version":1})
                assert result["status"] == "finalized"
                assert result["professional_review"] == "not_reviewed"
        with tenant_scope("bob"):
            assert store.list() == []
            with pytest.raises(KeyError):
                store.get(task["task_id"])
    asyncio.run(verify())


def test_store_idempotent_schema_and_stale_actions(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'business.sqlite'}")
    store = ResearchStore(engine)
    ResearchStore(engine)
    with tenant_scope("alice"):
        task = store.create("模拟")
        lease = store.claim(task["task_id"], 1)
        with pytest.raises(ResearchConflict):
            store.claim(task["task_id"], 1)
        store.save(task, lease)
        store.release(task["task_id"], lease)
        with pytest.raises(ResearchConflict):
            store.claim(task["task_id"], 1)


def test_report_cannot_cite_nonexistent_evidence():
    from services.research_evidence import validate_report
    with pytest.raises(ValueError):
        validate_report({"analysis":[{"evidence_ids":["invented"]}]}, [])


def test_fact_skip_review_edit_and_stale_version(tmp_path):
    async def verify():
        with tenant_scope("alice"):
            store = ResearchStore(create_engine(f"sqlite:///{tmp_path / 'facts.sqlite'}"))
            task = store.create("模拟业务研究")
            async with AsyncSqliteSaver.from_conn_string(str(tmp_path / 'facts-graph.sqlite')) as saver:
                agent = ResearchAgent(store, llm=ScriptedModel([understanding(), covered(), report()]),
                                      search=lambda *a, **k: [POLICY], workdir_root=tmp_path)
                agent._checkpointer = saver
                task = await agent.execute(task['task_id'], task['revision'])
                assert task['status'] == 'awaiting_facts'
                task = await agent.execute(task['task_id'], task['revision'], action={"action":"clarify", "skip_unknowns":True})
                assert task['status'] == 'awaiting_review' and task['clarification_rounds'] == 1
                old_revision = task['revision']
                task = await agent.execute(task['task_id'], task['revision'], action={"action":"revise", "report_version":1, "report":report()})
                assert task['report_version'] == 2 and len(store.history(task['task_id'])) == 2
                with pytest.raises(ResearchConflict):
                    await agent.execute(task['task_id'], old_revision, action={"action":"confirm", "report_version":1})
    asyncio.run(verify())


def test_budget_and_no_evidence_stop_without_claiming_verified(tmp_path):
    async def verify():
        with tenant_scope("alice"):
            store = ResearchStore(create_engine(f"sqlite:///{tmp_path / 'budget.sqlite'}"))
            task = store.create("模拟", budget={"clarifications":0, "llm_calls":0, "tool_calls":0})
            async with AsyncSqliteSaver.from_conn_string(str(tmp_path / 'budget-graph.sqlite')) as saver:
                agent = ResearchAgent(store, llm=ScriptedModel([]), search=lambda *a, **k: pytest.fail('budget bypass'), workdir_root=tmp_path)
                agent._checkpointer = saver
                task = await agent.execute(task['task_id'], task['revision'])
                assert task['status'] == 'failed'
                assert task['stop_reason'] == 'budget_exhausted'
                assert task['failure']['code'] == 'research_budget_exhausted'
                assert not task.get('report') and store.history(task['task_id']) == []
    asyncio.run(verify())


def test_retrieval_replans_once_and_stops_on_no_new_evidence(tmp_path):
    async def verify():
        with tenant_scope('alice'):
            store = ResearchStore(create_engine(f"sqlite:///{tmp_path / 'repeat.sqlite'}"))
            task = store.create('模拟业务', budget={'clarifications':0})
            gap = {'coverage':[], 'next_queries':['补查模拟业务']}
            plan = {'steps':[{'step_id':'extra','question':'补查模拟业务','depends_on':[]}]}
            responses = [understanding(),gap,plan]
            async with AsyncSqliteSaver.from_conn_string(str(tmp_path / 'repeat-graph.sqlite')) as saver:
                agent = ResearchAgent(store,llm=ScriptedModel(responses),search=lambda *a,**k:[POLICY],workdir_root=tmp_path)
                agent._checkpointer=saver
                result = await agent.execute(task['task_id'],task['revision'])
                assert result['research_rounds'] == 2 and result['replans'] == 1
                assert result['stop_reason'] == 'no_new_evidence'
                assert result['failure']['code'] == 'insufficient_evidence'
    asyncio.run(verify())


@pytest.mark.parametrize(("search_errors", "status", "code"), [
    (False, "insufficient_evidence", "insufficient_evidence"),
    (True, "failed", "retrieval_failed"),
])
def test_empty_knowledge_and_retrieval_failure_have_distinct_outcomes(tmp_path, search_errors, status, code):
    def search(*args, **kwargs):
        if search_errors:
            raise ConnectionError("test vector backend unavailable")
        return []

    async def verify():
        with tenant_scope("alice"):
            store = ResearchStore(create_engine(f"sqlite:///{tmp_path / 'empty.sqlite'}"))
            task = store.create("模拟业务", budget={"clarifications": 0, "research_rounds": 1})
            async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "empty-graph.sqlite")) as saver:
                agent = ResearchAgent(store, llm=ScriptedModel([understanding()]), search=search, workdir_root=tmp_path)
                agent._checkpointer = saver
                result = await agent.execute(task["task_id"], task["revision"])
                assert result["status"] == status and result["failure"]["code"] == code
                assert not result.get("report") and not store.history(task["task_id"])
                assert result["metrics"]["tool_calls"][0]["status"] == ("error" if search_errors else "empty")
    asyncio.run(verify())


def test_research_skips_embeddings_when_policy_collection_is_missing_or_empty(monkeypatch):
    from knowledge.embedding import EmbeddingService, QdrantService
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams
    from services.tools.kb_tools import search_tax_policy

    client = QdrantClient(":memory:")
    cloud = QdrantService(read_only=True, config={"url":"http://localhost:6333", "api_key":""}, client=client)
    monkeypatch.setattr("knowledge.embedding.QdrantService", lambda **kwargs: cloud)
    monkeypatch.setattr(EmbeddingService, "_embed_texts", lambda *args: pytest.fail("empty corpus must not call embedding"))
    try:
        assert search_tax_policy("missing policy collection", research=True) == []
        client.create_collection(cloud.COLLECTION_NAME, vectors_config=VectorParams(size=4, distance=Distance.COSINE))
        assert search_tax_policy("empty policy collection", research=True) == []
        client.upsert(cloud.COLLECTION_NAME, [PointStruct(id=1, vector=[1., 0., 0., 0.],
                                                          payload={"source_type":"user_upload", "scope":"global"})])
        assert search_tax_policy("only personal files", research=True) == []
        assert client.count(cloud.COLLECTION_NAME).count == 1
    finally:
        client.close()


def test_first_research_saved_case_second_research_revalidates_policy(tmp_path):
    from services.research_candidate import candidate_from_task
    async def verify():
        with tenant_scope('alice'):
            store=ResearchStore(create_engine(f"sqlite:///{tmp_path / 'demo.sqlite'}"))
            searches=[]
            def search(query,**kwargs):
                searches.append(query)
                return [POLICY]
            responses=[understanding(),covered(),report(),understanding(),
                       {'steps':[{'step_id':'q1','question':'核对模拟业务政策','depends_on':[]}]},covered(),report()]
            async with AsyncSqliteSaver.from_conn_string(str(tmp_path / 'demo-graph.sqlite')) as saver:
                agent=ResearchAgent(store,llm=ScriptedModel(responses),search=search,workdir_root=tmp_path)
                agent._checkpointer=saver
                first=store.create('模拟业务首次研究',budget={'clarifications':0})
                first=await agent.execute(first['task_id'],first['revision'])
                first=await agent.execute(first['task_id'],first['revision'],action={'action':'confirm','report_version':1})
                preview=candidate_from_task(first)
                case=agent.case_store.create(**{k:preview[k] for k in ('content','source_task_id','source_report_version')})
                case=agent.case_store.change(case['case_id'],case['revision'],'consent')
                case=agent.case_store.change(case['case_id'],case['revision'],'publish')
                duplicate=agent.case_store.create(**{k:preview[k] for k in ('content','source_task_id','source_report_version')})
                assert duplicate['case_id'] == case['case_id']
                second=store.create('模拟业务第二次研究',use_cases=True,budget={'clarifications':0})
                second=await agent.execute(second['task_id'],second['revision'])
                assert second['case_references'][0]['case_id'] == case['case_id']
                assert len(searches) == 2 and second['evidence']
    asyncio.run(verify())


def test_explicit_calculation_never_supplies_default_rate(tmp_path):
    async def verify():
        with tenant_scope('alice'):
            store=ResearchStore(create_engine(f"sqlite:///{tmp_path / 'calc.sqlite'}"))
            args={'requested':True,'sales_amount':100000,'vat_rate':.13,'input_tax':8000,'sales_is_inclusive':False,
                  'source_quotes':{'sales_amount':'100000元','vat_rate':'13%','input_tax':'8000元','sales_is_inclusive':'不含税'}}
            query='一般纳税人，请计算增值税税额：不含税销售额100000元，税率13%，进项税额8000元。'
            agent=ResearchAgent(store,llm=ScriptedModel([args,args]),workdir_root=tmp_path)
            data=await agent.calculate({'query':query})
            assert data['calculations'][0]['result']['tax_amount'] == 5000
            data=await agent.calculate({'query':query.replace('税率13%，','')})
            assert not data.get('calculations') and data['warnings']
    asyncio.run(verify())


def test_invalid_model_json_repair_is_bounded_and_counted(tmp_path):
    from eval.research_eval import usage_summary
    async def verify():
        store=ResearchStore(create_engine(f"sqlite:///{tmp_path / 'json.sqlite'}"))
        agent=ResearchAgent(store,llm=ScriptedModel(['invalid json shape',understanding()]),workdir_root=tmp_path)
        data=await agent.understand({'query':'模拟业务'})
        assert data['facts'] == ['模拟业务'] and len(data['metrics']['llm_calls']) == 2
        assert usage_summary(data)['retry_count'] == 1
        assert data['metrics']['llm_calls'][0]['error_type'] == 'ValidationError'
    asyncio.run(verify())


def test_report_unsupported_citation_gets_one_source_bound_correction(tmp_path):
    async def verify():
        store=ResearchStore(create_engine(f"sqlite:///{tmp_path / 'report-fix.sqlite'}"))
        bad={**report(),'answer_draft':'依据财税〔2020〕999号第六条，可以保证免税。'}
        agent=ResearchAgent(store,llm=ScriptedModel([bad,report()]),workdir_root=tmp_path)
        data=await agent.report({'query':'模拟业务','facts':['模拟业务'],'unknowns':[],
                                 'evidence':[normalize_evidence(POLICY)],'report_version':0,'assessment':covered()})
        assert data['status']=='awaiting_review' and data['report_checks']['ok']
        assert '999' not in data['report']['answer_draft']
        assert len(data['metrics']['llm_calls'])==2
        assert '财税〔2020〕999号' in data['report_validation_attempts'][0]['unmatched_citations']
    asyncio.run(verify())


def test_dated_core_evidence_rejects_old_policy_and_ignores_unrelated_candidates(tmp_path):
    async def verify():
        store = ResearchStore(create_engine(f"sqlite:///{tmp_path / 'old-policy.sqlite'}"))
        content = "本模拟办法自2023年1月1日起施行。模拟业务规则仅用于测试。"
        # A start in 2023 alone does not prove expiry in 2026. Test known expiry.
        old = {**POLICY, "content": content, "is_current": False}
        eid = normalize_evidence(old)["evidence_id"]
        assessment = {"coverage": [{"question_id":"q1", "evidence_ids":[eid], "sufficient":True,
                                    "excerpts":{eid:content}, "period_excerpts":{eid:content}}]}
        with tenant_scope("alice"):
            task = store.create("模拟业务", budget={"clarifications":0})
            async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "old-graph.sqlite")) as saver:
                agent = ResearchAgent(store, llm=ScriptedModel([{**understanding(), "tax_period":"2026年第一季度"}, assessment]),
                                      search=lambda *a, **k: [old, old, POLICY], workdir_root=tmp_path)
                agent._checkpointer = saver
                result = await agent.execute(task["task_id"], task["revision"])
                assert result["status"] == "insufficient_evidence" and result["failure"]["code"] == "insufficient_evidence"
                assert len(result["candidates"]) == 2 and len(result["evidence"]) == 1
                assert result["completed_questions"] == 0
                assert "所属期" in result["assessment"]["coverage"][0]["gap"]
                with pytest.raises(ResearchConflict):
                    await agent.execute(task["task_id"], result["revision"], action={"action":"confirm","report_version":1})
    asyncio.run(verify())


def test_repeated_unsupported_citation_has_no_report_version(tmp_path):
    async def verify():
        store = ResearchStore(create_engine(f"sqlite:///{tmp_path / 'bad-report.sqlite'}"))
        bad = {**report(), "answer_draft":"依据财税〔2020〕999号。"}
        with tenant_scope("alice"):
            task = store.create("模拟业务", budget={"clarifications":0})
            async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "bad-graph.sqlite")) as saver:
                agent = ResearchAgent(store, llm=ScriptedModel([understanding(), covered(), bad, bad,
                                                               understanding(), covered(), report()]),
                                      search=lambda *a, **k: [POLICY], workdir_root=tmp_path)
                agent._checkpointer = saver
                result = await agent.execute(task["task_id"], task["revision"])
                assert result["failure"]["code"] == "citation_mismatch"
                assert len(result["report_validation_attempts"]) == 2
                assert not result.get("report") and store.history(task["task_id"]) == []
                restarted = await agent.execute(task["task_id"], result["revision"])
                assert restarted["status"] == "awaiting_review" and restarted["report_version"] == 1
    asyncio.run(verify())


def test_research_api_ownership_and_chat_compatibility(tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from web.research_routes import build_router
    store = ResearchStore(create_engine(f"sqlite:///{tmp_path / 'api.sqlite'}", connect_args={"check_same_thread":False}))
    agent = ResearchAgent(store, workdir_root=tmp_path)
    async def auth():
        with tenant_scope("alice"):
            yield
    app = FastAPI()
    app.include_router(build_router(auth, lambda: agent))
    with TestClient(app) as client:
        created = client.post('/api/research/tasks', json={"query":"模拟研究"})
        assert created.status_code == 200
        task = created.json()
        assert client.get('/api/research/tasks/' + task['task_id']).json()['query'] == '模拟研究'
        assert client.get('/api/research/tasks/not-owned').status_code == 404
        assert client.post('/api/research/tasks', json={"query":" "}).status_code == 422


def test_actual_sse_api_and_versioned_confirmation(tmp_path):
    from contextlib import asynccontextmanager
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from web.research_routes import build_router
    store=ResearchStore(create_engine(f"sqlite:///{tmp_path / 'stream.sqlite'}",connect_args={'check_same_thread':False}))
    agent=ResearchAgent(store,llm=ScriptedModel([understanding(),covered(),report()]),search=lambda *a,**k: [POLICY],workdir_root=tmp_path)
    @asynccontextmanager
    async def lifespan(app):
        async with AsyncSqliteSaver.from_conn_string(str(tmp_path / 'stream-graph.sqlite')) as saver:
            agent._checkpointer=saver
            yield
    async def auth():
        with tenant_scope('alice'):
            yield
    app=FastAPI(lifespan=lifespan)
    app.include_router(build_router(auth,lambda:agent))
    with tenant_scope('alice'):
        task=store.create('模拟业务',budget={'clarifications':0})
    with TestClient(app) as client:
        response=client.post(f'/api/research/tasks/{task["task_id"]}/stream',json={'revision':task['revision']})
        assert response.status_code == 200 and 'text/event-stream' in response.headers['content-type']
        events=[json.loads(line[6:]) for line in response.text.splitlines() if line.startswith('data: ')]
        assert events[0]['type'] == 'status' and events[0]['status'] == 'running'
        assert events[0]['activity'] == '研究执行已启动'
        result=events[-1]['task']
        assert result['status']=='awaiting_review'
        route=f'/api/research/tasks/{task["task_id"]}/review'
        assert client.post(route,json={'revision':result['revision'],'report_version':2,'action':'confirm'}).status_code == 409
        assert client.post(route,json={'revision':result['revision'],'report_version':1,'action':'confirm'}).json()['status']=='finalized'
