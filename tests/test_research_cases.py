import asyncio

import pytest
from sqlalchemy import create_engine

from services.research_cases import CaseStore
from services.research_candidate import candidate_from_task
from services.research_store import ResearchConflict
from services.security import tenant_scope
from tests.test_research import POLICY
from services.research_evidence import normalize_evidence


def content():
    return {"title":"模拟增值税研究", "question":"模拟业务进项", "facts":["一般纳税人"], "tax_types":["vat"],
            "region":"上海", "tax_period":"2024", "conclusion":"模拟历史意见，仍需核验", "evidence":[normalize_evidence(POLICY)]}


def publish(store, case):
    case = store.change(case['case_id'], case['revision'], 'consent')
    return store.change(case['case_id'], case['revision'], 'publish')


def test_case_lifecycle_isolation_and_old_version_not_retrievable(tmp_path):
    store = CaseStore(create_engine(f"sqlite:///{tmp_path / 'cases.sqlite'}"))
    with tenant_scope('alice'):
        case = store.create(content())
        assert store.search('增值税') == []
        with pytest.raises(ValueError):
            store.change(case['case_id'], case['revision'], 'publish')
        case = publish(store, case)
        results = store.search('增值税', facts={"region":"北京", "tax_period":"2025"})
        assert results[0]['professional_review']['status'] == 'not_reviewed'
        assert [c['comparison'] for c in results[0]['comparisons']][1:3] == ['different','different']
        case = store.change(case['case_id'], case['revision'], 'edit', content={**content(), "conclusion":"修订意见"})
        assert store.search('增值税') == []
        assert case['current_version'] == 2 and not case['save_confirmed']
        assert len(store.history(case['case_id'])['versions']) == 2
        case = publish(store, case)
        assert store.search('增值税')[0]['version'] == 2
        case = store.change(case['case_id'], case['revision'], 'withdraw')
        assert store.search('增值税') == []
    with tenant_scope('bob'):
        assert store.list() == [] and store.search('增值税') == []
        with pytest.raises(KeyError):
            store.get(case['case_id'])
    with tenant_scope('alice'):
        store.change(case['case_id'], case['revision'], 'delete')
        assert store.list() == []
        with store.engine.connect() as conn:
            assert conn.execute(store.versions.select()).all() == []


def test_save_confirmation_and_professional_review_are_distinct(tmp_path):
    store = CaseStore(create_engine(f"sqlite:///{tmp_path / 'review.sqlite'}"))
    with tenant_scope('alice'):
        case = publish(store, store.create(content()))
        assert case['save_confirmed'] and case['professional_review']['status'] == 'not_reviewed'
        case = store.change(case['case_id'], case['revision'], 'professional_review', notes='人工核对了模拟资料及业务差异；非真实税务审核')
        assert case['professional_review']['reviewer'] == 'alice'
        with pytest.raises(ResearchConflict):
            store.change(case['case_id'], 1, 'withdraw')


def test_candidate_requires_finalization_and_only_previews():
    task = {"task_id":"t", "query":"甲某有限公司收入100万元", "status":"awaiting_review"}
    with pytest.raises(ValueError):
        candidate_from_task(task)
    task.update(status='finalized', facts=['甲某有限公司收入100万元'], report_version=2, evidence=[],
                report={"answer_draft":"待核实", "unconfirmed_facts":[], "uncertainties":[]})
    preview = candidate_from_task(task, generalize=True)
    assert '100万元' not in preview['content']['facts'][0]
    assert preview['source_report_version'] == 2


@pytest.mark.parametrize("delete_cases", [None, False, True])
def test_delete_research_task_through_api(tmp_path, delete_cases):
    from contextlib import asynccontextmanager
    from fastapi import FastAPI, Header
    from fastapi.testclient import TestClient
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    from agents.research_agent import ResearchAgent
    from services.research_store import ResearchStore
    from web.research_routes import build_router

    store = ResearchStore(create_engine(f"sqlite:///{tmp_path / 'delete.sqlite'}"))
    agent = ResearchAgent(store, workdir_root=tmp_path)
    @asynccontextmanager
    async def lifespan(app):
        async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "graph.sqlite")) as saver:
            agent._checkpointer = saver
            if delete_cases is not None:
                with tenant_scope("alice"):
                    graph = await agent.get_graph()
                    await graph.aupdate_state(agent.config(task["task_id"]), {"data": task}, as_node="report")
            yield
    async def auth(user: str = Header(default="alice")):
        with tenant_scope(user):
            yield
    app = FastAPI(lifespan=lifespan)
    app.include_router(build_router(auth, lambda: agent))

    with tenant_scope("alice"):
        task = store.create("测试删除任务")
        linked = None
        if delete_cases is not None:
            lease = store.claim(task["task_id"], task["revision"])
            task.update(status="finalized", report_version=1, report={"answer_draft": "测试报告"})
            task = store.save(task, lease)
            store.release(task["task_id"], lease)
            linked = publish(agent.case_store, agent.case_store.create(content(), source_task_id=task["task_id"],
                                                                       source_report_version=1))
        separate = agent.case_store.create(content())
    with tenant_scope("bob"):
        other = agent.case_store.create(content(), source_task_id=task["task_id"], source_report_version=1)
    with TestClient(app) as client:
        path = "/api/research/tasks/" + task["task_id"]
        body = {"revision": task["revision"]}
        if delete_cases:
            body["delete_cases"] = True
        assert client.get(path).json()["linked_case_count"] == int(linked is not None)
        assert client.request("DELETE", path, json=body, headers={"user":"bob"}).status_code == 404
        assert client.request("DELETE", path, json={**body, "revision": task["revision"] + 1}).status_code == 409
        response = client.request("DELETE", path, json=body)
        assert response.status_code == 200, response.text
        assert response.json()["deleted_case_ids"] == ([linked["case_id"]] if delete_cases else [])
        assert client.get(path).status_code == 404
        async def read_checkpoint():
            with tenant_scope("alice"):
                return await agent.checkpointer.aget_tuple(agent.config(task["task_id"]))
        assert client.portal.call(read_checkpoint) is None
    with store.engine.connect() as conn:
        assert not conn.execute(store.reports.select()).all()
    with tenant_scope("alice"):
        assert {row["case_id"] for row in agent.case_store.list()} == (
            {separate["case_id"], linked["case_id"]} if linked and not delete_cases else {separate["case_id"]})
    with tenant_scope("bob"):
        assert agent.case_store.get(other["case_id"])
