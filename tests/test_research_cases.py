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
