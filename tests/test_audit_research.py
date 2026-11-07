import asyncio
import copy

import pytest
from sqlalchemy import create_engine
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from agents.research_agent import ResearchAgent, ResearchBudgetExceeded, ResearchReport
from services.policy_validity import period_status
from services.research_evidence import normalize_evidence
from services.research_store import ResearchStore
from services.security import tenant_scope
from tests.test_research import POLICY, ScriptedModel, understanding, covered, report


@pytest.mark.parametrize("quote,status", [
    ("自2023年1月1日起施行", "pending_verification"),
    ("本优惠执行期间为2023年1月1日至2027年12月31日", "supported"),
    ("优惠适用期间2023–2027年", "supported"),
    ("财政部2026年第10号公告", "unknown"),
    ("执行期间为2020年1月1日至2022年12月31日", "unknown"),
])
def test_policy_period_requires_interval_context(quote, status):
    assert period_status([quote], "2026年度") == status


def test_report_question_id_and_redaction_do_not_drop_valid_analysis():
    evidence = normalize_evidence(POLICY)
    assessment = covered()
    assessment["coverage"][0]["question_id"] = "q1"
    result = report()
    result["analysis"][0].update(question_id="q1", question="改写的问题", finding="满足上述证据条件时无需申报；电话13812345678")
    assert ResearchAgent.report_issues(result, [evidence], assessment)["ok"]
    assert "13812345678" not in result["analysis"][0]["finding"]


def test_model_timeout_can_resume_assessment_without_retrieval(tmp_path, monkeypatch):
    monkeypatch.setattr("agents.streaming.count_tokens", lambda text, name="": (len(str(text)), "test"))
    class TimeoutModel(ScriptedModel):
        def invoke(self, messages, **kwargs):
            if self.fail:
                self.fail = False
                raise TimeoutError("mock timeout")
            value = super().invoke(messages, **kwargs)
            self.fail = True
            return value
    async def verify():
        store = ResearchStore(create_engine(f"sqlite:///{tmp_path / 'research.db'}"))
        searches = []
        model = TimeoutModel([understanding()]); model.fail = False
        with tenant_scope("audit"):
            task = store.create("模拟业务", budget={"clarifications":0})
            async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "checkpoint.db")) as saver:
                agent = ResearchAgent(store, llm=model, search=lambda *a, **k: searches.append(1) or [POLICY], workdir_root=tmp_path)
                agent._checkpointer = saver
                failed = await agent.execute(task["task_id"], task["revision"])
                assert failed["failure"]["code"] == "model_call_failed"
                assert failed["resume_stage"] == "assess"
                ids = [e["evidence_id"] for e in failed["candidates"]]
                agent._llm = ScriptedModel([covered(), report()])
                resumed = await agent.execute(task["task_id"], failed["revision"])
                assert resumed["status"] == "awaiting_review"
                assert [e["evidence_id"] for e in resumed["candidates"]] == ids
                assert searches == [1]
    asyncio.run(verify())


def test_report_has_reserved_calls_and_specific_repair_errors(tmp_path, monkeypatch):
    monkeypatch.setattr("agents.streaming.count_tokens", lambda text, name="": (len(str(text)), "test"))
    captured = []
    def repaired(messages):
        captured.extend(str(message.content) for message in messages)
        return report()
    async def verify():
        store = ResearchStore(create_engine(f"sqlite:///{tmp_path / 'budget.db'}"))
        bad = report(); bad["analysis"][0]["evidence_ids"] = ["invented"]
        agent = ResearchAgent(store, llm=ScriptedModel([bad, repaired]), workdir_root=tmp_path)
        data = {"query":"模拟业务", "facts":[], "unknowns":[], "evidence":[normalize_evidence(POLICY)],
                "assessment":covered(), "metrics":{"llm_calls":[{}] * 12,"tool_calls":[]}}
        result = await agent.report(data)
        assert result["status"] == "awaiting_review"
        assert len(result["metrics"]["llm_calls"]) == 14
        assert any("allowed_evidence_ids" in text for text in captured)
    asyncio.run(verify())
