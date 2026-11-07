import asyncio

from eval.dataset import EvalSample
from eval.metrics import refusal_accuracy, refusal_detected
from eval.run_eval import run_answer_eval
from services.security import current_tenant_id


def test_failed_model_call_never_scores_as_successful_refusal(tmp_path, monkeypatch):
    seen = []
    class FailingAgent:
        def __init__(self, **kwargs):
            seen.append(kwargs["workdir_root"])
        async def run(self, query, context):
            from agents.base_agent import runtime_checkpoint_path, runtime_memory
            from services.security import read_only_retrieval
            assert read_only_retrieval.get() is True
            assert current_tenant_id().startswith("eval-")
            assert runtime_checkpoint_path.get().is_relative_to(tmp_path)
            assert runtime_memory.get().base_dir.is_relative_to(tmp_path)
            seen.append(context["thread_id"])
            raise RuntimeError("private provider failure")
    monkeypatch.setattr("agents.tax_agent.TaxAgent", FailingAgent)
    samples = [EvalSample("broken", "问题", "knowledge_query", [], must_refuse=True)]
    answers, details = asyncio.run(run_answer_eval(samples, "tax_agent", runtime_root=tmp_path))
    assert answers == [""]
    assert details[0]["execution_success"] is False and details[0]["refusal_correct"] is False
    assert details[0]["error_type"] == "RuntimeError"
    assert details[0]["keyword_recall"] == 0
    assert "private" not in str(details)
    assert refusal_accuracy(answers, [True], execution_success=[False]) == 0
    first_thread = seen[-1]
    asyncio.run(run_answer_eval(samples, "tax_agent", runtime_root=tmp_path))
    assert seen[-1] != first_thread


def test_apology_alone_is_not_refusal():
    assert refusal_detected("抱歉刚才表述有误，计算结果为1480元") is False


def test_professional_review_preserves_save_consent_audit(tmp_path):
    from sqlalchemy import create_engine
    from services.research_cases import CaseStore
    from services.security import tenant_scope
    from tests.test_research_cases import content
    store = CaseStore(create_engine(f"sqlite:///{tmp_path / 'cases.db'}"))
    with tenant_scope("audit"):
        case = store.create(content())
        case = store.change(case["case_id"], case["revision"], "consent")
        stamp = case["save_confirmed_at"]
        reviewed = store.change(case["case_id"], case["revision"], "professional_review", notes="测试复核说明")
        assert reviewed["save_confirmed_at"] == stamp
        assert reviewed["save_confirmed_by"] == "audit"
