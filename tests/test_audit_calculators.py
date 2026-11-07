import json
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from pydantic import ValidationError
from sqlalchemy import create_engine, text

from services.calculators.tools import VAT_TOOL, MONTHLY_WITHHOLDING_TOOL, STAMP_DUTY_TOOL
from agents.sub_agents.calculation_agent import CalculationAgent
from tests.test_agent_workflows import make_agent


@pytest.mark.parametrize("args", [{}, {"sales_amount":None}, {"sales_amount":100,"taxpayer_type":"foo"},
                                  {"sales_amount":100,"vat_rate":13}, {"sales_amount":float("nan")}])
def test_vat_input_rejects_missing_invalid_and_nonfinite(args):
    with pytest.raises(ValidationError):
        VAT_TOOL.invoke(args)


def test_unknown_stamp_category_is_error_and_missing_monthly_income_is_rejected():
    assert "error" in json.loads(STAMP_DUTY_TOOL.invoke({"category":"不存在", "contract_amount":100}))
    for args in ({"monthly_wage":10000,"month_index":13}, {"monthly_wage":10000,"month_index":2}):
        with pytest.raises(ValidationError):
            MONTHLY_WITHHOLDING_TOOL.invoke(args)


def test_specialist_missing_sales_never_generates_zero_tax():
    agent = CalculationAgent.__new__(CalculationAgent)
    assert agent._run_calculator("vat", {"input_tax":100}) is None
    assert agent._run_calculator("vat", {"sales_amount":100, "taxpayer_type":"foo"}) is None


def test_reflection_does_not_confuse_taxable_income_and_tax(tmp_path):
    agent = make_agent(tmp_path, [])
    answer = "应纳税所得额 2,800,000 元。应纳税额 700000 元。"
    result = agent.reflection_node({"final_answer":answer, "original_query":"计算企业所得税", "messages":[AIMessage(content=answer)],
        "tool_evidence":[{"tool":"calculate_corporate_income_tax", "args":{"taxable_income":2800000}, "status":"success"}]})
    assert result["final_answer"] == answer and result["reflection"]["ok"]


def test_invalid_recalculation_is_recoverable_not_unhandled(tmp_path):
    agent = make_agent(tmp_path, [])
    result = agent.reflection_node({"final_answer":"计算草稿", "original_query":"计算", "messages":[],
        "tool_evidence":[{"tool":"calculate_stamp_duty", "args":{"category":"买卖合同", "contract_amount":100,"is_small_scale":None}, "status":"success"}]})
    assert result["reflection"]["ok"] is False
    assert "未经核验" in result["final_answer"]


def test_kb_name_traversal_rejected_before_filesystem():
    from web.main import app
    response = TestClient(app).post("/api/knowledge/databases", headers={"Authorization":"Bearer test_token_123"}, json={"name":"../../evil", "kb_type":"qdrant"})
    assert response.status_code == 400


def test_sqlite_wal_and_parallel_transactions_preserve_all_rows(tmp_path):
    import database.models  # registers connection pragmas
    engine = create_engine(f"sqlite:///{tmp_path / 'concurrent.db'}")
    try:
        with engine.begin() as connection:
            assert connection.exec_driver_sql("PRAGMA journal_mode").scalar() == "wal"
            assert connection.exec_driver_sql("PRAGMA busy_timeout").scalar() == 10000
            connection.exec_driver_sql("CREATE TABLE audit_writes (id INTEGER PRIMARY KEY)")
        def write(index):
            with engine.begin() as connection:
                connection.execute(text("INSERT INTO audit_writes VALUES (:id)"), {"id":index})
        with ThreadPoolExecutor(max_workers=4) as workers:
            list(workers.map(write, range(40)))
        with engine.connect() as connection:
            assert connection.exec_driver_sql("SELECT COUNT(*) FROM audit_writes").scalar() == 40
    finally:
        engine.dispose()
