from pathlib import Path

import pytest

from agents.memory import ConversationMemory
from services.security import tenant_scope
from services.calculators.individual_income import calculate_individual_income_annual


@pytest.mark.parametrize("wage,paid,difference", [(50000, 1000, -1000), (100000, 3000, -1520), (100000, 0, 1480)])
def test_annual_settlement_direction(wage, paid, difference):
    result = calculate_individual_income_annual(annual_wage=wage, paid_tax=paid)
    assert any(f"汇算后应补退税额: {difference:.2f}" in note for note in result.notes)


def test_compression_keeps_history_and_retraction_rebuilds_summary(tmp_path):
    memory = ConversationMemory(tmp_path, compress_at=6, retain_recent=2, context_budget_tokens=600)
    with tenant_scope("audit"):
        for i in range(10):
            memory.record_turn("s", f"第{i}轮问题" + "资料" * 300, f"第{i}轮回复")
        session = memory.get_session("s")
        assert len(session["messages"]) == 20
        assert session["messages"][0]["content"].startswith("第0轮")
        assert session["summary"] and session["compressed_count"] > 0
        context = memory.context_for("s", "继续")
        assert context["context_budget"]["estimated_tokens"] <= 600
        assert context["memory_layers"]["working"] < 6
        memory.pop_last_turn("s", revision=session["_storage_version"])
        assert len(memory.get_session("s")["messages"]) == 18
        assert "第9轮" not in memory.context_for("s", "继续")["prompt_text"]
        memory.record_turn("s", "2026年收入200000元", "已记录")
        assert memory.get_session("s")["profile"]["tax_years"] == ["2026"]
        assert memory.delete_session("s")
        assert not memory.context_for("new", "继续")["profile"].get("tax_years")


def test_database_test_runner_never_targets_project_database():
    source = (Path(__file__).parent / "test_database_init.py").read_text(encoding="utf-8")
    assert "project_db" not in source
    assert ".unlink(" not in source
