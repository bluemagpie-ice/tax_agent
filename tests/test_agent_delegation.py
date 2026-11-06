"""真实专业Agent图、代码计算与共享黑板；模型决定委托，边界以故障注入验收。"""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from agents.base_agent import BaseAgent
from agents.tax_agent import AgentRegistry
from agents.sub_agents.planning_agent import PlanningAgent
from agents.sub_agents.calculation_agent import CalculationAgent
from services.security import tenant_scope, agent_workspace
from tests.test_agent_workflows import make_agent, ScriptedModel, call, execute


def board(owner="default", session="workflow"):
    return {"owner_id": owner, "session_id": session, "entries": [], "events": [], "delegation_count": 0}


def child(cls, tmp_path, responses):
    agent = cls.__new__(cls)
    BaseAgent.__init__(agent, str(tmp_path))
    agent._ensure_checkpointer = AsyncMock()
    agent.llm = ScriptedModel(responses)
    agent.retrieve_context = lambda *args, **kw: "本轮未提供优惠政策原文，不据此判定优惠资格。"
    return agent


def test_planning_llm_delegates_to_real_calculation_graph_then_observes_result(tmp_path, caplog):
    task = "一般纳税人不含税销售100000元，税率13%，进项税额8000元，只算税额"
    def observe(history):
        receipt = next(m for m in history if isinstance(m, ToolMessage))
        result = json.loads(receipt.content)
        assert result["status"] == "success" and "5000" in result["response"]
        return AIMessage(content="按计算Agent结果应纳税额5000元；筹划前先核对业务条件。")
    planning = child(PlanningAgent, tmp_path, [call("delegate_to_agent", {"agent": "calculation_agent", "task": task}), observe])
    calculation = child(CalculationAgent, tmp_path, [AIMessage(content="按用户条件，应纳税额5000元。")])
    calculation.extract_llm = ScriptedModel([AIMessage(content=json.dumps({"calculator_type": "vat", "params": {
        "taxpayer_type": "general", "sales_amount": 100000, "vat_rate": 0.13, "input_tax": 8000}, "reasoning": "明确条件"}))])
    registry = AgentRegistry(str(tmp_path))
    registry._cache.update(planning_agent=planning, calculation_agent=calculation)
    parent = make_agent(tmp_path, [call("consult_specialists", {"reason": "筹划需要先算税"})])
    parent.registry = registry
    parent.router = SimpleNamespace(classify=lambda query: {"intent": "tax_planning", "agents": ["planning_agent"]})
    with caplog.at_level("INFO"):
        result = execute(parent, "请筹划，先根据明确条件计算税额：" + task)
    assert "5000" in result["final_answer"]
    assert result["reflection"]["calculator_checks"] == 1
    assert any(item["tool"] == "calculate_vat" for item in result["tool_evidence"])
    shared = result["blackboard"]
    assert shared["delegation_count"] == 1
    assert [e["agent"] for e in shared["entries"]] == ["calculation_agent", "planning_agent"]
    assert shared["events"][0] == {"from": "planning_agent", "to": "calculation_agent", "status": "success", "sequence": 1}
    assert "from=planning_agent to=calculation_agent" in caplog.text
    assert task not in caplog.text
    assert agent_workspace.get() is None


def test_simple_specialist_answer_never_forces_delegation(tmp_path):
    planning = child(PlanningAgent, tmp_path, [AIMessage(content="请核对政策条件。")])
    shared = board()
    result = asyncio.run(planning.run("简单筹划说明", {"thread_id": "s", "blackboard": shared,
                                                      "root_session": "workflow",
                                                      "delegate": lambda *args: pytest.fail("不应委托")}))
    assert result == "请核对政策条件。" and shared["delegation_count"] == 0
    assert len(planning.llm.histories) == 1


def test_shared_completed_result_reused_in_same_round(tmp_path):
    registry = AgentRegistry(str(tmp_path))
    target = SimpleNamespace(run=AsyncMock(return_value="应纳税额5000元"))
    registry._cache["calculation_agent"] = target
    shared = board()
    context = {"blackboard": shared, "agent_path": ["planning_agent"], "root_session": "workflow"}
    async def verify():
        first = await registry.delegate("calculation_agent", "算增值税", context)
        second = await registry.delegate("calculation_agent", "算增值税", context)
        assert first["response"] == second["response"] and second["shared_result"]
    asyncio.run(verify())
    assert target.run.await_count == 1 and shared["delegation_count"] == 1


@pytest.mark.parametrize("target,path", [("tax_agent", ["planning_agent"]),
    ("planning_agent", ["planning_agent"]), ("filing_agent", ["planning_agent", "knowledge_agent", "invoice_agent"])])
def test_unregistered_cycle_and_depth_rejected_before_call(tmp_path, target, path):
    registry = AgentRegistry(str(tmp_path))
    shared = board()
    with pytest.raises(ValueError):
        asyncio.run(registry.delegate(target, "子问题", {"blackboard": shared, "agent_path": path, "root_session": "workflow"}))
    assert shared["delegation_count"] == 0 and not shared["entries"]


def test_delegation_budget_cannot_be_bypassed(tmp_path):
    registry = AgentRegistry(str(tmp_path))
    shared = board()
    shared["delegation_count"] = 3
    with pytest.raises(ValueError, match="最多3次"):
        asyncio.run(registry.delegate("calculation_agent", "算税", {"blackboard": shared, "agent_path": ["planning_agent"], "root_session": "workflow"}))


@pytest.mark.parametrize("owner,session", [("other", "workflow"), ("me", "other-session")])
def test_foreign_blackboard_rejected(tmp_path, owner, session):
    registry = AgentRegistry(str(tmp_path))
    shared = board(owner, session)
    with tenant_scope("me"), pytest.raises(PermissionError):
        asyncio.run(registry.delegate("calculation_agent", "算税", {"blackboard": shared, "agent_path": ["planning_agent"], "root_session": "workflow"}))
    assert not shared["entries"]


@pytest.mark.parametrize("response", ["可以虚开发票抵扣", "财税〔2024〕999号允许免税", ""])
def test_unsafe_or_empty_delegated_result_is_structured_failure(tmp_path, response):
    registry = AgentRegistry(str(tmp_path))
    registry._cache["calculation_agent"] = SimpleNamespace(run=AsyncMock(return_value=response))
    shared = board()
    result = asyncio.run(registry.delegate("calculation_agent", "算税", {"blackboard": shared, "agent_path": ["planning_agent"], "root_session": "workflow"}))
    assert result["status"] == "error" and shared["events"][0]["status"] == "error"
    assert not shared["entries"]
    if response:
        assert response not in result.get("message", "")


def test_peer_network_failure_visible_and_no_success_entry(tmp_path):
    registry = AgentRegistry(str(tmp_path))
    registry._cache["calculation_agent"] = SimpleNamespace(run=AsyncMock(side_effect=ConnectionError("secret tax data must not be logged")))
    shared = board()
    result = asyncio.run(registry.delegate("calculation_agent", "算税", {"blackboard": shared, "agent_path": ["planning_agent"], "root_session": "workflow"}))
    assert result["status"] == "error" and result["error_type"] == "ConnectionError"
    assert not shared["entries"] and "secret" not in json.dumps(result)


def test_blackboard_resets_each_new_task_and_is_checkpointed(tmp_path):
    from agents.base_agent import AsyncSqliteSaver
    async def verify():
        async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "board.sqlite")) as saver:
            await saver.setup()
            agent = make_agent(tmp_path, [AIMessage(content="你好"), AIMessage(content="你好")])
            agent._checkpointer = saver
            first = await agent.invoke_messages([HumanMessage(content="你好")], {"thread_id": "workflow"})
            assert first["blackboard"]["session_id"] == "workflow"
            persisted = await agent.workflow_state("workflow")
            assert persisted.values["blackboard"] == first["blackboard"]
            second = await agent.invoke_messages([HumanMessage(content="你好")], {"thread_id": "workflow"})
            assert second["blackboard"]["entries"] == [] and second["blackboard"]["delegation_count"] == 0
    asyncio.run(verify())
