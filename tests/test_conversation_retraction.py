import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from agents.memory import ConversationMemory
from services.security import tenant_scope


USER = {"Authorization": "Bearer test_token_123"}
OTHER = {"Authorization": "Bearer admin_token_456"}


@pytest.fixture
def client_and_memory(tmp_path, monkeypatch):
    import web.main as main
    monkeypatch.setenv("AUTH_USER_TOKEN", "test_token_123")
    monkeypatch.setenv("DEVELOPER_TOKEN", "admin_token_456")
    memory = ConversationMemory(tmp_path / "conversation")
    monkeypatch.setattr(main, "conversation_memory", memory)
    monkeypatch.setattr(main, "tax_agent", None)
    return TestClient(main.app), memory


def test_retract_only_last_pair_with_revision_and_reopen(client_and_memory):
    client, memory = client_and_memory
    with tenant_scope("user_001"):
        for query, answer in [("older question", "older answer"), ("last question", "last answer")]:
            memory.record_turn("saved", query, answer)
    base = "/api/conversations/saved/messages"
    before = client.get(base, headers=USER).json()
    assert before["can_retract"]
    assert client.delete(base + "/last", headers=OTHER).status_code == 404
    result = client.delete(base + "/last", params={"revision": before["revision"]}, headers=USER)
    assert result.status_code == 200 and result.json()["can_retract"]
    assert client.delete(base + "/last", params={"revision": before["revision"]}, headers=USER).status_code == 409
    with tenant_scope("user_001"):
        reopened = ConversationMemory(memory.base_dir)
        current = reopened.get_session("saved")
        assert [m["content"] for m in current["messages"]] == ["older question", "older answer"]
        assert current["turns"] == 1
        prompt = reopened.context_for("saved", "follow-up")
        assert prompt["history_retracted"] and "last answer" not in prompt["prompt_text"]


def test_empty_pending_and_missing_have_distinct_status(client_and_memory):
    client, memory = client_and_memory
    with tenant_scope("user_001"):
        memory.create_session("empty")
        memory.ensure_session("pending", "still waiting")
    for session_id, status in (("empty", 409), ("pending", 409), ("missing", 404)):
        result = client.delete(f"/api/conversations/{session_id}/messages/last", headers=USER)
        assert result.status_code == status and result.json()["detail"]
    assert client.get("/api/conversations/pending/messages", headers=USER).json()["messages"][0]["pending"]


def test_retraction_cannot_run_during_chat_or_approval(client_and_memory, monkeypatch):
    import web.main as main
    client, memory = client_and_memory
    with tenant_scope("user_001"):
        memory.record_turn("workflow", "question", "waiting")
    url = "/api/conversations/workflow/messages/last"
    lock = asyncio.Lock()
    asyncio.run(lock.acquire())
    agent = SimpleNamespace(conversation_lock=lambda _: lock, pending_workflow=AsyncMock(return_value={"kind":"approval"}))
    monkeypatch.setattr(main, "tax_agent", agent)
    assert client.delete(url, headers=USER).status_code == 409
    agent.pending_workflow.assert_not_called()
    lock.release()
    assert client.delete(url, headers=USER).status_code == 409
    assert len(client.get(url.rsplit("/", 1)[0], headers=USER).json()["messages"]) == 2


def test_retracted_checkpoint_messages_are_not_used_as_model_history():
    from agents.base_agent import BaseAgent
    from langchain_core.messages import HumanMessage
    from langgraph.graph import add_messages

    old = HumanMessage("removed question")
    new = HumanMessage("next question")
    assert add_messages([old], BaseAgent.checkpoint_input([new], {"history_retracted":True})) == [new]


def test_retract_chat_does_not_change_research_task_or_published_case(client_and_memory, tmp_path):
    from sqlalchemy import create_engine
    from services.research_cases import CaseStore
    from services.research_store import ResearchStore
    from tests.test_research_cases import content, publish

    client, memory = client_and_memory
    engine = create_engine(f"sqlite:///{tmp_path / 'research.sqlite'}")
    try:
        research, cases = ResearchStore(engine), CaseStore(engine)
        with tenant_scope("user_001"):
            task = research.create("independent research")
            case = publish(cases, cases.create(content()))
            memory.record_turn(task["task_id"], "question", "answer")
        assert client.delete(f"/api/conversations/{task['task_id']}/messages/last", headers=USER).status_code == 200
        with tenant_scope("user_001"):
            assert research.get(task["task_id"]) == task
            assert cases.get(case["case_id"]) == case
    finally:
        engine.dispose()
