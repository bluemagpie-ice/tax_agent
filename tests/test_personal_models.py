"""Personal settings and Research integration with an isolated DB and mock provider."""

from contextlib import asynccontextmanager
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from tests.auth_helpers import ADMIN_BEARER, USER_BEARER

PATH = "/api/settings/personal"
USER = {"Authorization": USER_BEARER}
ADMIN = {"Authorization": ADMIN_BEARER}
BASE = "https://personal-model.example/v1/"
KEY = "personal-key-fixture"
CONFIG = {"model": "personal-model", "base_url": BASE, "api_key": KEY}


@pytest.fixture
def client():
    from web.main import app

    return TestClient(app)


def test_personal_settings_isolation_and_retention(client):
    assert client.get(PATH).status_code == 401
    assert client.get(PATH, headers=ADMIN).json()["config"]["llm"]["api_key_set"] is False
    assert client.put(PATH, headers=USER, json=CONFIG).json() == {"success": True}
    assert client.put(PATH, headers=ADMIN, json={**CONFIG, "model": "admin-personal", "api_key": ""}).status_code == 400
    assert client.put(PATH, headers=ADMIN, json={**CONFIG, "model": "admin-personal", "api_key": "admin-key-fixture"}).status_code == 200
    assert client.put(PATH, headers=USER, json={**CONFIG, "model": "renamed", "api_key": ""}).status_code == 200
    changed_address = {**CONFIG, "base_url": "https://other.example/v1", "api_key": ""}
    assert client.put(PATH, headers=USER, json=changed_address).status_code == 400
    assert client.put(PATH, headers=USER, json={**CONFIG, "base_url": "https://169.254.169.254"}).status_code == 400
    # Subsequent reads use the same persisted row, with no key or masked key fragments.
    response = client.get(PATH + "?owner_id=user_000", headers=USER)
    assert response.headers["cache-control"] == "no-store"
    assert response.json()["config"]["llm"] == {
        "base_url": BASE, "model": "renamed", "provider": "自定义", "api_key_set": True,
    }
    admin = client.get(PATH + "?owner_id=user_001", headers=ADMIN)
    assert admin.json()["config"]["llm"]["model"] == "admin-personal"
    assert KEY not in response.text + admin.text
    assert "admin-key-fixture" not in admin.text
    assert "renamed" not in client.get("/api/settings", headers=ADMIN).text


@pytest.mark.parametrize(("failure", "expected"), [
    (401, "authentication_failed"),
    (403, "model_not_found"),
    (404, "invalid_base_url"),
    (302, "invalid_base_url"),
    (503, "provider_error"),
    ("timeout", "timeout"),
    ("connect", "network_error"),
])
def test_connection_diagnostics_leave_saved_config_intact(client, monkeypatch, caplog, failure, expected):
    requests = []

    def provider(self, request):
        requests.append(request)
        assert request.headers["host"] == "personal-model.example"
        assert request.url.host == "93.184.216.34"
        assert request.extensions["sni_hostname"] == "personal-model.example"
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer new-test-key"
        assert json.loads(request.read())["model"] == "candidate-model"
        if failure == "timeout":
            raise httpx.ReadTimeout("new-test-key")
        if failure == "connect":
            raise httpx.ConnectError("new-test-key")
        code = "model_access_denied" if failure == 403 else "fixture_error"
        return httpx.Response(failure, headers={"location": "https://169.254.169.254"}, json={
            "error": {"message": "provider echoed new-test-key", "code": code},
        })

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", provider)
    assert client.put(PATH, headers=USER, json=CONFIG).status_code == 200
    response = client.post(PATH + "/connection", headers=USER,
                           json={**CONFIG, "model": "candidate-model", "api_key": "new-test-key"})
    assert response.json()["code"] == expected
    assert len(requests) == 1
    assert "new-test-key" not in response.text + caplog.text
    assert client.get(PATH, headers=USER).json()["config"]["llm"]["model"] == CONFIG["model"]


def test_blank_test_key_is_personal_same_endpoint_only(client, monkeypatch):
    seen = []

    def provider(self, request):
        seen.append(request.headers["Authorization"])
        return httpx.Response(200, json={"model": "candidate-model", "choices": [
            {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "OK"}},
        ]})

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", provider)
    assert client.put(PATH, headers=USER, json=CONFIG).status_code == 200
    candidate = {**CONFIG, "model": "candidate-model", "api_key": ""}
    assert client.post(PATH + "/connection", headers=USER, json=candidate).json()["code"] == "connected"
    assert client.post(PATH + "/connection", headers=ADMIN, json=candidate).json()["code"] == "configuration_missing"
    assert client.post(PATH + "/connection", headers=USER,
                       json={**candidate, "base_url": "https://other.example/v1"}).json()["code"] == "configuration_missing"
    assert seen == ["Bearer " + KEY]


def test_private_dns_rebinding_is_blocked_before_request(client, monkeypatch):
    import socket

    assert client.put(PATH, headers=USER, json=CONFIG).status_code == 200
    addresses = iter(("93.184.216.34", "169.254.169.254"))

    def changing_dns(host, port, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (next(addresses), port))]

    def blocked(self, request):
        pytest.fail("Model HTTP transport must never receive a private rebound target")

    monkeypatch.setattr(socket, "getaddrinfo", changing_dns)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", blocked)
    response = client.post(PATH + "/connection", headers=USER, json=CONFIG)
    assert response.json()["code"] == "network_error"
    assert KEY not in response.text


def test_research_stream_uses_rotated_personal_key_and_selected_model(tmp_path, monkeypatch, caplog):
    from agents.research_agent import ResearchAgent
    from database.models import engine
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    from services.research_store import ResearchStore
    from web import main

    agent = ResearchAgent(ResearchStore(engine), workdir_root=tmp_path)
    rotated = "rotated-personal-fixture"
    seen = []

    def provider(self, request):
        payload = json.loads(request.read())
        seen.append((payload["model"], request.headers["Authorization"]))
        assert payload["messages"][0]["content"].startswith("输出类型：Understanding\n")
        assert "response_format" not in payload
        output = {"facts": [], "questions": ["需核对"], "unknowns": ["所在地待确认 " + rotated]}
        return httpx.Response(200, json={"model": payload["model"], "choices": [
            {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": json.dumps(output)}},
        ]})

    @asynccontextmanager
    async def lifespan(app):
        async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "research.sqlite")) as saver:
            agent._checkpointer = saver
            yield

    monkeypatch.setattr(main.app.state, "research_service", agent, raising=False)
    monkeypatch.setattr(main.app.router, "lifespan_context", lifespan)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", provider)
    with TestClient(main.app) as client:
        assert client.put(PATH, headers=ADMIN, json={**CONFIG, "api_key": "admin-fixture", "model": "admin-model"}).status_code == 200
        assert client.put(PATH, headers=USER, json=CONFIG).status_code == 200
        assert client.put(PATH, headers=USER, json={**CONFIG, "api_key": rotated}).status_code == 200
        task = client.post("/api/research/tasks", headers=USER, json={"query": "模拟研究", "use_cases": False}).json()
        route = f'/api/research/tasks/{task["task_id"]}'
        response = client.post(route + "/stream", headers=USER, json={"revision": task["revision"]})
        assert response.status_code == 200
        events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
        assert events[-1]["task"]["status"] == "awaiting_facts"
        assert seen == [(CONFIG["model"], "Bearer " + rotated)]
        assert rotated not in response.text + client.get(route, headers=USER).text + caplog.text
        assert client.get(route, headers=ADMIN).status_code == 404
