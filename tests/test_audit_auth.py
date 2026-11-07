from tests.auth_helpers import ADMIN_BEARER, ADMIN_CREDENTIAL, USER_BEARER, USER_CREDENTIAL
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from services.network_security import validate_outbound_url, fetch_policy_response

USER = {"Authorization": USER_BEARER}
ADMIN = {"Authorization": ADMIN_BEARER}


@pytest.mark.parametrize("method,path,body", [
    ("get", "/api/settings", None), ("post", "/api/settings", {}),
    ("post", "/api/settings/models", {"section":"llm", "base_url":"http://127.0.0.1"}),
    ("post", "/api/settings/connection", {}), ("get", "/api/settings/qdrant", None),
    ("get", "/api/policy/sources", None), ("post", "/api/policy/update", None),
    ("get", "/api/policy/changes", None), ("put", "/api/policy/schedule", {}),
])
def test_configuration_is_admin_only(method, path, body):
    from web.main import app
    response = TestClient(app).request(method, path, headers=USER, json=body)
    assert response.status_code == 403


def test_authentication_fails_closed_and_cors_rejects_foreign_origin(monkeypatch):
    from web.main import app
    client = TestClient(app)
    assert client.get("/api/me").status_code == 401
    monkeypatch.delenv("AUTH_USER_TOKEN", raising=False)
    monkeypatch.delenv("AUTH_ADMIN_TOKEN", raising=False)
    monkeypatch.delenv("DEVELOPER_TOKEN", raising=False)
    assert client.get("/api/me", headers=USER).status_code == 401
    assert client.get("/api/me", headers=ADMIN).status_code == 401
    response = client.options("/api/settings", headers={"Origin":"http://evil.example", "Access-Control-Request-Method":"POST"})
    assert "access-control-allow-origin" not in response.headers


@pytest.mark.parametrize(("authorization", "detail"), [
    (None, "缺少访问 Token"),
    ("Bearer", "无效的认证格式"),
    ("Basic test_token_123", "无效的认证格式"),
    ("Bearer test_token_123 extra", "无效的认证格式"),
    ("Bearer invalid", "访问 Token 无效或已失效"),
])
def test_authentication_errors_distinguish_missing_malformed_and_invalid(authorization, detail):
    from web.main import app

    headers = {"Authorization": authorization} if authorization else {}
    response = TestClient(app).get("/api/me", headers=headers)
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert detail in response.json()["detail"]


def test_case_insensitive_bearer_and_rotated_credential(monkeypatch):
    from web.main import app

    client = TestClient(app)
    response = client.get("/api/me", headers={"Authorization": USER_BEARER.replace("Bearer", "bearer")})
    assert response.status_code == 200
    assert response.json()["role"] == "user"
    monkeypatch.setenv("AUTH_USER_TOKEN", "rotated-user-fixture")
    assert client.get("/api/me", headers=USER).status_code == 401
    login = client.post("/api/auth/login", json={"credential": "rotated-user-fixture"})
    assert login.status_code == 200
    headers = {"Authorization": "Bearer " + login.json()["access_token"]}
    assert client.get("/api/me", headers=headers).json()["id"] == response.json()["id"]


@pytest.mark.parametrize(("credential", "user_id", "role"), [
    (USER_CREDENTIAL, "user_001", "user"),
    (ADMIN_CREDENTIAL, "user_000", "admin"),
])
def test_login_issues_session_for_existing_identity_and_conversations(credential, user_id, role, monkeypatch):
    from agents.base_agent import _get_shared_memory
    from services.security import tenant_scope
    from web import main

    memory = _get_shared_memory()
    monkeypatch.setattr(main, "conversation_memory", memory)
    with tenant_scope("user_001"):
        memory.ensure_session("ordinary-session", "普通用户会话")
    with tenant_scope("user_000"):
        memory.ensure_session("admin-session", "管理员会话")
    client = TestClient(main.app)
    login = client.post("/api/auth/login", json={"credential": credential})
    assert login.status_code == 200 and login.headers["cache-control"] == "no-store"
    token = login.json()["access_token"]
    assert credential not in token
    assert client.get("/api/me", headers={"Authorization": "Bearer " + credential}).status_code == 401
    headers = {"Authorization": "Bearer " + token}
    identity = client.get("/api/me", headers=headers).json()
    assert (identity["id"], identity["role"]) == (user_id, role)
    conversations = client.get("/api/conversations", headers=headers).json()["conversations"]
    assert [row["session_id"] for row in conversations] == [("ordinary-session" if role == "user" else "admin-session")]
    assert client.get("/api/settings", headers=headers).status_code == (403 if role == "user" else 200)


def test_invalid_credential_tampered_session_and_expiry(monkeypatch):
    from types import SimpleNamespace
    from services import access_sessions
    from web.main import app

    clock = SimpleNamespace(time=lambda: 100_000)
    monkeypatch.setattr(access_sessions, "time", clock)
    client = TestClient(app)
    for credential in ("user_000", "user_001", "invalid-test-credential"):
        response = client.post("/api/auth/login", json={"credential": credential})
        assert response.status_code == 401 and credential not in response.text
    response = client.post("/api/auth/login", json={"credential": {"invalid": USER_CREDENTIAL}})
    assert response.status_code == 422 and USER_CREDENTIAL not in response.text
    issued = client.post("/api/auth/login", json={"credential": USER_CREDENTIAL}).json()
    token = issued["access_token"]
    forged = token.replace("v1.user.", "v1.admin.")
    assert client.get("/api/me", headers={"Authorization": "Bearer " + forged}).status_code == 401
    clock.time = lambda: issued["expires_at"]
    assert client.get("/api/me", headers={"Authorization": "Bearer " + token}).status_code == 401


@pytest.mark.parametrize("url", ["http://127.0.0.1", "https://169.254.169.254/", "https://[::1]/", "https://10.0.0.1/", "file:///private"])
def test_policy_and_model_urls_deny_private_addresses(url):
    with pytest.raises(ValueError):
        validate_outbound_url(url)
    with pytest.raises(ValueError):
        validate_outbound_url(url, model=True)


def test_local_model_requires_operator_origin_allowlist(monkeypatch):
    monkeypatch.setenv("TAX_AGENT_MODEL_ALLOWED_ORIGINS", "http://localhost:11434")
    assert validate_outbound_url("http://localhost:11434/v1/", model=True)
    with pytest.raises(ValueError):
        validate_outbound_url("http://localhost:11435/v1/", model=True)


def test_policy_fetch_rejects_redirect_and_oversize():
    session = MagicMock()
    session.get.return_value.status_code = 302
    with pytest.raises(ValueError, match="重定向"):
        fetch_policy_response(session, "https://example.com/")
    session.get.return_value.status_code = 200
    session.get.return_value.iter_content.return_value = [b"x" * 20]
    with pytest.raises(ValueError, match="2MB"):
        fetch_policy_response(session, "https://example.com/", limit=10)
    assert session.get.call_args.kwargs["allow_redirects"] is False


def test_policy_source_admin_cannot_use_external_directory_or_private_url(tmp_path, monkeypatch):
    from web import main
    monkeypatch.setattr(main, "policy_updater", object())
    client = TestClient(main.app)
    for payload in ({"type":"local", "directory":str(tmp_path)},
                    {"type":"http", "urls":["http://169.254.169.254/metadata"]}):
        response = client.post("/api/policy/sources", headers=ADMIN, json=payload)
        assert response.status_code == 400


def test_saved_public_policy_source_does_not_need_dns_at_startup(monkeypatch):
    from web.main import policy_source_from_config
    def offline(*args, **kwargs):
        raise OSError("offline test")
    monkeypatch.setattr("socket.getaddrinfo", offline)
    source = policy_source_from_config({"source_type":"http", "source_id":"public", "urls":["https://example.com/"]})
    assert source.urls == ["https://example.com/"]


def test_templates_have_no_embedded_access_token():
    for path in (Path(__file__).parents[1] / "web" / "templates").glob("*.html"):
        source = path.read_text(encoding="utf-8")
        assert "admin_token_456" not in source
        assert "test_token_123" not in source
