from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from services.network_security import validate_outbound_url, fetch_policy_response

USER = {"Authorization": "Bearer test_token_123"}
ADMIN = {"Authorization": "Bearer admin_token_456"}


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
