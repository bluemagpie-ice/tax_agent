"""设置接口使用临时配置；网络与密钥按端点隔离，不改真实配置。"""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

USER = {"Authorization": "Bearer test_token_123"}


@pytest.fixture
def settings_app(tmp_path, monkeypatch):
    import services.config_manager as config
    import web.main as main
    monkeypatch.setattr(config, "_CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config, "_CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setenv("AUTH_USER_TOKEN", "test_token_123")
    monkeypatch.setenv("LLM_BASE_URL", config.MODEL_PROVIDERS["GLM"])
    monkeypatch.setenv("LLM_API_KEY", "original-glm-secret")
    monkeypatch.setenv("LLM_MODEL_ID", "glm-test")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-only-secret")
    monkeypatch.setattr(main, "tax_agent", None)
    monkeypatch.setattr(main, "knowledge_agent", None)
    monkeypatch.setattr(main, "knowledge_service", None)
    return main, TestClient(main.app), config


def test_models_use_matching_provider_env_and_never_return_secret(settings_app, monkeypatch):
    _, client, config = settings_app
    request = MagicMock()
    request.return_value.json.return_value = {"data": [{"id":"gemini-2.5-flash"}, {"id":"gemini-2.5-pro"}]}
    monkeypatch.setattr("requests.get", request)
    response = client.post("/api/settings/models", headers=USER,
                           json={"section":"llm", "base_url":config.MODEL_PROVIDERS["Gemini"]})
    assert response.json()["models"] == ["gemini-2.5-flash", "gemini-2.5-pro"]
    assert request.call_args.args[0] == config.MODEL_PROVIDERS["Gemini"] + "models"
    assert request.call_args.kwargs["headers"]["Authorization"] == "Bearer gemini-only-secret"
    assert request.call_args.kwargs["allow_redirects"] is False
    assert "secret" not in response.text


def test_bad_key_and_custom_endpoint_do_not_reuse_other_key(settings_app, monkeypatch):
    _, client, _ = settings_app
    import requests
    request = MagicMock()
    request.return_value.raise_for_status.side_effect = requests.HTTPError("401")
    monkeypatch.setattr("requests.get", request)
    response = client.post("/api/settings/models", headers=USER,
                           json={"section":"llm", "base_url":"https://custom.example/v1", "api_key":"wrong"})
    assert response.json() == {"success":False, "models":[], "message":"❌ 无法连接，请检查配置"}
    request.reset_mock()
    response = client.post("/api/settings/models", headers=USER,
                           json={"section":"llm", "base_url":"https://custom.example/v1"})
    assert not response.json()["success"]
    request.assert_not_called()
    assert client.post("/api/settings/models", headers={"Authorization":"Bearer bad"},
                       json={"section":"llm", "base_url":"https://custom.example"}).status_code == 401


def test_switch_provider_and_custom_model_save_without_old_key(settings_app):
    _, _, config = settings_app
    manager = config.ConfigManager()
    manager.save_config({"llm":{"base_url":config.MODEL_PROVIDERS["Gemini"], "model":"gemini-2.5-flash", "api_key":""}})
    assert manager.get_llm_config()["api_key"] == "gemini-only-secret"
    manager.save_config({"llm":{"base_url":"http://localhost:8000/v1", "model":"arbitrary-model", "api_key":"custom-secret"}})
    assert manager.get_llm_config()["model"] == "arbitrary-model"
    manager.save_config({"llm":{"api_key":""}})
    assert manager.get_llm_config()["api_key"] == "custom-secret"
    with pytest.raises(ValueError, match="API Key"):
        manager.save_config({"llm":{"base_url":"https://unknown.example/v1", "model":"arbitrary"}})
    assert "custom-secret" not in str(manager.get_all_config())
    manager.save_config({"llm":{"provider":"自定义", "base_url":config.MODEL_PROVIDERS["GLM"], "model":"glm-test"}})
    assert manager.get_llm_config()["api_key"] == "original-glm-secret"
    assert manager.get_all_config()["llm"]["provider"] == "自定义"


def test_qdrant_live_count_estimate_warning_and_optional_key(settings_app, monkeypatch):
    _, client, config = settings_app
    manager = config.ConfigManager()
    manager.save_config({"qdrant":{"url":"http://localhost:6333", "vector_limit":100}})
    assert manager.get_qdrant_config()["api_key"] == ""
    cloud = MagicMock()
    cloud.get_collections.return_value.collections = [SimpleNamespace(name="tax_chunks")]
    cloud.get_collection.return_value = SimpleNamespace(status="green", config=SimpleNamespace(
        params=SimpleNamespace(vectors=SimpleNamespace(size=1024))))
    cloud.count.return_value.count = 81
    factory = MagicMock(return_value=cloud)
    monkeypatch.setattr("qdrant_client.QdrantClient", factory)
    data = client.get("/api/settings/qdrant", headers=USER).json()
    assert data["success"] and data["vector_count"] == 81 and data["warning"]
    assert data["estimated_bytes"] == 81 * 1024 * 4
    cloud.count.assert_called_once_with("tax_chunks", exact=True)
    cloud.close.assert_called_once()
    assert factory.call_args.kwargs["api_key"] is None
    cloud.get_collections.side_effect = ConnectionError("offline")
    assert client.get("/api/settings/qdrant", headers=USER).json()["success"] is False


def test_empty_collection_monitor_does_not_create_collection(settings_app, monkeypatch):
    _, client, config = settings_app
    config.ConfigManager().save_config({"qdrant":{"url":"http://localhost:6333"}})
    cloud = MagicMock()
    cloud.get_collections.return_value.collections = []
    monkeypatch.setattr("qdrant_client.QdrantClient", lambda **kwargs: cloud)
    data = client.get("/api/settings/qdrant", headers=USER).json()
    assert data["success"] and data["vector_count"] == 0 and data["usage_ratio"] is None
    cloud.create_collection.assert_not_called()


def test_cloud_free_ram_reference_warns_without_fixed_vector_quota(settings_app, monkeypatch):
    _, client, config = settings_app
    config.ConfigManager().save_config({"qdrant":{"url":"https://demo.eu-central-1.aws.cloud.qdrant.io"}})
    cloud = MagicMock()
    cloud.get_collections.return_value.collections = [SimpleNamespace(name="tax_chunks")]
    cloud.get_collection.return_value = SimpleNamespace(status="green", config=SimpleNamespace(
        params=SimpleNamespace(vectors=SimpleNamespace(size=1024))))
    cloud.count.return_value.count = 150000
    monkeypatch.setattr("qdrant_client.QdrantClient", lambda **kwargs: cloud)
    data = client.get("/api/settings/qdrant", headers=USER).json()
    assert data["reference_ram_bytes"] == 1024 ** 3 and data["vector_limit"] == 0
    assert data["estimated_ram_bytes"] == 150000 * 1024 * 4 * 1.5
    assert data["warning"] and data["usage_ratio"] > .8


def test_invalid_qdrant_save_keeps_previous_configuration(settings_app, monkeypatch):
    _, client, config = settings_app
    cloud = MagicMock()
    cloud.get_collections.side_effect = ConnectionError("invalid credentials")
    monkeypatch.setattr("qdrant_client.QdrantClient", lambda **kwargs: cloud)
    before = config.ConfigManager().get_qdrant_config()
    response = client.post("/api/settings", headers=USER,
                           json={"qdrant":{"url":"http://localhost:6333", "api_key":"wrong"}})
    assert response.status_code == 400
    assert "原配置已保留" in response.json()["detail"]
    assert config.ConfigManager().get_qdrant_config() == before
    assert not config._CONFIG_FILE.exists()
    cloud.close.assert_called_once()
