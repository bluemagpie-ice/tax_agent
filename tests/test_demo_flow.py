"""演示闭环回归：真实图编译、SSE 编排、配置与网络异常；外部 LLM 使用替身。"""
import asyncio
import base64
import importlib
import io
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import UploadFile
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    main = importlib.import_module("web.main")
    import services.config_manager as config
    monkeypatch.setattr(config, "_CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config, "_CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setenv("AUTH_USER_TOKEN", "test_token_123")
    fake = SimpleNamespace(run=AsyncMock(return_value='答复含 "引号"\n第二行'), registry=SimpleNamespace(_cache={}))
    monkeypatch.setattr(main, "tax_agent", fake)
    monkeypatch.setattr(main, "knowledge_agent", SimpleNamespace())
    monkeypatch.setattr(main, "conversation_memory", MagicMock())
    return main, TestClient(main.app), fake


def test_sse_uses_orchestrator_and_preserves_text(runtime):
    main, client, agent = runtime
    resp = client.post("/api/chat/stream", headers={"Authorization": "Bearer test_token_123"},
                       json={"user_id": "u", "query": "进项税是什么", "session_id": "demo", "web_search": True})
    assert resp.status_code == 200
    assert "event: done" in resp.text
    token = next(line[6:] for block in resp.text.split("\n\n") if block.startswith("event: token")
                 for line in block.splitlines() if line.startswith("data: "))
    assert json.loads(token)["text"] == '答复含 "引号"\n第二行'
    assert agent.run.await_args.kwargs["context"]["web_search"] is True
    main.conversation_memory.record_turn.assert_not_called()  # run 负责记录，路由不再重复记录


@pytest.mark.parametrize("endpoint", ["/api/chat", "/api/chat/stream"])
def test_blank_query_rejected(runtime, endpoint):
    _, client, agent = runtime
    resp = client.post(endpoint, headers={"Authorization": "Bearer test_token_123"},
                       json={"user_id": "u", "query": "  "})
    assert resp.status_code == 400
    agent.run.assert_not_called()


def test_config_mask_and_invalid_url_do_not_overwrite_key(tmp_path, monkeypatch):
    import services.config_manager as config
    monkeypatch.setattr(config, "_CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config, "_CONFIG_FILE", tmp_path / "config.json")
    manager = config.ConfigManager()
    manager.save_config({"llm": {"api_key": "demo-secret-key", "model": "custom-model"}})
    masked = manager.get_all_config()["llm"]["api_key"]
    manager.save_config({"llm": {"api_key": masked, "model": "next-model"}})
    assert manager.get_llm_config()["api_key"] == "demo-secret-key"
    with pytest.raises(ValueError):
        manager.save_config({"llm": {"base_url": "not-a-url"}})
    assert manager.get_llm_config()["model"] == "next-model"


def test_settings_refreshes_existing_client(runtime):
    _, client, agent = runtime
    agent.llm = SimpleNamespace(temperature=0.3)
    resp = client.post("/api/settings", headers={"Authorization": "Bearer test_token_123"},
                       json={"llm": {"api_key": "demo-secret-key", "model": "custom-model", "base_url": "https://example.com/v1"}})
    assert resp.status_code == 200
    assert agent.llm.model_name == "custom-model"
    assert agent.llm.openai_api_key.get_secret_value() == "demo-secret-key"


def test_real_policy_graph_compiles_and_runs(tmp_path):
    from agents.base_agent import BaseAgent
    from agents.sub_agents.policy_agent import PolicyAgent
    agent = PolicyAgent.__new__(PolicyAgent)
    BaseAgent.__init__(agent, workdir_root=str(tmp_path))
    agent._ensure_checkpointer = AsyncMock()
    agent.retrieve_context = MagicMock(return_value="政策依据")
    agent.llm = MagicMock()
    agent.llm.invoke.return_value = AIMessage(content="图已完成答复")
    assert asyncio.run(agent.run("政策说明", {"thread_id": "demo"})) == "图已完成答复"


@pytest.mark.parametrize("found", [True, False])
def test_web_search_injected_and_memory_keeps_original_query(tmp_path, found):
    from agents.base_agent import BaseAgent
    from agents.tax_agent import TaxAgent
    agent = TaxAgent.__new__(TaxAgent)
    BaseAgent.__init__(agent, workdir_root=str(tmp_path))
    agent.get_memory_context = MagicMock(return_value={"prompt_text": "历史背景"})
    agent.record_turn = MagicMock()
    agent.invoke_messages = AsyncMock(return_value={"final_answer": "回答"})
    results = [{"title": "政策标题", "url": "https://www.chinatax.gov.cn/demo", "snippet": "检索依据"}] if found else []
    with patch("services.tools.kb_tools.search_web_policy", return_value=results) as search, \
         patch("services.tools.kb_tools.read_web_page", return_value={"status": "success", "content": "检索依据", "url": "https://www.chinatax.gov.cn/demo"}):
        answer = asyncio.run(agent.run("什么是进项税", {"thread_id": "demo", "web_search": True}))
    search.assert_any_call("什么是进项税", source="gov_cn", limit=3)
    assert search.call_count == (1 if found else 2)
    assert agent.record_turn.call_args.kwargs["user_message"] == "什么是进项税"
    assert agent.record_turn.call_args.kwargs["assistant_message"] == answer
    if found:
        assert "检索依据" in agent.invoke_messages.await_args.args[0][0].content
        assert "https://www.chinatax.gov.cn/demo" in answer
    else:
        assert "未核验最新政策" in answer


def test_bing_redirect_decoded_and_fake_gov_host_rejected():
    from services.tools.kb_tools import search_web_policy
    url = "https://www.chinatax.gov.cn/demo"
    encoded = "a1" + base64.urlsafe_b64encode(url.encode()).decode().rstrip("=")
    html = f'<li class="b_algo"><h2><a href="https://www.bing.com/ck/a?u={encoded}">政策</a></h2><p>摘要</p></li>'
    html += '<li class="b_algo"><h2><a href="https://evil.example/gov.cn">伪官网</a></h2></li>'
    with patch("requests.get", return_value=SimpleNamespace(text=html, raise_for_status=lambda: None)):
        results = search_web_policy("税务政策", source="gov_cn")
    assert [r["url"] for r in results] == [url]


def test_upload_path_stays_in_upload_dir(runtime, tmp_path):
    main, _, _ = runtime
    service = main.KnowledgeService.__new__(main.KnowledgeService)
    service.upload_dir = str(tmp_path)
    file = UploadFile(filename="../../outside.md", file=io.BytesIO(b"tax policy"))
    path = service._save_file(file)
    assert path.parent == tmp_path
    assert path.read_bytes() == b"tax policy"
    with pytest.raises(ValueError, match="空文件"):
        service._save_file(UploadFile(filename="empty.md", file=io.BytesIO(b"")))


def test_image_parser_default_argument_and_ocr_failure(tmp_path):
    from knowledge.parser import ImageParser
    path = tmp_path / "demo.png"
    path.write_bytes(b"invalid image")
    parser = ImageParser(str(path))
    with patch.object(parser, "_get_ocr_engine", side_effect=ImportError("missing runtime")):
        assert "OCR 错误" in parser.parse()


def test_missing_config_real_lifespan_starts_and_stops(tmp_path, monkeypatch):
    main = importlib.import_module("web.main")
    import agents.base_agent as base
    from agents.memory import ConversationMemory
    from sqlalchemy import create_engine
    import database.models as models
    monkeypatch.setattr(main, "BASE_DIR", tmp_path)
    monkeypatch.setitem(main.PATHS, "data", str(tmp_path / "data"))
    monkeypatch.setattr(main.ConfigManager, "get_llm_config", lambda self: {"model": "", "api_key": "", "base_url": ""})
    monkeypatch.setattr(base, "_conversation_memory", ConversationMemory(base_dir=tmp_path / "memory"))
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    monkeypatch.setattr(models, "engine", engine)
    monkeypatch.delenv("TAX_CN_LISTING_URL", raising=False)
    with TestClient(main.app) as client:
        assert client.get("/").status_code == 200
        response = client.post("/api/chat", headers={"Authorization": "Bearer test_token_123"},
                               json={"user_id": "u", "query": "税务问题"})
        assert response.status_code == 503
    engine.dispose()
