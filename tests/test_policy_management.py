"""政策管理持久化、动态调度与进度，不连接真实网站或云端。"""
import threading
from types import SimpleNamespace

from fastapi.testclient import TestClient

from knowledge.policy_updater import PolicyUpdater, TaxGovCnSource

USER = {"Authorization":"Bearer test_token_123"}


def test_sources_and_schedule_survive_reconstruction(tmp_path, monkeypatch):
    import web.main as main
    import services.config_manager as config
    monkeypatch.setattr(config, "_CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config, "_CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setenv("AUTH_USER_TOKEN", "test_token_123")
    updater = PolicyUpdater(str(tmp_path / "tracker.db"))
    monkeypatch.setattr(main, "policy_updater", updater)
    client = TestClient(main.app)
    response = client.post("/api/policy/sources", headers=USER,
                           json={"type":"website", "urls":["https://example.gov.cn/policy/list.html"], "source_id":"province", "category":"省税务局"})
    assert response.status_code == 200, response.text
    assert client.get("/api/policy/sources", headers=USER).json()["sources"][0]["listing_url"].startswith("https://example.gov.cn")
    assert client.post("/api/policy/sources", headers=USER,
                       json={"type":"website", "urls":["https://example.gov.cn"], "source_id":"province"}).status_code == 409
    assert client.put("/api/policy/schedule", headers=USER, json={"interval_hours":12}).status_code == 200
    assert updater.get_scheduler_status()["interval_seconds"] == 43200
    cfg = config.ConfigManager().get_policy_config()
    assert cfg["interval_hours"] == 12
    restored = main.policy_source_from_config(cfg["sources"][0])
    assert isinstance(restored, TaxGovCnSource) and restored.source_id == "province"
    assert restored.max_items == 5
    assert client.delete("/api/policy/sources/province", headers=USER).status_code == 200
    assert config.ConfigManager().get_policy_config()["sources"] == []
    assert client.post("/api/policy/sources", headers=USER,
                       json={"type":"website", "urls":["file:///private"]}).status_code == 400
    assert client.put("/api/policy/schedule", headers=USER, json={"interval_hours":0}).status_code == 422
    updater.tracker.close()


def test_dynamic_interval_wakes_actual_scheduler(tmp_path):
    updater = PolicyUpdater(str(tmp_path / "tracker.db"))
    called = threading.Event()
    updater.add_notify_hook(lambda report: called.set())
    try:
        updater.start_scheduler(3600, run_immediately=False)
        assert not called.wait(.05)
        updater.set_interval(.02)
        assert called.wait(1), "新间隔应应用到已运行的线程"
        assert updater.get_report_history()
    finally:
        updater.stop_scheduler()
        updater.tracker.close()


def test_progress_is_visible_while_scan_runs_and_reports_saved(tmp_path, monkeypatch):
    import web.main as main
    monkeypatch.setenv("AUTH_USER_TOKEN", "test_token_123")
    entered, release = threading.Event(), threading.Event()
    def scan():
        entered.set()
        assert release.wait(2)
        return []
    updater = PolicyUpdater(str(tmp_path / "tracker.db"))
    updater.add_source(SimpleNamespace(source_id="test-site", source_type="http", scan=scan))
    monkeypatch.setattr(main, "policy_updater", updater)
    client = TestClient(main.app)
    result = []
    thread = threading.Thread(target=lambda: result.append(client.post("/api/policy/update", headers=USER)))
    thread.start()
    try:
        assert entered.wait(1)
        status = client.get("/api/policy/status", headers=USER).json()["scheduler"]
        assert status["updating"] and status["progress"]["source"] == "test-site"
    finally:
        release.set(); thread.join(2)
    assert result[0].status_code == 200
    assert client.get("/api/policy/reports", headers=USER).json()["reports"][0]["failed"] == 0
    updater.tracker.close()


def test_mof_parser_uses_article_title_and_removes_style(monkeypatch):
    from unittest.mock import MagicMock
    source = TaxGovCnSource("https://szs.mof.gov.cn/zhengcefabu/")
    response = MagicMock()
    response.headers = {"Content-Type":"text/html;charset=utf-8"}
    response.text = '<meta name="ArticleTitle" content="税务政策标题"><meta name="PubDate" content="2024-01-01"><h3>财政部微信</h3><div class="TRS_Editor"><style>.css{bad}</style><p>有效政策正文</p></div>'
    monkeypatch.setattr(source._session, "get", lambda *args, **kwargs: response)
    item = source._fetch_policy_page("https://szs.mof.gov.cn/zhengcefabu/202401/t20240101_1.htm")
    assert item.title == "税务政策标题" and item.content == "有效政策正文"
    assert item.published_date == "2024-01-01"
