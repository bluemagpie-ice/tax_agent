"""政策流水线回归：真实扫描/SQLite/调度，云端入库和 HTTP 故障使用可控替身。"""
import logging
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from bs4 import BeautifulSoup

from knowledge.policy_updater import (
    LocalDirectorySource, PolicyItem, PolicyTracker, PolicyUpdater, TaxGovCnSource, logging_notify_hook,
)


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    calls = []
    cloud = SimpleNamespace(qdrant=object())

    async def ingest(path, **kwargs):
        calls.append((Path(path).read_text(encoding="utf-8"), kwargs))
        return {"status": "success", "chunk_count": 1, "file_id": kwargs["file_id"]}

    kb = SimpleNamespace(ingest_file=ingest, _ensure_embedding=lambda: cloud)
    monkeypatch.setattr("knowledge.kb.KnowledgeBaseFactory.create", lambda *args, **kwargs: kb)
    directory = tmp_path / "policies"
    directory.mkdir()
    updater = PolicyUpdater(tracker_db=str(tmp_path / "tracker.db"), kb_work_dir=str(tmp_path / "kb"),
                            notify_hooks=[logging_notify_hook])
    updater.add_source(LocalDirectorySource(str(directory)))
    yield updater, directory, calls, cloud, kb
    updater.stop_scheduler()
    updater.tracker.close()


def test_full_flow_change_diff_and_retry(pipeline, caplog):
    updater, directory, calls, cloud, _ = pipeline
    path = directory / "税率_2024-01-01.md"
    path.write_text("增值税税率 13%", encoding="utf-8")
    with caplog.at_level(logging.INFO):
        first = updater.run_once()
        second = updater.run_once()
    assert first.ingested == 1 and second.duplicated == 1 and second.ingested == 0
    assert len(calls) == 1 and len(updater.get_report_history()) == 2
    assert sum("[政策更新报告]" in r.message for r in caplog.records) == 2
    old = updater._sources[0].scan()[0]
    path.write_text("增值税税率 13%，小规模纳税人减免至 1%", encoding="utf-8")
    cloud.qdrant = None
    failed = updater.run_once()
    assert failed.failed == 1 and failed.ingested == 0 and "Qdrant" in failed.errors[0]
    assert not updater.tracker.exists(updater._sources[0].scan()[0])
    cloud.qdrant = object()
    changed = updater.run_once()
    assert changed.changed == changed.ingested == 1 and changed.duplicated == 0
    assert len(calls) == 2 and calls[0][1]["file_id"] != calls[1][1]["file_id"]
    assert calls[1][1]["source_type"] == "default" and calls[1][1]["scope"] == "system"
    diff = updater.tracker.get_change_diff(old.unique_key)
    assert diff["old_hash"] == old.content_hash and diff["old_snapshot_available"]
    assert "-增值税税率 13%" in diff["diff"] and "+增值税税率 13%，小规模纳税人减免至 1%" in diff["diff"]
    assert updater.get_latest_report()["changed"] == 1
    reopened = PolicyTracker(updater.tracker.db_path)
    try:
        assert reopened.get_change_diff(old.unique_key) == diff
    finally:
        reopened.close()


def test_same_batch_hash_dedup_and_failed_result(pipeline):
    updater, directory, calls, _, kb = pipeline
    for name in ("a.md", "b.md"):
        (directory / name).write_text("相同正文", encoding="utf-8")

    async def failed_ingest(*args, **kwargs):
        return {"status": "error", "message": "网络不可用"}

    success = kb.ingest_file
    kb.ingest_file = failed_ingest
    report = updater.run_once()
    assert report.failed == 1 and report.duplicated == 1 and report.ingested == 0
    assert "网络不可用" in report.errors[0]
    kb.ingest_file = success
    assert updater.run_once().ingested == 1
    assert updater.run_once().duplicated == 2 and len(calls) == 1


def test_date_window_and_unknown_date(pipeline):
    updater, directory, calls, _, _ = pipeline
    updater.since_date = __import__("datetime").date(2022, 1, 1)
    for filename, text in [("policy_2020-01-01.md", "2020旧政策"),
                           ("new.md", "发布日期：2024年01月01日\n2024新政策"), ("unknown.md", "无日期")]:
        (directory / filename).write_text(text, encoding="utf-8")
    report = updater.run_once()
    assert report.scanned == 3 and report.filtered == 2 and report.ingested == 1
    assert "缺少发布日期" in report.errors[0]
    assert "2024新政策" in calls[0][0]
    assert updater.get_latest_report()["filtered"] == 2


@pytest.mark.parametrize("kwargs,expected", [
    ({"item_id": "policy_2024_02_29.md"}, "2024-02-29"),
    ({"item_id": "x", "content": "发布日期：2020/12/31"}, "2020-12-31"),
    ({"item_id": "x", "published_date": "2022-01-01"}, "2022-01-01"),
    ({"item_id": "x"}, None),
])
def test_published_date_parse(kwargs, expected):
    assert PolicyItem(source_id="s", **kwargs).published_date == expected


def test_invalid_date_and_since_date(tmp_path):
    with pytest.raises(ValueError):
        PolicyItem("s", "policy_2024-02-30.md")
    with pytest.raises(ValueError):
        PolicyUpdater(tracker_db=str(tmp_path / "x.db"), since_date="invalid")


def test_bad_file_reported_without_blocking_valid_file(pipeline):
    updater, directory, calls, _, _ = pipeline
    (directory / "empty.md").write_text("", encoding="utf-8")
    (directory / "good.md").write_text("有效政策", encoding="utf-8")
    report = updater.run_once()
    assert report.failed == 1 and report.ingested == 1 and "empty.md" in report.errors[0]
    assert len(calls) == 1


def test_empty_sources_and_concurrent_manual_scheduler(pipeline):
    updater, directory, calls, _, _ = pipeline
    assert updater.run_once().scanned == 0
    updater._sources.clear()
    assert updater.run_once().ingested == 0 and len(updater.get_report_history()) == 2
    (directory / "a.md").write_text("并发政策", encoding="utf-8")
    updater.add_source(LocalDirectorySource(str(directory)))
    updater.start_scheduler(interval_seconds=3600)
    with ThreadPoolExecutor(max_workers=4) as executor:
        reports = list(executor.map(lambda _: updater.run_once(), range(8)))
    updater.stop_scheduler()
    assert not updater.get_scheduler_status()["running"]
    assert len(calls) == 1 and all(r.failed == 0 for r in reports)
    assert len(updater.get_report_history()) == 11


def test_tracker_thread_transactions_and_schema_migration(tmp_path):
    path = tmp_path / "old.db"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE policy_record (unique_key TEXT PRIMARY KEY, source_id TEXT, item_id TEXT, title TEXT, content_hash TEXT, status TEXT, file_id TEXT, created_at TEXT)")
    connection.execute("INSERT INTO policy_record VALUES ('old::1','old','1','保留旧行','hash','ingested','f','2020')")
    connection.commit()
    connection.close()
    tracker = PolicyTracker(str(path))
    try:
        def write(index):
            item = PolicyItem("thread", str(index), content=f"政策{index}")
            tracker.upsert_pending(item)
            tracker.mark_ingested(item.unique_key)
            tracker.get_pending()
            return tracker.exists(item)
        with ThreadPoolExecutor(max_workers=8) as executor:
            assert all(executor.map(write, range(40)))
        assert tracker._conn.execute("SELECT COUNT(*) FROM policy_record").fetchone()[0] == 41
        assert tracker._conn.execute("SELECT title FROM policy_record WHERE unique_key='old::1'").fetchone()[0] == "保留旧行"
    finally:
        tracker.close()


def test_full_snapshots_failed_draft_and_legacy_diff(tmp_path):
    tracker = PolicyTracker(str(tmp_path / "x.db"))
    try:
        old = PolicyItem("s", "x", content="前文\n" * 1100 + "旧税率")
        tracker.upsert_pending(old)
        tracker.mark_ingested(old.unique_key)
        draft = PolicyItem("s", "x", content="失败草稿")
        tracker.upsert_pending(draft)
        tracker.mark_failed(draft.unique_key)
        new = PolicyItem("s", "x", content=old.content.replace("旧税率", "新税率"))
        tracker.upsert_pending(new)
        diff = tracker.get_change_diff(new.unique_key)
        assert diff["old_hash"] == old.content_hash and "-旧税率" in diff["diff"] and "+新税率" in diff["diff"]
        tracker._conn.execute("UPDATE policy_record SET previous_content_snapshot=NULL")
        tracker._conn.commit()
        assert not tracker.get_change_diff(new.unique_key)["old_snapshot_available"]
        assert "无法还原" in tracker.get_change_diff(new.unique_key)["diff"]
    finally:
        tracker.close()


def test_tax_gov_selector_adaptation_and_diagnostics():
    source = TaxGovCnSource("https://example.com/list.html", link_selector=".obsolete", title_selector=".old-title", content_selector=".old-body")
    assert source._build_page_url(1) == "https://example.com/list_1.html"
    assert source._select_links(BeautifulSoup('<a href="/a/content.html">政策</a>', "lxml"))
    with pytest.raises(ValueError, match="link_selector.*obsolete"):
        source._select_links(BeautifulSoup("<body>无链接</body>", "lxml"))
    response = MagicMock()
    response.headers = {}
    response.text = '<meta name="ArticleTitle" content="标题"><meta name="PubDate" content="2024-01-01 12:00:00"><div class="content">分享<div class="article">正文</div></div>'
    source._session.get = MagicMock(return_value=response)
    item = source._fetch_policy_page("https://example.com/a/content.html")
    assert item.title == "标题" and item.content == "正文" and item.published_date == "2024-01-01"
    response.text = "<h1>标题</h1><body>只有导航</body>"
    with pytest.raises(ValueError, match="content_selector.*old-body"):
        source._fetch_policy_page(item.url)
    response.text = '<div class="article">正文</div>'
    with pytest.raises(ValueError, match="title_selector.*old-title"):
        source._fetch_policy_page(item.url)
    response.raise_for_status.side_effect = RuntimeError("HTTP 403")
    assert source.scan() == [] and "HTTP 403" in source.last_errors[0]


def test_runtime_error_is_not_ingested_twice(pipeline):
    updater, directory, _, _, kb = pipeline
    (directory / "a.md").write_text("正文", encoding="utf-8")
    attempts = []
    async def broken(*args, **kwargs):
        attempts.append(1)
        raise RuntimeError("Cloud unavailable")
    kb.ingest_file = broken
    assert updater.run_once().failed == 1
    assert len(attempts) == 1


def test_stop_does_not_claim_a_busy_thread_has_exited(pipeline):
    updater, _, _, _, _ = pipeline
    busy = SimpleNamespace(join=MagicMock(), is_alive=lambda: True)
    updater._scheduler_thread = busy
    try:
        with pytest.raises(RuntimeError, match="尚未退出"):
            updater.stop_scheduler()
        assert updater._stop_event.is_set() and updater._scheduler_thread is busy
    finally:
        updater._scheduler_thread = None


def test_changes_api_accepts_windows_path(pipeline, monkeypatch):
    import web.main as main
    from fastapi.testclient import TestClient
    from urllib.parse import quote
    updater, directory, _, _, _ = pipeline
    path = directory / "政策'文档.md"
    path.write_text("旧正文", encoding="utf-8")
    updater.run_once()
    path.write_text("新正文", encoding="utf-8")
    updater.run_once()
    monkeypatch.setattr(main, "policy_updater", updater)
    monkeypatch.setenv("AUTH_USER_TOKEN", "test_token_123")
    key = updater.tracker.get_changes()[0]["unique_key"]
    client = TestClient(main.app)
    response = client.get("/api/policy/changes/" + quote(key, safe=""), headers={"Authorization": "Bearer test_token_123"})
    assert response.status_code == 200
    assert response.json()["unique_key"] == key and "-旧正文\n+新正文" in response.json()["diff"]
    assert client.get("/api/policy/changes/" + quote(key, safe=""), headers={"Authorization": "Bearer invalid-token"}).status_code == 401


def test_lifespan_stops_actual_scheduler_and_closes_tracker(tmp_path, monkeypatch):
    import web.main as main
    import agents.base_agent as base
    import database.models as models
    from agents.memory import ConversationMemory
    from fastapi.testclient import TestClient
    from sqlalchemy import create_engine
    monkeypatch.setattr(main, "BASE_DIR", tmp_path)
    monkeypatch.setitem(main.PATHS, "data", str(tmp_path / "data"))
    monkeypatch.setattr(main.ConfigManager, "get_llm_config", lambda self: {"model": "", "api_key": "", "base_url": ""})
    monkeypatch.setattr(base, "_conversation_memory", ConversationMemory(base_dir=tmp_path / "memory"))
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    monkeypatch.setattr(models, "engine", engine)
    monkeypatch.delenv("TAX_CN_LISTING_URL", raising=False)
    with TestClient(main.app) as client:
        updater = main.app.state.policy_updater
        thread = updater._scheduler_thread
        assert thread is not None and client.get("/training").status_code == 200
    assert updater._stop_event.is_set() and not thread.is_alive()
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        updater.tracker.get_reports()
    engine.dispose()
