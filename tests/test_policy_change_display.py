"""政策变更展示：真实文件与 SQLite，HTTP/Cloud 不在本测试中调用。"""
import sqlite3
from datetime import datetime
from urllib.parse import quote

import pytest
from fastapi.testclient import TestClient

from knowledge.policy_updater import LocalDirectorySource, PolicyItem, PolicyTracker, PolicyUpdater


def test_local_heading_date_summary_and_persistence(tmp_path, monkeypatch):
    directory = tmp_path / "policies"
    directory.mkdir()
    path = directory / "demo_vat_2024-01-01.md"
    before = "<!-- 模拟政策 -->\n# 增值税测试政策 ##\n发布日期：2024年2月29日\n增值税税率 13%"
    path.write_text(before, encoding="utf-8-sig")
    updater = PolicyUpdater(str(tmp_path / "tracker.db"))
    updater.add_source(LocalDirectorySource(str(directory), source_id="local_policies"))
    monkeypatch.setattr(updater, "_ingest_item", lambda item: {"status":"success", "chunk_count":1})
    try:
        assert updater.run_once().ingested == 1
        path.write_text(before + "，小规模纳税人减免至 1%", encoding="utf-8-sig")
        assert updater.run_once().changed == 1
        change = updater.tracker.get_changes()[0]
        assert change["title"] == "增值税测试政策"
        assert change["published_date"] == "2024-02-29"
        assert change["source_name"] == "本地政策库" and change["url"] == ""
        assert change["summary"] == "增值税税率 13%，小规模纳税人减免至 1%"
        assert change["updated_at"] != change["published_date"]
        diff = updater.tracker.get_change_diff(change["unique_key"])["diff"]
        assert "-增值税税率 13%\n+增值税税率 13%，小规模纳税人减免至 1%" in diff
        assert updater.run_once().duplicated == 1
        reopened = PolicyTracker(updater.tracker.db_path)
        try:
            assert reopened.get_changes() == [change]
        finally:
            reopened.close()
    finally:
        updater.tracker.close()


def test_web_origin_and_explicit_published_survive_change(tmp_path):
    tracker = PolicyTracker(str(tmp_path / "tracker.db"))
    url = "https://example.gov.cn/content.html?id=2020-01-01&name=%27policy%27"
    before = PolicyItem("chinatax", "2020-01-01", title="测试政策", content="旧规定",
                        published=datetime(2024, 5, 6, 15, 30), url=url)
    after = PolicyItem("chinatax", before.item_id, title="测试政策", content="新规定",
                       published=before.published, url=url)
    try:
        assert before.published == datetime(2024, 5, 6, 15, 30)
        tracker.upsert_pending(before)
        tracker.mark_ingested(before.unique_key)
        tracker.upsert_pending(after)
        change = tracker.get_changes()[0]
        assert change["published_date"] == "2024-05-06" and change["url"] == url
        assert change["source_name"] == "国家税务总局"
        assert change["old_hash"] == before.content_hash and change["new_hash"] == after.content_hash
        custom = PolicyItem("province", "1", title="地方政策", content="旧条文", category="某省税务局")
        tracker.upsert_pending(custom)
        tracker.mark_ingested(custom.unique_key)
        custom.content = "新条文"
        tracker.upsert_pending(custom)
        assert next(c for c in tracker.get_changes() if c["source_id"] == "province")["source_name"] == "某省税务局"
    finally:
        tracker.close()


@pytest.mark.parametrize("old,new,expected", [
    ("保留\n删除旧条文", "保留", "删除：删除旧条文"),
    ("旧", "新" * 120, "新" * 99 + "…"),
    ("旧", "++特殊条文", "++特殊条文"),
    ("保留", "保留\n", "仅空白或格式变更。"),
])
def test_summary_is_real_delta_and_within_100_characters(tmp_path, old, new, expected):
    tracker = PolicyTracker(str(tmp_path / "tracker.db"))
    try:
        item = PolicyItem("mof", "1", content=old)
        tracker.upsert_pending(item)
        tracker.mark_ingested(item.unique_key)
        item.content = new
        tracker.upsert_pending(item)
        change = tracker.get_changes()[0]
        assert change["summary"] == expected and len(change["summary"]) <= 100
        assert change["source_name"] == "财政部"
    finally:
        tracker.close()


def test_legacy_migration_is_repeatable_and_metadata_refresh_keeps_dedup(tmp_path):
    path = tmp_path / "old.db"
    before = PolicyItem("chinatax", "opaque-id", title="旧标题", content="新正文")
    conn = sqlite3.connect(path)
    conn.execute("""CREATE TABLE policy_record (unique_key TEXT PRIMARY KEY, source_id TEXT, item_id TEXT,
                 title TEXT, content_hash TEXT, previous_content_hash TEXT, content_snapshot TEXT,
                 previous_content_snapshot TEXT, status TEXT, file_id TEXT, created_at TEXT)""")
    conn.execute("INSERT INTO policy_record VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                 (before.unique_key, before.source_id, before.item_id, before.title, before.content_hash,
                  "old-hash", "新正文", "旧正文", "ingested", "keep-file-id", "2025-01-01T12:00:00"))
    conn.commit()
    conn.close()
    for _ in range(2):
        tracker = PolicyTracker(str(path))
        assert len(tracker.get_changes()) == 1
        tracker.close()
    tracker = PolicyTracker(str(path))
    try:
        item = PolicyItem("chinatax", before.item_id, title="政策标题", content="新正文",
                          published_date="2024-01-01", url="https://example.gov.cn/policy.html")
        assert tracker.upsert_pending(item) == (False, False)
        change = tracker.get_changes()[0]
        assert change["published_date"] == "2024-01-01" and change["url"] == item.url
        assert change["title"] == item.title and change["summary"] == "新正文"
        assert change["updated_at"] == "2025-01-01T12:00:00" and change["old_hash"] == "old-hash"
        row = tracker._conn.execute("SELECT * FROM policy_record").fetchone()
        assert row["file_id"] == "keep-file-id" and row["status"] == "ingested"
        tracker._conn.execute("UPDATE policy_record SET previous_content_snapshot=NULL")
        tracker._conn.commit()
        assert "缺少旧正文" in tracker.get_changes()[0]["summary"]
        assert "无法还原" in tracker.get_change_diff(item.unique_key)["diff"]
    finally:
        tracker.close()


def test_changes_api_preserves_existing_fields_and_diff(tmp_path, monkeypatch):
    import web.main as main
    updater = PolicyUpdater(str(tmp_path / "tracker.db"))
    try:
        item = PolicyItem("local_policies", r"D:\政策\demo_2024-01-01.md", title="demo_2024-01-01", content="# 真正标题#\n旧")
        updater.tracker.upsert_pending(item)
        updater.tracker.mark_ingested(item.unique_key)
        item.content = "# 真正标题#\n新"
        updater.tracker.upsert_pending(item)
        monkeypatch.setattr(main, "policy_updater", updater)
        monkeypatch.setenv("AUTH_USER_TOKEN", "test_token_123")
        client = TestClient(main.app)
        headers = {"Authorization":"Bearer test_token_123"}
        response = client.get("/api/policy/changes", headers=headers)
        assert response.status_code == 200
        change = response.json()["changes"][0]
        assert {"unique_key","source_id","item_id","title","old_hash","new_hash","snapshot","updated_at"} <= change.keys()
        assert change["published_date"] == "2024-01-01" and change["summary"] == "新"
        assert change["title"] == "真正标题#"
        diff = client.get("/api/policy/changes/" + quote(item.unique_key, safe=""), headers=headers)
        assert diff.status_code == 200 and "-旧\n+新" in diff.json()["diff"]
        assert client.get("/api/policy/changes", headers={"Authorization":"Bearer invalid"}).status_code == 401
    finally:
        updater.tracker.close()
