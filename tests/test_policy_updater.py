"""
T14 — PolicyTracker 去重逻辑测试

重点：exists / upsert_pending / mark_ingested 的各种状态组合
"""

import tempfile
from pathlib import Path

import pytest

from knowledge.policy_updater import PolicyTracker, PolicyItem

# ================================================================== fixture
@pytest.fixture
def tracker(tmp_path):
    """每个测试用独立的临时 SQLite 文件"""
    db_path = str(tmp_path / "tracker.sqlite3")
    t = PolicyTracker(db_path)
    yield t
    t.close()

def make_item(
    source_id="gov_cn",
    item_id="doc_001",
    title="增值税法实施条例",
    content="这是增值税法的具体条文内容...",
):
    """快速构造 PolicyItem 的工厂"""
    return PolicyItem(
        source_id=source_id,
        item_id=item_id,
        title=title,
        content=content,
    )

# ================================================================== 1. PolicyItem 属性辅助
class TestPolicyItemProps:
    def test_unique_key_format(self):
        item = make_item(source_id="src", item_id="abc")
        assert item.unique_key == "src::abc"

    def test_content_hash_is_sha256(self):
        item = make_item(content="hello world")
        # 手动算一遍
        import hashlib
        expected = hashlib.sha256(b"hello world").hexdigest()
        assert item.content_hash == expected

    def test_content_hash_falls_back_to_title(self):
        """content 为空时，content_hash 用 title 计算"""
        item = make_item(content="", title="仅标题")
        import hashlib
        expected = hashlib.sha256("仅标题".encode("utf-8")).hexdigest()
        assert item.content_hash == expected

    def test_content_hash_empty_when_both_empty(self):
        item = make_item(content="", title="")
        import hashlib
        expected = hashlib.sha256(b"").hexdigest()
        assert item.content_hash == expected

    def test_same_content_same_hash(self):
        a = make_item(content="same text")
        b = make_item(content="same text")
        assert a.content_hash == b.content_hash

    def test_different_content_different_hash(self):
        a = make_item(content="aaa")
        b = make_item(content="bbb")
        assert a.content_hash != b.content_hash

# ================================================================== 2. exists() 两级去重
class TestExists:
    """exists 只拦截 status='ingested' 的记录"""

    # ---------- 新 item → False ----------
    def test_fresh_item_not_exists(self, tracker):
        item = make_item()
        assert tracker.exists(item) is False

    # ---------- 一级：unique_key 命中 ----------
    def test_unique_key_ingested_exists(self, tracker):
        item = make_item()
        tracker.upsert_pending(item)
        tracker.mark_ingested(item.unique_key, file_id="f_001")
        # 再来同一个
        assert tracker.exists(make_item()) is True

    def test_unique_key_pending_not_exists(self, tracker):
        """同 unique_key 但还没 ingested → exists 返回 False（不拦截 pending）"""
        item = make_item()
        tracker.upsert_pending(item)  # 只 upsert_pending，不 mark_ingested
        assert tracker.exists(make_item()) is False

    def test_unique_key_failed_not_exists(self, tracker):
        tracker.upsert_pending(item := make_item())
        tracker.mark_failed(item.unique_key)
        assert tracker.exists(make_item()) is False

    # ---------- 二级：content_hash 命中 ----------
    def test_content_hash_ingested_exists(self, tracker):
        """unique_key 不同但 content 相同 → ingested 后 exists 拦截"""
        item_a = make_item(item_id="url_a", content="完全相同的内容")
        tracker.upsert_pending(item_a)
        tracker.mark_ingested(item_a.unique_key)

        item_b = make_item(item_id="url_b", content="完全相同的内容")
        assert item_b.unique_key != item_a.unique_key
        assert item_b.content_hash == item_a.content_hash
        assert tracker.exists(item_b) is True

    def test_content_hash_pending_not_exists(self, tracker):
        """content 相同但只 upsert_pending → 不拦截"""
        tracker.upsert_pending(make_item(item_id="url_a", content="same"))
        assert tracker.exists(make_item(item_id="url_b", content="same")) is False

    # ---------- 混合场景 ----------
    def test_both_key_and_hash_different_not_exists(self, tracker):
        tracker.upsert_pending(make_item(item_id="a", content="c1"))
        tracker.mark_ingested("gov_cn::a")
        assert tracker.exists(make_item(item_id="b", content="c2")) is False

# ================================================================== 3. upsert_pending
class TestUpsertPending:
    def test_first_insert_works(self, tracker):
        item = make_item()
        tracker.upsert_pending(item)
        pending = tracker.get_pending()
        assert len(pending) == 1
        row = pending[0]
        assert row["unique_key"] == item.unique_key
        assert row["status"] == "pending"
        assert row["title"] == item.title

    def test_duplicate_is_ignored(self, tracker):
        """INSERT OR IGNORE → 重复插入不报错、不产生新行"""
        item = make_item()
        tracker.upsert_pending(item)
        tracker.upsert_pending(item)  # 重复
        assert len(tracker.get_pending()) == 1  # 还是 1 条

    def test_upsert_pending_marks_changed_on_content_change(self, tracker):
        """内容变化时：更新记录 + 记录 previous_content_hash（支撑政策变更追踪）"""
        v1 = make_item(title="标题v1", content="内容v1")
        tracker.upsert_pending(v1)

        v2 = make_item(title="标题v2", content="内容v2")  # 同 unique_key，内容不同
        is_new, is_changed = tracker.upsert_pending(v2)

        assert is_changed is True  # 标记为内容变更
        pending = tracker.get_pending()
        assert len(pending) == 1
        assert pending[0]["title"] == "标题v2"  # 标题已刷新为新内容

        # 变更追踪：previous_content_hash 保存旧 hash，content_hash 更新为新 hash
        row = tracker._conn.execute(
            "SELECT previous_content_hash, content_hash FROM policy_record WHERE unique_key=?",
            (v1.unique_key,),
        ).fetchone()
        assert row["previous_content_hash"] == v1.content_hash
        assert row["content_hash"] == v2.content_hash

    def test_different_items_both_inserted(self, tracker):
        tracker.upsert_pending(make_item(item_id="a"))
        tracker.upsert_pending(make_item(item_id="b"))
        assert len(tracker.get_pending()) == 2

# ================================================================== 4. mark_ingested + mark_failed
class TestMarkStatus:
    def test_mark_ingested_changes_status_and_file_id(self, tracker):
        item = make_item()
        tracker.upsert_pending(item)
        tracker.mark_ingested(item.unique_key, file_id="qdrant_file_999")

        # pending 队列没了
        assert len(tracker.get_pending()) == 0
        # exists 现在能拦截了
        assert tracker.exists(make_item()) is True

        # 直接查状态
        row = tracker._conn.execute(
            "SELECT status, file_id FROM policy_record WHERE unique_key=?",
            (item.unique_key,),
        ).fetchone()
        assert row["status"] == "ingested"
        assert row["file_id"] == "qdrant_file_999"

    def test_mark_failed_does_not_block_exists(self, tracker):
        item = make_item()
        tracker.upsert_pending(item)
        tracker.mark_failed(item.unique_key)

        # exists 不拦截（status=failed）
        assert tracker.exists(make_item()) is False
        # 也不在 pending 了
        assert len(tracker.get_pending()) == 0

    def test_mark_ingested_default_file_id_empty(self, tracker):
        item = make_item()
        tracker.upsert_pending(item)
        tracker.mark_ingested(item.unique_key)  # 不传 file_id

        row = tracker._conn.execute(
            "SELECT file_id FROM policy_record WHERE unique_key=?",
            (item.unique_key,),
        ).fetchone()
        assert row["file_id"] == ""

# ================================================================== 5. get_pending
class TestGetPending:
    def test_get_pending_returns_only_pending(self, tracker):
        tracker.upsert_pending(item1 := make_item(item_id="a"))
        tracker.upsert_pending(item2 := make_item(item_id="b"))
        tracker.upsert_pending(item3 := make_item(item_id="c"))
        tracker.mark_ingested(item2.unique_key)
        tracker.mark_failed(item3.unique_key)

        pending = tracker.get_pending()
        keys = [r["unique_key"] for r in pending]
        assert keys == [item1.unique_key]  # 只剩 a

    def test_get_pending_limit(self, tracker):
        for i in range(10):
            tracker.upsert_pending(make_item(item_id=f"url_{i}"))
        pending = tracker.get_pending(limit=3)
        assert len(pending) == 3

    def test_get_pending_default_limit_500(self, tracker):
        for i in range(600):
            tracker.upsert_pending(make_item(item_id=f"url_{i}"))
        pending = tracker.get_pending()  # 默认 limit=500
        assert len(pending) == 500

# ================================================================== 6. 完整去重流水线（端到端）
class TestDeduplicationPipeline:
    def test_scan_ingest_rescan_flow(self, tracker):
        """模拟 run_once 的去重流程：scan → exists 判断 → upsert → ingested → 再次 scan 被拦截"""
        item = make_item(item_id="gov_doc_001", content="新发布的增值税政策")

        # 第一次扫描
        assert tracker.exists(item) is False
        tracker.upsert_pending(item)
        tracker.mark_ingested(item.unique_key, file_id="q_001")

        # 第二次扫描（同一条）
        assert tracker.exists(item) is True  # 被拦截
        assert len(tracker.get_pending()) == 0

    def test_content_dedup_catches_duplicate_from_different_source(self, tracker):
        """不同 source_id + 不同 item_id，但 content 相同 → 二级去重拦截"""
        item_a = make_item(source_id="site_a", item_id="page_1", content="完全一样的政策正文")
        tracker.upsert_pending(item_a)
        tracker.mark_ingested(item_a.unique_key)

        item_b = make_item(source_id="site_b", item_id="page_99", content="完全一样的政策正文")
        assert item_a.unique_key != item_b.unique_key
        assert tracker.exists(item_b) is True  # 二级拦截


# ================================================================== 7. TaxGovCnSource 正文抓取
class TestTaxGovCnSource:
    def test_fetch_policy_page_builds_valid_item(self):
        """_fetch_policy_page 应正确构造 PolicyItem。

        回归：旧实现给 PolicyItem 传了不存在的字段（publish_date/unique_key/
        content_hash/extra），一调用就 TypeError，且漏掉 content。
        """
        from unittest.mock import MagicMock
        from knowledge.policy_updater import TaxGovCnSource, PolicyItem

        src = TaxGovCnSource(listing_url="https://example.com/list.html", source_id="chinatax")
        fake_resp = MagicMock()
        fake_resp.headers = {"Content-Type": "text/html; charset=utf-8"}
        fake_resp.text = (
            "<html><body>"
            "<h3>财政部 税务总局公告</h3>"
            "<div class='info'>发布日期：2024年01月15日</div>"
            "<div id='fontzoom'>增值税小规模纳税人减免政策正文</div>"
            "</body></html>"
        )
        src._session = MagicMock()
        src._session.get.return_value = fake_resp

        item = src._fetch_policy_page("https://example.com/chinatax/n371/c100/content.html")

        assert isinstance(item, PolicyItem)
        assert item.title == "财政部 税务总局公告"
        assert item.content == "增值税小规模纳税人减免政策正文"
        assert item.url == "https://example.com/chinatax/n371/c100/content.html"
        assert item.content_hash  # 有正文，内容哈希非空

    def test_scan_crawls_listing_and_fetches_pages(self):
        """scan() 应能抓取列表页的链接并逐条解析正文。"""
        from unittest.mock import MagicMock
        from knowledge.policy_updater import TaxGovCnSource, PolicyItem

        src = TaxGovCnSource(
            listing_url="https://example.com/list.html", source_id="chinatax", max_pages=1
        )

        listing_html = (
            "<html><body>"
            '<a href="n371/a1/content.html">政策一</a>'
            '<a href="n371/a2/content.html">政策二</a>'
            "</body></html>"
        )

        def fake_get(url, timeout=None):
            resp = MagicMock()
            resp.headers = {"Content-Type": "text/html; charset=utf-8"}
            if url == "https://example.com/list.html":
                resp.text = listing_html
            else:
                resp.text = (
                    "<html><body><h3>某政策标题</h3>"
                    "<div class='info'>发布日期：2024年01月15日</div>"
                    "<div id='fontzoom'>某政策正文内容</div></body></html>"
                )
            return resp

        src._session = MagicMock()
        src._session.get.side_effect = fake_get

        items = src.scan()

        assert len(items) == 2
        assert all(isinstance(i, PolicyItem) for i in items)
        assert items[0].content == "某政策正文内容"