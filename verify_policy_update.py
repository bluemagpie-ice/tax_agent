"""独立政策更新验收：真实 Embedding/Qdrant，独立集合与本地记录，无需启动 FastAPI。"""
import argparse
import json
import logging
import sys
import time
import uuid
from pathlib import Path

from knowledge.policy_updater import (
    LocalDirectorySource, PolicyTracker, PolicyUpdater, TaxGovCnSource, logging_notify_hook,
)

DEMO_MARKER = "<!-- policy-update-demo -->\n# 模拟政策，仅用于软件验收，不作为税务依据\n"


def verify_connectivity(listing_url):
    """独立验证列表、选择器及一篇正文；失败直接给出 HTTP/selector 错误。"""
    from bs4 import BeautifulSoup
    source = TaxGovCnSource(listing_url=listing_url, max_pages=1)
    try:
        response = source._session.get(listing_url, timeout=source.request_timeout)
        response.raise_for_status()
        response.encoding = source._detect_encoding(response)
        soup = BeautifulSoup(response.text, "lxml")
        links = source._select_links(soup)
        url = source._resolve_url(links[0]["href"])
        item = source._fetch_policy_page(url)
        assert item and item.title.strip() and item.content.strip() and item.url.strip()
        evidence = {"list_status": response.status_code, "link_selector": source.link_selector,
                    "link_count": len(links), "title_selector": source.title_selector,
                    "content_selector": source.content_selector, "title": item.title,
                    "content_length": len(item.content), "url": item.url,
                    "published_date": item.published_date}
        print(json.dumps(evidence, ensure_ascii=False, indent=2))
        return evidence
    finally:
        source._session.close()


class ReportLogCapture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.reports = []

    def emit(self, record):
        if record.getMessage().startswith("[政策更新报告]"):
            self.reports.append(record.getMessage())


def verify_ingestion(updater, collection):
    from knowledge.embedding import QdrantService
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    service = QdrantService()
    evidence = []
    try:
        with updater.tracker._lock:
            rows = updater.tracker._conn.execute("SELECT * FROM policy_record WHERE status='ingested'").fetchall()
        for row in rows:
            query = Filter(must=[FieldCondition(key="file_id", match=MatchValue(value=row["file_id"]))])
            count = 0
            for _ in range(5):
                count = service.client.count(collection_name=collection, count_filter=query, exact=True).count
                if count:
                    break
                time.sleep(0.2)
            assert count > 0, f"Cloud 未找到向量：{row['file_id']}"
            assert row["content_snapshot"], "正文快照未落库"
            evidence.append({"file_id": row["file_id"], "cloud_vectors": count,
                             "snapshot_length": len(row["content_snapshot"])})
        assert rows, "没有成功入库记录"
        return evidence
    finally:
        service.client.close()


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["all", "flow", "change", "window", "connectivity"], default="all")
    parser.add_argument("--policies-dir", default="data/policies")
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--listing-url", default="https://fgk.chinatax.gov.cn/zcfgk/c100027/list.html")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    work = Path(args.work_dir or f"data/policy_verification/{uuid.uuid4().hex[:12]}").resolve()
    if any(work.glob("*.sqlite3")):
        parser.error("验收目录已有数据库，请使用新目录或省略 --work-dir；保留已有验收证据。")
    work.mkdir(parents=True, exist_ok=True)
    evidence = {}
    capture = ReportLogCapture()
    logging.getLogger("knowledge.policy_updater").addHandler(capture)

    if args.task != "connectivity":
        from knowledge.embedding import QdrantService, PersistentDocumentStore
        collection = "tax_policy_verify_" + uuid.uuid4().hex[:12]
        QdrantService.COLLECTION_NAME = collection  # 仅此独立进程；不触碰应用的 tax_chunks 集合。
        PersistentDocumentStore(file_path=work / "document_store.json")
        evidence["collection"] = collection
        print(f"验收目录：{work}\n隔离 Cloud 集合：{collection}")

    for task in ("flow", "change", "window"):
        if args.task not in (task, "all"):
            continue
        directory = Path(args.policies_dir).resolve() if task == "flow" else work / task
        directory.mkdir(parents=True, exist_ok=True)
        if task == "flow" and not any(directory.iterdir()):
            (directory / "demo_vat_2024-01-01.md").write_text(DEMO_MARKER + "增值税税率 13%", encoding="utf-8")
            (directory / "demo_income_2020-01-01.md").write_text(DEMO_MARKER + "企业所得税测试条文：25%", encoding="utf-8")
        if task == "change":
            path = directory / "模拟增值税_2024-01-01.md"
            path.write_text(DEMO_MARKER + "增值税税率 13%", encoding="utf-8")
        if task == "window":
            (directory / "政策_2020-01-01.md").write_text(DEMO_MARKER + "2020 年测试旧政策", encoding="utf-8")
            (directory / "政策_2024-01-01.md").write_text(DEMO_MARKER + "2024 年测试新政策", encoding="utf-8")
        db = work / f"{task}.sqlite3"
        with PolicyUpdater(tracker_db=str(db), kb_work_dir=str(work / task / "kb"), kb_id=f"verify_{task}",
                           since_date="2022-01-01" if task == "window" else None,
                           notify_hooks=[logging_notify_hook]) as updater:
            source = LocalDirectorySource(str(directory), source_id=f"verify_{task}")
            updater.add_source(source)
            items = source.scan()
            assert items and not source.last_errors, source.last_errors
            old_item = items[0]
            log_count = len(capture.reports)
            first = updater.run_once()
            print(f"[{task}] 第一次完整 UpdateReport\n" + json.dumps(first.to_dict(), ensure_ascii=False, indent=2))
            assert first.failed == 0 and not first.errors and first.ingested > 0, first.to_dict()
            rows = verify_ingestion(updater, collection)
            local_chunks = json.loads((work / "document_store.json").read_text(encoding="utf-8"))
            metadata = json.loads((work / task / "kb" / f"verify_{task}_meta.json").read_text(encoding="utf-8"))
            for row in rows:
                assert any(chunk["file_id"] == row["file_id"] and chunk["content"] for chunk in local_chunks)
                assert any(record["file_id"] == row["file_id"] and record["status"] == "active" for record in metadata["files"])
            if task == "change":
                path.write_text(DEMO_MARKER + "增值税税率 13%，小规模纳税人减免至 1%", encoding="utf-8")
                second = updater.run_once()
                assert second.changed == 1 and second.ingested == 1 and second.duplicated == 0, second.to_dict()
                diff = updater.tracker.get_change_diff(old_item.unique_key)
                assert diff["old_hash"] == old_item.content_hash and diff["new_hash"] != diff["old_hash"]
                assert diff["old_snapshot_available"] and "-增值税税率 13%" in diff["diff"]
                assert "+增值税税率 13%，小规模纳税人减免至 1%" in diff["diff"]
                evidence["diff"] = diff
                rows = verify_ingestion(updater, collection)
                print("is_changed=True, previous_content_hash=" + diff["old_hash"] + "\n" + diff["diff"])
            else:
                second = updater.run_once()
                assert second.ingested == 0 and second.duplicated == first.ingested, second.to_dict()
            print(f"[{task}] 第二次完整 UpdateReport\n" + json.dumps(second.to_dict(), ensure_ascii=False, indent=2))
            if task == "window":
                assert first.scanned == 2 and first.ingested == 1 and first.filtered == 1
                with updater.tracker._lock:
                    titles = [r[0] for r in updater.tracker._conn.execute("SELECT title FROM policy_record")]
                assert titles == ["政策_2024-01-01"], titles
            reports = updater.get_report_history()
            assert len(reports) == 2 and reports[0]["changed"] == second.changed
            assert len(capture.reports) - log_count == 2, "logging_notify_hook 未触发两次"
            evidence[task] = {"first": first.to_dict(), "second": second.to_dict(), "ingestion": rows,
                              "persisted_reports": len(reports), "logging_hook_calls": 2}
        if task == "change":
            with_db = PolicyTracker(str(db))
            try:
                assert with_db.get_change_diff(old_item.unique_key) == evidence["diff"], "重开数据库 diff 不一致"
            finally:
                with_db.close()

    if args.task in ("all", "connectivity"):
        evidence["connectivity"] = verify_connectivity(args.listing_url)
    assert "web.main" not in sys.modules, "独立脚本意外加载 FastAPI 应用"
    (work / "evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"验收完成，完整证据：{work / 'evidence.json'}；验收文件与隔离集合保留，未自动删除。")


if __name__ == "__main__":
    main()
