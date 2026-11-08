"""Read-only inventory of the four administrator-uploaded Research policies."""
import json
import logging
import os
from pathlib import Path
import re
import sqlite3
from contextlib import closing
from collections import Counter
from urllib.parse import urlsplit
from uuid import uuid4

from eval.research_phase1 import ROOT, fields, read_json

POLICIES = {
    "income_confirmation": "确认企业所得税收入",
    "digital_invoice": "数字化电子发票",
    "vat_regulation": "增值税法实施条例",
    "income_tax_regulation": "企业所得税法实施条例",
}
METADATA = ("title document_number issuer publish_date published_date effective_date "
            "is_current validity_check source_type scope status embedding_model")


def page_heading_only(content):
    return bool(re.fullmatch(r"\s*## 第 \d+ 页\s*", content))


def ro_database(path):
    return closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=2))


def local_inventory(data_dir, business_db):
    files = {}
    for name in ("kb/tax_policy_meta.json", "user_documents_meta.json"):
        path = data_dir / name
        if path.is_file():
            for record in read_json(path).get("files", []):
                files[record["file_id"]] = (record, name)
    store = data_dir / "document_store.json"
    chunks = read_json(store) if store.is_file() else []
    tracker_db = data_dir / "policy_tracker.sqlite3"
    for label, term in POLICIES.items():
        for file_id, (record, source) in files.items():
            metadata = record.get("tax_metadata", {})
            if metadata.get("source_type") != "default" or term not in record.get("file_name", ""):
                continue
            local = [c for c in chunks if c["file_id"] == file_id]
            source_exists = bool(record.get("file_path") and Path(record["file_path"]).is_file())
            row = {"policy": label, "file_id": file_id, "metadata_record": source,
                   "original_exists": source_exists,
                   **fields(record, "status chunk_count vector_write_started"),
                   "metadata": fields(metadata, METADATA),
                   "source_url_present": bool(metadata.get("source_url") or metadata.get("url")),
                   "local_chunks": len(local),
                   "nonempty_local_chunks": sum(bool(c.get("content", "").strip()) for c in local),
                   "page_heading_only_chunks": sum(page_heading_only(c.get("content", "")) for c in local),
                   "local_text_chars": sum(len(c.get("content", "")) for c in local),
                   "sql_files": None, "sql_chunks": None, "tracker_rows": None}
            if business_db.is_file():
                with ro_database(business_db) as db:
                    rows = db.execute("SELECT id FROM knowledge_files WHERE business_id=?", (file_id,)).fetchall()
                    row["sql_files"] = len(rows)
                    row["sql_chunks"] = sum(db.execute(
                        "SELECT count(*) FROM knowledge_chunks WHERE file_id=?", entry).fetchone()[0] for entry in rows)
            if tracker_db.is_file():
                with ro_database(tracker_db) as db:
                    row["tracker_rows"] = db.execute(
                        "SELECT count(*) FROM policy_record WHERE file_id=?", (file_id,)).fetchone()[0]
            yield row


def audit():
    from dotenv import load_dotenv
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    load_dotenv(ROOT / ".env")
    from knowledge.embedding import QdrantService
    from services.config_manager import ConfigManager
    from services.paths import DATA_DIR
    from services.research_evidence import normalize_evidence, restore_policy_metadata
    from services.security import read_only_retrieval, tenant_scope
    from services.tools.kb_tools import search_tax_policy
    db_url = os.environ.get("DATABASE_URL", "sqlite:///tax_agent.db")
    if urlsplit(db_url).scheme != "sqlite" or not db_url.startswith("sqlite:///"):
        raise ValueError("inventory requires a SQLite business database")
    rows = list(local_inventory(DATA_DIR, Path(db_url.removeprefix("sqlite:///"))))
    report = {"files": rows, "not_found": sorted(POLICIES.keys() - {r["policy"] for r in rows})}
    client = None
    guard = read_only_retrieval.set(True)
    try:
        qdrant = QdrantService(read_only=True)
        client = qdrant.client
        collection = qdrant.COLLECTION_NAME
        info = client.get_collection(collection)
        report["qdrant"] = {"status": "success", "collection": collection, "point_count": info.points_count,
                           "vector_size": getattr(info.config.params.vectors, "size", None)}
        for row in rows:
            try:
                selector = Filter(must=[FieldCondition(key="file_id", match=MatchValue(value=row["file_id"]))])
                offset = None
                dimensions, visibility = Counter(), Counter()
                nonempty = heading_only = 0
                while True:
                    points, offset = client.scroll(collection, scroll_filter=selector, limit=64, offset=offset,
                        with_payload=["status", "source_type", "scope", "content"],
                        with_vectors=True)
                    for point in points:
                        dimensions[str(len(point.vector)) if isinstance(point.vector, list) else "other_or_missing"] += 1
                        payload = point.payload or {}
                        visibility[(payload.get("source_type"), payload.get("scope"), payload.get("status"))] += 1
                        content = payload.get("content") or ""
                        nonempty += bool(content.strip())
                        heading_only += page_heading_only(content)
                    if offset is None:
                        break
                row["cloud"] = {"points": sum(dimensions.values()), "vector_dimensions": dict(dimensions),
                                "nonempty_chunks": nonempty, "page_heading_only_chunks": heading_only,
                                "visibility": [{"source_type": a, "scope": b, "status": c, "count": n}
                                               for (a, b, c), n in visibility.items()]}
            except Exception as exc:
                row["cloud"] = {"error_type": type(exc).__name__}
    except Exception as exc:
        report["qdrant"] = {"status": "error", "error_type": type(exc).__name__,
                            "cause_type": type(getattr(exc, "source", exc.__cause__)).__name__,
                            "http_status": getattr(exc, "status_code", None)}
    finally:
        if client is not None:
            client.close()
        read_only_retrieval.reset(guard)

    with tenant_scope("research-policy-audit-" + uuid4().hex):
        for row in rows:
            try:
                hits = search_tax_policy(POLICIES[row["policy"]], limit=4, research=True)
                restore_policy_metadata(hits)
                matching = [normalize_evidence(hit) for hit in hits
                            if hit.get("file_id") == row["file_id"] and hit.get("source_type") == "default"]
                row["research_probe"] = {"status": "hit" if matching else "no_match", "top_k": 4,
                    "matching_chunks": len(matching), "returned_chunks": len(hits),
                    "target_source_checks": dict(Counter(hit["source_check"] for hit in matching))}
            except Exception as exc:
                row["research_probe"] = {"status": "error", "error_type": type(exc).__name__,
                                         "cause_type": type(exc.__cause__ or getattr(exc, "source", None)).__name__}
    report["embedding_model"] = ConfigManager().get_embedding_config()["model"]
    return report


if __name__ == "__main__":
    logging.disable(logging.CRITICAL)
    try:
        print(json.dumps(audit(), ensure_ascii=False, indent=2))
    except Exception as error:
        print(json.dumps({"status": "error", "error_type": type(error).__name__}))
        raise SystemExit(1) from None
