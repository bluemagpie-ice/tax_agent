import asyncio
from datetime import date

from knowledge.chunking import split_text
from services.policy_metadata import extract_policy_metadata
from services.tools.date_extractor import extract_query_date
from tests.test_runtime_document_flow import document_app


def test_metadata_never_promotes_publication_to_validity():
    metadata = extract_policy_metadata("财政部税务总局公告2026年第10号\n发布日期2026年1月1日\n增值税规则")
    assert metadata["is_current"] is None
    assert metadata["validity_check"] == "unknown"
    assert "effective_date" not in metadata
    dated = extract_policy_metadata("自2023年1月1日至2027年12月31日执行增值税优惠")
    assert dated["effective_date"] == "2023-01-01"
    assert dated["expiry_date"] == "2027-12-31"


def test_law_chunks_keep_preamble_articles_and_stable_ids():
    text = "前置说明政策背景。\n第一条 " + "长条文内容。" * 100 + "\n第二条 其他规定。"
    chunks = split_text(text, "laws", file_id="audit-law", chunk_size=120, chunk_overlap=10)
    assert chunks == split_text(text, "laws", file_id="audit-law", chunk_size=120, chunk_overlap=10)
    assert chunks[0]["content"].startswith("前置说明")
    parts = [chunk for chunk in chunks if "长条文" in chunk["content"]]
    assert len(parts) > 1 and all(chunk["content"].startswith("第一条") for chunk in parts)


def test_precise_dates_and_document_years():
    assert extract_query_date("2023年3月15日", date(2026, 11, 7)) == "2023-03-15"
    assert extract_query_date("今年", date(2026, 11, 7)) == "2026-11-07"
    assert extract_query_date("财政部公告2026年第10号") is None


def test_same_file_retry_is_idempotent_in_local_qdrant(document_app, tmp_path):
    _, _, service, embedding, _ = document_app
    path = tmp_path / "policy.txt"
    path.write_text("增值税模拟条文，不是真实政策。", encoding="utf-8")
    async def verify():
        for _ in range(2):
            result = await service._kb.ingest_file(str(path), file_id="retry-policy", source_type="default", scope="system")
            assert result["status"] == "success"
        rows = [row for row in await service._kb.list_files() if row["file_id"] == "retry-policy"]
        assert len(rows) == 1 and rows[0]["status"] == "active"
        assert embedding.qdrant.client.count(embedding.qdrant.COLLECTION_NAME).count == result["chunk_count"]
        assert len(embedding.store.get_chunks()) == result["chunk_count"]
    asyncio.run(verify())


def test_missing_neo4j_driver_leaves_qdrant_and_bm25_independent(document_app, monkeypatch):
    import uuid

    import knowledge.embedding as module

    _, _, _, existing, _ = document_app
    monkeypatch.setattr(module, "_NEO4J_AVAILABLE", False)
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setattr(module, "QdrantService", lambda: existing.qdrant)
    monkeypatch.setattr(module.EmbeddingService, "_embed_texts", lambda self, texts: existing._embed_texts(texts))
    store = existing.store
    for index, text in enumerate(["增值税抵扣政策", "住房租赁登记", "企业所得税扣除"], start=1):
        store.add_chunks([module.ChunkDocument(str(uuid.uuid4()), text, str(index), {"source_type": "default"})])
    service = module.EmbeddingService(store, use_neo4j=True)
    service.embedding_model = "fixture"
    assert not service.use_neo4j
    assert service.search("增值税", mode="semantic")
    assert service.search("增值税", mode="bm25")[0]["file_id"] == "1"
