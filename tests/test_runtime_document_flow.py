"""文档真实 CRUD/过滤使用 Qdrant 本地引擎；模型使用可控流，绝不操作真实云端。"""
from tests.auth_helpers import USER_BEARER, ADMIN_BEARER
import asyncio
import importlib
import json
from threading import Event
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, AIMessageChunk
from qdrant_client import QdrantClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from knowledge.embedding import EmbeddingService, PersistentDocumentStore, QdrantService
from services.security import tenant_scope, document_session, blocked_document_ids


@pytest.fixture
def document_app(tmp_path, monkeypatch):
    main = importlib.import_module("web.main")
    import database.repositories as repos
    from database.models import Base
    engine = create_engine(f"sqlite:///{tmp_path / 'business.db'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    monkeypatch.setattr(repos, "SessionLocal", sessionmaker(bind=engine))
    monkeypatch.setitem(main.PATHS, "data", str(tmp_path / "data"))
    monkeypatch.setattr("services.paths.DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(main, "policy_updater", None)
    monkeypatch.setitem(main.PATHS, "uploads", str(tmp_path / "uploads"))
    (tmp_path / "uploads").mkdir()
    from agents.memory import ConversationMemory
    memory = ConversationMemory(base_dir=tmp_path / "memory")
    service = main.KnowledgeService()
    cloud = QdrantService.__new__(QdrantService)
    cloud.client = QdrantClient(":memory:")
    from qdrant_client.models import VectorParams, Distance
    cloud.client.create_collection(cloud.COLLECTION_NAME, vectors_config=VectorParams(size=4, distance=Distance.COSINE))
    embedding = EmbeddingService.__new__(EmbeddingService)
    monkeypatch.setattr(PersistentDocumentStore, "_file_path", tmp_path / "document_store.json")
    monkeypatch.setattr(PersistentDocumentStore, "_shared_chunks", [])
    monkeypatch.setattr(PersistentDocumentStore, "_loaded", False)
    embedding.store = PersistentDocumentStore()
    embedding.qdrant = cloud
    embedding.use_qdrant = True
    embedding.use_bm25 = embedding.use_neo4j = False
    embedding._qdrant_payload_cache = {}
    embedding.embedding_model = "test-vector"
    embedding._embed_texts = lambda texts: np.tile([1., 0., 0., 0.], (len(texts), 1))
    service._kb._embedding_service = embedding
    monkeypatch.setattr(main, "knowledge_service", service)
    monkeypatch.setattr(main, "conversation_memory", memory)
    monkeypatch.setattr(main, "tax_agent", SimpleNamespace(delete_thread=AsyncMock(return_value=True)))
    monkeypatch.setenv("AUTH_USER_TOKEN", "test_token_123")
    monkeypatch.setenv("DEVELOPER_TOKEN", "admin_token_456")
    blocked_document_ids.clear()
    yield main, TestClient(main.app), service, embedding, memory
    cloud.client.close()
    engine.dispose()
    blocked_document_ids.clear()


USER = {"Authorization": USER_BEARER}
ADMIN = {"Authorization": ADMIN_BEARER}


def upload(client, scope="global", session="", headers=USER, text="A document about input tax."):
    response = client.post("/api/knowledge/upload", headers=headers,
                           data={"scope": scope, "session_id": session}, files={"file": ("demo.md", text.encode(), "text/markdown")})
    assert response.status_code == 200, response.text
    return response.json()["file_id"]


def test_upload_browse_scopes_restart_and_delete(document_app):
    main, client, service, embedding, memory = document_app
    client.post("/api/conversations", headers=USER, json={"session_id": "one"})
    global_id = upload(client)
    from database.repositories import KnowledgeFileRepository, KnowledgeChunkRepository
    sql_record = KnowledgeFileRepository().get_by_business_id(global_id)
    assert sql_record is not None
    assert KnowledgeChunkRepository().count_by_file_id(sql_record.id) > 0
    session_id = upload(client, "session", "one", text="A private session-only document.")
    system_id = upload(client, "system", headers=ADMIN, text="Administrator source.")
    assert {r["file_id"] for r in client.get("/api/knowledge/files", headers=USER).json()["files"]} == {global_id, session_id}
    assert client.get(f"/api/knowledge/files/{system_id}/content", headers=USER).status_code == 404
    assert client.delete(f"/api/knowledge/files/{system_id}", headers=USER).status_code == 404
    assert client.get("/api/knowledge/files?include_system=true", headers=USER).status_code == 403
    assert "private session-only" in client.get(f"/api/knowledge/files/{session_id}/content", headers=USER).json()["content"]
    assert client.get("/api/search?q=tax").status_code == 401
    with tenant_scope("user_001"):
        token = document_session.set("one")
        try:
            assert {r["file_id"] for r in embedding.search("tax", limit=20)} == {global_id, session_id, system_id}
            document_session.set("two")
            assert {r["file_id"] for r in embedding.search("tax", limit=20)} == {global_id, system_id}
        finally:
            document_session.reset(token)
    with tenant_scope("another-user"):
        assert {r["file_id"] for r in embedding.search("tax", limit=20)} == {system_id}
    from knowledge.kb import QdrantKnowledgeBase
    reopened = QdrantKnowledgeBase(main.PATHS["data"])
    assert len(asyncio.run(reopened.list_files())) == 3
    assert client.delete("/api/conversations/one", headers=USER).status_code == 200
    assert {r["file_id"] for r in client.get("/api/knowledge/files", headers=USER).json()["files"]} == {global_id}
    assert not [p for p in embedding.qdrant.client.scroll(embedding.qdrant.COLLECTION_NAME)[0] if p.payload["file_id"] == session_id]
    path = next(r["file_path"] for r in service._kb._metadata["files"] if r["file_id"] == global_id)
    assert client.delete(f"/api/knowledge/files/{global_id}", headers=USER).status_code == 200
    from pathlib import Path
    assert not Path(path).exists()
    from database.repositories import KnowledgeFileRepository
    assert KnowledgeFileRepository().get_by_business_id(global_id) is None
    assert all(f["file_id"] != global_id for f in asyncio.run(QdrantKnowledgeBase(main.PATHS["data"]).list_files()))
    PersistentDocumentStore._loaded = False
    PersistentDocumentStore._shared_chunks = []
    assert all(chunk.file_id != global_id for chunk in PersistentDocumentStore().get_chunks())


@pytest.mark.parametrize("connection_error", [True, False])
def test_cloud_failure_retains_mapping_and_retry_finishes(document_app, connection_error):
    _, client, service, embedding, _ = document_app
    fid = upload(client)
    from pathlib import Path
    path = Path(service._kb._metadata["files"][0]["file_path"])
    failure = ConnectionError("simulated cloud failure") if connection_error else None
    with patch.object(embedding.qdrant.client, "delete", side_effect=failure) as delete:
        response = client.delete(f"/api/knowledge/files/{fid}", headers=USER)
    assert delete.call_args.kwargs["wait"] is True
    assert response.status_code == 500
    assert path.exists()
    assert service._kb._metadata["files"][0]["status"] == "delete_failed"
    assert embedding.qdrant.client.count(embedding.qdrant.COLLECTION_NAME).count > 0
    with tenant_scope("user_001"):
        assert embedding.search("tax") == []
    assert client.delete(f"/api/knowledge/files/{fid}", headers=USER).status_code == 200
    assert not path.exists()
    assert embedding.qdrant.client.count(embedding.qdrant.COLLECTION_NAME).count == 0


def test_legacy_files_hidden_but_admin_can_manage(document_app):
    _, client, service, _, _ = document_app
    service._kb._metadata["files"].append({"file_id": "legacy", "file_name": "old.md", "tax_metadata": {"source_type": "user_upload"}})
    service._kb.save_metadata()
    assert client.get("/api/knowledge/files", headers=USER).json()["files"] == []
    assert client.get("/api/knowledge/files/legacy/content", headers=USER).status_code == 404
    rows = client.get("/api/knowledge/files?include_system=true", headers=ADMIN).json()["files"]
    assert rows[0]["scope"] == "legacy"


def test_vector_upload_failure_is_not_reported_as_success(document_app):
    _, client, service, embedding, _ = document_app
    with patch.object(embedding.qdrant, "upsert_chunks", side_effect=ConnectionError("simulated upload failure")):
        response = client.post("/api/knowledge/upload", headers=USER, data={"scope": "global"}, files={"file": ("demo.md", b"tax")})
    assert response.status_code == 502
    assert service._kb._metadata["files"][0]["status"] == "upload_failed"


def test_delete_rejects_conflicting_cloud_owner_without_removing_either_file(document_app):
    from pathlib import Path
    from qdrant_client.models import PointStruct

    _, client, service, embedding, _ = document_app
    fid = upload(client)
    cloud = embedding.qdrant
    original = cloud.client.scroll(cloud.COLLECTION_NAME)[0][0]
    cloud.client.upsert(cloud.COLLECTION_NAME, [
        PointStruct(id=77, vector=[1., 0., 0., 0.], payload={**original.payload, "owner_id": "other-user"})
    ])
    record = service._kb._metadata["files"][0]
    assert client.delete(f"/api/knowledge/files/{fid}", headers=USER).status_code == 500
    assert Path(record["file_path"]).exists() and record["status"] == "active"
    assert cloud.client.count(cloud.COLLECTION_NAME).count == 2
    assert any(c.file_id == fid for c in embedding.store.get_chunks())


def test_legacy_web_batch_delete_stays_within_tenant(document_app, monkeypatch):
    from uuid import uuid4
    from knowledge.embedding import ChunkDocument

    _, client, _, embedding, _ = document_app
    documents = [
        ChunkDocument(str(uuid4()), "legacy web result", fid,
                      {"source_type": "web_search", "scope": "global", "owner_id": owner})
        for fid, owner in [("my-web", "user_001"), ("other-web", "user_000")]
    ]
    embedding.add_chunks(documents)
    embedding.store.add_chunks(documents)
    monkeypatch.setattr("knowledge.embedding.EmbeddingService", lambda **kwargs: embedding)
    assert client.delete("/api/knowledge/web-search?file_id=other-web", headers=USER).status_code == 404
    response = client.delete("/api/knowledge/web-search", headers=USER)
    assert response.status_code == 200 and response.json()["deleted_count"] == 1
    assert {chunk.file_id for chunk in embedding.store.get_chunks()} == {"other-web"}
    assert embedding.qdrant.client.count(embedding.qdrant.COLLECTION_NAME).count == 1


def test_abandoned_upload_before_cloud_write_can_be_deleted_without_qdrant(document_app):
    from pathlib import Path

    _, client, service, embedding, _ = document_app
    embedding.qdrant = None
    response = client.post("/api/knowledge/upload", headers=USER, data={"scope": "global"},
        files={"file": ("unfinished.md", b"unfinished document")})
    assert response.status_code == 502
    record = service._kb._metadata["files"][0]
    record["status"] = "uploading"
    service._kb.save_metadata()
    url = f"/api/knowledge/files/{record['file_id']}"
    assert client.delete(url, headers=USER).status_code == 200
    assert not Path(record["file_path"]).exists()
    assert client.delete(url, headers=USER).status_code == 404


@pytest.mark.parametrize("status", ["active", "delete_failed", "upload_failed"])
def test_delete_relocated_session_upload(document_app, tmp_path, status):
    from pathlib import Path
    from shutil import copyfile
    from database.models import KnowledgeFiles
    from database.repositories import KnowledgeChunkRepository, SessionLocal

    _, client, service, embedding, _ = document_app
    assert client.post("/api/conversations", headers=USER, json={"session_id": "moved"}).status_code == 200
    if status == "upload_failed":
        with patch.object(embedding, "_embed_texts", side_effect=RuntimeError("test failed upload")):
            response = client.post("/api/knowledge/upload", headers=USER,
                                   data={"scope": "session", "session_id": "moved"},
                                   files={"file": ("demo.md", b"test file")})
        assert response.status_code == 502
    else:
        upload(client, "session", "moved")
    record = service._kb._metadata["files"][0]
    fid = record["file_id"]
    current = Path(record["file_path"])
    historical = tmp_path / "old_checkout" / "uploads" / current.name
    historical.parent.mkdir(parents=True)
    copyfile(current, historical)
    record.update(file_path=str(historical), status=status)
    service._kb.save_metadata()
    with SessionLocal() as session:
        sql = session.query(KnowledgeFiles).filter_by(business_id=fid).first()
        sql_id = sql.id if sql else None
        if sql:
            sql.path = str(historical)
            if status == "delete_failed":
                sql.business_id = None
            session.commit()
    if status == "delete_failed":
        embedding.qdrant.delete_chunks_by_file_id(fid, ownership=record["tax_metadata"])

    neighbor = upload(client, "session", "moved", text="Another upload with the same display name.")
    url = f"/api/knowledge/files/{fid}"
    assert client.delete(url, headers=ADMIN).status_code == 404
    assert client.delete(url, headers=USER).status_code == 200
    assert not current.exists() and historical.exists()
    assert {r["file_id"] for r in client.get("/api/knowledge/files", headers=USER).json()["files"]} == {neighbor}
    points, _ = embedding.qdrant.client.scroll(embedding.qdrant.COLLECTION_NAME)
    assert {p.payload["file_id"] for p in points} == {neighbor}
    assert {chunk.file_id for chunk in embedding.store.get_chunks()} == {neighbor}
    if sql_id is not None:
        with SessionLocal() as session:
            assert session.get(KnowledgeFiles, sql_id) is None
        assert KnowledgeChunkRepository().count_by_file_id(sql_id) == 0


def test_delete_rejects_unrecognized_path_before_changing_index(document_app, tmp_path):
    from pathlib import Path

    _, client, service, embedding, _ = document_app
    fid = upload(client)
    record = service._kb._metadata["files"][0]
    current = Path(record["file_path"])
    outside = tmp_path / "outside" / "uploads" / "demo.md"
    outside.parent.mkdir(parents=True)
    outside.write_text("unrelated", encoding="utf-8")
    record["file_path"] = str(outside)
    service._kb.save_metadata()
    assert client.delete(f"/api/knowledge/files/{fid}", headers=USER).status_code == 500
    assert current.exists() and outside.exists() and record["status"] == "active"
    assert embedding.qdrant.client.count(embedding.qdrant.COLLECTION_NAME).count > 0


def test_preset_deletion_keeps_source_and_disables_reimport(document_app, tmp_path):
    main, client, service, _, _ = document_app
    preset = tmp_path / "preset.md"
    preset.write_text("preset source", encoding="utf-8")
    with tenant_scope("user_000"):
        result = asyncio.run(service.ingest_preset_file(str(preset)))
    assert client.delete(f"/api/knowledge/files/{result['file_id']}", headers=ADMIN).status_code == 200
    assert preset.exists()
    assert asyncio.run(service.ingest_preset_file(str(preset)))["skipped"] is True


def test_policy_delete_and_confirmed_reset_keep_personal_and_unrelated_data(document_app, tmp_path, monkeypatch):
    from database.repositories import KnowledgeFileRepository
    from knowledge.policy_updater import LocalDirectorySource, PolicyTracker, PolicyUpdater
    from qdrant_client.models import PointStruct

    main, client, service, embedding, _ = document_app
    data = tmp_path / "data"
    updater = PolicyUpdater(str(tmp_path / "policy.sqlite"), kb_work_dir=str(data / "kb"), kb_id="tax_policy")
    monkeypatch.setattr(main, "policy_updater", updater)
    sources = data / "policies"
    sources.mkdir()
    source_path = sources / "system.md"
    source_path.write_text("System input tax evidence", encoding="utf-8")
    item = LocalDirectorySource(str(sources)).scan()[0]
    file_id = updater.tracker.ingest_file_id(item)
    updater.tracker.upsert_pending(item)
    updater.tracker.mark_ingested(item.unique_key, file_id)
    kb = service._policy_kb()
    kb._embedding_service = embedding
    assert asyncio.run(kb.ingest_file(str(source_path), file_id=file_id, source_type="default", scope="system"))["status"] == "success"
    repo = KnowledgeFileRepository()
    repo.create(str(source_path), business_id=file_id)
    personal = upload(client)
    uploaded_policy = upload(client, scope="system", headers=ADMIN)
    assert file_id in {f["file_id"] for f in client.get("/api/knowledge/files?include_system=true", headers=ADMIN).json()["files"]}
    assert client.delete(f"/api/knowledge/files/{file_id}", headers=USER).status_code == 404
    assert client.delete(f"/api/knowledge/files/{personal}", headers=ADMIN).status_code == 404
    assert client.delete(f"/api/knowledge/files/{file_id}", headers=ADMIN).status_code == 200
    assert not source_path.exists()
    assert file_id not in {f["file_id"] for f in asyncio.run(service._policy_kb().list_files())}
    assert all(r.business_id != file_id for r in repo.get_all())
    assert updater.tracker.upsert_pending(item) == (False, False)

    cloud = embedding.qdrant
    def point(index, fid, source_type):
        return PointStruct(id=index, vector=[1., 0., 0., 0.], payload={"file_id":fid, "source_type":source_type, "content":"evidence"})
    cloud.client.upsert(cloud.COLLECTION_NAME, [point(1, "cloud-policy", "default"), point(2, "unrelated", "other")])
    path = "/api/policy/library/reset"
    assert client.post(path + "-preview", headers=USER).status_code == 403
    preview = client.post(path + "-preview", headers=ADMIN).json()
    assert {f["file_id"] for f in preview["files"]} == {"cloud-policy", uploaded_policy}
    assert client.post(path, headers=ADMIN, json={"preview_token":preview["preview_token"]}).status_code == 422
    body = {"preview_token":preview["preview_token"], "confirmation":"重置系统政策库"}
    cloud.client.upsert(cloud.COLLECTION_NAME, [point(3, "new-policy", "default")])
    assert client.post(path, headers=ADMIN, json=body).status_code == 409
    preview = client.post(path + "-preview", headers=ADMIN).json()
    body["preview_token"] = preview["preview_token"]
    result = client.post(path, headers=ADMIN, json=body).json()
    assert result["success"] and set(result["deleted"]) == {"cloud-policy", "new-policy", uploaded_policy}, result
    assert client.post(path, headers=ADMIN, json=body).status_code == 409
    points, _ = cloud.client.scroll(cloud.COLLECTION_NAME)
    assert {p.payload["file_id"] for p in points} == {personal, "unrelated"}
    restored = PolicyTracker(str(tmp_path / "policy.sqlite"))
    try:
        assert restored.is_paused()
        assert restored.upsert_pending(item) == (False, False)
    finally:
        restored.close()
        updater.tracker.close()


def test_real_orchestrator_emits_tokens_before_finishing(tmp_path, monkeypatch):
    from agents.base_agent import BaseAgent
    from agents.tax_agent import TaxAgent
    from agents.intent_router import HybridIntentRouter
    from agents.streaming import invoke_answer
    agent = TaxAgent.__new__(TaxAgent)
    BaseAgent.__init__(agent, str(tmp_path))
    agent._ensure_checkpointer = AsyncMock()
    agent.get_memory_context = MagicMock(return_value={})
    agent.record_turn = MagicMock()
    agent.router = HybridIntentRouter()
    finished = []
    first_token_delivered = Event()
    class Model:
        def stream(self, messages):
            for text in ("first ", "second ", "last"):
                yield AIMessageChunk(content=text)
    class Child:
        async def run(self, *args, **kwargs):
            response = await asyncio.to_thread(invoke_answer, Model(), [])
            # Guardrails publish only after the whole model response has been checked.
            assert await asyncio.to_thread(first_token_delivered.wait, 5)
            finished.append(True)
            return response.content
    agent.registry = SimpleNamespace(get=lambda name: Child())
    main = importlib.import_module("web.main")
    monkeypatch.setattr(main, "tax_agent", agent)
    monkeypatch.setattr(main, "conversation_memory", None)
    async def collect():
        response = await main.tax_chat_stream(main.TaxQueryRequest(user_id="u", query="hello", session_id="stream"), main.CurrentUser(id="u", user_name="u"))
        parts, before_finish = [], False
        async for block in response.body_iterator:
            if block.startswith("event: token"):
                before_finish |= not finished
                parts.append(json.loads(block.split("data: ")[1])["text"])
                first_token_delivered.set()
        assert before_finish
        assert parts == ["first ", "second ", "last"]
    asyncio.run(collect())


def test_independent_agents_run_concurrently(tmp_path):
    from agents.base_agent import BaseAgent
    from agents.tax_agent import TaxAgent
    agent = TaxAgent.__new__(TaxAgent)
    BaseAgent.__init__(agent, str(tmp_path))
    async def verify():
        started, ready = [], asyncio.Event()
        class Child:
            async def run(self, *args, **kwargs):
                started.append(True)
                if len(started) == 2:
                    ready.set()
                await asyncio.wait_for(ready.wait(), 1)
                return "ok"
        agent.registry = SimpleNamespace(get=lambda name: Child())
        result = await agent.dispatch_batch_node({"messages": [], "collaboration_chain": ["one", "two"], "thread_id": "s"})
        assert [r["agent"] for r in result["agent_outputs"]] == ["one", "two"]
        assert all(r["error"] is None for r in result["agent_outputs"])
    asyncio.run(verify())


def test_sql_failure_is_visible_and_document_is_quarantined(document_app):
    _, client, service, embedding, _ = document_app
    with patch("database.repositories.KnowledgeFileRepository.create", side_effect=OSError("disk full")):
        response = client.post("/api/knowledge/upload", headers=USER, data={"scope": "global"}, files={"file": ("demo.md", b"tax")})
    assert response.status_code == 502
    assert service._kb._metadata["files"][0]["status"] == "upload_failed"
    with tenant_scope("user_001"):
        assert embedding.search("tax") == []
    fid = service._kb._metadata["files"][0]["file_id"]
    assert client.delete(f"/api/knowledge/files/{fid}", headers=USER).status_code == 200


def test_local_delete_failure_preserves_disk_and_retries(document_app):
    _, client, service, embedding, _ = document_app
    fid = upload(client)
    path = PersistentDocumentStore._file_path
    original = path.read_bytes()
    import knowledge.embedding as module
    real_replace = module.os.replace
    def replace(source, target):
        if target == path:
            raise OSError("simulated disk failure")
        return real_replace(source, target)
    with patch.object(module.os, "replace", side_effect=replace):
        assert client.delete(f"/api/knowledge/files/{fid}", headers=USER).status_code == 500
    assert path.read_bytes() == original
    assert any(c.file_id == fid for c in PersistentDocumentStore().get_chunks())
    assert service._kb._metadata["files"][0]["status"] == "delete_failed"
    assert embedding.qdrant.client.count(embedding.qdrant.COLLECTION_NAME).count == 0
    assert client.delete(f"/api/knowledge/files/{fid}", headers=USER).status_code == 200
    assert not any(c.file_id == fid for c in PersistentDocumentStore().get_chunks())


def test_final_manifest_failure_keeps_retry_record(document_app):
    _, client, service, _, _ = document_app
    fid = upload(client)
    real_save = service._kb.save_metadata
    def save():
        if not service._kb._metadata["files"]:
            raise OSError("manifest write failed")
        real_save()
    with patch.object(service._kb, "save_metadata", side_effect=save):
        assert client.delete(f"/api/knowledge/files/{fid}", headers=USER).status_code == 500
    assert service._kb._metadata["files"][0]["file_id"] == fid
    assert client.delete(f"/api/knowledge/files/{fid}", headers=USER).status_code == 200


def test_corrupt_manifest_is_not_overwritten(tmp_path):
    from knowledge.kb import QdrantKnowledgeBase
    path = tmp_path / "user_documents_meta.json"
    path.write_text("{broken", encoding="utf-8")
    with pytest.raises(ValueError):
        QdrantKnowledgeBase(str(tmp_path))
    assert path.read_text(encoding="utf-8") == "{broken"


def test_checkpoint_cleanup_does_not_require_model(tmp_path, monkeypatch):
    import agents.base_agent as base
    main = importlib.import_module("web.main")
    monkeypatch.setattr(base, "_CHECKPOINT_DB_PATH", tmp_path / "checkpoints.sqlite")
    monkeypatch.setattr(main, "tax_agent", None)
    async def verify():
        import aiosqlite
        async with aiosqlite.connect(str(base._CHECKPOINT_DB_PATH)) as connection:
            await connection.execute("CREATE TABLE checkpoints (thread_id TEXT)")
            await connection.execute("CREATE TABLE writes (thread_id TEXT)")
            for table in ("checkpoints", "writes"):
                await connection.executemany(f"INSERT INTO {table} VALUES (?)", [("u:s",), ("u:s:knowledge_agent",), ("other:s",)])
            await connection.commit()
        with tenant_scope("u"):
            await main.clear_conversation_checkpoints("s")
        async with aiosqlite.connect(str(base._CHECKPOINT_DB_PATH)) as connection:
            for table in ("checkpoints", "writes"):
                cursor = await connection.execute(f"SELECT thread_id FROM {table}")
                assert await cursor.fetchall() == [("other:s",)]
    asyncio.run(verify())


def test_model_can_call_search_then_read_page():
    from agents.streaming import invoke_answer, answer_visible, web_tools_enabled, response_events
    from services.tools import kb_tools
    class Model:
        def __init__(self):
            self.calls = 0
        def bind_tools(self, tools):
            assert {t.name for t in tools} == {"search_web_policy", "read_web_page"}
            return self
        def stream(self, history):
            self.calls += 1
            if self.calls == 1:
                yield AIMessageChunk(content="", tool_call_chunks=[{"name":"search_web_policy", "args":'{"query":"tax", "source":"gov_cn", "limit":1}', "id":"search", "index":0}])
            elif self.calls == 2:
                assert any(getattr(m, "tool_call_id", "") == "search" for m in history)
                yield AIMessageChunk(content="", tool_call_chunks=[{"name":"read_web_page", "args":'{"url":"https://example.com/policy"}', "id":"page", "index":0}])
            else:
                assert any(getattr(m, "tool_call_id", "") == "page" and "actual policy" in m.content for m in history)
                yield AIMessageChunk(content="Verified source: ")
                yield AIMessageChunk(content="https://example.com/policy")
    events = []
    tokens = (answer_visible.set(True), web_tools_enabled.set(True), response_events.set(lambda event, data: events.append((event, data))))
    try:
        with patch.object(kb_tools, "search_web_policy", return_value=[{"url":"https://example.com/policy", "snippet":"summary"}]), patch.object(kb_tools, "read_web_page", return_value={"status":"success", "content":"actual policy", "url":"https://example.com/policy"}):
            result = invoke_answer(Model(), [])
        assert result.content == "Verified source: https://example.com/policy"
        assert len([e for e in events if e[0] == "token"]) == 2
    finally:
        answer_visible.reset(tokens[0])
        web_tools_enabled.reset(tokens[1])
        response_events.reset(tokens[2])


def test_web_reader_rejects_private_address_and_non_text():
    from services.tools.kb_tools import read_web_page
    with patch("services.tools.kb_tools.socket.getaddrinfo", return_value=[(2, 1, 6, "", ("127.0.0.1", 80))]), patch("services.tools.kb_tools.requests.get") as request:
        assert read_web_page("http://localhost/private")["status"] == "error"
        request.assert_not_called()
    response = MagicMock()
    response.__enter__.return_value = response
    response.status_code = 200
    response.is_redirect = False
    response.headers = {"Content-Type": "application/pdf"}
    with patch("services.tools.kb_tools.socket.getaddrinfo", return_value=[(2, 1, 6, "", ("93.184.216.34", 443))]), patch("services.tools.kb_tools.requests.get", return_value=response):
        assert read_web_page("https://example.com/doc.pdf")["status"] == "error"


def test_old_sqlite_file_table_is_upgraded_without_losing_rows(tmp_path, monkeypatch):
    import database.models as models
    from sqlalchemy import text, inspect
    engine = create_engine(f"sqlite:///{tmp_path / 'old.db'}")
    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE knowledge_files (id INTEGER PRIMARY KEY, path TEXT NOT NULL, create_at DATETIME NOT NULL, tags VARCHAR(200))"))
        connection.execute(text("INSERT INTO knowledge_files VALUES (1, 'old.md', '2026-10-08', 'legacy')"))
    monkeypatch.setattr(models, "engine", engine)
    models.create_tables()
    models.create_tables()
    assert {c["name"] for c in inspect(engine).get_columns("knowledge_files")} >= {"business_id", "file_name", "chunk_count"}
    with engine.connect() as connection:
        assert connection.execute(text("SELECT path, tags, chunk_count FROM knowledge_files WHERE id=1")).one() == ("old.md", "legacy", 0)
    engine.dispose()


def test_checkpoint_pruning_cleans_real_saver_tables(tmp_path):
    from agents.base_agent import BaseAgent, AsyncSqliteSaver
    from agents.tax_agent import TaxAgent
    agent = TaxAgent.__new__(TaxAgent)
    BaseAgent.__init__(agent, str(tmp_path))
    async def verify():
        async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "prune.db")) as saver:
            await saver.setup()
            for i in range(5):
                await saver.conn.execute("INSERT INTO checkpoints (thread_id, checkpoint_id) VALUES (?, ?)", ("u:s", str(i)))
                await saver.conn.execute("INSERT INTO writes (thread_id, checkpoint_id, task_id, idx, channel) VALUES (?, ?, 'task', 0, 'messages')", ("u:s", str(i)))
            await saver.conn.commit()
            agent._checkpointer = saver
            await agent._prune_checkpoints("u:s", keep_recent=2)
            for table in ("checkpoints", "writes"):
                cursor = await saver.conn.execute(f"SELECT checkpoint_id FROM {table} ORDER BY checkpoint_id")
                assert await cursor.fetchall() == [("3",), ("4",)]
    asyncio.run(verify())


def test_glm_final_answer_uses_supported_low_effort():
    from agents.streaming import invoke_answer, answer_visible, response_events
    model = MagicMock()
    model.model_name = "glm-5.3"
    model.stream.return_value = iter([AIMessageChunk(content="ok")])
    tokens = (answer_visible.set(True), response_events.set(lambda *args: None))
    try:
        assert invoke_answer(model, []).content == "ok"
        model.stream.assert_called_once_with([], reasoning_effort="low")
    finally:
        answer_visible.reset(tokens[0])
        response_events.reset(tokens[1])
