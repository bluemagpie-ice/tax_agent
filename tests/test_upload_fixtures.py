"""原始fixture解析回归；HTTP持久化使用隔离SQL/Qdrant，真实OCR与云端另行实测。"""
from pathlib import Path
from types import SimpleNamespace
import json
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage

from tests.test_runtime_document_flow import document_app, USER
from tests.test_agent_workflows import make_agent

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize("name,expected", [
    ("test_txt.txt", "13%"), ("test_docx_long.docx", "印花税"),
    ("test_csv.csv", "156000"), ("test_xlsx.xlsx", "945000"),
    ("test_pdf_medium.pdf", "纳税人识别号"), ("test_pdf_long.pdf", "免、抵、退"),
])
def test_original_fixture_text_is_preserved(name, expected):
    from knowledge.parser import ParserFactory
    text = ParserFactory.get_parser(str(FIXTURES / name)).parse()
    assert expected in text
    assert not text.startswith(("文件解析错误", "依赖错误", "错误："))


def test_all_text_pdf_pages_are_present():
    import re
    from knowledge.parser import PdfParser
    text = PdfParser(str(FIXTURES / "test_pdf_long.pdf")).parse()
    assert re.findall(r"## 第 (\d+) 页", text) == [str(i) for i in range(1, 33)]


@pytest.mark.parametrize("name,body,status", [
    ("empty.txt", b"", 400), ("unsupported.exe", b"data", 400),
    ("too_large.txt", b"x" * (20 * 1024 * 1024 + 1), 413),
], ids=["empty", "unsupported", "over-20-mib"])
def test_rejected_upload_has_no_disk_or_database_side_effect(document_app, name, body, status):
    main, client, service, embedding, _ = document_app
    response = client.post("/api/knowledge/upload", headers=USER, files={"file": (name, body)})
    assert response.status_code == status
    assert response.json()["detail"]
    assert list(Path(main.PATHS["uploads"]).iterdir()) == []
    assert service._kb._metadata.get("files", []) == []
    assert embedding.qdrant.client.count(embedding.qdrant.COLLECTION_NAME).count == 0


def test_original_csv_upload_metadata_and_persistence(document_app):
    _, client, service, embedding, _ = document_app
    path = FIXTURES / "test_csv.csv"
    response = client.post("/api/knowledge/upload", headers=USER,
                           files={"file": (path.name, path.read_bytes(), "text/csv")})
    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == path.name and data["size"] == path.stat().st_size
    assert data["mime_type"] == "text/csv"
    assert data["file_name"] == path.name  # 原API字段仍保留。
    assert client.get(data["url"], headers=USER).json()["content"].find("156000") >= 0
    assert client.get(data["url"]).status_code == 422
    record = service._kb._metadata["files"][0]
    assert Path(record["file_path"]).read_bytes() == path.read_bytes()
    from database.repositories import KnowledgeFileRepository, KnowledgeChunkRepository
    row = KnowledgeFileRepository().get_by_business_id(data["file_id"])
    assert row and KnowledgeChunkRepository().count_by_file_id(row.id) == data["chunk_count"]
    assert embedding.qdrant.client.count(embedding.qdrant.COLLECTION_NAME).count == data["chunk_count"]


def test_ocr_empty_result_is_failure_and_quarantined(document_app, monkeypatch):
    _, client, service, embedding, _ = document_app
    from knowledge.parser import ImageParser
    monkeypatch.setattr(ImageParser, "_get_ocr_engine", lambda self: SimpleNamespace(predict=lambda image: []))
    response = client.post("/api/knowledge/upload", headers=USER,
                           files={"file": ("test_jpeg.jpeg", (FIXTURES / "test_jpeg.jpeg").read_bytes(), "image/jpeg")})
    assert response.status_code == 502
    assert "OCR" in response.json()["detail"]
    assert service._kb._metadata["files"][0]["status"] == "upload_failed"
    assert embedding.qdrant.client.count(embedding.qdrant.COLLECTION_NAME).count == 0


def test_whitespace_document_is_not_success(document_app):
    _, client, service, embedding, _ = document_app
    response = client.post("/api/knowledge/upload", headers=USER, files={"file": ("blank.txt", b" \n\t")})
    assert response.status_code == 502
    assert "有效正文" in response.json()["detail"]
    assert service._kb._metadata["files"][0]["status"] == "upload_failed"
    assert embedding.qdrant.client.count(embedding.qdrant.COLLECTION_NAME).count == 0


def test_pdf_ocr_page_failure_does_not_become_document(monkeypatch):
    import pdf2image
    from PIL import Image
    from knowledge.parser import PdfParser
    parser = PdfParser(str(FIXTURES / "test_pdf_2.pdf"))
    parser._ocr_engine = SimpleNamespace(predict=MagicMock(side_effect=[
        [{"rec_texts": ["真实第1页"], "rec_scores": [0.99]}], RuntimeError("engine failure")]))
    monkeypatch.setattr(pdf2image, "pdfinfo_from_path", lambda *a, **k: {"Pages": 2})
    rendered = []
    def render(*args, **kwargs):
        rendered.append((kwargs["first_page"], kwargs["last_page"]))
        return [Image.new("RGB", (10, 10))]
    monkeypatch.setattr(pdf2image, "convert_from_path", render)
    with pytest.raises(RuntimeError, match="第 2 页OCR识别失败"):
        parser.pdf_ocr_parse()
    assert rendered == [(1, 1), (2, 2)]


@pytest.mark.parametrize("stream", [False, True])
def test_uploaded_text_reaches_autonomous_graph_and_critic(document_app, tmp_path, monkeypatch, stream):
    main, client, service, _, _ = document_app
    client.post("/api/conversations", headers=USER, json={"session_id": "fixture-chat"})
    response = client.post("/api/knowledge/upload", headers=USER,
        data={"scope": "session", "session_id": "fixture-chat"},
        files={"file": ("test_txt.txt", (FIXTURES / "test_txt.txt").read_bytes(), "text/plain")})
    fid = response.json()["file_id"]
    agent = make_agent(tmp_path, [AIMessage(content="test_txt.txt：13%、9%、6%。")])
    agent.registry = SimpleNamespace(workdir_root=str(tmp_path), _cache={})
    monkeypatch.setattr(main, "tax_agent", agent)
    response = client.post("/api/chat/stream" if stream else "/api/chat", headers=USER,
        json={"user_id": "u", "query": "附件有哪些税率？", "session_id": "fixture-chat", "uploaded_file_ids": [fid]})
    assert response.status_code == 200
    assert "13%" in response.text
    assert "进项税额可以抵扣" in agent.llm.histories[0][-1].content
    from services.security import tenant_scope, document_session
    with tenant_scope("other-user"):
        with pytest.raises(PermissionError):
            agent._enrich_query_with_files("query", {"uploaded_file_ids": [fid]})
    with tenant_scope("user_001"):
        token = document_session.set("another-chat")
        try:
            with pytest.raises(PermissionError):
                agent._enrich_query_with_files("query", {"uploaded_file_ids": [fid]})
        finally:
            document_session.reset(token)
