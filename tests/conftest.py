"""
pytest 共用 fixtures 和 mock 工具
=================================
"""
import sys
import os
import uuid
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock

# Establish isolation before test-module collection imports any application code.
_TEST_ROOT = Path(__file__).resolve().parents[1] / ".test_runs" / ("state-" + uuid.uuid4().hex)
_TEST_ROOT.mkdir(parents=True)
os.environ["PYTHON_DOTENV_DISABLED"] = "1"
os.environ["DATABASE_URL"] = "sqlite:///" + (_TEST_ROOT / "business.db").as_posix()
os.environ["TAX_AGENT_DATA_DIR"] = str(_TEST_ROOT / "data")
os.environ["TAX_AGENT_UPLOAD_DIR"] = str(_TEST_ROOT / "uploads")
os.environ["AUTH_USER_TOKEN"] = "test_token_123"
os.environ["AUTH_ADMIN_TOKEN"] = "admin_token_456"
os.environ.pop("DEVELOPER_TOKEN", None)
os.environ["AUTH_SESSION_SECRET"] = "test-only-signing-fixture"
for _key in ("LLM_API_KEY", "EMBEDDING_API_KEY", "QDRANT_API_KEY", "QDRANT_CLUSTER_URL"):
    os.environ[_key] = ""


def pytest_collection_modifyitems(config, items):
    for item in items:
        if item.path.name == "test_api_connectivity.py" or "test_rag_end_to_end" in item.nodeid:
            item.add_marker(pytest.mark.live)
        if item.get_closest_marker("live"):
            item.add_marker(pytest.mark.skip(reason="真实 API 测试单独执行，默认禁止付费调用与生产 Qdrant 写入"))


@pytest.fixture(autouse=True)
def isolate_runtime(tmp_path, monkeypatch, request):
    import socket
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    import services.config_manager as config
    import database.models as models
    import database.repositories as repositories
    import database.init_db as init_db
    import agents.base_agent as base
    import agents.memory as memory

    engine = create_engine("sqlite:///" + (tmp_path / "business.db").as_posix(),
                           connect_args={"check_same_thread": False})
    sessions = sessionmaker(bind=engine)
    for module in (models, repositories, init_db):
        if hasattr(module, "engine"):
            monkeypatch.setattr(module, "engine", engine)
        monkeypatch.setattr(module, "SessionLocal", sessions)
    monkeypatch.setattr(models, "DATABASE_URL", str(engine.url))
    models.create_tables()
    monkeypatch.setattr(config, "_CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config, "_CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(base, "_CHECKPOINT_DB_PATH", tmp_path / "checkpoints.sqlite")
    monkeypatch.setattr(base, "_conversation_memory", None)
    monkeypatch.setattr(base.BaseAgent, "base_workdir", str(tmp_path / "agents"))
    monkeypatch.setattr(memory, "DATA_DIR", tmp_path)
    if request.path.name in ("test_knowledge_agent.py", "test_rag_sub_agent.py"):
        monkeypatch.setenv("LLM_API_KEY", "unit-test-placeholder")
        monkeypatch.setenv("LLM_MODEL_ID", "unit-test-model")
        monkeypatch.setenv("LLM_BASE_URL", "https://provider.example/v1")
        monkeypatch.setattr("knowledge.embedding._NEO4J_AVAILABLE", False)
    monkeypatch.setenv("TAX_AGENT_MODEL_ALLOWED_ORIGINS", "http://localhost:8000,http://localhost:6333")
    # DNS is a test dependency, never resolve production hosts from unit tests.
    def public_dns(host, port, *args, **kwargs):
        import ipaddress
        try:
            address = str(ipaddress.ip_address(host))
        except ValueError:
            address = "127.0.0.1" if host == "localhost" else "93.184.216.34"
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, port))]
    monkeypatch.setattr(socket, "getaddrinfo", public_dns)
    original_connect = socket.socket.connect
    def reject_network(sock, address):
        # Windows asyncio implements its self-pipe with a loopback socketpair.
        import inspect
        caller = inspect.currentframe().f_back
        if caller.f_code.co_name == "_fallback_socketpair" and caller.f_globals.get("__name__") == "socket":
            return original_connect(sock, address)
        raise OSError("测试禁止外部网络；请使用 Mock 或独立 live 验证")
    monkeypatch.setattr(socket.socket, "connect", reject_network)
    monkeypatch.setattr(socket.socket, "connect_ex", reject_network)
    yield
    # Legacy helpers replace module globals directly; monkeypatch restores originals.
    for module in (models, repositories, init_db):
        replacement = getattr(module, "engine", engine)
        if replacement is not engine:
            replacement.dispose()
    engine.dispose()

# ============================================================
# Mock EmbeddingService — 给税法检索 Tool 用
# ============================================================

@pytest.fixture
def mock_embedding_service():
    """
    模拟一个 EmbeddingService，search() 返回硬编码的税法条文结果。
    让 test_search_tax_policy 不需要真实 Qdrant 就能跑。

    使用 create_autospec 创建带真实类签名约束的 mock：
      - 调用不存在的方法（如 .nonexistent()）会抛 TypeError
      - 这样能防止"测试锁定了不存在接口"的假阳性
    """
    fake_results = [
        {
            "id": "chunk_001",
            "score": 0.92,
            "content": "增值税暂行条例第一条：在中华人民共和国境内销售货物或者提供加工、修理修配劳务以及进口货物的单位和个人，为增值税的纳税人。",
            "file_id": "file_vat_001",
            "title": "中华人民共和国增值税暂行条例",
            "document_number": "国务院令第 538 号",
            "issuer": "国务院",
            "effective_date": "2009-01-01",
            "law_type": "行政法规",
            "is_current": True,
            "tax_type": "vat",
        },
        {
            "id": "chunk_002",
            "score": 0.85,
            "content": "财政部 税务总局公告 2023 年第 19 号：小规模纳税人季度销售额不超过 30 万元免征增值税。",
            "file_id": "file_vat_002",
            "title": "关于增值税小规模纳税人减免政策的公告",
            "document_number": "财政部 税务总局公告 2023 年第 19 号",
            "issuer": "财政部 国家税务总局",
            "effective_date": "2023-01-01",
            "law_type": "公告",
            "is_current": True,
            "tax_type": "vat",
        },
        {
            "id": "chunk_003",
            "score": 0.78,
            "content": "企业所得税法第四条：企业所得税的税率为 25%。",
            "file_id": "file_corp_001",
            "title": "中华人民共和国企业所得税法",
            "document_number": "主席令第 63 号",
            "issuer": "全国人大常委会",
            "effective_date": "2008-01-01",
            "law_type": "法律",
            "is_current": True,
            "tax_type": "corporate_income",
        },
    ]

    # create_autospec 会读取 EmbeddingService 的真实方法名
    # 调用不存在的方法会抛 AttributeError/TypeError
    from unittest.mock import create_autospec
    from knowledge.embedding import EmbeddingService

    mock_svc = create_autospec(EmbeddingService, instance=True)
    mock_svc.search.return_value = fake_results
    return mock_svc

@pytest.fixture
def patch_embedding(mock_embedding_service):
    """把 knowledge.embedding.EmbeddingService 替换成 mock"""
    with patch(
        "knowledge.embedding.EmbeddingService",
        return_value=mock_embedding_service,
    ):
        with patch("knowledge.embedding.PersistentDocumentStore"):
            yield mock_embedding_service

# ============================================================
# Mock Bing —— 给联网搜索 Tool 用
# ============================================================

@pytest.fixture
def patch_bing_search():
    """把 requests.get 替换成 mock，返回固定 Bing 结果页 HTML。"""
    fake_bing_html = """
    <html><body>
      <li class="b_algo"><h2><a href="https://www.mof.gov.cn/zhengce/202401/t.html">财政部 税务总局关于增值税小规模纳税人减免政策的公告</a></h2>
        <div class="b_caption"><p>2024 年小规模纳税人月销售额 10 万以下免征增值税...</p></div></li>
      <li class="b_algo"><h2><a href="https://www.chinatax.gov.cn/chinatax/n371/c1020000/content.html">国家税务总局：小型微利企业所得税优惠政策延续</a></h2>
        <div class="b_caption"><p>小型微利企业年应纳税所得额 300 万以下部分减按 25% 计入...</p></div></li>
    </body></html>
    """
    fake_resp = MagicMock()
    fake_resp.text = fake_bing_html
    fake_resp.apparent_encoding = "utf-8"
    with patch("requests.get", return_value=fake_resp) as mock_get:
        yield mock_get


@pytest.fixture(autouse=True)
def _isolate_persistent_document_store(tmp_path):
    """隔离 PersistentDocumentStore 到临时文件，防止测试读写/删除真实 data/document_store.json。"""
    from knowledge.embedding import PersistentDocumentStore
    PersistentDocumentStore._file_path = tmp_path / "test_document_store.json"
    PersistentDocumentStore._shared_chunks = []
    PersistentDocumentStore._loaded = False
    yield
    PersistentDocumentStore._file_path = None
    PersistentDocumentStore._shared_chunks = []
    PersistentDocumentStore._loaded = False
