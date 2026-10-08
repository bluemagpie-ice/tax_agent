"""
pytest 共用 fixtures 和 mock 工具
=================================
"""
import sys
import pytest
from unittest.mock import patch, MagicMock

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