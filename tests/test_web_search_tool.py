"""
tests/test_web_search_tool.py
==============================
测试 services/tools/kb_tools.py 中的联网搜索 Tool。
使用 conftest.py 提供的 mock，不需要真实网络。
"""
import json
import pytest

from services.tools import WEB_SEARCH_TOOL
from services.tools.kb_tools import search_web_policy

# ============================================================
# 1. 纯函数 search_web_policy
# ============================================================

@pytest.mark.usefixtures("patch_bing_search")
class TestSearchWebPolicy:
    def test_returns_list(self):
        results = search_web_policy("2024 增值税")
        assert isinstance(results, list)
        assert len(results) > 0

    def test_result_has_required_fields(self):
        results = search_web_policy("test")
        item = results[0]
        for key in ("title", "url", "snippet", "source"):
            assert key in item, f"缺少字段 {key}"

    def test_source_tag_bing(self):
        """通用搜索结果来源应标记为 bing"""
        results = search_web_policy("test", source="duckduckgo")
        assert results and all(r["source"] == "bing" for r in results)

    def test_limit_respected(self, patch_bing_search):
        search_web_policy("test", source="duckduckgo", limit=3)
        # source=duckduckgo 只走一次通用搜索，params.count 应为 3
        first_call = patch_bing_search.call_args_list[0]
        assert first_call.kwargs.get("params", {}).get("count") == 3

# ============================================================
# 2. LangChain WEB_SEARCH_TOOL
# ============================================================

@pytest.mark.usefixtures("patch_bing_search")
class TestWebSearchTool:
    def test_tool_exists(self):
        assert WEB_SEARCH_TOOL is not None
        assert hasattr(WEB_SEARCH_TOOL, "name")

    def test_tool_invoke_returns_json(self):
        out = WEB_SEARCH_TOOL.invoke({"query": "2024 增值税最新政策"})
        data = json.loads(out)
        assert "found" in data
        assert data["found"] > 0
        assert "results" in data
        assert isinstance(data["results"], list)

    def test_tool_with_source_filter(self):
        out = WEB_SEARCH_TOOL.invoke({
            "query": "企业所得税",
            "source": "duckduckgo",
            "limit": 5,
        })
        data = json.loads(out)
        assert "found" in data

    def test_tool_description(self):
        assert WEB_SEARCH_TOOL.description and "联网" in WEB_SEARCH_TOOL.description

# ============================================================
# 3. 异常 / 无网络
# ============================================================

class TestWebSearchFailure:
    def test_network_error_returns_empty(self):
        """Bing 网络超时 → 返回空列表，不崩溃"""
        from unittest.mock import patch as _patch
        with _patch("requests.get", side_effect=ConnectionError("Network timeout")):
            results = search_web_policy("test")
            assert isinstance(results, list)
            assert results == []


# ============================================================
# 4. 删除联网检索结果（delete_web_search_results）
# ============================================================
def test_delete_all_web_search_results():
    """删除全部 web_search 结果时，只删 web_search 来源，不误删其它来源。

    回归：旧实现调用不存在的 store.get_all_chunks() 且把 ChunkDocument 当 dict 用，
    导致「删除全部」分支必然抛异常。
    """
    from unittest.mock import patch, MagicMock
    from services.tools.web_search_tool import delete_web_search_results
    from knowledge.embedding import ChunkDocument

    fake_store = MagicMock()
    fake_store.get_chunks.return_value = [
        ChunkDocument(id="c1", content="x", file_id="web_1",
                      metadata={"source_type": "web_search"}),
        ChunkDocument(id="c2", content="y", file_id="file_1",
                      metadata={"source_type": "user_upload"}),
    ]
    fake_embedding = MagicMock()
    fake_embedding.qdrant.delete_chunks_by_file_id.return_value = 1

    with patch("knowledge.embedding.PersistentDocumentStore", return_value=fake_store), \
         patch("knowledge.embedding.EmbeddingService", return_value=fake_embedding):
        result = delete_web_search_results(file_id=None)

    assert result["deleted_count"] == 1
    # 只应删除 web_search 来源对应的 file_id
    fake_embedding.qdrant.delete_chunks_by_file_id.assert_called_once_with("web_1")


# ============================================================
# 5. 去重 / gov_cn 过滤 / 联网检索入库
# ============================================================
@pytest.mark.usefixtures("patch_bing_search")
def test_search_web_policy_dedup():
    """source=all 时，通用搜索与 gov 搜索的重复 URL 应去重。"""
    results = search_web_policy("test", source="all", limit=10)
    urls = [r["url"] for r in results]
    assert len(urls) == len(set(urls)), "URL 不应重复"


@pytest.mark.usefixtures("patch_bing_search")
def test_search_web_policy_gov_cn_source_filters():
    """source=gov_cn 时只返回 gov.cn 域名结果。"""
    results = search_web_policy("test", source="gov_cn", limit=5)
    assert results and all("gov.cn" in r["url"] for r in results)
    assert all(r["source"] == "gov_cn" for r in results)


def test_search_and_ingest_web_ingests_chunks():
    """联网检索入库：应把搜索结果分块并写入 store。"""
    from unittest.mock import patch, MagicMock
    from services.tools.web_search_tool import search_and_ingest_web

    fake_results = [{"title": "t1", "url": "http://x/1", "snippet": "最新政策内容", "source": "duckduckgo"}]
    fake_store = MagicMock()
    fake_store.get_chunks.return_value = []
    fake_embedding = MagicMock()

    with patch("services.tools.kb_tools.search_web_policy", return_value=fake_results), \
         patch("knowledge.embedding.PersistentDocumentStore", return_value=fake_store), \
         patch("knowledge.embedding.EmbeddingService", return_value=fake_embedding):
        result = search_and_ingest_web("增值税最新政策", limit=1)

    assert result["ingested"] >= 1
    assert result["file_id"].startswith("web_")
    assert fake_embedding.add_chunks.called