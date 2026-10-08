"""
tests/test_tax_policy_tool.py
==============================
测试 services/tools/kb_tools.py 中的税法检索 Tool。
使用 conftest.py 提供的 mock，不需要真实 Qdrant。
"""
import json
import pytest

from services.tools import search_tax_policy, TAX_POLICY_TOOL

# ============================================================
# 1. 纯函数 search_tax_policy — 需要 mock EmbeddingService
# ============================================================

@pytest.mark.usefixtures("patch_embedding")
class TestSearchTaxPolicy:
    def test_returns_list(self):
        results = search_tax_policy("增值税")
        assert isinstance(results, list)
        assert len(results) > 0

    def test_result_has_required_fields(self):
        """
        验证搜索结果包含税法检索所需的全部元数据字段。
        两层防线：
          1. tax_type 必须是顶层 key（不是嵌套在 metadata 里）——证明经过了 payload 透出
          2. "metadata" 这个键不能存在于返回中（真实 Qdrant payload 是打平存储的）
        """
        results = search_tax_policy("增值税")
        item = results[0]
        required_keys = {
            "id", "score", "content", "file_id",  # 顶层固定字段
            "title", "document_number", "issuer",
            "effective_date", "law_type", "is_current", "tax_type",
        }
        assert required_keys.issubset(set(item.keys()))
        # 核心：tax_type 是平铺字段，不是 metadata[:子键]
        assert item.get("tax_type") == "vat"
        # metadata 键不应存在（锁死不再有嵌套结构回归）
        assert "metadata" not in item

    def test_tax_type_filter_applied_to_filters_dict(self, patch_embedding):
        """
        tax_type="vat" 应当被组装成 filters={"tax_type": "vat"} 传给 EmbeddingService.search。
        用 assert_called_once_with 精确锁定，防止 MagicMock 的假阳性。
        """
        search_tax_policy("抵扣", tax_type="vat")
        patch_embedding.search.assert_called_once()
        called_kwargs = patch_embedding.search.call_args.kwargs
        assert called_kwargs["filters"]["tax_type"] == "vat"

    def test_is_current_filter(self, patch_embedding):
        """is_current=True 应当被组装进 filters"""
        search_tax_policy("优惠", is_current=True)
        patch_embedding.search.assert_called_once()
        called_kwargs = patch_embedding.search.call_args.kwargs
        assert called_kwargs["filters"]["is_current"] is True

    def test_limit_pass_through(self, patch_embedding):
        """limit 参数应当原封不动透传给 EmbeddingService.search"""
        search_tax_policy("test", limit=10)
        patch_embedding.search.assert_called_once()
        assert patch_embedding.search.call_args.kwargs["limit"] == 10

# ============================================================
# 2. LangChain TAX_POLICY_TOOL 封装
# ============================================================

@pytest.mark.usefixtures("patch_embedding")
class TestTaxPolicyTool:
    def test_tool_exists(self):
        assert TAX_POLICY_TOOL is not None
        assert hasattr(TAX_POLICY_TOOL, "name")
        assert "tax_policy" in TAX_POLICY_TOOL.name or "search" in TAX_POLICY_TOOL.name.lower()

    def test_tool_invoke_returns_json(self):
        out = TAX_POLICY_TOOL.invoke({"query": "增值税"})
        data = json.loads(out)
        assert "found" in data
        assert data["found"] > 0
        assert "results" in data
        assert isinstance(data["results"], list)

    def test_tool_invoke_with_tax_type(self):
        out = TAX_POLICY_TOOL.invoke({
            "query": "企业所得税",
            "tax_type": "corporate_income",
            "limit": 3,
        })
        data = json.loads(out)
        assert "found" in data

    def test_tool_description_not_empty(self):
        assert TAX_POLICY_TOOL.description and len(TAX_POLICY_TOOL.description) > 10

# ============================================================
# 3. 无结果时的行为
# ============================================================

class TestEmptyResult:
    def test_embedding_failure_returns_empty_list(self):
        """EmbeddingService 完全连不上 → 返回空列表，不应崩溃"""
        from unittest.mock import patch as _patch
        with _patch(
            "knowledge.embedding.EmbeddingService",
            side_effect=ConnectionError("Qdrant offline"),
        ):
            results = search_tax_policy("任何查询")
            assert results == []

    def test_tool_handles_no_result(self):
        """Tool 层面：空列表要返回合法 JSON 而不是崩"""
        from unittest.mock import patch as _patch
        with _patch(
            "knowledge.embedding.EmbeddingService",
            side_effect=ConnectionError("Qdrant offline"),
        ):
            out = TAX_POLICY_TOOL.invoke({"query": "随便搜"})
            data = json.loads(out)
            assert "found" in data
            assert data["found"] == 0