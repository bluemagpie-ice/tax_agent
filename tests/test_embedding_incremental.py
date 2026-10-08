"""增量索引回归测试。

重点：add_chunks() 只对新 chunk 做向量化，不重算 store 里已有的旧 chunk，
避免每次上传新文档都触发全库 embedding。
"""

from unittest.mock import MagicMock, patch

from knowledge.embedding import ChunkDocument, EmbeddingService


def test_add_chunks_only_embeds_new_chunks():
    old = [
        ChunkDocument(id="old1", content="旧内容一", file_id="f1"),
        ChunkDocument(id="old2", content="旧内容二", file_id="f1"),
    ]
    new = [ChunkDocument(id="new1", content="新内容一", file_id="f2")]

    fake_store = MagicMock()
    fake_store.get_chunks.return_value = old

    # 关闭 Qdrant/BM25/Neo4j，避免网络；mock 掉 embedding 调用
    with patch.object(
        EmbeddingService, "_embed_texts", return_value=[[0.1, 0.2, 0.3]]
    ) as mock_embed:
        svc = EmbeddingService(store=fake_store, use_qdrant=False, use_bm25=False, use_neo4j=False)
        # __init__ 里 rebuild() 已对 old 做过一次 embedding，这里只看增量
        svc.add_chunks(new)

        # 最后一次 _embed_texts 的入参应只有新增的那 1 条，而不是 old+new
        embedded_texts = mock_embed.call_args[0][0]
        assert embedded_texts == ["新内容一"], f"❌ 应只向量化新增块，实际 {embedded_texts}"


def test_add_chunks_skips_empty_chunks():
    fake_store = MagicMock()
    fake_store.get_chunks.return_value = []

    with patch.object(
        EmbeddingService, "_embed_texts", return_value=[[0.1, 0.2, 0.3]]
    ) as mock_embed:
        svc = EmbeddingService(store=fake_store, use_qdrant=False, use_bm25=False, use_neo4j=False)
        mock_embed.reset_mock()

        empty_or_blank = [
            ChunkDocument(id="e1", content="", file_id="f3"),
            ChunkDocument(id="e2", content="   ", file_id="f3"),
        ]
        svc.add_chunks(empty_or_blank)

        # 全部为空/空白 → 不应触发任何 embedding
        assert mock_embed.call_count == 0
