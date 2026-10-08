"""
联网检索 → 入库 Tool
====================

职责：
  1. 接收用户 query → 调用搜索引擎获取最新税务信息
  2. 将搜索结果分块 → 向量化 → 写入 Qdrant（source_type="web_search"）
  3. 提供 LangChain StructuredTool 供 Agent 调用
  4. 提供删除接口，按条件清理联网检索结果

与 kb_tools.py 的区别：
  kb_tools 的 search_web_policy 只返回搜索结果（不入库）；
  本模块在搜索的基础上增加了入库 + 可删除管理的能力。
"""

import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def search_and_ingest_web(
    query: str,
    limit: int = 5,
) -> dict:
    """
    联网搜索并入库（纯函数入口）。

    流程：搜索 → 分块 → 向量化 → Qdrant

    :param query: 搜索关键词
    :param limit: 返回前 N 条
    :return: {"ingested": N, "file_id": "...", "results": [...]}
    """
    from services.tools.kb_tools import search_web_policy

    # 1. 联网搜索
    raw_results = search_web_policy(query, source="all", limit=limit)
    if not raw_results:
        return {"ingested": 0, "file_id": "", "results": [], "message": "联网搜索未返回结果"}

    # 2. 组装可入库的文本
    chunks_for_ingest = []
    for i, r in enumerate(raw_results):
        title = r.get("title", "无标题")
        url = r.get("url", "")
        snippet = r.get("snippet", "")
        source = r.get("source", "web")
        # 拼接为一条可读的文本块
        text = f"标题：{title}\n来源：{url}\n摘要：{snippet}"
        chunks_for_ingest.append({
            "content": text,
            "title": title,
            "url": url,
            "source": source,
            "index": i,
        })

    # 3. 入库到 Qdrant
    import time
    from knowledge.chunking import split_text
    from knowledge.embedding import ChunkDocument, EmbeddingService, PersistentDocumentStore
    from pathlib import Path

    file_id = f"web_{int(time.time())}_{Path(query).stem if query else 'search'}"

    store = PersistentDocumentStore()
    embedding = EmbeddingService(store=store, use_qdrant=True)

    documents = []
    for chunk in chunks_for_ingest:
        # 对每条搜索结果做一个最小分块（按句子切分，块大小 1024）
        sub_chunks = split_text(
            text=chunk["content"],
            preset_id="general",
            file_id=file_id,
            chunk_size=1024,
            chunk_overlap=128,
        )
        for sc in sub_chunks:
            documents.append(ChunkDocument(
                id=sc["id"],
                content=sc["content"],
                file_id=file_id,
                metadata={
                    "source_type": "web_search",
                    "title": chunk["title"],
                    "url": chunk.get("url", ""),
                    "is_current": True,
                },
            ))

    if documents:
        store.add_chunks(documents)
        embedding.add_chunks(documents)
        logger.info(f"[web_search_tool] 已入库 {len(documents)} 个分块，file_id={file_id}")

    return {
        "ingested": len(documents),
        "file_id": file_id,
        "results": raw_results,
        "message": f"已入库 {len(documents)} 个分块",
    }

def delete_web_search_results(file_id: Optional[str] = None) -> dict:
    """
    删除联网检索结果。

    :param file_id: 指定要删除的 file_id；为 None 时删除全部 web_search 来源的数据
    :return: {"deleted_count": N, "message": "..."}
    """
    try:
        from knowledge.embedding import EmbeddingService, PersistentDocumentStore

        store = PersistentDocumentStore()
        embedding = EmbeddingService(store=store, use_qdrant=True)

        if file_id:
            count = embedding.qdrant.delete_chunks_by_file_id(file_id)
            PersistentDocumentStore.remove_by_file_id(file_id)
            return {"deleted_count": count, "message": f"已删除 file_id={file_id} 的 {count} 条联网检索结果"}
        else:
            total = 0
            web_file_ids = []
            for chunk in store.get_chunks():
                meta = getattr(chunk, "metadata", {}) or {}
                if isinstance(meta, dict) and meta.get("source_type") == "web_search":
                    fid = getattr(chunk, "file_id", "") or ""
                    if fid and fid not in web_file_ids:
                        web_file_ids.append(fid)
            for fid in web_file_ids:
                count = embedding.qdrant.delete_chunks_by_file_id(fid)
                total += count
                PersistentDocumentStore.remove_by_file_id(fid)
            return {"deleted_count": total, "message": f"已删除全部 {len(web_file_ids)} 批联网检索结果，共 {total} 条"}
    except Exception as e:
        logger.error(f"[web_search_tool] 删除失败: {e}")
        return {"deleted_count": 0, "message": f"删除失败: {str(e)}"}

# ---- LangChain Tool 封装 ----

try:
    from langchain_core.tools import StructuredTool
    from pydantic import BaseModel, Field

    class WebIngestInput(BaseModel):
        query: str = Field(description="要联网搜索的税务关键词，如 '2024年增值税最新政策'")
        limit: int = Field(default=5, description="返回前 N 条，默认 5，最多 10")

    def _tool_search_and_ingest(query: str, limit: int = 5) -> str:
        result = search_and_ingest_web(query, limit=limit)
        return json.dumps(result, ensure_ascii=False, indent=2)

    WEB_INGEST_TOOL = StructuredTool.from_function(
        func=_tool_search_and_ingest,
        name="search_and_ingest_web_policy",
        description=(
            "联网搜索最新税务政策并自动入库。\n"
            "适用场景：用户问及最新政策、今年新规、时效性强的税务问题时。\n"
            "搜索范围：DuckDuckGo 通用搜索 + 财政部/税务总局官网。\n"
            "搜索结果会自动写入知识库（标记为联网检索来源），后续 RAG 检索可命中。"
        ),
        args_schema=WebIngestInput,
        return_direct=False,
    )
except ImportError:
    WEB_INGEST_TOOL = None
