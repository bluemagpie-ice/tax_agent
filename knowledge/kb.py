"""
知识库基类与工厂
---------------------------------
KnowledgeBase      : 所有具体知识库的抽象基类（定义通用接口）
KnowledgeBaseFactory : 根据类型创建对应知识库实例的工厂

设计说明：

- 采用"模板方法"思想：基类规定了 get_database_info / search / ingest_file 等标准接口；

- 后续若要接入"本地文件型知识库""Milvus 向量库""Elastic 全文检索"等，
  只需继承 KnowledgeBase 并实现对应方法即可。
"""

from __future__ import annotations

import abc
import json
import logging
import os
from pathlib import Path
from typing import Any, Optional



logger = logging.getLogger(__name__)

class KnowledgeBase(abc.ABC):
    """
    知识库抽象基类。

    一个"知识库"至少需要具备：
    1. 元数据管理（名称、描述、创建者、分块策略等）
    2. 文件入库（解析 → 分块 → 向量化 → 存储）
    3. 检索（基于向量 / 关键词的搜索）
    """

    # 子类必须覆盖
    kb_type: str = "base"

    def __init__(self, work_dir: str, kb_id: Optional[str] = None):
        """
        Args:
            work_dir: 该知识库使用的工作目录（存放元数据文件等）
            kb_id:    知识库唯一标识，可为空（首次创建时生成）
        """
        self.work_dir = work_dir
        os.makedirs(self.work_dir, exist_ok=True)
        self.kb_id = kb_id or self._generate_kb_id()
        self._metadata: dict[str, Any] = {}

        # 自动加载已有元数据
        if self._metadata_file_path.exists():
            self._load_metadata()

    # -------------------- 元数据相关 -------------------- #
    @property
    def _metadata_file_path(self) -> Path:
        return Path(self.work_dir) / f"{self.kb_id}_meta.json"

    def _generate_kb_id(self) -> str:
        import uuid
        return str(uuid.uuid4())

    def _load_metadata(self) -> dict[str, Any]:
        """从磁盘加载元数据（JSON 格式）"""
        if not self._metadata_file_path.exists():
            return {}
        try:
            with open(self._metadata_file_path, "r", encoding="utf-8") as f:
                self._metadata = json.load(f)
        except Exception as e:
            logger.warning(f"加载元数据失败: {e}")
            self._metadata = {}
        return self._metadata

    def save_metadata(self) -> None:
        """将元数据持久化到磁盘"""
        with open(self._metadata_file_path, "w", encoding="utf-8") as f:
            json.dump(self._metadata, f, ensure_ascii=False, indent=2)

    def get_database_info(self, kb_id: str, include_files: bool = False) -> Optional[dict]:
        """
        返回指定知识库的信息字典。
        默认实现只返回本地元数据；子类可扩展以携带文件列表等。
        """
        if kb_id != self.kb_id:
            return None
        info = dict(self._metadata)
        info["kb_id"] = self.kb_id
        info["kb_type"] = self.kb_type
        return info

    def normalize_additional_params(self, additional_params: Any) -> dict:
        """规范 additional_params，子类可覆盖"""
        if additional_params is None:
            return {}
        if isinstance(additional_params, dict):
            return additional_params
        if isinstance(additional_params, str):
            try:
                return json.loads(additional_params)
            except Exception:
                return {}
        return {}

    # -------------------- 抽象方法（必须由子类实现） -------------------- #
    @abc.abstractmethod
    async def ingest_file(self, file_path: str, **kwargs) -> dict:
        """
        文件入库：解析 → 分块 → 向量化 → 存储
        返回 {"status": ..., "file_id": ..., "chunk_count": ...}
        """
        ...

    @abc.abstractmethod
    async def search(self, query: str, limit: int = 5) -> list[dict]:
        """
        语义检索，返回相关文本块列表
        """
        ...

    @abc.abstractmethod
    async def delete_file(self, file_id: str) -> bool:
        """删除指定文件及其所有分块"""
        ...

    @abc.abstractmethod
    async def list_files(self) -> list[dict]:
        """列出当前知识库中所有文件"""
        ...

# ---------------------------------------------------------------------------- #

# 针对当前项目的默认实现：Qdrant + EmbeddingService

# ---------------------------------------------------------------------------- #
class QdrantKnowledgeBase(KnowledgeBase):
    """
    基于 Qdrant 的知识库默认实现。

    复用 knowledge/embedding.py 中的 EmbeddingService + QdrantService。
    """

    kb_type = "qdrant"

    def __init__(self, work_dir: str, kb_id: Optional[str] = None):
        super().__init__(work_dir=work_dir, kb_id=kb_id)

        # 运行期对象（懒加载，避免构造时强制依赖环境变量）
        self._store = None
        self._embedding_service = None

    # -------------------- 懒加载 embedding 服务 -------------------- #
    def _ensure_embedding(self):
        if self._embedding_service is not None:
            return self._embedding_service
        from knowledge.embedding import EmbeddingService, PersistentDocumentStore
        self._store = PersistentDocumentStore()
        self._embedding_service = EmbeddingService(store=self._store, use_qdrant=True)
        return self._embedding_service

    def is_file_ingested(self, file_id: str) -> bool:
        """
        检查 file_id 是否已入库。

        通过 Qdrant scroll 查询 payload 中 file_id 是否存在来判断。
        用于预置知识库去重——重启时跳过已入库的 preset 文件。
        """
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            embedding = self._ensure_embedding()
            if embedding.qdrant is None:
                return False
            records, _ = embedding.qdrant.client.scroll(
                collection_name=embedding.qdrant.COLLECTION_NAME,
                scroll_filter=Filter(
                    must=[FieldCondition(key="file_id", match=MatchValue(value=file_id))]
                ),
                limit=1,
            )
            return len(records) > 0
        except Exception as e:
            logger.warning(f"[is_file_ingested] 查询失败（默认返回 False 触发入库）: {e}")
            return False

    # -------------------- 实现抽象接口 -------------------- #
    async def ingest_file(self, file_path: str, **kwargs) -> dict:
        """
        入库流程：
        parser -> chunking -> embedding -> qdrant
        """
        import time
        from knowledge.parser import ParserFactory
        from knowledge.chunking import split_text
        from knowledge.embedding import ChunkDocument

        parser = ParserFactory.get_parser(file_path)
        markdown_text = parser.parse()
        if not markdown_text or "错误" in markdown_text:
            return {"status": "error", "message": markdown_text or "解析失败"}

        preset_id = kwargs.get("chunk_preset_id", "general")
        chunk_size = kwargs.get("chunk_size", 1024)
        chunk_overlap = kwargs.get("chunk_overlap", 128)
        file_id = kwargs.get("file_id") or f"file_{int(time.time())}"
        # ---- 收集税务元数据（可选，会被打平写入 Qdrant payload）----
        tax_metadata: dict = {}
        for key in ("tax_type", "document_number", "issuer", "effective_date",
                    "law_type", "is_current", "title", "source_type",
                    "expiry_date", "superseded_by"):
            val = kwargs.get(key)
            if val is not None and val != "":
                tax_metadata[key] = val
        # 若外部未传 source_type，默认标记为 user_upload（安全兜底）
        if "source_type" not in tax_metadata:
            tax_metadata["source_type"] = "user_upload"


        chunks = split_text(
            text=markdown_text,
            preset_id=preset_id,
            file_id=file_id,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        # 写入 store 并触发向量化
        embedding = self._ensure_embedding()
        documents = [
            ChunkDocument(id=c["id"], content=c["content"], file_id=c["file_id"], metadata=tax_metadata)
            for c in chunks
        ]
        embedding.store.add_chunks(documents)
        embedding.add_chunks(documents)

        # 记录元数据
        self._metadata.setdefault("files", [])
        self._metadata["files"].append({
            "file_id": file_id,
            "file_name": Path(file_path).name,
            "file_path": file_path,
            "chunk_count": len(chunks),
            "tax_metadata": tax_metadata,
        })
        self.save_metadata()

        return {
            "status": "success",
            "file_id": file_id,
            "chunk_count": len(chunks),
            "tax_metadata": tax_metadata,
            "message": f"共 {len(chunks)} 个分块已入库",
        }

    async def search(self, query: str, limit: int = 5) -> list[dict]:
        embedding = self._ensure_embedding()
        return embedding.search(query=query, limit=limit)

    async def delete_file(self, file_id: str) -> bool:
        """
        删除指定文件：元数据 + Qdrant 向量 + 内存 store 全链路清理

        Returns:
            True 表示至少删除了一项（元数据或向量）；False 表示该 file_id 不存在
        """
        deleted_any = False

        # 1. 清理元数据
        files = self._metadata.get("files", [])
        original_len = len(files)
        self._metadata["files"] = [f for f in files if f.get("file_id") != file_id]
        meta_removed = original_len - len(self._metadata["files"])
        if meta_removed > 0:
            self.save_metadata()
            deleted_any = True
            logger.info(f"[delete_file] 元数据中移除 file_id={file_id}")

        # 2. 清理 Qdrant 向量（按 payload.file_id 过滤）
        try:
            embedding = self._ensure_embedding()
            if embedding.qdrant is not None:
                qdrant_removed = embedding.qdrant.delete_chunks_by_file_id(file_id)
                if qdrant_removed > 0:
                    deleted_any = True
        except Exception as e:
            logger.error(f"[delete_file] Qdrant 删除失败: {e}")

        # 3. 清理内存 store（通过类方法操作类级别共享列表 + 同步到磁盘文件）
        try:
            from knowledge.embedding import PersistentDocumentStore
            removed_from_store = PersistentDocumentStore.remove_by_file_id(file_id)
            if removed_from_store > 0:
                logger.info(
                    f"[delete_file] 内存 store 清理了 {removed_from_store} 个 chunk"
                )
        except Exception as e:
            logger.error(f"[delete_file] 内存 store 清理失败: {e}")

        # 4. 清理磁盘上的原始文件
        file_path = None
        for f in self._metadata.get("files", []):
            if f.get("file_id") == file_id:
                file_path = f.get("file_path")
        if file_path:
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    logger.info(f"[delete_file] 已删除磁盘文件: {file_path}")
                    deleted_any = True
            except Exception as e:
                logger.error(f"[delete_file] 磁盘文件删除失败: {e}")

        return deleted_any
    async def list_files(self) -> list[dict]:
        return list(self._metadata.get("files", []))

# ---------------------------------------------------------------------------- #

# 工厂

# ---------------------------------------------------------------------------- #
class KnowledgeBaseFactory:
    """
    知识库工厂：根据 kb_type 创建对应的 KnowledgeBase 实例。
    """
    # 已注册的类型表
    _registry: dict[str, type[KnowledgeBase]] = {
        "qdrant": QdrantKnowledgeBase,
    }

    @classmethod
    def register(cls, kb_type: str, kb_class: type[KnowledgeBase]) -> None:
        """注册一种新的知识库类型（扩展点）"""
        cls._registry[kb_type] = kb_class
        logger.info(f"已注册知识库类型: {kb_type} -> {kb_class.__name__}")

    @classmethod
    def is_type_supported(cls, kb_type: str) -> bool:
        return kb_type in cls._registry

    @classmethod
    def create(cls, kb_type: str, work_dir: str, kb_id: Optional[str] = None) -> KnowledgeBase:
        if kb_type not in cls._registry:
            supported = list(cls._registry.keys())
            raise ValueError(
                f"不支持的知识库类型: {kb_type}，当前支持: {supported}"
            )
        kb_class = cls._registry[kb_type]
        return kb_class(work_dir=work_dir, kb_id=kb_id)