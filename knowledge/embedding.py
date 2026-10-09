"""
提供向量化服务、存储、检索服务
使用硅基流动api

多路召回架构：
    1. Qdrant 向量数据库（语义检索）
    2. Neo4j 图数据库（实体关系检索）
    3. BM25 词法索引（关键词检索）
三者通过 RRF (Reciprocal Rank Fusion) 融合重排。

"""
# todo：进阶：检索时采用源文档保留功能，同时增加父文档检索和多向量检索功能。
# todo：embedding和retrieve的代码内容是否需要分开？

import os
import json
import time
import threading
from pathlib import Path
from typing import List, Optional, Tuple

from openai import OpenAI
from dotenv import load_dotenv
from services.config_manager import ConfigManager
from services.cache_utils import TTLCache
from services.security import current_tenant_id, document_session, document_visible, blocked_document_ids
import logging
import numpy as np

_embed_cache = TTLCache(ttl_seconds=1800, maxsize=512)
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, PayloadSchemaType

# ---------- 可选依赖（缺少时降级） ---------- #
try:
    from neo4j import GraphDatabase
    _NEO4J_AVAILABLE = True
except ImportError:
    _NEO4J_AVAILABLE = False
    logging.warning("neo4j 未安装，图数据库功能不可用。请执行: pip install neo4j")

try:
    from rank_bm25 import BM25Okapi
    _BM25_AVAILABLE = True
except ImportError:
    _BM25_AVAILABLE = False
    logging.warning("rank-bm25 未安装，BM25 词法检索不可用。请执行: pip install rank-bm25")

try:
    import jieba
    jieba.setLogLevel(logging.WARNING)  # 抑制 jieba 冗余日志
    _JIEBA_AVAILABLE = True
except ImportError:
    _JIEBA_AVAILABLE = False
    logging.warning("jieba 未安装，中文分词将使用简单切分。请执行: pip install jieba")

load_dotenv()

# ==================== 工具函数 ==================== #
def default_preprocessing_func(text: str) -> List[str]:
    """
    默认预处理函数（参考 LangChain BM25Retriever.default_preprocessing_func）

    LangChain 默认实现是 `text.split()`，这里针对中文场景做了增强：
    ─────────────────────────────────────────────────────
    英文文本 → 直接 split() 按空白切分（兼容 LangChain 默认行为）
    中文文本 → jieba 分词；若 jieba 不可用则回退到 2-gram 切分
    ─────────────────────────────────────────────────────
    为什么这样设计？
    - 英文：自然空格分隔，split() 即可
    - 中文：没有空格，需要 jieba 分词
    - 回退方案：防止 jieba 未安装时整个 BM25 不可用
    """
    if not text:
        return []
    # 检测是否包含中文（CJK Unified Ideographs 范围）
    has_chinese = any('\u4e00' <= ch <= '\u9fff' for ch in text)

    if has_chinese and _JIEBA_AVAILABLE:
        # 中文 + jieba：最理想情况
        return [t.strip() for t in jieba.cut(text) if t.strip()]

    if not has_chinese:
        # 纯英文/数字：LangChain 默认行为 text.split()
        return text.split()

    # 含中文但 jieba 不可用 → 2-gram 回退（效果较差，但保证可用）
    if len(text) <= 1:
        return [text]
    return [text[i:i+2] for i in range(len(text)-1)]



# ==================== RRF 融合核心（参考 LangChain EnsembleRetriever） ==================== #
def weighted_reciprocal_rank(
    result_lists: List[List[Tuple[str, float]]],
    weights: Optional[List[float]] = None,
    c: int = 60,
) -> List[Tuple[str, float]]:
    """
    加权 RRF (Weighted Reciprocal Rank Fusion)
    参考：LangChain EnsembleRetriever.weighted_reciprocal_rank

    公式：score(d) = Σ weight_i / (c + rank_i(d))

    与朴素 RRF 的区别：
    ──────────────────────────────────────────────────
    朴素 RRF       : score = Σ 1 / (k + rank)
    加权 RRF (原版) : score = Σ weight / (c + rank)
    ──────────────────────────────────────────────────

    参数：
        result_lists:  多路召回结果，每路为 [(doc_id, raw_score), ...]
                       注意：raw_score 在 RRF 中被忽略，只用到排序位置
                       假设结果分别为[语义结果(doc_id, raw_score), BM25结果, 图结果]
        weights:       每一路的权重，None 时默认等权 1.0
                       例：[0.6, 0.3, 0.1] 表示向量:BM25:图 = 6:3:1
        c:             平滑常数（LangChain 用 c 命名，默认 60）
                       c 越大，名次差异对分数的影响越平缓
    返回：
        [(doc_id, fused_score), ...] 按 fused_score 降序
    """
    # 权重校验
    num_lists = len(result_lists)
    if weights is None:
        weights = [1.0] * num_lists
    if len(weights) != num_lists:
        raise ValueError(
            f"weights 长度 ({len(weights)}) 必须与 result_lists 长度 ({num_lists}) 一致"
        )

    # 融合分数累加器，累计每个 doc 的融合分数
    fused_scores: dict[str, float] = {}
    # 记录每个 doc 出现于哪些召回源（调试/可解释性用）
    doc_sources: dict[str, List[str]] = {}
    #
    for list_idx, (ranked_list, weight) in enumerate(zip(result_lists, weights)):
        # 为每一路召回源生成标签（语义/BM25/图）
        source_tag = ["semantic", "bm25", "graph", f"custom_{list_idx}"]
        tag = source_tag[list_idx] if list_idx < len(source_tag) else f"custom_{list_idx}"
        # 超过已有的路径，则设置为自定义标签（custom_{list_idx}）

        for rank, (doc_id, _raw_score) in enumerate(ranked_list, start=1):
            # 核心公式：weight / (c + rank)
            contribution = weight / (c + rank)
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + contribution

            # 记录来源（便于后续调试日志）
            doc_sources.setdefault(doc_id, []).append(tag)

    # 按融合分数降序排列
    sorted_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

    # 可选：打印 Top3 结果的来源分布（debug 时打开）
    # for doc_id, score in sorted_results[:3]:
    #     logging.debug(f"  {doc_id}: {score:.6f} <- {doc_sources[doc_id]}")

    return sorted_results



# Qdrant向量数据库服务

class ChunkDocument:
    """
    自定义文档块类（增强版：支持税务领域元数据）
    """
    def __init__(self, id: str, content: str, file_id: str, metadata: dict | None = None):
        self.id = id
        self.content = content
        self.file_id = file_id
        # 税务领域元数据（可选，Qdrant payload 会原样存储）
        # 典型字段：tax_type / document_number / issuer / effective_date / law_type / is_current
        self.metadata = metadata or {}


class QdrantService:
    """
    Qdrant 向量数据库服务
    提供向量存储、向量检索、集合功能
    """

    # Qdrant集合名称
    COLLECTION_NAME = "tax_chunks"
    # 距离度量方式：余弦相似度（Cosine Similarity）
    DISTANCE_METRIC = Distance.COSINE


    def __init__(self):
        """
        初始化 Qdrant 客户端
        """
        self.api_key = os.getenv("QDRANT_API_KEY")
        self.url = os.getenv("QDRANT_CLUSTER_URL")

        if not self.api_key or not self.url:
            raise ValueError("缺少 QDRANT_CLUSTER_URL 或 QDRANT_API_KEY 环境变量")

        self.client = QdrantClient(
            url=self.url,
            api_key=self.api_key,
            timeout=30.0,
        )

        # 确保集合存在 or 创建集合
        self._ensure_collection()

    def _ensure_collection(self):
        """
        确保Qdrant中存在指定的集合
        如果集合存在，则跳过创建
        """
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]

        if self.COLLECTION_NAME not in collection_names:
            self.client.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=int(os.getenv("VECTOR_SIZE")),
                    distance=self.DISTANCE_METRIC
                )
            )
            logging.info(f"成功创建集合 {self.COLLECTION_NAME}")
        else:
            logging.info(f"集合 {self.COLLECTION_NAME} 已存在")

        # 为常用过滤字段创建 payload 索引（Qdrant Cloud 按字段过滤必须建索引，幂等操作）
        indexed = self.client.get_collection(self.COLLECTION_NAME).payload_schema or {}
        for field_name in ("file_id", "source_type", "owner_id", "scope", "session_id", "status"):
            if field_name not in indexed:
                self.client.create_payload_index(
                    collection_name=self.COLLECTION_NAME,
                    field_name=field_name,
                    field_schema=PayloadSchemaType.KEYWORD,
                    wait=True,
                )

    def upsert_chunks(self, chunks: list[ChunkDocument], embeddings):
        """
        将文本块及向量更新到Qdrant（增强版：payload 带完整元数据）
        :param chunks: 文本块列表（含 metadata 字段）
        :param embeddings: 向量矩阵，与 chunks 列表对应
        """
        points = []
        for chunk, embedding in zip(chunks, embeddings):
            # payload = 固定字段 + 税务元数据（全部打平到顶层，方便 Qdrant Filter 查询）
            payload = {
                "content": chunk.content,
                "file_id": chunk.file_id,
            }
            payload.update(chunk.metadata or {})   # 税务元数据打平注入
            points.append(
                PointStruct(
                    id=chunk.id,
                    vector=embedding,
                    payload=payload,
                )
            )

        self.client.upsert(
            collection_name=self.COLLECTION_NAME,
            points=points,
        )
        logging.info(f"成功插入 {len(points)} 条向量到集合 {self.COLLECTION_NAME}")

    def search(self, query_vector, limit: int = 5, filters: dict | None = None) -> list:
        """
        查询向量在 Qdrant 中最相似的结果（支持元数据过滤）
        :param query_vector: 查询向量
        :param limit: 返回最相似的 N 条结果
        :param filters: Qdrant Filter dict，例 {"tax_type": "vat", "is_current": true}
        :return: 列表，每项包括 {id, score, payload}
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny

        if isinstance(query_vector, np.ndarray):
            query_vector = query_vector.tolist()

        # 构建 Qdrant Filter（支持单值 MatchValue 和多值 MatchAny）
        qdrant_filter = None
        if filters:
            must_conditions = []
            for key, value in filters.items():
                if key == "__access":
                    owner, session_id = value
                    branches = [
                        Filter(must=[FieldCondition(key="source_type", match=MatchValue(value="default"))]),
                        Filter(must=[FieldCondition(key="scope", match=MatchValue(value="system"))]),
                        Filter(must=[FieldCondition(key="owner_id", match=MatchValue(value=owner)),
                                     FieldCondition(key="scope", match=MatchValue(value="global"))]),
                    ]
                    if session_id:
                        branches.append(Filter(must=[
                            FieldCondition(key="owner_id", match=MatchValue(value=owner)),
                            FieldCondition(key="scope", match=MatchValue(value="session")),
                            FieldCondition(key="session_id", match=MatchValue(value=session_id))]))
                    must_conditions.append(Filter(should=branches))
                    continue
                if value is None:
                    continue
                if isinstance(value, list):
                    must_conditions.append(
                        FieldCondition(key=key, match=MatchAny(any=value))
                    )
                else:
                    must_conditions.append(
                        FieldCondition(key=key, match=MatchValue(value=value))
                    )
            if must_conditions:
                excluded = [FieldCondition(key="status", match=MatchAny(any=["uploading", "deleting", "delete_failed", "upload_failed"]))]
                if blocked_document_ids:
                    excluded.append(FieldCondition(key="file_id", match=MatchAny(any=list(blocked_document_ids))))
                qdrant_filter = Filter(must=must_conditions, must_not=excluded)

        results = self.client.query_points(
            collection_name=self.COLLECTION_NAME,
            query=query_vector,
            limit=limit,
            with_payload=True,
            query_filter=qdrant_filter,  # None 时 Qdrant 忽略
        )

        return [
            {
                "id": point.id,
                "score": point.score,
                "payload": point.payload,
            }
            for point in results.points
        ]

    def delete_chunks_by_file_id(self, file_id: str) -> int:
        """
        按 file_id 删除 Qdrant 中对应的所有向量点
        :param file_id: 入库时关联的文件 id
        :return: 删除的点数量
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        # 先查一下有多少条匹配的（用于日志和返回值）
        count_result = self.client.count(
            collection_name=self.COLLECTION_NAME,
            count_filter=Filter(
                must=[
                    FieldCondition(key="file_id", match=MatchValue(value=file_id))
                ]
            ),
        )
        deleted_count = count_result.count

        # 执行删除
        self.client.delete(
            collection_name=self.COLLECTION_NAME,
            points_selector=Filter(
                must=[
                    FieldCondition(key="file_id", match=MatchValue(value=file_id))
                ]
            ),
            wait=True,
        )
        remaining = self.client.count(
            collection_name=self.COLLECTION_NAME,
            count_filter=Filter(must=[FieldCondition(key="file_id", match=MatchValue(value=file_id))]),
            exact=True,
        ).count
        if remaining:
            raise RuntimeError(f"云端仍有 {remaining} 个分块，删除未完成")

        logging.info(
            f"Qdrant 按 file_id='{file_id}' 删除了 {deleted_count} 条向量"
        )
        return deleted_count

    def delete_all_chunks(self):
        """
        删除整个集合
        """
        self.client.delete_collection(
            collection_name=self.COLLECTION_NAME
        )
        logging.info(f"集合{self.COLLECTION_NAME}已删除")

# ==================== Neo4j 图数据库服务 ==================== #

class Neo4jService:
    """
    Neo4j 图数据库服务

    用途：存储税务领域实体及其关系，补充向量检索在"多跳关联查询"上的不足
    图模型：(chunk:Chunk)-[:CONTAINS]->(entity:Entity)-[:RELATED_TO]->(entity)
    """

    LABEL_CHUNK = "Chunk"
    LABEL_ENTITY = "Entity"
    REL_CONTAINS = "CONTAINS"
    REL_RELATED_TO = "RELATED_TO"

    def __init__(self):
        if not _NEO4J_AVAILABLE:
            self.driver = None
            logging.warning("Neo4j Python 驱动未安装，图数据库服务不可用")
            return

        self.uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = os.getenv("NEO4J_USER", "neo4j")
        self.password = os.getenv("NEO4J_PASSWORD", "neo4j")

        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
            )
            self.driver.verify_connectivity()
            logging.info("Neo4j 图数据库连接成功")
        except Exception as e:
            logging.error(f"Neo4j 连接失败: {e}")
            self.driver = None

    def _extract_entities(self, text: str) -> List[str]:
        """轻量实体抽取：基于分词 + 2字以上词元"""
        tokens = default_preprocessing_func(text)
        entities = list(set(t for t in tokens if len(t) >= 2))
        return entities[:20]
    def upsert_chunks_with_entities(self, chunks: List["ChunkDocument"]):
        """将文本块写入图数据库，抽实体并建立关系"""
        if self.driver is None:
            return

        query_chunk = f"""
        MERGE (c:{self.LABEL_CHUNK} {{id: $chunk_id}})
        SET c.content = $content, c.file_id = $file_id
        WITH c
        UNWIND $entities AS entity_name
        MERGE (e:{self.LABEL_ENTITY} {{name: entity_name}})
        MERGE (c)-[:{self.REL_CONTAINS}]->(e)
        """
        query_related = f"""
        MATCH (e1:{self.LABEL_ENTITY})-[r1:{self.REL_CONTAINS}]->(c:{self.LABEL_CHUNK})<-[r2:{self.REL_CONTAINS}]-(e2:{self.LABEL_ENTITY})
        WHERE e1.name <> e2.name
        MERGE (e1)-[:{self.REL_RELATED_TO}]->(e2)
        """

        try:
            with self.driver.session() as session:
                for chunk in chunks:
                    entities = self._extract_entities(chunk.content)
                    session.run(
                        query_chunk,
                        chunk_id=str(chunk.id),
                        content=chunk.content[:2000],
                        file_id=chunk.file_id,
                        entities=entities,
                    )
                session.run(query_related)
            logging.info(f"Neo4j 写入完成：{len(chunks)} 个 Chunk 节点")
        except Exception as e:
            logging.error(f"Neo4j 写入失败: {e}")

    def search(self, query: str, limit: int = 5) -> List[Tuple[str, float]]:
        """
        图数据库检索：匹配查询相关的实体，返回关联最多的 Chunk
        """
        if self.driver is None:
            return []

        query_terms = default_preprocessing_func(query)
        if not query_terms:
            return []

        cypher = f"""
        UNWIND $terms AS term
        MATCH (e:{self.LABEL_ENTITY})-[*0..1]-(related:Entity)
        WHERE related.name CONTAINS term OR term CONTAINS related.name
        MATCH (c:{self.LABEL_CHUNK})-[:{self.REL_CONTAINS}]->(related)
        WITH c, count(DISTINCT related) AS match_count
        ORDER BY match_count DESC
        LIMIT $limit
        RETURN c.id AS chunk_id, match_count AS score
        """

        try:
            with self.driver.session() as session:
                result = session.run(cypher, terms=query_terms, limit=limit * 2)
                return [(record["chunk_id"], float(record["score"])) for record in result]
        except Exception as e:
            logging.warning(f"Neo4j 图检索失败: {e}")
            return []

    def close(self):
        if self.driver is not None:
            self.driver.close()
            logging.info("Neo4j 驱动已关闭")

# ==================== BM25 词法检索服务 ==================== #
# ==================== BM25 词法检索服务（参考 LangChain BM25Retriever） ==================== #

class BM25Service:
    """
    BM25 词法检索服务

    参考：langchain_community.retrievers.bm25.BM25Retriever
    ──────────────────────────────────────────────────────────────
    与 LangChain 原版对齐的设计：
      1. preprocess_func: 可注入的分词器（默认 default_preprocessing_func）
      2. from_texts()   : 工厂方法，一步构建索引
      3. add_chunks()   : 支持增量添加文档（LangChain 的 add_documents）
      4. get_top_n()    : 直接返回 Top N（LangChain 的 _get_relevant_documents）
      5. _k / _scorer  : BM25 超参数可配置（rank_bm25 的 BM25Okapi 参数）
    ──────────────────────────────────────────────────────────────

    与 LangChain 的差异：
      - LangChain 存 Document 对象，我们存 ChunkDocument（业务适配）
      - LangChain 检索返回 Document List，我们返回 [(chunk_id, score)]（为了 RRF 融合）
    """

    def __init__(
        self,
        preprocess_func=None,
        k: int = 5,
        bm25_scorer: str = "okapi",
        **bm25_kwargs,
    ):
        """
        :param preprocess_func: 分词器函数，签名 func(text: str) -> List[str]
                                None 则使用 default_preprocessing_func（中英双语增强版）
                                LangChain 默认也是 preprocess_func=None → text.split()
        :param k:              默认返回的 Top N 数量
        :param bm25_scorer:    rank_bm25 的评分器，"okapi" | "bm25l" | "bm25plus"
        :param bm25_kwargs:    传递给 BM25Okapi 的额外参数
                                如 k1, b, epsilon 等（参考 rank_bm25 文档）
        """
        if not _BM25_AVAILABLE:
            logging.warning("BM25Service: rank-bm25 未安装，无法使用")
            return

        # ── 对齐 LangChain: preprocess_func 可配置 ──
        self.preprocess_func = preprocess_func or default_preprocessing_func
        self._k = k                          # LangChain 的 self.k。返回的文本个数
        self._scorer = bm25_scorer           # LangChain 的 vectorizer 类型。评分算法
        self._bm25_kwargs = bm25_kwargs      # rank_bm25 超参数透传

        # ── 内部状态（对应 LangChain 的 self.docs / self.vectorizer） ──
        self.bm25: Optional[BM25Okapi] = None
        self.chunks: List["ChunkDocument"] = []   # LangChain 的 self.docs（存完整对象）
        self._corpus_tokens: List[List[str]] = []

    # ==================== 构建/增量 ==================== #
    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        ids: Optional[List[str]] = None,
        preprocess_func=None,
        k: int = 5,
        **kwargs,
    ) -> "BM25Service":
        """
        工厂方法（对应 LangChain BM25Retriever.from_texts / from_documents）

        LangChain 原版：
            BM25Retriever.from_documents(docs, preprocess_func=None, k=4)

        这里封装得更方便，直接传文本列表即可：
            bm25 = BM25Service.from_texts(
                texts=[chunk.content for chunk in chunks],
                ids=[str(chunk.id) for chunk in chunks],
                k=5,
            )
        """
        service = cls(preprocess_func=preprocess_func, k=k, **kwargs)
        if texts:
            # 构造 ChunkDocument 列表（LangChain 用 Document）
            if ids is None:
                ids = [str(i) for i in range(len(texts))]
            fake_chunks = [
                ChunkDocument(id=cid, content=text, file_id="bm25_batch")
                for cid, text in zip(ids, texts)
            ]
            service._build_index(fake_chunks) # 核心方法
        return service

    def _build_index(self, chunks: List["ChunkDocument"]):
        """
        构建 BM25 索引（对应 LangChain 的初始化阶段）

        LangChain 内部流程：
            corpus = [preprocess_func(d.page_content) for d in docs]
            vectorizer = BM25Okapi(corpus)
        """
        if not chunks:
            return

        self.chunks = chunks                              # LangChain: self.docs = docs
        self._corpus_tokens = [
            self.preprocess_func(c.content)               # LangChain: preprocess_func
            for c in chunks
        ]

        # 根据 scorer 选择 rank_bm25 的类（LangChain 只支持 okapi，我们额外支持 bm25l/bm25plus）
        scorer_map = {
            "okapi": BM25Okapi,
        }
        try:
            from rank_bm25 import BM25L, BM25Plus
            scorer_map["bm25l"] = BM25L
            scorer_map["bm25plus"] = BM25Plus
        except ImportError:
            pass

        BM25Class = scorer_map.get(self._scorer, BM25Okapi)
        self.bm25 = BM25Class(self._corpus_tokens, **self._bm25_kwargs)
        logging.info(
            f"BM25 索引构建完成：{len(chunks)} 个文档块 "
            f"(scorer={self._scorer}, preprocess={self.preprocess_func.__name__})"
        )

    def add_chunks(self, new_chunks: List["ChunkDocument"]):
        """
        增量添加文档（对应 LangChain BM25Retriever.add_documents）

        注意：rank_bm25 的 BM25Okapi 不支持原地增量，所以这里是全量重建。
        如果你的场景需要频繁增量，建议换 Elasticsearch / OpenSearch。
        """
        if not new_chunks:
            return
        all_chunks = self.chunks + new_chunks
        self._build_index(all_chunks)
        logging.info(f"BM25 增量添加 {len(new_chunks)} 个文档，总计 {len(self.chunks)}")

    # ==================== 检索 ==================== #
    def search(self, query: str, limit: Optional[int] = None) -> List[Tuple[str, float]]:
        """
        BM25 检索（对应 LangChain BM25Retriever._get_relevant_documents）

        LangChain 原版流程：
            query_tokens = preprocess_func(query)
            docs_and_scores = vectorizer.get_scores_and_top_n(query_tokens, docs, k)
            return docs_and_scores

        :param query: 用户查询文本
        :param limit: 返回数量，None 则用初始化时的 self._k
        :return: [(chunk_id, score), ...]
        """
        if self.bm25 is None:
            return []

        n = limit or self._k                              # LangChain: self.k
        query_tokens = self.preprocess_func(query)        # LangChain: preprocess_func(query)
        if not query_tokens:
            return []

        # ── 获取所有分数 ──
        scores = self.bm25.get_scores(query_tokens)       # rank_bm25 API

        # ── 取 Top N（LangChain 用 heapq.nlargest，这里复用 np.argsort） ──
        top_indices = np.argsort(scores)[::-1][:n]

        results: List[Tuple[str, float]] = []
        for idx in top_indices:
            if scores[idx] <= 0:
                break
            results.append((str(self.chunks[idx].id), float(scores[idx])))

        return results

    def get_top_n(self, query: str, n: Optional[int] = None) -> List[Tuple[str, float]]:
        """
        get_top_n 别名（LangChain BM25Retriever 的 API 命名风格）

        LangChain: retriever._get_relevant_documents(query) → List[Document]
        这里：     bm25.get_top_n(query) → List[Tuple[str, float]]

        两种叫法等价，统一用 search() 即可。
        """
        return self.search(query, limit=n)


# store 数据存储对象，新增的修改内容
class PersistentDocumentStore:
    """
    全局共享的文档分块存储。

    使用类级别变量 `_shared_chunks`，确保在任何地方（Agent / Service / 路由）
    调用 PersistentDocumentStore() 得到的都是同一份数据，避免多实例互不
    可见的问题。

    分块数据会序列化到 JSON 文件中：
        - 首次实例化时自动从文件加载已有 chunks
        - add_chunks() 后自动同步写入文件
        - clear() 同时清空内存和磁盘文件
        - 进程重启后自动恢复，真正实现持久化
    """

    # —— 类级别属性：所有实例共享 ——
    _shared_chunks: list = []
    _loaded: bool = False
    _file_path: Path | None = None
    _lock = threading.Lock()

    # 默认持久化文件路径：<项目根>/data/document_store.json
    _DEFAULT_FILE = Path(__file__).resolve().parent.parent / "data" / "document_store.json"

    def __init__(self, file_path: str | Path | None = None):
        # 文件路径只在首次实例化时设置（类级别共享）
        if PersistentDocumentStore._file_path is None:
            PersistentDocumentStore._file_path = (
                Path(file_path) if file_path else self._DEFAULT_FILE
            )
        # 确保目录存在
        PersistentDocumentStore._file_path.parent.mkdir(parents=True, exist_ok=True)
        # 从文件加载（仅首次实例化时执行）
        if not PersistentDocumentStore._loaded:
            self._load_from_file()
            PersistentDocumentStore._loaded = True

    def _load_from_file(self):
        """从 JSON 文件恢复 chunks 到内存"""
        path = PersistentDocumentStore._file_path
        if path is None or not path.exists():
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            loaded = []
            for item in data:
                loaded.append(ChunkDocument(
                    id=item.get("id", ""),
                    content=item.get("content", ""),
                    file_id=item.get("file_id", ""),
                    metadata=item.get("metadata", {}),
                ))
            PersistentDocumentStore._shared_chunks = loaded
            logging.info(f"从文件加载了 {len(loaded)} 个文档块: {path}")
        except Exception as e:
            logging.error(f"加载文档块文件失败，保留原文件: {type(e).__name__}")
            raise

    def _save_to_file(self):
        """将内存中的 chunks 序列化写入 JSON 文件"""
        path = PersistentDocumentStore._file_path
        if path is None:
            return
        try:
            data = []
            for chunk in PersistentDocumentStore._shared_chunks:
                data.append({
                    "id": chunk.id,
                    "content": chunk.content,
                    "file_id": chunk.file_id,
                    "metadata": chunk.metadata,
                })
            temporary = path.with_suffix(".json.tmp")
            with open(temporary, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(temporary, path)
        except Exception as e:
            logging.error(f"保存文档块到文件失败: {e}")
            raise

    def add_chunks(self, chunks):
        with PersistentDocumentStore._lock:
            before = len(PersistentDocumentStore._shared_chunks)
            PersistentDocumentStore._shared_chunks.extend(chunks)
            try:
                self._save_to_file()
            except Exception:
                PersistentDocumentStore._shared_chunks[before:] = []
                raise

    def get_chunks(self) -> list:
        return PersistentDocumentStore._shared_chunks

    def get_chunk_by_id(self, chunk_id):
        for chunk in PersistentDocumentStore._shared_chunks:
            if chunk_id == chunk.id:
                return chunk
        return None

    @classmethod
    def clear(cls):
        """清空所有分块（内存 + 磁盘文件），用于测试或重新入库场景"""
        with cls._lock:
            cls._shared_chunks[:] = []
            cls._loaded = False
            if cls._file_path and cls._file_path.exists():
                try:
                    cls._file_path.unlink()
                except Exception as e:
                    logging.warning(f"删除文档块文件失败: {e}")

    @classmethod
    def remove_by_file_id(cls, file_id: str) -> int:
        """按 file_id 移除所有关联分块（内存 + 同步到文件）

        Returns:
            实际移除的 chunk 数量
        """
        with cls._lock:
            before = len(cls._shared_chunks)
            previous = list(cls._shared_chunks)
            cls._shared_chunks[:] = [
                c for c in cls._shared_chunks if c.file_id != file_id
            ]
            removed = before - len(cls._shared_chunks)
            if removed > 0:
                # 立即同步到磁盘文件
                if cls._file_path is not None:
                    try:
                        data = []
                        for chunk in cls._shared_chunks:
                            data.append({
                                "id": chunk.id,
                                "content": chunk.content,
                                "file_id": chunk.file_id,
                                "metadata": chunk.metadata,
                            })
                        temporary = cls._file_path.with_suffix(".json.tmp")
                        with open(temporary, "w", encoding="utf-8") as f:
                            json.dump(data, f, ensure_ascii=False, indent=2)
                        os.replace(temporary, cls._file_path)
                    except Exception as e:
                        cls._shared_chunks[:] = previous
                        logging.error(f"移除分块后保存文件失败: {e}")
                        raise
            return removed

    @classmethod
    def count(cls) -> int:
        """返回当前分块总数"""
        return len(cls._shared_chunks)

# EmbeddingService
class EmbeddingService:
    """
    向量化服务 + 多路召回融合

    召回源：
        1. Qdrant 向量语义检索（必选）
        2. BM25 词法检索（可选）
        3. Neo4j 图数据库检索（可选）
    融合策略：RRF (Reciprocal Rank Fusion)
    """

    # RRF 融合配置（参考 LangChain EnsembleRetriever）
    RRF_C = 60  # 平滑常数 c
    RRF_WEIGHTS = [0.6, 0.3, 0.1]  # 三路默认权重：向量:BM25:图 = 6:3:1

    # 可按业务调优：
    #   精准匹配场景 → BM25 权重调高，如 [0.4, 0.5, 0.1]
    #   模糊语义场景 → 向量权重调高，如 [0.8, 0.15, 0.05]

    def __init__(
        self,
        store,
        use_qdrant: bool = True,
        use_bm25: bool = True,
        use_neo4j: bool = False,
        skip_rebuild: bool = False,
    ):
        """
        :param store:        数据存储对象，需有 get_chunks() 方法
        :param use_qdrant:   是否使用 Qdrant 向量存储
        :param use_bm25:     是否启用 BM25 词法检索
        :param use_neo4j:    是否启用 Neo4j 图数据库检索
        :param skip_rebuild: 跳过索引重建（Agent 复用已有 Qdrant 数据时设为 True）
        """
        self.store = store
        self.embedding_model = self._load_embedding_model()
        self.embedding_matrix = None
        self._index_chunks: list = []
        self._qdrant_payload_cache: dict = {}

        # ---------- Qdrant ----------
        self.use_qdrant = use_qdrant
        self.qdrant: Optional[QdrantService] = None
        if use_qdrant:
            try:
                self.qdrant = QdrantService()
                logging.info("Qdrant 服务已初始化")
            except Exception as e:
                logging.error(f"Qdrant 服务初始化失败: {e}")
                self.qdrant = None

        # ---------- BM25 ----------
        self.use_bm25 = use_bm25 and _BM25_AVAILABLE and _JIEBA_AVAILABLE
        self.bm25: Optional[BM25Service] = None
        if self.use_bm25:
            self.bm25 = BM25Service()
            if not skip_rebuild:
                logging.info("BM25 词法服务已启用")
        else:
            if not skip_rebuild:
                logging.info("BM25 词法服务未启用（缺少依赖或配置）")

        # ---------- Neo4j ----------
        self.use_neo4j = use_neo4j
        self.neo4j: Optional[Neo4jService] = None
        if use_neo4j:
            try:
                self.neo4j = Neo4jService()
                if self.neo4j.driver is not None and not skip_rebuild:
                    logging.info("Neo4j 图数据库服务已启用")
                elif self.neo4j.driver is None:
                    self.use_neo4j = False
            except Exception as e:
                logging.warning(f"Neo4j 服务初始化失败，已禁用图检索: {e}")
                self.use_neo4j = False

        # ---------- 构建所有索引（Agent 检索模式下跳过，Qdrant 已有数据） ----------
        if not skip_rebuild:
            self.rebuild()
        else:
            logging.info("EmbeddingService 跳过索引重建（复用 Qdrant 已有数据）")

    def _load_embedding_model(self):
        """
        加载向量化模型名称（优先 ConfigManager 自定义，否则 .env 默认）
        """
        return ConfigManager().get_embedding_config()["model"] or os.getenv("EMBEDDING_MODEL_ID")
    # todo：加载模型时，需要检查env文件中写入了ID和URL

    def _get_client(self):
        """
        获取OpenAI客户端（优先 ConfigManager 自定义，否则 .env 默认）
        """
        emb_cfg = ConfigManager().get_embedding_config()
        return OpenAI(
            api_key=emb_cfg["api_key"] or os.getenv("EMBEDDING_API_KEY"),
            base_url=emb_cfg["base_url"] or os.getenv("EMBEDDING_BASE_URL"),
            timeout=15.0,
            max_retries=1,
        )

    # Embedding 模型最大输入长度保护
    # BAAI/bge-large-zh-v1.5 最大 512 tokens，约 1000 中文字，截断到 1500 字符留安全余量
    _MAX_TEXT_LEN = 1500

    def _embed_texts(self,texts:list[str])->np.ndarray:
        """
        对文本列表进行向量化处理（严格保持输入输出行对齐）。

        :param texts: 文本列表
        :return: 向量矩阵，行数与 len(texts) 严格一致；
                 空文本用全零向量占位，保证调用方按索引一一对应，
                 避免"过滤空文本后向量与文本错位"的静默 bug。
        """
        client = self._get_client()
        model_name = self.embedding_model or "BAAI/bge-m3"

        try:
            return self._embed_texts_with_cache(client, model_name, texts)
        except (KeyError, TypeError, ValueError):
            logging.warning("_embed_texts: 缓存逻辑异常，降级为无缓存调用", exc_info=True)
            return self._embed_texts_no_cache(client, model_name, texts)

    def _embed_texts_no_cache(
        self, client, model_name: str, texts: list[str]
    ) -> np.ndarray:
        """原始无缓存向量化（兜底路径）。"""
        non_empty_idx = [i for i, t in enumerate(texts) if t and t.strip()]
        if not non_empty_idx:
            logging.warning("_embed_texts: 所有文本均为空，返回全零矩阵")
            return np.zeros((len(texts), 0))
        safe_texts = [texts[i][:self._MAX_TEXT_LEN] for i in non_empty_idx]
        response = client.embeddings.create(model=model_name, input=safe_texts)
        dim = len(response.data[0].embedding)
        out = np.zeros((len(texts), dim))
        for j, i in enumerate(non_empty_idx):
            out[i] = response.data[j].embedding
        return out

    def _embed_texts_with_cache(
        self, client, model_name: str, texts: list[str]
    ) -> np.ndarray:
        """带缓存的向量化（主路径）。"""
        missing_indices = []
        missing_texts = []
        cached_vectors: dict = {}
        for i in range(len(texts)):
            t = texts[i]
            if not t or not t.strip():
                continue
            key = t[:self._MAX_TEXT_LEN]
            cache_key = (current_tenant_id(), model_name, str(client.base_url), key)
            cached = _embed_cache.get(cache_key)
            if cached is not None:
                cached_vectors[i] = cached
            else:
                missing_indices.append(i)
                missing_texts.append(key)

        if not missing_indices and not cached_vectors:
            logging.warning("_embed_texts: 所有文本均为空，返回全零矩阵")
            return np.zeros((len(texts), 0))

        response = None
        if missing_texts:
            response = client.embeddings.create(
                model=model_name,
                input=missing_texts,
            )

        dim = 0
        if missing_texts and response is not None and response.data:
            dim = len(response.data[0].embedding)
        elif cached_vectors:
            dim = len(next(iter(cached_vectors.values())))
        out = np.zeros((len(texts), dim))
        if response is not None:
            for j, i in enumerate(missing_indices):
                vec = response.data[j].embedding
                out[i] = vec
                _embed_cache.set((current_tenant_id(), model_name, str(client.base_url), missing_texts[j]), vec)
        for i, vec in cached_vectors.items():
            out[i] = vec
        return out

    # ==================== 索引构建 ==================== #
    def rebuild(self):
        """重构所有索引（向量 + BM25 + 图数据库）"""
        chunks = self.store.get_chunks()
        # 过滤空文本块：空块无检索价值，且零向量会导致 Qdrant 余弦相似度未定义
        valid_chunks = [c for c in chunks if c.content and c.content.strip()]
        if not valid_chunks:
            logging.warning("store 中没有有效文本块，无法构建索引")
            self._index_chunks = []
            self.embedding_matrix = None
            return

        self._index_chunks = valid_chunks
        texts = [chunk.content for chunk in valid_chunks]

        # ---------- 1. 向量索引 ----------
        try:
            embeddings = self._embed_texts(texts)

            if self.use_qdrant and self.qdrant:
                self.qdrant.upsert_chunks(valid_chunks, embeddings)
                logging.info(f" Qdrant 向量索引构建成功：{len(valid_chunks)} 条")
            else:
                self.embedding_matrix = embeddings
                logging.info(f" 本地向量矩阵构建成功：{len(valid_chunks)} 条")
        except Exception as e:
            logging.error(f" 向量索引构建失败: {e}")
            self.embedding_matrix = None

        # ---------- 2. BM25 词法索引 ----------
        if self.use_bm25 and self.bm25:
            try:
                self.bm25._build_index(valid_chunks)
            except Exception as e:
                logging.error(f" BM25 索引构建失败: {e}")

        # ---------- 3. Neo4j 图索引 ----------
        if self.use_neo4j and self.neo4j:
            try:
                self.neo4j.upsert_chunks_with_entities(valid_chunks)
            except Exception as e:
                logging.error(f" Neo4j 图索引构建失败: {e}")


    # ==================== 增量入库（仅处理新增 chunk，避免全库重算） ==================== #
    def add_chunks(self, chunks: list) -> None:
        """增量入库：只对新 chunk 做向量化 + 写入索引，不重算全库。

        与 rebuild() 的区别：rebuild() 会对 store 里全部 chunk 重新 embedding，
        适合首次初始化；本方法只处理本次新增的 chunk，用于上传新文档 / 联网检索入库。
        """
        valid = [c for c in chunks if c.content and c.content.strip()]
        if not valid:
            return
        texts = [c.content for c in valid]

        # 1. 向量索引：只向量化新增块
        try:
            if self.use_qdrant and self.qdrant is None:
                raise RuntimeError("Qdrant 未连接，文件尚未完成云端入库")
            embeddings = self._embed_texts(texts)
            if self.use_qdrant and self.qdrant:
                self.qdrant.upsert_chunks(valid, embeddings)
            else:
                if self.embedding_matrix is None:
                    self.embedding_matrix = embeddings
                    self._index_chunks = valid
                else:
                    self.embedding_matrix = np.vstack([self.embedding_matrix, embeddings])
                    self._index_chunks = self._index_chunks + valid
        except Exception as e:
            logging.error(f"增量向量化失败: {e}")
            raise

        # 2. BM25 词法索引（内部全量重建，但不涉及 embedding，代价低）
        if self.use_bm25 and self.bm25:
            try:
                self.bm25.add_chunks(valid)
            except Exception as e:
                logging.error(f"BM25 增量失败: {e}")

        # 3. Neo4j 图索引增量
        if self.use_neo4j and self.neo4j:
            try:
                self.neo4j.upsert_chunks_with_entities(valid)
            except Exception as e:
                logging.error(f"Neo4j 增量失败: {e}")

    # ==================== 单路检索方法 ==================== #

    def _semantic_search(
        self, query: str, limit: int, filters: dict | None = None
    ) -> List[Tuple[str, float]]:
        """语义检索：向量相似度匹配（支持透传 Qdrant 元数据过滤）"""
        if self.embedding_model is None:
            return []
        if not query or not query.strip():
            return []
        try:
            query_vec = self._embed_texts([query])[0]

            if self.use_qdrant and self.qdrant:
                results = self.qdrant.search(query_vec, limit=limit, filters=filters)
                # 把 payload 缓存下来，供 search() 最终组装时用
                for r in results:
                    self._qdrant_payload_cache[r["id"]] = r.get("payload", {})
                return [(result["id"], result["score"]) for result in results]

            if self.embedding_matrix is None:
                return []
            scores = self.embedding_matrix @ query_vec.T
            top_k = np.argsort(scores)[::-1][:limit]
            # 本地矩阵模式下，用 _index_chunks 按行号取真实块 id
            # （不能用 str(idx) 当 id 查：矩阵行号是索引，不是 chunk.id）
            chunks = self._index_chunks
            return [
                (chunks[idx].id, float(scores[idx]))
                for idx in top_k
                if idx < len(chunks) and scores[idx] > 0
            ]
        except Exception as e:
            logging.warning(f"语义检索失效: {e}")
            return []

    def _bm25_search(self, query: str, limit: int) -> List[Tuple[str, float]]:
        """BM25 词法检索包装"""
        if not (self.use_bm25 and self.bm25):
            return []
        return self.bm25.search(query, limit=limit)

    def _graph_search(self, query: str, limit: int) -> List[Tuple[str, float]]:
        """Neo4j 图检索包装"""
        if not (self.use_neo4j and self.neo4j):
            return []
        return self.neo4j.search(query, limit=limit)

    # ==================== RRF 融合检索（LangChain EnsembleRetriever 风格） ==================== #
    def _rrf_fused_search(
        self,
        query: str,
        limit: int,
        weights: Optional[List[float]] = None,
    ) -> List[Tuple[str, float]]:
        """
        多路召回 + 加权 RRF 融合
        参考：LangChain EnsembleRetriever._get_combined_documents

        流程：
            1. 每一路召回 limit * 2 个候选（给融合更多选择空间）
            2. 按顺序收集三路结果：[semantic, bm25, graph]
            3. 调用 weighted_reciprocal_rank 做加权融合
            4. 去重（按 doc_id，LangChain 是按 id_key/page_content 去重）
            5. 截断到 limit

        :param weights: 自定义权重，None 则用类属性 RRF_WEIGHTS
        """
        recall_size = limit * 2

        # ---------- 1. 三路召回 ----------
        semantic_results = self._semantic_search(query, limit=recall_size)
        bm25_results = self._bm25_search(query, limit=recall_size)
        graph_results = self._graph_search(query, limit=recall_size)

        # ---------- 2. 按固定顺序组装（顺序决定权重对应关系） ----------
        # 顺序：[semantic, bm25, graph] → 对应 weights 里的 [w1, w2, w3]
        recall_lists: List[List[Tuple[str, float]]] = []
        effective_weights: List[float] = []

        source_map = [
            ("semantic", semantic_results),
            ("bm25",     bm25_results),
            ("graph",    graph_results),
        ]

        default_weights = weights or self.RRF_WEIGHTS

        for i, (tag, results) in enumerate(source_map):
            if results:
                recall_lists.append(results)
                # 如果权重列表足够长，取对应权重；否则默认 1.0
                w = default_weights[i] if i < len(default_weights) else 1.0
                effective_weights.append(w)

        if not recall_lists:
            return []

        # ---------- 3. 日志（LangChain 会打印每路贡献） ----------
        src_counts = {
            tag: len(results) for tag, results in source_map if results
        }
        logging.info(f"RRF 多路召回候选: {src_counts}, 权重={effective_weights}, c={self.RRF_C}")

        # ---------- 4. 加权 RRF 融合 ----------
        fused = weighted_reciprocal_rank(
            result_lists=recall_lists,
            weights=effective_weights,
            c=self.RRF_C,
        )

        # ---------- 5. 去重 + 截断 ----------
        # LangChain 的 unique_by_key 逻辑：doc_id 即为唯一键，
        # RRF 累加时天然已经合并了相同 doc_id，这里只是防御性去重
        seen_ids = set()
        deduped: List[Tuple[str, float]] = []
        for doc_id, score in fused:
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                deduped.append((doc_id, score))
            if len(deduped) >= limit:
                break

        return deduped

    # ==================== 对外检索接口 ==================== #
    def search(
            self,
            query: str,
            limit: int = 5,
            mode: str = "rrf",
            filters: dict | None = None,
    ) -> List[dict]:
        """
        统一检索入口

        :param query:  用户查询文本
        :param limit:  返回前 N 条结果
        :param mode:   "rrf"(默认) / "semantic" / "bm25" / "graph"
        :param filters: Qdrant 元数据过滤 dict，如 {"tax_type": "vat", "is_current": True}
                       注意：传 filters 时会强制走 semantic 模式（BM25/图检索不支持过滤）
        :return: 列表，每项包含 {'id', 'score', 'content', 'file_id'} 以及 Qdrant payload 中的全部元数据
        """
        # 有 filters 时强制走 semantic（只有 Qdrant 向量检索支持元数据过滤）
        filters = dict(filters or {})
        if current_tenant_id() != "default":
            filters["__access"] = (current_tenant_id(), document_session.get())
        effective_mode = "semantic" if filters else mode

        if effective_mode == "semantic":
            id_score_list = self._semantic_search(query, limit=limit, filters=filters)
        elif effective_mode == "bm25":
            id_score_list = self._bm25_search(query, limit=limit)
        elif effective_mode == "graph":
            id_score_list = self._graph_search(query, limit=limit)
        else:  # "rrf" 或其他未知值
            id_score_list = self._rrf_fused_search(query, limit=limit)

        results: List[dict] = []
        for chunk_id, score in id_score_list:
            # 优先从 Qdrant payload 取（持久化的、进程无关的数据源）
            # 拿不到再 fallback 到内存 store（兼容本地无 Qdrant 的场景）
            payload = self._qdrant_payload_cache.get(chunk_id, {})
            chunk = self.store.get_chunk_by_id(chunk_id)

            content = payload.get("content") or (chunk.content if chunk else "")
            file_id = payload.get("file_id") or (chunk.file_id if chunk else "")

            # 两边都没拿到 content → 跳过这条（确实不应该出现）
            if not content:
                continue

            # 从 Qdrant payload 带出全部元数据（payload 已在入库时打平注入）
            # 保留 content/file_id 在顶层方便直接访问，其他字段原样透出
            record = {
                "id": chunk_id,
                "score": round(score, 6),
                "content": content,
                "file_id": file_id,
            }
            for key, val in payload.items():
                if key not in record:  # 不覆盖顶层已有的字段
                    record[key] = val
            if chunk and not payload:
                record.update(chunk.metadata or {})
            if current_tenant_id() != "default" and not document_visible(record):
                continue
            if file_id in blocked_document_ids:
                continue
            results.append(record)
        return results
