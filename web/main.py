"""
FastAPI 应用入口

config: 配置文件
/ # 首页
/api/chat # 对话接口
/api/skills # 技能
/training # 训练接口
/tax # 税务项目接口
/api/agent
/knowledge #添加文本到知识库（RAG）
/api/search # 搜索接口
/api/memory # 长期记忆
...（其他可能尚未考虑的接口）

税务知识库 MVP 基础架构
├── 1. 发票合规与报销规范 (高频实用)
├── 2. 核心三大税种基础查验 (增值税/企税/个税)
└── 3. 税务名词与基础概念词典 (辅助 Agent 理解)

1、发票合规与报销规范（最适合新手起步）：
知识点内容：
发票抬头与税号规则：企业报销发票必须包含哪些要素（抬头、纳税人识别号等）。
发票分类与异常处理：增值税专用发票 vs 普通发票、数纸电发票（电子发票）的保存规范、作废与红字发票流程。
不得抵扣/报销的范围：例如餐饮发票抵扣限制、福利费发票等常见误区。
为什么先建这个：提问格式非常标准（如“餐饮发票能开专票抵扣吗？”），Agent 非常容易做准确的向量检索。

2、三大主税种基础速查（只存“干货”）
增值税（VAT）：
小规模纳税人 vs 一般纳税人认定标准及征收率/税率表（例如 13%、9%、6%、3% 对应的适用行业）。
企业所得税（CIT）：
小型微利企业的优惠税率及判定条件（如年应纳税所得额、从业人数、资产总额限制）。
个人所得税（IIT）：
综合所得税率表（预扣预缴表）、专项附加扣除（子女教育、住房贷款利息、3岁以下婴幼儿照护等）的扣除标准与条件。


3、基础概念和税务术语
知识点内容：
把常见的税务缩写、专有名词做成 词典/Q&A。例如：什么是“进项税”、“销项税”、“预扣预缴”、“汇算清缴”。
为什么需要这个：当用户提问“什么是进项税”时，Agent 可以直接调用定义，避免去几万字的文件里盲目检索。

todo: templates部分的html文件格式需了解。
todo：更新后的ingest_file和TaxAgent需要理解
"""

# ======日记配置=====
import json
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)

import os
import inspect
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException, status, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.params import Depends
from starlette.responses import HTMLResponse, Response, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, StrictBool
from services.security import bind_tenant, reset_tenant
from services.config_manager import ConfigManager, MODEL_PROVIDERS, validate_api_url, infer_provider


logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
PATHS = {
    "templates": str(Path(__file__).resolve().parent / "templates"),
    "uploads": str(BASE_DIR / "uploads"),
    "data": str(BASE_DIR / "data"),
}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
os.makedirs(PATHS["uploads"], exist_ok=True)
os.makedirs(PATHS["data"], exist_ok=True)

from contextlib import asynccontextmanager

# ===== FastAPI lifespan：管理全局 async 资源的初始化和清理 =====
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ---- startup ----
    logger.info("[lifespan] 正在初始化全局 Agent...")
    from agents.knowledge_agent import KnowledgeAgent
    from agents.tax_agent import TaxAgent
    from agents.base_agent import _get_shared_memory
    from web.main import KnowledgeService

    # ---- 初始化 SQL 业务表（首次启动自动建表，已存在则跳过） ----
    from database.models import create_tables as init_sql_tables
    init_sql_tables()
    logger.info("[lifespan] SQL 业务表已就绪")

    app.state.knowledge_agent = None
    app.state.tax_agent = None
    llm_cfg = ConfigManager().get_llm_config()
    if all(llm_cfg.values()) and not llm_cfg["api_key"].startswith("your_"):
        try:
            app.state.knowledge_agent = KnowledgeAgent()
            app.state.tax_agent = TaxAgent()
            app.state.tax_agent.registry._cache["knowledge_agent"] = app.state.knowledge_agent
        except Exception as e:
            logger.warning("[lifespan] Agent 初始化失败（%s），请在模型配置弹窗检查配置", type(e).__name__)
    else:
        logger.warning("[lifespan] 未配置有效 LLM，页面可用；请打开模型配置弹窗")
    app.state.knowledge_service = KnowledgeService()
    app.state.conversation_memory = _get_shared_memory()

    # ---- 知识库管理器（多知识库 + 权限过滤） ----
    from knowledge.kb_manager import KnowledgeBaseManager
    app.state.kb_manager = KnowledgeBaseManager(work_dir=str(BASE_DIR / "data" / "kb"))

    # —— 预置默认知识库自动入库（knowledge/presets/ 目录下的 .md 文件） ——
    presets_dir = BASE_DIR / "knowledge" / "presets"
    if presets_dir.is_dir():
        preset_files = sorted(presets_dir.glob("*.md"))
        emb_cfg = ConfigManager().get_embedding_config()
        rag_configured = (all(emb_cfg.values()) and not emb_cfg["api_key"].startswith("your_")
                          and bool(ConfigManager().get_qdrant_config()["url"])
                          and "your-cluster" not in ConfigManager().get_qdrant_config()["url"])
        if preset_files and rag_configured:
            logger.info(f"[lifespan] 检测到 {len(preset_files)} 个预置知识文件，开始入库...")
            for pf in preset_files:
                try:
                    result = await app.state.knowledge_service.ingest_preset_file(
                        file_path=str(pf),
                    )
                    logger.info(f"[lifespan] 预置文件入库: {pf.name} → {result.get('message', result)}")
                except Exception as e:
                    logger.warning(f"[lifespan] 预置文件入库失败 {pf.name}: {e}")
        else:
            logger.info("[lifespan] 预置目录为空或 RAG 未配置，跳过自动入库；可配置后上传预置文件")
    else:
        os.makedirs(presets_dir, exist_ok=True)
        logger.info(f"[lifespan] 已创建预置知识目录: {presets_dir}")

    # 初始化政策自动更新器
    from knowledge.policy_updater import (
        PolicyUpdater, LocalDirectorySource, logging_notify_hook, TaxGovCnSource,
    )
    policies_dir = str(BASE_DIR / "data" / "policies")
    os.makedirs(policies_dir, exist_ok=True)

    policy_updater = PolicyUpdater(
        tracker_db=str(BASE_DIR / "data" / "policy_tracker.sqlite3"),
        kb_work_dir=str(BASE_DIR / "data" / "kb"),
        kb_id="tax_policy",
        notify_hooks=[logging_notify_hook],
    )
    policy_updater.add_source(LocalDirectorySource(
        directory=policies_dir,
        source_id="local_policies",
        recursive=True,
        category="税务政策",
    ))
    # 预置官网源；保存的源清单优先（删除后不会在重启时重新出现）。
    chinatax_url = os.getenv("TAX_CN_LISTING_URL", "")
    policy_cfg = ConfigManager().get_policy_config()
    defaults = [
        {"source_type":"tax_gov_cn", "source_id":"chinatax", "category":"国家税务总局法规库",
         "listing_url":chinatax_url or "https://fgk.chinatax.gov.cn/zcfgk/c100027/list.html", "max_pages":1, "max_items":5},
        {"source_type":"tax_gov_cn", "source_id":"mof", "category":"财政部税政司",
         "listing_url":"https://szs.mof.gov.cn/zhengcefabu/", "link_selector":"a[href*='/t20'], a[href^='./20']",
         "title_selector":"meta[name='ArticleTitle'], h2.title_con", "content_selector":".TRS_Editor", "max_pages":1, "max_items":5},
    ]
    if "sources" in policy_cfg:
        policy_updater.remove_source("local_policies")
    for source in policy_cfg.get("sources", defaults):
        policy_updater.add_source(policy_source_from_config(source))
    # 启动后台定时扫描（每 6 小时一次）
    policy_updater.start_scheduler(interval_seconds=policy_cfg.get("interval_hours", 6) * 3600, run_immediately=False)
    app.state.policy_updater = policy_updater

    # 让模块级全局变量也能被旧代码引用（兼容向后）
    import web.main as _this
    _this.knowledge_agent = app.state.knowledge_agent
    _this.tax_agent = app.state.tax_agent
    _this.knowledge_service = app.state.knowledge_service
    _this.conversation_memory = app.state.conversation_memory
    _this.policy_updater = app.state.policy_updater
    _this.kb_manager = app.state.kb_manager

    logger.info("[lifespan] 初始化完成（含长记忆模块 + 政策自动更新）")
    yield

    # ---- shutdown ----
    logger.info("[lifespan] 正在清理资源...")
    # 停止政策更新调度器
    import asyncio
    if hasattr(app.state, "policy_updater") and app.state.policy_updater:
        try:
            await asyncio.to_thread(app.state.policy_updater.stop_scheduler)
            app.state.policy_updater.tracker.close()
        except Exception as e:
            logger.warning("[lifespan] 停止调度器失败（%s）", type(e).__name__)
    # 取消所有残余 async 任务，避免 "event loop is closed"
    import asyncio
    # SSE 请求自身负责取消任务；不取消服务器和其他组件拥有的任务。
    agents = [app.state.knowledge_agent, app.state.tax_agent]
    if app.state.tax_agent is not None:
        agents.extend(app.state.tax_agent.registry._cache.values())
    for agent in list(agents):
        if agent is not None:
            agents.extend(child for name in ("policy_agent", "definition_agent")
                          if (child := vars(agent).get(name)) is not None)
    tasks = []
    for agent in agents:
        if agent is not None and agent.checkpointer is not None:
            await agent.checkpointer.conn.close()
    for t in tasks:
        t.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    logger.info("[lifespan] 清理完成")


app = FastAPI(
    title="税务Agent",
    version="1.0",
    description="实现税务计算、咨询筹划相关的智能体",
    lifespan=lifespan,
)

# todo:加入注入依赖到路由函数中。

# 作用：允许前端页面从任意地址跨域请求本后端 API
# 注意：allow_credentials 与 allow_origins=["*"] 互斥（浏览器会拒绝带凭据的跨域请求），
# 本应用用 Bearer 头鉴权、不依赖 Cookie，故关闭 credentials。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


templates = Jinja2Templates(directory=str(PATHS["templates"]))

def render_page(request: Request, name: str) -> Response:
    """渲染模板页面，兼容不同版本 Starlette 模板 API"""
    parameters = inspect.signature(templates.TemplateResponse).parameters
    if next(iter(parameters), "") == "request":
        return templates.TemplateResponse(request=request, name=name)
    return templates.TemplateResponse(name, {"request": request})

# 核心业务逻辑：知识库服务：
# 这是文件入库的管道：用户上传 → 保存 → 解析 → 分块 → 向量化 → 存储
# 目前仅实现保存，后续需对接 knowledge/ 下的模块

class KnowledgeService:
    """
    知识库文件入库服务【核心】

    数据流：
        上传文件 → _save_file() → QdrantKnowledgeBase.ingest_file()
                  → Parser → Chunking → Embedding → Qdrant + 元数据持久化

    QdrantKnowledgeBase 负责：
        - 解析/分块/向量化/入库
        - 元数据 JSON 持久化（刷新不丢失）
        - 全链路删除
    """

    def __init__(self):
        self.upload_dir = PATHS["uploads"]
        from knowledge.kb import QdrantKnowledgeBase
        self._kb = QdrantKnowledgeBase(work_dir=PATHS["data"])
        self._kb.upload_root = Path(self.upload_dir)

    async def ingest_file(self, file: UploadFile, source_type: str = "user_upload", **metadata) -> dict:
        """
        文件入库管道【核心方法】

        保存 → 委托 QdrantKnowledgeBase（解析→分块→向量化→Qdrant+元数据持久化）

        :param source_type: 知识来源，default / user_upload / web_search
        """
        # 1. 保存文件到本地
        import asyncio
        file_path = await asyncio.to_thread(self._save_file, file)
        try:
            import time

            file_id = f"file_{uuid.uuid4().hex}"
            result = await self._kb.ingest_file(
                file_path=str(file_path),
                file_id=file_id,
                source_type=source_type,
                file_name=file.filename,
                managed_upload=True,
                **metadata,
            )

            if result.get("status") == "success":
                result["message"] = f"文件 {file.filename} 已完成向量化，共 {result.get('chunk_count', 0)} 个分块"
                result["file_name"] = file.filename
                import mimetypes
                result.update(filename=Path((file.filename or "").replace("\\", "/")).name,
                              size=file_path.stat().st_size,
                              mime_type="text/csv" if file_path.suffix.lower() == ".csv" else mimetypes.guess_type(file.filename or "")[0] or file.content_type or "application/octet-stream",
                              url=f"/api/knowledge/files/{file_id}/content")
                record = next(f for f in self._kb._metadata["files"] if f["file_id"] == file_id)
                record.update({key: result[key] for key in ("filename", "size", "mime_type", "url")})
                self._kb.save_metadata()

                # 4. 同步写入 SQL 业务表（KnowledgeFiles + KnowledgeChunks）
                try:
                    from database.repositories import KnowledgeFileRepository, KnowledgeChunkRepository
                    file_repo = KnowledgeFileRepository()
                    chunk_repo = KnowledgeChunkRepository()

                    sql_record = file_repo.create(
                        path=str(file_path),
                        business_id=file_id,
                        file_name=file.filename,
                        chunk_count=result.get("chunk_count", 0),
                    )
                    # 从内存 store 读取刚入库的 chunks 写入 SQL
                    from knowledge.embedding import PersistentDocumentStore
                    chunks_for_sql = [
                        {"id": c.id, "content": c.content}
                        for c in PersistentDocumentStore().get_chunks()
                        if c.file_id == file_id
                    ]
                    if chunks_for_sql:
                        chunk_repo.create_batch(sql_record.id, chunks_for_sql)
                    logger.info(
                        f"[ingest_file] SQL 记录已写入: file_id={file_id}, "
                        f"sql_id={sql_record.id}, chunks={len(chunks_for_sql)}"
                    )
                except Exception as sql_err:
                    logger.error("[ingest_file] SQL 写入失败：%s", type(sql_err).__name__)
                    record = next(f for f in self._kb._metadata["files"] if f["file_id"] == file_id)
                    record["status"] = "upload_failed"
                    from services.security import blocked_document_ids
                    blocked_document_ids.add(file_id)
                    self._kb.save_metadata()
                    raise RuntimeError("业务记录保存失败，文档已隔离，可在文档管理中删除后重新上传") from sql_err

            return result

        except Exception as e:
            return {
                "status": "error",
                "message": f"文件处理失败：{str(e)}",
                "file_path": str(file_path),
            }

    async def list_files(self) -> list[dict]:
        """列出所有已上传文件（从持久化元数据读取，刷新不丢失）"""
        return await self._kb.list_files()

    async def refresh_cloud_files(self):
        """管理员显式同步云端清单，包括只有云端分块的历史资料。"""
        import asyncio
        def discover():
            embedding = self._kb._ensure_embedding()
            if embedding.qdrant is None:
                raise RuntimeError("请先配置 Qdrant Cloud 地址、密钥和 Embedding")
            client = embedding.qdrant.client
            offset, grouped = None, {}
            while True:
                points, offset = client.scroll(collection_name=embedding.qdrant.COLLECTION_NAME,
                    limit=256, offset=offset, with_vectors=False,
                    with_payload=["file_id", "file_name", "title", "source_type", "owner_id", "scope", "session_id", "status"])
                for point in points:
                    payload = point.payload or {}
                    fid = payload.get("file_id")
                    if not fid:
                        continue
                    record = grouped.setdefault(fid, {"file_id": fid, "file_name": payload.get("file_name") or payload.get("title") or fid,
                        "file_path": "", "chunk_count": 0, "cloud_only": True, "tax_metadata": payload})
                    record["chunk_count"] += 1
                if offset is None:
                    return grouped
        grouped = await asyncio.to_thread(discover)
        existing = {f["file_id"]: f for f in self._kb._metadata.get("files", [])}
        for fid, record in grouped.items():
            existing.setdefault(fid, record)
        self._kb._metadata["files"] = list(existing.values())
        self._kb.save_metadata()
        return len(grouped)

    async def read_cloud_chunks(self, file_id):
        import asyncio
        def read():
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            embedding = self._kb._ensure_embedding()
            if embedding.qdrant is None:
                raise RuntimeError("Qdrant Cloud 未连接")
            points, _ = embedding.qdrant.client.scroll(collection_name=embedding.qdrant.COLLECTION_NAME,
                scroll_filter=Filter(must=[FieldCondition(key="file_id", match=MatchValue(value=file_id))]),
                limit=256, with_vectors=False, with_payload=["content"])
            return [(p.payload or {}).get("content", "") for p in points]
        return await asyncio.to_thread(read)

    async def list_files_from_sql(self) -> list[dict]:
        """从 SQL 业务表读取已上传文件列表（含分块数）"""
        try:
            from database.repositories import KnowledgeFileRepository, KnowledgeChunkRepository
            file_repo = KnowledgeFileRepository()
            chunk_repo = KnowledgeChunkRepository()
            records = file_repo.get_all()
            result = []
            for r in records:
                actual_chunks = chunk_repo.count_by_file_id(r.id) if r.id else r.chunk_count
                result.append({
                    "sql_id": r.id,
                    "business_id": r.business_id,
                    "file_name": r.file_name,
                    "path": r.path,
                    "chunk_count": actual_chunks,
                    "create_at": r.create_at.isoformat() if r.create_at else None,
                })
            return result
        except Exception as e:
            logger.warning(f"[list_files_from_sql] 查询失败: {e}")
            return []

    async def delete_file(self, file_id: str) -> dict:
        """删除指定文件：元数据 + Qdrant 向量 + 内存 store + SQL 表 + 磁盘原文件"""
        record = next((f for f in await self._kb.list_files() if f.get("file_id") == file_id), {})
        # 同步清理 SQL 业务表
        def finalize_sql():
            from database.repositories import KnowledgeFileRepository
            repo = KnowledgeFileRepository()
            sql_deleted = repo.delete_by_business_id(file_id)
            # 历史 SQL 行尚无业务 ID，仅按已授权清单中的精确原路径清理关联记录。
            if record.get("file_path"):
                from database.models import KnowledgeFiles, KnowledgeChunks
                from database.repositories import SessionLocal
                with SessionLocal() as session:
                    old_rows = session.query(KnowledgeFiles).filter(
                        KnowledgeFiles.business_id.is_(None), KnowledgeFiles.path == record["file_path"]).all()
                    for row in old_rows:
                        session.query(KnowledgeChunks).filter(KnowledgeChunks.file_id == row.id).delete()
                        session.delete(row)
                    session.commit()
            if sql_deleted:
                logger.info(f"[delete_file] SQL 记录已清理: file_id={file_id}")
        ok = await self._kb.delete_file(file_id, finalize=finalize_sql)

        if ok:
            return {"status": "success", "message": f"文件 {file_id} 已删除", "file_id": file_id}
        else:
            return {"status": "error", "message": f"文件 {file_id} 不存在或删除失败", "file_id": file_id}

    async def ingest_preset_file(self, file_path: str) -> dict:
        """
        预置知识文件入库（启动时自动调用）

        与 ingest_file 的区别：
        - 入参是本地文件路径（非 UploadFile），source_type 固定为 "default"
        - 不保存额外的磁盘副本（预设文件已在 presets/ 目录下）
        - file_id 基于文件名生成（稳定），重启时自动跳过已入库的文件
        """
        from pathlib import Path as _Path

        file_name = _Path(file_path).stem
        file_id = f"preset_{file_name}"
        if file_id in self._kb._metadata.get("disabled_presets", []):
            return {"message": "已由管理员停用，跳过自动入库", "file_id": file_id, "skipped": True}

        # —— 去重：检查该 preset 是否已入库 ——
        if self._kb.is_file_ingested(file_id):
            return {"message": f"[{file_name}] 已入库，跳过", "file_id": file_id, "skipped": True}

        return await self._kb.ingest_file(
            file_path=file_path,
            file_id=file_id,
            source_type="default",
            scope="system",
        )

    def _save_file(self, file: UploadFile) -> Path:
        """
        将上传文件保存到 uploads/ 目录
        文件名加时间戳前缀，防止同名文件冲突
        """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        from knowledge.parser import ParserFactory
        name = Path((file.filename or "").replace("\\", "/")).name
        if not name or not ParserFactory.is_supported(name):
            raise ValueError("请上传支持的文本、PDF、Office 或图片文件")
        safe_name = f"{ts}_{uuid.uuid4().hex}_{name}"
        file_path = Path(self.upload_dir) / safe_name
        content = file.file.read(MAX_UPLOAD_BYTES + 1)
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="文件超过20 MiB上限，请拆分后上传")
        if not content:
            raise ValueError("不能上传空文件")
        with open(file_path, "wb") as f:
            f.write(content)
        return file_path


# 框架代码：Pydantic 请求/响应数据模型
# 作用：自动校验 HTTP 请求体格式，自动生成 OpenAPI 文档
class CurrentUser(BaseModel):
    id:str
    user_name:str
    role:str = "user" # 默认用户


class WorkflowResume(BaseModel):
    id: str = Field(min_length=1, max_length=200)
    value: StrictBool


class TaxQueryRequest(BaseModel):
    user_id:str = Field(..., description="用户ID")
    query: str = Field(..., example="餐饮发票能开专票抵扣吗？",description="税务查询语句")
    session_id: Optional[str] = Field(None, description="会话ID")
    source_types: Optional[List[str]] = Field(None, description="知识来源筛选：default / user_upload / web_search；None 表示不限定")
    web_search: bool = Field(False, description="本轮主动联网检索；检索摘要直接注入，不依赖向量入库")
    uploaded_file_ids: Optional[List[str]] = Field(None, max_length=10, description="本轮对话中用户上传的文件ID列表")
    resume: WorkflowResume | None = None

class ReferenceSource(BaseModel):
    """引用知识库来源"""
    title: str
    content: str

class TaxQueryResponse(BaseModel):
    """Agent 返回给前端的结果"""
    session_id:str
    answer: str = Field(..., description="智能体回答")
    references: List[ReferenceSource] = Field(..., description="引用知识库来源")
    status:str = "success"
    pending: dict | None = None


class PolicySourceRequest(BaseModel):
    """添加政策源的请求体"""
    type: str = Field(..., description="源类型：local 或 http")
    source_id: str | None = Field(None, description="来源标识")
    directory: str | None = Field(None, description="本地目录路径（type=local 时必填）")
    urls: list[str] | None = Field(None, description="URL 列表（type=http 时必填）")
    recursive: bool = Field(True, description="是否递归扫描子目录（仅 local 有效）")
    category: str = Field("", description="政策分类标签")

# 定义依赖函数，新增的修改内容
async def get_current_user(authorization: str = Header(...)):
    """
    从请求头解析 Token → 构造 CurrentUser → 绑定租户上下文 → 请求结束后自动 reset。

    使用 async generator + yield 是为了让 FastAPI 在请求生命周期内自动处理 try/finally，
    确保 bind_tenant() 对应的 reset_tenant() 一定被调用，避免 contextvar 泄漏到下一个请求。
    """
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的认证格式，请使用 Bearer Token",
        )

    raw_token = authorization.split(" ")[1]

    # 演示用 token，支持通过环境变量覆盖（生产环境务必替换为真实鉴权）
    # - AUTH_USER_TOKEN : 普通用户 token
    # - DEVELOPER_TOKEN : 开发者 token（优先），回退到 AUTH_ADMIN_TOKEN
    user_token = os.getenv("AUTH_USER_TOKEN", "test_token_123")
    developer_token = os.getenv("DEVELOPER_TOKEN") or os.getenv("AUTH_ADMIN_TOKEN", "admin_token_456")

    if raw_token == user_token:
        current_user = CurrentUser(id="user_001", user_name="张三", role="user")
    elif raw_token == developer_token:
        current_user = CurrentUser(id="user_000", user_name="开发者", role="admin")
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的凭证，请重新登录",
        )

    # —— 关键：把用户 ID 绑定到租户上下文 ——
    tenant_token = bind_tenant(str(current_user.id))
    try:
        yield current_user
    finally:
        # 请求结束（无论成功/异常）都释放，防止 contextvar 跨请求污染
        reset_tenant(tenant_token)

# 路由处理函数
# 每个路由函数内部只做"参数接收 → 调用核心逻辑 → 返回结果"



# 全局单例：延迟初始化（真正的初始化在 lifespan startup 里完成）
# 这里用 None 占位，确保导入不会触发模块级副作用
knowledge_agent = None
tax_agent = None
knowledge_service = None
conversation_memory = None
policy_updater = None
kb_manager = None

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    """首页 - 对话界面"""
    return render_page(request, "index.html")

@app.get("/knowledge", response_class=HTMLResponse)
def knowledge(request: Request):
    """知识库管理页面"""
    return render_page(request, "knowledge.html")

@app.get("/training", response_class=HTMLResponse)
@app.get("/policy", response_class=HTMLResponse)
def training(request: Request):
    """训练测试页面"""
    return render_page(request, "training.html")

# 新增的修改内容
@app.post("/api/chat")
async def tax_chat(body: TaxQueryRequest, current_user: CurrentUser = Depends(get_current_user)):
    """【同步对话接口】接收税务查询，经 TaxAgent 编排器路由多子 Agent 协作后一次性返回。

    TaxAgent.run() 内部已完成：意图路由 → 多 Agent 调度 → 汇总 → 长记忆注入与记录。
    """
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="请输入税务问题")
    if tax_agent is None:
        raise HTTPException(status_code=503, detail="请先在 ⚙ 模型配置弹窗填写模型、API Key 和 Base URL")
    logger.info("[同步请求] query_length=%d", len(body.query))

    try:
        session_id = body.session_id or f"session_{int(time.time())}"
        if conversation_memory:
            conversation_memory.ensure_session(session_id, body.query)
        answer = await tax_agent.run(
            body.query,
            context={
                "thread_id": session_id,
                "uid": current_user.id,
                "source_types": body.source_types,
                "web_search": body.web_search,
                "uploaded_file_ids": body.uploaded_file_ids,
                "resume": body.resume.model_dump() if body.resume else None,
            },
        )

        pending = await tax_agent.pending_workflow(session_id) if inspect.iscoroutinefunction(getattr(tax_agent, "pending_workflow", None)) else None
        return TaxQueryResponse(
            session_id=session_id,
            answer=answer or "（Agent 未返回有效内容）",
            references=[],
            status="waiting" if pending else "success", pending=pending,
        )

    except Exception as e:
        logger.warning("[同步请求错误] %s", type(e).__name__)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="查询失败，请检查模型配置或网络连接后重试",
        )

# 新增的修改内容
# -------------------- 流式对话接口（SSE 协议） --------------------
@app.post("/api/chat/stream")
async def tax_chat_stream(
    body: TaxQueryRequest,
    current_user: CurrentUser = Depends(get_current_user),
):
    """【SSE 对话接口】接收税务查询，推送处理状态和实时生成的最终正文。

    架构链路：
        浏览器 EventSource
          ↓  (text/event-stream)
        FastAPI StreamingResponse
          ↓  (async generator)
        status / token / done / error 事件
          ↓
        tax_agent.run() ← 全局编排器（正文生成时发出 token，完成后核对最终结果）
          ↓  (LangGraph graph.ainvoke())
        LangGraph Agent + RAG 检索 + LLM 调用
    """
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="请输入税务问题")
    if tax_agent is None:
        raise HTTPException(status_code=503, detail="请先在 ⚙ 模型配置弹窗填写模型、API Key 和 Base URL")
    logger.info("[流式请求] query_length=%d", len(body.query))

    from langchain_core.messages import HumanMessage
    from agents.streaming import agent_to_sse

    session_id = body.session_id or f"session_{int(time.time())}"

    if conversation_memory:
        try:
            conversation_memory.ensure_session(session_id, body.query)
        except Exception as exc:
            logger.error("[流式] 创建会话失败（%s）", type(exc).__name__)
            raise HTTPException(status_code=503, detail="会话保存失败，请检查本地磁盘后重试") from exc

    try:
        input_context = {
            "thread_id": session_id,
            "uid": current_user.id,
            "user_id": current_user.id,
            "user_name": current_user.user_name,
            "source_types": body.source_types,
            "web_search": body.web_search,
            "uploaded_file_ids": body.uploaded_file_ids,
            "resume": body.resume.model_dump() if body.resume else None,
        }

        # 与同步接口共用 TaxAgent 编排；长记忆注入和记录统一由 run 完成。
        # SSE 保留 status / token / done / error 协议，正文随模型生成即时输出。
        import asyncio

        async def sse_generator():
            task = None
            try:
                yield f"event: status\ndata: {json.dumps({'text': ' 正在分析问题并检索相关知识…'}, ensure_ascii=False)}\n\n"
                from agents.streaming import response_events, _sse_line
                queue = asyncio.Queue()
                loop = asyncio.get_running_loop()
                token_count = 0
                streamed_text = ""
                stream_open = True
                stream_started = time.perf_counter()
                def emit(event, data):
                    if stream_open:
                        loop.call_soon_threadsafe(queue.put_nowait, (event, data))
                callback_token = response_events.set(emit)
                try:
                    task = asyncio.create_task(tax_agent.run(body.query, context=input_context))
                finally:
                    response_events.reset(callback_token)
                pending_event = None
                try:
                    while not task.done() or not queue.empty():
                        pending_event = asyncio.create_task(queue.get())
                        ready, _ = await asyncio.wait({task, pending_event}, timeout=10,
                                                      return_when=asyncio.FIRST_COMPLETED)
                        if pending_event in ready:
                            event, data = pending_event.result()
                            if event == "token":
                                streamed_text += data.get("text", "")
                                token_count += 1
                                if token_count == 1:
                                    logger.info("[流式] 首字耗时 %.2fs", time.perf_counter() - stream_started)
                            yield _sse_line(event, data)
                        else:
                            pending_event.cancel()
                            await asyncio.gather(pending_event, return_exceptions=True)
                            if not ready:
                                yield ": keepalive\n\n"
                finally:
                    if pending_event is not None and not pending_event.done():
                        pending_event.cancel()
                        await asyncio.gather(pending_event, return_exceptions=True)
                answer = await task
                if token_count == 0:
                    yield _sse_line("token", {"text": answer or "（未返回内容）"})
                elif streamed_text != answer:
                    # 最终正文以编排结果为准，覆盖工具草稿或补齐未流出的降级正文。
                    yield _sse_line("answer", {"text": answer or "（未返回内容）"})
                logger.info("[流式] 完成耗时 %.2fs", time.perf_counter() - stream_started)
                yield "event: done\ndata: null\n\n"
            except asyncio.CancelledError:
                logger.info("[流式] 客户端断开或服务关闭")
                raise
            except Exception as e:
                logger.warning("[流式] 查询失败（%s）", type(e).__name__)
                message = ("模型不存在或当前 API Key 无权使用；请在设置中重新获取列表，并选择可用的对话模型"
                           if getattr(e, "status_code", None) == 404 else "查询失败，请检查模型配置或网络连接后重试")
                yield f"event: error\ndata: {json.dumps({'msg': message}, ensure_ascii=False)}\n\n"
            finally:
                stream_open = False
                if task is not None and not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)

        return StreamingResponse(
            sse_generator(),
            media_type="text/event-stream",
        )

    except Exception as e:
        err_msg = f"流式对话初始化失败: {str(e)}"
        logger.error(f"[流式请求错误] {err_msg}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=err_msg,
        )

# 新增的修改内容
@app.get("/api/skills")
def skills():
    if knowledge_agent is None:
        raise HTTPException(status_code=503, detail="请先在模型配置弹窗完成配置")
    """获取 Agent 支持的所有税务技能列表"""
    return {
        "success": True,
        "data": {
            "name": knowledge_agent.name,
            "description": knowledge_agent.description,
            "capabilities": knowledge_agent.capabilities,
            # 子 Agent 能力
            "sub_agents": {
                "policy_agent": "政策条文解读与适用条件说明",
                "definition_agent": "税务术语定义与概念解释",
            },
        },
    }

@app.post("/api/agent")
async def agent_endpoint(body: TaxQueryRequest, current_user: CurrentUser = Depends(get_current_user)):
    """Agent 调用接口（预留扩展，复用 KnowledgeAgent 核心逻辑）"""
    from langchain_core.messages import HumanMessage
    try:
        session_id = body.session_id or f"session_{int(time.time())}"
        input_context = {
            "thread_id": session_id,
            "user_id": current_user.id,
            "user_name": current_user.user_name,
            "source_types": body.source_types,
        }
        messages = [HumanMessage(content=body.query)]
        result = await knowledge_agent.invoke_messages(messages, input_context=input_context)

        answer = ""
        for msg in reversed(result["messages"]):
            if hasattr(msg, "content") and not isinstance(msg, HumanMessage):
                answer = msg.content
                break

        return TaxQueryResponse(
            session_id=session_id,
            answer=answer or "（Agent 未返回有效内容）",
            references=[],
            status="success",
        )
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Agent 调用失败: {str(e)}")
@app.get("/api/search")
def global_search_api(q: str = "", limit: int = 12, current_user: CurrentUser = Depends(get_current_user)):
    """全文搜索接口：对接本地 RAG 知识库检索。"""
    if not q or not q.strip():
        return {
            "success": True,
            "query": q,
            "results": [],
            "timestamp": datetime.now().isoformat(),
            "message": "请输入搜索关键词",
        }
    try:
        from services.tools.kb_tools import search_tax_policy
        results = search_tax_policy(q, limit=max(1, min(limit, 20)))
        return {
            "success": True,
            "query": q,
            "results": results,
            "total": len(results),
            "timestamp": datetime.now().isoformat(),
            "message": "" if results else "未检索到相关内容",
        }
    except Exception as e:
        logger.warning(f"[global_search_api] 搜索失败: {e}")
        return {
            "success": True,
            "query": q,
            "results": [],
            "timestamp": datetime.now().isoformat(),
            "message": f"搜索暂时不可用：{e}",
        }

@app.post("/api/tax")
def create_tax(request: Request):
    """创建税务项目（预留接口）"""
    return {"success": True, "message": "税务项目创建接口（预留）"}


# ---------------------------------------------------------------------------
# 知识库管理（多知识库）
# ---------------------------------------------------------------------------

class KnowledgeBaseCreateRequest(BaseModel):
    name: str = Field(..., description="知识库名称")
    kb_type: str = Field("qdrant", description="知识库类型，默认 qdrant")


@app.get("/api/knowledge/databases")
async def list_knowledge_databases(current_user: CurrentUser = Depends(get_current_user)):
    """列出当前用户可见的知识库。"""
    if not kb_manager:
        raise HTTPException(status_code=503, detail="知识库管理器未初始化")
    user_info = {"uid": current_user.id, "role": current_user.role, "department_id": None}
    return await kb_manager.get_databases_by_user(user_info)


@app.post("/api/knowledge/databases")
async def create_knowledge_database(
    body: KnowledgeBaseCreateRequest,
    current_user: CurrentUser = Depends(get_current_user),
):
    """新建一个知识库（默认仅创建者可见）。"""
    from knowledge.kb import KnowledgeBaseFactory
    from database.repositories import KnowledgeBaseRepository

    name = (body.name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="知识库名称不能为空")
    if len(name) > 30:
        raise HTTPException(status_code=400, detail="知识库名称不能超过 30 个字符")
    kb_type = body.kb_type or "qdrant"
    if not KnowledgeBaseFactory.is_type_supported(kb_type):
        raise HTTPException(status_code=400, detail=f"不支持的知识库类型: {kb_type}")

    repo = KnowledgeBaseRepository()
    if repo.get_by_kb_id(name) is not None:
        raise HTTPException(status_code=409, detail=f"知识库 {name} 已存在")

    share_config = json.dumps({
        "access_level": "user",
        "department_ids": [],
        "user_uids": [current_user.id],
    })
    kb = repo.create(name=name, created_by=current_user.id, kb_type=kb_type, share_config=share_config)
    return {"success": True, "kb_id": kb.name, "name": kb.name, "kb_type": kb.kb_type}


@app.post("/api/knowledge/upload")
async def upload_knowledge(
    file: UploadFile = File(...),
    scope: str | None = Form(None),
    session_id: str | None = Form(None),
    current_user: CurrentUser = Depends(get_current_user),
):
    """【文件入库接口】接收上传文件，调用 KnowledgeService 保存。

    admin 上传 → source_type="default"（发布者默认知识库）
    普通用户上传 → source_type="user_upload"（用户个人知识库）
    """
    try:
        scope = scope or ("system" if current_user.role == "admin" else "global")
        if scope not in ("global", "session", "system"):
            raise HTTPException(status_code=400, detail="文档范围必须为 global/session/system")
        if scope == "system" and current_user.role != "admin":
            raise HTTPException(status_code=403, detail="只有管理员可上传系统资料")
        if scope == "session":
            if not session_id or not conversation_memory or conversation_memory.get_session(session_id) is None:
                raise HTTPException(status_code=400, detail="请先创建或选择本人对话，再上传单次对话文档")
        source_type = "default" if scope == "system" else "user_upload"
        logger.info(f"[上传] 用户={current_user.user_name}, role={current_user.role}, source_type={source_type}")
        result = await knowledge_service.ingest_file(file, source_type=source_type,
                    owner_id=current_user.id, scope=scope, session_id=session_id if scope == "session" else "")
        if result.get("status") != "success":
            raise HTTPException(status_code=502, detail=result.get("message", "文件入库未完成，请在文档页检查"))
        return result
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"文件上传失败: {str(e)}")


@app.get("/api/knowledge/files")
async def list_knowledge_files(include_system: bool = False, refresh_cloud: bool = False,
                               current_user: CurrentUser = Depends(get_current_user)):
    """【已上传文件列表】从持久化元数据读取，刷新不丢失"""
    try:
        if (include_system or refresh_cloud) and current_user.role != "admin":
            raise HTTPException(status_code=403, detail="仅管理员可查看系统和历史隔离资料")
        if refresh_cloud:
            await knowledge_service.refresh_cloud_files()
        files = await knowledge_service.list_files()
        files = [public_file_info(f) for f in files if can_manage_file(f, current_user, include_system)]
        return {"success": True, "files": files, "total": len(files)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"获取文件列表失败: {str(e)}")


@app.get("/api/knowledge/files/sql")
async def list_knowledge_files_sql(current_user: CurrentUser = Depends(get_current_user)):
    """【SQL 业务表文件列表】从 KnowledgeFiles + KnowledgeChunks 表读取"""
    try:
        files = await knowledge_service.list_files_from_sql()
        visible = {f["file_id"] for f in await knowledge_service.list_files() if can_manage_file(f, current_user)}
        files = [f for f in files if f.get("business_id") in visible]
        return {"success": True, "files": files, "total": len(files)}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"获取 SQL 文件列表失败: {str(e)}")


@app.delete("/api/knowledge/files/{file_id}")
async def delete_knowledge_file(file_id: str, current_user: CurrentUser = Depends(get_current_user)):
    """【删除文件】全链路清理：元数据 + Qdrant 向量 + 磁盘原文件"""
    try:
        record = await authorized_file(file_id, current_user, admin=True)
        metadata = record.get("tax_metadata", {})
        if conversation_memory is not None and metadata.get("owner_id"):
            from services.security import tenant_scope
            with tenant_scope(metadata["owner_id"]):
                sessions = ([{"session_id": metadata.get("session_id")}]
                            if metadata.get("scope") == "session" else conversation_memory.list_sessions(limit=1000))
                for session in sessions:
                    sid = session.get("session_id")
                    if sid:
                        conversation_memory.invalidate_document_context(sid)
                        await clear_conversation_checkpoints(sid)
        result = await knowledge_service.delete_file(file_id)
        if result.get("status") == "error":
            raise HTTPException(status_code=404, detail=result.get("message"))
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"删除文件失败: {str(e)}")


def can_manage_file(record, user, include_system=False):
    metadata = record.get("tax_metadata", {})
    if include_system and user.role == "admin":
        return True
    if user.role == "admin" and metadata.get("source_type") == "default":
        return True
    return (metadata.get("owner_id") == user.id and metadata.get("scope") in ("global", "session")
            and metadata.get("source_type") != "default")


def public_file_info(record):
    metadata = record.get("tax_metadata", {})
    return {"file_id": record["file_id"], "file_name": record.get("file_name", record["file_id"]),
            "filename": record.get("filename", record.get("file_name", record["file_id"])),
            "size": record.get("size"), "mime_type": record.get("mime_type"),
            "url": record.get("url", f"/api/knowledge/files/{record['file_id']}/content"),
            "chunk_count": record.get("chunk_count", 0), "status": record.get("status", "active"),
            "scope": metadata.get("scope") or ("system" if metadata.get("source_type") == "default" else "legacy"),
            "session_id": metadata.get("session_id", ""), "cloud_only": record.get("cloud_only", False)}


async def authorized_file(file_id, user, admin=False):
    if knowledge_service is None:
        raise HTTPException(status_code=503, detail="知识库服务未初始化")
    record = next((f for f in await knowledge_service.list_files() if f.get("file_id") == file_id), None)
    if record is None or not can_manage_file(record, user, admin):
        raise HTTPException(status_code=404, detail="文档不存在或不属于当前用户")
    return record


@app.get("/api/knowledge/files/{file_id}/content")
async def preview_knowledge_file(file_id: str, current_user: CurrentUser = Depends(get_current_user)):
    record = await authorized_file(file_id, current_user, admin=True)
    from knowledge.embedding import PersistentDocumentStore
    store = PersistentDocumentStore()
    chunks = [c.content for c in store.get_chunks() if c.file_id == file_id]
    if not chunks and record.get("cloud_only"):
        chunks = await knowledge_service.read_cloud_chunks(file_id)
    if not chunks and record.get("file_path"):
        from knowledge.parser import ParserFactory
        import asyncio
        chunks = [await asyncio.to_thread(ParserFactory.get_parser(record["file_path"]).parse)]
    if not chunks:
        raise HTTPException(status_code=404, detail="文档正文未找到，请检查原文件或云端连接")
    text = "\n\n".join(chunks)
    return {"success": True, **public_file_info(record), "content": text[:200000],
            "truncated": len(text) > 200000 or (record.get("cloud_only", False) and len(chunks) < record.get("chunk_count", 0))}

@app.get("/api/memory")
def get_memory(user_id: str, session_id: str | None = None, current_user: CurrentUser = Depends(get_current_user)):
    """获取用户长期记忆（含会话列表、指定会话详情、健康度统计）"""
    if session_id:
        session = conversation_memory.get_session(session_id)
        if session is None:
            return {"success": False, "message": f"会话 {session_id} 不存在"}
        ctx = conversation_memory.context_for(session_id, "")
        return {
            "success": True,
            "user_id": user_id,
            "session_id": session_id,
            "profile": session.get("profile", {}),
            "summary": session.get("summary", ""),
            "turns": session.get("turns", 0),
            "memory_layers": ctx.get("memory_layers", {}),
            "context_budget": ctx.get("context_budget", {}),
        }

    sessions = conversation_memory.list_sessions(limit=50)
    stats = conversation_memory.stats()
    return {
        "success": True,
        "user_id": user_id,
        "stats": stats,
        "sessions": sessions,
    }


@app.delete("/api/memory/{session_id}")
async def delete_memory(session_id: str, current_user: CurrentUser = Depends(get_current_user)):
    """删除指定会话的长记忆（含 LangGraph checkpointer 状态）"""
    return await delete_conversation(session_id, current_user)


# ---------------------------------------------------------------------------
# 对话记录侧边栏 API（供前端左侧对话列表使用）
# ---------------------------------------------------------------------------

@app.get("/api/conversations")
async def list_conversations(user_id: str = "web_user", current_user: CurrentUser = Depends(get_current_user)):
    """获取对话列表（用于侧边栏展示）

    返回：
        - session_id: 对话 ID（前端已有当前 session_id，无需额外区分）
        - title: 自动生成的对话主题
        - first_message: 第一条用户消息（预览用）
        - turns: 对话轮数
        - updated_at: 最后更新时间
    """
    sessions = conversation_memory.list_sessions(limit=100)
    result = []
    for s in sessions:
        sid = s.get("session_id", "")
        title = s.get("title", "")
        first_msg = ""
        # 从完整会话中提取第一条用户消息作为预览
        try:
            full = conversation_memory.get_session(sid)
            if full and full.get("messages"):
                for m in full["messages"]:
                    if m.get("role") == "user":
                        first_msg = m.get("content", "")[:50]
                        break
        except Exception as e:
            logger.debug("[会话预览] 读取失败（%s）", type(e).__name__)
        # 若无 title（存量数据），用第一条消息生成
        if not title and first_msg:
            # 警告：此操作不应在只读查询中修改数据，这里仅做临时展示
            title = first_msg[:20] + ("…" if len(first_msg) > 20 else "")
        result.append({
            "session_id": sid,
            "title": title or "新对话",
            "first_message": first_msg,
            "turns": s.get("turns", 0),
            "updated_at": s.get("updated_at", ""),
        })
    # 按更新时间倒序排列
    result.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    return {"success": True, "conversations": result}


class CreateConversationRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=120, pattern=r"^[A-Za-z0-9_-]+$")


@app.post("/api/conversations")
async def create_conversation(body: CreateConversationRequest, current_user: CurrentUser = Depends(get_current_user)):
    if conversation_memory is None:
        raise HTTPException(status_code=503, detail="会话服务未初始化")
    conversation_memory.create_session(body.session_id)
    return {"success": True, "session_id": body.session_id}


@app.get("/api/conversations/{session_id}/messages")
async def get_conversation_messages(session_id: str, current_user: CurrentUser = Depends(get_current_user)):
    """获取指定对话的完整消息历史（用于恢复对话）"""
    session = conversation_memory.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"对话 {session_id} 不存在")
    messages = session.get("messages", [])
    title = session.get("title", "")
    return {
        "success": True,
        "session_id": session_id,
        "title": title,
        "messages": messages,
    }


@app.get("/api/conversations/{session_id}/workflow")
async def get_conversation_workflow(session_id: str, current_user: CurrentUser = Depends(get_current_user)):
    if conversation_memory is None or conversation_memory.get_session(session_id) is None:
        raise HTTPException(status_code=404, detail="对话不存在或不属于当前用户")
    if tax_agent is None or not callable(getattr(tax_agent, "workflow_state", None)):
        raise HTTPException(status_code=503, detail="工作流服务未就绪，请检查模型配置")
    snapshot = await tax_agent.workflow_state(session_id)
    values = snapshot.values if snapshot else {}
    pending = await tax_agent.pending_workflow(session_id)
    return {"pending": pending, "trace": values.get("workflow_trace", []),
            "artifacts": [{"id": aid, "title": item["title"]} for aid, item in values.get("artifacts", {}).items()]}


@app.post("/api/conversations/{session_id}/resume")
async def resume_conversation(session_id: str, body: WorkflowResume, current_user: CurrentUser = Depends(get_current_user)):
    await get_conversation_workflow(session_id, current_user)  # 先验证归属与服务状态。
    try:
        answer = await tax_agent.run("人工批准生成文件" if body.value else "人工拒绝生成文件",
                                    {"thread_id": session_id, "uid": current_user.id, "resume": body.model_dump()})
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("[workflow] resume_failed=%s", type(exc).__name__)
        raise HTTPException(status_code=503, detail="恢复失败，请保留待审批记录并重试") from exc
    return {"answer": answer, **await get_conversation_workflow(session_id, current_user)}


@app.get("/api/conversations/{session_id}/artifacts/{artifact_id}")
async def download_workflow_artifact(session_id: str, artifact_id: str, current_user: CurrentUser = Depends(get_current_user)):
    await get_conversation_workflow(session_id, current_user)
    snapshot = await tax_agent.workflow_state(session_id)
    artifact = snapshot.values.get("artifacts", {}).get(artifact_id) if snapshot else None
    if artifact is None:
        raise HTTPException(status_code=404, detail="文件未生成或已随对话删除")
    return Response(content=json.dumps(artifact, ensure_ascii=False, indent=2), media_type="application/json",
                    headers={"Content-Disposition": 'attachment; filename="filing-draft.json"'})


class RenameConversationRequest(BaseModel):
    title: str = Field(..., description="新标题")


@app.patch("/api/conversations/{session_id}")
async def rename_conversation(
    session_id: str,
    body: RenameConversationRequest,
    current_user: CurrentUser = Depends(get_current_user),
):
    """重命名会话标题。"""
    if not conversation_memory:
        raise HTTPException(status_code=503, detail="记忆模块未初始化")
    title = (body.title or "").strip()
    if not title:
        raise HTTPException(status_code=400, detail="标题不能为空")
    ok = conversation_memory.rename_session(session_id, title)
    if not ok:
        raise HTTPException(status_code=404, detail=f"对话 {session_id} 不存在")
    return {"success": True, "session_id": session_id, "title": title}


@app.delete("/api/conversations/{session_id}")
async def delete_conversation(session_id: str, current_user: CurrentUser = Depends(get_current_user)):
    """删除指定对话（记忆 + LangGraph checkpointer 状态一并清理）"""
    if not conversation_memory or conversation_memory.get_session(session_id) is None:
        raise HTTPException(status_code=404, detail="对话不存在或不属于当前用户")
    if knowledge_service is not None:
        for record in await knowledge_service.list_files():
            metadata = record.get("tax_metadata", {})
            if (metadata.get("owner_id") == current_user.id and metadata.get("scope") == "session"
                    and metadata.get("session_id") == session_id):
                await delete_knowledge_file(record["file_id"], current_user)
    # 同时清理 LangGraph checkpointer 里的多轮历史（若已启用）
    await clear_conversation_checkpoints(session_id)
    ok = conversation_memory.delete_session(session_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"对话 {session_id} 不存在")
    return {"success": True, "session_id": session_id}


async def clear_conversation_checkpoints(session_id):
    names = ("knowledge_agent", "planning_agent", "calculation_agent", "invoice_agent", "compliance_agent", "filing_agent")
    if tax_agent is None:
        # 清理历史状态不依赖模型配置；复用原有 SQLite Saver，不创建模型客户端。
        from agents.base_agent import _CHECKPOINT_DB_PATH, AsyncSqliteSaver
        from services.security import checkpoint_thread_id
        if not _CHECKPOINT_DB_PATH.exists():
            return
        try:
            async with AsyncSqliteSaver.from_conn_string(str(_CHECKPOINT_DB_PATH)) as saver:
                for thread_id in [session_id, *(f"{session_id}:{name}" for name in names)]:
                    await saver.adelete_thread(checkpoint_thread_id(thread_id))
        except Exception as exc:
            raise HTTPException(status_code=503, detail="历史上下文清理失败，请保留记录并重试") from exc
        return
    for thread_id in [session_id, *(f"{session_id}:{name}" for name in names)]:
        if not await tax_agent.delete_thread(thread_id):
            raise HTTPException(status_code=503, detail="对话上下文清理未完成，请保留对话并重试")


@app.delete("/api/conversations/{session_id}/messages/last")
async def retract_last_turn(session_id: str, current_user: CurrentUser = Depends(get_current_user)):
    """撤回最后一轮对话（移除最新的 user + assistant 消息对）。"""
    if not conversation_memory:
        raise HTTPException(status_code=503, detail="记忆模块未初始化")
    ok = conversation_memory.pop_last_turn(session_id)
    if not ok:
        raise HTTPException(status_code=404, detail="无可撤回的消息")
    return {"success": True, "session_id": session_id}


# ================================================================
# 政策自动更新 API
# ================================================================

def policy_source_from_config(data):
    from knowledge.policy_updater import LocalDirectorySource, HttpCrawlSource, TaxGovCnSource
    kind = data.get("source_type") or data.get("type")
    if kind in ("local", "local_directory"):
        return LocalDirectorySource(directory=data["directory"], source_id=data["source_id"],
                                    recursive=data.get("recursive", True), category=data.get("category", ""))
    if kind == "http":
        for url in data["urls"]:
            validate_api_url(url)
        return HttpCrawlSource(source_id=data["source_id"], urls=data["urls"], category=data.get("category", ""))
    url = data.get("listing_url") or (data.get("urls") or [""])[0]
    validate_api_url(url)
    options = {key:data[key] for key in ("max_pages", "max_items", "link_selector", "title_selector", "content_selector", "date_selector") if key in data}
    if kind == "website":
        options.update(link_selector="a[href*='content.html'], a[href*='/t20'], a[href^='./20'], main a[href$='.html'], .list a[href$='.html']",
                       title_selector="meta[name='ArticleTitle'], h1, h2.title_con, .title",
                       content_selector=".TRS_Editor, #zoom, #fontzoom, article, .article, .xxgk_content", max_pages=1, max_items=5)
    return TaxGovCnSource(listing_url=url, source_id=data["source_id"], category=data.get("category", ""), **options)


class PolicyScheduleRequest(BaseModel):
    interval_hours: int = Field(default=6, ge=1, le=168)


@app.put("/api/policy/schedule")
def set_policy_schedule(body: PolicyScheduleRequest, current_user: CurrentUser = Depends(get_current_user)):
    if not policy_updater:
        raise HTTPException(status_code=503, detail="政策更新器未初始化")
    ConfigManager().save_policy_config({"interval_hours":body.interval_hours})
    policy_updater.set_interval(body.interval_hours * 3600)
    return {"success":True, "scheduler":policy_updater.get_scheduler_status()}

@app.post("/api/policy/update")
def trigger_policy_update(current_user: CurrentUser = Depends(get_current_user)):
    """手动触发一次政策更新扫描（扫描→去重→增量入库）"""
    if not policy_updater:
        raise HTTPException(status_code=503, detail="政策更新器未初始化")
    try:
        report = policy_updater.run_once()
        return {
            "success": True,
            "report": report.to_dict(),
        }
    except Exception as e:
        logger.error(f"[政策更新] 手动触发失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"政策更新失败: {e}")


@app.get("/api/policy/status")
def get_policy_status(current_user: CurrentUser = Depends(get_current_user)):
    """获取政策更新器运行状态与最近一次更新报告"""
    if not policy_updater:
        raise HTTPException(status_code=503, detail="政策更新器未初始化")
    status = policy_updater.get_scheduler_status()
    latest = policy_updater.get_latest_report()
    return {
        "success": True,
        "scheduler": status,
        "latest_report": latest,
    }


@app.get("/api/policy/sources")
def list_policy_sources(current_user: CurrentUser = Depends(get_current_user)):
    """列出所有已配置的政策源"""
    if not policy_updater:
        raise HTTPException(status_code=503, detail="政策更新器未初始化")
    return {
        "success": True,
        "sources": policy_updater.get_sources_info(),
    }


@app.post("/api/policy/sources")
def add_policy_source(body: PolicySourceRequest, current_user: CurrentUser = Depends(get_current_user)):
    """动态添加一个政策源（本地目录 或 HTTP 抓取）"""
    if not policy_updater:
        raise HTTPException(status_code=503, detail="政策更新器未初始化")

    from knowledge.policy_updater import LocalDirectorySource, HttpCrawlSource

    try:
        if body.type == "local":
            if not body.directory:
                raise HTTPException(status_code=400, detail="type=local 时必须提供 directory")
            src = LocalDirectorySource(
                directory=body.directory,
                source_id=body.source_id or f"local_{int(time.time())}",
                recursive=body.recursive,
                category=body.category,
            )
        elif body.type in ("http", "website"):
            if not body.urls:
                raise HTTPException(status_code=400, detail="type=http 时必须提供 urls")
            src = policy_source_from_config({**body.model_dump(), "source_id":body.source_id or "site_" + uuid.uuid4().hex[:12]})
        else:
            raise HTTPException(status_code=400, detail=f"不支持的政策源类型: {body.type}，仅支持 local / http / website")

        if any(s["source_id"] == src.source_id for s in policy_updater.get_sources_info()):
            raise HTTPException(status_code=409, detail="来源 ID 已存在")
        # 先保存完整清单再应用，写盘失败时不会出现虚假的成功。
        info = policy_updater.get_sources_info()
        from knowledge.policy_updater import PolicyUpdater
        descriptor = {"source_type": src.source_type, "source_id": src.source_id, "category":src.category}
        if isinstance(src, LocalDirectorySource):
            descriptor.update(directory=src.directory, recursive=src.recursive)
        elif isinstance(src, HttpCrawlSource):
            descriptor.update(urls=list(src.urls))
        else:
            descriptor.update(listing_url=src.listing_url, link_selector=src.link_selector,
                              title_selector=src.title_selector, content_selector=src.content_selector,
                              max_pages=src.max_pages, max_items=src.max_items)
        ConfigManager().save_policy_config({"sources":info + [descriptor]})
        policy_updater.add_source(src)
        logger.info(f"[政策更新] 已添加政策源: type={body.type}, id={src.source_id}")
        return {
            "success": True,
            "source": {"type": body.type, "source_id": src.source_id},
        }
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error(f"[政策更新] 添加政策源失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"添加政策源失败: {e}")


@app.get("/api/policy/reports")
def get_policy_reports(limit: int = 20, current_user: CurrentUser = Depends(get_current_user)):
    """获取历史更新报告列表（按时间倒序）"""
    if not policy_updater:
        raise HTTPException(status_code=503, detail="政策更新器未初始化")
    return {
        "success": True,
        "reports": policy_updater.get_report_history(limit=max(1, min(limit, 200))),
    }


@app.delete("/api/policy/sources/{source_id}")
def remove_policy_source(source_id: str, current_user: CurrentUser = Depends(get_current_user)):
    """按 source_id 移除一个政策源"""
    if not policy_updater:
        raise HTTPException(status_code=503, detail="政策更新器未初始化")
    sources = policy_updater.get_sources_info()
    if not any(s["source_id"] == source_id for s in sources):
        raise HTTPException(status_code=404, detail=f"政策源不存在: {source_id}")
    ConfigManager().save_policy_config({"sources":[s for s in sources if s["source_id"] != source_id]})
    policy_updater.remove_source(source_id)
    return {"success": True, "removed": source_id}


@app.get("/api/policy/changes")
def get_policy_changes(limit: int = 20, current_user: CurrentUser = Depends(get_current_user)):
    """获取内容发生变更的政策列表"""
    if not policy_updater:
        raise HTTPException(status_code=503, detail="政策更新器未初始化")
    return {
        "success": True,
        "changes": policy_updater.tracker.get_changes(limit=max(1, min(limit, 200))),
    }


@app.get("/api/policy/changes/{unique_key:path}")
def get_policy_change_diff(unique_key: str, current_user: CurrentUser = Depends(get_current_user)):
    """获取某条政策的文字级变更 diff"""
    if not policy_updater:
        raise HTTPException(status_code=503, detail="政策更新器未初始化")
    diff = policy_updater.tracker.get_change_diff(unique_key=unique_key)
    if not diff:
        raise HTTPException(status_code=404, detail=f"未找到变更: {unique_key}")
    return {"success": True, **diff}


# ================================================================
# 模型配置 API（前端设置页）
# ================================================================

# 可选模型列表（前端下拉菜单用）
_KNOWN_MODELS = [
    {"value": "glm-5.3", "label": "GLM-5.3（智谱）"},
    {"value": "glm-4.6", "label": "GLM-4.6（智谱）"},
    {"value": "glm-4-flash", "label": "GLM-4-Flash（智谱，快速）"},
    {"value": "glm-4.7", "label": "GLM-4.7（智谱）"},
    {"value": "deepseek-v4-flash", "label": "DeepSeek V4 Flash"},
    {"value": "deepseek-v4", "label": "DeepSeek V4"},
    {"value": "BAAI/bge-large-zh-v1.5", "label": "BGE-Large（中文向量，1024维）"},
    {"value": "BAAI/bge-m3", "label": "BGE-M3（多语言向量，1024维）"},
]


@app.get("/api/me")
def me(current_user: CurrentUser = Depends(get_current_user)):
    """返回当前登录用户信息（前端据此决定是否展示开发者选项）。"""
    return {
        "success": True,
        "id": current_user.id,
        "user_name": current_user.user_name,
        "role": current_user.role,
        "is_developer": current_user.role == "admin",
    }


@app.get("/api/settings")
def get_settings(current_user: CurrentUser = Depends(get_current_user)):
    """获取当前模型配置（API Key 已掩码）"""
    return {
        "success": True,
        "config": ConfigManager().get_all_config(),
        "known_models": _KNOWN_MODELS,
        "providers": MODEL_PROVIDERS,
    }


class SettingsUpdateRequest(BaseModel):
    llm: dict | None = None
    embedding: dict | None = None
    qdrant: dict | None = None


class ModelConnectionRequest(BaseModel):
    section: str = Field(pattern="^(llm|embedding)$")
    base_url: str = Field(min_length=1, max_length=2048)
    api_key: str = Field(default="", max_length=4096)


@app.post("/api/settings/models")
def probe_models(body: ModelConnectionRequest, current_user: CurrentUser = Depends(get_current_user)):
    """服务端检测 OpenAI 兼容 /models；密钥不回传，也不写入配置。"""
    import requests
    try:
        url = validate_api_url(body.base_url.strip())
        key = ConfigManager().resolve_model_key(body.section, url, body.api_key)
        if not key:
            raise ValueError("请填写所选提供商的 API Key")
        response = requests.get(url + "models", headers={"Authorization": "Bearer " + key},
                                timeout=(5, 15), allow_redirects=False)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
            raise ValueError("模型列表必须是 OpenAI 兼容的 data 数组")
        models = sorted({row["id"] for row in payload["data"] if isinstance(row, dict)
                         and isinstance(row.get("id"), str) and row["id"].strip()})
        if infer_provider(url) == "Gemini":
            models = [model.removeprefix("models/") for model in models]
        if not models:
            raise ValueError("模型列表为空")
        return {"success": True, "models": models, "message": "✅ 连接成功"}
    except (requests.RequestException, ValueError, TypeError) as exc:
        logger.warning("[模型检测] 失败（%s）", type(exc).__name__)
        return {"success": False, "models": [], "message": "❌ 无法连接，请检查配置"}


@app.get("/api/settings/qdrant")
def qdrant_usage(current_user: CurrentUser = Depends(get_current_user)):
    """只读监控实际点数；空间是向量 float32 裸数据估算，不是云端账单。"""
    from qdrant_client import QdrantClient
    from knowledge.embedding import QdrantService
    cfg = ConfigManager().get_qdrant_config()
    client = None
    try:
        validate_api_url(cfg["url"])
        client = QdrantClient(url=cfg["url"], api_key=cfg["api_key"] or None, timeout=10,
                              check_compatibility=False)
        collection = QdrantService.COLLECTION_NAME
        names = {c.name for c in client.get_collections().collections}
        dimensions, total, state = 0, 0, "集合尚未创建"
        if collection in names:
            info = client.get_collection(collection)
            vectors = info.config.params.vectors
            dimensions = sum(v.size for v in vectors.values()) if isinstance(vectors, dict) else vectors.size
            total = client.count(collection, exact=True).count
            state = str(info.status)
        limit = cfg["vector_limit"]
        estimated_bytes = total * dimensions * 4
        # Cloud 免费档资源参考，不是实际 RAM 用量；自建实例不套用此参考。
        from urllib.parse import urlparse
        cloud_reference = (urlparse(cfg["url"]).hostname or "").endswith(".qdrant.io")
        reference_ram_bytes = 1024 ** 3 if cloud_reference else None
        estimated_ram_bytes = int(estimated_bytes * 1.5)
        usage_ratio = total / limit if limit else estimated_ram_bytes / reference_ram_bytes if reference_ram_bytes else None
        return {"success": True, "collection": collection, "status": state,
                "vector_count": total, "dimensions": dimensions, "estimated_bytes": estimated_bytes,
                "estimated_ram_bytes": estimated_ram_bytes, "reference_ram_bytes": reference_ram_bytes,
                "vector_limit": limit, "usage_ratio": usage_ratio,
                "warning": usage_ratio is not None and usage_ratio >= .8,
                "estimate_note": "仅估算 float32 向量正文，不含索引、payload 或其他集合；免费版按资源限制，并无固定向量数上限。"}
    except Exception as exc:
        logger.warning("[Qdrant监控] 查询失败（%s）", type(exc).__name__)
        return {"success": False, "collection": QdrantService.COLLECTION_NAME,
                "message": "Qdrant 无法连接，请检查 URL、API Key 或服务状态"}
    finally:
        if client is not None:
            client.close()


@app.post("/api/settings")
def update_settings(body: SettingsUpdateRequest, current_user: CurrentUser = Depends(get_current_user)):
    """保存模型配置到本地 config.json"""
    global knowledge_agent, tax_agent
    try:
        old_qdrant = ConfigManager().get_qdrant_config()
        incoming = body.model_dump(exclude_none=True)
        candidate = ConfigManager()._save_config_locked(incoming, persist=False)
        next_qdrant = candidate.get("qdrant", old_qdrant)
        changed_qdrant = any(next_qdrant.get(key) != old_qdrant.get(key) for key in ("url", "api_key"))
        if changed_qdrant:
            # 先检查新地址与密钥，失败时保留原配置和检索客户端。
            from qdrant_client import QdrantClient
            client = QdrantClient(url=next_qdrant["url"], api_key=next_qdrant["api_key"] or None,
                                  timeout=10, check_compatibility=False)
            try:
                client.get_collections()
            except Exception as exc:
                raise HTTPException(status_code=400, detail="Qdrant 无法连接，请检查 URL 和 API Key；原配置已保留") from exc
            finally:
                client.close()
        ConfigManager().save_config(incoming)
        from langchain_openai import ChatOpenAI
        cfg = ConfigManager().get_llm_config()
        if all(cfg.values()) and not cfg["api_key"].startswith("your_"):
            if tax_agent is None:
                from agents.tax_agent import TaxAgent
                tax_agent = TaxAgent()
                app.state.tax_agent = tax_agent
            if knowledge_agent is None:
                from agents.knowledge_agent import KnowledgeAgent
                knowledge_agent = KnowledgeAgent()
                app.state.knowledge_agent = knowledge_agent
            # 更新现有实例的客户端，保留会话、graph、checkpointer 和 API。
            agents = [tax_agent, knowledge_agent]
            registry = vars(tax_agent).get("registry")
            if registry is not None:
                registry._cache["knowledge_agent"] = knowledge_agent
                agents.extend(vars(registry).get("_cache", {}).values())
            for agent in list(agents):
                agents.extend(child for name in ("policy_agent", "definition_agent")
                              if (child := vars(agent).get(name)) is not None)
            for agent in agents:
                for attr in ("llm", "extract_llm"):
                    old = vars(agent).get(attr)
                    if old is not None:
                        import httpx as _httpx
                        setattr(agent, attr, ChatOpenAI(
                            model=cfg["model"], api_key=cfg["api_key"], base_url=cfg["base_url"],
                            temperature=getattr(old, "temperature", 0.3),
                            timeout=_httpx.Timeout(connect=10.0, read=45.0, write=10.0, pool=5.0),
                            max_retries=2,
                        ))
                svc = vars(agent).get("_embedding_service")
                if svc is not None:
                    svc.embedding_model = ConfigManager().get_embedding_config()["model"]
            if conversation_memory is not None:
                conversation_memory.set_title_llm(ChatOpenAI(
                    model=cfg["model"], api_key=cfg["api_key"], base_url=cfg["base_url"],
                    temperature=0.2, timeout=30.0, max_retries=2,
                ))
        if body.embedding or changed_qdrant:
            # 更新检索客户端，保留原有文档、索引和 graph。
            from knowledge.embedding import EmbeddingService, PersistentDocumentStore, QdrantService
            cloud = QdrantService() if changed_qdrant else None
            targets = [knowledge_agent, tax_agent]
            if tax_agent is not None and vars(tax_agent).get("registry") is not None:
                targets.extend(tax_agent.registry._cache.values())
            for agent in list(targets):
                if agent is not None:
                    targets.extend(child for name in ("policy_agent", "definition_agent") if (child := vars(agent).get(name)) is not None)
            for agent in targets:
                if agent is not None and "_embedding_service" in vars(agent):
                    if agent._embedding_service is None:
                        agent._store = PersistentDocumentStore()
                        agent._embedding_service = EmbeddingService(agent._store, skip_rebuild=True)
                    agent._embedding_service.embedding_model = ConfigManager().get_embedding_config()["model"]
                    if changed_qdrant:
                        agent._embedding_service.qdrant = cloud
                        agent._embedding_service._qdrant_payload_cache.clear()
            if knowledge_service is not None:
                svc = knowledge_service._kb._embedding_service
                if svc is not None:
                    svc.embedding_model = ConfigManager().get_embedding_config()["model"]
                    if changed_qdrant:
                        svc.qdrant = cloud
                        svc._qdrant_payload_cache.clear()
        return {"success": True, "message": "配置已保存，下次对话生效"}
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.warning("[配置] 更新失败（%s）", type(e).__name__)
        raise HTTPException(status_code=500, detail="配置更新失败，请检查模型配置；已保存的配置可在弹窗中查看")

# —— 联网检索结果管理 ——

class WebSearchIngestRequest(BaseModel):
    query: str = Field(..., description="搜索关键词")
    limit: int = Field(default=5, ge=1, le=10, description="返回条数，默认 5，最多 10")


@app.post("/api/knowledge/web-search/ingest")
async def ingest_web_search(body: WebSearchIngestRequest, current_user: CurrentUser = Depends(get_current_user)):
    """联网检索并入库。

    搜索结果将被分块、向量化、写入 Qdrant（source_type=web_search），
    后续 RAG 检索可命中。
    """
    try:
        from services.tools.web_search_tool import search_and_ingest_web
        if not body.query.strip():
            raise HTTPException(status_code=400, detail="请输入搜索关键词")
        import asyncio
        result = await asyncio.to_thread(search_and_ingest_web, query=body.query, limit=body.limit)
        if result.get("file_id") and result.get("ingested"):
            knowledge_service._kb._metadata.setdefault("files", []).append({
                "file_id": result["file_id"], "file_name": body.query[:80], "file_path": "",
                "chunk_count": result["ingested"], "status": "active", "cloud_only": True,
                "tax_metadata": {"source_type": "web_search", "owner_id": current_user.id, "scope": "global", "session_id": ""}})
            knowledge_service._kb.save_metadata()
        return {"success": True, **result}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"联网检索入库失败: {str(e)}")



# 联网检索结果管理
@app.delete("/api/knowledge/web-search")
async def delete_web_search_results(file_id: str | None = None, current_user: CurrentUser = Depends(get_current_user)):
    """删除联网检索结果。

    不传 file_id → 删除全部 web_search 来源的数据
    传入 file_id → 删除指定批次
    """
    try:
        from services.tools.web_search_tool import delete_web_search_results as do_delete
        records = await knowledge_service.list_files()
        managed = [r for r in records if (not file_id or r["file_id"] == file_id)
                   and r.get("tax_metadata", {}).get("owner_id") == current_user.id
                   and r.get("tax_metadata", {}).get("source_type") == "web_search"]
        for record in managed:
            await delete_knowledge_file(record["file_id"], current_user)
        if file_id and managed:
            return {"success": True, "deleted_files": len(managed)}
        result = do_delete(file_id=file_id)
        if result.get("status") == "error":
            raise HTTPException(status_code=502, detail=result["message"])
        return {"success": True, "deleted_files": len(managed), **result}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除联网检索结果失败: {str(e)}")
