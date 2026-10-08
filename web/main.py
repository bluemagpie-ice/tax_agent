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
todo：更新后的ingest_file和TaxAgent需要阅读理解
todo:需要更新security.py中的租户信息。
#todo：html中提取主题是通过概括的方式提取，如果不能成果再提取前20词（除了ai外有无更好的方法）。
#todo：html中增加删除对话（同时删除对应的id和其他存储的内容）
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
from datetime import datetime
from pathlib import Path
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, UploadFile, File, HTTPException, status, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.params import Depends
from starlette.responses import HTMLResponse, Response, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from services.security import bind_tenant, reset_tenant
from services.config_manager import ConfigManager


logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
PATHS = {
    "templates": str(Path(__file__).resolve().parent / "templates"),
    "uploads": str(BASE_DIR / "uploads"),
    "data": str(BASE_DIR / "data"),
}
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

    app.state.knowledge_agent = KnowledgeAgent()
    app.state.tax_agent = TaxAgent()
    app.state.knowledge_service = KnowledgeService()
    app.state.conversation_memory = _get_shared_memory()

    # ---- 知识库管理器（多知识库 + 权限过滤） ----
    from knowledge.kb_manager import KnowledgeBaseManager
    app.state.kb_manager = KnowledgeBaseManager(work_dir=str(BASE_DIR / "data" / "kb"))

    # —— 预置默认知识库自动入库（knowledge/presets/ 目录下的 .md 文件） ——
    presets_dir = BASE_DIR / "knowledge" / "presets"
    if presets_dir.is_dir():
        preset_files = sorted(presets_dir.glob("*.md"))
        if preset_files:
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
            logger.info("[lifespan] knowledge/presets/ 目录为空，跳过预置知识入库")
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
    # 可选：对接国家税务总局法规库（配置 TAX_CN_LISTING_URL 环境变量激活）
    chinatax_url = os.getenv("TAX_CN_LISTING_URL", "")
    if chinatax_url:
        policy_updater.add_source(TaxGovCnSource(
            listing_url=chinatax_url,
            source_id="chinatax",
            category=os.getenv("TAX_CN_CATEGORY", "税务法规"),
            max_pages=int(os.getenv("TAX_CN_MAX_PAGES", "3")),
        ))
        logger.info(f"[lifespan] 已注册国家税务总局法规源: {chinatax_url}")
    # 启动后台定时扫描（每 6 小时一次）
    policy_updater.start_scheduler(interval_seconds=6 * 3600)
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
    if hasattr(app.state, "policy_updater") and app.state.policy_updater:
        try:
            app.state.policy_updater.stop_scheduler()
        except Exception:
            pass
    # 取消所有残余 async 任务，避免 "event loop is closed"
    import asyncio
    tasks = [t for t in asyncio.all_tasks()
             if t is not asyncio.current_task() and not t.done()]
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

    async def ingest_file(self, file: UploadFile, source_type: str = "user_upload") -> dict:
        """
        文件入库管道【核心方法】

        保存 → 委托 QdrantKnowledgeBase（解析→分块→向量化→Qdrant+元数据持久化）

        :param source_type: 知识来源，default / user_upload / web_search
        """
        # 1. 保存文件到本地
        file_path = self._save_file(file)
        try:
            import time

            file_id = f"file_{int(time.time())}"
            result = await self._kb.ingest_file(
                file_path=str(file_path),
                file_id=file_id,
                source_type=source_type,
            )

            if result.get("status") == "success":
                result["message"] = f"文件 {file.filename} 已完成向量化，共 {result.get('chunk_count', 0)} 个分块"
                result["file_name"] = file.filename

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
                        for c in PersistentDocumentStore.get_chunks()
                        if c.file_id == file_id
                    ]
                    if chunks_for_sql:
                        chunk_repo.create_batch(sql_record.id, chunks_for_sql)
                    logger.info(
                        f"[ingest_file] SQL 记录已写入: file_id={file_id}, "
                        f"sql_id={sql_record.id}, chunks={len(chunks_for_sql)}"
                    )
                except Exception as sql_err:
                    logger.warning(f"[ingest_file] SQL 写入失败（不影响主流程）: {sql_err}")

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
        ok = await self._kb.delete_file(file_id)

        # 同步清理 SQL 业务表
        try:
            from database.repositories import KnowledgeFileRepository
            repo = KnowledgeFileRepository()
            sql_deleted = repo.delete_by_business_id(file_id)
            if sql_deleted:
                logger.info(f"[delete_file] SQL 记录已清理: file_id={file_id}")
        except Exception as sql_err:
            logger.warning(f"[delete_file] SQL 清理失败（不影响主流程）: {sql_err}")

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

        # —— 去重：检查该 preset 是否已入库 ——
        if self._kb.is_file_ingested(file_id):
            return {"message": f"[{file_name}] 已入库，跳过", "file_id": file_id, "skipped": True}

        return await self._kb.ingest_file(
            file_path=file_path,
            file_id=file_id,
            source_type="default",
        )

    def _save_file(self, file: UploadFile) -> Path:
        """
        将上传文件保存到 uploads/ 目录
        文件名加时间戳前缀，防止同名文件冲突
        """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = f"{ts}_{file.filename}"
        file_path = Path(self.upload_dir) / safe_name
        with open(file_path, "wb") as f:
            f.write(file.file.read())
        return file_path


# 框架代码：Pydantic 请求/响应数据模型
# 作用：自动校验 HTTP 请求体格式，自动生成 OpenAPI 文档
class CurrentUser(BaseModel):
    id:str
    user_name:str
    role:str = "user" # 默认用户


class TaxQueryRequest(BaseModel):
    user_id:str = Field(..., description="用户ID")
    query: str = Field(..., example="餐饮发票能开专票抵扣吗？",description="税务查询语句")
    session_id: Optional[str] = Field(None, description="会话ID")
    source_types: Optional[List[str]] = Field(None, description="知识来源筛选：default / user_upload / web_search；None 表示不限定")

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
def training(request: Request):
    """训练测试页面"""
    return render_page(request, "training.html")

# 新增的修改内容
@app.post("/api/chat")
async def tax_chat(body: TaxQueryRequest, current_user: CurrentUser = Depends(get_current_user)):
    """【同步对话接口】接收税务查询，经 TaxAgent 编排器路由多子 Agent 协作后一次性返回。

    TaxAgent.run() 内部已完成：意图路由 → 多 Agent 调度 → 汇总 → 长记忆注入与记录。
    """
    logger.info(f"[同步请求] 用户={current_user.user_name}(id={current_user.id}) query={body.query[:30]}...")

    try:
        session_id = body.session_id or f"session_{int(time.time())}"
        answer = await tax_agent.run(
            body.query,
            context={
                "thread_id": session_id,
                "uid": current_user.id,
                "source_types": body.source_types,
            },
        )

        return TaxQueryResponse(
            session_id=session_id,
            answer=answer or "（Agent 未返回有效内容）",
            references=[],
            status="success",
        )

    except Exception as e:
        logger.error(f"[同步请求错误] {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"查询失败: {str(e)}",
        )

# 新增的修改内容
# -------------------- 流式对话接口（SSE 协议） --------------------
@app.post("/api/chat/stream")
async def tax_chat_stream(
    body: TaxQueryRequest,
    current_user: CurrentUser = Depends(get_current_user),
):
    """【流式对话接口】接收税务查询，实时推送 LLM 生成的 token 流（SSE 协议）

    架构链路：
        浏览器 EventSource
          ↓  (text/event-stream)
        FastAPI StreamingResponse
          ↓  (async generator)
        agent_to_sse() ← agents/streaming.py 封装层
          ↓  (LangGraph stream_mode="messages")
        knowledge_agent.stream_messages() ← 全局单例
          ↓  (LangGraph graph.astream())
        LangGraph Agent + RAG 检索 + LLM 调用
    """
    logger.info(f"[流式请求] 用户={current_user.user_name}(id={current_user.id}) query={body.query[:30]}...")

    from langchain_core.messages import HumanMessage
    from agents.streaming import agent_to_sse

    try:
        session_id = body.session_id or f"session_{int(time.time())}"
        input_context = {
            "thread_id": session_id,
            "user_id": current_user.id,
            "user_name": current_user.user_name,
            "source_types": body.source_types,
        }

        # 注入长记忆上下文
        enriched_query = body.query
        try:
            mem_ctx = conversation_memory.context_for(session_id, body.query)
            if mem_ctx.get("prompt_text"):
                enriched_query = (
                    f"[以下为该用户的历史对话背景，供回答时参考]\n"
                    f"{mem_ctx['prompt_text']}\n\n"
                    f"[用户当前问题]\n{body.query}"
                )
        except Exception as mem_err:
            logger.debug(f"[长记忆] 上下文注入失败: {mem_err}")

        messages = [HumanMessage(content=enriched_query)]

        import asyncio

        # 用于收集流式输出的完整回答文本
        collected_answer: list[str] = []

        async def sse_generator():
            nonlocal collected_answer
            try:
                async for sse_line in agent_to_sse(
                    knowledge_agent.stream_messages(messages, input_context=input_context)
                ):
                    # 收集 token 类型事件中的文本，用于记录长记忆
                    if '"type": "token"' in sse_line or 'event: token' in sse_line:
                        import re as _re
                        match = _re.search(r'"text":\s*"([^"]*)"', sse_line)
                        if match:
                            collected_answer.append(match.group(1))
                    yield sse_line
            except asyncio.CancelledError:
                logger.info("[流式] 客户端断开或服务关闭，SSE generator 已终止")
                return
            except Exception as e:
                logger.error(f"[流式] generator 异常: {e}")
                err_line = f"event: error\ndata: {json.dumps({'msg': str(e)})}\n\n"
                yield err_line
            finally:
                # 流式结束后记录本轮对话到长记忆
                full_answer = "".join(collected_answer)
                if full_answer.strip():
                    try:
                        conversation_memory.record_turn(
                            session_id=session_id,
                            user_message=body.query,
                            assistant_message=full_answer,
                        )
                    except Exception as mem_err:
                        logger.warning(f"[长记忆] 流式记录对话失败: {mem_err}")

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
def global_search_api(q: str = "", limit: int = 12):
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
    current_user: CurrentUser = Depends(get_current_user),
):
    """【文件入库接口】接收上传文件，调用 KnowledgeService 保存。

    admin 上传 → source_type="default"（发布者默认知识库）
    普通用户上传 → source_type="user_upload"（用户个人知识库）
    """
    try:
        source_type = "default" if current_user.role == "admin" else "user_upload"
        logger.info(f"[上传] 用户={current_user.user_name}, role={current_user.role}, source_type={source_type}")
        return await knowledge_service.ingest_file(file, source_type=source_type)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"文件上传失败: {str(e)}")


@app.get("/api/knowledge/files")
async def list_knowledge_files(current_user: CurrentUser = Depends(get_current_user)):
    """【已上传文件列表】从持久化元数据读取，刷新不丢失"""
    try:
        files = await knowledge_service.list_files()
        return {"success": True, "files": files, "total": len(files)}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"获取文件列表失败: {str(e)}")


@app.get("/api/knowledge/files/sql")
async def list_knowledge_files_sql(current_user: CurrentUser = Depends(get_current_user)):
    """【SQL 业务表文件列表】从 KnowledgeFiles + KnowledgeChunks 表读取"""
    try:
        files = await knowledge_service.list_files_from_sql()
        return {"success": True, "files": files, "total": len(files)}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"获取 SQL 文件列表失败: {str(e)}")


@app.delete("/api/knowledge/files/{file_id}")
async def delete_knowledge_file(file_id: str, current_user: CurrentUser = Depends(get_current_user)):
    """【删除文件】全链路清理：元数据 + Qdrant 向量 + 磁盘原文件"""
    try:
        result = await knowledge_service.delete_file(file_id)
        if result.get("status") == "error":
            raise HTTPException(status_code=404, detail=result.get("message"))
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"删除文件失败: {str(e)}")

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
    ok = conversation_memory.delete_session(session_id)
    if tax_agent is not None:
        try:
            await tax_agent.delete_thread(session_id)
        except Exception as e:
            logger.warning(f"[delete_memory] checkpointer 清理失败: {e}")
    return {"success": ok, "session_id": session_id}


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
        except Exception:
            pass
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
    ok = conversation_memory.delete_session(session_id)
    # 同时清理 LangGraph checkpointer 里的多轮历史（若已启用）
    if tax_agent is not None:
        try:
            await tax_agent.delete_thread(session_id)
        except Exception as e:
            logger.warning(f"[delete_conversation] checkpointer 清理失败: {e}")
    if not ok:
        raise HTTPException(status_code=404, detail=f"对话 {session_id} 不存在")
    return {"success": True, "session_id": session_id}


# ================================================================
# 政策自动更新 API
# ================================================================

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
        elif body.type == "http":
            if not body.urls:
                raise HTTPException(status_code=400, detail="type=http 时必须提供 urls")
            src = HttpCrawlSource(
                source_id=body.source_id or f"http_{int(time.time())}",
                urls=body.urls,
                category=body.category,
            )
        else:
            raise HTTPException(status_code=400, detail=f"不支持的政策源类型: {body.type}，仅支持 local / http")

        policy_updater.add_source(src)
        logger.info(f"[政策更新] 已添加政策源: type={body.type}, id={src.source_id}")
        return {
            "success": True,
            "source": {"type": body.type, "source_id": src.source_id},
        }
    except HTTPException:
        raise
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
    ok = policy_updater.remove_source(source_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"政策源不存在: {source_id}")
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
    }


class SettingsUpdateRequest(BaseModel):
    llm: dict | None = None
    embedding: dict | None = None


@app.post("/api/settings")
def update_settings(body: SettingsUpdateRequest, current_user: CurrentUser = Depends(get_current_user)):
    """保存模型配置到本地 config.json"""
    try:
        ConfigManager().save_config(body.model_dump(exclude_none=True))
        return {"success": True, "message": "配置已保存，下次对话生效"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"保存配置失败: {e}")

# —— 联网检索结果管理 ——

class WebSearchIngestRequest(BaseModel):
    query: str = Field(..., description="搜索关键词")
    limit: int = Field(default=5, description="返回条数，默认 5，最多 10")


@app.post("/api/knowledge/web-search/ingest")
async def ingest_web_search(body: WebSearchIngestRequest, current_user: CurrentUser = Depends(get_current_user)):
    """联网检索并入库。

    搜索结果将被分块、向量化、写入 Qdrant（source_type=web_search），
    后续 RAG 检索可命中。
    """
    try:
        from services.tools.web_search_tool import search_and_ingest_web
        result = search_and_ingest_web(query=body.query, limit=body.limit)
        return {"success": True, **result}
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
        result = do_delete(file_id=file_id)
        return {"success": True, **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除联网检索结果失败: {str(e)}")