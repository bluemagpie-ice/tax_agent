# 税务 Agent（TaxAgent）

一个面向中文财税场景的 **多 Agent + RAG 智能问答系统**：接收用户税务咨询，自动识别意图、调度多个专业子 Agent 协作，结合本地知识库检索与联网实时检索给出带依据的回答，并具备跨会话记忆与多租户隔离。

## 技术栈

- **Web 层**：FastAPI + Jinja2（SSE 流式对话）
- **编排**：LangGraph（StateGraph + SQLite checkpointer 多轮持久化）
- **LLM**：智谱 GLM（`langchain-openai` 兼容接口）
- **向量库**：Qdrant（云端）+ BM25 词法 + Neo4j 图（可选），RRF 融合
- **Embedding**：SiliconFlow（`BAAI/bge-large-zh-v1.5`，1024 维）
- **存储**：SQLite（SQLAlchemy 业务表 + 原生键值存储层）
- **文档解析/OCR**：Docling、PyPDF、PaddleOCR
- **联网检索**：DuckDuckGo（财政部/税务总局定向）

## 核心能力

1. **意图路由 + 多 Agent 协作**：关键词 + n-gram 余弦相似度的混合路由，把问题分派给知识问答/筹划/计算/发票/合规/申报 6 类子 Agent，再由编排器用 LLM 汇总。
2. **RAG 检索**：文档解析 → 分块（含法律条文结构化切分）→ 向量化 → 多路召回（语义 + 词法 + 图）→ RRF 融合，回答带标题/文号/发布机关/生效日期等引用溯源。
3. **精确税务计算**：增值税/企业所得税/个人所得税/附加税费/印花税由纯函数硬编码公式计算（非 LLM 脑补），LLM 只做参数提取与解释。
4. **跨会话记忆**：工作记忆 + 会话摘要压缩 + 用户画像（跨会话聚合），按租户物理隔离，乐观锁防并发覆盖。
5. **时效性触发联网检索**：用户问「最新/最近政策」时自动触发联网实时检索并注入回答，而非只查本地库。
6. **政策自动更新**：本地目录 / 国家税务总局法规库爬取 → 去重 → 增量入库，后台定时扫描。
7. **离线评估框架**：内置 22 条评测集，量化意图准确率 / 关键词召回率 / 拒答正确率。
8. **增量索引**：上传新文档只向量化新增块，不重算全库。

## 快速开始

```bash
# 1. 配置环境变量（复制 .env.example 为 .env 并填写密钥）
#    LLM / Embedding / Qdrant 三组配置

# 2. 启动服务
.venv/Scripts/python.exe -m uvicorn web.main:app --reload
# 打开 http://127.0.0.1:8000

# 3. 运行测试（232 通过 + 1 集成测试默认跳过）
.venv/Scripts/python.exe -m pytest tests/ -q --ignore=_snapshot_20261006_164141

# 4. 离线评估（意图路由，无需 LLM/Qdrant）
.venv/Scripts/python.exe -m eval.run_eval --mode intent
```

## 目录结构

```
agents/           Agent 编排（tax_agent 编排器、intent_router、memory、sub_agents/*）
services/         计算器（calculators/*）、工具（tools/*）、配置/租户/存储
knowledge/        分块、embedding、知识库、政策更新、文档解析
database/         SQLAlchemy 模型与 Repository
web/              FastAPI 入口 + 前端模板
eval/             评估框架（数据集 + 指标 + 入口）
tests/            测试（pytest）
```

## 配置与鉴权

- 模型/Embedding 配置：前端设置页写入 `data/config.json`，回退 `.env`。
- 鉴权：请求头 `Authorization: Bearer <token>`。
  - `AUTH_USER_TOKEN`：普通用户（默认 `test_token_123`）
  - `DEVELOPER_TOKEN`：开发者，可看到设置/政策面板（回退 `AUTH_ADMIN_TOKEN` / `admin_token_456`）
  - 仅演示用途，生产环境请替换为真实鉴权。

## 已知待办（README 版）

- SQLAlchemy 业务表缺少 schema 迁移（Alembic）。
- 记忆无 PII 脱敏。
- 政策变更的文本级 diff 尚未真正实现（需存旧快照）。
- `web/templates/training.html` 政策面板代码被复制了三份，待去重。
