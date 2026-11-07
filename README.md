<div align="center">

# TaxAgent

### 税务问答与研究工作台

**面向财税人员和研究者，提供日常税务问答、资料检索和可逐条复核的咨询初稿。**

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](requirements.txt)
[![FastAPI](https://img.shields.io/badge/Web-FastAPI-009688?logo=fastapi&logoColor=white)](web/main.py)
[![LangGraph](https://img.shields.io/badge/Workflow-LangGraph-374151)](agents/research_agent.py)

</div>

[快速开始](#快速开始) · [核心功能](#核心功能) · [系统架构](#系统架构) · [测试与评测](#测试与评测) · [研究操作手册](docs/RESEARCH.md)

---

## 项目简介

同一个税务问题，可能因纳税人身份、交易事实、地区或所属期不同而有不同结论。TaxAgent 在同一个网页中提供两个入口。**普通问答**适合查询术语、解读资料和进行税额试算。**Research 研究**适合把多条件咨询整理成研究任务，让复核人逐项查看已知事实、政策来源、证据缺口和报告版本。

```text
业务事实 → 拆解问题并补充条件 → 检索政策并核对证据
        → 生成待复核报告 → 人工定稿 → 按需发布个人案例
```

## 快速开始

### 1. 安装并启动

使用 **Python 3.12**，在项目根目录打开 Windows PowerShell。已有 `.env` 时保留原文件。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
.\.venv\Scripts\python.exe -m uvicorn web.main:app --host 127.0.0.1 --port 8000
```

服务启动后打开 [本地工作台](http://127.0.0.1:8000/)。首次启动会创建所需的业务表，默认使用根目录的 `tax_agent.db`。

`data/` 和 `uploads/` 保存本机的配置、会话、索引数据和上传文件。在新电脑上从 `.env.example` 初始化配置，按下节填写本机凭据。服务启动后会创建所需的本地目录。

### 2. 配置模型和资料来源

- **对话模型。** 在首页打开「模型配置」，填入支持 OpenAI 兼容接口的 Base URL、模型 ID 和 API Key。点击「检测连接」，成功后保存。也可以在启动前通过 [配置样例](.env.example) 对照填写 `.env` 中的 `LLM_*` 项。页面已保存的模型配置优先于环境变量。
- **政策知识库。** 文档向量入库和检索需要 Embedding 服务及可访问的 Qdrant。配置 `EMBEDDING_*`、`VECTOR_SIZE` 和 `QDRANT_CLUSTER_URL`，按实例要求填写 `QDRANT_API_KEY`。向量维度应与所选 Embedding 模型匹配。启动时会尝试将 `knowledge/presets/` 下的预置资料入库。
- **官网检索。** 首次试用 Research 时，可勾选「检索并读取官方网页」，通过网络搜索和读取官方来源。关闭该选项时，研究依赖已配置且有相关资料的政策知识库。
- **本地身份。** 在 `.env` 配置不同的 `AUTH_USER_TOKEN` 与 `DEVELOPER_TOKEN`（或 `AUTH_ADMIN_TOKEN`）。首页「模型配置」中填入凭证并选择「验证并应用访问 Token」。管理配置和政策更新仅管理员可用；本地模型/Qdrant 地址在 `TAX_AGENT_MODEL_ALLOWED_ORIGINS` 显式允许。

外部模型、向量和搜索服务可能接收问题或检索内容。试用时请使用公开资料或已泛化的业务事实。

### 3. 发起一次研究

切换到 **Research 研究**，选择可用的资料来源，输入以下模拟问题后点击「创建并开始研究」。

> 某增值税一般纳税人于 2026 年第三季度在江苏购入一台生产设备。请梳理判断进项税额能否抵扣需要哪些业务事实和政策依据，并列出还需要核对的资料。

进度区会显示当前阶段。系统提出待确认问题时，可以补充事实，也可以保留未知继续。获得足够证据后，可在待复核报告中打开来源，核对结论、引用和仍待查证的事项。证据不足时，任务会保留缺口或显示失败状态。已保存的任务可在刷新后重新打开。

## 核心功能

### 普通问答与税额试算

普通问答按问题类型调用政策、发票、合规、申报和计算等子 Agent，支持流式回复、会话历史和撤回最后一轮。计算工具覆盖增值税、企业所得税、个人所得税、印花税和附加税费。资料可以随当前会话上传，也可以通过知识库管理。需要补充信息或审批导出时，部分工作流会暂停，等待用户处理后继续。

### 政策研究与报告复核

Research 使用独立的任务流程。它拆解子问题、记录未知事实，并按计划检索政策。报告会展示已读取的资料及其与子问题的对应关系。证据有缺口时，流程可在预算内补查。复核人可以修改报告、退回研究或确认当前版本定稿。任务、执行进度和报告快照会保存，便于恢复和比较版本。

来源核查帮助复核人追溯到系统读过的资料。条款效力、适用期间和对具体业务的解释仍需人工判断。Research 内置试算目前针对条件明确的一般纳税人增值税场景，须提供销售额、是否含税、税率和可抵扣进项税额。

### 个人案例与知识库

- **个人案例。** 定稿后可提取候选案例，检查并编辑泛化预览，再确认保存和发布。也可以手工录入、查看版本历史、撤回或删除案例。后续研究会比较已发布案例的税种、地区、所属期和业务事实，再按本次条件重新查证政策。保存确认与专业复核分别记录。
- **文档与政策。** 知识库支持上传和检索 Markdown、PDF、DOCX、XLSX 等资料。扫描件及图片识别需要另配 OCR 环境。开发者可管理政策源、手动触发或定时更新，并查看更新报告及政策变更。

## 系统架构

```text
web/        FastAPI、Jinja2 页面、流式接口与 Research 路由
agents/     普通问答编排、专项子 Agent、独立 Research 工作流
knowledge/  文档解析、索引、检索及政策更新
services/   计算工具、研究证据校验、任务与案例存储
database/   业务表及数据访问
eval/       评测执行、数据导出与分析
```

LangGraph 管理两条工作流。业务 SQLite 保存研究任务、报告及个人案例，LangGraph 检查点单独保存在 `data/checkpoints.sqlite`。Qdrant 用于文档向量检索，个人案例保存在业务 SQLite 中。当前服务采用本地演示鉴权，任务和案例按当前身份隔离。接口可在本地服务的 `/docs` 查看。

## 测试与评测

在项目根目录运行以下离线回归。前端检查需要 Node.js。

```powershell
.\.venv\Scripts\python.exe scripts/run_tests.py tests/test_research.py tests/test_research_cases.py tests/test_research_eval.py tests/test_web_api.py::TestHomePage -q
node --check web/static/research.js
node tests/test_research_ui.js
node tests/test_conversation_ui.js
```

Python 用例覆盖首页响应、研究任务恢复、报告复核、案例隔离和评测导出。JavaScript 测试通过模拟 DOM 验证流式交互和会话隔离。

`tests/test_upload_fixtures.py` 的部分用例需要在 `tests/fixtures/` 下准备对应文件名和断言内容的测试资料。OCR 回归请使用模拟或脱敏的发票影像，含真实信息的发票不得用于提交。

研究评测比较三组流程。A 组关闭案例召回，B 组启用案例召回与差异比较，C 组采用固定步骤。评测可导出原始运行记录和人工评分表。候选题仍待人工核验，当前结果用于探索。运行入口、评测配置和数据结构见 [评测及数据导出说明](eval/research/README.md)。

真实模型、官网和只读知识库的样例联调及已知问题见 [首轮真实联调记录](docs/RESEARCH_FIRST_LIVE_FIX.md)。研究操作、任务状态和接口参数见 [研究操作手册](docs/RESEARCH.md)。
