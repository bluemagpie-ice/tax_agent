# 阶段实施记录 · 2026-11-05

本轮基于已有未提交改动增量实现，未重建正式数据库、未修改真实 Qdrant 集合，未修改税务计算规则。

## P0-A

已实现独立 Research 入口、任务 ID、业务事实/未知条件、政策检索、证据快照、结构化报告及用户确认。
新增 research_tasks / research_reports，复用业务 SQLite 和 AsyncSqliteSaver，以 research 线程前缀隔离普通问答。
政策更新、KB 入库和工具返回链路补齐来源 URL、发布日期、地区等元数据，未知效力不填“有效”。
关键文件：agents/research_agent.py、services/research_store.py、services/research_evidence.py、web/research_routes.py、
web/static/research.js、knowledge/policy_updater.py、knowledge/kb.py、services/tools/kb_tools.py。
首轮针对研究、原 Web API、政策检索及政策更新的隔离回归：86 passed。
真实外部验收尚不在该数字内。

## P0-B

已实现最多两轮追问、保留未知继续、单问题轻量计划、四步依赖检查、三轮检索/一次重规划、
无新增证据停止、工具/模型/保守 token 预算、报告版本修改/退回/确认和恢复。
模型无效 JSON 允许一次预算内结构修复；规划仍失败时回退为已有子问题检索。
报告引用或合规失败只允许一次修正，再失败即停止发布，不把校验通过当作法律适用性证明。
共用原增值税计算工具的显式参数一般纳税人路径；其他静态政策默认值未开放为研究依据。
关键状态：awaiting_facts / running / awaiting_review / finalized / failed / interrupted。
新增接口以 task revision + report_version 校验审核请求；过期请求返回 409。

## P1

已实现 research_cases / research_case_versions / research_case_events，支持人工录入、保存确认、发布、修改、
撤回、删除和专业复核记录。用户确认保存与专业复核独立；专业复核仅记录用户声明，不验证资质。
仅当前用户、当前发布版本、已确认保存案例参与检索。修改即撤销旧版本当前可检索资格。
当前使用 SQLite 轻量关键词候选、结构化差异和政策线索比较，未引入 Qdrant 案例索引。

评测实现 A/B/C 执行器、独立数据/案例/政策快照、重复运行、运行级 CSV、完整 JSONL、空白人工评分表、
Markdown 摘要和按任务/案例家族汇总的探索性配对分析。失败调用、模型用量缺失、潜在泄漏均记录。
源码哈希、依赖版本和模型参数在后续实验 manifest 中保留。
数据包含 18 个正式评测候选题及 6 个调试题；全部人工核验状态仍为 pending，没有生成假标准答案。

实际替身执行：18 × A/B/C = 54 条流程记录。该结果仅验收流程、隔离和导出，不支持税务准确性结论。
最终替身导出目录：data/research_eval/2026-11-05T13-33-36.375247+00-00-352e4149，54 条全部完成流程，包含源码哈希和依赖版本。

真实模型调试使用公开模拟题和冻结官方节选，没有访问正式 Qdrant：

- 沙箱内 A/B/C 连通性尝试均连接失败，保留原始失败记录。
- 沙箱外最初尝试出现超时；随后复用普通问答的 GLM-5.3 low 推理参数。
- B/C 在 .test_runs/live-low-reasoning/2026-11-05T13-20-30.303657+00-00-e72cebd5 中生成待审核报告。
- A 的早期尝试分别遇到引用检查失败和规划 JSON 结构错误；这些失败没有从记录中删除。
- 加入结构修复及规划回退后，A 在 .test_runs/live-a-structured/2026-11-05T13-28-49.983535+00-00-a86cee8a 中生成待审核报告。

这些成功来自不同调试轮次/代码状态，不能拼接为正式配对比较，更未完成人工税务评分。
网络、时延和模型输出仍会波动；不宣称已证明案例增益或 Agent 优于 SOP。

## P2

已实现从定稿报告直接映射候选案例、泛化预览、字段修改、删除敏感内容、取消或确认保存发布。
不额外调用模型，不重复要求用户录入已有报告字段；不自动发布模型输出。
同一报告版本重复保存使用稳定案例 ID，避免重复创建。修改后的案例需重新确认发布。
泛化仅覆盖部分模式，不是完整匿名化保证；政策原文及法定数值不会被金额泛化规则静默改写。

闭环测试已覆盖“首次研究 → 定稿 → 候选案例 → 确认发布 → 第二次研究 → 召回比较 → 再检索政策”。
这是临时 SQLite 与脚本模型的集成测试，不是真实专业咨询验收。

## 验证结果与未完成项

```powershell
.\.venv\Scripts\python.exe scripts/run_tests.py tests/ -q --ignore=tests/test_api_connectivity.py --ignore=tests/test_files_embedding.py --disable-warnings --tb=short
node tests/test_conversation_ui.js
node tests/test_research_ui.js
node --check web/static/research.js
```

隔离回归：405 passed，41 warnings。随后新增/调整模块的最终定向回归：19 passed；已有聊天六项并发检查通过，
Research DOM 替身交互检查通过，JavaScript 语法检查通过。
完整外部连通性测试曾实际运行并出现 5 项连接失败；不把未运行的真实文件向量写入测试记为通过。

UI strict 静态审计目前仅剩 web/templates/training.html:325 的既有空链接；未顺便改动政策管理页面。
审计结果在 premium-audit.json。Computer Use 因无法可靠确认 Windows 浏览器 URL 而中止，
没有完成真实浏览器、窄屏、键盘和可访问性验收。DOM 替身不能替代这些项目。

下一步依赖：恢复可验证的浏览器通道、人工税务核验候选任务和参考判断、扩充政策快照、在固定代码/模型版本下
对同一批任务正式重复运行 A/B/C，并完成独立人工评分。暂不增加案例向量索引、团队权限或完整税法时序系统。

完整操作/API/数据模型见 RESEARCH.md；数据导出和评分见 ../eval/research/README.md。
