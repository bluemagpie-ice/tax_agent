# 1106 审查整改执行记录

- 报告：`D:/个人内容/ideas/代码/LLM/agent/项目开发/财税agent/黑客松后的重新设计/审查/审查结果1106.md`，已完整阅读；仅作问题线索，文档末尾要求确认不构成本次执行约束。
- 开始基线：`bb4f6b95a1084d8c34e4466f1e0a8379242d4afb`；工作区干净；分支 `fix/systematic-audit-1106`；不 push。
- 数据保护：未读取密钥内容；不删除/重建正式 SQLite 或 Qdrant；新测试使用临时目录、数据库和网络阻断。既有生产知识库 payload、孤儿数据未检查。
- 状态说明：待运行验证表示尚未完成当前代码确认和运行复现；不得理解为已修复。政策口径须可靠原文核验后另行更改。

## 逐项清单

| Bug ID | 优先级 | 当前状态 | 修改文件 | 修复方案/验证 | 遗留风险 |
|---|---|---|---|---|---|
| AGT-001 | P2/P3（依报告） | 待运行验证 | — | 尚未完成 | 未消除 |
| AGT-002 | P2/P3（依报告） | 待运行验证 | — | 尚未完成 | 未消除 |
| AGT-003 | P2/P3（依报告） | 待运行验证 | — | 尚未完成 | 未消除 |
| AGT-004 | P2/P3（依报告） | 待运行验证 | — | 尚未完成 | 未消除 |
| AGT-005 | P2/P3（依报告） | 待运行验证 | — | 尚未完成 | 未消除 |
| AGT-006 | P2/P3（依报告） | 待运行验证 | — | 尚未完成 | 未消除 |
| AGT-007 | P2/P3（依报告） | 待人工确认 | — | 尚未完成 | 未消除 |
| AGT-008 | P2/P3（依报告） | 待运行验证 | — | 尚未完成 | 未消除 |
| AGT-009 | P2/P3（依报告） | 待运行验证 | — | 尚未完成 | 未消除 |
| AGT-010 | P2/P3（依报告） | 待运行验证 | — | 尚未完成 | 未消除 |
| AGT-011 | P2/P3（依报告） | 待运行验证 | — | 尚未完成 | 未消除 |
| AGT-012 | P2/P3（依报告） | 待人工确认 | — | 尚未完成 | 未消除 |
| AGT-013 | P2/P3（依报告） | 待运行验证 | — | 尚未完成 | 未消除 |
| AGT-014 | P2/P3（依报告） | 待运行验证 | — | 尚未完成 | 未消除 |
| AGT-015 | P2/P3（依报告） | 待运行验证 | — | 尚未完成 | 未消除 |
| AGT-016 | P2/P3（依报告） | 待运行验证 | — | 尚未完成 | 未消除 |
| AGT-017 | P2/P3（依报告） | 待运行验证 | — | 尚未完成 | 未消除 |
| AGT-018 | P2/P3（依报告） | 待运行验证 | — | 尚未完成 | 未消除 |
| AGT-019 | P2/P3（依报告） | 待运行验证 | — | 尚未完成 | 未消除 |
| AGT-020 | P2/P3（依报告） | 待运行验证 | — | 尚未完成 | 未消除 |
| AGT-021 | P2/P3（依报告） | 待人工确认 | — | 尚未完成 | 未消除 |
| DB-1 | P2/P3（依报告） | 待运行验证 | — | 尚未完成 | 未消除 |
| DB-10 | P2/P3（依报告） | 待运行验证 | — | 尚未完成 | 未消除 |
| DB-11 | P2/P3（依报告） | 待运行验证 | — | 尚未完成 | 未消除 |
| DB-12 | P2/P3（依报告） | 待运行验证 | — | 尚未完成 | 未消除 |
| DB-13 | P2/P3（依报告） | 待运行验证 | — | 尚未完成 | 未消除 |
| DB-14 | P2/P3（依报告） | 待运行验证 | — | 尚未完成 | 未消除 |
| DB-2 | P2/P3（依报告） | 待运行验证 | — | 尚未完成 | 未消除 |
| DB-3 | P2/P3（依报告） | 待运行验证 | — | 尚未完成 | 未消除 |
| DB-4 | P2/P3（依报告） | 待运行验证 | database/models.py; services/paths.py | 默认业务库固定到项目根；保留 DATABASE_URL 覆盖；阶段1回归 56 passed | 未合并磁盘上的历史重复库 |
| DB-5 | P2/P3（依报告） | 待运行验证 | — | 尚未完成 | 未消除 |
| DB-6 | P2/P3（依报告） | 待运行验证 | — | 尚未完成 | 未消除 |
| DB-7 | P2/P3（依报告） | 待运行验证 | — | 尚未完成 | 未消除 |
| DB-8 | P2/P3（依报告） | 待运行验证 | — | 尚未完成 | 未消除 |
| DB-9 | P2/P3（依报告） | 待运行验证 | — | 尚未完成 | 未消除 |
| EVAL-1 | P2/P3（依报告） | 待运行验证 | — | 尚未完成 | 未消除 |
| EVAL-2 | P2/P3（依报告） | 待运行验证 | — | 尚未完成 | 未消除 |
| EVAL-3 | P2/P3（依报告） | 待运行验证 | — | 尚未完成 | 未消除 |
| EVAL-4 | P2/P3（依报告） | 待运行验证 | — | 尚未完成 | 未消除 |
| EVAL-5 | P2/P3（依报告） | 待运行验证 | — | 尚未完成 | 未消除 |
| EVAL-7 | P2/P3（依报告） | 待运行验证 | tests/test_research_eval.py | 使用跟踪的 config.example.json；阶段1回归 56 passed | 完整评测尚未执行 |
| P0-1 | P0 | 已确认并修复 | web/main.py; services/network_security.py | 管理配置/政策读写统一 require_admin；本地来源限制目录；出站地址阻止私网和重定向，2MB 限制 | DNS 校验与连接之间仍有 rebinding 窗口，未宣称完全消除 SSRF |
| P0-2 | P0 | 已确认并修复 | agents/memory.py; tests/test_memory.py; tests/test_audit_data_safety.py | 完整 messages + compressed_count；撤回重建摘要；上下文预算保持；阶段1回归 56 passed | 旧版已丢失历史无法恢复 |
| P0-3 | P0 | 已确认并修复 | services/calculators/individual_income.py; tests/test_audit_data_safety.py | 统一退补方向；3 个正负用例；阶段1回归 56 passed | 未调整任何税率或政策 |
| P0-4 | P0 | 已确认并修复 | tests/test_database_init.py; tests/test_audit_data_safety.py | 删除正式库 unlink；真实初始化器内存幂等测试；阶段1回归 56 passed | 无 |
| P1-1 | P1 | 已确认并修复 | web/main.py; web/templates/*.html; .env.example | 默认凭证拒绝；页面不内置 Token；CORS 白名单；缺头401 | 仍为两个显式配置身份，不是多用户注册系统 |
| P1-10 | P1 | 待运行验证 | — | 尚未完成 | 未消除 |
| P1-11 | P1 | 待运行验证 | — | 尚未完成 | 未消除 |
| P1-12 | P1 | 已确认并修复 | web/templates/index.html; tests/test_conversation_ui.js | 401 不切身份或 Token；会话存储按角色；独立访问凭证恢复按钮 | 需真实浏览器验收 |
| P1-2 | P1 | 待运行验证 | — | 尚未完成 | 未消除 |
| P1-3 | P1 | 待运行验证 | — | 尚未完成 | 未消除 |
| P1-4 | P1 | 已确认但暂未修复 | agents/research_agent.py | Research 入选优先关联题目及分数；普通聊天 RAG 日期排序待阶段4 | 尚未消除全链路风险 |
| P1-5 | P1 | 已确认并修复 | services/policy_validity.py; agents/research_agent.py | 期间区间判断，起始无终止标待核对；文号年份不通过 | 不做法律效力自动保证；区间解析只覆盖显式年份/日期 |
| P1-6 | P1 | 已确认并修复 | agents/research_agent.py; agents/streaming.py | 相关度排序、选集变更才评估、保留已充分题目、裁剪警告、Research 独立窗口环境变量 | 真实模型相关度与小窗口验收待执行 |
| P1-7 | P1 | 已确认并修复 | agents/research_agent.py; services/security.py | 预留报告与一次修正调用及token；按question_id；具体错误；敏感字段脱敏；有据非申报分析不误拦 | 联合文号及全报告字段脱敏还需专项覆盖 |
| P1-8 | P1 | 已确认并修复 | agents/research_agent.py; web/static/research.js | 瞬时模型/结构失败保留状态从失败节点恢复；其他失败明确从头执行 | 预算耗尽须重新研究；取消的线程调用不能强杀 |
| P1-9 | P1 | 待运行验证 | — | 尚未完成 | 未消除 |
| RAG-001 | P2/P3（依报告） | 待运行验证 | — | 尚未完成 | 未消除 |
| RAG-002 | P2/P3（依报告） | 待运行验证 | — | 尚未完成 | 未消除 |
| RAG-003 | P2/P3（依报告） | 待运行验证 | — | 尚未完成 | 未消除 |
| RAG-004 | P2/P3（依报告） | 待运行验证 | — | 尚未完成 | 未消除 |
| RAG-005 | P2/P3（依报告） | 待运行验证 | — | 尚未完成 | 未消除 |
| RAG-006 | P2/P3（依报告） | 待运行验证 | — | 尚未完成 | 未消除 |
| RAG-007 | P2/P3（依报告） | 待运行验证 | — | 尚未完成 | 未消除 |
| RAG-008 | P2/P3（依报告） | 待运行验证 | — | 尚未完成 | 未消除 |
| RAG-009 | P2/P3（依报告） | 待运行验证 | — | 尚未完成 | 未消除 |
| RAG-010 | P2/P3（依报告） | 待运行验证 | — | 尚未完成 | 未消除 |
| RAG-011 | P2/P3（依报告） | 待运行验证 | — | 尚未完成 | 未消除 |
| RAG-012 | P2/P3（依报告） | 待运行验证 | — | 尚未完成 | 未消除 |
| RAG-013 | P2/P3（依报告） | 待运行验证 | — | 尚未完成 | 未消除 |
| RAG-014 | P2/P3（依报告） | 待人工确认 | — | 尚未完成 | 未消除 |
| RAG-015 | P2/P3（依报告） | 待运行验证 | — | 尚未完成 | 未消除 |
| RAG-016 | P2/P3（依报告） | 待运行验证 | — | 尚未完成 | 未消除 |
| RAG-017 | P2/P3（依报告） | 待运行验证 | — | 尚未完成 | 未消除 |
| RAG-018 | P2/P3（依报告） | 待运行验证 | — | 尚未完成 | 未消除 |
| RAG-019 | P2/P3（依报告） | 待运行验证 | — | 尚未完成 | 未消除 |
| RES-001 | P2/P3（依报告） | 待运行验证 | — | 尚未完成 | 未消除 |
| RES-002 | P2/P3（依报告） | 待运行验证 | — | 尚未完成 | 未消除 |
| RES-003 | P2/P3（依报告） | 待运行验证 | — | 尚未完成 | 未消除 |
| RES-004 | P2/P3（依报告） | 待运行验证 | — | 尚未完成 | 未消除 |
| RES-005 | P2/P3（依报告） | 待运行验证 | — | 尚未完成 | 未消除 |
| RES-006 | P2/P3（依报告） | 待运行验证 | — | 尚未完成 | 未消除 |
| RES-007 | P2/P3（依报告） | 待运行验证 | — | 尚未完成 | 未消除 |
| RES-008 | P2/P3（依报告） | 待运行验证 | — | 尚未完成 | 未消除 |
| RES-009 | P2/P3（依报告） | 已确认并修复 | agents/research_agent.py | unknown_kinds分类，补充事实累加，无补充skip不重复调用understand | 模型事实分类仍需真实验收 |
| RES-010 | P2/P3（依报告） | 已确认并修复 | agents/research_agent.py | 可选计算失败降级警告；重新研究清理旧计算和告警 | 纯计算工具异常降级还需扩展 |
| RES-011 | P2/P3（依报告） | 已确认并修复 | agents/research_agent.py | 修改报告保留case_references，只绑定报告引用的政策证据 | 人工改稿需人工承担事实审核 |
| RES-012 | P2/P3（依报告） | 待运行验证 | — | 尚未完成 | 未消除 |
| RES-013 | P2/P3（依报告） | 已确认并修复 | agents/research_agent.py | 本地检索优先且网页预算留出本地检索次数 | 极小总预算仍不能完成所有工具 |
| RES-014 | P2/P3（依报告） | 已确认但暂未修复 | web/static/research.js | 已修运行计时、真实阶段与心跳状态；SQLite线程池/摘要查询优化待做 | 同步DB在事件循环及线程池竞争仍存在 |
| RES-015 | P3 | 已确认并修复 | web/static/research.js; tests/test_research_ui.js | SSE error/停止刷新，running显示中文阶段 | 真实浏览器断线/停止待验收 |
| RES-016 | P2/P3（依报告） | 待运行验证 | — | 尚未完成 | 未消除 |
| RES-017 | P2/P3（依报告） | 已确认但暂未修复 | agents/research_agent.py | 计划重复编号与未知依赖规范化，结构错误带字段反馈 | 超长模型列表尚无裁剪降级 |
| RES-018 | P3 | 待运行验证 | — | 尚未完成 | 未消除 |
| TEST-1 | P2/P3（依报告） | 已确认并修复 | tests/conftest.py; services/paths.py; database/models.py | collection 前隔离；每测试替换并恢复 engine；禁止外网；阶段1回归 56 passed | live 测试仍需独立环境 |
| TEST-2 | P2/P3（依报告） | 已确认并修复 | tests/conftest.py; tests/test_*.py | 删除导入时 chdir；隔离默认配置、上传和 checkpoint 路径；阶段1回归 56 passed | 少数显式传参由测试自行隔离 |
| TEST-3 | P2/P3（依报告） | 已确认并修复 | tests/test_database_init.py; database/init_db.py | 实跑初始化逻辑；修复 insert_initial_data 成功无返回；阶段1回归 56 passed | 示例数据政策内容未核验 |
| TEST-4 | P2/P3（依报告） | 待运行验证 | — | 尚未完成 | 未消除 |
| TEST-5 | P2/P3（依报告） | 待运行验证 | — | 尚未完成 | 未消除 |
| TEST-6 | P2/P3（依报告） | 待运行验证 | — | 尚未完成 | 未消除 |
| TEST-7 | P2/P3（依报告） | 待运行验证 | — | 尚未完成 | 未消除 |
| WEB-001 | P2/P3（依报告） | 待运行验证 | — | 尚未完成 | 未消除 |
| WEB-002 | P2/P3（依报告） | 待运行验证 | — | 尚未完成 | 未消除 |
| WEB-003 | P2/P3（依报告） | 待运行验证 | — | 尚未完成 | 未消除 |
| WEB-004 | P2/P3（依报告） | 待运行验证 | — | 尚未完成 | 未消除 |
| WEB-005 | P2/P3（依报告） | 已确认并修复 | services/network_security.py; services/config_manager.py | 本地模型 origin 由服务端允许；公共模型需 HTTPS 和全局 IP | DNS rebinding 仍待处理；既有配置未改写 |
| WEB-006 | P2/P3（依报告） | 已确认并修复 | web/templates/index.html | 令牌输入在配置读取失败时保持可用，独立验证后重新读取 | 需真实浏览器验收 |
| WEB-007 | P2/P3（依报告） | 待运行验证 | — | 尚未完成 | 未消除 |
| WEB-008 | P2/P3（依报告） | 待运行验证 | — | 尚未完成 | 未消除 |
| WEB-009 | P2/P3（依报告） | 已确认并修复 | web/templates/index.html | /me 错误不改身份，按角色存 session_id | 历史共享 key 不再自动使用，旧会话仍在列表 |
| WEB-010 | P2/P3（依报告） | 待运行验证 | web/templates/index.html | 重新生成先撤回服务端，成功后再提交问题 | 尚需专项浏览器测试 |
| WEB-011 | P2/P3（依报告） | 待运行验证 | — | 尚未完成 | 未消除 |
| WEB-012 | P2/P3（依报告） | 待运行验证 | — | 尚未完成 | 未消除 |
| WEB-013 | P2/P3（依报告） | 已存在修复，无须重复修改 | web/main.py; services/config_manager.py; tests/test_settings_management.py | 保留 prepare/commit 客户端热更新；测试证明 chat/research 使用新 Key | MockTransport 不代表供应商验收 |
| WEB-014 | P2/P3（依报告） | 待运行验证 | — | 尚未完成 | 未消除 |
| WEB-015 | P2/P3（依报告） | 待运行验证 | — | 尚未完成 | 未消除 |
| WEB-016 | P2/P3（依报告） | 待运行验证 | — | 尚未完成 | 未消除 |
| WEB-017 | P2/P3（依报告） | 待运行验证 | — | 尚未完成 | 未消除 |

## 阶段记录

### 阶段 1（执行中）

P0-2/3/4 已确认代码问题；修改完成，正在运行隔离回归。旧版本已压缩丢失的消息无法凭摘要恢复；保留已有摘要与剩余消息，不伪造历史。

阶段 1 验证：`.venv/Scripts/python.exe scripts/run_tests.py tests/test_audit_data_safety.py tests/test_memory.py tests/test_database_init.py tests/test_repositories.py tests/test_conversation_retraction.py tests/test_context_window.py -q` → **56 passed, 0 failed, 0 skipped**。包含临时 SQLite 本地集成、FastAPI TestClient；外部模型与 Qdrant 未调用。初次运行网络阻断误拦 Windows asyncio socketpair（4 failed），已限定允许标准库自身 socketpair 后全通过。

兼容：messages 接口字段未改；旧 session 无 compressed_count 时从 0 开始，仅压缩未来上下文。默认数据路径不搬迁；可用 TAX_AGENT_DATA_DIR / TAX_AGENT_UPLOAD_DIR 指向独立实例。未执行任何生产迁移。

### 阶段 2（核心修复完成）

认证/配置/政策定向测试：63 passed，0 failed，0 skipped；Node 对话脚本通过，增加 `/me` 401 身份/凭证/会话不变断言。使用 Mock DNS、MockTransport、Mock HTTP/Qdrant 与临时配置，真实模型调用 0。旧测试以普通 Token 管理配置的断言已改为管理员，另增普通身份 403 回归。政策 HTTP Mock 现在包含状态码和流式字节。

本地开发：设置不同的 AUTH_USER_TOKEN 与 DEVELOPER_TOKEN（或 AUTH_ADMIN_TOKEN），首页设置中“验证并应用访问 Token”；本地服务在 TAX_AGENT_MODEL_ALLOWED_ORIGINS 指定完整 origin。未更改真实 .env/config.json。

阶段 1 checkpoint：c9daecb。

### 阶段 3（核心路径完成，性能与边界项仍有遗留）

`test_audit_research.py test_research.py test_research_metadata.py test_research_cases.py test_research_eval.py test_agent_safety.py`：44 passed。随后 Research/上下文定向36 passed。Node Research DOM脚本通过，新增碎片SSE、keepalive、error后刷新、42秒计时和恢复按钮断言。全部为脚本模型/Mock与临时SQLite，无真实税务结论或模型验收。包含充分/不足证据、期间不确定、超时恢复、错误引用修正、预算不足与取消的既有及新增回归。

旧测试把“2023起施行就不适用2026”当作预期，已改为明确 is_current=False 的失效证据，并新增开放起始期间 pending_verification 回归。

阶段2 checkpoint：e7b4ac7。RESEARCH_CONTEXT_WINDOW / RESEARCH_MAX_OUTPUT_TOKENS 可独立设置，默认沿用聊天预算；不擅自放大真实模型上下文。
