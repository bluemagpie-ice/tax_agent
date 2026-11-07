# 1106 审查整改执行记录

- 报告：`D:/个人内容/ideas/代码/LLM/agent/项目开发/财税agent/黑客松后的重新设计/审查/审查结果1106.md`，已完整阅读；仅作问题线索，文档末尾要求确认不构成本次执行约束。
- 开始基线：`bb4f6b95a1084d8c34e4466f1e0a8379242d4afb`；工作区干净；分支 `fix/systematic-audit-1106`；不 push。
- 数据保护：未读取密钥内容；不删除/重建正式 SQLite 或 Qdrant；新测试使用临时目录、数据库和网络阻断。既有生产知识库 payload、孤儿数据未检查。
- 状态说明：待运行验证表示尚未完成当前代码确认和运行复现；不得理解为已修复。政策口径须可靠原文核验后另行更改。

## 逐项清单

| Bug ID | 优先级 | 当前状态 | 修改文件 | 修复方案/验证 | 遗留风险 |
|---|---|---|---|---|---|
| AGT-001 | P0 | 已确认并修复 | agents/memory.py; tests/test_memory.py; tests/test_audit_data_safety.py | 完整 messages + compressed_count；撤回重建摘要；上下文预算保持；阶段1回归 56 passed | 旧版已丢失历史无法恢复 |
| AGT-002 | P0 | 已确认并修复 | services/calculators/individual_income.py; tests/test_audit_data_safety.py | 统一退补方向；3 个正负用例；阶段1回归 56 passed | 未调整任何税率或政策 |
| AGT-003 | P1 | 已确认并修复 | agents/tax_agent.py; agents/sub_agents/calculation_agent.py | 复算异常安全降级，税额正则不匹配应纳税所得额，计算参数统一schema | 未改变反思模型本身的判断能力 |
| AGT-004 | P1 | 已确认并修复 | agents/tax_agent.py; agents/sub_agents/calculation_agent.py | 复算异常安全降级，税额正则不匹配应纳税所得额，计算参数统一schema | 未改变反思模型本身的判断能力 |
| AGT-005 | P1 | 已确认并修复 | services/calculators/*; agents/sub_agents/calculation_agent.py | 未知税目失败、缺销售额不返回0、非法枚举拒绝 | 收入额为0必须由用户显式提供 |
| AGT-006 | P1 | 已确认并修复 | services/calculators/*; agents/sub_agents/calculation_agent.py | 未知税目失败、缺销售额不返回0、非法枚举拒绝 | 收入额为0必须由用户显式提供 |
| AGT-007 | P2 | 待人工确认 | — | 尚未完成 | 未消除 |
| AGT-008 | P2 | 已确认但暂未修复 | services/calculators | 含特定期间的静态政策口径未变更；需可靠法定依据和所属期契约 | 历史/未来政策不应直接验收 |
| AGT-009 | P2 | 已确认但暂未修复 | services/calculators/tools.py | 修正个税税率文字、印花税最低1元误导、增加other_deduction | 经营所得Tool和部分描述仍待完善 |
| AGT-010 | P2 | 已确认并修复 | services/calculators/tools.py | 有限数、非负数、税率0..1、月份1..12、2月后prior_income必填 | 负数冲减业务不属于当前简单计算器，应明确拒绝 |
| AGT-011 | P2 | 已确认并修复 | services/calculators/vat.py | 小规模缺季度销售额时提示未核验免征条件 | 税收口径未经本次核验 |
| AGT-012 | P2 | 待人工确认 | — | 尚未完成 | 未消除 |
| AGT-013 | P2 | 已确认但暂未修复 | agents/tax_agent.py | 委托thread加入独立run_id，避免跨轮复用 | 旧派生线程清理尚未实现 |
| AGT-014 | P2 | 已确认并修复 | agents/tax_agent.py | 专业Agent使用包含补充信息的original_query | 需真实多轮验收 |
| AGT-015 | P2 | 已确认并修复 | agents/memory.py | 年份边界修正；撤回/删除重建画像 | 已丢失旧历史不可复原，旧摘要信息有限 |
| AGT-016 | P2 | 已确认并修复 | agents/tax_agent.py; agents/memory.py | 反思失败保留草稿标未核验；失败助手消息exclude_from_context | 危险建议仍由安全校验拦截，不能保留发布 |
| AGT-017 | P2 | 待运行验证 | — | 尚未完成 | 未消除 |
| AGT-018 | P3 | 待运行验证 | — | 尚未完成 | 未消除 |
| AGT-019 | P3 | 待运行验证 | — | 尚未完成 | 未消除 |
| AGT-020 | P3 | 待运行验证 | — | 尚未完成 | 未消除 |
| AGT-021 | P3 | 待人工确认 | — | 尚未完成 | 未消除 |
| DB-1 | P0 | 已确认并修复 | tests/test_database_init.py; tests/test_audit_data_safety.py | 删除正式库 unlink；真实初始化器内存幂等测试；阶段1回归 56 passed | 无 |
| DB-10 | P3 | 待运行验证 | — | 尚未完成 | 未消除 |
| DB-11 | P3 | 待运行验证 | — | 尚未完成 | 未消除 |
| DB-12 | P3 | 待运行验证 | — | 尚未完成 | 未消除 |
| DB-13 | P3 | 待运行验证 | — | 尚未完成 | 未消除 |
| DB-14 | P3 | 待运行验证 | — | 尚未完成 | 未消除 |
| DB-2 | P2 | 已确认并修复 | web/main.py; knowledge/kb.py; knowledge/kb_manager.py | 名称白名单，API拒绝路径穿越，路径创建前二次校验 | 未改名已有库 |
| DB-3 | P3 | 已确认并修复 | database/init_db.py | 初始化插入成功显式返回True | 旧示例数据不涉及真实任务 |
| DB-4 | P2 | 已确认并修复 | database/models.py; services/paths.py | 默认业务库固定到项目根，保留DATABASE_URL覆盖 | 未合并旧库；显式相对URL仍按部署配置解释 |
| DB-5 | P2 | 已确认并修复 | database/models.py | SQLite连接busy_timeout=10s、WAL；40行并发测试 | 长事务仍可能等待或超时 |
| DB-6 | P2 | 已确认但暂未修复 | web/main.py | SQL清单沿用文件可见性过滤并剔除绝对路径 | 业务表owner列和迁移未实施 |
| DB-7 | P3 | 待运行验证 | — | 尚未完成 | 未消除 |
| DB-8 | P3 | 待运行验证 | — | 尚未完成 | 未消除 |
| DB-9 | P2 | 已确认并修复 | services/research_cases.py | 基于原版本payload更新复核，保留确认审计字段 | 并发重复create仍待专门处理 |
| EVAL-1 | P2 | 已确认并修复 | eval/run_eval.py; eval/metrics.py; tests/test_audit_eval.py | 执行异常单列并计零，不作正确拒答 | 未执行真实模型评测 |
| EVAL-2 | P2 | 已确认但暂未修复 | eval/metrics.py | 收窄拒答词、执行失败计零 | 计算数值基准和关键词设计仍需人工标注 |
| EVAL-3 | P2 | 已确认并修复 | eval/run_eval.py; agents/base_agent.py; knowledge/embedding.py | 独立租户/运行目录/记忆/checkpoint、唯一线程；Qdrant只读；真实评测须--live | 禁止生产评测回流 |
| EVAL-4 | P2 | 已确认但暂未修复 | eval/research_eval.py | 导出每题冻结快照匹配来源数 | 仍只有有限政策节选，缺官方逐题人工核验 |
| EVAL-5 | P2 | 已确认并修复 | eval/research/cases.json; eval/research_eval.py; tests/test_research_eval.py | 显式家族映射、重叠标记和未来案例截止回归 | 真实案例家族划分须人工复核 |
| EVAL-7 | P2 | 已确认并修复 | tests/test_research_eval.py | 跟踪的config.example.json；隔离冒烟测试通过 | 真实配置不读取 |
| P0-1 | P0 | 已确认并修复 | web/main.py; services/network_security.py | 管理配置/政策读写统一 require_admin；本地来源限制目录；出站地址阻止私网和重定向，2MB 限制 | DNS 校验与连接之间仍有 rebinding 窗口，未宣称完全消除 SSRF |
| P0-2 | P0 | 已确认并修复 | agents/memory.py; tests/test_memory.py; tests/test_audit_data_safety.py | 完整 messages + compressed_count；撤回重建摘要；上下文预算保持；阶段1回归 56 passed | 旧版已丢失历史无法恢复 |
| P0-3 | P0 | 已确认并修复 | services/calculators/individual_income.py; tests/test_audit_data_safety.py | 统一退补方向；3 个正负用例；阶段1回归 56 passed | 未调整任何税率或政策 |
| P0-4 | P0 | 已确认并修复 | tests/test_database_init.py; tests/test_audit_data_safety.py | 删除正式库 unlink；真实初始化器内存幂等测试；阶段1回归 56 passed | 无 |
| P1-1 | P1 | 已确认并修复 | web/main.py; web/templates/*.html; .env.example | 默认凭证拒绝；页面不内置 Token；CORS 白名单；缺头401 | 仍为两个显式配置身份，不是多用户注册系统 |
| P1-10 | P1 | 已确认并修复 | agents/tax_agent.py; agents/sub_agents/calculation_agent.py | 复算异常安全降级，税额正则不匹配应纳税所得额，计算参数统一schema | 未改变反思模型本身的判断能力 |
| P1-11 | P1 | 已确认并修复 | services/calculators/*; agents/sub_agents/calculation_agent.py | 未知税目失败、缺销售额不返回0、非法枚举拒绝 | 收入额为0必须由用户显式提供 |
| P1-12 | P1 | 已确认并修复 | web/templates/index.html; tests/test_conversation_ui.js | 401 不切身份或 Token；会话存储按角色；独立访问凭证恢复按钮 | 需真实浏览器验收 |
| P1-2 | P1 | 已确认并修复 | services/policy_metadata.py; knowledge/kb.py; services/tools/kb_tools.py | 显式元数据抽取+未知效力；历史缺字段不硬过滤；Research透传状态 | 生产历史payload未检查；法律时效仍须核验 |
| P1-3 | P1 | 已确认并修复 | knowledge/policy_updater.py; knowledge/embedding.py | 新版成功后旧file_id标记 superseded，保留旧块审计；失败可重试 | 未回溯标记生产历史重复版本 |
| P1-4 | P1 | 已确认并修复 | agents/knowledge_agent.py; agents/sub_agents/rag_sub_agent.py; agents/research_agent.py | 相关分数优先；日期仅次序判定；Research按问题关联筛选 | 最低分阈值未校准，不宣称检索质量验收 |
| P1-5 | P1 | 已确认并修复 | services/policy_validity.py; agents/research_agent.py | 期间区间判断，起始无终止标待核对；文号年份不通过 | 不做法律效力自动保证；区间解析只覆盖显式年份/日期 |
| P1-6 | P1 | 已确认并修复 | agents/research_agent.py; agents/streaming.py | 相关度排序、选集变更才评估、保留已充分题目、裁剪警告、Research 独立窗口环境变量 | 真实模型相关度与小窗口验收待执行 |
| P1-7 | P1 | 已确认并修复 | agents/research_agent.py; services/security.py | 预留报告与一次修正调用及token；按question_id；具体错误；敏感字段脱敏；有据非申报分析不误拦 | 联合文号及全报告字段脱敏还需专项覆盖 |
| P1-8 | P1 | 已确认并修复 | agents/research_agent.py; web/static/research.js | 瞬时模型/结构失败保留状态从失败节点恢复；其他失败明确从头执行 | 预算耗尽须重新研究；取消的线程调用不能强杀 |
| P1-9 | P1 | 已确认并修复 | knowledge/embedding.py; services/tools/kb_tools.py | 故障向上传递，Tool返回status=error且结论未核验 | 真实外部故障未做网络验收 |
| RAG-001 | P0 | 已确认并修复 | web/main.py; services/network_security.py | 管理配置/政策读写统一 require_admin；本地来源限制目录；出站地址阻止私网和重定向，2MB 限制 | DNS 校验与连接之间仍有 rebinding 窗口，未宣称完全消除 SSRF |
| RAG-002 | P0 | 已确认并修复 | web/main.py; services/network_security.py | 管理配置/政策读写统一 require_admin；本地来源限制目录；出站地址阻止私网和重定向，2MB 限制 | DNS 校验与连接之间仍有 rebinding 窗口，未宣称完全消除 SSRF |
| RAG-003 | P1 | 已确认并修复 | knowledge/policy_updater.py; knowledge/embedding.py | 新版成功后旧file_id标记 superseded，保留旧块审计；失败可重试 | 未回溯标记生产历史重复版本 |
| RAG-004 | P1 | 已确认并修复 | services/policy_metadata.py; knowledge/kb.py; services/tools/kb_tools.py | 显式元数据抽取+未知效力；历史缺字段不硬过滤；Research透传状态 | 生产历史payload未检查；法律时效仍须核验 |
| RAG-005 | P2 | 已确认并修复 | agents/*agent.py | 相关性优先，不让新日期覆盖相关度 | 最低分阈值待真实数据校准 |
| RAG-006 | P1 | 已确认并修复 | knowledge/embedding.py; services/tools/kb_tools.py | 故障向上传递，Tool返回status=error且结论未核验 | 真实外部故障未做网络验收 |
| RAG-007 | P2 | 待运行验证 | — | 尚未完成 | 未消除 |
| RAG-008 | P2 | 已确认并修复 | knowledge/chunking.py; knowledge/kb.py; knowledge/embedding.py | 确定性chunk id；同file_id复用记录；store按id upsert | 旧版随机chunk尚未清理 |
| RAG-009 | P2 | 已确认并修复 | knowledge/kb.py; knowledge/embedding.py | 先uploading，落盘成功后云端激活，失败保持不可检索 | 跨进程与崩溃窗口仍需长期并发验收 |
| RAG-010 | P2 | 待运行验证 | — | 尚未完成 | 未消除 |
| RAG-011 | P2 | 待运行验证 | — | 尚未完成 | 未消除 |
| RAG-012 | P2 | 已确认并修复 | knowledge/chunking.py; knowledge/kb.py | 保留前言、长条文续块条号、payload chunk_index/articles | 旧向量不重建 |
| RAG-013 | P2 | 已确认并修复 | services/tools/date_extractor.py | 完整中文日期、今年为今天、去除公告文号年份 | 复杂自然语言时间仍需模型/人工确认 |
| RAG-014 | P2 | 待人工确认 | — | 尚未完成 | 未消除 |
| RAG-015 | P2 | 待运行验证 | — | 尚未完成 | 未消除 |
| RAG-016 | P2 | 已确认但暂未修复 | knowledge/embedding.py; services/config_manager.py | Embedding分批、数量/维度校验；禁止已有配置直接换模型 | 历史集合无模型指纹，完整模型/集合身份验证未完成 |
| RAG-017 | P2 | 待运行验证 | — | 尚未完成 | 未消除 |
| RAG-018 | P2 | 待运行验证 | — | 尚未完成 | 未消除 |
| RAG-019 | P2 | 待运行验证 | — | 尚未完成 | 未消除 |
| RES-001 | P1 | 已确认并修复 | agents/research_agent.py; web/static/research.js | 瞬时模型/结构失败保留状态从失败节点恢复；其他失败明确从头执行 | 预算耗尽须重新研究；取消的线程调用不能强杀 |
| RES-002 | P1 | 已确认并修复 | services/policy_validity.py; agents/research_agent.py | 期间区间判断，起始无终止标待核对；文号年份不通过 | 不做法律效力自动保证；区间解析只覆盖显式年份/日期 |
| RES-003 | P1 | 已确认并修复 | agents/research_agent.py; agents/streaming.py | 相关度排序、选集变更才评估、保留已充分题目、裁剪警告、Research 独立窗口环境变量 | 真实模型相关度与小窗口验收待执行 |
| RES-004 | P1 | 已确认并修复 | agents/knowledge_agent.py; agents/sub_agents/rag_sub_agent.py; agents/research_agent.py | 相关分数优先；日期仅次序判定；Research按问题关联筛选 | 最低分阈值未校准，不宣称检索质量验收 |
| RES-005 | P1 | 已确认并修复 | services/policy_metadata.py; knowledge/kb.py; services/tools/kb_tools.py | 显式元数据抽取+未知效力；历史缺字段不硬过滤；Research透传状态 | 生产历史payload未检查；法律时效仍须核验 |
| RES-006 | P1 | 已确认并修复 | agents/research_agent.py; services/security.py | 预留报告与一次修正调用及token；按question_id；具体错误；敏感字段脱敏；有据非申报分析不误拦 | 联合文号及全报告字段脱敏还需专项覆盖 |
| RES-007 | P1 | 已确认并修复 | agents/research_agent.py; services/security.py | 预留报告与一次修正调用及token；按question_id；具体错误；敏感字段脱敏；有据非申报分析不误拦 | 联合文号及全报告字段脱敏还需专项覆盖 |
| RES-008 | P1 | 已确认并修复 | agents/research_agent.py; agents/streaming.py | 相关度排序、选集变更才评估、保留已充分题目、裁剪警告、Research 独立窗口环境变量 | 真实模型相关度与小窗口验收待执行 |
| RES-009 | P2 | 已确认并修复 | agents/research_agent.py | unknown_kinds分类，补充事实累加，无补充skip不重复调用understand | 模型事实分类仍需真实验收 |
| RES-010 | P2 | 已确认并修复 | agents/research_agent.py | 可选计算失败降级警告；重新研究清理旧计算和告警 | 纯计算工具异常降级还需扩展 |
| RES-011 | P2 | 已确认并修复 | agents/research_agent.py | 修改报告保留case_references，只绑定报告引用的政策证据 | 人工改稿需人工承担事实审核 |
| RES-012 | P2 | 已确认但暂未修复 | services/research_cases.py | 重复来源保存不同编辑时显式冲突 | 召回最低分与并发重复仍待处理 |
| RES-013 | P2 | 已确认并修复 | agents/research_agent.py | 本地检索优先且网页预算留出本地检索次数 | 极小总预算仍不能完成所有工具 |
| RES-014 | P2 | 已确认但暂未修复 | web/static/research.js | 已修运行计时、真实阶段与心跳状态；SQLite线程池/摘要查询优化待做 | 同步DB在事件循环及线程池竞争仍存在 |
| RES-015 | P3 | 已确认并修复 | web/static/research.js; tests/test_research_ui.js | SSE error/停止刷新，running显示中文阶段 | 真实浏览器断线/停止待验收 |
| RES-016 | P2 | 已确认并修复 | agents/research_agent.py; services/research_evidence.py | Research政策证据限系统政策资料或官网原文，保留来源类型 | 用户上传不再作为政策证明 |
| RES-017 | P2 | 已确认但暂未修复 | agents/research_agent.py | 计划重复编号与未知依赖规范化，结构错误带字段反馈 | 超长模型列表尚无裁剪降级 |
| RES-018 | P3 | 待运行验证 | — | 尚未完成 | 未消除 |
| TEST-1 | P2 | 已确认并修复 | tests/conftest.py; services/paths.py; database/models.py | collection 前隔离；每测试替换并恢复 engine；禁止外网；阶段1回归 56 passed | live 测试仍需独立环境 |
| TEST-2 | P2 | 已确认并修复 | tests/conftest.py; tests/test_*.py | 删除导入时 chdir；隔离默认配置、上传和 checkpoint 路径；阶段1回归 56 passed | 少数显式传参由测试自行隔离 |
| TEST-3 | P2 | 已确认并修复 | tests/test_database_init.py; database/init_db.py | 实跑初始化逻辑；修复 insert_initial_data 成功无返回；阶段1回归 56 passed | 示例数据政策内容未核验 |
| TEST-4 | P3 | 已确认并修复 | tests/test_web_api.py | 删除sys.modules替换；上下文mock可恢复 | 无 |
| TEST-5 | P3 | 已确认并修复 | requirements.txt; tests/test_upload_fixtures.py | 声明pdf2image依赖；失败路径用render替身 | 真实Poppler/OCR未运行 |
| TEST-6 | P3 | 已确认但暂未修复 | scripts/run_tests.py | 隔离运行目录保留用于故障诊断 | 未清理历史.test_runs；将来可加按年龄清理本次生成目录 |
| TEST-7 | P1 | 已确认并修复 | web/main.py; web/templates/*.html; .env.example | 默认凭证拒绝；页面不内置 Token；CORS 白名单；缺头401 | 仍为两个显式配置身份，不是多用户注册系统 |
| WEB-001 | P1 | 已确认并修复 | web/main.py; web/templates/*.html; .env.example | 默认凭证拒绝；页面不内置 Token；CORS 白名单；缺头401 | 仍为两个显式配置身份，不是多用户注册系统 |
| WEB-002 | P0 | 已确认并修复 | web/main.py; services/network_security.py | 管理配置/政策读写统一 require_admin；本地来源限制目录；出站地址阻止私网和重定向，2MB 限制 | DNS 校验与连接之间仍有 rebinding 窗口，未宣称完全消除 SSRF |
| WEB-003 | P0 | 已确认并修复 | web/main.py; services/network_security.py | 管理配置/政策读写统一 require_admin；本地来源限制目录；出站地址阻止私网和重定向，2MB 限制 | DNS 校验与连接之间仍有 rebinding 窗口，未宣称完全消除 SSRF |
| WEB-004 | P1 | 已确认并修复 | web/main.py; web/templates/*.html; .env.example | 默认凭证拒绝；页面不内置 Token；CORS 白名单；缺头401 | 仍为两个显式配置身份，不是多用户注册系统 |
| WEB-005 | P2 | 已确认并修复 | services/network_security.py; services/config_manager.py | 本地模型 origin 由服务端允许；公共模型需 HTTPS 和全局 IP | DNS rebinding 仍待处理；既有配置未改写 |
| WEB-006 | P2 | 已确认并修复 | web/templates/index.html | 令牌输入在配置读取失败时保持可用，独立验证后重新读取 | 需真实浏览器验收 |
| WEB-007 | P1 | 已确认并修复 | web/templates/index.html; tests/test_conversation_ui.js | 401 不切身份或 Token；会话存储按角色；独立访问凭证恢复按钮 | 需真实浏览器验收 |
| WEB-008 | P2 | 已确认但暂未修复 | web/main.py | 文件底层删除失败503，区分不存在；对话多资源删除仍非原子 | 部分失败仍须保留会话记录重试 |
| WEB-009 | P2 | 已确认并修复 | web/templates/index.html | /me 错误不改身份，按角色存 session_id | 历史共享 key 不再自动使用，旧会话仍在列表 |
| WEB-010 | P2 | 已确认并修复 | web/templates/index.html; tests/test_conversation_ui.js | 只允许最后一轮；成功撤回后重发；Node覆盖旧回复拒绝/先撤回 | 真实浏览器与服务器联调待验收 |
| WEB-011 | P2 | 已确认但暂未修复 | web/main.py | 检索故障success=false，错误不回显内部异常 | 统一422前端格式化与旧agent状态码仍待收口 |
| WEB-012 | P2 | 已确认并修复 | agents/tax_agent.py | 附件清单使用统一绝对DATA_DIR | 自定义多实例需对应环境变量 |
| WEB-013 | P2 | 已存在修复，无须重复修改 | web/main.py; services/config_manager.py; tests/test_settings_management.py | 保留 prepare/commit 客户端热更新；测试证明 chat/research 使用新 Key | MockTransport 不代表供应商验收 |
| WEB-014 | P3 | 待运行验证 | — | 尚未完成 | 未消除 |
| WEB-015 | P3 | 待运行验证 | — | 尚未完成 | 未消除 |
| WEB-016 | P3 | 待运行验证 | — | 尚未完成 | 未消除 |
| WEB-017 | P3 | 待运行验证 | — | 尚未完成 | 未消除 |

## 阶段记录

### 阶段 1（完成）

P0-2/3/4 已确认代码问题；修改与隔离回归均完成。旧版本已压缩丢失的消息无法凭摘要恢复；保留已有摘要与剩余消息，不伪造历史。

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

### 阶段 4（新写入链路完成，历史数据迁移未执行）

RAG/入库/分块/政策更新定向：55 passed，0 failed，0 skipped。其中 Qdrant `:memory:` 为真实本地引擎，Embedding为固定向量替身；验证同文件重复入库记录与向量数不增长。政策更新使用Mock云端，断言旧file_id标失效/被替代。生产云端与真实Embedding未连接。

未执行删除、去重、迁移或重建。旧payload缺效力仍为unknown；若需补齐旧数据，应先导出只读快照、生成差异清单，经人工确认后逐file_id补元数据。不得基于文号年份、发布日期推断生效日。

阶段3 checkpoint：fa230ff。

### 阶段 5（核心缺陷完成，跨资源删除仍有遗留）

计算器/Agent/安全/撤回/知识库/案例定向：131 passed，0 failed，0 skipped。新增缺参、非法枚举、税率13、NaN、印花税未知税目、月度累计缺历史收入、反思所得额误匹配、schema异常不500、路径穿越、WAL四线程40行写入回归。仅Mock模型和临时SQLite。

旧负销售额测试原本接受静默0税额，改为明确拒绝负数并提示独立调整业务路径；未改法定税率或优惠口径。阶段4 checkpoint：c51db0a。

### 阶段 6（离线回归及评测安全）

- `eval/run_eval.py`：默认离线意图评测，真实答案需 `--live`；执行错误单列、评分为零，不能当作正确拒答；每轮独立工作目录、租户、记忆和检查点，关闭评测连接，向量检索只读。
- `eval/research_eval.py`：报告冻结快照来源匹配数；按显式家族映射标注潜在泄漏；晚于 `as_of` 的控制案例不召回。没有新增未经核验的政策条款或标准答案。
- `tests/`：修正缺凭证应401、有效凭证下坏请求仍422、管理员权限、政策流式字节Mock；清除跨测试模块Mock污染。OCR失败使用替身覆盖，真实OCR不计通过。
- 收尾定向回归：51 passed；已运行 Node 对话、Research 脚本以及 `node --check web/static/research.js`。对话断流、重新生成、身份失败；Research 心跳、报错刷新与状态标签用 DOM 替身验证。未完成真实浏览器端到端：浏览器控制仅有 MCP Apps，没有可用浏览器标签提供方。
- 前端严格静态审计：0 errors / 0 warnings；Python `compileall` 通过。单独执行 `python -m tests.test_database_init`，设置 `DATABASE_URL=sqlite:///:memory:` 和 `PYTHON_DOTENV_DISABLED=1`，5 passed，正式库不参与。
- 全套测试：`.venv/Scripts/python.exe scripts/run_tests.py -q -rs --tb=short` → **477 passed, 0 failed, 7 skipped**（70.69秒，3条弃用警告）。跳过6个真实API连通性测试与1个真实Embedding/Qdrant集成。日志 `.test_runs/audit-1106-verified.txt`。本次未发起真实模型、Embedding 或云Qdrant调用。

阶段5 checkpoint：e90ea7b。所有检查点仅本地，无 push。

## 验收边界与下一步

1. **模型、浏览器与政策仍需验收。** 建议先用专用账号和隔离 Qdrant，选少量 Research 成功、证据不足、超时恢复及取消案例跑真实模型；浏览器验证首次凭证、401恢复、聊天续聊、重新生成和 Research 全流程。Mock 通过不能代替真实输出验收。
2. **政策时效仍有历史缺口。** 生产历史 payload/重复版本未读取；未知效力不得显示为有效。AGT-007、AGT-012、RAG-014、AGT-021 需查官方原文与适用期；EVAL-4 需逐题补可核验政策来源。
3. **安全遗留。** 管理路由、目录与网络地址基础校验已生效；DNS解析与实际连接间仍有 TOCTOU，模型SDK重定向与代理需独立验收，面向不受信网络部署前建议增加出站控制。未修改真实密钥或配置。
4. **删除与并发遗留。** 删除对话仍跨SQLite、文件与Qdrant，未实现原子回滚；旧委托检查点仍待安全清理。WAL和忙等待提高并发能力，不等于跨服务事务。
5. **性能与基准。** Research工作线程不能强制打断正在进行的同步调用，SQLite摘要与线程池待优化；历史向量模型指纹、检索阈值、数值评测真值仍待验证。

结论：可以开始隔离环境的小规模产品验收；尚无依据宣布完整产品验收通过或上线就绪。
