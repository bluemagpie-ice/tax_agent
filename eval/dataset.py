"""
评测数据集模块
==============

定义评测样本结构 EvalSample，并提供一组内置的税务问答评测集，
用于衡量 Agent 的意图路由准确率、答案关键信息召回率与拒答正确性。

评测集设计原则：
- 覆盖 intent_router 定义的 6 类意图 + 拒答场景
- expected_keywords 只放"正确答案几乎必然包含"的稳定术语（避免数字格式差异导致误判）
- must_refuse=True 表示该问题在知识库无对应内容，Agent 应当拒答而非编造
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EvalSample:
    """一条评测样本。"""

    id: str                       # 样本唯一标识
    query: str                    # 用户问题
    expected_intent: str          # 期望意图（对应 intent_router 的 INTENTS 名）
    expected_keywords: list[str]  # 答案应包含的关键词（子串匹配，命中即得分）
    must_refuse: bool = False     # True = 应拒答（知识库无内容，不应编造）
    note: str = ""                # 备注（评测口径说明）


# 意图名与 intent_router.INTENTS 对齐
INTENT_KNOWLEDGE = "knowledge_query"
INTENT_PLANNING = "tax_planning"
INTENT_CALCULATION = "tax_calculation"
INTENT_INVOICE = "invoice_handling"
INTENT_COMPLIANCE = "compliance_check"
INTENT_FILING = "filing_guidance"


BUILTIN_DATASET: list[EvalSample] = [
    # ---------------- 知识问答 ----------------
    EvalSample("kb_001", "什么是进项税？", INTENT_KNOWLEDGE, ["进项税"]),
    EvalSample("kb_002", "什么是增值税？", INTENT_KNOWLEDGE, ["增值税"]),
    EvalSample("kb_003", "增值税一般纳税人的认定标准是什么？", INTENT_KNOWLEDGE, ["一般纳税人"]),
    EvalSample("kb_004", "什么是留抵退税？", INTENT_KNOWLEDGE, ["留抵"]),
    EvalSample("kb_005", "小型微利企业有什么企业所得税优惠？", INTENT_KNOWLEDGE, ["小型微利"]),
    EvalSample("kb_006", "进项税和销项税有什么区别？", INTENT_KNOWLEDGE, ["进项", "销项"]),

    # ---------------- 税务计算 ----------------
    EvalSample(
        "calc_001",
        "我是增值税一般纳税人，销售额100万不含税，进项5万，税率13%，应纳税额多少？",
        INTENT_CALCULATION, ["销项", "进项"],
        note="计算类答案应展示销项/进项与抵扣过程",
    ),
    EvalSample(
        "calc_002",
        "小规模纳税人季度销售额28万，要交增值税吗？",
        INTENT_CALCULATION, ["免征"],
        note="28万<30万，答案应提及免征",
    ),
    EvalSample(
        "calc_003",
        "企业所得税应纳税所得额200万，小型微利企业，要交多少税？",
        INTENT_CALCULATION, ["小型微利"],
    ),
    EvalSample(
        "calc_004",
        "月薪15000元，第一个月要预扣多少个人所得税？",
        INTENT_CALCULATION, ["预扣"],
    ),

    # ---------------- 税务筹划 ----------------
    EvalSample("plan_001", "小微企业怎么做税务筹划？", INTENT_PLANNING, ["筹划"]),
    EvalSample("plan_002", "如何合法地降低企业所得税税负？", INTENT_PLANNING, ["筹划", "优惠"]),
    EvalSample(
        "plan_003",
        "一般纳税人和小规模纳税人，选哪个更省税？",
        INTENT_PLANNING, ["小规模"],
    ),

    # ---------------- 发票处理 ----------------
    EvalSample("invoice_001", "餐饮发票能抵扣进项税吗？", INTENT_INVOICE, ["发票"]),
    EvalSample("invoice_002", "红字发票怎么开具？", INTENT_INVOICE, ["红字"]),
    EvalSample("invoice_003", "发票抬头写错了还能报销吗？", INTENT_INVOICE, ["抬头"]),

    # ---------------- 合规检查 ----------------
    EvalSample("comp_001", "企业有哪些常见的税务风险？", INTENT_COMPLIANCE, ["风险"]),
    EvalSample("comp_002", "公司账务处理不合规会有什么后果？", INTENT_COMPLIANCE, ["风险", "合规"]),

    # ---------------- 申报指导 ----------------
    EvalSample("file_001", "企业所得税汇算清缴什么时候截止？", INTENT_FILING, ["汇算清缴"]),
    EvalSample("file_002", "增值税纳税申报的期限是多久？", INTENT_FILING, ["申报"]),
    EvalSample("file_003", "个人所得税年度汇算怎么办理？", INTENT_FILING, ["汇算"]),

    # ---------------- 拒答场景（知识库无内容，应拒答而非编造） ----------------
    EvalSample(
        "refuse_001",
        "火星殖民地的企业所得税应该怎么申报？",
        INTENT_FILING, [], must_refuse=True,
        note="知识库不含此类内容，Agent 应明确告知无法回答（意图路由到申报类合理，考核点在拒答）",
    ),
    EvalSample(
        "refuse_002",
        "2080年才会生效的某部税法第999条是什么？",
        INTENT_KNOWLEDGE, [], must_refuse=True,
        note="无中生有的问题，Agent 不应编造条文",
    ),
]


def load_dataset() -> list[EvalSample]:
    """返回内置评测集（后续可从 JSON/YAML 扩展）。"""
    return list(BUILTIN_DATASET)
