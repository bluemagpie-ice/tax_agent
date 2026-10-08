"""
税务 Agent 意图路由器
--------------------
基于 关键词匹配 + 本地 n-gram 余弦相似度 的轻量混合路由：
1. 对每条 Intent 计算 pattern（关键词命中占比）与 semantic（示例句最大余弦相似度）
2. 加权融合得分，选出 primary intent
3. 同时输出多 Agent 协作链、紧急程度、税务实体等，供上层编排使用
todo：对于一些agent，到底是用agent方便还是用tools更方便？
"""

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple


@dataclass
class IntentDefinition:
    name: str
    agent: str
    keywords: Tuple[str, ...]
    examples: Tuple[str, ...]

INTENTS: Tuple[IntentDefinition, ...] = (
    IntentDefinition(
        "knowledge_query",
        "knowledge_agent",
        (
            "什么是", "是什么", "区别", "定义", "解释", "含义", "概念",
            "查询", "咨询", "知识", "制度", "政策", "范围", "优惠", "减免",
            "历史", "参考",
        ),
        ("什么是增值税", "增值税的征税范围是什么", "进项税和销项税的区别",
         "小型微利企业有什么企业所得税优惠"),
    ),
    IntentDefinition(
        "tax_planning",
        "planning_agent",
        (
            "税务筹划", "税收筹划", "避税", "方案", "建议", "优化", "结构",
            "安排", "税负", "省税", "节税", "降低", "合法",
        ),
        ("形成税务筹划方案", "如何合法降低企业所得税税负",
         "一般纳税人和小规模哪个省税", "如何优化企业税务结构"),
    ),
    IntentDefinition(
        "tax_calculation",
        "calculation_agent",
        (
            "计算", "金额", "应纳税额", "应纳税所得额", "税率", "税额", "税金",
            "算税", "预扣", "个税", "个人所得税", "工资", "月薪", "交税", "要交",
        ),
        ("计算应纳税额", "月薪15000预扣多少个人所得税",
         "计算增值税税额", "计算企业所得税"),
    ),
    IntentDefinition(
        "invoice_handling",
        "invoice_agent",
        (
            "发票", "开票", "抬头", "税号", "专票", "普票", "电子发票",
            "数电发票", "作废", "红字", "抵扣", "报销",
        ),
        ("发票能否抵扣", "如何开具红字发票", "餐饮发票能否报销"),
    ),
    IntentDefinition(
        "compliance_check",
        "compliance_agent",
        ("合规", "检查", "验证", "风险", "自查", "稽查", "合规性"),
        ("检查方案是否符合税务合规要求", "验证是否存在税务风险"),
    ),
    IntentDefinition(
        "filing_guidance",
        "filing_agent",
        (
            "申报", "报税", "缴税", "汇算", "汇算清缴", "申报期限",
            "逾期", "补税", "退税", "期限", "截止", "办理", "怎么办",
        ),
        ("如何进行年度申报", "汇算清缴如何办理",
         "增值税申报期限是多久", "逾期申报怎么办"),
    ),
)

# 时效性触发词：命中说明用户在询问"最新/最近"信息，应触发联网实时检索，而非只搜本地知识库
FRESHNESS_KEYWORDS = (
    "最新", "最近", "新政策", "新规定", "新规", "最新出台",
    "近期", "刚发布", "刚刚出台", "今年新",
)


def detect_freshness(text: str) -> bool:
    """检测用户是否在询问时效性强的信息（如"最新政策""最近有没有新规"）。"""
    if not text:
        return False
    return any(kw in text for kw in FRESHNESS_KEYWORDS)


class HybridIntentRouter:
    """
    混合意图路由器：关键词权重 0.62 + n-gram 语义权重 0.38
    """

    pattern_weight = 0.62
    semantic_weight = 0.38

    # --------------------------------------------------------------- #
    # 1. 主分类入口
    # --------------------------------------------------------------- #
    def classify(self, message: str) -> Dict[str, Any]:
        normalized = self._normalize(message)
        scored: List[Dict[str, Any]] = []

        for definition in INTENTS:
            matched = [kw for kw in definition.keywords if kw.lower() in normalized]
            pattern = min(1.0, len(matched) / 2) if definition.keywords else 0.0
            semantic = max(
                (
                    self._cosine(
                        self._vector(normalized),
                        self._vector(example),
                    )
                    for example in definition.examples
                ),
                default=0.0,
            )
            score = self.pattern_weight * pattern + self.semantic_weight * semantic
            scored.append(
                {
                    "intent": definition.name,
                    "agent": definition.agent,
                    "score": round(score, 4),
                    "signals": {
                        "pattern": round(pattern, 4),
                        "semantic": round(semantic, 4),
                    },
                    "matched_keywords": matched,
                }
            )

        scored.sort(key=lambda item: item["score"], reverse=True)
        primary = scored[0] #取最高分的INTENT

        # 兜底：若最高分过低，退化为通用税务咨询
        if primary["score"] < 0.14:
            primary = {
                "intent": "general_tax",
                "agent": "tax_agent",
                "score": 0.35,
                "signals": {"pattern": 0.0, "semantic": 0.0},
                "matched_keywords": [],
            }

        # 多 Agent 协作：选出得分不低于主意图 68% 的其他意图（最多 4 个）
        collaboration_floor = max(0.25, primary["score"] * 0.68)
        selected = [
            item
            for item in scored
            if item["score"] >= collaboration_floor
            and item["intent"] != primary["intent"]
        ][:4]

        agents = self._complete_collaboration(
            [primary["agent"], *[item["agent"] for item in selected]],
            message,
            primary["intent"],
        )

        confidence = min(0.98, 0.42 + primary["score"] * 0.54)
        return {
            "intent": primary["intent"],
            "confidence": round(confidence, 3),
            "urgency": self._urgency(message),
            "agents": agents,
            "multi_agent": len(agents) > 1,
            "collaboration_plan": self._collaboration_plan(agents),
            "entities": self._entities(message),
            "decision": primary,
            "alternatives": scored[1:4],
            "strategy": "pattern+local_ngram_similarity",
            "needs_web_search": detect_freshness(message),
        }

    # --------------------------------------------------------------- #
    # 2. 协作链补全（税务场景）
    # --------------------------------------------------------------- #
    def _complete_collaboration(
        self, agents: List[str], message: str, primary_intent: str
    ) -> List[str]:
        # 税务默认协作链：知识 → 发票 → 计算 → 申报 → 合规
        default_chain = [
            "knowledge_agent",
            "invoice_agent",
            "calculation_agent",
            "filing_agent",
            "compliance_agent",
            "planning_agent",
        ]

        # 根据消息关键词动态插入/追加
        lowered = message.lower()
        if any(term in lowered for term in ("筹划", "优化", "方案", "结构", "避税")):
            default_chain.append("planning_agent")
        if any(term in lowered for term in ("申报", "报税", "汇算", "补税", "退税")):
            default_chain.append("filing_agent")
        if any(term in lowered for term in ("发票", "抬头", "专票", "红字", "抵扣")):
            default_chain.insert(0, "invoice_agent")

        target_size = 8 if primary_intent != "general_tax" else 6
        merged = list(dict.fromkeys([*agents, *default_chain]))
        return merged[: max(target_size, min(len(merged), 10))]

    # --------------------------------------------------------------- #
    # 3. 协作计划（税务版）
    # --------------------------------------------------------------- #
    def _collaboration_plan(self, agents: List[str]) -> List[Dict[str, str]]:
        responsibilities = {
            "tax_agent":         "统筹整体税务咨询，作为最终答复责任人。",
            "knowledge_agent":   "检索税务法规、政策与术语定义，提供知识依据。",
            "invoice_agent":     "处理发票开具、抬头、抵扣、红字、报销等问题。",
            "calculation_agent": "计算应纳税额、税率、进项/销项、应纳税所得额等。",
            "filing_agent":      "指导纳税申报、汇算清缴、补税退税与申报期限。",
            "compliance_agent":  "检查税务合规性、识别风险、给出自查建议。",
            "planning_agent":    "设计税务筹划方案、优化纳税结构与选择。",
        }
        return [
            {
                "agent": agent,
                "responsibility": responsibilities.get(
                    agent, "根据主 Agent 调度承担补充职责。"
                ),
            }
            for agent in agents
        ]

    # --------------------------------------------------------------- #
    # 4. 文本归一化 / 向量化 / 余弦相似度
    # --------------------------------------------------------------- #
    def _normalize(self, text: str) -> str:
        return re.sub(r"\s+", "", text.lower())

    def _vector(self, text: str) -> Counter[str]:
        compact = self._normalize(text)
        features: Counter[str] = Counter()
        for token in re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", compact):
            features[f"t:{token}"] += 1
        for size in (2, 3):
            for index in range(max(0, len(compact) - size + 1)):
                features[f"g{size}:{compact[index:index + size]}"] += 1
        return features

    def _cosine(self, left: Counter[str], right: Counter[str]) -> float:
        """计算两个 Counter 向量的余弦相似度"""
        if not left or not right:
            return 0.0
        # 取两个 Counter 共有的特征键
        shared = set(left) & set(right)
        # 分子：点积（对应特征的计数乘积之和）
        numerator = sum(left[key] * right[key] for key in shared)
        # 分母：各向量的模长（L2 范数）
        left_norm = math.sqrt(sum(value * value for value in left.values()))
        right_norm = math.sqrt(sum(value * value for value in right.values()))
        # 防止除以零
        return numerator / max(left_norm * right_norm, 1e-9)

    # --------------------------------------------------------------- #
    # 5. 紧急程度判断（税务场景）
    # --------------------------------------------------------------- #
    def _urgency(self, text: str) -> str:
        critical = (
            "立即", "马上", "重大", "稽查",
            "处罚", "罚款", "滞纳金", "逾期未",
        )
        high = (
            "紧急", "高风险", "逾期", "补税", "退税", "投诉", "阻断",
        )
        if any(kw in text for kw in critical):
            return "critical"
        if any(kw in text for kw in high):
            return "high"
        return "normal"

    # --------------------------------------------------------------- #
    # 6. 税务实体抽取
    # --------------------------------------------------------------- #
    def _entities(self, text: str) -> Dict[str, List[str]]:
        # 常见税种
        tax_types = [
            item
            for item in (
                "增值税", "消费税", "企业所得税", "个人所得税",
                "印花税", "房产税", "城镇土地使用税", "土地增值税",
                "车辆购置税", "车船税", "契税", "关税",
            )
            if item in text
        ]

        # 发票类型
        invoice_types = [
            item
            for item in (
                "增值税专用发票", "增值税普通发票", "电子普通发票",
                "电子专用发票", "数电发票", "机动车销售统一发票",
                "红字发票", "发票",
            )
            if item in text
        ]

        # 纳税人类型
        taxpayer_types = [
            item
            for item in ("一般纳税人", "小规模纳税人", "小型微利企业", "个体工商户")
            if item in text
        ]

        # 税务条款/文号：如 "财税[2023]12号"、"国家税务总局公告2024年第3号"
        law_references = self._unique(
            [
                *re.findall(
                    r"(?:财税|国家税务总局|财政部|国税)[^\s，,。；;]{0,20}",
                    text,
                ),
                *re.findall(r"第\s*\d+\s*号", text),
                *re.findall(r"\[\d{4}\]\s*\d+\s*号", text),
            ]
        )

        # 金额/税率：如 "13%"、"6%"、"3%"、"100万"
        rates = self._unique(
            re.findall(r"\d+(?:\.\d+)?\s*%", text)
        )
        amounts = self._unique(
            re.findall(r"\d+(?:\.\d+)?\s*(?:万元|亿|万|元)", text)
        )

        return {
            "tax_types": tax_types[:6],
            "invoice_types": invoice_types[:6],
            "taxpayer_types": taxpayer_types[:4],
            "law_references": law_references[:6],
            "rates": rates[:6],
            "amounts": amounts[:6],
        }

    # --------------------------------------------------------------- #
    # 7. 工具方法
    # --------------------------------------------------------------- #
    def _unique(self, values: Iterable[str]) -> List[str]:
        return list(dict.fromkeys(value.strip() for value in values if value.strip()))