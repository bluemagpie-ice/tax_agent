"""
T11 — IntentRouter 意图分类测试

测试点：
  1. 知识查询类文案 → 命中 knowledge_query
  2. 税务计算类文案 → 命中 tax_calculation
  3. 发票处理类文案 → 命中 invoice_handling
  4. 模糊问候 → 兜底 general_tax
  5. 无意义字符 → 兜底 general_tax
  额外：输出结构完整性、实体抽取、紧急程度
"""

import pytest

from agents.intent_router import HybridIntentRouter, INTENTS

@pytest.fixture
def router():
    return HybridIntentRouter()

# ================================================================== 1. 主分类（5 种意图）
class TestIntentClassification:
    """输入文案 → 期望 intent（核心测试）"""

    @pytest.mark.parametrize("message,expected", [
        # ---- knowledge_query ----
        ("什么是增值税", "knowledge_query"),
        ("如何查询小微企业优惠政策", "knowledge_query"),
        ("增值税的定义是什么", "knowledge_query"),
        ("咨询一下最新政策", "knowledge_query"),

        # ---- tax_calculation ----
        ("帮我计算一下企业所得税", "tax_calculation"),
        ("增值税税率是多少", "tax_calculation"),
        ("计算应纳税额", "tax_calculation"),
        ("这个月税额多少", "tax_calculation"),

        # ---- invoice_handling ----
        ("发票抬头怎么写", "invoice_handling"),
        ("如何开具红字发票", "invoice_handling"),
        ("专票可以抵扣吗", "invoice_handling"),
        ("普票能报销吗", "invoice_handling"),

        # ---- filing_guidance ----
        ("汇算清缴怎么办理", "filing_guidance"),
        ("逾期申报怎么办", "filing_guidance"),

        # ---- tax_planning ----
        ("帮我做税务筹划方案", "tax_planning"),
        ("如何优化企业税务结构", "tax_planning"),

        # ---- compliance_check ----
        ("帮我做税务风险自查", "compliance_check"),
        ("合规性检查怎么做", "compliance_check"),

        # ---- 兜底 general_tax ----
        ("你好", "general_tax"),
        ("在吗", "general_tax"),
        ("帮我一下", "general_tax"),
        ("asdfqwer", "general_tax"),
        ("111???", "general_tax"),
    ])
    def test_classify_intent(self, router, message, expected):
        result = router.classify(message)
        assert result["intent"] == expected, (
            f"消息 '{message}' 期望 '{expected}'，实际是 '{result['intent']}'\n"
            f"完整结果: {result}"
        )

# ================================================================== 2. 兜底逻辑
class TestFallback:
    """score < 0.14 → general_tax，且兜底时 decision.score 强制为 0.35"""

    def test_greeting_fallback_has_forced_score(self, router):
        result = router.classify("你好")
        assert result["intent"] == "general_tax"
        assert result["decision"]["score"] == 0.35
        # general_tax 兜底时：tax_agent 作为统筹在首位 + 默认协作链（源码设计如此）
        assert result["agents"][0] == "tax_agent"
        assert len(result["agents"]) >= 2   # 至少有统筹 + 协作链里的 agent`

    def test_nonsense_fallback(self, router):
        result = router.classify("乱打的一串字符xyz")
        assert result["intent"] == "general_tax"

    def test_fallback_threshold(self, router):
        """兜底文案的原始最高分必须 < 0.14"""
        normalized = router._normalize("你好")
        max_raw = 0.0
        for definition in INTENTS:
            matched = [kw for kw in definition.keywords if kw.lower() in normalized]
            pattern = min(1.0, len(matched) / 2) if definition.keywords else 0.0
            semantic = max(
                (
                    router._cosine(
                        router._vector(normalized), router._vector(ex)
                    )
                    for ex in definition.examples
                ),
                default=0.0,
            )
            score = router.pattern_weight * pattern + router.semantic_weight * semantic
            max_raw = max(max_raw, score)
        assert max_raw < 0.14, f"原始最高分 {max_raw:.4f} 应 < 0.14 触发兜底"

# ================================================================== 3. 输出字段完整性
class TestOutputStructure:
    """验证 classify 返回的字典包含所有声明的字段"""

    REQUIRED_TOP_LEVEL = {
        "intent", "confidence", "urgency", "agents", "multi_agent",
        "collaboration_plan", "entities", "decision", "alternatives",
        "strategy",
    }

    def test_all_required_fields_present(self, router):
        result = router.classify("什么是增值税")
        missing = self.REQUIRED_TOP_LEVEL - set(result.keys())
        assert not missing, f"缺少顶层字段: {missing}"

    def test_decision_field_details(self, router):
        """decision 是 primary intent 的详细信息"""
        result = router.classify("计算应纳税额")
        decision = result["decision"]
        assert "intent" in decision
        assert "agent" in decision
        assert "score" in decision
        assert "signals" in decision       # {"pattern": 0.5, "semantic": 0.3}
        assert "matched_keywords" in decision
        assert isinstance(decision["matched_keywords"], list)

    def test_confidence_in_valid_range(self, router):
        result = router.classify("发票抬头怎么写")
        assert 0.0 < result["confidence"] <= 0.98

# ================================================================== 4. 实体抽取
class TestEntitiesExtraction:

    def test_extract_tax_types(self, router):
        result = router.classify("增值税和企业所得税怎么算")
        tax_types = result["entities"]["tax_types"]
        assert "增值税" in tax_types
        assert "企业所得税" in tax_types

    def test_extract_invoice_types(self, router):
        result = router.classify("增值税专用发票可以抵扣吗")
        invoice_types = result["entities"]["invoice_types"]
        assert any("发票" in t for t in invoice_types)

    def test_extract_rates_and_amounts(self, router):
        result = router.classify("13%的增值税率，金额100万元")
        rates = result["entities"]["rates"]
        amounts = result["entities"]["amounts"]
        assert any("13" in r for r in rates)
        assert any("100" in a for a in amounts)

    def test_entities_empty_when_no_tax(self, router):
        result = router.classify("你好")
        assert result["entities"]["tax_types"] == []
        assert result["entities"]["rates"] == []

# ================================================================== 5. 紧急程度
class TestUrgency:

    def test_critical(self, router):
        result = router.classify("稽查来了怎么办，立即处理")
        assert result["urgency"] == "critical"

    def test_high(self, router):
        result = router.classify("逾期申报了要补税")
        assert result["urgency"] == "high"

    def test_normal(self, router):
        result = router.classify("增值税税率是多少")
        assert result["urgency"] == "normal"