"""时效性检测 + 自动联网检索 回归测试。"""

from agents.intent_router import HybridIntentRouter, detect_freshness


def test_detect_freshness_true():
    for q in [
        "最新政策是什么",
        "最近有什么新规",
        "2026年增值税新政策",
        "刚刚出台的新规定",
        "近期有没有新的税收优惠",
    ]:
        assert detect_freshness(q), f"应识别为时效性查询: {q}"


def test_detect_freshness_false():
    for q in [
        "什么是增值税",
        "进项税怎么算",
        "餐饮发票能抵扣吗",
        "一般纳税人的认定标准",
    ]:
        assert not detect_freshness(q), f"不应识别为时效性查询: {q}"


def test_classify_includes_needs_web_search():
    router = HybridIntentRouter()
    plan = router.classify("最近有没有新的增值税优惠政策")
    assert plan["needs_web_search"] is True
    assert "needs_web_search" in router.classify("什么是增值税")
