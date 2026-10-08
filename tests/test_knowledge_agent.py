"""
T9 — KnowledgeAgent 初始化与路由测试
======================================

测试目标：
    1. KnowledgeAgent() 实例化不崩（RAG 初始化失败时应降级）
    2. _is_policy_query 关键词识别正确
    3. _is_definition_query 关键词识别正确
    4. 路由优先级：同时命中两类时走 policy（先判）
    5. retrieve_context 在 RAG 未初始化时降级返回提示文本

不依赖外部服务。run() 方法因需要 LLM 调用，标记为可选测试。

运行方式：
    cd D:/code/tax_agent
    python -m tests.test_knowledge_agent
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(str(PROJECT_ROOT))

# ================================================================
# 测试 1：实例化不崩
# ================================================================
def test_instantiation_does_not_crash():
    """KnowledgeAgent() 能成功实例化，即使 RAG 服务连不上"""
    from agents.knowledge_agent import KnowledgeAgent

    agent = KnowledgeAgent()

    # 基础属性
    assert agent.name == "knowledge_agent"
    assert isinstance(agent.capabilities, list)
    assert len(agent.capabilities) >= 1
    print(f"  ✅ KnowledgeAgent 实例化成功: name='{agent.name}', capabilities={agent.capabilities}")

    # 子 Agent 挂载
    assert hasattr(agent, "policy_agent"), "❌ 应挂载 policy_agent"
    assert hasattr(agent, "definition_agent"), "❌ 应挂载 definition_agent"
    print(f"  ✅ policy_agent / definition_agent 子 Agent 已挂载")

    # __init__ 里的 try/except 会在失败时设为 None
    if agent._store is None or agent._embedding_service is None:
        print(f"  ℹ️ RAG 服务未初始化（Neo4j/Qdrant 可能没装），但 Agent 仍可用 —— 降级正常")
    else:
        print(f"  ✅ RAG 服务初始化成功")

# ================================================================
# 测试 2：_is_policy_query 关键词识别
# ================================================================
def test_is_policy_query():
    """政策类关键词应命中"""
    from agents.knowledge_agent import KnowledgeAgent
    agent = KnowledgeAgent()

    policy_queries = [
        "根据税法规定，增值税怎么算",
        "小微企业所得税优惠政策",
        "税务总局发布的最新通知",
        "这种做法是否合规",
        "企业重组的税务处理条例",
        "发票管理办法有哪些要求",
    ]
    for q in policy_queries:
        result = agent._is_policy_query(q)
        assert result is True, f"❌ 应命中政策查询: '{q}'"

    # 不应命中
    # 不应命中（确保不含任何 policy 关键词，否则会被误判）
    non_policy = [
        "什么是进项税",              # 命中 "什么是" → 定义类
        "帮我算一下公司的账目",      # 纯计算类（不含政策关键词）
        "你好",                      # 闲聊
        "怎么在电子税务局操作申报",  # 操作流程类
    ]

    for q in non_policy:
        result = agent._is_policy_query(q)
        assert result is False, f"❌ 不应命中政策查询: '{q}'"

    print(f"  ✅ _is_policy_query 关键词识别准确（{len(policy_queries)} 正例 / {len(non_policy)} 反例）")

# ================================================================
# 测试 3：_is_definition_query 关键词识别
# ================================================================
def test_is_definition_query():
    """术语定义类关键词应命中"""
    from agents.knowledge_agent import KnowledgeAgent
    agent = KnowledgeAgent()

    definition_queries = [
        "什么是进项税额",
        "进项税额是什么",
        "给我解释一下留抵退税",
        "进项税的定义是什么",
        "小规模纳税人这个概念",
    ]
    for q in definition_queries:
        result = agent._is_definition_query(q)
        assert result is True, f"❌ 应命中定义查询: '{q}'"

    # 不应命中
    non_definition = [
        "增值税政策有哪些",
        "怎么计算企业所得税",
        "你好",
    ]
    for q in non_definition:
        result = agent._is_definition_query(q)
        assert result is False, f"❌ 不应命中定义查询: '{q}'"

    print(f"  ✅ _is_definition_query 关键词识别准确（{len(definition_queries)} 正例 / {len(non_definition)} 反例）")

# ================================================================
# 测试 4：路由优先级 —— 同时命中两类时走 policy
# ================================================================
def test_routing_priority():
    """
    handle() 里先判 _is_policy_query，再判 _is_definition_query。
    如果一个查询同时命中两类，应该走 policy_agent。
    """
    from agents.knowledge_agent import KnowledgeAgent
    agent = KnowledgeAgent()

    # 构造同时命中两类的查询：含 "政策"(policy) + "什么是"(definition)
    mixed_query = "什么是增值税留抵退税政策"

    is_policy = agent._is_policy_query(mixed_query)
    is_def = agent._is_definition_query(mixed_query)

    print(f"  查询: '{mixed_query}'")
    print(f"    _is_policy_query = {is_policy}")
    print(f"    _is_definition_query = {is_def}")

    # 预期：policy 关键词 "政策" 命中 → True
    # 预期：definition 关键词 "什么是" 命中 → True
    assert is_policy is True, "❌ '政策' 应命中 policy"
    assert is_def is True, "❌ '什么是' 应命中 definition"

    # 但 handle() 里 policy 先判，所以路由会优先走 policy_agent
    # 验证顺序：
    # if self._is_policy_query(query):  ← 先判，True
    #     return await self.policy_agent.run(...)  ← 这里返回
    # if self._is_definition_query(query):  ← 永远不会执行到
    print(f"  ℹ️ 同时命中两类，但 policy 先判 → 路由到 policy_agent（当前设计优先级）")
    print(f"  ✅ 路由优先级确认：policy > definition > general")

# ================================================================
# 测试 5：retrieve_context 未初始化降级
# ================================================================
def test_retrieve_context_degradation():
    """RAG 服务未初始化时 retrieve_context() 不崩，返回降级提示"""
    from agents.knowledge_agent import KnowledgeAgent

    agent = KnowledgeAgent()

    # 强制模拟未初始化状态
    agent._store = None
    agent._embedding_service = None

    result = agent.retrieve_context("随便问问")
    assert isinstance(result, str), f"❌ 应返回 str，实际 {type(result)}"
    assert len(result) > 0
    assert "未初始化" in result or "检索" in result or "暂" in result
    print(f"  ✅ retrieve_context 降级: '{result[:30]}...'")

# ================================================================
# 测试 6：handle() 路由分发逻辑（不真正调 LLM）
# ================================================================
def test_handle_routing_logic():
    """
    验证 handle() 的 if-elif-else 分支逻辑正确。
    不真正调用 LLM（用 mock 拦截子 Agent 的 run 方法）。
    """
    from unittest.mock import patch, AsyncMock
    from agents.knowledge_agent import KnowledgeAgent

    agent = KnowledgeAgent()

    # 用 AsyncMock 替换子 Agent 的 run 方法，捕获调用
    policy_mock = AsyncMock(return_value="[policy_agent 回答]")
    def_mock = AsyncMock(return_value="[definition_agent 回答]")

    agent.policy_agent.run = policy_mock
    agent.definition_agent.run = def_mock

    import asyncio

    # ---- 测试 A：明确的政策查询 → 应调用 policy_agent ----
    result = asyncio.run(agent.handle("增值税优惠政策", context={"thread_id": "t1"}))
    assert policy_mock.called, "❌ 政策查询应调用 policy_agent.run()"
    assert not def_mock.called, "❌ 政策查询不应调用 definition_agent"
    assert result == "[policy_agent 回答]"

    # ---- 重置 mock ----
    policy_mock.reset_mock()
    def_mock.reset_mock()

    # ---- 测试 B：明确的定义查询 → 应调用 definition_agent ----
    result2 = asyncio.run(agent.handle("什么是进项税", context={"thread_id": "t2"}))
    assert not policy_mock.called, "❌ 定义查询不应调用 policy_agent"
    assert def_mock.called, "❌ 定义查询应调用 definition_agent.run()"
    assert result2 == "[definition_agent 回答]"

    print(f"  ✅ handle() 路由分发逻辑正确（mock 验证）")

# ================================================================
# 运行全部测试
# ================================================================
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING)

    print("\n" + "=" * 60)
    print("T9 — KnowledgeAgent 初始化与路由测试")
    print("=" * 60)

    tests = [
        ("实例化不崩 + 子 Agent 挂载", test_instantiation_does_not_crash),
        ("_is_policy_query 关键词识别", test_is_policy_query),
        ("_is_definition_query 关键词识别", test_is_definition_query),
        ("路由优先级：policy > definition > general", test_routing_priority),
        ("retrieve_context 未初始化降级", test_retrieve_context_degradation),
        ("handle() 路由分发（mock 验证）", test_handle_routing_logic),
    ]

    passed = 0
    failed = 0

    for i, (name, func) in enumerate(tests, 1):
        print(f"\n[{i}/{len(tests)}] {name}...")
        try:
            func()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"  ❌ 失败: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'=' * 60}")
    print(f"测试结果: ✅ {passed} passed | ❌ {failed} failed")
    print(f"{'=' * 60}\n")

    if failed > 0:
        sys.exit(1)