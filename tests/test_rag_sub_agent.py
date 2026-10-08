"""
T8 — RAGSubAgent 基类 + 子类继承测试
====================================

测试目标：
    1. PolicyAgent / DefinitionAgent 能被正常实例化
    2. 继承后类属性（NODE_NAME / SYSTEM_PROMPT / RETRIEVE_PREFIX / REF_TAG / FALLBACK_TEXT）正确
    3. SYSTEM_PROMPT 含 {retrieved_context} 占位符
    4. retrieve_context() 在 RAG 服务未初始化时优雅降级（不崩，返回提示文本）
    5. 两个子类的 name / description / capabilities 元信息不重复

不依赖外部服务（Qdrant / LLM），纯本地测试。
实例化时即使 EmbeddingService 初始化失败，也会被 try/except 捕获并降级。

运行方式：
    cd D:/code/tax_agent
    python -m tests.test_rag_sub_agent
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(str(PROJECT_ROOT))

# ================================================================
# 测试 1：基类 RAGSubAgent 本身不能直接实例化（概念验证）
# ================================================================
def test_base_class_placeholder_values():
    """验证基类的默认占位值，确认子类确实覆盖了它们"""
    from agents.sub_agents.rag_sub_agent import RAGSubAgent

    # 基类的默认值应该是空/默认值
    assert RAGSubAgent.NODE_NAME == "rag"
    assert RAGSubAgent.RETRIEVE_PREFIX == ""
    assert RAGSubAgent.REF_TAG == "参考"
    assert RAGSubAgent.FALLBACK_TEXT == "（未返回有效内容）"
    assert RAGSubAgent.name == "rag_sub_agent"
    print(f"  ✅ RAGSubAgent 基类占位值正确")

# ================================================================
# 测试 2：PolicyAgent 实例化 + 类属性
# ================================================================
def test_policy_agent_instantiation():
    """PolicyAgent 能实例化，且类属性都覆盖了基类默认值"""
    from agents.sub_agents.policy_agent import PolicyAgent
    from agents.sub_agents.rag_sub_agent import RAGSubAgent

    agent = PolicyAgent()

    # —— 1. 是 RAGSubAgent 的子类实例 ——
    assert isinstance(agent, RAGSubAgent), "❌ PolicyAgent 应该是 RAGSubAgent 的实例"
    print(f"  ✅ PolicyAgent 继承自 RAGSubAgent")

    # —— 2. 关键类属性覆盖了基类默认值 ——
    assert agent.NODE_NAME == "policy", f"❌ NODE_NAME 应为 'policy'，实际 '{agent.NODE_NAME}'"
    assert agent.RETRIEVE_PREFIX == "政策", f"❌ RETRIEVE_PREFIX 应为 '政策'，实际 '{agent.RETRIEVE_PREFIX}'"
    assert agent.REF_TAG == "政策参考", f"❌ REF_TAG 应为 '政策参考'，实际 '{agent.REF_TAG}'"
    assert "政策" in agent.FALLBACK_TEXT, f"❌ FALLBACK_TEXT 应包含'政策'，实际 '{agent.FALLBACK_TEXT}'"
    print(f"  ✅ 类属性覆盖正确: NODE_NAME='{agent.NODE_NAME}', RETRIEVE_PREFIX='{agent.RETRIEVE_PREFIX}'")

    # —— 3. SYSTEM_PROMPT 必须含 {retrieved_context} 占位符 ——
    assert "{retrieved_context}" in agent.SYSTEM_PROMPT, \
        "❌ SYSTEM_PROMPT 必须含 {retrieved_context} 占位符"
    print(f"  ✅ SYSTEM_PROMPT 含 {{retrieved_context}} 占位符（长度 {len(agent.SYSTEM_PROMPT)}）")

    # —— 4. 元信息 ——
    assert agent.name == "policy_agent"
    assert "政策" in agent.description
    assert isinstance(agent.capabilities, list) and len(agent.capabilities) >= 1
    print(f"  ✅ 元信息: name='{agent.name}', capabilities={agent.capabilities}")

# ================================================================
# 测试 3：DefinitionAgent 实例化 + 类属性
# ================================================================
def test_definition_agent_instantiation():
    """DefinitionAgent 能实例化，且类属性都覆盖了基类默认值"""
    from agents.sub_agents.definition_agent import DefinitionAgent
    from agents.sub_agents.rag_sub_agent import RAGSubAgent

    agent = DefinitionAgent()

    assert isinstance(agent, RAGSubAgent)

    assert agent.NODE_NAME == "definition", f"❌ 应为 'definition'，实际 '{agent.NODE_NAME}'"
    assert agent.RETRIEVE_PREFIX == "术语解释", f"❌ 应为 '术语解释'，实际 '{agent.RETRIEVE_PREFIX}'"
    assert agent.REF_TAG == "术语参考", f"❌ 应为 '术语参考'，实际 '{agent.REF_TAG}'"
    assert "术语" in agent.FALLBACK_TEXT
    print(f"  ✅ 类属性覆盖正确: NODE_NAME='{agent.NODE_NAME}', RETRIEVE_PREFIX='{agent.RETRIEVE_PREFIX}'")

    assert "{retrieved_context}" in agent.SYSTEM_PROMPT
    print(f"  ✅ SYSTEM_PROMPT 占位符正常")

    assert agent.name == "definition_agent"
    assert "术语" in agent.description
    print(f"  ✅ 元信息: name='{agent.name}'")

# ================================================================
# 测试 4：两个子类的类属性不重复（防止复制粘贴后忘记改）
# ================================================================
def test_sub_agents_distinct():
    """确保 PolicyAgent 和 DefinitionAgent 的 NODE_NAME / RETRIEVE_PREFIX 不相同"""
    from agents.sub_agents.policy_agent import PolicyAgent
    from agents.sub_agents.definition_agent import DefinitionAgent

    assert PolicyAgent.NODE_NAME != DefinitionAgent.NODE_NAME, \
        "❌ 两个子类的 NODE_NAME 不应该相同"
    assert PolicyAgent.RETRIEVE_PREFIX != DefinitionAgent.RETRIEVE_PREFIX, \
        "❌ 两个子类的 RETRIEVE_PREFIX 不应该相同"
    assert PolicyAgent.REF_TAG != DefinitionAgent.REF_TAG, \
        "❌ 两个子类的 REF_TAG 不应该相同"
    assert PolicyAgent.name != DefinitionAgent.name, \
        "❌ 两个子类的 name 不应该相同"

    # SYSTEM_PROMPT 第一个字应该不同（一个是"你是...政策"，一个是"你是...术语"）
    assert PolicyAgent.SYSTEM_PROMPT != DefinitionAgent.SYSTEM_PROMPT

    print(f"  ✅ PolicyAgent vs DefinitionAgent 属性完全不同（无复制粘贴错误）")

# ================================================================
# 测试 5：retrieve_context 优雅降级（核心！）
# ================================================================
def test_retrieve_context_degradation():
    """
    当 RAG 服务未初始化时（Qdrant 连不上 / store 为 None），
    retrieve_context() 不应该抛异常，而是返回友好的降级提示。
    """
    from agents.sub_agents.policy_agent import PolicyAgent
    from agents.sub_agents.definition_agent import DefinitionAgent

    # —— 方式 A：直接实例化（__init__ 里有 try/except，
    #     如果 Qdrant 连不上，_store 和 _embedding_service 会是 None）——
    policy = PolicyAgent()
    definition = DefinitionAgent()

    # 强制模拟未初始化状态（不管真实初始化是否成功）
    policy._store = None
    policy._embedding_service = None
    definition._store = None
    definition._embedding_service = None

    # —— 关键：这两个调用**不应该抛异常** ——
    result_policy = policy.retrieve_context("增值税留抵退税政策")
    result_def = definition.retrieve_context("什么是进项税")

    # 应该返回字符串提示，而不是崩溃
    assert isinstance(result_policy, str), f"❌ 返回值应为 str，实际 {type(result_policy)}"
    assert isinstance(result_def, str), f"❌ 返回值应为 str，实际 {type(result_def)}"
    assert len(result_policy) > 0, "❌ 降级提示不应为空"
    assert len(result_def) > 0, "❌ 降级提示不应为空"

    # 应该包含"未初始化"或类似提示
    assert "未初始化" in result_policy or "检索" in result_policy or "暂" in result_policy, \
        f"❌ 降级提示应友好，实际: '{result_policy}'"

    print(f"  ✅ PolicyAgent.retrieve_context 在 RAG 未初始化时返回降级文本: '{result_policy[:30]}...'")
    print(f"  ✅ DefinitionAgent.retrieve_context 在 RAG 未初始化时返回降级文本: '{result_def[:30]}...'")

# ================================================================
# 测试 6：retrieve_context 增强 query 前缀拼接
# ================================================================
def test_retrieve_context_prefix_logic():
    """验证 RETRIEVE_PREFIX 正确拼接到查询里（通过 mock EmbeddingService 验证）"""
    from unittest.mock import MagicMock, patch
    from agents.sub_agents.policy_agent import PolicyAgent

    agent = PolicyAgent()

    # 手动 mock embedding_service.search()，捕获它收到的 query 参数
    mock_service = MagicMock()
    mock_service.search.return_value = []  # 返回空结果
    agent._embedding_service = mock_service
    agent._store = MagicMock()  # 非 None 即可通过第一个 if

    # 调用
    agent.retrieve_context("增值税税率", limit=3)

    # 验证 search() 被调用了，且 query 里拼了 "政策" 前缀
    mock_service.search.assert_called_once()
    called_query = mock_service.search.call_args[0][0]  # 第一个位置参数
    assert "政策" in called_query, f"❌ 增强查询应包含 '政策' 前缀，实际 '{called_query}'"
    assert "增值税税率" in called_query
    print(f"  ✅ RETRIEVE_PREFIX 正确拼接: search() 收到 '{called_query}'")

# ================================================================
# 测试 7：PlanningAgent 的 RETRIEVE_PREFIX 必须是字符串
# ================================================================
def test_planning_agent_prefix_is_str():
    """PlanningAgent.RETRIEVE_PREFIX 曾经是 list，导致检索 query 被拼成
    "['税务筹划', '税务合规计划'] ..." 污染。这里固定为字符串。"""
    from agents.sub_agents.planning_agent import PlanningAgent

    prefix = PlanningAgent.RETRIEVE_PREFIX
    assert isinstance(prefix, str), f"❌ RETRIEVE_PREFIX 应为 str，实际 {type(prefix)}"
    assert "[" not in prefix, f"❌ 前缀不应包含 list 的 repr，实际 '{prefix}'"

# ================================================================
# 运行全部测试
# ================================================================
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING)  # 测试时把 import 时的 warning 压低

    print("\n" + "=" * 60)
    print("T8 — RAGSubAgent 基类 + 子类继承测试")
    print("=" * 60)

    tests = [
        ("基类占位值正确", test_base_class_placeholder_values),
        ("PolicyAgent 实例化 + 类属性", test_policy_agent_instantiation),
        ("DefinitionAgent 实例化 + 类属性", test_definition_agent_instantiation),
        ("两个子类属性不同（无复制粘贴错误）", test_sub_agents_distinct),
        ("retrieve_context 优雅降级", test_retrieve_context_degradation),
        ("RETRIEVE_PREFIX 前缀拼接", test_retrieve_context_prefix_logic),
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