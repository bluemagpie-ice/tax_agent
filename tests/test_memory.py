"""
T10 — Memory 三层记忆测试

测试点：
  1. record_turn() 能正常存储对话消息
  2. context_for() 返回最近 N 条工作记忆
  3. _compress() 滚动窗口正确将旧消息压缩成摘要
  4. 多轮对话后能提取出纳税人类型、关注税种等用户画像
  5. 跨租户隔离：user_001 的记忆 user_002 看不到
"""

import tempfile
from pathlib import Path

import pytest

from agents.memory import ConversationMemory
from services.security import tenant_scope, current_tenant_id

# ------------------------------------------------------------------ 工具：创建临时 Memory 实例
@pytest.fixture
def memory(tmp_path):
    """使用 pytest 内置的临时目录，自动在测试结束后清理。"""
    return ConversationMemory(
        base_dir=tmp_path,
        compress_at=6,       # 压缩阈值调小，方便测试（默认18）
        retain_recent=2,     # 压缩后保留最近2条（默认8）
        context_budget_tokens=2400,
    )

SESSION = "test_session_001"

# ================================================================== 1. 基础存储
class TestBasicStorage:
    def test_record_turn_stores_messages(self, memory):
        """record_turn 能正常存储一轮（user + assistant）对话"""
        with tenant_scope("tenant_A"):
            result = memory.record_turn(
                session_id=SESSION,
                user_message="你好，我想咨询增值税发票问题",
                assistant_message="您好，请描述一下具体情况~",
            )
            assert result["turns"] == 1
            session = memory.get_session(SESSION)
            assert session is not None
            assert len(session["messages"]) == 2  # user + assistant

    def test_record_multiple_turns(self, memory):
        """多轮对话能累积存储（不触发压缩的情况下）"""
        with tenant_scope("tenant_A"):
            for i in range(2):  # 2轮×2条=4条，未达到compress_at=6，不会触发压缩
                memory.record_turn(
                    SESSION,
                    f"第{i+1}轮用户消息",
                    f"第{i+1}轮助手回复",
                )
            session = memory.get_session(SESSION)
            assert session["turns"] == 2
            assert len(session["messages"]) == 4

# ================================================================== 2. 工作记忆（最近 N 条）
class TestWorkingMemory:
    def test_context_for_returns_recent_messages(self, memory):
        """context_for 返回的 recent_messages 是最近的 N 条"""
        with tenant_scope("tenant_A"):
            for i in range(5):
                memory.record_turn(SESSION, f"用户{i}", f"助手{i}")
            ctx = memory.context_for(SESSION, "随便问点什么")

        # recent_messages 是 context_for 里截断后的最近 6 条（源码硬编码 [-6:]）
        assert len(ctx["recent_messages"]) <= 6
        # 最后一条应该是最新的助手回复
        assert ctx["recent_messages"][-1]["content"] == "助手4"

    def test_working_memory_layers_report(self, memory):
        """memory_layers 字典正确反映三层记忆规模"""
        with tenant_scope("tenant_A"):
            memory.record_turn(SESSION, "问1", "答1")
            ctx = memory.context_for(SESSION, "继续")
        layers = ctx["memory_layers"]
        assert layers["working"] == 2  # 2 条消息（1轮）
        assert "episodic" in layers
        assert "profile_fields" in layers

# ================================================================== 3. 滚动窗口压缩
class TestCompress:
    def test_compress_triggers_at_threshold(self, memory):
        """消息数 >= compress_at 时自动触发压缩"""
        # compress_at=6, retain_recent=2
        with tenant_scope("tenant_A"):
            # 连续 record_turn，每条加2个消息，3轮=6条 → 触发压缩
            for i in range(3):
                memory.record_turn(SESSION, f"用户{i}", f"助手{i}")

            session = memory.get_session(SESSION)

        # 压缩后 messages 只保留最近 retain_recent=2 条
        assert len(session["messages"]) == 2
        # 旧消息被压缩进 summary 或 episodes
        assert session["summary"], "摘要不应为空"
        assert len(session["episodes"]) >= 1, "至少生成了一个 episode"

    def test_compress_preserves_recent(self, memory):
        """压缩后最近的消息顺序和内容完好"""
        with tenant_scope("tenant_A"):
            for i in range(4):
                memory.record_turn(SESSION, f"用户{i}", f"助手{i}")

            session = memory.get_session(SESSION)

        # 最后一轮的助手回复应该还在
        contents = [m["content"] for m in session["messages"]]
        assert "助手3" in contents

    def test_multiple_compressions_accumulate(self, memory):
        """多轮压缩会累积 episodes（上限 20）"""
        with tenant_scope("tenant_A"):
            # 塞 20 轮 → 每次压缩保留 2 条，会触发多次压缩
            for i in range(20):
                memory.record_turn(SESSION, f"用户{i}", f"助手{i}")

            session = memory.get_session(SESSION)

        assert len(session["episodes"]) >= 1

# ================================================================== 4. 用户画像提取
class TestProfileExtraction:
    def test_taxpayer_type_extraction(self, memory):
        """能从对话中提取纳税人类型"""
        with tenant_scope("tenant_A"):
            memory.record_turn(SESSION, "我是小规模纳税人", "好的，已记录")
            memory.record_turn(SESSION, "之后转一般纳税人了", "好的~")
            session = memory.get_session(SESSION)

        types = session["profile"]["taxpayer_types"]
        assert "小规模纳税人" in types
        assert "一般纳税人" in types

    def test_tax_topics_extraction(self, memory):
        """能识别关注的税种/话题"""
        with tenant_scope("tenant_A"):
            memory.record_turn(SESSION, "我想咨询增值税发票抵扣和汇算清缴", "好的")
            session = memory.get_session(SESSION)

        topics = session["profile"]["topics"]
        assert "增值税" in topics
        assert "发票" in topics
        assert "抵扣" in topics
        assert "汇算清缴" in topics

    def test_tax_year_extraction(self, memory):
        """能识别年份"""
        with tenant_scope("tenant_A"):
            memory.record_turn(SESSION, "我要做2024年的企业所得税汇算清缴", "好的")
            session = memory.get_session(SESSION)

        years = session["profile"]["tax_years"]
        assert "2024" in years or "2024年" in str(years)

    def test_intent_and_agent_from_metadata(self, memory):
        """metadata 里的 intent/agent 能写入画像"""
        with tenant_scope("tenant_A"):
            memory.record_turn(
                SESSION, "问点东西", "答",
                metadata={"intent": "咨询政策", "agent": "policy_agent"},
            )
            session = memory.get_session(SESSION)

        assert session["profile"]["last_intent"] == "咨询政策"
        assert session["profile"]["last_agent"] == "policy_agent"

# ================================================================== 5. 跨租户隔离
class TestTenantIsolation:
    def test_user001_cannot_see_user002_session(self, memory):
        """user_001 创建的 session，user_002 看不到"""
        with tenant_scope("user_001"):
            memory.record_turn("session_X", "私密对话", "私密回复")

        with tenant_scope("user_002"):
            session = memory.get_session("session_X")
            assert session is None, "user_002 不应该看到 user_001 的会话"

    def test_list_sessions_isolated(self, memory):
        """list_sessions 只返回当前租户的"""
        with tenant_scope("user_001"):
            memory.record_turn("s1", "u1", "a1")
            memory.record_turn("s2", "u2", "a2")

        with tenant_scope("user_002"):
            memory.record_turn("s3", "u3", "a3")

        with tenant_scope("user_001"):
            sessions = memory.list_sessions()
            ids = {s["session_id"] for s in sessions}
            assert ids == {"s1", "s2"}

        with tenant_scope("user_002"):
            sessions = memory.list_sessions()
            ids = {s["session_id"] for s in sessions}
            assert ids == {"s3"}

    def test_stats_isolated(self, memory):
        """stats() 按租户隔离"""
        with tenant_scope("user_001"):
            for _ in range(3):
                memory.record_turn("s1", "u", "a")
            s1 = memory.stats()

        with tenant_scope("user_002"):
            memory.record_turn("s2", "u", "a")
            s2 = memory.stats()

        assert s1["turns"] == 3
        assert s2["turns"] == 1


# ================================================================== 6. 跨会话用户画像（记忆效果检测）
class TestCrossSessionProfile:
    def test_profile_accumulates_across_sessions(self, memory):
        """同一租户下，不同会话的画像应合并进跨会话用户画像。"""
        with tenant_scope("tenant_A"):
            memory.record_turn("session_1", "我是小规模纳税人，关注增值税", "好的")
            memory.record_turn("session_2", "我是个体工商户，关注个人所得税", "好的")
            ctx = memory.context_for("session_3", "继续问")

        profile = ctx["profile"]
        assert "小规模纳税人" in profile.get("taxpayer_types", [])
        assert "个体工商户" in profile.get("taxpayer_types", [])
        assert "增值税" in profile.get("topics", [])
        assert "个人所得税" in profile.get("topics", [])

    def test_prompt_injects_cross_session_profile(self, memory):
        """context_for 的 prompt_text 应包含跨会话画像信息（真正注入给 LLM）。"""
        with tenant_scope("tenant_A"):
            memory.record_turn("s1", "我是小规模纳税人", "好的")
            memory.record_turn("s2", "我是个体工商户", "好的")
            ctx = memory.context_for("s3", "继续")

        assert "用户画像" in ctx["prompt_text"]
        assert "小规模纳税人" in ctx["prompt_text"]
        assert "个体工商户" in ctx["prompt_text"]

    def test_profile_is_tenant_isolated(self, memory):
        """跨会话画像同样按租户隔离。"""
        with tenant_scope("user_001"):
            memory.record_turn("s1", "我是小规模纳税人", "好的")
        with tenant_scope("user_002"):
            memory.record_turn("s2", "我是个体工商户", "好的")

        with tenant_scope("user_001"):
            ctx1 = memory.context_for("s3", "问")
        with tenant_scope("user_002"):
            ctx2 = memory.context_for("s4", "问")

        assert "小规模纳税人" in ctx1["profile"].get("taxpayer_types", [])
        assert "个体工商户" not in ctx1["profile"].get("taxpayer_types", [])
        assert "个体工商户" in ctx2["profile"].get("taxpayer_types", [])


# ================================================================== 7. 对话标题生成
class TestTitleGeneration:
    def test_title_falls_back_to_truncation_without_llm(self, memory):
        """未配置标题 LLM 时，回退为前 20 字截断。"""
        msg = "你好我想咨询一下增值税一般纳税人的认定标准是什么"
        with tenant_scope("tenant_A"):
            memory.record_turn("s1", msg, "答")
            session = memory.get_session("s1")
        assert session["title"] == msg.strip()[:20] + "…"

    def test_title_uses_llm_when_available(self, memory):
        """配置了标题 LLM 时，使用 LLM 概括出的标题。"""
        class _FakeLLM:
            def invoke(self, messages):
                class _R:
                    content = "增值税一般纳税人认定"
                return _R()

        memory.set_title_llm(_FakeLLM())
        with tenant_scope("tenant_A"):
            memory.record_turn("s1", "你好我想咨询一下增值税一般纳税人的认定标准是什么", "答")
            session = memory.get_session("s1")
        assert session["title"] == "增值税一般纳税人认定"

    def test_title_falls_back_on_llm_error(self, memory):
        """LLM 概括抛异常时，回退为前 20 字截断（不崩）。"""
        class _BadLLM:
            def invoke(self, messages):
                raise RuntimeError("LLM down")

        memory.set_title_llm(_BadLLM())
        msg = "你好我想咨询一下增值税一般纳税人的认定标准是什么"
        with tenant_scope("tenant_A"):
            memory.record_turn("s1", msg, "答")
            session = memory.get_session("s1")
        assert session["title"] == msg.strip()[:20] + "…"


# ================================================================== 8. 会话重命名
class TestRenameSession:
    def test_rename_session(self, memory):
        with tenant_scope("tenant_A"):
            memory.record_turn("s1", "你好", "答")
            assert memory.rename_session("s1", "新标题") is True
            session = memory.get_session("s1")
            assert session["title"] == "新标题"

    def test_rename_missing_session_returns_false(self, memory):
        with tenant_scope("tenant_A"):
            assert memory.rename_session("no_such", "x") is False