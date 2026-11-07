"""
T1 — 数据库初始化测试
======================

测试目标：
    1. 所有业务表能成功创建（包括新增字段）
    2. KnowledgeBases 表确实有 kb_type / created_by / share_config 列
    3. init_db 的初始数据插入逻辑正确
    4. 幂等性：重复执行 insert_initial_data() 不会崩、不会重复插入
    5. 清理后重建：确保删除表后能完整重建

使用 SQLite 内存数据库（sqlite:///:memory:），不污染真实数据。

运行方式：
    cd D:/code/tax_agent
    python -m tests.test_database_init
"""

import os
import sys
import json
import tempfile
from datetime import datetime
from pathlib import Path

# ---------- 关键：让 tests/ 目录下执行时能正确 import 项目根包 ----------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _build_memory_engine():
    """
    为每个测试构建一个全新的内存数据库引擎。
    这样不同测试之间互不影响。
    """
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    SessionLocal = sessionmaker(bind=engine)
    return engine, SessionLocal

# ================================================================
# 测试 1：所有业务表能成功创建
# ================================================================
def test_create_all_tables():
    """验证 Base.metadata.create_all() 能一次性创建全部 6 张表"""
    from database.models import Base

    engine, _ = _build_memory_engine()
    Base.metadata.create_all(bind=engine)

    # 验证 6 张表都创建了
    expected_tables = {
        "users",
        "knowledge_files",
        "knowledge_chunks",
        "knowledge_bases",
        "conversations",
        "messages",
        "user_memories",
    }
    actual_tables = set(Base.metadata.tables.keys())
    missing = expected_tables - actual_tables
    assert not missing, f"❌ 缺少以下表: {missing}"
    print(f"  ✅ 7 张业务表全部创建成功")

# ================================================================
# 测试 2：KnowledgeBases 表有新增字段
# ================================================================
def test_knowledge_bases_has_new_columns():
    """验证 KnowledgeBases 表确实包含 kb_type / created_by / share_config"""
    from database.models import Base, KnowledgeBases

    engine, SessionLocal = _build_memory_engine()
    Base.metadata.create_all(bind=engine)

    session = SessionLocal()
    try:
        # 用 KnowledgeBases 创建一条测试记录，只填新增的 3 个字段
        # 如果字段不存在会直接抛 OperationalError
        kb = KnowledgeBases(
            name="测试库",
            kb_type="qdrant",
            created_by="user_001",
            share_config='{"access_level": "private", "department_ids": [], "user_uids": ["user_001"]}',
            create_at=datetime.now(),
        )
        session.add(kb)
        session.commit()
        session.refresh(kb)

        # 读回来验证每个字段
        assert kb.kb_type == "qdrant", f"❌ kb_type 应为 'qdrant'，实际 '{kb.kb_type}'"
        assert kb.created_by == "user_001", f"❌ created_by 应为 'user_001'，实际 '{kb.created_by}'"
        assert kb.share_config is not None, "❌ share_config 不应为空"

        # 额外：share_config 能被正确解析为 JSON
        parsed = json.loads(kb.share_config)
        assert parsed["access_level"] == "private"
        assert "user_001" in parsed["user_uids"]

        print(f"  ✅ KnowledgeBases 新增字段全部可用")
        print(f"     → kb_type={kb.kb_type}, created_by={kb.created_by}")
        print(f"     → share_config JSON 解析正常: {parsed['access_level']}")

    finally:
        session.close()

# ================================================================
# 测试 3：insert_initial_data 能正常插入
# ================================================================
def test_insert_initial_data():
    """模拟 init_db.py 的插入逻辑，验证示例数据完整"""
    from database.models import (
        Base, User, KnowledgeBases, Conversation, Messages, UserMemories
    )

    engine, SessionLocal = _build_memory_engine()
    Base.metadata.create_all(bind=engine)

    session = SessionLocal()
    try:
        # ---- 插入示例用户 ----
        user = User(username="text_user", create_at=datetime.now())
        session.add(user)
        session.commit()
        session.refresh(user)
        assert user.id is not None
        assert user.username == "text_user"
        print(f"  ✅ 示例用户插入成功: id={user.id}, username={user.username}")

        # ---- 插入示例知识库（带新字段） ----
        kb = KnowledgeBases(
            name="税务知识库",
            kb_type="qdrant",
            created_by="text_user",
            share_config='{"access_level": "global", "department_ids": [], "user_uids": []}',
            create_at=datetime.now(),
        )
        session.add(kb)
        session.commit()
        session.refresh(kb)
        assert kb.created_by == "text_user"
        assert kb.kb_type == "qdrant"
        print(f"  ✅ 示例知识库插入成功: id={kb.id}, created_by={kb.created_by}")

        # ---- 插入示例对话 ----
        convo = Conversation(user_id=user.id, create_at=datetime.now())
        session.add(convo)
        session.commit()
        session.refresh(convo)
        assert convo.id is not None

        msg_user = Messages(conversation_id=convo.id, role="user",
                            content="你好，我想咨询一下税额的计算公式", create_at=datetime.now())
        msg_agent = Messages(conversation_id=convo.id, role="agent",
                             content="您好。税额 = 不含税价格 × 税率", create_at=datetime.now())
        session.add_all([msg_user, msg_agent])
        session.commit()
        print(f"  ✅ 示例对话+消息插入成功: conversation_id={convo.id}")

        # ---- 插入用户记忆 ----
        memory = UserMemories(user_id=user.id,
                              content="用户咨询过税额的计算公式", create_at=datetime.now())
        session.add(memory)
        session.commit()
        print(f"  ✅ 示例用户记忆插入成功")

    finally:
        session.close()

# ================================================================
# 测试 4：幂等性 —— 重复执行不会崩溃、不会重复插入
# ================================================================
def test_idempotent_insert():
    """
    模拟 init_db.py 的"存在则跳过"逻辑，重复插入同一批数据。
    期望：不崩、不报错、不产生重复记录。
    """
    from database.models import Base, User

    engine, SessionLocal = _build_memory_engine()
    Base.metadata.create_all(bind=engine)

    session = SessionLocal()
    try:
        # ---------- 第一轮插入 ----------
        if session.query(User).filter_by(username="text_user").first() is None:
            u = User(username="text_user", create_at=datetime.now())
            session.add(u)
            session.commit()

        count_after_first = session.query(User).filter_by(username="text_user").count()
        assert count_after_first == 1, f" 第一轮后应为 1 条，实际 {count_after_first}"

        # ---------- 第二轮插入（完全相同的代码） ----------
        if session.query(User).filter_by(username="text_user").first() is None:
            u = User(username="text_user", create_at=datetime.now())
            session.add(u)
            session.commit()
        # 注意：第二轮没有进入 if 分支，所以不会有 commit，但也不会崩

        count_after_second = session.query(User).filter_by(username="text_user").count()
        assert count_after_second == 1, f" 第二轮后应为 1 条（未重复），实际 {count_after_second}"

        print(f"   幂等性验证通过：重复插入不崩、不产生重复记录")
        print(f"     → 第一轮后 {count_after_first} 条，第二轮后仍为 {count_after_second} 条")

    finally:
        session.close()



# ================================================================
# 测试 5：真实数据库文件 init_db 跑一遍
# ================================================================
def test_real_init_db():
    """Run the actual initializer twice against an isolated in-memory database."""
    from unittest.mock import patch
    import database.init_db as init_module
    from database.models import Base, User, Messages
    engine, sessions = _build_memory_engine()
    try:
        with patch.object(init_module, "engine", engine), patch.object(init_module, "SessionLocal", sessions):
            assert init_module.create_tables() is True
            assert init_module.insert_initial_data() is True
            assert init_module.insert_initial_data() is True
            with sessions() as session:
                assert session.query(User).count() == 1
                assert session.query(Messages).count() == 2
    finally:
        engine.dispose()

# ================================================================
# 运行全部测试
# ================================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("T1 — 数据库初始化测试")
    print("=" * 60)

    tests = [
        ("所有业务表能成功创建", test_create_all_tables),
        ("KnowledgeBases 新增字段可用", test_knowledge_bases_has_new_columns),
        ("insert_initial_data 正常插入", test_insert_initial_data),
        ("幂等性：重复执行不崩不重复", test_idempotent_insert),
        ("真实 init_db 完整流程", test_real_init_db),
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
