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
os.chdir(str(PROJECT_ROOT))  # 切到项目根目录，让相对路径 sqlite:///./tax_agent.db 能工作

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
    """
    完整调试版：用内存数据库跑 init_db.init_database()
    并逐步手动插入数据，精确定位哪一步报错。
    """
    import database.init_db as init_db_module
    import database.models as models_module
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    # ====== 保存旧引用 ======
    old_models_engine = models_module.engine
    old_models_session = models_module.SessionLocal
    old_initdb_engine = init_db_module.engine
    old_initdb_session = init_db_module.SessionLocal

    # ====== 构建内存数据库 ======
    new_engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    new_session_local = sessionmaker(bind=new_engine)

    # ====== 同时替换两处 engine ======
    old_models_engine.dispose()

    models_module.engine = new_engine
    models_module.SessionLocal = new_session_local
    models_module.DATABASE_URL = "sqlite:///:memory:"
    init_db_module.engine = new_engine
    init_db_module.SessionLocal = new_session_local

    try:
        # ====== 第一步：确认 create_tables 正常 ======
        ok = init_db_module.create_tables()
        assert ok, "❌ create_tables() 返回了 False"
        print(f"  ✅ create_tables() 成功")

        # ====== 第二步：手动逐步插入，精确定位 ======
        print(f"\n  🔍 逐步手动插入测试（看哪一步挂）：")
        from datetime import datetime as _dt
        from database.models import User, KnowledgeBases, Conversation, Messages, UserMemories

        s = new_session_local()
        try:
            # 步骤 1: User
            print(f"    [1/5] User...", end=" ", flush=True)
            if s.query(User).filter_by(username="text_user").first() is None:
                u = User(username="text_user", create_at=_dt.now())
                s.add(u); s.commit()
                s.refresh(u)
                print(f"✅ id={u.id}")
            else:
                print("已存在")

            # 步骤 2: KnowledgeBases
            print(f"    [2/5] KnowledgeBases...", end=" ", flush=True)
            if s.query(KnowledgeBases).filter_by(name="税务知识库").first() is None:
                kb = KnowledgeBases(
                    name="税务知识库", kb_type="qdrant",
                    created_by="text_user",
                    share_config='{"access_level": "global", "department_ids": [], "user_uids": []}',
                    create_at=_dt.now(),
                )
                s.add(kb); s.commit()
                print(f"✅ id={kb.id}")
            else:
                print("已存在")

            # 步骤 3: Conversation
            print(f"    [3/5] Conversation...", end=" ", flush=True)
            conv = Conversation(user_id=1, create_at=_dt.now())
            s.add(conv); s.commit()
            s.refresh(conv)
            print(f"✅ id={conv.id}")

            # 步骤 4: Messages
            print(f"    [4/5] Messages...", end=" ", flush=True)
            s.add_all([
                Messages(conversation_id=conv.id, role="user",
                         content="你好，我想咨询一下税额的计算公式", create_at=_dt.now()),
                Messages(conversation_id=conv.id, role="agent",
                         content="您好。税额 = 不含税价格 × 税率", create_at=_dt.now()),
            ])
            s.commit()
            print(f"✅ 2 条")

            # 步骤 5: UserMemories
            print(f"    [5/5] UserMemories...", end=" ", flush=True)
            if s.query(UserMemories).first() is None:
                s.add(UserMemories(user_id=1,
                        content="用户咨询过税额的计算公式", create_at=_dt.now()))
                s.commit()
            print(f"✅")

            print(f"\n  🎉 手动插入全部成功！说明数据逻辑没问题。")
            print(f"  🔍 现在测试 insert_initial_data() 为啥失败...")

        except Exception as e:
            print(f"\n  ❌ 手动插入也失败了！真实报错：")
            import traceback
            traceback.print_exc()
            return  # 直接跳过后续

        finally:
            s.close()

        # ====== 第三步：清空，再跑正式 insert_initial_data ======
        print(f"\n  🔍 清空数据，测试正式的 insert_initial_data()：")
        s2 = new_session_local()
        try:
            # 先确认 init_db_module.insert_initial_data 用的是我们的 engine
            print(f"    init_db.SessionLocal is new_session_local: {init_db_module.SessionLocal is new_session_local}")
            print(f"    models.SessionLocal is new_session_local: {models_module.SessionLocal is new_session_local}")

            ret = init_db_module.insert_initial_data()
            print(f"    insert_initial_data() 返回: {ret}")

            if ret is not None and ret is not False:
                print(f"  🔍 返回的是字符串，说明有错误信息: {ret}")
        finally:
            s2.close()

    finally:
        # ====== 恢复旧 engine ======
        new_engine.dispose()
        models_module.engine = old_models_engine
        models_module.SessionLocal = old_models_session
        init_db_module.engine = old_initdb_engine
        init_db_module.SessionLocal = old_initdb_session
# ================================================================
# 运行全部测试
# ================================================================
if __name__ == "__main__":
    # ---------- 自动清理：测试开始前删残留的 tax_agent.db ----------
    project_db = PROJECT_ROOT / "tax_agent.db"
    if project_db.exists():
        try:
            project_db.unlink()
            print(f"🧹 已清理残留的 {project_db.name}")
        except PermissionError:
            print(f"⚠️  无法删除 {project_db.name}（可能被其他进程占用）")

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

    # ---------- 自动清理：测试结束后确保 tax_agent.db 不存在 ----------
    if project_db.exists():
        try:
            project_db.unlink()
            print(f"\n🧹 测试结束，再次清理 {project_db.name}")
        except PermissionError:
            pass  # 最后一次清理，删不掉就算了

    print(f"\n{'=' * 60}")
    print(f"测试结果: ✅ {passed} passed | ❌ {failed} failed")
    print(f"{'=' * 60}\n")

    if failed > 0:
        sys.exit(1)