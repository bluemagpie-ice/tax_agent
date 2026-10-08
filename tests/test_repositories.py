"""
T2 — Repositories 数据访问层测试
================================

测试目标：
    1. KnowledgeBaseRepository.create() 的新字段（created_by / kb_type / share_config）能真存真读
    2. get_all() / get_by_kb_id() / delete() 正常
    3. UserRepository / KnowledgeFileRepository 基本 CRUD

使用内存数据库（sqlite:///:memory:），不污染真实数据。
每个测试前重新建 engine + 建表，互不干扰。

运行方式：
    cd D:/code/tax_agent
    python -m tests.test_repositories
"""

import os
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(str(PROJECT_ROOT))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# ================================================================
# 测试 0：测试基础设施 —— 建内存数据库 + 替换 engine
# ================================================================
def _setup_memory_db():
    """
    创建全新的内存数据库，同时替换 models 和 repositories 的 engine。
    返回 models_module（方便后续查数据）。
    """
    import database.models as models_module
    import database.repositories as repo_module

    new_engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    new_session = sessionmaker(bind=new_engine)

    # 替换两处
    models_module.engine = new_engine
    models_module.SessionLocal = new_session
    repo_module.SessionLocal = new_session  # repositories 自己 import 了 SessionLocal

    # 建表
    models_module.Base.metadata.create_all(bind=new_engine)
    return models_module, repo_module

# ================================================================
# 测试 1：KnowledgeBaseRepository.create() —— 新字段真存真读
# ================================================================
def test_kb_create_with_new_fields():
    """验证 create() 传入的 created_by / kb_type / share_config 能正确存入"""
    import json
    mm, rm = _setup_memory_db()
    repo = rm.KnowledgeBaseRepository()

    share = json.dumps({
        "access_level": "private",
        "department_ids": ["dept_001"],
        "user_uids": ["user_001", "user_002"],
    })

    kb = repo.create(
        name="测试知识库",
        created_by="user_001",
        kb_type="qdrant",
        share_config=share,
    )

    # —— 刚 create 完就能读到正确值 ——
    assert kb.id is not None, "❌ create 后应该有 id"
    assert kb.name == "测试知识库"
    assert kb.created_by == "user_001", f"❌ created_by 应为 'user_001'，实际 '{kb.created_by}'"
    assert kb.kb_type == "qdrant", f"❌ kb_type 应为 'qdrant'，实际 '{kb.kb_type}'"
    assert kb.share_config == share, "❌ share_config 应该完整存下"

    # —— 用 get_all 读回来再验证一次 ——
    all_kbs = repo.get_all()
    assert len(all_kbs) == 1
    read_back = all_kbs[0]
    assert read_back.created_by == "user_001"
    assert read_back.kb_type == "qdrant"

    # —— get_by_kb_id 也能读到 ——
    found = repo.get_by_kb_id("测试知识库")
    assert found is not None
    assert found.created_by == "user_001"
    assert json.loads(found.share_config)["access_level"] == "private"

    print(f"  ✅ create 新字段真存真读: created_by='{kb.created_by}', kb_type='{kb.kb_type}'")

# ================================================================
# 测试 2：KnowledgeBaseRepository 全 CRUD
# ================================================================
def test_kb_full_crud():
    """create / get_all / get_by_kb_id / delete 完整链路"""
    mm, rm = _setup_memory_db()
    repo = rm.KnowledgeBaseRepository()

    # Create
    repo.create(name="库A", created_by="alice")
    repo.create(name="库B", created_by="bob")
    repo.create(name="库C", created_by="carol")

    all_kbs = repo.get_all()
    assert len(all_kbs) == 3
    print(f"  ✅ create × 3 成功, get_all 返回 3 条")

    # Read
    kb = repo.get_by_kb_id("库B")
    assert kb is not None
    assert kb.created_by == "bob"
    print(f"  ✅ get_by_kb_id('库B') 命中, created_by='{kb.created_by}'")

    # Delete 存在的
    ok = repo.delete("库A")
    assert ok is True
    remaining = repo.get_all()
    assert len(remaining) == 2
    assert repo.get_by_kb_id("库A") is None
    print(f"  ✅ delete('库A') 成功, 剩余 2 条")

    # Delete 不存在的 → 返回 False
    ok2 = repo.delete("不存在的库")
    assert ok2 is False
    print(f"  ✅ delete('不存在') 返回 False（幂等性 OK）")

# ================================================================
# 测试 3：KnowledgeBaseRepository.create() 默认值
# ================================================================
def test_kb_create_defaults():
    """不传 created_by / kb_type / share_config 时应该用默认值"""
    mm, rm = _setup_memory_db()
    repo = rm.KnowledgeBaseRepository()

    kb = repo.create(name="最小化知识库")

    assert kb.name == "最小化知识库"
    assert kb.created_by == "", f"❌ created_by 默认应为空字符串，实际 '{kb.created_by}'"
    assert kb.kb_type == "qdrant", f"❌ kb_type 默认应为 'qdrant'，实际 '{kb.kb_type}'"
    assert kb.share_config is None, f"❌ share_config 默认应为 None，实际 {kb.share_config}"

    print(f"  ✅ 默认值正确: created_by='{kb.created_by}', kb_type='{kb.kb_type}', share_config={kb.share_config}")

# ================================================================
# 测试 4：UserRepository 基本查询
# ================================================================
def test_user_repository():
    """UserRepository.get_by_uid"""
    mm, rm = _setup_memory_db()

    # 先直接往内存数据库里塞一个用户
    from database.models import User
    s = mm.SessionLocal()
    s.add(User(username="alice", create_at=datetime.now()))
    s.add(User(username="bob", create_at=datetime.now()))
    s.commit()
    s.close()

    repo = rm.UserRepository()

    # 查到存在的
    u = repo.get_by_uid("alice")
    assert u is not None
    assert u.username == "alice"
    print(f"  ✅ get_by_uid('alice') 命中")

    # 查不存在的 → 返回 None
    u2 = repo.get_by_uid("charlie")
    assert u2 is None
    print(f"  ✅ get_by_uid('charlie') 返回 None（不存在时）")

# ================================================================
# 测试 5：KnowledgeFileRepository
# ================================================================
def test_knowledge_file_repository():
    """KnowledgeFileRepository.create + get_by_id"""
    mm, rm = _setup_memory_db()
    repo = rm.KnowledgeFileRepository()

    # Create
    f1 = repo.create(path="/docs/tax_policy.pdf", tags="政策")
    assert f1.id is not None
    assert f1.path == "/docs/tax_policy.pdf"
    assert f1.tags == "政策"
    print(f"  ✅ create 成功: id={f1.id}, path='{f1.path}'")

    # Create 第二个
    f2 = repo.create(path="/docs/definition.docx")
    assert f2.id is not None
    print(f"  ✅ 第二个 create 成功: id={f2.id}")

    # get_by_id
    found = repo.get_by_id(f1.id)
    assert found is not None
    assert found.path == "/docs/tax_policy.pdf"
    print(f"  ✅ get_by_id({f1.id}) 命中")

    # get_by_id 不存在
    missing = repo.get_by_id(9999)
    assert missing is None
    print(f"  ✅ get_by_id(9999) 返回 None（不存在时）")

# ================================================================
# 运行全部测试
# ================================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("T2 — Repositories 数据访问层测试")
    print("=" * 60)

    tests = [
        ("KnowledgeBaseRepository.create 新字段真存真读", test_kb_create_with_new_fields),
        ("KnowledgeBaseRepository 全 CRUD 链路", test_kb_full_crud),
        ("KnowledgeBaseRepository.create 默认值", test_kb_create_defaults),
        ("UserRepository 基本查询", test_user_repository),
        ("KnowledgeFileRepository create + get_by_id", test_knowledge_file_repository),
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