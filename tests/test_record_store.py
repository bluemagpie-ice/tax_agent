"""
T4 — RecordStore 多租户隔离 + T6 — PersistentDocumentStore 类级共享
====================================================================

测试目标：
    RecordStore（SQLite 多租户键值存储）：
        1. 绑租户后 put/get/list/delete 正常
        2. 没绑租户时 put() 抛 PermissionError
        3. A 租户的数据 B 租户看不到（物理隔离）
        4. 跨租户写入被拒（TenantBoundaryError）

    PersistentDocumentStore（类级共享分块存储）：
        5. 两个实例共享同一份 chunks
        6. count() / clear() 类方法正确
        7. add_chunks → get_chunk_by_id 链路通

不依赖外部服务，全部用临时 SQLite 文件 + 内存。

运行方式：
    cd D:/code/tax_agent
    python -m tests.test_record_store
"""

import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(str(PROJECT_ROOT))

from services.security import (
    current_tenant_id,
    bind_tenant,
    reset_tenant,
    tenant_scope,
)
from services.record_store import (
    SQLiteRecordStore,
    TenantBoundaryError,
    RecordConflictError,
)

# ================================================================
# RecordStore 基础设施
# ================================================================
def _make_store():
    """创建一个指向临时文件的 RecordStore（自动清理）"""
    tmp = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False)
    tmp.close()
    store = SQLiteRecordStore(path=Path(tmp.name))
    store._tmp_path = Path(tmp.name)  # 挂上去方便后面清理
    return store

# ================================================================
# T4-1：绑租户后 put/get 正常
# ================================================================
def test_put_get_with_tenant():
    """绑定租户后 put 成功、get 能读回来"""
    store = _make_store()

    with tenant_scope("user_alice"):
        version = store.put(
            namespace="profile",
            record_id="default",
            payload={"nickname": "爱丽丝", "level": 3},
        )
        assert version == 1, f"❌ 首次 put 应返回 version=1，实际 {version}"

        record = store.get(namespace="profile", record_id="default")
        assert record is not None
        assert record["nickname"] == "爱丽丝"
        assert record["level"] == 3
        # tenant_id 被强制注入
        assert record["tenant_id"] == "user_alice"

        # 更新同一 record_id → version 递增
        version2 = store.put(
            namespace="profile",
            record_id="default",
            payload={"nickname": "爱丽丝", "level": 5},
        )
        assert version2 == 2

    print(f"  ✅ 绑租户后 put/get 正常，version 递增正确")

# ================================================================
# T4-2：没绑租户时 put() 抛 PermissionError
# ================================================================
def test_put_without_tenant_raises():
    """default 状态下调用 put 必须抛 PermissionError"""
    store = _make_store()

    # 先确保是 default
    assert current_tenant_id() == "default"

    try:
        store.put(namespace="profile", record_id="x", payload={"a": 1})
        assert False, "❌ 这里应该抛 PermissionError"
    except PermissionError as e:
        assert "未绑定租户" in str(e), f"❌ 错误消息应含提示，实际 '{e}'"
        print(f"  ✅ 未绑租户时 put() 正确抛 PermissionError")

# ================================================================
# T4-3：A 租户写的，B 租户看不到
# ================================================================
def test_tenant_isolation():
    """两个租户各写各的，互相隔离"""
    store = _make_store()

    # Alice 写 profile
    with tenant_scope("user_alice"):
        store.put(namespace="profile", record_id="x", payload={"name": "爱丽丝"})

    # Bob 写 profile（同一个 record_id，但属于不同租户）
    with tenant_scope("user_bob"):
        store.put(namespace="profile", record_id="x", payload={"name": "鲍勃"})
        # Bob 看不到 Alice 的
        bob_record = store.get(namespace="profile", record_id="x")
        assert bob_record["name"] == "鲍勃", f"❌ Bob 应读到自己的数据，实际 {bob_record}"

    # 切回 Alice → 看到自己的，看不到 Bob 的
    with tenant_scope("user_alice"):
        alice_record = store.get(namespace="profile", record_id="x")
        assert alice_record["name"] == "爱丽丝", f"❌ Alice 应读到自己的数据"

    # list() 也隔离
    with tenant_scope("user_alice"):
        alice_list = store.list(namespace="profile")
        assert len(alice_list) == 1

    with tenant_scope("user_bob"):
        bob_list = store.list(namespace="profile")
        assert len(bob_list) == 1

    print(f"  ✅ 两个租户数据互相隔离（物理隔离 OK）")

# ================================================================
# T4-4：跨租户写入被拒绝（TenantBoundaryError）
# ================================================================
def test_cross_tenant_write_rejected():
    """payload 里偷偷塞别人的 tenant_id 应该被拒绝"""
    store = _make_store()

    with tenant_scope("user_alice"):
        try:
            store.put(
                namespace="profile",
                record_id="hack",
                payload={"tenant_id": "user_bob", "hack": True},  # 恶意伪造
            )
            assert False, "❌ 应该抛 TenantBoundaryError"
        except TenantBoundaryError as e:
            assert "跨租户" in str(e)
            print(f"  ✅ 跨租户伪造 tenant_id 被正确拒绝")

# ================================================================
# T4-5：delete 只删自己租户的
# ================================================================
def test_delete_isolated():
    """delete 只影响当前租户的数据"""
    store = _make_store()

    with tenant_scope("user_alice"):
        store.put(namespace="session", record_id="s1", payload={"data": "alice_data"})

    with tenant_scope("user_bob"):
        store.put(namespace="session", record_id="s1", payload={"data": "bob_data"})
        ok = store.delete(namespace="session", record_id="s1")
        assert ok is True

    # Alice 的还在
    with tenant_scope("user_alice"):
        r = store.get(namespace="session", record_id="s1")
        assert r is not None, "❌ Alice 的数据不应该被 Bob 的 delete 影响"

    # Bob 的没了
    with tenant_scope("user_bob"):
        r2 = store.get(namespace="session", record_id="s1")
        assert r2 is None

    print(f"  ✅ delete 也是租户隔离的")

# ================================================================
# T6-1：PersistentDocumentStore 类级共享
# ================================================================
def test_persistent_store_shared():
    """两个实例 add_chunks / get_chunks 互相可见"""
    from knowledge.embedding import PersistentDocumentStore, ChunkDocument

    # 先清干净
    PersistentDocumentStore.clear()
    assert PersistentDocumentStore.count() == 0

    store_a = PersistentDocumentStore()
    store_b = PersistentDocumentStore()

    # store_a 加 2 个 chunk
    store_a.add_chunks([
        ChunkDocument(id="chunk_001", content="增值税是第一大税种", file_id="f1"),
        ChunkDocument(id="chunk_002", content="企业所得税基本税率25%", file_id="f1"),
    ])

    # store_b 应该立刻能看到
    chunks = store_b.get_chunks()
    assert len(chunks) == 2, f"❌ store_b 应看到 2 个，实际 {len(chunks)}"

    # store_b 再加 1 个
    store_b.add_chunks([
        ChunkDocument(id="chunk_003", content="小规模纳税人征收率3%", file_id="f2"),
    ])

    # store_a 也能看到
    chunks_a = store_a.get_chunks()
    assert len(chunks_a) == 3
    assert PersistentDocumentStore.count() == 3

    print(f"  ✅ PersistentDocumentStore 两个实例完全共享: {len(chunks_a)} 个 chunk")

# ================================================================
# T6-2：get_chunk_by_id 跨实例可用
# ================================================================
def test_get_chunk_by_id_cross_instance():
    """一个实例 add 的 chunk，另一个实例能通过 id 查到"""
    from knowledge.embedding import PersistentDocumentStore, ChunkDocument

    PersistentDocumentStore.clear()

    s1 = PersistentDocumentStore()
    s2 = PersistentDocumentStore()

    s1.add_chunks([
        ChunkDocument(id="abc_123", content="测试内容", file_id="f1"),
    ])

    found = s2.get_chunk_by_id("abc_123")
    assert found is not None
    assert found.content == "测试内容"
    assert found.file_id == "f1"

    # 不存在的 id → 返回 None
    missing = s2.get_chunk_by_id("not_exist")
    assert missing is None

    print(f"  ✅ get_chunk_by_id 跨实例可用，不存在返回 None")

# ================================================================
# T6-3：clear() 类方法清空所有实例
# ================================================================
def test_clear_class_method():
    """PersistentDocumentStore.clear() 清空后所有实例都变空"""
    from knowledge.embedding import PersistentDocumentStore, ChunkDocument

    # —— 步骤 1：先强制清干净，确认起始状态 ——
    PersistentDocumentStore.clear()
    before = PersistentDocumentStore.count()
    print(f"    clear() 后起始 count={before}")
    assert before == 0, f"❌ clear() 后应该是 0，实际 {before}（clear 没生效！）"

    # —— 步骤 2：加 2 个 chunk ——
    s1 = PersistentDocumentStore()
    s2 = PersistentDocumentStore()

    s1.add_chunks([
        ChunkDocument(id="c1", content="a", file_id="f1"),
        ChunkDocument(id="c2", content="b", file_id="f1"),
    ])
    middle = PersistentDocumentStore.count()
    print(f"    add 2 个后 count={middle}")
    assert middle == 2, f"❌ add 2 个后应该是 2，实际 {middle}"

    # —— 步骤 3：再 clear ——
    PersistentDocumentStore.clear()
    after = PersistentDocumentStore.count()
    print(f"    第二次 clear 后 count={after}")
    assert after == 0, f"❌ 第二次 clear 后应该是 0，实际 {after}"

    # —— 步骤 4：跨实例验证都空了 ——
    assert len(s1.get_chunks()) == 0
    assert len(s2.get_chunks()) == 0
    assert s2.get_chunk_by_id("c1") is None

    print(f"  ✅ PersistentDocumentStore.clear() 清空所有实例")
# ================================================================
# 运行全部测试 + 自动清场
# ================================================================
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING)

    print("\n" + "=" * 60)
    print("T4 — RecordStore 多租户 + T6 — PersistentDocumentStore 共享")
    print("=" * 60)

    tests = [
        ("RecordStore: 绑租户后 put/get 正常", test_put_get_with_tenant),
        ("RecordStore: 未绑租户 put 抛 PermissionError", test_put_without_tenant_raises),
        ("RecordStore: 两个租户物理隔离", test_tenant_isolation),
        ("RecordStore: 跨租户伪造 tenant_id 被拒", test_cross_tenant_write_rejected),
        ("RecordStore: delete 也是租户隔离的", test_delete_isolated),
        ("PersistentDocumentStore: 类级共享", test_persistent_store_shared),
        ("PersistentDocumentStore: get_chunk_by_id 跨实例", test_get_chunk_by_id_cross_instance),
        ("PersistentDocumentStore: clear() 类方法清空", test_clear_class_method),
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
        finally:
            # 每个测试后强制清场 tenant，防污染
            try:
                with tenant_scope("__cleanup__"):
                    pass
            except Exception:
                pass

    print(f"\n{'=' * 60}")
    print(f"测试结果: ✅ {passed} passed | ❌ {failed} failed")
    print(f"{'=' * 60}\n")

    # 最后：清空 PersistentDocumentStore 避免残留
    try:
        from knowledge.embedding import PersistentDocumentStore
        PersistentDocumentStore.clear()
    except Exception:
        pass

    if failed > 0:
        sys.exit(1)