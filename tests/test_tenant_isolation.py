"""
T3 — 租户隔离（security.py）测试
================================

测试目标：
    1. bind_tenant / current_tenant_id / reset_tenant 基础链路
    2. tenant_scope 上下文管理器自动 reset
    3. require_tenant 未绑定时抛 PermissionError
    4. record_visible 可见性规则全分支
    5. 并发（多线程）场景下租户 ID 不串

运行方式：
    cd D:/code/tax_agent
    python -m tests.test_tenant_isolation
"""

import os
import sys
import threading
import asyncio
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(str(PROJECT_ROOT))

from services.security import (
    current_tenant_id,
    bind_tenant,
    reset_tenant,
    tenant_scope,
    require_tenant,
    record_visible,
)

# ================================================================
# 测试 1：bind_tenant / current_tenant_id / reset_tenant 基础链路
# ================================================================
def test_bind_reset_basic():
    """绑定 → 读取 → reset → 恢复默认"""
    # 初始状态应该是 "default"
    assert current_tenant_id() == "default", f"❌ 初始应为 'default'，实际 '{current_tenant_id()}'"

    # 绑定 user_001
    token = bind_tenant("user_001")
    assert current_tenant_id() == "user_001", f"❌ 绑定后应为 'user_001'，实际 '{current_tenant_id()}'"

    # 再绑一个（模拟嵌套，token 可以层层 reset）
    token2 = bind_tenant("user_002")
    assert current_tenant_id() == "user_002"

    # 先 reset 第二层
    reset_tenant(token2)
    assert current_tenant_id() == "user_001", f"❌ reset token2 后应回到 'user_001'，实际 '{current_tenant_id()}'"

    # 再 reset 第一层 → 恢复默认
    reset_tenant(token)
    assert current_tenant_id() == "default", f"❌ reset token 后应回到 'default'，实际 '{current_tenant_id()}'"

    print(f"  ✅ bind/reset 基础链路 + 嵌套 reset 正常")

# ================================================================
# 测试 2：tenant_scope 上下文管理器
# ================================================================
def test_tenant_scope():
    """tenant_scope 退出 with 块后应自动恢复"""
    assert current_tenant_id() == "default"

    with tenant_scope("user_alice"):
        assert current_tenant_id() == "user_alice"
        with tenant_scope("user_bob"):
            assert current_tenant_id() == "user_bob"
        # 内层退出 → 回到外层
        assert current_tenant_id() == "user_alice"

    # 全部退出 → 回到 default
    assert current_tenant_id() == "default", f"❌ scope 退出后应恢复 default，实际 '{current_tenant_id()}'"

    # —— 即使 with 块里抛异常，也能正确恢复 ——
    try:
        with tenant_scope("should_be_reset"):
            raise ValueError("模拟异常")
    except ValueError:
        pass
    assert current_tenant_id() == "default", f"❌ 异常后也应恢复 default，实际 '{current_tenant_id()}'"

    print(f"  ✅ tenant_scope 自动恢复 + 异常安全")

# ================================================================
# 测试 3：require_tenant 未绑定时抛异常
# ================================================================
def test_require_tenant():
    """未绑定 → 抛 PermissionError；已绑定 → 正常返回 tenant_id"""
    # 先确保是 default
    assert current_tenant_id() == "default"

    # 未绑定必须抛异常
    try:
        require_tenant()
        assert False, "❌ 这里应该抛 PermissionError"
    except PermissionError as e:
        assert "未绑定租户上下文" in str(e), f"❌ 错误消息应包含提示，实际 '{e}'"
        print(f"  ✅ 未绑定时 require_tenant() 正确抛 PermissionError")

    # 已绑定 → 正常返回
    token = bind_tenant("user_active")
    tid = require_tenant()
    assert tid == "user_active", f"❌ 已绑定时应返回 tenant_id，实际 '{tid}'"
    print(f"  ✅ 已绑定时 require_tenant() 返回 '{tid}'")
    reset_tenant(token)

# ================================================================
# 测试 4：record_visible 可见性规则
# ================================================================
def test_record_visible():
    """覆盖 record_visible 的所有分支。全程用 tenant_scope 管理，最后确保回到 default。"""

    # ===== 规则 1：record 为空 → 不可见 =====
    assert record_visible(None) is False
    assert record_visible({}) is False
    print(f"  ✅ 规则1: 空 record 不可见")

    # ===== 规则 2-5：有租户上下文时的可见性 =====
    with tenant_scope("user_001"):
        # 规则 2：无 tenant_id → 可见
        assert record_visible({"content": "历史数据"}) is True

        # 规则 3：tenant_id = "public" → 可见
        assert record_visible({"tenant_id": "public"}) is True

        # 规则 4：匹配 → 可见
        assert record_visible({"tenant_id": "user_001"}) is True

        # 规则 5：不匹配 → 不可见
        assert record_visible({"tenant_id": "user_999"}) is False

    print(f"  ✅ 规则2-5: tenant_scope('user_001') 下四条分支全对")

    # ===== 规则 6：default 状态下 =====
    # 到这里 tenant_scope 已经退出，主线程是 default
    assert current_tenant_id() == "default"

    # default 状态 + 有隔离 tenant_id → 不可见
    assert record_visible({"tenant_id": "user_001"}) is False
    # default 状态 + 无 tenant_id / public → 仍可见
    assert record_visible({"content": "历史"}) is True
    assert record_visible({"tenant_id": "public"}) is True

    print(f"  ✅ 规则6: default 状态下隔离数据不可见，但公开数据仍可见")

# ================================================================
# 测试 5：多线程并发 —— 租户 ID 不串（关键！）
# ================================================================
def test_thread_isolation():
    """
    多线程场景下，每个线程绑定自己的租户，互不影响。
    contextvars 会为每个线程/协程维护独立的上下文副本。
    """
    results = {}          # thread_id → (绑定的租户, 读到的租户)
    errors = []
    barrier = threading.Barrier(3)  # 3 个线程同时启动

    def worker(thread_name, tenant_id):
        try:
            # 所有线程同时到 barrier 再开始，增加并发竞争
            barrier.wait()

            token = bind_tenant(tenant_id)
            # 在绑定后再读一次，确认读到的是自己的
            actual = current_tenant_id()
            results[thread_name] = (tenant_id, actual)

            # 随便做点什么，模拟业务逻辑
            _ = [record_visible({"tenant_id": tenant_id}) for _ in range(100)]

            reset_tenant(token)
        except Exception as e:
            errors.append((thread_name, str(e)))

    threads = [
        threading.Thread(target=worker, args=("t1", "user_alpha")),
        threading.Thread(target=worker, args=("t2", "user_beta")),
        threading.Thread(target=worker, args=("t3", "user_gamma")),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"❌ 线程执行出错: {errors}"
    assert len(results) == 3, f"❌ 应该有 3 个线程结果，实际 {len(results)}"

    for name, (expected, actual) in results.items():
        assert expected == actual, f"❌ {name}: 期望 '{expected}', 实际 '{actual}'（租户串了！）"

    # 主线程自己也应该不受影响
    assert current_tenant_id() == "default", f"❌ 主线程应仍是 default，实际 '{current_tenant_id()}'"

    print(f"  ✅ 3 线程并发租户隔离正常，无串扰")
    for name, (expected, actual) in sorted(results.items()):
        print(f"     {name}: bind='{expected}', read='{actual}' ✅")

    # 线程函数里的 bind/reset 是各线程独立的，不影响主线程
    # 但测试 4 可能污染了主线程，这里显式清场
    with tenant_scope("main_thread_cleanup"):
        pass  # 进入再退出，确保当前是 default
    # 确认主线程真的是 default
    assert current_tenant_id() == "default"

    print(f"  ✅ 3 线程并发租户隔离正常，无串扰")
    # ... 后面不变
# ================================================================
# 测试 6：asyncio 协程并发 —— 同样不串
# ================================================================
def test_asyncio_isolation():
    """
    在同一个线程里跑多个协程，contextvars 也能正确隔离。
    （FastAPI 用 async，这个测试很重要）
    """
    results = {}

    async def coro(name, tenant_id):
        bind_tenant(tenant_id)
        # 模拟异步 I/O
        await asyncio.sleep(0.01)
        actual = current_tenant_id()
        results[name] = (tenant_id, actual)
        reset_tenant(bind_tenant("dummy"))  # 这其实不对，简化写法下面修正
        # 更规范：在协程开始时保存 token，结束时 reset

    # 修正版：用 tenant_scope
    async def coro_correct(name, tenant_id):
        with tenant_scope(tenant_id):
            await asyncio.sleep(0.01)
            actual = current_tenant_id()
            results[name] = (tenant_id, actual)

    async def main():
        await asyncio.gather(
            coro_correct("async1", "async_user_A"),
            coro_correct("async2", "async_user_B"),
            coro_correct("async3", "async_user_C"),
        )

    asyncio.run(main())

    assert len(results) == 3
    for name, (expected, actual) in results.items():
        assert expected == actual, f"❌ {name}: 期望 '{expected}', 实际 '{actual}'"

    assert current_tenant_id() == "default"

    print(f"  ✅ 3 协程并发租户隔离正常，无串扰")

# ================================================================
# 运行全部测试
# ================================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("T3 — 租户隔离（security.py）测试")
    print("=" * 60)

    tests = [
        ("bind/reset 基础链路 + 嵌套", test_bind_reset_basic),
        ("tenant_scope 上下文管理器", test_tenant_scope),
        ("require_tenant 未绑定抛异常", test_require_tenant),
        ("record_visible 可见性规则", test_record_visible),
        ("多线程并发隔离", test_thread_isolation),
        ("asyncio 协程隔离", test_asyncio_isolation),
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
            # —— 无论成功失败，都强制清场回 default ——
            # 防止某个测试忘 reset 导致污染后续测试
            try:
                while current_tenant_id() != "default":
                    # 如果不是 default，随便 bind 一个再 reset 掉
                    token = bind_tenant("__cleanup__")
                    reset_tenant(token)
            except Exception:
                pass  # 清场本身出错就不管了

    print(f"\n{'=' * 60}")
    print(f"测试结果: ✅ {passed} passed | ❌ {failed} failed")
    print(f"{'=' * 60}\n")

    if failed > 0:
        sys.exit(1)