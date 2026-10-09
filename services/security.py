"""
多租户隔离模块

基于 contextvars 实现请求级租户上下文绑定，确保每条会话/记忆
都严格隔离在各自的租户（企业/用户）范围内。

对外提供：
    - current_tenant_id()  获取当前线程/协程的租户 ID
    - bind_tenant()        绑定租户上下文（返回 token，用于 reset）
    - tenant_scope()       上下文管理器，方便 with 块绑定
    - record_visible()    判定某条记录当前是否对本租户可见
"""

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Dict, Optional

# 租户上下文变量；默认值 "default" 表示尚未绑定具体租户，
# 此时任何受租户保护的写入都应被拒绝。
_tenant_context: ContextVar[str] = ContextVar("tenant_id", default="default")
document_session: ContextVar[str] = ContextVar("document_session", default="")
blocked_document_ids = set()


def document_visible(metadata: dict) -> bool:
    """RAG 文档权限：系统资料共享；用户文件必须有归属，单次文件限定会话。"""
    if metadata.get("file_id") in blocked_document_ids:
        return False
    if metadata.get("status", "active") != "active":
        return False
    if metadata.get("source_type") == "default" or metadata.get("scope") == "system":
        return True
    owner = metadata.get("owner_id")
    if not owner or str(owner) != current_tenant_id():
        return False
    return metadata.get("scope") == "global" or (
        metadata.get("scope") == "session" and bool(document_session.get())
        and metadata.get("session_id") == document_session.get())


def checkpoint_thread_id(thread_id: str) -> str:
    tenant = current_tenant_id()
    return f"{tenant}:{thread_id}" if tenant != "default" else thread_id

# 公共可见的租户占位 —— 用于标识"租户无关"的公开数据
_PUBLIC_TENANT = "public"

def current_tenant_id() -> str:
    """获取当前上下文的租户 ID。"""
    return _tenant_context.get()

def bind_tenant(tenant_id: str):
    """绑定租户 ID，返回可用于 reset 的 token。"""
    return _tenant_context.set(tenant_id)

def reset_tenant(token) -> None:
    """将租户上下文恢复到 set 之前的状态。"""
    _tenant_context.reset(token)


@contextmanager
def tenant_scope(tenant_id: str):
    """
    使用 `with tenant_scope("tenant_xxx"):` 形式绑定租户上下文，
    退出块后自动恢复，便于在 FastAPI 依赖或测试中使用。
    """
    token = bind_tenant(tenant_id)
    try:
        yield tenant_id
    finally:
        reset_tenant(token)

def require_tenant() -> str:
    """
    断言当前已绑定真实租户，否则抛出异常。
    供写操作入口（如 RecordStore.put）调用。
    """
    tenant_id = current_tenant_id()
    if tenant_id == "default":
        raise PermissionError("未绑定租户上下文：请先在请求入口调用 bind_tenant(...)")
    return tenant_id

def record_visible(record: Optional[Dict[str, Any]]) -> bool:
    """
    判断某条记录对当前租户是否可见。

    规则：
        1) 记录为空 → 不可见
        2) 记录未声明 tenant_id（历史数据/公开数据）→ 可见，由上层决定是否迁移
        3) 记录 tenant_id 为 "public" → 对所有租户可见
        4) 当前租户已绑定，且匹配记录 tenant_id → 可见
        5) 其他情况 → 不可见
    """
    if not record:
        return False
    record_tenant = record.get("tenant_id")
    if record_tenant in (None, "", _PUBLIC_TENANT):
        return True
    current = current_tenant_id()
    if current == "default":
        # 未绑定租户时看不到任何租户隔离数据
        return False
    return str(record_tenant) == current
