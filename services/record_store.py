"""
为用户提供文档存储服务
实现项目持久化

基于 SQLite 的多租户键值存储层，供会话记忆、用户画像等模块调用。
特性：
    - 租户级物理隔离（tenant_id 作为主键的一部分）
    - 乐观锁版本控制，防止并发覆盖
    - WAL 模式提升并发读写性能

"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


from services.security import current_tenant_id, require_tenant

class TenantBoundaryError(PermissionError):
    """跨租户写入被拒绝时抛出。"""

class RecordConflictError(RuntimeError):
    """乐观锁检测到记录已过期（版本冲突）时抛出。"""

class SQLiteRecordStore:
    """
    基于 SQLite 的多租户键值存储。

    所有读写都会按当前 ``current_tenant_id()`` 自动过滤；
    写入前会通过 ``require_tenant()`` 确保租户上下文已绑定。
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    # ------------------------------------------------------------------ 写
    def put(
        self,
        namespace: str,
        record_id: str,
        payload: Dict[str, Any],
        *,
        expected_version: Optional[int] = None,
    ) -> int:
        """
        写入或更新一条记录（带乐观锁）。

        :param namespace:  记录命名空间，如 ``"conversation_session"``
        :param record_id:  记录主键（同命名空间下唯一）
        :param payload:    要存储的 JSON 可序列化字典
        :param expected_version: 若提供，则仅当当前版本等于该值时才写入
        :return: 写入后的新版本号
        :raises TenantBoundaryError:  租户 ID 与当前上下文不一致
        :raises RecordConflictError:  乐观锁版本不一致
        """
        tenant_id = require_tenant()
        # 避免恶意改造payload
        claimed_tenant = str(payload.get("tenant_id") or tenant_id)
        if claimed_tenant != tenant_id:
            raise TenantBoundaryError(
                f"跨租户写入被拒绝：当前租户={tenant_id!r}，记录声明租户={claimed_tenant!r}"
            )
        now = datetime.now(timezone.utc).isoformat()
        # 拿到锁，确保线程安全。
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT version FROM records WHERE namespace = ? AND tenant_id = ? AND record_id = ?",
                (namespace, tenant_id, record_id),
            ).fetchone()
            current_version = int(row[0]) if row else 0
            if expected_version is not None and current_version != int(expected_version):
                connection.rollback()
                raise RecordConflictError(
                    f"记录 {namespace}/{record_id} 版本冲突："
                    f"期望版本 {expected_version}，当前版本 {current_version}"
                )
            # 封装存储数据
            next_version = current_version + 1
            stored_payload = {
                **payload, #展开业务数据
                "tenant_id": tenant_id, #强制覆盖租户ID
                "_storage_version": next_version, #写入的新版本号
            }
            serialized = json.dumps(
                stored_payload,
                ensure_ascii=False,
                sort_keys=True,
                default=str)
            # UPDATE or INSERT
            if row: # 存在记录，更新版本
                connection.execute(
                    """
                    UPDATE records
                    SET payload = ?, updated_at = ?, version = ?
                    WHERE namespace = ? AND tenant_id = ? AND record_id = ?
                    """,
                    (serialized, now, next_version, namespace, tenant_id, record_id),
                )
            else: # 不存在记录，插入新记录
                connection.execute(
                    """
                    INSERT INTO records(namespace, tenant_id, record_id, payload, created_at, updated_at, version)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (namespace, tenant_id, record_id, serialized, now, now, next_version),
                )
            connection.commit()
            return next_version

    # ------------------------------------------------------------------ 读
    def get(self, namespace: str, record_id: str) -> Optional[Dict[str, Any]]:
        """读取一条记录；不存在或租户不匹配时返回 None。"""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload FROM records WHERE namespace = ? AND tenant_id = ? AND record_id = ?",
                (namespace, current_tenant_id(), record_id),
            ).fetchone()
        return json.loads(row[0]) if row else None

    def list(self, namespace: str, limit: int = 100) -> List[Dict[str, Any]]:
        """按更新时间倒序列出当前租户下的记录。"""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT payload FROM records
                WHERE namespace = ? AND tenant_id = ?
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (namespace, current_tenant_id(), max(1, int(limit))),
            ).fetchall()
        return [json.loads(row[0]) for row in rows]

    # ------------------------------------------------------------------ 删
    def delete(self, namespace: str, record_id: str) -> bool:
        """删除当前租户下的指定记录；返回是否实际删除了行。"""
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                "DELETE FROM records WHERE namespace = ? AND tenant_id = ? AND record_id = ?",
                (namespace, current_tenant_id(), record_id),
            )
            connection.commit()
            return cursor.rowcount > 0

    # ------------------------------------------------------------------ 诊断
    def health(self) -> Dict[str, Any]:
        """返回底层 SQLite 健康状态，供 /health 类接口使用。"""
        try:
            with self._connect() as connection:
                mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
                quick_check = connection.execute("PRAGMA quick_check").fetchone()[0]
                schema_version = connection.execute("PRAGMA user_version").fetchone()[0]
                count = connection.execute("SELECT COUNT(*) FROM records").fetchone()[0]
            return {
                "ready": mode.lower() == "wal" and quick_check == "ok",
                "journal_mode": mode,
                "quick_check": quick_check,
                "schema_version": schema_version,
                "records": count,
                "path": str(self.path),
            }
        except sqlite3.Error as exc:
            return {"ready": False, "error": str(exc), "path": str(self.path)}

    # ------------------------------------------------------------------ 内部
    # 建表，数据库初始化
    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA user_version=1")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS records(
                    namespace TEXT NOT NULL,
                    tenant_id TEXT NOT NULL,
                    record_id TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    version INTEGER NOT NULL DEFAULT 1,
                    PRIMARY KEY(namespace, tenant_id, record_id)
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_records_listing ON records(namespace, tenant_id, updated_at DESC)"
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.execute("PRAGMA busy_timeout=10000")
        try:
            yield connection
        finally:
            connection.close()