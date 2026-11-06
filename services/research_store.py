"""Research business records. No chat-memory or policy-index writes.

Schema v1 is additive and idempotent: only these prefixed tables are created.
The caller supplies an engine in tests; production reuses the business engine.
"""
from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, Integer, MetaData, String, Table, Text, Float, select, update

from services.security import require_tenant


def utcnow():
    return datetime.now(timezone.utc).isoformat()


class ResearchConflict(ValueError):
    pass


class ResearchStore:
    def __init__(self, engine=None):
        if engine is None:
            from database.models import engine
        self.engine = engine
        meta = MetaData()
        self.tasks = Table(
            "research_tasks", meta,
            Column("task_id", String(64), primary_key=True),
            Column("owner_id", String(120), nullable=False, index=True),
            Column("revision", Integer, nullable=False),
            Column("payload", Text, nullable=False),
            Column("lease", String(64)), Column("lease_until", Float),
        )
        self.reports = Table(
            "research_reports", meta,
            Column("task_id", String(64), primary_key=True),
            Column("version", Integer, primary_key=True),
            Column("owner_id", String(120), nullable=False),
            Column("payload", Text, nullable=False),
        )
        meta.create_all(engine)

    def _owned(self, task_id):
        return (self.tasks.c.task_id == task_id) & (self.tasks.c.owner_id == require_tenant())

    def create(self, query, *, web_search=False, **options):
        task_id = uuid.uuid4().hex
        state = {"task_id": task_id, "owner_id": require_tenant(), "query": query,
                 "web_search": web_search, "status": "created", "stage": "created",
                 "created_at": utcnow(), "updated_at": utcnow(), "report_version": 0,
                 "evidence": [], "facts": [], "unknowns": [], "trace": [], **options}
        with self.engine.begin() as conn:
            conn.execute(self.tasks.insert().values(task_id=task_id, owner_id=require_tenant(),
                         revision=1, payload=json.dumps(state, ensure_ascii=False)))
        return {**state, "revision": 1}

    def get(self, task_id):
        with self.engine.connect() as conn:
            row = conn.execute(select(self.tasks).where(self._owned(task_id))).mappings().first()
        if row is None:
            raise KeyError("研究任务不存在或不属于当前用户")
        return {**json.loads(row["payload"]), "revision": row["revision"]}

    def list(self):
        with self.engine.connect() as conn:
            rows = conn.execute(select(self.tasks.c.payload, self.tasks.c.revision).where(
                self.tasks.c.owner_id == require_tenant())).all()
        return sorted([{**json.loads(row[0]), "revision": row[1]} for row in rows],
                      key=lambda item: item["updated_at"], reverse=True)

    def claim(self, task_id, revision):
        """CAS + expiring lease protects actions across workers, including stale reviews."""
        self.get(task_id)
        lease = uuid.uuid4().hex
        with self.engine.begin() as conn:
            result = conn.execute(update(self.tasks).where(
                self._owned(task_id), self.tasks.c.revision == revision,
                (self.tasks.c.lease_until.is_(None)) | (self.tasks.c.lease_until < time.time()),
            ).values(lease=lease, lease_until=time.time() + 600))
            if result.rowcount != 1:
                raise ResearchConflict("任务正在执行或版本已变化，请刷新后重试")
        return lease

    def save(self, state, lease):
        task_id = state["task_id"]
        state = {k: v for k, v in state.items() if k != "revision"}
        state["updated_at"] = utcnow()
        with self.engine.begin() as conn:
            result = conn.execute(update(self.tasks).where(self._owned(task_id), self.tasks.c.lease == lease).values(
                payload=json.dumps(state, ensure_ascii=False), revision=self.tasks.c.revision + 1,
                lease_until=time.time() + 600))
            if result.rowcount != 1:
                raise ResearchConflict("研究任务执行租约已失效")
            if state.get("report"):
                existing = conn.execute(select(self.reports.c.version).where(
                    self.reports.c.task_id == task_id, self.reports.c.version == state["report_version"])).first()
                payload = json.dumps(state["report"], ensure_ascii=False)
                if existing is None:
                    conn.execute(self.reports.insert().values(task_id=task_id, owner_id=require_tenant(),
                                 version=state["report_version"], payload=payload))
        return self.get(task_id)

    def release(self, task_id, lease):
        with self.engine.begin() as conn:
            conn.execute(update(self.tasks).where(self._owned(task_id), self.tasks.c.lease == lease).values(
                lease=None, lease_until=None))

    def history(self, task_id):
        self.get(task_id)
        with self.engine.connect() as conn:
            rows = conn.execute(select(self.reports).where(self.reports.c.task_id == task_id,
                self.reports.c.owner_id == require_tenant()).order_by(self.reports.c.version)).mappings().all()
        return [{"version": row["version"], "report": json.loads(row["payload"])} for row in rows]
