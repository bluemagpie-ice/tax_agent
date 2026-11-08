"""Personal case library. SQLite is authoritative; no policy/chat vectors are used."""
import json
import re
import uuid

from pydantic import BaseModel, Field
from sqlalchemy import Column, Integer, MetaData, String, Table, Text, select, update, delete

from services.research_store import ResearchConflict, utcnow
from services.security import require_tenant


class CaseContent(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    question: str = Field(default="", max_length=16000)
    facts: list[str] = Field(default_factory=list, max_length=30)
    tax_types: list[str] = Field(default_factory=list, max_length=5)
    region: str | None = None
    tax_period: str | None = None
    business_type: str | None = None
    enterprise_type: str | None = None
    conditions: list[str] = Field(default_factory=list, max_length=30)
    evidence: list[dict] = Field(default_factory=list, max_length=50)
    conclusion: str = Field(max_length=30000)
    unresolved: list[str] = Field(default_factory=list, max_length=30)


def terms(value):
    text = str(value).lower()
    words = set(re.findall(r"[a-z0-9_]+", text))
    for chunk in re.findall(r"[\u4e00-\u9fff]+", text):
        words.update(chunk[i:i+2] for i in range(max(1, len(chunk)-1)))
    return words


class CaseStore:
    def __init__(self, engine):
        self.engine = engine
        meta = MetaData()
        self.cases = Table("research_cases", meta,
            Column("case_id", String(64), primary_key=True), Column("owner_id", String(120), nullable=False, index=True),
            Column("revision", Integer, nullable=False), Column("current_version", Integer, nullable=False),
            Column("published_version", Integer), Column("status", String(24), nullable=False),
            Column("source_task_id", String(64)), Column("created_at", String(40)), Column("updated_at", String(40)))
        self.versions = Table("research_case_versions", meta,
            Column("case_id", String(64), primary_key=True), Column("version", Integer, primary_key=True),
            Column("payload", Text, nullable=False))
        self.events = Table("research_case_events", meta,
            Column("event_id", String(64), primary_key=True), Column("case_id", String(64), nullable=False),
            Column("owner_id", String(120), nullable=False), Column("payload", Text, nullable=False))
        meta.create_all(engine)

    def _owned(self, case_id):
        return (self.cases.c.case_id == case_id) & (self.cases.c.owner_id == require_tenant())

    def _event(self, conn, case_id, action, version, **details):
        conn.execute(self.events.insert().values(event_id=uuid.uuid4().hex, case_id=case_id, owner_id=require_tenant(),
            payload=json.dumps({"action": action, "version": version, "at": utcnow(), "actor": require_tenant(), **details}, ensure_ascii=False)))

    def create(self, content, *, source_task_id=None, source_report_version=None):
        content = CaseContent.model_validate(content).model_dump()
        case_id = (uuid.uuid5(uuid.NAMESPACE_URL, f"research-case:{require_tenant()}:{source_task_id}:{source_report_version}").hex
                   if source_task_id and source_report_version is not None else uuid.uuid4().hex)
        if source_task_id and source_report_version is not None:
            with self.engine.connect() as conn:
                existing = conn.execute(select(self.cases.c.status).where(self._owned(case_id))).scalar_one_or_none()
            if existing is not None:
                if existing == "deleted":
                    raise ResearchConflict("此报告对应案例已删除，不能由重复保存请求恢复")
                current = self.get(case_id)
                if current["content"] != content:
                    raise ResearchConflict("来源报告已保存为案例，当前编辑不同；请打开已有案例修改")
                return current
        with self.engine.begin() as conn:
            conn.execute(self.cases.insert().values(case_id=case_id, owner_id=require_tenant(), revision=1,
                current_version=1, status="draft", source_task_id=source_task_id, created_at=utcnow(), updated_at=utcnow()))
            self._insert_version(conn, case_id, 1, content)
            self._event(conn, case_id, "created", 1)
        return self.get(case_id)

    def _insert_version(self, conn, case_id, version, content):
        conn.execute(self.versions.insert().values(case_id=case_id, version=version,
            payload=json.dumps({"content": content, "save_confirmed": False,
                                "professional_review": {"status": "not_reviewed"}}, ensure_ascii=False)))

    def get(self, case_id):
        with self.engine.connect() as conn:
            row = conn.execute(select(self.cases).where(self._owned(case_id))).mappings().first()
            if row is None or row["status"] == "deleted":
                raise KeyError("案例不存在或不属于当前用户")
            version = conn.execute(select(self.versions.c.payload).where(
                self.versions.c.case_id == case_id, self.versions.c.version == row["current_version"])).scalar_one()
        return {**dict(row), **json.loads(version)}

    def list(self):
        with self.engine.connect() as conn:
            ids = conn.execute(select(self.cases.c.case_id).where(self.cases.c.owner_id == require_tenant(),
                               self.cases.c.status != "deleted")).scalars().all()
        return [self.get(case_id) for case_id in ids]

    def linked(self, task_id):
        with self.engine.connect() as conn:
            return [dict(row) for row in self._linked_rows(conn, task_id)]

    def _linked_rows(self, conn, task_id):
        return conn.execute(select(self.cases.c.case_id, self.cases.c.revision, self.cases.c.current_version).where(
            self.cases.c.owner_id == require_tenant(), self.cases.c.source_task_id == task_id,
            self.cases.c.status != "deleted")).mappings().all()

    def delete_linked(self, conn, task_id):
        rows = self._linked_rows(conn, task_id)
        for row in rows:
            self._delete(conn, row["case_id"], row["revision"], row["current_version"])
        return [row["case_id"] for row in rows]

    def _delete(self, conn, case_id, revision, version):
        changed = conn.execute(update(self.cases).where(
            self._owned(case_id), self.cases.c.revision == revision, self.cases.c.status != "deleted"
        ).values(revision=revision+1, updated_at=utcnow(), published_version=None, status="deleted"))
        if changed.rowcount != 1:
            raise ResearchConflict("案例已被其他操作修改")
        conn.execute(delete(self.versions).where(self.versions.c.case_id == case_id))
        self._event(conn, case_id, "delete", version)

    def change(self, case_id, revision, action, *, content=None, notes=""):
        case = self.get(case_id)
        if revision != case["revision"]:
            raise ResearchConflict("案例版本已变化，请刷新后重试")
        version = case["current_version"]
        if action == "delete":
            with self.engine.begin() as conn:
                self._delete(conn, case_id, revision, version)
            return {"case_id": case_id, "status": "deleted"}
        with self.engine.begin() as conn:
            changed = conn.execute(update(self.cases).where(self._owned(case_id), self.cases.c.revision == revision).values(
                revision=revision+1, updated_at=utcnow()))
            if changed.rowcount != 1:
                raise ResearchConflict("案例已被其他操作修改")
            if action == "edit":
                parsed = CaseContent.model_validate(content).model_dump()
                version += 1
                self._insert_version(conn, case_id, version, parsed)
                conn.execute(update(self.cases).where(self._owned(case_id)).values(current_version=version, published_version=None, status="draft"))
            elif action in ("consent", "professional_review"):
                payload = json.loads(conn.execute(select(self.versions.c.payload).where(
                    self.versions.c.case_id == case_id, self.versions.c.version == version)).scalar_one())
                if action == "consent":
                    payload.update(save_confirmed=True, save_confirmed_at=utcnow(), save_confirmed_by=require_tenant())
                else:
                    if not notes.strip():
                        raise ValueError("专业复核记录需要填写判断依据")
                    payload["professional_review"] = {"status": "reviewed", "reviewer": require_tenant(), "notes": notes,
                        "at": utcnow(), "basis": "user_attestation_credentials_not_verified"}
                conn.execute(update(self.versions).where(self.versions.c.case_id == case_id, self.versions.c.version == version).values(
                    payload=json.dumps(payload, ensure_ascii=False)))
            elif action == "publish":
                if not case["save_confirmed"] or not case["content"]["facts"] or not case["content"]["conclusion"].strip():
                    raise ValueError("发布需要确认保存、业务事实和研究结论")
                if not any(e.get("content") and e.get("evidence_id") for e in case["content"]["evidence"]):
                    raise ValueError("发布需要保留实际政策证据，不能仅保存无来源结论")
                conn.execute(update(self.cases).where(self._owned(case_id)).values(published_version=version, status="published"))
            elif action == "withdraw":
                conn.execute(update(self.cases).where(self._owned(case_id)).values(published_version=None, status="withdrawn"))
            else:
                raise ValueError("未知案例操作")
            self._event(conn, case_id, action, version)
        return self.get(case_id)

    def history(self, case_id):
        self.get(case_id)
        with self.engine.connect() as conn:
            versions = conn.execute(select(self.versions).where(self.versions.c.case_id == case_id).order_by(self.versions.c.version)).mappings().all()
            events = conn.execute(select(self.events.c.payload).where(self.events.c.case_id == case_id,
                self.events.c.owner_id == require_tenant())).scalars().all()
        return {"versions": [{"version": row["version"], **json.loads(row["payload"])} for row in versions],
                "events": [json.loads(event) for event in events]}

    def search(self, query, *, facts=None, limit=3):
        query_terms = terms(query + " " + " ".join((facts or {}).get("facts", [])))
        candidates = []
        for case in self.list():
            if case["status"] != "published" or case["published_version"] != case["current_version"] or not case["save_confirmed"]:
                continue
            content = case["content"]
            # Candidate ranking uses facts and conditions, not historical conclusions.
            indexed = terms(" ".join([content["title"], content["question"], *content["facts"], *content["conditions"], *content["tax_types"]]))
            score = len(query_terms & indexed) / max(len(query_terms | indexed), 1)
            if score <= 0:
                continue
            comparisons = []
            for field in ("tax_types", "region", "tax_period", "business_type", "enterprise_type"):
                old, new = content.get(field), (facts or {}).get(field)
                comparisons.append({"field":field, "historical":old, "current":new,
                    "comparison":"unknown" if not old or not new else "same" if old == new else "different"})
            candidates.append({"case_id":case["case_id"], "version":case["current_version"], "score":round(score,6),
                "content":content, "comparisons":comparisons, "professional_review":case["professional_review"],
                "shared_facts": sorted(set(content["facts"]) & set((facts or {}).get("facts", []))),
                "historical_only_facts": sorted(set(content["facts"]) - set((facts or {}).get("facts", []))),
                "current_only_facts": sorted(set((facts or {}).get("facts", [])) - set(content["facts"])),
                "conditions_to_recheck": content["conditions"],
                "notice":"历史经验仅作研究线索，必须重新检索官方政策；保存确认不代表专业审核"})
        return sorted(candidates, key=lambda row: (-row["score"], row["case_id"]))[:limit]
