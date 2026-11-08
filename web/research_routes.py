"""Research API is mounted with the existing authentication dependency."""
import asyncio
import json
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, StrictBool, StrictInt, model_validator
from starlette.responses import StreamingResponse

from services.research_store import ResearchConflict
from services.research_cases import CaseContent
from services.research_evidence import EVIDENCE_GAP_CODES, legacy_failed_report


class CreateResearch(BaseModel):
    query: str = Field(min_length=1, max_length=16000)
    web_search: bool = False
    use_cases: bool = True


class RunResearch(BaseModel):
    revision: StrictInt = Field(ge=1)


class StreamRun(RunResearch):
    model_config = {"extra": "forbid"}


class DeleteResearch(RunResearch):
    model_config = {"extra": "forbid"}
    delete_cases: StrictBool = False


class ReviewResearch(RunResearch):
    action: Literal["confirm", "revise", "research_again", "clarify"]
    report_version: StrictInt | None = Field(default=None, ge=1)
    report: dict | None = None
    reply: str = Field(default="", max_length=16000)
    skip_unknowns: bool = False

    @model_validator(mode="after")
    def valid_action(self):
        if self.action != "clarify" and self.report_version is None:
            raise ValueError("审核动作需要报告版本")
        if self.action == "revise" and not self.report:
            raise ValueError("请提供修改后的报告")
        if self.action == "clarify" and not self.reply.strip() and not self.skip_unknowns:
            raise ValueError("请补充信息或选择保留未知")
        return self


class CreateCaseRequest(BaseModel):
    content: CaseContent
    source_task_id: str | None = Field(default=None, max_length=64)
    source_report_version: StrictInt | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def source_version(self):
        if bool(self.source_task_id) != bool(self.source_report_version):
            raise ValueError("来源任务和报告版本必须同时提供")
        return self


class ChangeCaseRequest(RunResearch):
    action: Literal["edit", "consent", "publish", "withdraw", "delete", "professional_review"]
    content: CaseContent | None = None
    notes: str = Field(default="", max_length=8000)

    @model_validator(mode="after")
    def edit_content(self):
        if self.action == "edit" and self.content is None:
            raise ValueError("修改案例需要提供内容")
        return self


def build_router(auth, service):
    router = APIRouter(prefix="/api/research", dependencies=[Depends(auth)])

    def lookup(task_id):
        try:
            return service().store.get(task_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    def visible(task):
        if task["status"] == "failed" and (task.get("failure") or {}).get("code") in EVIDENCE_GAP_CODES:
            return {**task, "status": "insufficient_evidence"}
        if not legacy_failed_report(task):
            return task
        if task["status"] == "awaiting_review":
            return {**task, "status": "failed", "report": None, "failure": {
                "code": "legacy_invalid_report", "message": "历史初稿未通过引用校验。可以重新研究，原版本仍保存在历史记录中。"
            }}
        return {**task, "report": None}

    @router.post("/tasks")
    async def create(body: CreateResearch):
        if not body.query.strip():
            raise HTTPException(422, "请输入研究问题")
        return service().store.create(body.query.strip(), web_search=body.web_search, use_cases=body.use_cases)

    @router.get("/tasks")
    async def listing():
        return [{k: row.get(k) for k in ("task_id", "query", "stage", "status", "revision", "updated_at")}
                for row in map(visible, service().store.list())]

    @router.get("/tasks/{task_id}")
    async def read(task_id: str):
        task = visible(lookup(task_id))
        return {**task, "linked_case_count": len(service().case_store.linked(task_id))}

    @router.delete("/tasks/{task_id}")
    async def delete_task(task_id: str, body: DeleteResearch):
        try:
            return await service().delete_task(task_id, body.revision, delete_cases=body.delete_cases)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ResearchConflict as exc:
            raise HTTPException(409, str(exc)) from exc
        except Exception as exc:
            raise HTTPException(503, "研究任务删除未完成，请稍后重试") from exc

    @router.get("/tasks/{task_id}/reports")
    async def history(task_id: str):
        lookup(task_id)
        return service().store.history(task_id)

    @router.get("/tasks/{task_id}/case-preview")
    async def preview(task_id: str, generalize: bool = False):
        from services.research_candidate import candidate_from_task
        try:
            return candidate_from_task(lookup(task_id), generalize=generalize)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @router.get("/cases")
    async def cases():
        return service().case_store.list()

    @router.post("/cases")
    async def create_case(body: CreateCaseRequest):
        from services.research_cases import CaseContent
        from services.research_evidence import normalize_evidence
        body = body.model_dump()
        try:
            if isinstance(body.get("content"), dict):
                body["content"]["evidence"] = [e if e.get("evidence_id") else normalize_evidence(e,kind="user_supplied")
                                                for e in body["content"].get("evidence",[])]
            content = CaseContent.model_validate(body.get("content", {})).model_dump()
            source = body.get("source_task_id")
            if source:
                source_task = lookup(source)
                if source_task["status"] != "finalized" or body.get("source_report_version") != source_task["report_version"]:
                    raise ResearchConflict("来源研究报告尚未定稿或版本已变化")
                if source_task.get("evaluation_run"):
                    raise ResearchConflict("评测结果不得回流正式案例库")
                store = service().store
                lease = store.claim(source, source_task["revision"])
                try:
                    return service().case_store.create(content, source_task_id=source, source_report_version=body["source_report_version"])
                finally:
                    store.release(source, lease)
            return service().case_store.create(content, source_task_id=source, source_report_version=body.get("source_report_version"))
        except (ValueError, ResearchConflict) as exc:
            raise HTTPException(422, str(exc)) from exc

    @router.get("/cases/{case_id}/history")
    async def case_history(case_id: str):
        try:
            return service().case_store.history(case_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    @router.post("/cases/{case_id}")
    async def change_case(case_id: str, body: ChangeCaseRequest):
        body = body.model_dump()
        try:
            if type(body.get("revision")) is not int:
                raise ValueError("操作必须携带案例版本")
            return service().case_store.change(case_id, body["revision"], body.get("action"), content=body.get("content"), notes=body.get("notes", ""))
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ResearchConflict as exc:
            raise HTTPException(409, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @router.post("/tasks/{task_id}/run")
    async def run(task_id: str, body: RunResearch):
        lookup(task_id)
        try:
            return await service().execute(task_id, body.revision)
        except ResearchConflict as exc:
            raise HTTPException(409, str(exc)) from exc
        except Exception as exc:
            raise HTTPException(503, "研究执行失败，已保存完成的步骤；请检查模型配置后恢复任务") from exc

    @router.post("/tasks/{task_id}/review")
    async def review(task_id: str, body: ReviewResearch):
        lookup(task_id)
        try:
            return await service().execute(task_id, body.revision, action=body.model_dump())
        except ResearchConflict as exc:
            raise HTTPException(409, str(exc)) from exc
        except Exception as exc:
            raise HTTPException(503, "报告确认失败，请刷新任务后重试") from exc

    @router.post("/tasks/{task_id}/stream")
    async def stream(task_id: str, body: ReviewResearch | StreamRun):
        lookup(task_id)
        action = body.model_dump() if isinstance(body, ReviewResearch) else None

        async def generate():
            queue = asyncio.Queue()

            async def emit(kind, payload):
                await queue.put({"type": kind, **payload})

            async def work():
                try:
                    result = await service().execute(task_id, body.revision, action=action, emit=emit)
                    await emit("done", {"task": visible(result)})
                except ResearchConflict as exc:
                    await emit("error", {"message": str(exc)})
                except Exception:
                    await emit("error", {"message": "研究未完成，已保存进度，请检查配置并恢复任务"})

            task = asyncio.create_task(work())
            try:
                while True:
                    try:
                        event = await asyncio.wait_for(queue.get(), 10)
                    except asyncio.TimeoutError:
                        yield ": keepalive\n\n"
                        continue
                    yield "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"
                    if event["type"] in ("done", "error"):
                        break
            finally:
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)

        return StreamingResponse(generate(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    return router
