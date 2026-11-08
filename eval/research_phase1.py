"""Run the Research graph across two independent retrieval switches."""
import argparse
import asyncio
from collections import Counter
from hashlib import sha256
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
from time import monotonic
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "eval/research/phase1_cases.json"
REFERENCES = ROOT / "eval/research/phase1_references.json"
SNAPSHOT = ROOT / "eval/research/policy_snapshot.json"
MODES = {"A": (False, False), "B": (False, True), "C": (True, False), "D": (True, True)}
RETRIEVAL = ("search_tax_policy", "search_web_policy", "read_web_page")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, data):
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pending.replace(path)


def empty_search(*args, **kwargs):
    return []


def fields(record, names):
    return {name: record[name] for name in names.split() if name in record}


def official_location(url):
    from services.research_evidence import official_url

    if official_url(url):
        parts = urlsplit(url)
        return urlunsplit((parts.scheme, parts.hostname, parts.path, "", ""))
    return None


def classify(data, retrieval):
    failure = data.get("failure") or {}
    code = failure.get("code")
    categories = []
    if code == "structured_output_invalid":
        categories.append("structured_output_failed")
    if retrieval["search_tax_policy"].get("error"):
        categories.append("local_policy_failed")
    if (retrieval["search_web_policy"].get("error") or retrieval["read_web_page"].get("error")
            or code == "policy_original_unavailable"):
        categories.append("official_web_failed")
    if data.get("status") == "insufficient_evidence":
        return ["completed_insufficient_evidence", *categories]
    if data.get("status") == "reference_analysis":
        return ["completed_unverified_analysis", *categories]
    if data.get("report"):
        report = data["report"]
        coverage = (data.get("assessment") or {}).get("coverage", [])
        sufficient = coverage and len(coverage) == len(data.get("questions", [])) and all(
            item.get("sufficient") for item in coverage)
        from services.research_evidence import traceable_evidence
        traceable_ids = {item["evidence_id"] for item in report.get("policy_evidence", []) if traceable_evidence(item)}
        if not traceable_ids or any(not item.get("evidence_ids") or set(item["evidence_ids"]) - traceable_ids
                                    for item in report.get("analysis", [])):
            return ["other_failure", *categories]
        return ["completed_sufficient_evidence" if sufficient else "completed_insufficient_evidence", *categories]
    if not categories or code not in ("structured_output_invalid", "retrieval_failed"):
        categories.append("other_failure")
    return categories


def summarize(data):
    metrics = data.get("metrics") or {}
    llm = [fields(c, "purpose stage outcome latency_ms error_type finish_reason retry_kind output_limit usage model_id response_model_id "
                     "format_repairs prompt_evidence_count prompt_passage_count thinking json_mode")
           for c in metrics.get("llm_calls", [])]
    raw_tools = metrics.get("tool_calls", [])
    tools = []
    for call in raw_tools:
        tool = fields(call, "tool stage status error_type error_code latency_ms")
        result = call.get("result")
        if isinstance(result, list):
            tool["result_count"] = len(result)
            if call["tool"] == "search_tax_policy":
                tool["system_policy_hits"] = [fields(row, "file_id id score") for row in result
                                              if row.get("source_type") == "default"]
            if call["tool"] == "search_web_policy":
                tool["urls"] = [url for row in result if (url := official_location(row.get("url")))]
        if call["tool"] == "read_web_page":
            args = call.get("args", {}).get("positional") or [None]
            tool["url"] = official_location(args[0])
        tools.append(tool)
    retrieval = {name: dict(Counter(c.get("status", "in_progress") for c in tools if c["tool"] == name))
                 for name in RETRIEVAL}
    outcomes = Counter(c.get("status", "in_progress") for c in tools)
    status = data.get("status", "unknown")
    failure = data.get("failure") or {}
    candidates = data.get("candidates", [])
    return {
        "status": status, "stage": data.get("stage"), "graph_elapsed_ms": data.get("elapsed_ms"),
        "categories": classify(data, retrieval),
        "failure": fields(failure, "code stage") | fields(
            failure.get("details") or {}, "output_error previous_output_error schema provider_error_type error_type tools"),
        "stage_trace": [fields(t, "stage outcome elapsed_ms") for t in data.get("trace", [])],
        "model": {"count": len(llm), "latency_ms": sum(c.get("latency_ms", 0) for c in llm),
                  "outcomes": dict(Counter(c.get("outcome", "in_progress") for c in llm)), "calls": llm},
        "tools": {"count": len(tools), "outcomes": dict(outcomes),
                  "success_rate": (outcomes["success"] + outcomes["empty"]) / len(tools) if tools else None,
                  "retrieval": retrieval, "calls": tools},
        "stop_reason": data.get("stop_reason"),
        "evidence": {"candidates": dict(Counter(c["kind"] for c in candidates)),
                     "candidate_source_checks": dict(Counter(c.get("source_check", "unknown") for c in candidates)),
                     "question_count": len(data.get("questions", [])),
                     "coverage": [fields(c, "question_id sufficient evidence_ids period_status")
                                  for c in (data.get("assessment") or {}).get("coverage", [])]},
        "report_generated": bool(data.get("report")),
        "reference_analysis_generated": bool(data.get("reference_analysis")),
        "clarification_rounds": data.get("clarification_rounds", 0),
    }


def live_model():
    from langchain_openai import ChatOpenAI
    from services.config_manager import ConfigManager, validate_api_url

    cfg = ConfigManager().get_llm_config()
    if not all(cfg.values()) or cfg["api_key"].startswith("your_"):
        raise ValueError("model_configuration")
    validate_api_url(cfg["base_url"])
    return ChatOpenAI(**cfg, temperature=0, max_retries=0, timeout=25)


async def run_case(job, workdir):
    from dotenv import load_dotenv
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    from sqlalchemy import create_engine

    if job["live"]:
        load_dotenv(ROOT / ".env")
    from agents.research_agent import ResearchAgent, _run_metrics, _run_trace
    from services.research_store import ResearchStore
    from services.security import read_only_retrieval, tenant_scope

    case = job["case"]
    online, populated = MODES[job["mode"]]
    if job["live"]:
        try:
            llm = live_model()
        except ValueError:
            return summarize({"status": "failed", "stage": "setup",
                              "failure": {"code": "model_configuration", "stage": "setup"}})
        options = {"search": None if populated else empty_search}
    else:
        from eval.research_eval import SmokeModel, SnapshotTools
        from langchain_core.messages import AIMessage

        snapshot = SnapshotTools(read_json(SNAPSHOT)["records"])

        class ClarifyingSmokeModel(SmokeModel):
            def invoke(self, messages, **kwargs):
                if messages[0].content.startswith("输出类型：ClarificationUpdate\n"):
                    reply = json.loads(messages[-1].content)["reply"]
                    return AIMessage(content=json.dumps({"facts": [reply], "answered": [],
                        "kept_unknown": ["f1"], "needs_reanalysis": False}, ensure_ascii=False))
                return super().invoke(messages, **kwargs)

        llm = ClarifyingSmokeModel()
        options = {"search": snapshot.search if populated else empty_search,
                   "web_search": snapshot.web_search, "read_page": snapshot.page}

    engine = create_engine("sqlite:///:memory:")
    guard = read_only_retrieval.set(True)
    progress = workdir / "progress.json"
    reply_used = False
    try:
        with tenant_scope("research-phase1-" + uuid4().hex):
            async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
                store = ResearchStore(engine)
                agent = ResearchAgent(store, llm=llm, workdir_root=workdir, **options)
                agent._checkpointer = saver
                task = store.create(case["question"], web_search=online, use_cases=case.get("use_cases", False),
                                    budget=job["budget"], evaluation_run=True,
                                    allow_reference_analysis=job.get("allow_reference_analysis", False))

                async def emit(_, event):
                    current = store.get(task["task_id"])
                    current.update(metrics=_run_metrics.get(), trace=_run_trace.get() or [])
                    write_json(progress, summarize(current) | fields(event, "stage status"))

                async def execute():
                    nonlocal reply_used
                    result = await agent.execute(task["task_id"], task["revision"], emit=emit)
                    if result["status"] == "awaiting_facts" and case.get("clarification"):
                        reply_used = True
                        result = await agent.execute(task["task_id"], result["revision"], emit=emit,
                            action={"action": "clarify", "reply": case["clarification"], "skip_unknowns": False})
                    while result["status"] == "awaiting_facts":
                        result = await agent.execute(task["task_id"], result["revision"], emit=emit,
                            action={"action": "clarify", "skip_unknowns": True})
                    return result

                try:
                    task = await asyncio.wait_for(execute(), timeout=job["timeout"] - min(5, job["timeout"] / 3))
                except TimeoutError:
                    task = store.get(task["task_id"])
                    task["failure"] = {"code": "case_timeout", "stage": task["stage"]}
                    task["status"] = "timeout"
                result = summarize(task)
                result["clarification_sent"] = reply_used
                if job.get("save_drafts") and task.get("report"):
                    from services.security import redact_identifiers

                    draft = redact_identifiers(task["report"]["answer_draft"])
                    # The application also supports credentials stored outside the process environment.
                    from services.config_manager import ConfigManager
                    config = ConfigManager()
                    for get_config in (config.get_llm_config, config.get_embedding_config, config.get_qdrant_config):
                        secret = get_config().get("api_key")
                        if secret:
                            draft = draft.replace(secret, "[密钥已脱敏]")
                    write_json(workdir / "draft.json", {"answer_draft": draft})
                return result
    finally:
        engine.dispose()
        read_only_retrieval.reset(guard)


def worker(workdir):
    logging.disable(logging.CRITICAL)
    job = read_json(workdir / "job.json")

    async def run():
        try:
            result = await run_case(job, workdir)
        except Exception as exc:
            result = summarize({"status": "failed", "stage": "worker",
                                "failure": {"code": "worker_error", "details": {"error_type": type(exc).__name__}}})
        write_json(workdir / "result.json", result)

    asyncio.run(run())


def write_review_queue(output, rows):
    # Reference material is loaded in the coordinator only after all workers exit.
    references = read_json(REFERENCES)["references"]
    review = []
    for row in rows:
        case_id, mode = row["case_id"], row["mode"]
        if case_id not in references:
            continue
        draft = output / f"{mode}-{case_id}" / "draft.json"
        if not row["live"]:
            state = "mock"
        elif row["status"] == "skipped":
            state = "case_skipped"
        elif row.get("metrics_partial"):
            state = "worker_result_incomplete"
        elif not row.get("report_generated"):
            state = "no_report"
        else:
            state = "pending_human_review" if draft.is_file() else "draft_not_exported"
        reference = references[case_id]
        target = reference.get("source_url")
        tools = row.get("tools", {}).get("calls", [])
        review.append({"case_id": case_id, "mode": mode, "status": state, "reference": reference,
                       "target_found": any(target in call.get("urls", []) for call in tools) if target else None,
                       "target_read": any(call.get("url") == target and call.get("status") == "success"
                                          for call in tools) if target else None,
                       "draft": draft.relative_to(output).as_posix() if draft.is_file() else None})
    write_json(output / "reference_review.json", review)


def run_matrix(options):
    cases = read_json(DATASET)["cases"]
    if options.case:
        cases = [case for case in cases if case["id"] in options.case]
        if {case["id"] for case in cases} != set(options.case):
            raise ValueError("unknown case ID")
    output = ROOT / ".test_runs" / ("research-phase1-" + uuid4().hex[:12])
    output.mkdir(parents=True)
    budget = {"llm_calls": options.max_model_calls, "tool_calls": options.max_tool_calls,
              "token_upper_budget": options.max_tokens, "research_rounds": options.rounds, "replans": 1,
              "clarifications": 2}
    env = dict(os.environ, LANGCHAIN_TRACING_V2="false", LANGCHAIN_TRACING="false", LANGSMITH_TRACING="false",
               LANGSMITH_TRACING_V2="false")
    if not options.live:
        env["PYTHON_DOTENV_DISABLED"] = "1"
    inputs = [DATASET, Path(__file__), ROOT / "agents/research_agent.py", ROOT / "agents/research_output.py",
              ROOT / "services/tools/kb_tools.py", ROOT / "knowledge/embedding.py"]
    if not options.live:
        inputs.append(SNAPSHOT)
    write_json(output / "manifest.json", {
        "execution": "live" if options.live else "mock", "modes": options.modes, "budget_per_case": budget,
        "case_timeout_s": options.case_timeout, "total_timeout_s": options.total_timeout,
        "case_ids": [case["id"] for case in cases], "python": sys.version.split()[0],
        "allow_reference_analysis": getattr(options, "allow_reference_analysis", False),
        "sha256": {path.relative_to(ROOT).as_posix(): sha256(path.read_bytes()).hexdigest() for path in inputs},
    })
    print(output, flush=True)
    deadline = monotonic() + options.total_timeout
    rows = []
    for mode in options.modes:
        for case in cases:
            started = monotonic()
            timeout = min(options.case_timeout, deadline - started)
            row = {"case_id": case["id"], "mode": mode, "web_search": MODES[mode][0],
                   "populated_policy": MODES[mode][1], "live": options.live}
            if timeout <= 3:
                row.update(status="skipped", categories=["other_failure"], failure={"code": "total_timeout"})
            else:
                workdir = output / f"{mode}-{case['id']}"
                workdir.mkdir()
                write_json(workdir / "job.json", {"case": case, "mode": mode, "live": options.live,
                           "budget": budget, "timeout": timeout, "save_drafts": options.save_drafts,
                           "allow_reference_analysis": getattr(options, "allow_reference_analysis", False)})
                timed_out = False
                try:
                    subprocess.run([sys.executable, "-m", "eval.research_phase1", "--worker", str(workdir)],
                                   cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   check=False, timeout=timeout)
                except subprocess.TimeoutExpired:
                    timed_out = True
                result = workdir / "result.json"
                if result.exists():
                    row.update(read_json(result))
                else:
                    progress = workdir / "progress.json"
                    if progress.is_file():
                        row.update(read_json(progress))
                    row.update(status="timeout" if timed_out else "failed", categories=["other_failure"],
                               failure={"code": "case_timeout" if timed_out else "worker_exit",
                                        "stage": row.get("stage", "setup")},
                               metrics_partial=True)
                row["worker_hard_timeout"] = timed_out
            row["wall_ms"] = round((monotonic() - started) * 1000)
            rows.append(row)
            write_json(output / "results.json", rows)
            print(mode, case["id"], row["status"], ",".join(row["categories"]), row["wall_ms"], flush=True)
    write_review_queue(output, rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="call configured real services")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--case", action="append")
    parser.add_argument("--case-timeout", type=int, default=75)
    parser.add_argument("--total-timeout", type=int, default=600)
    parser.add_argument("--max-model-calls", type=int, default=10)
    parser.add_argument("--max-tool-calls", type=int, default=12)
    parser.add_argument("--max-tokens", type=int, default=45000)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--save-drafts", action="store_true", help="export redacted public-case drafts for local review")
    parser.add_argument("--allow-reference-analysis", action="store_true")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    options = parser.parse_args()
    if options.worker:
        worker(options.worker)
        return 0
    if (min(options.case_timeout, options.total_timeout) <= 3 or options.max_model_calls < 3
            or min(options.max_tool_calls, options.max_tokens, options.rounds) < 1):
        parser.error("invalid timeout or call budget")
    try:
        rows = run_matrix(options)
    except ValueError as exc:
        parser.error(str(exc))
    return int(any(row["status"] not in ("awaiting_review", "finalized", "insufficient_evidence", "reference_analysis") for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
