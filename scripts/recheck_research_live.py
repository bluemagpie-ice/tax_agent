"""Opt-in live check with public facts, temporary SQLite, and read-only policy retrieval.

Run from the repository root: python -m scripts.recheck_research_live
Use --with-sources to test explicitly supplied official sources separately.
"""
import argparse
import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic

from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from sqlalchemy import create_engine

from agents.research_agent import ResearchAgent
from services.research_store import ResearchStore
from services.security import tenant_scope


async def run(root, amount, with_sources):
    question = (
        "我们是按季申报的小规模纳税人。2026年第一季度跨地级行政区提供建筑服务，"
        f"在项目所在地实现不含税销售额{amount}万元。需要在项目所在地预缴增值税吗？"
        "企业注册在江苏省南京市鼓楼区，项目在江苏省苏州市吴中区，承接办公楼室内装修，"
        "2026年1月开工、3月完工，只开具普通发票，未开专票，无其他销售、价款或预收款。"
    )
    if with_sources:
        question += ("\n供核对的政策原文：\n"
                     "https://fgk.chinatax.gov.cn/zcfgk/c102416/c5247497/content.html\n"
                     "https://fgk.chinatax.gov.cn/zcfgk/c102416/c5247434/content.html")
    engine = create_engine(f"sqlite:///{root / f'{amount}.sqlite'}")
    try:
        with tenant_scope("isolated_research_recheck"):
            async with AsyncSqliteSaver.from_conn_string(str(root / f"{amount}-graph.sqlite")) as saver:
                store = ResearchStore(engine)
                agent = ResearchAgent(store, workdir_root=root)
                agent._checkpointer = saver
                task = store.create(question, web_search=True, use_cases=False)
                started = monotonic()

                async def emit(_, event):
                    print(amount, f"{monotonic() - started:.2f}s", event["stage"], event["status"],
                          event.get("activity", ""), flush=True)

                task = await agent.execute(task["task_id"], task["revision"], emit=emit)
                if task["status"] == "awaiting_facts":
                    print(amount, "待确认事实", task["unknowns"], flush=True)
                    task = await agent.execute(task["task_id"], task["revision"], emit=emit,
                                               action={"action": "clarify", "skip_unknowns": True})
                print(amount, task["task_id"], task["status"], task.get("failure"), flush=True)
                print(amount, "初步证据覆盖", [(c["question"], c["sufficient"], c["gap"])
                                               for c in task.get("assessment", {}).get("coverage", [])], flush=True)
                print(amount, "引用", [(e["title"], e["source_url"]) for e in task.get("evidence", [])], flush=True)
                print(amount, "初稿", (task.get("report") or {}).get("answer_draft"), flush=True)
                for call in task.get("metrics", {}).get("tool_calls", []):
                    print(amount, call["tool"], call["args"]["positional"], call["status"], call["latency_ms"], flush=True)
                return task["status"] == "awaiting_review"
    finally:
        engine.dispose()


async def main(with_sources):
    with TemporaryDirectory(prefix="tax-research-recheck-") as directory:
        return all(await asyncio.gather(*(run(Path(directory), amount, with_sources) for amount in (30, 29))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-sources", action="store_true")
    options = parser.parse_args()
    load_dotenv(".env")
    raise SystemExit(0 if asyncio.run(main(options.with_sources)) else 1)
