"""Paired A/B/C research experiments. No production writes and no automatic gold scores.

python -m eval.research_eval --config eval/research/config.json --smoke
python -m eval.research_eval --config eval/research/config.json --live --allow-unverified
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import random
import time
import uuid
import importlib.metadata
import platform
from pathlib import Path

from langchain_core.messages import AIMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from sqlalchemy import create_engine

from agents.research_agent import ResearchAgent
from services.research_cases import CaseStore, terms
from services.research_store import ResearchStore, utcnow
from services.security import tenant_scope

QUALITY_FIELDS = ["conclusion_correct", "major_tax_error", "citation_real", "citation_supports_conclusion",
    "temporal_applicability", "regional_applicability", "missing_facts_identified", "case_differences_identified",
    "inherited_historical_error", "report_completeness", "human_edit_count"]


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


class SnapshotTools:
    """Frozen public excerpts; lexical RAG and official web replay, NOT live Cloud acceptance."""
    def __init__(self, records):
        self.records = records

    def search(self, query, limit=4, **kwargs):
        query_terms = terms(query)
        ranked = sorted(self.records, key=lambda r: len(query_terms & terms(r["title"] + r["content"])), reverse=True)
        return [r for r in ranked if query_terms & terms(r["title"] + r["content"])][:limit]

    def web_search(self, query, limit=2, **kwargs):
        return [{"title": r["title"], "url": r["source_url"]} for r in self.search(query, limit)]

    def page(self, url, **kwargs):
        record = next((r for r in self.records if r["source_url"] == url), None)
        return {**record, "url":url, "status":"success"} if record else {"status":"error", "url":url}


class SmokeModel:
    """Exercises transport/schema only. Never a tax-quality experiment or standard answer."""
    model_name = "scripted-smoke-not-an-llm"

    def invoke(self, messages, **kwargs):
        instruction = messages[0].content
        payload = json.loads(messages[-1].content)
        if '"title": "Understanding"' in instruction:
            result = {"facts":[], "unknowns":["模拟执行未进行业务事实核验"], "questions":[payload['query'][:500]]}
        elif '"title": "ResearchPlan"' in instruction:
            result = {"steps":[{"step_id":"s1", "question":payload['questions'][0], "depends_on":[]}]}
        elif '"title": "Assessment"' in instruction:
            result = {"coverage":[], "next_queries":[]}
        else:
            result = {"scope":"替身流程测试", "analysis":[], "uncertainties":["未进行真实模型研究"],
                      "follow_up_questions":[], "answer_draft":"这是流程替身输出，不是税务研究答案。"}
        return AIMessage(content=json.dumps(result, ensure_ascii=False))


def usage_summary(data, pricing=None):
    calls = data.get('metrics', {}).get('llm_calls', [])
    tools = data.get('metrics', {}).get('tool_calls', [])
    usage = [call.get('usage') for call in calls]
    complete = bool(calls) and all(isinstance(u, dict) and all(u.get(k) is not None for k in ('input_tokens','output_tokens','total_tokens')) for u in usage)
    totals = {k:sum(u[k] for u in usage) if complete else None for k in ('input_tokens','output_tokens','total_tokens')}
    cost = None
    if complete and pricing and all(k in pricing for k in ('input_per_million', 'output_per_million', 'currency')):
        cost = (totals['input_tokens'] * pricing['input_per_million'] + totals['output_tokens'] * pricing['output_per_million']) / 1_000_000
    return {**totals, "usage_complete":complete, "estimated_cost":cost, "cost_currency":(pricing or {}).get('currency'),
        "tool_call_count":len(tools), "retrieval_count":sum(t['tool'].startswith('search_') for t in tools),
        "llm_call_count":len(calls), "retry_count":sum(bool(c.get('schema_repair')) for c in calls),
        "retry_count_basis":"schema_repair_calls_counted; live_sdk_max_retries_zero",
        "reported_response_models":sorted({c['response_model_id'] for c in calls if c.get('response_model_id')})}


async def fixed_sop(agent, data):
    """C: fixed understand → retrieve → assess → report, with the same evidence gate."""
    data = await agent.understand(data)
    data['plan'] = [{"question":query} for query in data['questions']]
    data = await agent.retrieve(data)
    data = await agent.calculate(data)
    data = await agent.assess(data)
    return await agent.report(data)


def write_csv(path, rows, fields=None):
    fields = fields or list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict,list)) else v for k,v in row.items() if k in fields})


async def run_experiments(config, *, smoke=False, allow_unverified=False, limit=None, output=None):
    base = Path(config.get('_base', '.'))
    dataset = json.loads((base / config['dataset']).read_text(encoding='utf-8'))
    policy = json.loads((base / config['policy_snapshot']).read_text(encoding='utf-8'))
    cases = json.loads((base / config['case_library']).read_text(encoding='utf-8'))
    tasks = [t for t in dataset['tasks'] if t.get('split') == config.get('split','evaluation')]
    if not allow_unverified and any(t.get('human_validation', {}).get('status') != 'verified' for t in tasks) and not smoke:
        raise ValueError('测试题尚未完成人工核验。请先核验，或使用 --allow-unverified 明确开展探索性运行')
    if limit:
        tasks = tasks[:limit]
    if not tasks:
        raise ValueError('所选数据划分没有测试任务')
    case_families = {c['case_family'] for c in cases['cases']}
    development_families = {t['case_family'] for t in dataset['tasks'] if t['split'] == 'development'}
    evaluation_families = {t['case_family'] for t in dataset['tasks'] if t['split'] == 'evaluation'}
    if development_families & evaluation_families:
        raise ValueError('开发集和评测集存在相同案例家族')
    out = Path(output or config.get('output_dir','data/research_eval')) / (utcnow().replace(':','-') + '-' + uuid.uuid4().hex[:8])
    out.mkdir(parents=True)
    tools = SnapshotTools(policy['records'])
    engine = create_engine(f"sqlite:///{(out / 'business.sqlite').resolve()}")
    store, case_store = ResearchStore(engine), CaseStore(engine)
    groups = config.get('groups',['A','B','C'])
    if not groups or any(group not in ('A','B','C') for group in groups):
        raise ValueError('experiment groups must be A/B/C')
    if not smoke:
        # Snapshot config once per experiment; never serialize credentials.
        from services.config_manager import ConfigManager
        from langchain_openai import ChatOpenAI
        from dotenv import load_dotenv
        load_dotenv()
        cfg = ConfigManager().get_llm_config()
        model = ChatOpenAI(model=cfg['model'], api_key=cfg['api_key'], base_url=cfg['base_url'], temperature=0, max_retries=0, timeout=45)
    else:
        model = SmokeModel()
    controls = {"model_id":getattr(model,'model_name',None), "model_parameters":{"temperature":0, "max_retries":0,
        "reasoning_effort":"low" if getattr(model,'model_name',None) == 'glm-5.3' else None},
        "dataset_version":dataset['version'], "dataset_hash":digest(dataset), "policy_snapshot_id":digest(policy),
        "case_library_version":digest(cases), "research_guidance_version":"research-v1", "tool_backend":"frozen_public_keyword_rag_and_web_replay",
        "budget":config.get('budget',{}), "schedule_seed":config.get('schedule_seed',17),
        "smoke_test":smoke, "source_accessed_at":policy.get('accessed_at'), "random_seed":None,
        "exploratory":True, "human_gold_available":False}
    project = Path(__file__).resolve().parents[1]
    source_files = [project / 'agents/research_agent.py', project / 'agents/streaming.py',
                    *sorted((project / 'services').glob('research_*.py')),
                    *sorted((project / 'eval').glob('research_*.py'))]
    source_hashes = {str(p.relative_to(project)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    controls.update(code_snapshot_id=digest(source_hashes),source_hashes=source_hashes,python_version=platform.python_version(),
        dependency_versions={name:importlib.metadata.version(name) for name in ('langgraph','langchain-core','langchain-openai','SQLAlchemy','openai')})
    (out / 'manifest.json').write_text(json.dumps(controls, ensure_ascii=False, indent=2),encoding='utf-8')
    # Freeze copies, not just labels: later edits to input files cannot alter this experiment.
    for filename, value in [('dataset.json',dataset),('policy_snapshot.json',policy),('case_library.json',cases)]:
        (out / filename).write_text(json.dumps(value, ensure_ascii=False, indent=2),encoding='utf-8')
    rows = []
    schedule = [(t,repeat,g) for t in tasks for repeat in range(1,config.get('repeats',1)+1) for g in groups]
    random.Random(controls['schedule_seed']).shuffle(schedule)
    with tenant_scope('eval-isolated-owner'):
        availability = {}
        for item in cases['cases']:
            with tenant_scope('eval-foreign-owner' if item.get('foreign_owner') else 'eval-isolated-owner'):
                case = case_store.create(item['content'])
                availability[case['case_id']] = item.get('available_at')
                if item.get('publish', True):
                    case = case_store.change(case['case_id'],case['revision'],'consent')
                    case = case_store.change(case['case_id'],case['revision'],'publish')
                if item.get('withdrawn'):
                    case_store.change(case['case_id'],case['revision'],'withdraw')
        async with AsyncSqliteSaver.from_conn_string(str(out / 'checkpoints.sqlite')) as saver:
            agent = ResearchAgent(store, case_store=case_store, llm=model, search=tools.search,
                web_search=tools.web_search, read_page=tools.page, workdir_root=out)
            agent._checkpointer = saver
            for test, repeat, group in schedule:
                started = time.perf_counter()
                run_id = uuid.uuid4().hex
                data = store.create(test['query'], web_search=True, use_cases=group == 'B',
                    budget={**config.get('budget',{}), 'clarifications':0}, evaluation_run=True)
                class CutoffCases:
                    def search(self, query, facts=None, limit=3):
                        candidates = case_store.search(query,facts=facts,limit=max(len(availability),limit))
                        cutoff = test.get('as_of')
                        return [c for c in candidates if not cutoff or (availability.get(c['case_id']) and availability[c['case_id']] <= cutoff)][:limit]
                agent.case_store = CutoffCases()
                error = None
                try:
                    if group == 'C':
                        lease = store.claim(data['task_id'],data['revision'])
                        try:
                            data = await fixed_sop(agent,data)
                            data = store.save(data,lease)
                        except Exception as exc:
                            data.update(status="failed", error_type=getattr(exc, "code", type(exc).__name__))
                            data = store.save(data,lease)
                            raise
                        finally:
                            store.release(data['task_id'],lease)
                    else:
                        data = await agent.execute(data['task_id'],data['revision'])
                except Exception as exc:
                    data = store.get(data['task_id'])
                    error = data.get("error_type") or type(exc).__name__
                error = error or (data.get("failure") or {}).get("code")
                row = {"run_id":run_id,"task_id":test['task_id'],"runtime_task_id":data['task_id'],
                    "experiment_group":group,"repeat_index":repeat,"case_family":test['case_family'],
                    "test_category":test['test_category'],"run_timestamp":utcnow(), **controls,
                    "latency_ms":round((time.perf_counter()-started)*1000), **usage_summary(data,config.get('pricing')),
                    "task_status":"failed" if error else data['status'],"error_type":error,
                    "human_validation_status":test.get('human_validation',{}).get('status','unknown'),
                    "leakage_flags":test.get('leakage_flags',[]) + (["case_family_overlap"] if test['case_family'] in case_families else []),
                    "clarification_policy":"use_dataset_facts_then_keep_unknowns_in_all_groups"}
                rows.append(row)
                with (out / 'runs.jsonl').open('a',encoding='utf-8') as f:
                    f.write(json.dumps({"run":row,"task":data},ensure_ascii=False)+'\n')
                # Incremental export survives partial/provider-failure runs.
                write_csv(out / 'runs.csv',rows)
    quality = [{"run_id":row['run_id'],"scorer_id":"","scoring_method":"human","scored_at":"",
                "rubric_version":"v1","basis":"", **{field:None for field in QUALITY_FIELDS}} for row in rows]
    write_csv(out / 'quality_scores_template.csv',quality)
    (out / 'summary.md').write_text(
        '# Research experiment\n\n' + ('Scripted smoke test: no tax-quality conclusions.\n\n' if smoke else 'Exploratory model runs; human quality scores are pending.\n\n')
        + f'Runs: {len(rows)}. Failed: {sum(bool(r["error_type"]) for r in rows)}.\n\n'
        + 'A/B isolate case reuse; C uses fixed SOP without the research graph. Frozen public excerpts are incomplete and are not a live Qdrant/network benchmark. '
        + 'No significance or superiority claim is supported by this export. Missing usage/cost/quality values remain blank.\n', encoding='utf-8')
    engine.dispose()
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='eval/research/config.json')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--smoke',action='store_true'); mode.add_argument('--live',action='store_true')
    parser.add_argument('--allow-unverified',action='store_true'); parser.add_argument('--limit',type=int)
    parser.add_argument('--output')
    args = parser.parse_args()
    path = Path(args.config).resolve()
    config = json.loads(path.read_text(encoding='utf-8')); config['_base'] = str(path.parent)
    print(asyncio.run(run_experiments(config,smoke=args.smoke,allow_unverified=args.allow_unverified,limit=args.limit,output=args.output)))


if __name__ == '__main__':
    main()
