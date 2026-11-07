import asyncio
import csv
import json
from pathlib import Path

import pytest

from eval.research_eval import run_experiments, usage_summary
from eval.research_analysis import summarize


def test_paired_exports_keep_missing_usage_and_quality_unscored(tmp_path,monkeypatch):
    monkeypatch.setattr('agents.streaming.count_tokens',lambda text,name='':(len(str(text)), 'test_estimate'))
    root = Path(__file__).resolve().parents[1] / 'eval' / 'research'
    config = json.loads((root / 'config.example.json').read_text(encoding='utf-8'))
    config.update(_base=str(root),repeats=2)
    out = asyncio.run(run_experiments(config,smoke=True,limit=1,output=tmp_path))
    rows = list(csv.DictReader((out / 'runs.csv').open(encoding='utf-8-sig')))
    assert len(rows) == 6
    assert {r['experiment_group'] for r in rows} == {'A','B','C'}
    assert all(r['total_tokens'] == '' and r['estimated_cost'] == '' for r in rows)
    assert all(r['task_status'] == 'failed' for r in rows)
    raw = [json.loads(line) for line in (out / 'runs.jsonl').read_text(encoding='utf-8').splitlines()]
    assert all('task' in r and r['run']['smoke_test'] for r in raw)
    assert all(not r['task'].get('case_references') for r in raw if r['run']['experiment_group'] != 'B')
    assert all('case_references' in r['task'] for r in raw if r['run']['experiment_group'] == 'B')
    scores = list(csv.DictReader((out / 'quality_scores_template.csv').open(encoding='utf-8-sig')))
    assert all(s['conclusion_correct'] == '' for s in scores)
    paired = summarize(out)
    assert paired and all(row['paired_tasks'] == 1 for row in paired)


def test_unverified_dataset_cannot_silently_become_formal_eval(tmp_path):
    root = Path(__file__).resolve().parents[1] / 'eval' / 'research'
    config = json.loads((root / 'config.example.json').read_text(encoding='utf-8'))
    config['_base'] = str(root)
    with pytest.raises(ValueError,match='人工核验'):
        asyncio.run(run_experiments(config,output=tmp_path))


def test_usage_and_price_only_use_reported_provider_values():
    data={'metrics':{'llm_calls':[{'usage':{'input_tokens':100,'output_tokens':50,'total_tokens':150}}], 'tool_calls':[]}}
    measured=usage_summary(data,{'input_per_million':1,'output_per_million':2,'currency':'CNY'})
    assert measured['total_tokens'] == 150 and measured['estimated_cost'] == .0002
    data['metrics']['llm_calls'].append({'error_type':'Timeout'})
    assert usage_summary(data)['total_tokens'] is None
