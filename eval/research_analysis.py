"""Exploratory paired summaries; quality data must come from attributed human scoring."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eval.research_eval import QUALITY_FIELDS


def summarize(directory, scores=None):
    directory = Path(directory)
    runs = pd.read_csv(directory / 'runs.csv')
    if runs['smoke_test'].astype(str).str.lower().eq('true').any():
        note = 'Contains scripted smoke runs; no tax-quality or model-efficiency inference is valid.'
    else:
        note = 'Exploratory only. Intervals resample task families; repeated runs are not independent tasks.'
    if scores:
        quality = pd.read_csv(scores)
        required = ['run_id','scorer_id','scoring_method','scored_at','basis']
        if quality[required].isna().any().any() or not quality.scoring_method.eq('human').all():
            raise ValueError('Quality scores require attributed human reviewer, timestamp and scoring basis')
        if not set(quality.run_id).issubset(set(runs.run_id)):
            raise ValueError('Scores contain unknown run IDs')
        if quality.duplicated(['run_id','scorer_id']).any():
            raise ValueError('Duplicate scorer/run records')
        for field in QUALITY_FIELDS:
            quality[field] = pd.to_numeric(quality[field],errors='raise')
        binary = [f for f in QUALITY_FIELDS if f not in ('report_completeness','human_edit_count')]
        if any(not quality[f].dropna().isin([0,1]).all() for f in binary):
            raise ValueError('Binary quality values must be 0/1 or missing')
        if not quality.report_completeness.dropna().between(0,1).all() or (quality.human_edit_count.dropna() < 0).any():
            raise ValueError('Completeness must be 0..1; edit count must be nonnegative')
        quality.to_csv(directory / 'quality_scores.csv',index=False,encoding='utf-8-sig')
        # Rater means are descriptive ratings, not automatically adjudicated gold answers.
        averaged = quality.groupby('run_id')[QUALITY_FIELDS].mean()
        runs = runs.merge(averaged,on='run_id',how='left',validate='one_to_one')
    metrics = [m for m in ['latency_ms','total_tokens','tool_call_count','retrieval_count','llm_call_count','estimated_cost',*QUALITY_FIELDS] if m in runs]
    for metric in metrics:
        runs[metric] = pd.to_numeric(runs[metric],errors='coerce')
    descriptive = runs.groupby('experiment_group')[metrics].agg(['count','mean','std'])
    descriptive.to_csv(directory / 'descriptive.csv',encoding='utf-8-sig')
    paired, differences = [], []
    rng = np.random.default_rng(17)
    for metric in metrics:
        # First collapse repeats by task, then resample whole families.
        table = runs.groupby(['task_id','case_family','experiment_group'])[metric].mean().unstack('experiment_group')
        for first,second in [('A','B'),('A','C'),('B','C')]:
            if first not in table or second not in table:
                continue
            values = (table[second] - table[first]).dropna()
            if values.empty:
                continue
            for (task_id,family),value in values.items():
                differences.append({'metric':metric,'comparison':second+'-'+first,'task_id':task_id,'case_family':family,'difference':value})
            family_values = values.groupby(level='case_family').mean()
            interval = [None,None]
            if len(family_values) >= 2:
                boot = rng.choice(family_values.to_numpy(),size=(2000,len(family_values)),replace=True).mean(axis=1)
                interval = np.quantile(boot,[.025,.975]).tolist()
            paired.append({'metric':metric,'comparison':second+'-'+first,'paired_tasks':len(values),'case_families':len(family_values),
                'mean_task_difference':float(values.mean()), 'mean_family_difference':float(family_values.mean()),
                'family_bootstrap_95_low':interval[0],'family_bootstrap_95_high':interval[1]})
    pd.DataFrame(differences).to_csv(directory / 'paired_task_differences.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(paired).to_csv(directory / 'paired_summary.csv',index=False,encoding='utf-8-sig')
    (directory / 'analysis_notes.json').write_text(json.dumps({'notice':note,'ratings':'Human rater means are not adjudicated gold.',
        'interval_method':'percentile bootstrap over case-family means, 2000 resamples, seed 17; small samples remain exploratory'},indent=2),encoding='utf-8')
    return paired


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory'); parser.add_argument('--scores')
    args = parser.parse_args()
    print(json.dumps(summarize(args.directory,args.scores),ensure_ascii=False,indent=2))
