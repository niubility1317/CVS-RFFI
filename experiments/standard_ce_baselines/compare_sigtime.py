"""Reuse completed scores only; no checkpoint, IQ, truth or training reads."""
import csv
from pathlib import Path
import statistics
from .common import PROJECT,RUN,EXTENSION_RUN,SIG_SEEDS,read,write
from .model import COMMON


def compare(spec):
    records=[]
    for variant in (*COMMON,'residual_fusion','sig_time'):
        for seed in SIG_SEEDS:
            parent=spec['run_id'] if variant=='sig_time' else RUN if seed<2026092705 else EXTENSION_RUN
            score=read(Path(PROJECT)/'runs'/parent/f'{variant}-s{seed}'/'scores/scores.json')
            if score.get('status')!='SCORED' or score.get('truth_last') is not True:raise ValueError('Historical score incomplete')
            overall=next(r for r in score['results'] if r['dimension']=='overall')
            if overall['model_seed']!=seed or overall['variant']!=variant or overall['view']!='clean':raise ValueError('Historical row differs')
            records.append(dict(variant=variant,model_seed=seed,run_id=parent,
                                accuracy=overall['accuracy'],macro_f1=overall['macro_f1']))
    summary=[];lookup={(r['variant'],r['model_seed']):r for r in records}
    for variant in (*COMMON,'residual_fusion','sig_time'):
        values=[r for r in records if r['variant']==variant]
        diffs=[lookup['sig_time',s]['accuracy']-lookup[variant,s]['accuracy'] for s in SIG_SEEDS]
        summary.append(dict(variant=variant,actual_seed_count=len(values),
            accuracy_mean=statistics.mean(r['accuracy'] for r in values),
            accuracy_sample_sd=statistics.stdev(r['accuracy'] for r in values),
            macro_f1_mean=statistics.mean(r['macro_f1'] for r in values),
            macro_f1_sample_sd=statistics.stdev(r['macro_f1'] for r in values),
            sig_time_minus_model_accuracy_mean=statistics.mean(diffs),
            sig_time_minus_model_accuracy_sample_sd=statistics.stdev(diffs),
            sig_time_positive_pairs=sum(x>0 for x in diffs),paired_seeds=8))
    root=Path(spec['runtime_root'])
    write(root/'comparison_eight_seed.json',dict(status='ANALYZED',rows=records,
         groups=summary,truth_reopened=False,target_feedback_forbidden=True,
         claim='Matched split/E200/default-head pureCE comparison, not a field-deployment validation'))
    for name,rows in [('comparison_per_seed.csv',records),('comparison_summary.csv',summary)]:
        with (root/name).open('x',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
