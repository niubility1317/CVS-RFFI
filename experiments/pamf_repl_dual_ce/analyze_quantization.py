"""Aggregate discarded-raw-step upper bounds for completed dual-anchor training."""
import argparse
from collections import Counter
import gzip
import json
import math
from pathlib import Path

GROUPS=('t2','cls_head','other')


def aggregate(segment,*,offset,steps):
    folder=Path(segment)
    done=json.loads((folder/'complete.json').read_text(encoding='utf-8'))
    if (done['status']!='SOURCE_FROZEN' or done['updates']!=steps or done['target_access']
            or done['U_truth_read']):
        raise ValueError('Complete source-only frozen segment required')
    epochs=[json.loads(x) for x in (folder/'metrics_epoch.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
    if len(epochs)*222!=steps:raise ValueError('Every epoch must have 222 source steps')
    result={offset+i+1:{'lr':float(row['lr']),'groups':{g:Counter() for g in GROUPS}}
            for i,row in enumerate(epochs)}
    if any(not math.isfinite(v['lr']) or v['lr']<0 for v in result.values()):
        raise ValueError('Finite nonnegative logged learning rates required')
    n=0
    with gzip.open(folder/'step_guard_metrics.jsonl.gz','rt',encoding='utf-8') as stream:
        for n,line in enumerate(stream,1):
            row=json.loads(line)
            if row['step']!=n:raise ValueError('All ordered actual parameter-step records required')
            epoch=offset+(n-1)//222+1
            for group in GROUPS:
                d=row['groups'][group];raw=float(d['raw_increment_norm']);applied=float(d['applied_increment_norm'])
                if not all(math.isfinite(v) and v>=0 for v in (raw,applied)):
                    raise ValueError('Finite nonnegative actual increment norms required')
                if not d['actual_parameter_check'] or not d['norm_nonexpansion']:
                    raise ValueError('Actual step checks required')
                fallback=d['quantization_fallback']
                if fallback and applied!=0:raise ValueError('Fallback must produce a zero extra step')
                c=result[epoch]['groups'][group];c['steps']+=1;c['fallback_steps']+=int(fallback)
                c['raw_norm_sum']+=raw;c['applied_norm_sum']+=applied
                if fallback:c['fallback_raw_norm_sum']+=raw
    if n!=steps or any(v['groups'][g]['steps']!=222 for v in result.values() for g in GROUPS):
        raise ValueError('Complete aligned source steps required')
    return {str(epoch):{'lr':v['lr'],'groups':{g:dict(c) for g,c in v['groups'].items()}}
            for epoch,v in result.items()}


def analyze(source):
    source=Path(source);done=json.loads((source/'complete.json').read_text(encoding='utf-8'))
    if (done['status']!='SOURCE_FROZEN' or done['epoch']!=50 or done['updates']!=11100
            or done['target_access'] or done['U_truth_read']):
        raise ValueError('Complete source-only E50 trajectory required')
    epochs={}
    for end,offset,steps in ((30,20,2220),(50,30,4440)):
        epochs.update(aggregate(source/f'segment{end}',offset=offset,steps=steps))
    totals={}
    for group in GROUPS:
        total=Counter()
        for epoch in epochs.values():total.update(epoch['groups'][group])
        total=dict(total)
        total['fallback_raw_share_upper_bound']=(total.get('fallback_raw_norm_sum',0)/total['raw_norm_sum']
                                                   if total['raw_norm_sum'] else None)
        totals[group]=total
    return dict(status='ANALYZED_DUAL_QUANTIZATION_MAGNITUDES',row_id=done['config']['row_id'],
        epochs=epochs,totals=totals,target_inputs_read=False,U_truth_read=False,
        interpretation='Fallback raw norm bounds any nonexpansive projected extra step, but is not '
        'the exact recoverable step, parameter displacement, loss or clean performance effect. '
        'Norm sums do not account for cancellation between steps or parameter groups.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);a=p.parse_args()
    print(json.dumps(analyze(a.source),ensure_ascii=False,allow_nan=False))
