"""Describe full-trajectory conflict across within-epoch labeled-CE quantiles."""
import argparse
from collections import Counter
import gzip
from itertools import zip_longest
import json
import math
from pathlib import Path


def quintiles(rows):
    if len(rows)!=222:raise ValueError('One complete 222-step epoch required')
    rows=sorted(rows,key=lambda r:r['CE']);out=[]
    for i in range(5):
        bucket=rows[i*222//5:(i+1)*222//5];c=Counter()
        for row in bucket:
            c['steps']+=1;c['CE_sum']+=row['CE'];c['L_norm_sum']+=row['L_norm']
            for name in ('full','ce'):
                value=row[name]
                if value is None:c[name+'_undefined']+=1
                else:
                    if not math.isfinite(value):raise ValueError('Finite direction required')
                    c[name+'_defined']+=1;c[name+'_conflict']+=int(value < -1e-6)
        out.append(dict(c))
    return out


def norm_strata(rows):
    """Within one epoch, compare CE by conflict status at similar L norm."""
    if len(rows)!=222:raise ValueError('One complete 222-step epoch required')
    rows=sorted(rows,key=lambda r:r['L_norm']);result=[]
    for i in range(5):
        bucket=rows[i*222//5:(i+1)*222//5];entry={}
        for anchor in ('full','ce'):
            c=Counter()
            for row in bucket:
                value=row[anchor]
                if value is None:continue
                if not math.isfinite(value):raise ValueError('Finite direction required')
                status='conflict' if value < -1e-6 else 'nonconflict'
                c[status+'_count']+=1;c[status+'_CE_sum']+=row['CE']
            entry[anchor]=dict(c)
        result.append(entry)
    return result


def records(path):
    with gzip.open(path,'rt',encoding='utf-8') as stream:
        for line in stream:yield json.loads(line)


def analyze(source):
    source=Path(source);done=json.loads((source/'complete.json').read_text(encoding='utf-8'))
    if (done['status']!='SOURCE_FROZEN' or done['epoch']!=50 or done['updates']!=11100
            or done['target_access'] or done['U_truth_read']):
        raise ValueError('Complete source-only E50 trajectory required')
    totals=[Counter() for _ in range(5)];epochs={};stratified={k:Counter() for k in ('full','ce')}
    for end,offset,expected in ((30,20,2220),(50,30,4440)):
        folder=source/f'segment{end}';seg=json.loads((folder/'complete.json').read_text(encoding='utf-8'))
        if seg['status']!='SOURCE_FROZEN' or seg['updates']!=expected or seg['target_access'] or seg['U_truth_read']:
            raise ValueError('Complete source-only segment required')
        rows=[];n=0
        for n,(metric,guard) in enumerate(zip_longest(records(folder/'step_metrics.jsonl.gz'),
                                                      records(folder/'step_guard_metrics.jsonl.gz')),1):
            if metric is None or guard is None or metric['step']!=n or guard['step']!=n:
                raise ValueError('Complete aligned step records required')
            m=metric['metrics'];ce=float(m['train/loss_tx_labeled']);ln=float(m['HPL_GP_t2_L_norm'])
            if not all(math.isfinite(v) and v>=0 for v in (ce,ln)):raise ValueError('Finite nonnegative CE and norm required')
            angles=guard['groups']['t2']['anchor_cosines_before']
            rows.append(dict(CE=ce,L_norm=ln,full=angles['L_full'],ce=angles['L_clean']))
            if n%222==0:
                bins=quintiles(rows);strata=norm_strata(rows);epoch=offset+n//222;epochs[str(epoch)]=bins
                for total,entry in zip(totals,bins):total.update(entry)
                for entry in strata:
                    for anchor,v in entry.items():
                        a,b=v.get('conflict_count',0),v.get('nonconflict_count',0)
                        if not a or not b:continue
                        diff=v['conflict_CE_sum']/a-v['nonconflict_CE_sum']/b;weight=min(a,b)
                        c=stratified[anchor];c['valid_strata']+=1;c['positive_strata']+=int(diff>0)
                        c['negative_strata']+=int(diff<0);c['weight_sum']+=weight;c['weighted_CE_difference_sum']+=weight*diff
                rows=[]
        if n!=expected or rows:raise ValueError('All source steps required')
    strata_result={}
    for anchor,c in stratified.items():
        value=dict(c);weight=value.get('weight_sum',0)
        value['weighted_conflict_minus_nonconflict_CE']=(value.get('weighted_CE_difference_sum',0)/weight if weight else None)
        strata_result[anchor]=value
    return dict(status='ANALYZED_DUAL_BATCH_DIFFICULTY',row_id=done['config']['row_id'],
        epochs=epochs,overall_quintiles=[dict(v) for v in totals],target_inputs_read=False,U_truth_read=False,
        within_epoch_L_norm_quintile_stratification=strata_result,
        interpretation='Within-epoch batch ranking by ordinary L CE. Conflict is the actual extra '
        'AdamW descent vs current anchor before projection. Training batches are repeated observations; '
        'associations are descriptive, not independent samples or causal effects.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);a=p.parse_args()
    print(json.dumps(analyze(a.source),ensure_ascii=False,allow_nan=False))
