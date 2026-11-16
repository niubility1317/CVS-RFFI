"""Fixed matched reference comparisons after all runs independently score."""
import argparse
from collections import Counter
import gzip
import json
import math
from pathlib import Path

SETTINGS=('ten_class','source_rx_6')
WEIGHTS=(3.,6.)
CAPS=(.25,.5)
GROUPS=('t2','cls_head','other')


def comparison_plan():
    return [dict(setting=s,weight=w,cap=c,comparison=name,baseline=base,candidate=candidate)
            for s in SETTINGS for w in WEIGHTS for c in CAPS
            for name,base,candidate in (('CE_vs_full','full','CE'),('full_vs_dual','dual','full'),('CE_vs_dual','dual','CE'))]


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def dominance_stats(source,cap):
    total={g:Counter() for g in GROUPS}
    for end,steps in ((30,2220),(50,4440)):
        p=Path(source)/f'segment{end}'/'step_guard_metrics.jsonl.gz';n=0
        with gzip.open(p,'rt',encoding='utf-8') as stream:
            for n,line in enumerate(stream,1):
                value=json.loads(line)
                if value['step']!=n:raise ValueError('Ordered step guard telemetry required')
                for group in GROUPS:
                    d=value['groups'][group]
                    if not d.get('reference_dominance_checked') or d['reference_extra_cap']!=cap:
                        raise ValueError('Every nonempty group must verify its registered reference cap')
                    ratio=d['reference_extra_ratio'];cosine=d['reference_dominance_cosine']
                    if ratio is not None and (not math.isfinite(ratio) or ratio>cap*(1+1e-6)):
                        raise ValueError('Actual extra step exceeds reference cap')
                    if cosine is not None and (not math.isfinite(cosine) or cosine<math.sqrt(1-cap*cap)-1e-6):
                        raise ValueError('Actual update is outside the registered reference cone')
                    total[group]['reference_zero_fallback']+=int(d['reference_zero_fallback'])
                    total[group]['reference_rounding_fallback']+=int(d['reference_rounding_fallback'])
                    total[group]['reference_cosine_undefined']+=int(cosine is None)
                    if ratio is not None:
                        total[group]['ratio_sum']+=ratio;total[group]['ratio_defined']+=1
                    if cosine is not None:
                        total[group]['cosine_sum']+=cosine;total[group]['cosine_defined']+=1
        if n!=steps:raise ValueError('All actual reference step records required')
    return {g:dict(v) for g,v in total.items()}


def analyze(dual_root,full_root,ce_root):
    import numpy as np
    from experiments.pamf_repl_dual_ce.analyze_segment import analyze as segment
    from experiments.pamf_repl_weight.analyze_paired import compare,prediction
    roots={k:Path(v) for k,v in dict(dual=dual_root,full=full_root,CE=ce_root).items()}
    cells={};context=None
    for mode,root in roots.items():
        if read(root/'pipeline_state.json')['status']!='ANALYZED':
            raise ValueError('All registered rows and independent scores must finish')
        runtime=read(root.parents[1]/'releases'/(root.name+'-release')/'runtime.json')
        current_context=read(runtime['context'])
        if context is None:context=current_context
        for rid,entry in read(root/'sources.json').items():
            source=Path(entry['source_output']);done=read(source/'complete.json');cfg=done['config']
            setting=next((s for s in SETTINGS if rid.startswith(s+'-')),None)
            if (setting is None or cfg['row_id']!=rid or cfg['model_seed']!=2026092701
                    or done['status']!='SOURCE_FROZEN' or done['epoch']!=50 or done['updates']!=11100
                    or done['target_access'] or done['U_truth_read']):
                raise ValueError('Matching complete source-only candidate required')
            recipe=cfg['tuning_recipe'];weight=float(recipe['weight'])
            cap=None if mode=='dual' else float(recipe['reference_cap'])
            key=(mode,setting,weight,cap)
            if key in cells:raise ValueError('Duplicate registered reference cell')
            scored=read(Path(entry['scores'])/'scores.json');sdone=read(Path(entry['scores'])/'complete.json')
            if sdone['status']!='SCORED_COMPLETE' or not scored['truth_last']:
                raise ValueError('Frozen independent clean score required')
            items=[x for x in scored['results'] if x['test'] in ('cross_rx','cross_both')
                   and x['readout']=='raw' and x['view']=='clean' and x['dimension']=='overall']
            if len(items)!=2 or {x['test'] for x in items}!={'cross_rx','cross_both'}:
                raise ValueError('Both main clean tests required')
            count=sum(x['count'] for x in items)
            correct=sum(sum(row[i] for i,row in enumerate(x['confusion'])) for x in items)
            uroot=root/'readouts'/rid/'U_diagnostic';u=read(uroot/'scores.json');udone=read(uroot/'complete.json')
            if (udone['status']!='SCORED_U_DIAGNOSTIC' or not udone['truth_last']
                    or udone['source_epoch']!=50 or u['training_or_selection_use'] is not False):
                raise ValueError('Independent frozen U diagnostic required')
            segments=[segment(source/f'segment{end}',offset=offset,expected_steps=steps)
                      for end,offset,steps in ((30,20,2220),(50,30,4440))]
            if current_context['handles'][cfg['protocol_id']]['source_contract']!=context['handles'][cfg['protocol_id']]['source_contract']:
                raise ValueError('Matched source contract handle required')
            o=u['overall']; cells[key]=dict(row_id=rid,entry=entry,protocol_id=cfg['protocol_id'],
                clean=dict(count=count,correct=correct,accuracy=correct/count),segments=segments,
                U=dict(seen=o['seen'],accepted=o['accepted'],accepted_correct=o['accepted_correct'],
                       acceptance=o['accepted']/o['seen'],accuracy=o['accepted_correct']/o['accepted'] if o['accepted'] else None,
                       unique_total=u['unique_U_total'],unique_accepted=u['unique_accepted']),
                dominance=None if cap is None else dominance_stats(source,cap),
                extra_semantics='U relative to full-L-only' if mode!='CE' else 'other L and U relative to ordinary CE-only')
    expected={(m,s,w,c) for m in ('full','CE') for s in SETTINGS for w in WEIGHTS for c in CAPS}
    expected.update(('dual',s,w,None) for s in SETTINGS for w in WEIGHTS)
    if set(cells)!=expected:raise ValueError('Complete matched reference matrices required')
    truths={}
    for setting in SETTINGS:
        protocols={v['protocol_id'] for k,v in cells.items() if k[1]==setting}
        if len(protocols)!=1:raise ValueError('One target physical-ID contract per setting required')
        protocol=protocols.pop();truths[setting]={task:read(context['handles'][protocol]['tests'][task]['truth'])
                                                   for task in ('cross_rx','cross_both')}
    predictions={k:{task:prediction(v['entry']['predictions'],task) for task in truths[k[1]]} for k,v in cells.items()}
    comparisons=[]
    for plan in comparison_plan():
        s,w,c=plan['setting'],plan['weight'],plan['cap']
        def key(mode):return (mode,s,w,None if mode=='dual' else c)
        base,candidate=key(plan['baseline']),key(plan['candidate']);total=Counter();per_task={}
        for task,truth in truths[s].items():
            ids,pred=predictions[candidate][task];old_ids,old=predictions[base][task]
            if not np.array_equal(ids,old_ids):raise ValueError('Exact same physical IDs required')
            per_task[task]=compare(ids,pred,old,truth);total.update(per_task[task]['overall'])
        if total['candidate_correct']!=cells[candidate]['clean']['correct'] or total['baseline_correct']!=cells[base]['clean']['correct']:
            raise ValueError('Paired predictions differ from independent score')
        comparisons.append(dict(**plan,combined=dict(total),per_test=per_task,delta_pp=100*total['net']/total['count']))
    return dict(status='ANALYZED_MATCHED_REFERENCE_COMPARISONS',
        cells=[dict(key=list(key),**{k:v for k,v in value.items() if k!='entry'}) for key,value in sorted(cells.items(),key=lambda x:str(x[0]))],
        comparisons=comparisons,
        interpretation='All fixed comparisons on previously exposed targets and selected seed; CE-reference extra step includes other L and U. No feedback to training or selection.')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('dual-root','full-root','ce-root'):p.add_argument('--'+name,required=True)
    a=p.parse_args();print(json.dumps(analyze(a.dual_root,a.full_root,a.ce_root),ensure_ascii=False,allow_nan=False))
