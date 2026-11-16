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


def comparison_plan(include_repair=False):
    plans = [dict(setting=s,weight=w,cap=c,comparison=name,baseline=base,candidate=candidate)
            for s in SETTINGS for w in WEIGHTS for c in CAPS
            for name,base,candidate in (('CE_vs_full','full','CE'),('full_vs_dual','dual','full'),('CE_vs_dual','dual','CE'),
                ('EMA_vs_full','full','EMA'),('EMA_vs_CE','CE','EMA'),('EMA_vs_dual','dual','EMA'))]
    if include_repair:
        plans.extend(dict(setting=s,weight=w,cap=.25,comparison='repair_vs_full',baseline='full',candidate='repair')
                     for s in SETTINGS for w in WEIGHTS)
    return plans


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def dominance_stats(source,cap,mode=None):
    total={g:Counter() for g in GROUPS}
    for end,steps in ((30,2220),(50,4440)):
        p=Path(source)/f'segment{end}'/'step_guard_metrics.jsonl.gz';n=0
        if mode=='EMA':
            state=read(p.parent/'repl_state.json')['U_parameter_step_guard']
            if state['reference_gradient_EMA_updates']!=steps or state['reference_gradient_beta']!=.9:
                raise ValueError('Every source step must update the registered labeled-gradient EMA')
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
                    raw=float(d['raw_increment_norm']);reference=float(d['reference_step_norm'])
                    if not all(math.isfinite(v) and v>=0 for v in (raw,reference)):
                        raise ValueError('Finite nonnegative actual step norms required')
                    upper=min(raw,cap*reference)
                    total[group]['raw_increment_norm_sum']+=raw
                    total[group]['capped_increment_upper_sum']+=upper
                    if d['reference_rounding_fallback']:
                        total[group]['reference_fallback_raw_norm_sum']+=raw
                        total[group]['reference_fallback_capped_upper_sum']+=upper
                    total[group]['reference_zero_fallback']+=int(d['reference_zero_fallback'])
                    total[group]['reference_rounding_fallback']+=int(d['reference_rounding_fallback'])
                    total[group]['reference_cosine_undefined']+=int(cosine is None)
                    if mode=='repair':
                        total[group]['reference_cap_repair_attempts']+=int(d['reference_cap_repair_attempts'])
                        total[group]['reference_cap_repair_used']+=int(d['reference_cap_repair_used'])
                        total[group]['reference_cap_repair_nonzero']+=int(d['reference_cap_repair_nonzero'])
                    if ratio is not None:
                        total[group]['ratio_sum']+=ratio;total[group]['ratio_defined']+=1
                    if cosine is not None:
                        total[group]['cosine_sum']+=cosine;total[group]['cosine_defined']+=1
                    if mode=='EMA':
                        fact=value['reference']['current_full_L_vs_reference_gradient'][group]
                        direction=fact['cosine']
                        total[group]['current_L_vs_EMA_undefined']+=int(direction is None)
                        if direction is not None:
                            if not math.isfinite(direction):raise ValueError('Finite current L vs EMA direction required')
                            total[group]['current_L_vs_EMA_cosine_sum']+=direction
                            total[group]['current_L_vs_EMA_defined']+=1
                            total[group]['current_L_vs_EMA_conflict']+=int(direction < -1e-6)
        if n!=steps:raise ValueError('All actual reference step records required')
    result={g:dict(v) for g,v in total.items()}
    for values in result.values():
        raw=values.get('raw_increment_norm_sum',0.)
        values['reference_fallback_raw_norm_share']=(values.get('reference_fallback_raw_norm_sum',0.)/raw if raw else None)
    return result


def analyze(dual_root,full_root,ce_root,ema_root,repair_root=None,repair_only=False):
    import numpy as np
    from experiments.pamf_repl_dual_ce.analyze_segment import analyze as segment
    from experiments.pamf_repl_weight.analyze_paired import compare,prediction
    if repair_only and repair_root is None:raise ValueError('Repair root required for repair-only comparison')
    selected = dict(full=full_root) if repair_only else dict(dual=dual_root,full=full_root,CE=ce_root,EMA=ema_root)
    if any(v is None for v in selected.values()):raise ValueError('All requested comparison roots required')
    roots={k:Path(v) for k,v in selected.items()}
    if repair_root is not None:roots['repair']=Path(repair_root)
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
                dominance=None if cap is None else dominance_stats(source,cap,mode),
                extra_semantics={'dual':'U relative to full-L-only','full':'U relative to full-L-only',
                    'CE':'other L and U relative to ordinary CE-only','EMA':'current L minus EMA reference and all U',
                    'repair':'U relative to full-L-only, with inward rounding repair'}[mode])
    modes=('full',) if repair_only else ('full','CE','EMA')
    expected={(m,s,w,c) for m in modes for s in SETTINGS for w in WEIGHTS for c in CAPS}
    if not repair_only:expected.update(('dual',s,w,None) for s in SETTINGS for w in WEIGHTS)
    if repair_root is not None:
        expected.update(('repair',s,w,.25) for s in SETTINGS for w in WEIGHTS)
    if set(cells)!=expected:raise ValueError('Complete matched reference matrices required')
    truths={}
    for setting in SETTINGS:
        protocols={v['protocol_id'] for k,v in cells.items() if k[1]==setting}
        if len(protocols)!=1:raise ValueError('One target physical-ID contract per setting required')
        protocol=protocols.pop();truths[setting]={task:read(context['handles'][protocol]['tests'][task]['truth'])
                                                   for task in ('cross_rx','cross_both')}
    predictions={k:{task:prediction(v['entry']['predictions'],task) for task in truths[k[1]]} for k,v in cells.items()}
    comparisons=[]
    plans=comparison_plan(include_repair=repair_root is not None)
    if repair_only:plans=[p for p in plans if p['comparison']=='repair_vs_full']
    for plan in plans:
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
        interpretation='All fixed comparisons on previously exposed targets and selected seed; CE and EMA reference extra steps include supervised differences and U. No feedback to training or selection.')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('dual-root','full-root','ce-root','ema-root','repair-root'):p.add_argument('--'+name)
    p.add_argument('--repair-only',action='store_true')
    a=p.parse_args();print(json.dumps(analyze(a.dual_root,a.full_root,a.ce_root,a.ema_root,a.repair_root,a.repair_only),ensure_ascii=False,allow_nan=False))
