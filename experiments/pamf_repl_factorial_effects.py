"""Fixed factorial contrasts and paired-ID attribution after complete scoring."""
import argparse
from collections import Counter
import gzip
import json
import math
from pathlib import Path

WEIGHTS = (3.,6.)
CELLS = {(w,a,b) for w in WEIGHTS for a in (0,1) for b in (0,1)}
GROUPS = ('t2','cls_head','other')


def contrast_plan():
    plans=[]
    for weight in WEIGHTS:
        for b in (0,1):
            plans.append(dict(factor='A',condition=dict(weight=weight,B=b),baseline=(weight,0,b),candidate=(weight,1,b)))
        for a in (0,1):
            plans.append(dict(factor='B',condition=dict(weight=weight,A=a),baseline=(weight,a,0),candidate=(weight,a,1)))
    for a in (0,1):
        for b in (0,1):
            plans.append(dict(factor='weight_6_vs_3',condition=dict(A=a,B=b),baseline=(3.,a,b),candidate=(6.,a,b)))
    return plans


def summarize_effects(cells):
    if set(cells) != CELLS:
        raise ValueError('Exact full factorial cells required')
    by_weight={}
    for w in WEIGHTS:
        a0=cells[w,1,0]-cells[w,0,0];a1=cells[w,1,1]-cells[w,0,1]
        b0=cells[w,0,1]-cells[w,0,0];b1=cells[w,1,1]-cells[w,1,0]
        by_weight[str(w)]={'A_at_B0':a0,'A_at_B1':a1,'A_average':(a0+a1)/2,
                           'B_at_A0':b0,'B_at_A1':b1,'B_average':(b0+b1)/2,
                           'AB_interaction':a1-a0}
    return dict(by_weight=by_weight,weight_effects={f'A{a}_B{b}':cells[6.,a,b]-cells[3.,a,b]
                                                    for a in (0,1) for b in (0,1)})


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def source_metrics(source):
    source=Path(source); done=read(source/'complete.json')
    if (done['status']!='SOURCE_FROZEN' or done['epoch']!=50 or done['updates']!=11100
            or done['target_access'] or done['U_truth_read']):
        raise ValueError('Only complete source-only E50 models are comparable')
    total=Counter(); last_v=None
    for end,steps in ((30,2220),(50,4440)):
        folder=source/f'segment{end}'
        epochs=[json.loads(x) for x in (folder/'metrics_epoch.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
        if len(epochs)*222!=steps:raise ValueError('Every source validation epoch required')
        last_v=epochs[-1]['val_tx_acc']; n=0
        with gzip.open(folder/'step_metrics.jsonl.gz','rt',encoding='utf-8') as stream:
            for n,line in enumerate(stream,1):
                row=json.loads(line); m=row['metrics']
                if row['step']!=n:raise ValueError('Ordered complete source steps required')
                keys=['train/loss_tx_labeled']+[f'HPL_GP_{g}_{k}' for g in GROUPS
                    for k in ('L_norm','U_norm','applied_U_norm','conflict','capped')]
                for key in keys:
                    value=float(m[key])
                    if not math.isfinite(value):raise ValueError('Finite source telemetry required')
                    total[key]+=value
        if n!=steps:raise ValueError('Incomplete source telemetry')
    return dict(L_CE=total['train/loss_tx_labeled']/6660,source_val_E50=last_v,
        groups={g:{k:total[f'HPL_GP_{g}_{k}']/6660 for k in
                   ('L_norm','U_norm','applied_U_norm','conflict','capped')} for g in GROUPS})


def coordinates(recipe,matrix):
    if matrix=='shallow_extrap':
        return float(recipe['weight']),int(recipe['shallow_factor']>0),int(recipe['extrap_coefficient']>0)
    if matrix=='joint_kl':
        return float(recipe['weight']),int(recipe['joint_weight']>0),int(recipe['kl_factor']>0)
    raise ValueError('Unknown preregistered factorial matrix')


def analyze(root,matrix):
    import numpy as np
    from experiments.pamf_repl_weight.analyze_paired import prediction, compare
    root=Path(root);state=read(root/'pipeline_state.json')
    if state['status']!='ANALYZED':raise ValueError('All registered models, predictions and scores must finish')
    sources=read(root/'sources.json')
    runtime=read(root.parents[1]/'releases'/(root.name+'-release')/'runtime.json')
    context=read(runtime['context']); settings={}
    for rid,entry in sources.items():
        source=Path(entry['source_output']);done=read(source/'complete.json');cfg=done['config']
        setting=next((x for x in ('ten_class','source_rx_6') if rid.startswith(x+'-')),None)
        if setting is None or cfg['row_id']!=rid or cfg['model_seed']!=2026092701:
            raise ValueError('Exact preregistered setting, row and seed required')
        key=coordinates(cfg['tuning_recipe'],matrix)
        bucket=settings.setdefault(setting,{})
        if key in bucket:raise ValueError('Duplicate factorial cell')
        scored=read(Path(entry['scores'])/'scores.json')
        if read(Path(entry['scores'])/'complete.json')['status']!='SCORED_COMPLETE' or not scored['truth_last']:
            raise ValueError('Independent truth-last clean score required')
        records=[x for x in scored['results'] if x['test'] in ('cross_rx','cross_both')
                 and x['readout']=='raw' and x['view']=='clean' and x['dimension']=='overall']
        if len(records)!=2 or {x['test'] for x in records}!={'cross_rx','cross_both'}:
            raise ValueError('Both preregistered main clean tests required')
        count=sum(x['count'] for x in records)
        correct=sum(sum(row[i] for i,row in enumerate(x['confusion'])) for x in records)
        uroot=root/'readouts'/rid/'U_diagnostic';u=read(uroot/'scores.json');udone=read(uroot/'complete.json')
        if (udone['status']!='SCORED_U_DIAGNOSTIC' or not udone['truth_last']
                or udone['source_epoch']!=50 or u['training_or_selection_use'] is not False):
            raise ValueError('Independent post-freeze U score required')
        o=u['overall']
        bucket[key]=dict(row_id=rid,entry=entry,protocol_id=cfg['protocol_id'],source=source_metrics(source),
            clean=dict(count=count,correct=correct,accuracy=correct/count),
            U=dict(seen=o['seen'],accepted=o['accepted'],accepted_correct=o['accepted_correct'],
                   acceptance=o['accepted']/o['seen'],accuracy=o['accepted_correct']/o['accepted'] if o['accepted'] else None,
                   unique_total=u['unique_U_total'],unique_seen=u['unique_seen'],unique_accepted=u['unique_accepted']))
    if set(settings)!={'ten_class','source_rx_6'} or any(set(v)!=CELLS for v in settings.values()):
        raise ValueError('Complete two-setting 2x2x2 matrix required')
    results={}
    for setting,cells in settings.items():
        protocols={v['protocol_id'] for v in cells.values()}
        if len(protocols)!=1:raise ValueError('One exact source/test contract per setting required')
        protocol=protocols.pop(); truths={task:read(context['handles'][protocol]['tests'][task]['truth'])
                                         for task in ('cross_rx','cross_both')}
        predictions={k:{task:prediction(v['entry']['predictions'],task) for task in truths} for k,v in cells.items()}
        paired=[]
        for plan in contrast_plan():
            base,candidate=plan['baseline'],plan['candidate'];per_task={};total=Counter()
            for task in truths:
                ids,pred=predictions[candidate][task];old_ids,old=predictions[base][task]
                if not np.array_equal(ids,old_ids):raise ValueError('Exact paired physical IDs required')
                per_task[task]=compare(ids,pred,old,truths[task]);total.update(per_task[task]['overall'])
            if total['candidate_correct']!=cells[candidate]['clean']['correct'] or total['baseline_correct']!=cells[base]['clean']['correct']:
                raise ValueError('Paired predictions and independent clean score differ')
            paired.append(dict(**plan,combined=dict(total),per_test=per_task,
                               delta_pp=100*total['net']/total['count']))
        summaries={name:summarize_effects({key:100*value['clean']['accuracy'] for key,value in cells.items()}) for name in ('clean_pp',)}
        summaries['source_val_pp']=summarize_effects({key:value['source']['source_val_E50'] for key,value in cells.items()})
        results[setting]=dict(cells=[dict(coordinates=list(key),**{k:v for k,v in value.items() if k!='entry'})
                                     for key,value in sorted(cells.items())],contrasts=paired,effects=summaries)
    return dict(status='ANALYZED_FULL_FACTORIAL_POST_SCORE',matrix=matrix,settings=results,
        interpretation='All fixed conditional contrasts and interactions; target and seed already exposed, '
                       'one seed, no test or U truth feedback to training or selection.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-root',required=True)
    p.add_argument('--matrix',choices=('shallow_extrap','joint_kl'),required=True);a=p.parse_args()
    print(json.dumps(analyze(a.run_root,a.matrix),ensure_ascii=False,allow_nan=False))
