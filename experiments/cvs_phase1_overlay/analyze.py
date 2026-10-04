"""Independent post-prediction recount and paired factorial contrasts; no tuning."""
import argparse
from collections import defaultdict
import math
import numpy as np
from experiments.cvs_phase1_overlay.contract import *
from experiments.cvs_phase1_overlay.dispatch import validate_matrix,validate_freeze


def recount_metrics(y,p):
    cm=np.bincount(6*y+p,minlength=36).reshape(6,6)
    den=cm.sum(0)+cm.sum(1)
    f=np.divide(2*np.diag(cm),den,out=np.zeros(6),where=den>0)
    return cm,float(np.trace(cm)/cm.sum()),float(f.mean())


def contrasts(v):
    return dict(leo_without_mix=v['leo']-v['ce'],leo_with_mix=v['leo_mixstyle']-v['mixstyle'],
        mix_without_leo=v['mixstyle']-v['ce'],mix_with_leo=v['leo_mixstyle']-v['leo'],
        both_vs_ce=v['leo_mixstyle']-v['ce'],interaction=v['leo_mixstyle']-v['leo']-v['mixstyle']+v['ce'])


def analyze(spec):
    validate_matrix(spec);root=Path(spec['runtime_root']);freeze=read(root/'source_matrix_frozen.json');validate_freeze(freeze)
    if read(root/'scoring_p1_complete.json')['status']!='SCORED_COMPLETE':raise ValueError('Independent scoring incomplete')
    results=read(root/'phase1_scored_results.json')['results']
    expected={(r['row_id'],v,rx) for r in spec['rows'] for v in ['clean','satellite']+SCENES for rx in ['ALL','0','2','5','7','9','10','11']}
    # Receiver values in the existing truth are physical receiver names, not necessarily indices.
    truth=read(spec['p1_truth']);ix=np.load(Path(CAPSULE)/'index.npz',allow_pickle=False)
    ids=ix['ids'];y=np.array([truth[s]['label'] for s in ids],dtype=np.int64);rx=np.array([str(truth[s]['receiver']) for s in ids])
    receivers=['ALL']+sorted(set(rx));expected={(r['row_id'],v,rxi) for r in spec['rows'] for v in ['clean','satellite']+SCENES for rxi in receivers}
    actual={(r['row_id'],r['view'],r['receiver']):r for r in results}
    if set(actual)!=expected or len(results)!=len(expected):raise ValueError('Incomplete scored strata')
    pertx=[];verified=0
    for row in spec['rows']:
        pred=np.load(Path(row['output_root'])/'phase1_predictions.npz',allow_pickle=False)
        if not np.array_equal(pred['ids'],ids):raise ValueError('Independent physical ID recount mismatch')
        for view in ['clean','satellite']+SCENES:
            mask=np.ones(len(y),dtype=bool) if view in ('clean','satellite') else ix['scenes']==SCENES.index(view)
            p=pred['clean' if view=='clean' else 'satellite']
            for receiver in receivers:
                use=mask if receiver=='ALL' else mask&(rx==receiver)
                cm,acc,f1=recount_metrics(y[use],p[use]);r=actual[(row['row_id'],view,receiver)]
                if cm.tolist()!=r['confusion'] or int(cm.sum())!=r['query_count'] or abs(acc-r['accuracy'])>1e-12 or abs(f1-r['macro_f1'])>1e-12:raise ValueError('Independent score mismatch')
                verified+=1
                for label,tx in enumerate(CLASSES):
                    n=int(cm[label].sum());pertx.append(dict(row_id=row['row_id'],arm=row['arm'],model_seed=row['model_seed'],view=view,receiver=receiver,transmitter=tx,count=n,accuracy=float(cm[label,label]/n) if n else None))
    write(root/'independent_recount.json',dict(status='VERIFIED',records=verified,rows=16,query_per_row=len(y),physical_ids='EXACT_MATCH',method='numpy.bincount independent confusion/accuracy/F1',target_feedback_forbidden=True))
    write(root/'per_transmitter_results.json',dict(results=pertx))
    effects=[]
    for view in ['clean','satellite']+SCENES:
        for receiver in receivers:
            for metric in ['accuracy','macro_f1']:
                values=defaultdict(list)
                for seed in SEEDS:
                    v={a:100*actual[(a+'-s'+str(seed),view,receiver)][metric] for a in ARMS}
                    for name,delta in contrasts(v).items():values[name].append(delta)
                effects.extend(dict(view=view,receiver=receiver,metric=metric,contrast=k,model_seeds=list(SEEDS),seed_deltas_pp=v,mean_pp=float(np.mean(v)),seed_sd_pp=float(np.std(v,ddof=1))) for k,v in values.items())
    worst=[]
    for a in ARMS:
        for view in ['clean','satellite']+SCENES:
            vals=[min(actual[(a+'-s'+str(s),view,rx)]['accuracy'] for rx in receivers[1:]) for s in SEEDS]
            worst.append(dict(arm=a,view=view,mean=float(np.mean(vals)),seed_sd=float(np.std(vals,ddof=1)),seed_values=vals))
    write(root/'paired_effects.json',dict(unit='percentage_points',results=effects,source_continuation_arm_frozen_before_query=freeze['continuation_arm'],target_selection=False))
    write(root/'worst_receiver_results.json',dict(results=worst))
    lines=['# reference_response 原 Phase1 机制第一轮结果','',
        '状态：全部16行完成固定预测、独立评分及第二实现复算。既有暴露基准上的固定设计结果，不作全新盲测声明。',
        '', '源域预先冻结的后续起点：`'+freeze['continuation_arm']+'`。测试结果不改变该选择。',
        '', '| 机制 | 视图 | Accuracy（%，四seed均值±样本SD） | Macro-F1（%） | 最差RX（%） |','|---|---|---:|---:|---:|']
    for a in ARMS:
        for view in ['clean','satellite']+SCENES:
            av=np.array([actual[(a+'-s'+str(s),view,'ALL')]['accuracy'] for s in SEEDS])*100
            fv=np.array([actual[(a+'-s'+str(s),view,'ALL')]['macro_f1'] for s in SEEDS])*100
            w=next(x for x in worst if x['arm']==a and x['view']==view)
            lines.append(f'| {a} | {view} | {av.mean():.3f} ± {av.std(ddof=1):.3f} | {fv.mean():.3f} ± {fv.std(ddof=1):.3f} | {100*w["mean"]:.3f} ± {100*w["seed_sd"]:.3f} |')
    lines+=['','逐seed、RX、TX及混淆矩阵见同目录JSON/CSV。paired_effects.json保留两种条件下的简单效应和交互项（both−LEO−MixStyle+CE）。所有负结果保留。',
        '', '本轮为Phase1六类闭集测试；Phase2适应、新类注册、K及H均N/A。历史完整CVS训练步数不同，不作等计算量比较。']
    (root/'analysis.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);a=p.parse_args();analyze(read(a.spec))
