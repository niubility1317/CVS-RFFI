"""Truth-last independent recount and fixed paired contrasts for all136 rows."""
import argparse
from collections import defaultdict
import numpy as np
from experiments.cvs_phase1_stack.design import *
from experiments.cvs_phase1_overlay.analyze import recount_metrics


def analyze(spec):
    if spec['run_id']!=RUN or spec['runtime_root']!=BASE or spec['p1_truth']!=TRUTH or {r['row_id'] for r in spec['rows']}!={r['row_id'] for r in rows()} or len(spec['rows'])!=136:raise ValueError('Scoring matrix mismatch')
    root=Path(BASE)
    if read(root/'scoring_p1_complete.json')['status']!='SCORED_COMPLETE':raise ValueError('Scoring incomplete')
    data=read(root/'phase1_scored_results.json')['results']
    truth=read(TRUTH)
    with np.load(Path(CAPSULE)/'index.npz',allow_pickle=False) as ix:ids=ix['ids'].copy();scenes=ix['scenes'].copy()
    y=np.asarray([truth[s]['label'] for s in ids],dtype=np.int64);rx=np.asarray([str(truth[s]['receiver']) for s in ids])
    receivers=['ALL']+sorted(set(rx));views=['clean','satellite']+SCENES
    index={(r['row_id'],r['view'],r['receiver']):r for r in data}
    expected={(r['row_id'],v,x) for r in rows() for v in views for x in receivers}
    if set(index)!=expected or len(data)!=len(expected):raise ValueError('Missing/duplicate scoring strata')
    bytx=[]
    for row in rows():
        with np.load(root/row['row_id']/'prediction/phase1_predictions.npz',allow_pickle=False) as pred:
            if not np.array_equal(ids,pred['ids']):raise ValueError('Physical ID mismatch')
            for view in views:
                mask=np.ones(len(ids),dtype=bool) if view in ['clean','satellite'] else scenes==SCENES.index(view)
                p=pred['clean' if view=='clean' else 'satellite']
                for receiver in receivers:
                    use=mask if receiver=='ALL' else mask&(rx==receiver)
                    cm,acc,f1=recount_metrics(y[use],p[use]);r=index[(row['row_id'],view,receiver)]
                    if cm.tolist()!=r['confusion'] or int(cm.sum())!=r['query_count'] or abs(acc-r['accuracy'])>1e-12 or abs(f1-r['macro_f1'])>1e-12:raise ValueError('Independent recount mismatch')
                    for label,tx in enumerate(CLASSES):
                        n=int(cm[label].sum());bytx.append(dict(**row,view=view,receiver=receiver,transmitter=tx,count=n,accuracy=float(cm[label,label]/n) if n else None))
    write(root/'independent_recount.json',dict(status='VERIFIED',rows=136,records=len(index),physical_ids='EXACT_MATCH',implementation='bincount',target_feedback_forbidden=True))
    write(root/'per_transmitter_results.json',dict(results=bytx))
    paired=[];summary=[]
    for stage,arms in STAGES.items():
        control='bridge' if stage=='r2' else ('full' if stage=='r6' else 'base')
        for arm in arms:
            for view in views:
                av=[index[(stage+'-'+arm+'-s'+str(s),view,'ALL')]['accuracy'] for s in SEEDS]
                fv=[index[(stage+'-'+arm+'-s'+str(s),view,'ALL')]['macro_f1'] for s in SEEDS]
                worst=[min(index[(stage+'-'+arm+'-s'+str(s),view,r)]['accuracy'] for r in receivers[1:]) for s in SEEDS]
                summary.append(dict(stage=stage,arm=arm,view=view,accuracy_mean=float(np.mean(av)),accuracy_sd=float(np.std(av,ddof=1)),macro_f1_mean=float(np.mean(fv)),macro_f1_sd=float(np.std(fv,ddof=1)),worst_rx_mean=float(np.mean(worst)),worst_rx_sd=float(np.std(worst,ddof=1))))
                for receiver in receivers:
                    for metric in ['accuracy','macro_f1']:
                        deltas=[100*(index[(stage+'-'+arm+'-s'+str(s),view,receiver)][metric]-index[(stage+'-'+control+'-s'+str(s),view,receiver)][metric]) for s in SEEDS]
                        paired.append(dict(stage=stage,arm=arm,control=control,view=view,receiver=receiver,metric=metric,seed_deltas_pp=deltas,mean_pp=float(np.mean(deltas)),seed_sd_pp=float(np.std(deltas,ddof=1))))
    # R5 DAOT×RC4 interaction from the same four fixed seeds.
    interactions=[]
    for view in views:
        d=[100*(index[('r5-both-s'+str(s),view,'ALL')]['accuracy']-index[('r5-daot-s'+str(s),view,'ALL')]['accuracy']-index[('r5-rc4-s'+str(s),view,'ALL')]['accuracy']+index[('r5-base-s'+str(s),view,'ALL')]['accuracy']) for s in SEEDS]
        interactions.append(dict(view=view,seed_deltas_pp=d,mean_pp=float(np.mean(d)),seed_sd_pp=float(np.std(d,ddof=1))))
    write(root/'paired_results.json',dict(results=paired,r5_interactions=interactions,target_selection=False))
    write(root/'compact_summary.json',dict(results=summary))
    lines=['# reference_response剩余Phase1机制实验结果','','全部136行预测、独立评分和第二实现复算完成。全部源域选择发生在本批query之前。','',
        '| 轮次 | 实验组 | 视图 | Accuracy（%±seed SD） | Macro-F1（%±seed SD） | 最差RX（%） |','|---|---|---|---:|---:|---:|']
    for r in summary:lines.append(f'| {r["stage"]} | {r["arm"]} | {r["view"]} | {100*r["accuracy_mean"]:.3f} ± {100*r["accuracy_sd"]:.3f} | {100*r["macro_f1_mean"]:.3f} ± {100*r["macro_f1_sd"]:.3f} | {100*r["worst_rx_mean"]:.3f} |')
    lines+=['','固定配对差值见paired_results.json；逐TX/RX与混淆矩阵保留。全部负结果保留。',
        '', 'R5共同MUSE桥接、R6移除伪标签时的MUSE/RC4依赖及有标签DG/concat限定见预登记。架构属既有暴露基准上的固定设计，不声明全新盲测。Phase2适应/注册、K、新类及H均N/A。']
    (root/'analysis.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);a=p.parse_args();analyze(read(a.spec))
