"""A separate process joins target truth only after all four predictions are fixed."""
import gzip,json,time
from pathlib import Path
import numpy as np
from . import design as d

def predictions(root,expected_ids):
    done=d.read(root/'complete.json')
    if (done['status']!='PREDICTIONS_COMPLETE' or done['config']!=d.configuration() or done['count']!=d.COUNT
            or done['views']!=list(d.VIEWS) or done['truth_read'] or done['query_fit'] or not done['model_state_unchanged']
            or not done['all_parameters_frozen'] or any(done[k] for k in ('optimizer_updates','EMA_updates','head_fit_updates'))):
        raise ValueError('Truth remains closed: incomplete or changed frozen predictions')
    result={}
    for view in d.VIEWS:
        with np.load(root/(view+'.npz'),allow_pickle=False) as f:
            logits=f['logits']
            if not np.array_equal(f['ids'],expected_ids) or logits.shape!=(d.COUNT,6) or not np.isfinite(logits).all():
                raise ValueError('Truth remains closed: missing/reordered/nonfinite predictions')
            result[view]=logits.copy()
    return result,done

def metric(y,logits):
    p=logits.argmax(1);cm=np.bincount(y*6+p,minlength=36).reshape(6,6)
    denominator=cm.sum(0)+cm.sum(1);f1=np.divide(2*np.diag(cm),denominator,out=np.zeros(6),where=denominator>0).mean()
    return dict(count=len(y),correct=int((p==y).sum()),accuracy=float((p==y).mean()),macro_f1=float(f1),confusion=cm.tolist())

def run():
    # This module has no model, source loader, optimizer or teacher import.
    with np.load(d.FORMAL_VIEWS/'index.npz',allow_pickle=False) as f:all_ids=f['ids'].copy()
    ids=all_ids[d.sample_positions(all_ids)];root=d.BASE/'prediction'
    values,done=predictions(root,ids)
    with gzip.open(root/'channel_metadata.jsonl.gz','rt',encoding='utf-8') as f:metadata=[json.loads(line) for line in f]
    if [v['id'] for v in metadata]!=ids.tolist():raise ValueError('Truth remains closed: metadata identity/order differs')
    out=d.BASE/'scoring'
    if out.exists():raise FileExistsError('Existing scores must be reconciled, never overwritten')
    # The only truth access occurs here, after completion and full prediction checks.
    truth=d.read(d.TRUTH)
    if set(truth)!=set(all_ids.tolist()):raise ValueError('Validated target truth ID coverage differs')
    y=np.asarray([truth[p]['label'] for p in ids]);rx=np.asarray([str(truth[p]['receiver']) for p in ids]);days=np.asarray([str(truth[p]['day']) for p in ids])
    if y.dtype.kind not in 'iu' or set(y)!={0,1,2,3,4,5}:raise ValueError('Fixed target class mapping differs')
    fields=dict(receiver=rx,day=days,state=np.asarray([v['state'] for v in metadata]),lock=np.asarray([v['lock'] for v in metadata]),
        added_noise_bin=np.asarray(['unknown' if v['added_noise_snr_db'] is None else '<3' if v['added_noise_snr_db']<3 else '3-8' if v['added_noise_snr_db']<8 else '8-15' if v['added_noise_snr_db']<15 else '>=15' for v in metadata]))
    rows=[]
    for view,logits in values.items():
        rows.append(dict(view=view,dimension='overall',stratum='ALL',**metric(y,logits)))
        for field,array in fields.items():
            for value in sorted(set(array)):
                mask=array==value;rows.append(dict(view=view,dimension=field,stratum=str(value),**metric(y[mask],logits[mask])))
    pairs=[]
    for treatment,control in ((d.VIEWS[2],d.VIEWS[0]),(d.VIEWS[1],d.VIEWS[0]),(d.VIEWS[2],d.VIEWS[1]),(d.VIEWS[3],d.VIEWS[0])):
        now=values[treatment].argmax(1)==y;old=values[control].argmax(1)==y
        pairs.append(dict(treatment=treatment,control=control,count=d.COUNT,repaired=int((now&~old).sum()),
            broken=int((~now&old).sum()),accuracy_difference=float(now.mean()-old.mean())))
    out.mkdir();result=dict(status='ANALYZED',scope='TARGET_HQ_COUNTERFACTUAL_NOT_FORMAL',config=d.configuration(),
        model_seed=2026092701,model_component='ema_model',count=d.COUNT,results=rows,pairs=pairs,
        truth_last=True,independent_scorer_process=True,target_feedback_forbidden=True,model_updates=0,
        formal_test_IQ_modified=False,cross_seed_confirmation='NOT_RUN',prediction=done)
    d.write(out/'results.json',result);d.write(out/'complete.json',dict(status='SCORED_COMPLETE',views=list(d.VIEWS),count=d.COUNT,truth_last=True))
    d.write(d.BASE/'complete.json',dict(status='ANALYZED',scope=result['scope'],count=d.COUNT,truth_last=True,models_trained=0))
    lines=['# 冻结 EMA：目标域 HQ 配对反事实诊断','',
        '固定4200个匿名ID抽样；同一A0 final50 EMA和其默认头。0更新；完整缓存IQ保持，HQ只降低模拟器新增噪声。不是正式full practical性能改善，也不是训练收益。','',
        '|视图|准确率 %|Macro-F1 %|正确/总数|','|---|---:|---:|---:|']
    for r in rows:
        if r['dimension']=='overall':lines.append(f"|{r['view']}|{100*r['accuracy']:.4f}|{100*r['macro_f1']:.4f}|{r['correct']}/{r['count']}|")
    lines+=['','|配对比较|变化 pp|修复|新增错误|','|---|---:|---:|---:|']
    for r in pairs:lines.append(f"|{r['treatment']} − {r['control']}|{100*r['accuracy_difference']:+.4f}|{r['repaired']}|{r['broken']}|")
    lines+=['','原始完整test有168000物理样本，本小试只覆盖4200；单模型seed和单独立噪声实现，不推断全体或跨seed保证。分层含全部实际RX/day/state/lock/新增噪声区间；分层重叠，新增噪声SNR不是总SNR。目标结果不得选择/调整主线训练、候选或选择性重跑。']
    (d.BASE/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='ANALYZED',count=d.COUNT,overall=[r for r in rows if r['dimension']=='overall'],pairs=pairs)),flush=True)
    return result

if __name__=='__main__':run()
