"""Post-hoc probability arithmetic for the already frozen E30 IQ readouts."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
from .enhancement_pair_score import RUN,MODEL_COMMIT,VIEWS


def probabilities(logits):
    values=np.asarray(logits,dtype=np.float64)
    shifted=values-values.max(1,keepdims=True)
    logp=shifted-np.log(np.exp(shifted).sum(1,keepdims=True))
    return logp,np.exp(logp)


def statistics(logits,labels):
    logp,p=probabilities(logits);index=np.arange(len(labels))
    predicted=p.argmax(1);confidence=p.max(1);right=predicted==labels
    other=np.asarray(logits,dtype=np.float64).copy();other[index,labels]=-np.inf
    margin=np.asarray(logits,dtype=np.float64)[index,labels]-other.max(1)
    brier=(p*p).sum(1)-2*p[index,labels]+1
    bins=[]
    for i in range(10):
        mask=(confidence>=i/10)&(confidence<(i+1)/10 if i<9 else confidence<=1.)
        bins.append(dict(lower=i/10,upper=(i+1)/10,count=int(mask.sum()),
            confidence=float(confidence[mask].mean()) if mask.any() else None,
            accuracy=float(right[mask].mean()) if mask.any() else None))
    ece=sum(b['count']/len(labels)*abs(b['confidence']-b['accuracy']) for b in bins if b['count'])
    return dict(count=len(labels),correct=int(right.sum()),accuracy=float(right.mean()),
        mean_max_probability=float(confidence.mean()),mean_true_probability=float(p[index,labels].mean()),
        NLL_nats=float(-logp[index,labels].mean()),Brier_sum6=float(brier.mean()),
        mean_true_margin=float(margin.mean()),ECE_10_equal_width=float(ece),calibration_bins=bins)


def pair_statistics(first,second,labels):
    _,p=probabilities(first);_,q=probabilities(second);index=np.arange(len(labels))
    a=p.argmax(1)==labels;b=q.argmax(1)==labels;confidence_change=q.max(1)-p.max(1)
    result={}
    for name,mask in dict(repaired=~a&b,damaged=a&~b,both_correct=a&b,both_wrong=~a&~b).items():
        result[name]=dict(count=int(mask.sum()),
            mean_max_probability_change=float(confidence_change[mask].mean()) if mask.any() else None,
            mean_true_probability_change=float((q[index,labels]-p[index,labels])[mask].mean()) if mask.any() else None)
    result['more_confident_and_wrong_after']=int(((confidence_change>0)&~b).sum())
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--analysis-commit',required=True);args=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU truth-side analysis required')
    root=args.root;run=root/'runs'/RUN
    read=lambda p:json.loads(p.read_text())
    if read(run/'complete.json')['status']!='ANALYZED':raise ValueError('Completed official row required')
    formal=root/'runs/20261019-phase1-restoration-formal-frozen-residual-s2026092701-r05/reference'
    ids=np.load(formal/'physical_ids.npy',allow_pickle=False);arrays={}
    if ids.shape!=(168000,):raise ValueError('Original population required')
    for arm in ('physical','scratch_joint'):
        folder=run/arm/'target_prediction';done=read(folder/'complete.json')
        expected=dict(status='PREDICTIONS_COMPLETE',count=168000,views=list(VIEWS),role='target',arm=arm,
            code_commit=MODEL_COMMIT,F_run='20261016-phase1-F-supcon-e50-clean-manysig-m9-r02',F_seed=2026092701,
            original_default_head=True,TX_truth_reads=0,state_updates=0,head_fit_updates=0)
        if any(done.get(k)!=v for k,v in expected.items()):raise ValueError('Both complete registered readouts required')
        if not np.array_equal(np.load(folder/'physical_ids.npy',allow_pickle=False),ids):raise ValueError('Physical ID mismatch')
        arrays[arm]={}
        for view in VIEWS:
            values=np.load(folder/(view+'.logits.npy'),mmap_mode='r',allow_pickle=False)
            if values.shape!=(168000,6) or values.dtype!=np.float32 or not np.isfinite(values).all():
                raise ValueError('Complete finite FP32 logits required')
            arrays[arm][view]=values
    # All fourteen prediction arrays and their identities are fixed before truth.
    truth=read(root/'runs/20260927-phase1-baselines-final-clean-satellite-m5-r01/data/truth.json')
    labels=np.asarray([truth[str(pid)]['label'] for pid in ids],dtype=np.int64)
    overall={}
    summaries={arm:read(run/arm/'target_summary.json') for arm in arrays}
    for view in VIEWS:
        row={arm:statistics(arrays[arm][view],labels) for arm in arrays}
        for arm,current in row.items():
            if current['correct']!=summaries[arm]['overall'][view]['correct']:raise ValueError('Official correct count disagrees')
        row['paired']=pair_statistics(arrays['physical'][view],arrays['scratch_joint'][view],labels)
        overall[view]=row
    print(json.dumps(dict(status='ANALYZED_POSTHOC',run=RUN,model_commit=MODEL_COMMIT,analysis_commit=args.analysis_commit,
        count_per_view=168000,all_fourteen_predictions_before_truth=True,new_model_forwards=0,updates=0,
        target_feedback=False,bins='10 fixed equal-width maximum-probability bins;ECE fractions',
        overall=overall)))


if __name__=='__main__':main()
