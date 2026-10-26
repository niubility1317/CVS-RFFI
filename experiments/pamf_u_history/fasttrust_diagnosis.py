"""Read saved source-V outputs only; assess information and FastTrust-style routes.

This is a diagnostic, not a calibrated labeler or a training/selection routine.
The fixed .95 candidate sets below carry no coverage guarantee on new receivers.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from .analyze import dump_csv


def probability(logits):
    x=np.asarray(logits,dtype=np.float64)
    if x.ndim!=2 or x.shape[1]!=6 or not np.isfinite(x).all():
        raise ValueError('Finite six-class source logits required')
    x=x-x.max(1,keepdims=True);p=np.exp(x)
    return p/p.sum(1,keepdims=True)


def mass_set(p, mass=.95):
    if not 0<mass<1 or p.ndim!=2 or not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(1),1):
        raise ValueError('Normalized finite probabilities and a fixed mass required')
    order=np.argsort(-p,axis=1,kind='stable')
    sorted_p=np.take_along_axis(p,order,axis=1)
    # Include the first class crossing the mass, including exact-mass ties once.
    size=1+(np.cumsum(sorted_p,axis=1)[:,:-1]<mass).sum(1)
    selected=np.arange(p.shape[1])[None,:]<size[:,None]
    result=np.zeros_like(p,dtype=bool)
    np.put_along_axis(result,order,selected,axis=1)
    return result


def quality(mask, prediction, truth):
    n=int(mask.sum());correct=int((mask&(prediction==truth)).sum())
    return dict(n=n,correct=correct,wrong=n-correct,coverage_percent=100*n/len(truth),
                precision_percent=100*correct/n if n else None)


def set_quality(selected, mask, truth):
    n=int(mask.sum());hit=selected[np.arange(len(truth)),truth]
    good=int((mask&hit).sum())
    return dict(n=n,truth_in_set=good,false_exclusion=n-good,
                coverage_percent=100*n/len(truth),
                inclusion_percent=100*good/n if n else None,
                mean_size=float(selected[mask].sum(1).mean()) if n else None)


def view_gradient(p, pseudo, truth, mask):
    target=np.eye(6)[pseudo];true=np.eye(6)[truth]
    hard=p-target;oracle=p-true;smooth=p-(.99*target+.01/6)
    dot=(smooth*oracle).sum(1)
    norm=np.linalg.norm(smooth,axis=1)*np.linalg.norm(oracle,axis=1)
    cosine=np.divide(dot,norm,out=np.zeros_like(dot),where=norm>0)
    correct=mask&(pseudo==truth)
    return dict(n=int(mask.sum()),correct_pseudo=int(correct.sum()),
        hard_equals_oracle_on_correct=bool(np.array_equal(hard[correct],oracle[correct])),
        hard_gradient_norm_mean=float(np.linalg.norm(hard[mask],axis=1).mean()) if mask.any() else None,
        LS01_vs_oracle_conflicts=int((correct&(dot<0)).sum()),
        LS01_vs_oracle_cosine_correct_mean=float(cosine[correct].mean()) if correct.any() else None,
        interpretation='logit gradients only; LS opposition is not proof of harmful parameter updates')


def pair_purity(pseudo, accepted, rx, truth):
    def cross_receiver_pairs(mask):
        n=int(mask.sum());return (n*(n-1)-sum(int((mask&(rx==r)).sum())*(int((mask&(rx==r)).sum())-1) for r in np.unique(rx)))//2
    pairs=good=0
    for label in range(6):
        m=accepted&(pseudo==label);pairs+=cross_receiver_pairs(m)
        good+=sum(cross_receiver_pairs(m&(truth==y)) for y in range(6))
    return dict(positive_pairs=pairs,true_same_TX_pairs=good,false_positive_pairs=pairs-good,
                false_positive_percent=100*(pairs-good)/pairs if pairs else None,
                scope='all saved V cross-RX pairs; not the actual minibatch contrast graph')


def diagnose(path, output):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    with np.load(path,allow_pickle=False) as a:
        truth=a['y'];rx=a['rx'];ids=a['ids'];views=a['views'].tolist()
        if len(truth)!=27000 or len(set(ids.tolist()))!=27000 or set(rx.tolist())!={1,3,4,6,8} or set(truth.tolist())!=set(range(6)):
            raise ValueError('Complete original six-class source V27000 and five RX required')
        indices=np.flatnonzero(a['methods']=='PLControl')
        if a['seeds'][indices].tolist()!=[2026092701,2026092702,2026092703]:
            raise ValueError('The three completed PL controls are diagnostic inputs, never retrained')
        rows=[];gradients=[];sets=[];strata=[];pairs=[]
        for j in indices:
            seed=int(a['seeds'][j]);raw=probability(a['logits'][j,views.index('raw')]);pseudo=raw.argmax(1)
            accepted=raw.max(1)>=.95;wrong=pseudo!=truth
            rows.append(dict(seed=seed,scope='raw',**quality(accepted,pseudo,truth),
                all_errors=int(wrong.sum()),rejected_errors=int((~accepted&wrong).sum()),
                fraction_errors_rejected_percent=100*int((~accepted&wrong).sum())/int(wrong.sum())))
            for dim,values in (('TX',truth),('RX',rx),('day',a['day'])):
                for v in np.unique(values):
                    m=values==v
                    strata.append(dict(seed=seed,dimension=dim,value=int(v),**quality(accepted[m],pseudo[m],truth[m])))
            candidates=mass_set(raw);size=candidates.sum(1)
            for scope,mask in (('all',np.ones(len(truth),bool)),('raw_rejected',~accepted),('raw_rejected_size2or3',~accepted&(size>=2)&(size<=3))):
                sets.append(dict(seed=seed,scope=scope,rule='raw smallest cumulative probability mass >= .95; no fitted calibration',
                                 **set_quality(candidates,mask,truth)))
            pairs.append(dict(seed=seed,**pair_purity(pseudo,accepted,rx,truth)))
            for k,view in enumerate(views):
                p=probability(a['logits'][j,k]);pred=p.argmax(1);conf=p.max(1)
                for scope,mask in (('ordinary_PL_accepted',accepted),('additional_hard_trust',accepted&(pred==pseudo)&(conf>=.95))):
                    n=int(mask.sum());good=mask&(pseudo==truth)
                    rows.append(dict(seed=seed,view=view,scope=scope,**quality(mask,pseudo,truth),
                        view_correct=int((mask&(pred==truth)).sum()),
                        view_accuracy_percent=100*int((mask&(pred==truth)).sum())/n if n else None,
                        raw_correct_view_wrong=int((good&(pred!=truth)).sum()),
                        pseudo_CE=float((-np.log(p[np.arange(len(truth)),pseudo].clip(1e-300)))[mask].mean()) if n else None))
                    gradients.append(dict(seed=seed,view=view,scope=scope,**view_gradient(p,pseudo,truth,mask)))
    result=dict(status='ANALYZED_SAVED_SOURCE_ONLY',source_records=27000,source_method='completed PLControl SIG_TIME_X3 E50',
        seeds=[2026092701,2026092702,2026092703],views=views,source=str(path),
        new_model_forwards=0,new_IQ_reads=0,optimizer_updates=0,U_truth_reads=0,new_target_reads=0,
        ordinary_PL_changed=False,calibration_fitted=False,quality=rows,partial_sets=sets,pair_purity=pairs,
        limitations=['X3 completed controls are evidence proxies, not the current SIG_TIME single-view model',
                     'V truth used only for diagnostic arithmetic; no V-trained calibrator or classifier',
                     'fixed-mass sets have no empirical or theoretical safety guarantee on target',
                     'gradient diagnostics are at logits; no causal attribution of accuracy changes'])
    (out/'source_diagnosis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    dump_csv(out/'source_view_routes.csv',rows);dump_csv(out/'source_gradient_diagnosis.csv',gradients)
    dump_csv(out/'source_PL_strata.csv',strata);dump_csv(out/'source_partial_sets.csv',sets)
    print(json.dumps(dict(status=result['status'],raw=[r for r in rows if r.get('scope')=='raw'],partial_sets=sets,pairs=pairs)))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();diagnose(args.source,args.output)
