"""CPU diagnostics on already scored, frozen predictions. Never changes decisions."""
import argparse
import json
import os
from pathlib import Path
import numpy as np

VIEWS=('clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban')
F_RUN='20261016-phase1-F-supcon-e50-clean-manysig-m9-r02'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def decisions(*arrays):
    values=[np.asarray(a) for a in arrays]
    if (not values or any(a.ndim!=1 or a.shape!=values[0].shape or not np.issubdtype(a.dtype,np.integer)
            or np.any((a<0)|(a>=6)) for a in values)):
        raise ValueError('Aligned one-dimensional six-class decisions required')
    return values


def percent(n,d):
    return float(100*n/d) if d else None


def paired_readouts(truth,clean,before,after):
    y,c,b,r=decisions(truth,clean,before,after);n=len(y)
    count=lambda mask:int(np.count_nonzero(mask))
    good=c==y;bad=~good;repair=(b!=y)&(r==y);damage=(b==y)&(r!=y)
    ng=count(good);nb=count(bad)
    return dict(count=n,baseline_accuracy_percent=percent(count(b==y),n),accuracy_percent=percent(count(r==y),n),
        original_clean_accuracy_percent=percent(ng,n),baseline_clean_agreement_percent=percent(count(b==c),n),
        restored_clean_agreement_percent=percent(count(r==c),n),repaired=count(repair),damaged=count(damage),
        clean_correct_count=ng,clean_correct_baseline_correct=count(good&(b==y)),clean_correct_restored_correct=count(good&(r==y)),
        clean_correct_baseline_accuracy_percent=percent(count(good&(b==y)),ng),clean_correct_restored_accuracy_percent=percent(count(good&(r==y)),ng),
        clean_wrong_count=nb,clean_wrong_baseline_correct=count(bad&(b==y)),clean_wrong_restored_correct=count(bad&(r==y)),
        clean_wrong_restored_accuracy_percent=percent(count(bad&(r==y)),nb),clean_wrong_inherited_error=count(bad&(r==c)),
        repaired_clean_correct=count(repair&good),repaired_clean_wrong=count(repair&bad),damaged_to_clean_wrong_decision=count(damage&bad&(r==c)))


def correction_components(clean,received,restored):
    x,r,y=[np.asarray(a,dtype=np.float64) for a in (clean,received,restored)]
    if x.ndim!=3 or x.shape[1]!=2 or r.shape!=x.shape or y.shape!=x.shape or not all(np.isfinite(a).all() for a in (x,r,y)):
        raise ValueError('Finite aligned IQ arrays required')
    error=x-r;delta=y-r;power=np.maximum((x*x).mean((1,2)),1e-4)
    e2=(error*error).mean((1,2));d2=(delta*delta).mean((1,2));inner=(delta*error).mean((1,2))
    defined=e2>1e-12;direction=defined&(d2>1e-12)
    alpha=np.divide(inner,e2,out=np.zeros_like(inner),where=defined)
    projected=inner*alpha;orthogonal=np.maximum(d2-projected,0.)
    cosine=np.divide(inner,np.sqrt(e2*d2),out=np.zeros_like(inner),where=direction)
    after=((y-x)**2).mean((1,2))/power
    return dict(input_nmse=e2/power,restored_nmse=after,projection_fraction=alpha,projection_defined=defined,
        cosine=np.clip(cosine,-1,1),cosine_defined=direction,orthogonal_correction_nmse=orthogonal/power,
        introduced_error_nmse=np.where(~defined,after,0.))


def robust_views(truth,before,after):
    y=np.asarray(truth);b=np.asarray(before);r=np.asarray(after)
    if b.shape!=(6,len(y)) or r.shape!=b.shape:raise ValueError('Six aligned impaired views required')
    for raw,new in zip(b,r):decisions(y,raw,new)
    nb=(b==y).sum(0);nr=(r==y).sum(0)
    return dict(count=len(y),baseline_all_six_correct=int((nb==6).sum()),restored_all_six_correct=int((nr==6).sum()),
        baseline_number_correct_histogram=np.bincount(nb,minlength=7).tolist(),restored_number_correct_histogram=np.bincount(nr,minlength=7).tolist(),
        decisions_changed=False,scope='Descriptive same-physical-sample counts across correlated views;not a runtime ensemble or extra independent samples')


def waveform_summary(values,mask):
    good=mask&values['projection_defined'];direction=mask&values['cosine_defined'];zero=mask&~values['projection_defined']
    mean=lambda key,select:float(values[key][select].mean()) if np.any(select) else None
    return dict(input_nmse=mean('input_nmse',mask),restored_nmse=mean('restored_nmse',mask),
        projection_defined_count=int(good.sum()),mean_projection_fraction=mean('projection_fraction',good),
        mean_correction_error_cosine=mean('cosine',direction),cosine_defined_count=int(direction.sum()),
        orthogonal_correction_nmse=mean('orthogonal_correction_nmse',good),zero_input_error_count=int(zero.sum()),
        introduced_error_nmse=mean('introduced_error_nmse',zero))


def analyze(args,*,truth_loader=read_json):
    if os.environ.get('CUDA_VISIBLE_DEVICES','')!='':raise ValueError('Independent CPU analysis required')
    summary=read_json(args.row/'summary.json');done=read_json(args.row/'prediction/complete.json')
    if (summary.get('status')!='ANALYZED' or summary.get('role')!='target' or summary.get('count')!=168000
            or not summary.get('all_predictions_before_truth') or not summary.get('original_default_head')
            or summary.get('F_run')!=F_RUN or summary.get('F_seed')!=2026092701
            or done.get('status')!='PREDICTIONS_COMPLETE' or done.get('views')!=list(VIEWS)):
        raise ValueError('Already scored full same-F target row required before any truth diagnostics')
    baseline=read_json(args.baseline/'complete.json');reference=read_json(args.reference/'manifest.json')
    if (baseline.get('status')!='PREDICTIONS_COMPLETE' or baseline.get('views')!=list(VIEWS) or baseline.get('count')!=168000
            or baseline.get('F_run')!=F_RUN or baseline.get('F_seed')!=2026092701 or not baseline.get('original_default_head')
            or reference.get('status')!='FORMAL_REFERENCE_READY' or reference.get('count')!=168000 or reference.get('views')!=list(VIEWS)):
        raise ValueError('Original completed same-F baseline and formal reference required')
    ids=np.load(args.row/'prediction/physical_ids.npy',allow_pickle=False)
    if ids.shape!=(168000,) or not all(np.array_equal(ids,np.load(p/'physical_ids.npy',allow_pickle=False)) for p in (args.baseline,args.reference)):
        raise ValueError('Paired physical IDs differ')
    def logits(path):
        a=np.load(path,mmap_mode='r',allow_pickle=False)
        if a.shape!=(len(ids),6) or not np.isfinite(a).all():raise ValueError('Complete six-class logits required')
        return a.argmax(1)
    before={v:logits(args.baseline/(v+'.identity.logits.npy')) for v in VIEWS}
    after={v:logits(args.row/'prediction'/(v+'.restored.logits.npy')) for v in VIEWS}
    truth=truth_loader(args.truth)  # All predictions fixed and the original independent scorer already finished.
    labels=np.asarray([truth[str(pid)]['label'] for pid in ids],dtype=np.int64)
    rx=np.asarray([str(truth[str(pid)]['receiver']) for pid in ids]);day=np.asarray([str(truth[str(pid)].get('day','unavailable')) for pid in ids])
    groups=[('overall','all',np.ones(len(ids),bool))]
    for dimension,values in (('TX',labels),('RX',rx),('day',day)):
        groups += [(dimension,str(value),values==value) for value in np.unique(values)]
    clean=np.load(args.reference/'clean.npy',mmap_mode='r',allow_pickle=False);records=[]
    for view in VIEWS:
        received=np.load(args.reference/(view+'.npy'),mmap_mode='r',allow_pickle=False)
        restored=np.load(args.row/'prediction'/(view+'.restored.iq.npy'),mmap_mode='r',allow_pickle=False)
        parts={}
        for start in range(0,len(ids),2048):
            values=correction_components(clean[start:start+2048],received[start:start+2048],restored[start:start+2048])
            for key,value in values.items():parts.setdefault(key,[]).append(value)
        values={key:np.concatenate(value) for key,value in parts.items()}
        for dim,value,mask in groups:
            paired=paired_readouts(labels[mask],before['clean'][mask],before[view][mask],after[view][mask])
            records.append(dict(view=view,dimension=dim,value=value,**paired,correction=waveform_summary(values,mask)))
        result=records[-len(groups)];original=next(r for r in summary['records'] if r['view']==view and r['dimension']=='overall' and r['readout']=='restored')
        if abs(result['accuracy_percent']-original['accuracy'])>1e-9 or result['repaired']!=original['repaired'] or result['damaged']!=original['damaged']:
            raise ValueError('Paired diagnostic differs from original independent score')
    return dict(schema='paired_identity_v1',status='ANALYZED',count=len(ids),views=list(VIEWS),F_run=F_RUN,F_seed=2026092701,
        records=records,all_six_robustness=robust_views(labels,np.stack([before[v] for v in VIEWS[1:]]),np.stack([after[v] for v in VIEWS[1:]])),
        runtime_decisions_changed=False,parameters_fitted=0,extra_model_predictions=0,truth_after_completed_score=True,
        correction_definition='e=clean-received,delta=restored-received;alpha=<delta,e>/<e,e>,unclipped;orthogonal=||delta||²-<delta,e>²/||e||²;perpacketcleanpowernormalization',
        claim_scope='Paired waveform/readout evidence only;full-IQ correction can be dominated by channel correction;matching a mistaken clean F decision is not TX correctness;not per-detail authenticity proof;not a new blind benchmark')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('row','baseline','reference','truth','output'):parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args();value=analyze(args)
    with args.output.open('x',encoding='utf-8') as stream:json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(dict(status=value['status'],count=value['count'],classification_groups=len(value['records']),extra_model_predictions=0)))


if __name__=='__main__':main()
