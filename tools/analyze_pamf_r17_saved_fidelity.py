"""Same-row source analysis of frozen R17/R02 predictions; no model forwards."""
import json
from pathlib import Path
import numpy as np

TEACHER='20261020-phase1-restoration-unrolled-selfsup-s2026092701-m2-r17'
COMPACT='20261020-phase1-restoration-r17-fidelity-source-s2026092701-m2-r02'
ACTION='20261019-phase1-restoration-learned-action-source-s2026092701-r07'
READER='20261016-phase1-F-supcon-e50-clean-manysig-m9-r02'
ARMS=('step_norm_fidelity','factor_half_fidelity')
VIEWS=('clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban')


def _finite(value,shape,name):
    value=np.asarray(value,dtype=np.float64)
    if value.shape!=shape or not np.isfinite(value).all():raise ValueError('Finite matched '+name+' required')
    return value


def fidelity(teacher,student,labels):
    labels=np.asarray(labels)
    if labels.ndim!=1 or not len(labels) or labels.dtype.kind not in 'iu' or (labels<0).any() or (labels>=6).any():
        raise ValueError('Actual same-row six-class labels required')
    n=len(labels);a=_finite(teacher['logits'],(n,6),'teacher logits');b=_finite(student['logits'],(n,6),'student logits')
    features=[_finite(value['features'],(n,160),'features') for value in (teacher,student)]
    iq=[_finite(value['iq'],(n,2,256),'saved restored IQ') for value in (teacher,student)]
    before,after=a.argmax(1),b.argmax(1);old=before==labels;new=after==labels;flipped=before!=after
    sorted_a=np.sort(a,axis=1);margin=sorted_a[:,-1]-sorted_a[:,-2]
    change=np.max(abs(a-b),axis=1);prob=np.exp(a-a.max(1,keepdims=True));confidence=prob.max(1)/prob.sum(1)
    units=[v/np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-12) for v in features]
    feature_distance=np.clip(1-(units[0]*units[1]).sum(1),0,2)
    nmse=((iq[0]-iq[1])**2).mean((1,2))/np.maximum((iq[0]**2).mean((1,2)),1e-4)
    lost=old&~new;gained=~old&new
    cases={}
    for name,mask in (('all',np.ones(n,bool)),('decision_changed',flipped),('R17_correct_student_wrong',lost),
                      ('R17_wrong_student_correct',gained),('decision_unchanged',~flipped)):
        cases[name]=dict(count=int(mask.sum()),**{key:float(values[mask].mean()) if mask.any() else None for key,values in (
            ('teacher_confidence_mean',confidence),('teacher_logit_margin_mean',margin),('logit_max_absolute_change_mean',change),
            ('feature_cosine_distance_to_R17_mean',feature_distance),('saved_IQ_NMSE_to_R17_mean',nmse))})
    bins=[]
    for low,high in ((0.,.6),(.6,.9),(.9,.99),(.99,1.0000001)):
        mask=(confidence>=low)&(confidence<high)
        bins.append(dict(teacher_confidence_low=low,teacher_confidence_high=min(high,1.),count=int(mask.sum()),
            decision_changed=int((mask&flipped).sum()),lost=int((mask&lost).sum()),gained=int((mask&gained).sum())))
    result=dict(count=n,teacher_correct=int(old.sum()),student_correct=int(new.sum()),
        teacher_accuracy=100*float(old.mean()),student_accuracy=100*float(new.mean()),
        decision_changed=int(flipped.sum()),lost=int(lost.sum()),gained=int(gained.sum()),
        net_correct_delta=int(gained.sum()-lost.sum()),both_wrong_changed=int((~old&~new&flipped).sum()),
        confidence_bins=bins,cases=cases)
    if result['student_correct']-result['teacher_correct']!=result['net_correct_delta']:
        raise ValueError('Paired decision arithmetic differs')
    return result


def _read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def source_arrays(folder):
    """All seven prediction sets validate before any source label open."""
    folder=Path(folder);done=_read(folder/'prediction/complete.json');summary=_read(folder/'summary.json')
    expected=dict(status='PREDICTIONS_COMPLETE',kind='native_map_prior',role='source',count=1800,views=list(VIEWS),
        readouts=['restored'],TX_truth_reads=0,state_updates=0,head_fit_updates=0,F_run=READER,F_seed=2026092701)
    if any(done.get(k)!=v for k,v in expected.items()) or any(summary.get(k)!=v for k,v in
            dict(status='ANALYZED',role='source',count=1800,F_run=READER,F_seed=2026092701,original_default_head=True,
                 all_predictions_before_truth=True,target_access=False,head_fit_updates=0).items()):
        raise ValueError('Only complete already-scored source predictions allowed')
    ids=np.load(folder/'prediction/physical_ids.npy',allow_pickle=False)
    if ids.shape!=(1800,) or len(np.unique(ids))!=1800:raise ValueError('Same complete physical source panel required')
    result={}
    for view in VIEWS:
        result[view]={}
        for key,shape in (('logits',(1800,6)),('features',(1800,160)),('iq',(1800,2,256))):
            value=np.load(folder/'prediction'/(view+'.restored.'+key+'.npy'),mmap_mode='r',allow_pickle=False)
            if value.shape!=shape or not np.isfinite(value).all():raise ValueError('All finite frozen source predictions required before labels')
            result[view][key]=value
    return ids,result


def analyze_saved_source(runs,source_cache):
    runs,source_cache=Path(runs),Path(source_cache)
    if (_read(runs/TEACHER/'launch.json')['commit']!='cd97744884b5e460926e48c8b7efb7d78c6da0f4'
            or _read(runs/COMPACT/'launch.json')['commit']!='1515d73c5de3e962a822bbaca34182e28a6ae584'):
        raise ValueError('Original frozen R17 and R02 method identities required')
    done=_read(runs/COMPACT/'complete.json')
    if done.get('status')!='ANALYZED' or done.get('target_rows')!=[] or done.get('source_only') is not True:
        raise ValueError('Completed source-only R02 rows required')
    ids,teacher=source_arrays(runs/TEACHER/'unroll_unshared/native_map_prior');students={}
    for arm in ARMS:
        candidate_ids,value=source_arrays(runs/COMPACT/arm/'native_map_prior')
        if not np.array_equal(ids,candidate_ids):raise ValueError('Paired prediction physical IDs differ')
        students[arm]=value
    reference=runs/ACTION/'cache/validation'
    if not np.array_equal(ids,np.load(reference/'physical_ids.npy',allow_pickle=False)):
        raise ValueError('Original source reference panel differs')
    manifest=_read(source_cache/'manifest.json')
    if manifest.get('status')!='READY' or manifest.get('native_role_ids')!='EXACT_MATCH':
        raise ValueError('Existing source data handles required')
    role_ids=np.load(source_cache/'V/physical_ids.npy',allow_pickle=False);positions={str(pid):i for i,pid in enumerate(role_ids)}
    if any(str(pid) not in positions for pid in ids):raise ValueError('Only source V physical IDs allowed')
    index=np.asarray([positions[str(pid)] for pid in ids],dtype=np.int64)
    rx=np.load(source_cache/'V/rx_i.npy',allow_pickle=False)[index];day=np.load(source_cache/'V/day_i.npy',allow_pickle=False)[index]
    if not np.array_equal(rx,np.load(reference/'rx.npy',allow_pickle=False)) or not np.array_equal(day,np.load(reference/'day.npy',allow_pickle=False)):
        raise ValueError('Matched source receiver/day grouping required')
    # Source V labels are evaluated only after all three rows x seven views validate.
    labels=np.load(source_cache/'V/labels.npy',allow_pickle=False)[index]
    output=dict(status='SOURCE_SAVED_PREDICTIONS_ANALYSIS',teacher_run=TEACHER,source_run=COMPACT,count=1800,
        F_run=READER,F_seed=2026092701,original_default_head=True,all_predictions_fixed_before_source_truth=True,
        source_V_labels_evaluated=1800,raw_IQ_reads=0,model_forwards=0,checkpoint_weight_reads=0,
        target_reads=0,training_updates=0,head_fit_updates=0,arms={},
        limitation='Post-hoc paired source V diagnostics only;not target evidence or a causal attribution;R03 remains fixed;no new prediction or model fitting')
    for arm,student in students.items():
        views={}
        for view in VIEWS:
            metrics=fidelity(teacher[view],student[view],labels);groups={}
            for dimension,values in (('TX',labels),('RX',rx),('day',day)):
                groups[dimension]={}
                for value in np.unique(values):
                    mask=values==value
                    record=fidelity({k:v[mask] for k,v in teacher[view].items()},
                                    {k:v[mask] for k,v in student[view].items()},labels[mask])
                    groups[dimension][str(value)]={key:record[key] for key in ('count','teacher_correct','student_correct','decision_changed','lost','gained','net_correct_delta','both_wrong_changed')}
                if any(sum(r[key] for r in groups[dimension].values())!=metrics[key] for key in ('count','teacher_correct','student_correct','decision_changed','lost','gained','net_correct_delta','both_wrong_changed')):
                    raise ValueError('Paired source group counters must partition overall results')
            views[view]=dict(**metrics,groups=groups)
        output['arms'][arm]=dict(views=views)
    return output
