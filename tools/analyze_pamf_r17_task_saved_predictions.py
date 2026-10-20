"""Post-hoc paired R17/R04 saved predictions; no raw IQ or model forwards.

This analysis never affects selection, predictions, or the registered target gate.
"""
import json
import os
from pathlib import Path
import numpy as np

from experiments.pamf_r17_task import design as d
from tools.analyze_pamf_r17_saved_fidelity import fidelity

METHOD='a310b3b5ff30e23584e7bc950e8a0ff96ac685fb'
FORMAL='20261019-phase1-restoration-formal-frozen-residual-s2026092701-r05'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def saved_arrays(folder,role,*,task_knowledge):
    if role not in ('source','target'):raise ValueError('Registered role required before IO')
    folder=Path(folder);count=1800 if role=='source' else 168000
    done=read(folder/'prediction/complete.json');summary=read(folder/'summary.json')
    expected=dict(status='PREDICTIONS_COMPLETE',kind=d.KIND,role=role,count=count,views=list(d.VIEWS),
        readouts=['restored'],TX_truth_reads=0,state_updates=0,head_fit_updates=0,F_run=d.F_RUN,F_seed=d.F_SEED)
    required=dict(status='ANALYZED',role=role,count=count,F_run=d.F_RUN,F_seed=d.F_SEED,original_default_head=True,
        all_predictions_before_truth=True,target_access=role=='target',head_fit_updates=0,
        training_F_access=task_knowledge,training_TX_labels_read=0)
    if any(done.get(k)!=v for k,v in expected.items()) or any(summary.get(k)!=v for k,v in required.items()):
        raise ValueError('Complete frozen same-role original-head predictions and honest F access required')
    ids=np.load(folder/'prediction/physical_ids.npy',allow_pickle=False)
    if ids.shape!=(count,) or len(np.unique(ids))!=count:raise ValueError('Complete unique physical panel required')
    arrays={}
    for view in d.VIEWS:
        arrays[view]={}
        for key,shape in (('logits',(count,6)),('features',(count,160)),('iq',(count,2,256))):
            value=np.load(folder/'prediction'/(view+'.restored.'+key+'.npy'),mmap_mode='r',allow_pickle=False)
            if value.shape!=shape or not np.isfinite(value).all():raise ValueError('All seven finite saved views required before truth')
            arrays[view][key]=value
    return ids,arrays,summary


def _overall(summary,view):
    rows=[r for r in summary['records'] if (r['view'],r['readout'],r['dimension'],r['value'])==(view,'restored','overall','all')]
    if len(rows)!=1:raise ValueError('Unique overall score required')
    matrix=np.asarray(rows[0]['confusion'])
    if matrix.shape!=(6,6) or matrix.dtype.kind not in 'iu' or (matrix<0).any() or matrix.sum()!=summary['count']:
        raise ValueError('Integer complete overall confusion required')
    if not np.isclose(rows[0]['accuracy'],100*matrix.trace()/matrix.sum(),rtol=0,atol=1e-9):
        raise ValueError('Existing accuracy and confusion disagree')
    return int(matrix.trace())


def paired_views(teacher,student,labels,rx,day,teacher_summary,student_summary):
    n=len(labels)
    if np.asarray(rx).shape!=(n,) or np.asarray(day).shape!=(n,):raise ValueError('Matched physical grouping required')
    result={};keys=('count','teacher_correct','student_correct','decision_changed','lost','gained','net_correct_delta','both_wrong_changed')
    for view in d.VIEWS:
        value=fidelity(teacher[view],student[view],labels)
        if (value['teacher_correct'],value['student_correct'])!=(_overall(teacher_summary,view),_overall(student_summary,view)):
            raise ValueError('Saved decisions disagree with original independent score')
        groups={}
        for dimension,values in (('TX',labels),('RX',rx),('day',day)):
            groups[dimension]={}
            for item in np.unique(values):
                mask=values==item
                group=fidelity({k:v[mask] for k,v in teacher[view].items()},
                               {k:v[mask] for k,v in student[view].items()},labels[mask])
                groups[dimension][str(item)]={k:group[k] for k in keys}
            if any(sum(r[k] for r in groups[dimension].values())!=value[k] for k in keys):
                raise ValueError('Paired group counts must partition the same physical panel')
        result[view]=dict(**value,groups=groups)
    return result


def source_truth(runs,cache,ids):
    reference=runs/d.ACTION_RUN/'cache/validation'
    if not np.array_equal(ids,np.load(reference/'physical_ids.npy',allow_pickle=False)):
        raise ValueError('Original source reference physical panel differs')
    manifest=read(cache/'manifest.json')
    if manifest.get('status')!='READY' or manifest.get('native_role_ids')!='EXACT_MATCH':
        raise ValueError('Existing validated source handles required')
    roles=np.load(cache/'V/physical_ids.npy',allow_pickle=False);positions={str(pid):i for i,pid in enumerate(roles)}
    if any(str(pid) not in positions for pid in ids):raise ValueError('Source V roles required')
    index=np.asarray([positions[str(pid)] for pid in ids],np.int64)
    rx=np.load(cache/'V/rx_i.npy',allow_pickle=False)[index];day=np.load(cache/'V/day_i.npy',allow_pickle=False)[index]
    if not np.array_equal(rx,np.load(reference/'rx.npy',allow_pickle=False)) or not np.array_equal(day,np.load(reference/'day.npy',allow_pickle=False)):
        raise ValueError('Source receiver/day partition differs')
    return np.load(cache/'V/labels.npy',allow_pickle=False)[index],rx,day


def analyze_saved(runs,source_cache,target_truth):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Independent CPU-only analysis required')
    runs,cache=Path(runs),Path(source_cache);root=runs/d.RUN
    if read(root/'launch.json')['commit']!=METHOD or read(runs/d.PARENT_RUN/'launch.json')['commit']!=d.PARENT_COMMIT:
        raise ValueError('Actual immutable R04 and original R17 method identities required')
    done=read(root/'complete.json');decision=read(root/'source_decision.json')
    selected=decision.get('selected')
    rows=[selected] if selected else []
    if (done.get('status')!='ANALYZED' or done.get('source_rows')!=list(d.ARMS) or done.get('target_rows')!=rows
            or decision.get('status')!='SOURCE_DECISION_FROZEN' or decision.get('target_rows')!=rows
            or decision.get('target_feedback') is not False or decision.get('target_reads_before_decision')!=0
            or selected is not None and selected not in d.ARMS):
        raise ValueError('Completed fixed source matrix and only its selected target required')
    ids,teacher,teacher_score=saved_arrays(runs/d.PARENT_RUN/'unroll_unshared'/d.KIND,'source',task_knowledge=False)
    students={}
    for arm in d.ARMS:
        current,arrays,score=saved_arrays(root/arm/d.KIND,'source',task_knowledge=True)
        if not np.array_equal(current,ids):raise ValueError('Paired source physical IDs differ')
        if arm!=selected and (root/arm/'target_confirmation').exists():raise ValueError('Unselected row accessed target')
        students[arm]=(arrays,score)
    target=None
    if selected:
        target_ids,a,a_score=saved_arrays(runs/d.PARENT_RUN/'unroll_unshared/target_confirmation'/d.KIND,'target',task_knowledge=False)
        current,b,b_score=saved_arrays(root/selected/'target_confirmation'/d.KIND,'target',task_knowledge=True)
        manifest=read(runs/FORMAL/'reference/manifest.json')
        if (manifest.get('status')!='FORMAL_REFERENCE_READY' or manifest.get('count')!=168000 or manifest.get('views')!=list(d.VIEWS)
                or not np.array_equal(current,target_ids) or not np.array_equal(target_ids,np.load(runs/FORMAL/'reference/physical_ids.npy',allow_pickle=False))):
            raise ValueError('Original formal target physical panel required')
        target=(target_ids,a,b,a_score,b_score)
    # Every registered source row and selected target validates before either truth open.
    labels,rx,day=source_truth(runs,cache,ids)
    output=dict(status='SAVED_PAIRED_PREDICTIONS_ANALYZED',run_id=d.RUN,teacher_run=d.PARENT_RUN,method_commit=METHOD,
        F_run=d.F_RUN,F_seed=d.F_SEED,original_default_head=True,training_F_access=True,F_parameter_updates=0,
        all_predictions_fixed_before_truth=True,source_labels_evaluated=1800,target_labels_evaluated=0,
        raw_IQ_reads=0,model_forwards=0,checkpoint_weight_reads=0,training_updates=0,head_fit_updates=0,
        source={arm:paired_views(teacher,v,labels,rx,day,teacher_score,s) for arm,(v,s) in students.items()},target={},
        limitation='Post-hoc same-row fixed-set decision preservation;not new prediction,causal evidence or selection feedback')
    if target:
        target_ids,a,b,a_score,b_score=target;truth=read(target_truth)
        labels=np.asarray([truth[str(pid)]['label'] for pid in target_ids],np.int64)
        rx=np.asarray([str(truth[str(pid)]['receiver']) for pid in target_ids]);day=np.asarray([str(truth[str(pid)].get('day','unavailable')) for pid in target_ids])
        output['target'][selected]=paired_views(a,b,labels,rx,day,a_score,b_score)
        output['target_labels_evaluated']=168000
    return output
