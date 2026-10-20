"""Nonblocking paired analysis of already-scored fixed R17 target predictions."""
import os
from pathlib import Path
import numpy as np
from experiments.pamf_r17_target_compare import design as d
from tools.analyze_pamf_r17_task_saved_predictions import read,saved_arrays,paired_views

METHOD='81c39cb908373d4e19b70ec8406f4083529da99a'
FORMAL='20261019-phase1-restoration-formal-frozen-residual-s2026092701-r05'


def analyze_saved_target(runs,target_truth):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Independent CPU analysis required')
    runs=Path(runs);root=runs/d.RUN
    if read(root/'launch.json')['commit']!=METHOD or read(runs/d.PARENT_RUN/'launch.json')['commit']!=d.PARENT_COMMIT:
        raise ValueError('Actual frozen comparison and original R17 identities required')
    expected=dict(status='ANALYZED',target_rows=list(d.ARMS),new_training_updates=0,target_feedback=False)
    if read(root/'complete.json')!=expected or read(root/'comparison_plan.json')!=d.configuration():
        raise ValueError('Both fixed target rows must complete before paired analysis')
    ids,teacher,teacher_score=saved_arrays(runs/d.PARENT_RUN/'unroll_unshared/target_confirmation'/d.KIND,'target',task_knowledge=False)
    students={}
    for arm in d.ARMS:
        current,arrays,score=saved_arrays(root/arm/'target_confirmation'/d.KIND,'target',task_knowledge=True)
        if not np.array_equal(ids,current):raise ValueError('Paired target physical IDs differ')
        students[arm]=(arrays,score)
    reference=runs/FORMAL/'reference';manifest=read(reference/'manifest.json')
    if (manifest.get('status')!='FORMAL_REFERENCE_READY' or manifest.get('count')!=168000 or manifest.get('views')!=list(d.VIEWS)
            or len(ids)!=168000 or not np.array_equal(ids,np.load(reference/'physical_ids.npy',allow_pickle=False))):
        raise ValueError('Original complete formal target panel required before truth')
    # All three rows x seven saved prediction sets validate before this truth open.
    truth=read(target_truth);labels=np.asarray([truth[str(pid)]['label'] for pid in ids],np.int64)
    rx=np.asarray([str(truth[str(pid)]['receiver']) for pid in ids]);day=np.asarray([str(truth[str(pid)].get('day','unavailable')) for pid in ids])
    return dict(status='SAVED_TARGET_PAIRED_PREDICTIONS_ANALYZED',run_id=d.RUN,teacher_run=d.PARENT_RUN,
        method_commit=METHOD,target_labels_evaluated=168000,all_three_rows_seven_views_before_truth=True,
        raw_IQ_reads=0,checkpoint_weight_reads=0,model_forwards=0,training_updates=0,head_fit_updates=0,target_feedback=False,
        rows={arm:paired_views(teacher,arrays,labels,rx,day,teacher_score,score) for arm,(arrays,score) in students.items()},
        limitation='Post-hoc same-row fixed benchmark diagnostics;not new prediction,selection feedback or causal attribution')
