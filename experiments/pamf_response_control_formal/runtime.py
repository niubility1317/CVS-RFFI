import math
import os
from pathlib import Path
import numpy as np
from . import design as d
from experiments.pamf_response_restore.runtime import load_checkpoint as original_load
from experiments.pamf_restoration.runner import read_json
from experiments.pamf_uncertainty_restore.evaluation import reference as original_reference,VIEWS
from experiments.pamf_fcrossrx_readout.frozen import REFERENCE_RUN

ROOT=Path(__file__).resolve().parents[2]


def source_counts(summary,readout):
    expected=dict(status='ANALYZED',role='source',count=1800,F_run=REFERENCE_RUN,F_seed=2026092701,
        original_default_head=True,all_predictions_before_truth=True,training_TX_labels_read=0,
        training_F_access=False,head_fit_updates=0,target_access=False,
        kind='reference_controls' if readout=='identity' else d.KIND)
    if any(summary.get(k)!=v for k,v in expected.items()):raise ValueError('Same frozen F50 source-only evidence required')
    rows=[r for r in summary['records'] if r['readout']==readout and r['dimension']=='overall']
    if len(rows)!=7 or {r['view'] for r in rows}!=set(VIEWS):raise ValueError('All seven source views required')
    result={};population=None
    for row in rows:
        cm=np.asarray(row['confusion']);accuracy=row['accuracy']
        if (cm.shape!=(6,6) or not np.issubdtype(cm.dtype,np.number) or not np.isfinite(cm).all()
                or (cm<0).any() or not np.equal(cm,np.floor(cm)).all() or int(cm.sum())!=1800
                or row['count']!=1800 or row['value']!='all' or not math.isfinite(accuracy)):
            raise ValueError('Finite complete integer source confusion counts required')
        counts=cm.sum(1)
        if population is not None and not np.array_equal(counts,population):raise ValueError('Same source population across views required')
        population=counts;correct=int(np.trace(cm))
        if abs(accuracy-100*correct/1800)>1e-8:raise ValueError('Source accuracy and confusion disagree')
        result[row['view']]=correct
    return result


def qualify(candidate,reference,raw):
    new,old,base=source_counts(candidate,'restored'),source_counts(reference,'restored'),source_counts(raw,'identity')
    gain=sum(new[v]-old[v] for v in VIEWS[1:]);clean_lost=base['clean']-new['clean']
    qualified=gain>0 and clean_lost*100<=1800*.1
    return dict(status='SOURCE_DECISION_FROZEN',qualified=bool(qualified),candidate=d.CANDIDATE,
        target_candidates=[d.KIND] if qualified else [],source_policy=d.POLICY,source_count=1800,
        six_scene_net_correct_gain=gain,mean6_gain_pp=100*gain/(1800*6),clean_drop_vs_raw_pp=100*clean_lost/1800,
        per_view_gain_pp={v:100*(new[v]-old[v])/1800 for v in VIEWS},
        source_correct_counts=dict(candidate=new,reference=old,raw=base),
        source_reference=d.UNROLLED,identity_uses_source_V_labels=True,source_V_fit_updates=0,
        new_training_updates=0,training_TX_F_access=False,parent_decisions_modified=False,
        target_reads_before_decision=0,target_feedback=False,goal_complete=False)


def validate_training(value):
    expected=dict(status='SOURCE_TRAINED',candidate=d.CANDIDATE,epochs=20,updates=4940,
        L_exposures=126000,U_exposures=1134000,L_waveform_forwards=378000,U_waveform_forwards=3402000,
        TX_labels_read=0,F_outputs_read=0,F_weights_read=0,target_reads=0,frozen_context_updates=0,
        prior_initialization='scratch',inherited_prior_epochs=0,response_constraint_active=False)
    if any(value.get(k)!=v for k,v in expected.items()):raise ValueError('Original complete TX/F-free R32 control training required')


def load_control(path,device):
    model,value=original_load(path,device)
    if value.get('candidate')!=d.CANDIDATE or value.get('epoch')!=20 or value.get('updates')!=4940:
        raise ValueError('Only complete R32 paired control checkpoint allowed')
    return model.requires_grad_(False).eval(),value


def parent_selection(args):
    parent=args.parent_run
    if parent.name!=d.PARENT or read_json(parent/'launch.json')['commit']!=d.PARENT_COMMIT:
        raise ValueError('Fixed original R32 parent required')
    complete=read_json(parent/'complete.json')
    if complete.get('status')!='ANALYZED' or complete.get('source_rows')!=['pair_aug_control','observable_response']:
        raise ValueError('R32 must finish before this separate evaluation')
    # Preserve the existing parent worker and controller while they are alive.
    records=[read_json(parent/'launch.json'),*read_json(parent/'workers.json')['rows']]
    for record in records:
        proc=Path('/proc')/str(record['pid'])
        try:
            stat=(proc/'stat').read_text();state=stat[stat.rfind(')')+2:].split()[0]
            cwd=os.readlink(proc/'cwd')
        except FileNotFoundError:continue
        if state!='Z' and cwd==record['cwd']:raise ValueError('Parent process still alive;do not launch')
    for root,name,commit in [(args.unrolled_run,d.UNROLLED,d.UNROLLED_COMMIT),(args.reference_run,d.REFERENCE,d.REFERENCE_COMMIT)]:
        if root.name!=name or read_json(root/'launch.json')['commit']!=commit:raise ValueError('Original source comparison lineage required')
    row=parent/d.CANDIDATE;validate_training(read_json(row/'training_complete.json'))
    done=read_json(row/'complete.json')
    if (done.get('status')!='ANALYZED' or done.get('candidate')!=d.CANDIDATE or done.get('role')!='source'
            or done.get('all_predictions_before_truth') is not True or done.get('training_TX_F_access') is not False):
        raise ValueError('Completed original control source scoring required')
    paths=[row/d.KIND/'summary.json',args.unrolled_run/'unroll_unshared'/d.KIND/'summary.json',args.reference_run/'reference_controls/summary.json']
    panel=None
    for path in paths:
        ids=np.load(path.parent/'prediction/physical_ids.npy',allow_pickle=False)
        if ids.shape!=(1800,) or (panel is not None and not np.array_equal(ids,panel)):raise ValueError('Same source physical panel required')
        panel=ids
    return qualify(*[read_json(path) for path in paths])


def audit_target(args):
    if read_json(args.output/'resolved_config.json')!=d.configuration(args.expected_commit):raise ValueError('Registered evaluation contract required')
    decision=read_json(args.output/'source_decision.json')
    required=dict(status='SOURCE_DECISION_FROZEN',qualified=True,candidate=d.CANDIDATE,source_policy=d.POLICY,
        target_candidates=[d.KIND],new_training_updates=0,parent_decisions_modified=False,
        target_reads_before_decision=0,target_feedback=False)
    if any(decision.get(k)!=v for k,v in required.items()) or args.kind!=d.KIND:
        raise ValueError('Only source-positive paired control may access target')


def reference(args,role):
    if role!='target':raise ValueError('This run only reuses completed source evidence')
    audit_target(args)
    return original_reference(args,role)
