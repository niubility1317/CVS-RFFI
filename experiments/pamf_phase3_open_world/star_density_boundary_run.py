"""Frozen source boundary students with one unchanged source density state."""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import re
import sys
import numpy as np
import torch
from . import star_dura_best_run as best
from . import star_class_density_run as parent
from . import star_class_density_core as numerical
from . import star_local_cov_run as frozen
from . import star_signal_open_eval as source_eval
from . import star_macro12_risk as risk

RUN_ID='20261031-phase3-density-boundary-s2026092701-r01'
SCHEMA='pamf_phase3_density_boundary_prediction_v1'
STATUS='DENSITY_BOUNDARY_PREDICTIONS_COMPLETE'
MODELS_STATUS='DENSITY_BOUNDARY_MODELS_FROZEN'
CALIBRATION_STATUS='DENSITY_BOUNDARY_THRESHOLDS_FROZEN'
ARMS=('E0','REPLAY','BOUNDARY')
METHODS=('E0_pooled','E0_predicted','REPLAY_pooled','REPLAY_predicted','BOUNDARY_pooled','BOUNDARY_predicted')
PRIMARY_METHOD=METHODS[5]
CONDITIONS=('source_DURA','clean_DURA','star_DURA')
COUNTS=dict(L=6300,OE=6134,V=27000,novel=3150)
Q_COUNT=864
FIELDS={'run_id','code_commit','train_config','output_root','batch_size','Q_conditions'}
KNOWN_FIELDS,NOVEL_FIELDS,TARGET_FIELDS=best.KNOWN_FIELDS,best.NOVEL_FIELDS,best.TARGET_FIELDS
_read,_write,_same=best._read,best._write,best._same


def _train():
    from . import star_density_boundary_train
    return star_density_boundary_train


def validate_config(c):
    if (not isinstance(c,dict) or set(c)!=FIELDS or c.get('run_id')!=RUN_ID or re.fullmatch('[0-9a-f]{40}',str(c.get('code_commit'))) is None
            or type(c.get('batch_size')) is not int or c['batch_size']!=128 or not isinstance(c.get('output_root'),str) or not c['output_root'].strip()):
        raise ValueError('Exact source-only boundary evaluation required; no truth/U/K')
    t=_train().validate_config(c['train_config'])
    if t['run_id']!=RUN_ID or t['code_commit']!=c['code_commit'] or c['Q_conditions']!=t['reference_config']['Q_conditions']:
        raise ValueError('Same completed training and fixed original clean/star public roles required')
    out=Path(c['output_root']).resolve();p=t['reference_config']
    roots=[t['output_root'],p['output_root'],p['train_config']['IQ_root']]+[v['public_root'] for v in c['Q_conditions'].values()]
    if any(out.is_relative_to(Path(v).resolve()) or Path(v).resolve().is_relative_to(out) for v in roots):raise ValueError('New prediction output cannot overlap frozen inputs')
    return c


def _reference(c):return frozen._reference(c['train_config'])
def _backend(c,device):return _train()._backend(c['train_config'],device)
def _load_model(c,arm,device):return _train().load_trained_model(c['train_config'],arm,device)
def _iq_role(c,role,identity):return source_eval._iq_role(c['train_config']['reference_config']['train_config'],role,identity)
def _public(c,name):return parent._public(c,name)
def _raw(state,z,closed):return numerical.raw_scores(state,z,closed)[:,[1,3]]


def _validate(a,role):
    n=COUNTS['V' if role=='known' else 'novel'] if role in ('known','novel') else Q_COUNT
    if role in ('known','novel'):
        best._validate_rows(dict(a,signal_features=a['signal_features'][:,0],raw_scores=a['raw_scores'][:,:2],scores=a['scores'][:,:2]),role)
    elif (set(a)!=TARGET_FIELDS or a['ids'].shape!=(n,) or a['ids'].dtype.kind!='U' or len(set(a['ids'].tolist()))!=n
            or a['default_logits'].shape!=(n,6) or a['default_logits'].dtype!=np.float32 or not np.isfinite(a['default_logits']).all()
            or a['closed'].shape!=(n,) or a['closed'].dtype!=np.int64 or not np.array_equal(a['closed'],a['default_logits'].argmax(1))
            or not np.array_equal(a['baseline_closed'],a['closed'])):raise ValueError('Complete original same-DURA F0 query identity required')
    z=a['signal_features']
    if (z.shape!=(n,3,160) or z.dtype!=np.float32 or not np.isfinite(z).all() or not np.allclose(np.linalg.norm(z.astype(np.float64),axis=2),1,rtol=0,atol=2e-6)
            or any(a[k].shape!=(n,6) or a[k].dtype!=np.float64 or not np.isfinite(a[k]).all() for k in ('raw_scores','scores'))):
        raise ValueError('Three frozen unit160 representations and six FP64 readouts required')
    return a


def _source_rows(root,condition,role):
    if condition!='source_DURA' or role not in ('known','novel'):raise ValueError('Original source-DURA role required')
    with np.load(Path(root)/'source'/condition/(role+'_predictions.npz'),allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
    return _validate(a,role)


def _target_rows(root,condition):
    if condition not in CONDITIONS[1:]:raise ValueError('Fixed clean/star DURA condition required')
    with np.load(Path(root)/'targets'/condition/'predictions.npz',allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
    return _validate(a,'Q')


def _forward(b,models,values,device,with_parent=False):
    x=np.asarray(values)
    if x.dtype!=np.float32 or x.shape!=(len(x),2,256) or not np.isfinite(x).all():raise ValueError('Finite public full256 IQ required')
    with torch.no_grad(),torch.autocast(device_type=torch.device(device).type,enabled=False):
        tensor=torch.tensor(x.tolist(),dtype=torch.float32,device=device);restored=b['dura'][1](tensor)[0].detach().clone()
        if restored.dtype!=torch.float32 or restored.shape!=tensor.shape or not bool(torch.isfinite(restored).all()):raise ValueError('Same finite DURA output must reach original F0 and both students')
        logits=b['reader'].classify_features(b['reader'].features(restored));zs=[models[k](restored) for k in (ARMS if with_parent else ARMS[1:])]
    return dict(default_logits=np.asarray(logits.detach().cpu().tolist(),np.float32),signal_features=np.stack([np.asarray(z.detach().cpu().tolist(),np.float32) for z in zs],axis=1))


def _smoke(b,models,device):
    from .star_open_feature_export import _synthetic
    a=_forward(b,models,np.asarray(_synthetic(device).detach().cpu().tolist(),np.float32),device,True)
    if (a['signal_features'].shape!=(2,3,160) or not np.isfinite(a['signal_features']).all()
            or not np.allclose(np.linalg.norm(a['signal_features'].astype(np.float64),axis=2),1,atol=2e-6,rtol=0)
            or a['default_logits'].shape!=(2,6) or not np.isfinite(a['default_logits']).all()):raise ValueError('Actual F0/G/three-open-model no-query smoke failed')
    return dict(status='PASS',actual_F0_signal_DURA_checkpoints=True,real_data_reads=0,optimizer_updates=0,
                synthetic_signal_forward_rows=6,synthetic_F_forward_rows=2,G_synthetic_forward_rows=2)


def evaluate(c,device='cuda:0'):
    validate_config(c);out=Path(c['output_root'])
    if out.exists():raise FileExistsError('Observe original output; no replacement or automatic restart')
    b=_backend(c,device);models={'E0':b['model']};identities={'E0':b['model_identity']}
    for arm in ARMS[1:]:models[arm],identities[arm]=_load_model(c,arm,device)
    # All neural training must complete before even reading the old V calibration.
    pm,state,old_tau,old_cal=_reference(c)
    if b['model_identity']!=pm['model_identity'] or b['reader_identity']!=pm['parent_F0_identity'] or b['dura_identity']!=pm['loaded_dura_identity']:
        raise ValueError('Original F0/r03/DURA must match the selected source-only reference')
    for arm in ARMS[1:]:
        i=identities[arm]
        if (i.get('arm')!=arm or i.get('updates')!=_train().STEPS or i.get('parent_signal_identity')!=pm['model_identity']
                or i.get('parent_F0_identity')!=pm['parent_F0_identity'] or i.get('dura_identity')!=pm['dura_identity'] or not i.get('source_only') or i.get('target_U_used')):
            raise ValueError('Both new students require explicit original source-only ancestry')
    for model in models.values():
        if any(p.requires_grad or p.dtype!=torch.float32 for p in model.parameters()) or any(m.training for m in model.modules()):raise ValueError('All neural states must freeze before V/novel/query')
    with np.load(Path(c['train_config']['output_root'])/'density.npz',allow_pickle=False) as z:training_density={k:z[k].copy() for k in z.files}
    if not _same(training_density,state):raise ValueError('One original density must be fixed throughout training and prediction')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    snapshots=[(m,best._snapshot_model(m)) for m in [b['reader'],b['dura'][0],*models.values()]];original=deepcopy(state)
    def check():
        for model,before in snapshots:best._check_model(model,before)
        if not _same(state,original):raise ValueError('Fixed source density changed')
    smoke=_smoke(b,models,device);check();out.mkdir(parents=True);_write(out/'resolved_config.json',c);_write(out/'smoke.json',smoke)
    best._save(out/'models/density.npz',state);_write(out/'models/complete.json',dict(schema=SCHEMA,status=MODELS_STATUS,run_id=RUN_ID,code_commit=c['code_commit'],arm_order=list(ARMS),identities=identities,
        states_shared_and_fixed=True,source_density_models_fitted=0,normalizer_fit=False,source_V_fit=False,novel_fit=False,target_Q_fit=False,neural_states_frozen=True))
    root=c['train_config']['reference_config']['output_root'];source_ids=set()
    def source(role):
        old=parent._rows(root,role);fields=KNOWN_FIELDS if role=='known' else NOVEL_FIELDS;a={k:old[k].copy() for k in fields}
        iq=_iq_role(c,'V' if role=='known' else 'novel',pm['model_identity'])
        try:
            source_eval._align(iq,a,'V' if role=='known' else 'novel');ids=set(a['ids'].tolist())
            if ids&source_ids:raise ValueError('Source known and novel roles overlap')
            source_ids.update(ids);z=[old['signal_features']]+[source_eval._project(models[k],iq['restored'],device) for k in ARMS[1:]]
            raw=np.concatenate([old['raw_scores'][:,[1,3]]]+[_raw(state,z[j],a['closed']) for j in (1,2)],axis=1)
            return dict(a,signal_features=np.stack(z,axis=1),raw_scores=raw)
        finally:source_eval._close_role(iq)
    V=source('known');tau=np.empty((6,6),np.float64);tau[:2]=old_tau
    reports={METHODS[j]:deepcopy(old_cal['methods'][parent.METHODS[k]]) for j,k in enumerate((1,3))}
    for j in range(2,6):
        fitted,report=risk.calibrate_known(V['ids'],V['y'],V['closed'],V['baseline_closed'],V['raw_scores'][:,j],V['rx_i'],V['scenario']);tau[j]=fitted['conditional_thresholds'];reports[METHODS[j]]=report;check()
    best._save(out/'thresholds.npz',dict(conditional=tau));(out/'calibration').mkdir();_write(out/'calibration/complete.json',dict(schema=SCHEMA,status=CALIBRATION_STATUS,run_id=RUN_ID,code_commit=c['code_commit'],method_names=list(METHODS),primary_method=PRIMARY_METHOD,
        known_count=COUNTS['V'],group_count=42,hard_names=list(risk.HARD_NAMES),diagnostic_names=list(risk.DIAGNOSTIC_NAMES),methods=reports,conditional_thresholds=tau.tolist(),known_only=True,unknown_inputs_used=False,target_reads=0,
        source_V_neural_fit=False,source_V_density_fit=False,source_V_threshold_fit=True,all_shared_source_thresholds=True))
    V['scores']=V['raw_scores']-tau[:,V['closed']].T;_validate(V,'known');best._save(out/'source/source_DURA/known_predictions.npz',V);frozen_tau=tau.copy();check();print(CALIBRATION_STATUS,flush=True)
    N=source('novel');N['scores']=N['raw_scores']-tau[:,N['closed']].T;_validate(N,'novel');best._save(out/'source/source_DURA/novel_predictions.npz',N);check();manifests={}
    for name in CONDITIONS[1:]:
        iq=None
        try:
            old=parent._target_rows(root,name);manifest,ids,iq=_public(c,name)
            if not np.array_equal(ids,old['ids']) or source_ids&set(ids.tolist()):raise ValueError('Same source-disjoint original Q physical records required')
            blocks=[_forward(b,models,iq[start:start+128],device) for start in range(0,len(ids),128)];logits=np.concatenate([a['default_logits'] for a in blocks]);new_z=np.concatenate([a['signal_features'] for a in blocks])
            if not np.array_equal(logits.argmax(1),old['closed']) or not np.allclose(logits,old['default_logits'],rtol=1e-6,atol=1e-6):raise ValueError('Repeated same-DURA original F0 identity differs')
            z=np.concatenate((old['signal_features'][:,None],new_z),axis=1);raw=np.concatenate([old['raw_scores'][:,[1,3]]]+[_raw(state,z[:,j],old['closed']) for j in (1,2)],axis=1)
            a=dict(old,signal_features=z,raw_scores=raw,scores=raw-tau[:,old['closed']].T);_validate(a,'Q');best._save(out/'targets'/name/'predictions.npz',a);manifests[name]=manifest;check();print('DENSITY_BOUNDARY_Q_COMPLETE',name,len(ids),flush=True)
        finally:
            if hasattr(iq,'_mmap'):iq._mmap.close()
    if not np.array_equal(tau,frozen_tau):raise ValueError('Shared source thresholds changed after novel/query')
    done=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c,method_names=list(METHODS),primary_method=PRIMARY_METHOD,arm_order=list(ARMS),conditions=list(CONDITIONS),
        known_count=COUNTS['V'],novel_count=COUNTS['novel'],Q_counts={name:Q_COUNT for name in CONDITIONS[1:]},Q_manifests=manifests,reference_prediction_identity=pm,model_identities=identities,parent_F0_identity=pm['parent_F0_identity'],dura_identity=pm['dura_identity'],loaded_dura_identity=b['dura_identity'],
        classification_path='same_DURA_output_original_F0',identity_baseline_path='same_DURA_output_original_F0',classification_unchanged=True,score_threshold=0.,source_models_frozen_before_V=True,all_thresholds_frozen_before_novel_and_Q=True,
        all_three_DURA_conditions_complete=True,all_conditions_through_DURA=True,source_E0_controls_reused_exactly=True,source_states_unchanged=True,frozen_parameters_buffers_unchanged=True,states_shared_and_fixed=True,
        source_density_models_fitted=0,new_threshold_sets_fitted=4,source_V_neural_fit=False,source_V_density_fit=False,source_V_threshold_fit=True,source_IQ_reads=COUNTS['V']+COUNTS['novel'],target_Q_IQ_reads=2*Q_COUNT,
        target_U_reads=0,training_target_U_count=0,private_truth_reads=0,U_truth_reads=0,source_novel_truth_reads=0,target_truth_reads=0,original_U_s_reads=0,new_neural_updates=0,inherited_new_REPLAY_updates=_train().STEPS,inherited_new_BOUNDARY_updates=_train().STEPS,
        real_F_forward_rows=2*Q_COUNT,G_forward_rows=2*Q_COUNT,real_signal_forward_rows=2*(COUNTS['V']+COUNTS['novel'])+4*Q_COUNT,
        synthetic_F_forward_rows=2,G_synthetic_forward_rows=2,synthetic_signal_forward_rows=6,Q_E0_new_forward_rows=0,source_bank_deployed=False,U_bank_deployed=False,formal_target_accessed=False,final_phase3_complete=False,full_goal_status='ACTIVE',A_N=None,H_OW=None)
    check();_write(out/'complete.json',done);print(STATUS,flush=True);return done


run=evaluate
def main():
    print(json.dumps(dict(event='DENSITY_BOUNDARY_EVAL_PROCESS',pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv)),flush=True)
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args();evaluate(_read(a.config),a.device)


if __name__=='__main__':main()
