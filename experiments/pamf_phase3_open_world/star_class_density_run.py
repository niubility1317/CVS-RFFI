"""Frozen predicted-class densities; source and two cross-RX tests use DURA."""
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
from . import star_class_density_core as core
from . import star_raw_open_eval as reference
from . import star_signal_open_train as signal_train
from . import star_signal_open_eval as signal_eval
from . import star_macro12_risk as risk
from . import star_relative_source as relative
from .development_transfer import _snapshot_model,_check_model

RUN_ID='20261031-phase3-class-density-s2026092701-r01'
SCHEMA='pamf_phase3_class_density_prediction_v1'
STATUS='CLASS_DENSITY_PREDICTIONS_COMPLETE'
MODELS_STATUS='CLASS_DENSITY_MODELS_FROZEN'
CALIBRATION_STATUS='CLASS_DENSITY_THRESHOLDS_FROZEN'
REFERENCE_RUN='20261031-phase3-star-raw-open-s2026092701-r01'
REFERENCE_CODE='617681c8b2f24e7d05c1e8f5a48cb74d99a1c8cb'
SIGNAL_RUN='20261030-phase3-star-signal-open-s2026092701-r03'
SIGNAL_CODE='dc01a402582f93e452dbaa3ef59a9fbb44151818'
METHODS=core.METHODS
PRIMARY_METHOD=core.PRIMARY_METHOD
PRIMARY_INDEX=3
CONDITIONS=('clean_DURA','star_DURA')
COUNTS=dict(L=6300,OE=6134,V=27000,novel=3150)
Q_COUNT=864
FIELDS={'run_id','code_commit','train_config','reference_prediction_root','output_root','batch_size','Q_conditions','dura_checkpoint','dura_release'}
TARGET_FIELDS={'ids','default_logits','closed','baseline_closed','signal_features','raw_scores','scores'}
KNOWN_FIELDS,NOVEL_FIELDS=reference.KNOWN_FIELDS,reference.NOVEL_FIELDS
_read,_write,_npz,_same=reference._read,reference._write,reference._npz,reference._same


def validate_config(c):
    if (not isinstance(c,dict) or set(c)!=FIELDS or c.get('run_id')!=RUN_ID or re.fullmatch('[0-9a-f]{40}',str(c.get('code_commit'))) is None
            or type(c.get('batch_size')) is not int or c['batch_size']!=128 or not isinstance(c.get('train_config'),dict)
            or any(not isinstance(c.get(k),str) or not c[k].strip() for k in ('reference_prediction_root','output_root','dura_checkpoint','dura_release'))):
        raise ValueError('Exact nine-field frozen three-DURA-test configuration required; no truth/U/K')
    t=signal_train.validate_config(c['train_config'])
    if (t['run_id'],t['code_commit'],t['arm'])!=(SIGNAL_RUN,SIGNAL_CODE,'E_SIGNAL'):raise ValueError('Only the original r03 P1 signal checkpoint is allowed')
    q=c['Q_conditions']
    if not isinstance(q,dict) or set(q)!=set(CONDITIONS):raise ValueError('Both clean_DURA and star_DURA must be preregistered')
    roots=[]
    for name,family in zip(CONDITIONS,('clean_cross_receiver','practical_cross_receiver')):
        spec=q[name]
        if (not isinstance(spec,dict) or set(spec)!={'public_root','capsule_id','split_id','expected_count','view_family'}
                or type(spec['expected_count']) is not int or spec['expected_count']!=Q_COUNT or spec['view_family']!=family
                or any(not isinstance(spec[k],str) or not spec[k].strip() for k in ('public_root','capsule_id','split_id'))):raise ValueError('Exact separate clean/star future-Q public handles required')
        root=Path(spec['public_root'])
        if root.name!='public' or root.parent.name!='evaluation':raise ValueError('Only evaluation public may be opened; no discovery or truth')
        roots.append(root.resolve())
    if roots[0]==roots[1]:raise ValueError('Clean and star inputs must have separate public roots')
    output=Path(c['output_root']).resolve()
    for path in [*roots,Path(c['reference_prediction_root']),Path(t['output_root']),Path(c['dura_release']),Path(c['dura_checkpoint'])]:
        p=path.resolve()
        if output.is_relative_to(p) or p.is_relative_to(output):raise ValueError('Immutable output cannot overlap frozen model/public inputs')
    return c


def _reference(root,train_config):
    """Load source metadata and P1 states only; no P0 checkpoint or novel rows."""
    root=Path(root);m=_read(root/'complete.json')
    expected=dict(schema=reference.SCHEMA,status=reference.STATUS,run_id=REFERENCE_RUN,code_commit=REFERENCE_CODE,arm='E_RAW',method_names=list(reference.METHODS),
        classification_path='P1',known_count=COUNTS['V'],novel_count=COUNTS['novel'],source_only=True,private_truth_reads=0,target_reads=0,original_U_s_reads=0,
        identity_predictions_unchanged=True,models_frozen_before_V=True,all_thresholds_frozen_before_novel=True,source_states_unchanged=True,model_parameters_buffers_unchanged=True)
    if any(m.get(k)!=v for k,v in expected.items()):raise ValueError('Completed fixed raw-open P1 controls required')
    reference.validate_config(m['config']);sm=m.get('signal_reference_identity',{})
    if (sm.get('run_id')!=SIGNAL_RUN or sm.get('code_commit')!=SIGNAL_CODE or sm.get('config',{}).get('train_config')!=train_config
            or signal_eval._identity(sm['model_identity'],sm['config'])!=m['parent_F0_identity'] or sm['model_identity']['dura_identity']!=m['dura_identity']):
        raise ValueError('P1 r03 source identity must be used, never the parent P0 model identity')
    with np.load(root/'models/P1_known12.npz',allow_pickle=False) as z:known={k:z[k].copy() for k in z.files}
    with np.load(root/'models/P1_ratio12.npz',allow_pickle=False) as z:state={k:z[k].copy() for k in z.files}
    core.density.validate_state(known,'known_only');core.density.validate_state(state,'density_ratio')
    if state['known_means'].shape!=(90,160) or any(not np.array_equal(v,state[k]) for k,v in known.items()):raise ValueError('Shared original D160 ninety-mode P1 known statistics required')
    fit=_read(root/'models/complete.json')
    if any(fit.get(k)!=v for k,v in dict(schema=reference.SCHEMA,status=reference.MODELS_STATUS,run_id=REFERENCE_RUN,code_commit=REFERENCE_CODE,
            P1_models_reused=True,P1_known_statistics_shared=True,source_V_fit=False,novel_fit=False).items()):raise ValueError('Original source-only P1 model freeze required')
    with np.load(root/'thresholds.npz',allow_pickle=False) as z:
        if set(z.files)!={'conditional'}:raise ValueError('Complete parent conditional thresholds required')
        tau=z['conditional'].copy()
    cal=_read(root/'calibration/complete.json')
    if (tau.shape!=(4,6) or tau.dtype!=np.float64 or not np.isfinite(tau).all()
            or any(cal.get(k)!=v for k,v in dict(schema=reference.SCHEMA,status=reference.CALIBRATION_STATUS,run_id=REFERENCE_RUN,code_commit=REFERENCE_CODE,
                method_names=list(reference.METHODS),known_count=COUNTS['V'],group_count=42,known_only=True,unknown_inputs_used=False,target_reads=0).items())
            or not np.array_equal(tau,cal.get('conditional_thresholds'))):raise ValueError('Exact frozen P1 known12/ratio12 calibration required')
    for j,name in enumerate(reference.METHODS[:2]):
        if cal['methods'][name]['conditional_thresholds']!=tau[j].tolist():raise ValueError('Pooled source threshold/report mismatch')
    for v in state.values():v.setflags(write=False)
    return m,state,tau[:2].copy(),cal


def _rows(root,role):
    if role not in ('known','novel'):raise ValueError('Only fixed source known or novel public rows allowed')
    with np.load(Path(root)/(role+'_predictions.npz'),allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
    if set(a)!=(KNOWN_FIELDS if role=='known' else NOVEL_FIELDS):raise ValueError('Exact public source fields required; novel truth forbidden')
    signal_eval._validate_rows({k:v for k,v in a.items() if k!='raw_signal_features'},role,4,True)
    if a['raw_signal_features'].shape!=(len(a['ids']),160) or a['raw_signal_features'].dtype!=np.float32 or not np.isfinite(a['raw_signal_features']).all():raise ValueError('Copied parent feature field differs')
    return a


def _backend(c,device):
    from .star_channel import _channel_api
    _channel_api();n=c['train_config']['native_config']
    spec=dict(reader_config={k:n[k] for k in ('base_release','frozen_sources','source_output','model_seed')},source_contract=n['source_contract'],
        dura_checkpoint=c['dura_checkpoint'],dura_release=c['dura_release'])
    backend=relative._backend(spec,device,with_dura=True)
    model,identity=signal_train.load_trained_model(c['train_config'],Path(c['train_config']['output_root'])/'final.pt',device)
    return dict(backend,model=model,model_identity=identity)


def _current(b,values,device):
    values=np.asarray(values)
    if values.dtype!=np.float32 or values.shape!=(len(values),2,256) or not np.isfinite(values).all():raise ValueError('Finite public FP32 full256 IQ required')
    with torch.no_grad(),torch.autocast(device_type=torch.device(device).type,enabled=False):
        x=torch.tensor(values.tolist(),dtype=torch.float32,device=device);restored=b['dura'][1](x)[0].detach().clone()
        if restored.dtype!=torch.float32 or restored.shape!=x.shape or not bool(torch.isfinite(restored).all()):raise ValueError('The same finite DURA output must reach F0 and signal')
        logits=b['reader'].classify_features(b['reader'].features(restored));z=b['model'](restored)
    logits=np.asarray(logits.detach().cpu().tolist(),np.float32);z=np.asarray(z.detach().cpu().tolist(),np.float32)
    if (logits.shape!=(len(values),6) or z.shape!=(len(values),160) or not np.isfinite(logits).all() or not np.isfinite(z).all()
            or not np.allclose(np.linalg.norm(z.astype(np.float64),axis=1),1.,rtol=0,atol=2e-6)):raise ValueError('Complete default logits and unit160 signal features required')
    closed=logits.argmax(1).astype(np.int64)
    return dict(default_logits=logits,closed=closed,baseline_closed=closed.copy(),signal_features=z)


def _smoke(b,device):
    from .star_open_feature_export import _synthetic
    values=np.asarray(_synthetic(device).detach().cpu().tolist(),np.float32);result=_current(b,values,device)
    if len(result['closed'])!=2:raise ValueError('Two actual-checkpoint synthetic G10 rows required')
    return dict(status='PASS',actual_F0_signal_DURA_checkpoints=True,G_synthetic_forward_rows=2,synthetic_F_forward_rows=2,
        synthetic_signal_forward_rows=2,real_data_reads=0,optimizer_updates=0)


def _public(c,name):
    spec=c['Q_conditions'][name]
    if name=='clean_DURA':
        from .development_clean import load_clean_public
        m,ids,iq=load_clean_public(spec['public_root'])
    else:
        from .star_relative_target import _open
        m,ids,iq=_open(dict(Q_public_root=spec['public_root'],Q_capsule_id=spec['capsule_id'],Q_split_id=spec['split_id'],Q_expected_count=Q_COUNT),'Q')
    if (m.get('capsule_id')!=spec['capsule_id'] or m.get('split_id')!=spec['split_id'] or m.get('count')!=Q_COUNT
            or m.get('registered_classes')!=signal_train.native.CLASSES or ids.shape!=(Q_COUNT,) or ids.dtype.kind!='U' or len(set(ids.tolist()))!=Q_COUNT):
        if hasattr(iq,'_mmap'):iq._mmap.close()
        raise ValueError('Declared future Q capsule/split/old-six public identity differs')
    return m,ids,iq


def _target_rows(root,condition):
    if condition not in CONDITIONS:raise ValueError('Declared DURA condition required')
    with np.load(Path(root)/'targets'/condition/'predictions.npz',allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
    if set(a)!=TARGET_FIELDS:raise ValueError('Exact public query prediction fields required; truth/scene/RX inputs forbidden')
    return a


def run(c,device='cuda:0'):
    validate_config(c);out=Path(c['output_root'])
    if out.exists():raise FileExistsError('New immutable three-condition output root required')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;b=_backend(c,device)
    signal_eval._identity(b['model_identity'],dict(code_commit=SIGNAL_CODE,train_config=c['train_config']))
    snapshots=[(model,_snapshot_model(model)) for model in (b['reader'],b['model'],b['dura'][0])]
    def models_unchanged():
        for model,before in snapshots:_check_model(model,before)
    smoke=_smoke(b,device);models_unchanged()
    if smoke.get('status')!='PASS' or smoke.get('actual_F0_signal_DURA_checkpoints') is not True:raise ValueError('Actual three-model DURA smoke must pass before source/query access')
    pm,state,old_tau,pc=_reference(c['reference_prediction_root'],c['train_config']);identity=pm['signal_reference_identity']['model_identity']
    if b['model_identity']!=identity or b['reader_identity']!=identity['parent_F0_identity']:raise ValueError('Actual original F0/P1 branch differs; P0 checkpoint excluded')
    expected=identity['dura_identity'];actual=b['dura_identity']
    if any(actual.get(k)!=v or expected.get(k)!=v for k,v in dict(deployment_id='dura_v1',parameters=266951,epoch=20,updates=4940).items()) or any(actual.get(k)!=expected[k] for k in ('checkpoint','arm','training_commit','reuse','graph','native','precision','TF32') if k in expected):raise ValueError('All three tests require the same frozen DURA v1 computation')
    frozen=deepcopy(state);out.mkdir(parents=True,exist_ok=False);_write(out/'resolved_config.json',c);_write(out/'smoke.json',smoke);(out/'models').mkdir()
    _npz(out/'models/density.npz',state);_write(out/'models/complete.json',dict(schema=SCHEMA,status=MODELS_STATUS,run_id=RUN_ID,code_commit=c['code_commit'],
        models_frozen=True,models_refitted=False,mode_order=core.MODE_ORDER,source_V_fit=False,novel_fit=False,target_fit=False))
    def check():
        models_unchanged()
        if not _same(state,frozen):raise ValueError('Source density state changed during prediction')
    tau=np.empty((4,6),np.float64);tau[:2]=old_tau;methods={METHODS[j]:deepcopy(pc['methods'][reference.METHODS[j]]) for j in (0,1)}
    def source_scores(a):
        raw=core.raw_scores(state,a['signal_features'],a['closed'])
        if not np.allclose(raw[:,:2],a['raw_scores'][:,:2],rtol=1e-12,atol=1e-9) or not np.array_equal(a['scores'][:,:2],a['raw_scores'][:,:2]-old_tau[:,a['closed']].T):raise ValueError('Exact pooled source controls do not reconstruct')
        raw[:,:2]=a['raw_scores'][:,:2];return raw
    V=_rows(c['reference_prediction_root'],'known');raw=source_scores(V);check()
    for j in (2,3):
        calibrated,report=risk.calibrate_known(V['ids'],V['y'],V['closed'],V['baseline_closed'],raw[:,j],V['rx_i'],V['scenario']);tau[j]=calibrated['conditional_thresholds'];methods[METHODS[j]]=report;check()
    _npz(out/'thresholds.npz',dict(conditional=tau));(out/'calibration').mkdir()
    _write(out/'calibration/complete.json',dict(schema=SCHEMA,status=CALIBRATION_STATUS,run_id=RUN_ID,code_commit=c['code_commit'],method_names=list(METHODS),primary_method=PRIMARY_METHOD,
        known_count=COUNTS['V'],group_count=42,hard_names=list(risk.HARD_NAMES),diagnostic_names=list(risk.DIAGNOSTIC_NAMES),methods=methods,conditional_thresholds=tau.tolist(),
        known_only=True,unknown_inputs_used=False,target_reads=0,source_V_neural_fit=False,source_V_density_fit=False,source_V_threshold_fit=True,shared_thresholds_for_all_conditions=True))
    def source_prediction(a,raw):
        result=dict(a,raw_scores=raw,scores=raw-tau[:,a['closed']].T)
        if not np.array_equal(result['scores'][:,:2],a['scores'][:,:2]):raise ValueError('Pooled source operating points changed')
        return result
    _npz(out/'known_predictions.npz',source_prediction(V,raw));frozen_tau=tau.copy();check();print(CALIBRATION_STATUS,flush=True)
    N=_rows(c['reference_prediction_root'],'novel')
    if set(V['ids'].tolist())&set(N['ids'].tolist()):raise ValueError('Known and novel source records overlap')
    _npz(out/'novel_predictions.npz',source_prediction(N,source_scores(N)));check();public_manifests={};source_ids=set(V['ids'].tolist())|set(N['ids'].tolist())
    for name in CONDITIONS:
        iq=None
        try:
            manifest,ids,iq=_public(c,name)
            if source_ids&set(ids.tolist()):raise ValueError('Source and future query roles overlap')
            blocks=[]
            for start in range(0,len(ids),128):
                a=_current(b,iq[start:start+128],device);scores=core.raw_scores(state,a['signal_features'],a['closed'])
                blocks.append(dict(a,raw_scores=scores,scores=scores-tau[:,a['closed']].T));check()
                print('CLASS_DENSITY_Q_PROGRESS',name,start+len(a['closed']),len(ids),flush=True)
            result={k:np.concatenate([a[k] for a in blocks]) for k in blocks[0]};result['ids']=ids.copy();folder=out/'targets'/name;folder.mkdir(parents=True)
            _npz(folder/'predictions.npz',result);public_manifests[name]=manifest
        finally:
            if hasattr(iq,'_mmap'):iq._mmap.close()
    check()
    if not np.array_equal(tau,frozen_tau):raise ValueError('Shared source thresholds changed after query/novel access')
    done=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c,method_names=list(METHODS),primary_method=PRIMARY_METHOD,
        classification_path='P1',identity_baseline_path='same_DURA_output_P1',score_threshold=0.,known_count=COUNTS['V'],novel_count=COUNTS['novel'],
        source_known_count=COUNTS['V'],source_novel_count=COUNTS['novel'],Q_counts={name:Q_COUNT for name in CONDITIONS},Q_manifests=public_manifests,
        reference_prediction_identity=pm,model_identity=identity,parent_F0_identity=identity['parent_F0_identity'],dura_identity=identity['dura_identity'],loaded_dura_identity=b['dura_identity'],
        source_only=True,source_IQ_reads=0,source_L_reads=0,source_OE_reads=0,original_U_s_reads=0,target_U_reads=0,private_truth_reads=0,source_novel_truth_reads=0,target_truth_reads=0,
        formal_target_accessed=False,neural_updates=0,native_updates=0,density_fit_updates=0,source_V_threshold_fit=True,
        real_signal_forward_rows=2*Q_COUNT,real_F_forward_rows=2*Q_COUNT,G_forward_rows=2*Q_COUNT,synthetic_signal_forward_rows=2,synthetic_F_forward_rows=2,G_synthetic_forward_rows=2,
        models_frozen_before_V=True,all_thresholds_frozen_before_novel_and_Q=True,all_three_conditions_complete=True,all_conditions_through_DURA=True,
        source_states_unchanged=True,model_parameters_buffers_unchanged=True,controls_reused_exactly=True,identity_predictions_unchanged=True,
        source_bank_deployed=False,per_record_source_rows_deployed=0,final_phase3_complete=False,full_goal_status='ACTIVE',A_N=None,H_OW=None)
    _write(out/'complete.json',done);print(STATUS,flush=True);return done


evaluate=run


def main(argv=None):
    print(json.dumps(dict(event='CLASS_DENSITY_PROCESS',pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv)),flush=True)
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args(argv);run(_read(a.config),a.device)


if __name__=='__main__':main()
