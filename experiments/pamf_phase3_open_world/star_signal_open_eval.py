"""Frozen source-P1 signal branch and two exact frozen cross-set controls."""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import re
import sys
import time
import numpy as np
import torch
from . import star_crossset_metric_eval as parent
from . import star_representation_cache as IQ
from . import star_class_risk as risk

RUN_ID='20261030-phase3-star-signal-open-s2026092701-r03'
SCHEMA='pamf_phase3_star_signal_open_prediction_v1'
STATUS='STAR_SIGNAL_OPEN_PREDICTIONS_COMPLETE'
DENSITY_STATUS='STAR_SIGNAL_OPEN_DENSITIES_FROZEN'
CALIBRATION_STATUS='STAR_SIGNAL_OPEN_THRESHOLDS_FROZEN'
ARMS=('E_SIGNAL',)
PRIMARY_ARM=ARMS[0]
METHODS=('frozen_known_support','signal_known_support','frozen_density_ratio','signal_density_ratio')
PRIMARY_METHOD=METHODS[1]
REFERENCE_RUN=parent.RUN_ID
REFERENCE_CODE='fb016284e6c1f4715da4c690e2333efce89103b5'
FIELDS={'run_id','code_commit','arm','train_config','IQ_root','feature_root','reference_prediction_root','output_root','batch_size'}
COUNTS=dict(L=6300,OE=6134,V=27000,novel=3150)
KNOWN_FIELDS=parent.KNOWN_FIELDS|{'signal_features'}
NOVEL_FIELDS=parent.NOVEL_FIELDS|{'signal_features'}
_read,_write,_npz,_same=parent._read,parent._write,parent._npz,parent._same
_snapshot,_unchanged=parent._snapshot,parent._unchanged
_core=parent._core


def _train():
    from . import star_signal_open_train
    return star_signal_open_train


def validate_config(c):
    if (not isinstance(c,dict) or set(c)!=FIELDS or c.get('run_id')!=RUN_ID or c.get('arm') not in ARMS
            or re.fullmatch('[0-9a-f]{40}',str(c.get('code_commit'))) is None or not isinstance(c.get('train_config'),dict)
            or type(c.get('batch_size')) is not int or c['batch_size']!=128
            or any(not isinstance(c.get(k),str) or not c[k].strip() for k in ('IQ_root','feature_root','reference_prediction_root','output_root'))):
        raise ValueError('Exact nine-field source signal evaluation config required; no truth/target')
    t=_train().validate_config(c['train_config'])
    if any(t.get(k)!=c[k] for k in ('run_id','code_commit','arm','IQ_root','feature_root')) or t.get('control_root')!=c['reference_prediction_root']:
        raise ValueError('Actual trained signal row, IQ, feature and frozen-control inputs must agree')
    out=Path(c['output_root']).resolve()
    if any(out.is_relative_to(Path(p).resolve()) or Path(p).resolve().is_relative_to(out) for p in (c['IQ_root'],c['feature_root'],c['reference_prediction_root'],t['output_root'])):
        raise ValueError('Immutable signal evaluation output must be separate from every input')
    return c


def _identity(i,c):
    expected=dict(run_id=RUN_ID,code_commit=c['code_commit'],arm=PRIMARY_ARM,open_signal_updates=2220,native_updates=0,original_parent_updates=11100,
        signal_feature_cumulative_updates=13320,signal_parameters=334721,signal_feature_parameters=232033,metric_parameters=102688,
        original_F0_parameters=232993,combined_active_parameters=567714,identity_head_replaced=False,input_path='cached_P1_IQ_only')
    if any(i.get(k)!=v for k,v in expected.items()) or Path(i.get('checkpoint','')).resolve()!=(Path(c['train_config']['output_root'])/'final.pt').resolve():
        raise ValueError('Actual frozen 2220-step headless signal checkpoint required')
    p=i.get('parent_F0_identity',{});d=i.get('dura_identity',{})
    if (any(p.get(k)!=v for k,v in dict(model='original_F0_E50',parameters=232993,epoch=50,updates=11100,model_seed=2026092701).items())
            or any(d.get(k)!=v for k,v in dict(deployment_id='dura_v1',parameters=266951,epoch=20,updates=4940).items())):
        raise ValueError('Original F0/DURA source ancestry must remain unchanged')
    return p


def _backend(c,device):return _train().load_trained_model(c['train_config'],Path(c['train_config']['output_root'])/'final.pt',device)
def _load_role(root,role):return IQ.load_role(Path(root)/role,role)
def _close_role(a):IQ.close_role(a)


def _project(model,values,device):
    result=[]
    for start in range(0,len(values),128):
        batch=np.asarray(values[start:start+128])
        if batch.dtype!=np.float32 or batch.shape!=(len(batch),2,256) or not np.isfinite(batch).all():raise ValueError('Finite cached P1 FP32 full256 IQ required')
        with torch.no_grad(),torch.autocast(device_type=torch.device(device).type,enabled=False):
            z=model(torch.tensor(batch.tolist(),dtype=torch.float32,device=device))
        a=np.asarray(z.detach().cpu().tolist(),np.float32)
        if a.shape!=(len(batch),160) or not np.isfinite(a).all() or not np.allclose(np.linalg.norm(a.astype(np.float64),axis=1),1.,rtol=0,atol=2e-6):
            raise ValueError('Actual signal checkpoint must return same-row FP32 unit160')
        result.append(a)
    return np.concatenate(result)


def _smoke(model,device):
    from .star_open_feature_export import _synthetic
    values=np.asarray(_synthetic(device).detach().cpu().tolist(),np.float32)
    if _project(model,values,device).shape!=(2,160):raise ValueError('Actual signal checkpoint synthetic IQ smoke failed')
    return dict(status='PASS',actual_checkpoint=True,synthetic_signal_forward_rows=2,synthetic_input='G10_P0_interface_only',
        real_signal_forward_rows=0,real_F_forward_rows=0,synthetic_F_forward_rows=0,G_forward_rows=0,source_role_reads=0,target_reads=0,optimizer_updates=0)


def _iq_role(c,role,identity):
    a=_load_role(c['IQ_root'],role);m=a['manifest']
    if (m.get('parent_F0_identity')!=identity['parent_F0_identity'] or m.get('dura_identity')!=identity['dura_identity']
            or m.get('role')!=role or m.get('count')!=COUNTS[role] or m.get('source_only') is not True or m.get('restored_input_path')!='P1'):
        _close_role(a);raise ValueError('Only matching original source G10 restored P1 IQ roles may be read')
    return a


def _reference(root,identity,source):
    root=Path(root);m=_read(root/'complete.json')
    expected=dict(run_id=REFERENCE_RUN,code_commit=REFERENCE_CODE,schema=parent.SCHEMA,status=parent.STATUS,arm='S_SEP_TOPO',method_names=list(parent.METHODS),
        primary_method=parent.PRIMARY_METHOD,classification_path='P1',source_only=True,private_truth_reads=0,target_reads=0,identity_predictions_unchanged=True,
        models_frozen_before_V=True,all_thresholds_frozen_before_novel=True,source_states_unchanged=True,model_parameters_buffers_unchanged=True,
        parent_F0_identity=identity['parent_F0_identity'],dura_identity=identity['dura_identity'],source_feature_manifest=source,learned_known_statistics_shared=True)
    if any(m.get(k)!=v for k,v in expected.items()):raise ValueError('Exact frozen cross-set S_SEP_TOPO controls with shared source ancestry required')
    parent._identity(m['model_identity'],m['config'])
    states={}
    for new,old,method in ((METHODS[0],parent.METHODS[1],'known_only'),(METHODS[2],parent.METHODS[2],'density_ratio')):
        with np.load(root/'models'/(old+'.npz'),allow_pickle=False) as z:states[new]={k:z[k].copy() for k in z.files}
        _core().validate_state(states[new],method)
        if states[new]['whitener'].shape!=(160,160):raise ValueError('Frozen cross-set controls must remain D160')
    if any(not np.array_equal(v,states[METHODS[2]][k]) for k,v in states[METHODS[0]].items()):raise ValueError('Frozen controls must share their original known statistics')
    with np.load(root/'thresholds.npz',allow_pickle=False) as z:tau=z['conditional'][[1,2]].copy();glob=z['global'][[1,2]].copy()
    cal=_read(root/'calibration/complete.json')
    if any(cal.get(k)!=v for k,v in dict(run_id=REFERENCE_RUN,code_commit=REFERENCE_CODE,status=parent.CALIBRATION_STATUS,known_only=True,unknown_inputs_used=False).items()):
        raise ValueError('Original frozen known-only cross-set calibration required')
    for j,name in enumerate(parent.METHODS[1:]):
        if cal['methods'][name]['conditional_thresholds']!=tau[j].tolist():raise ValueError('Original frozen control thresholds differ')
    return m,states,tau,glob,cal


def _rows(root,role):
    fields=parent.KNOWN_FIELDS if role=='known' else parent.NOVEL_FIELDS
    with np.load(Path(root)/(role+'_predictions.npz'),allow_pickle=False) as z:
        if set(z.files)!=fields:raise ValueError('Exact parent public prediction fields required, without private novel labels')
        a={k:z[k].copy() for k in z.files}
    _validate_rows(a,role,3,False)
    return a


def _validate_rows(a,role,columns,signal):
    n=COUNTS['V' if role=='known' else 'novel'];fields=(KNOWN_FIELDS if role=='known' else NOVEL_FIELDS) if signal else (parent.KNOWN_FIELDS if role=='known' else parent.NOVEL_FIELDS)
    if set(a)!=fields or a['ids'].shape!=(n,) or a['ids'].dtype.kind!='U' or len(set(a['ids'].tolist()))!=n:raise ValueError('Exact complete role IDs and public fields required')
    for key,d,typ in (('scores',columns,np.float64),('raw_scores',columns,np.float64),('default_logits',6,np.float32),('contribution_features',640,np.float32),('unit_features',160,np.float32),('learned_features',160,np.float32)):
        if a[key].shape!=(n,d) or a[key].dtype!=typ or not np.isfinite(a[key]).all():raise ValueError('Source prediction shape/dtype/finite mismatch: '+key)
    for key in ('learned_features','unit_features')+ (('signal_features',) if signal else ()):
        if a[key].shape!=(n,160) or a[key].dtype!=np.float32 or not np.isfinite(a[key]).all() or not np.allclose(np.linalg.norm(a[key].astype(np.float64),axis=1),1.,rtol=0,atol=2e-6):raise ValueError('Same-row unit160 features required')
    for key,values in (('closed',range(6)),('rx_i',(1,3,4,6,8)),('day_i',(1,2,3))):
        if a[key].shape!=(n,) or a[key].dtype!=np.int64 or not np.isin(a[key],values).all():raise ValueError('Invalid public source identity/domain')
    if not np.array_equal(a['closed'],a['default_logits'].argmax(1)):raise ValueError('Original F0 default-head identity must be preserved')
    if role=='known' and (any(a[k].shape!=(n,) or a[k].dtype!=np.int64 or not np.isin(a[k],range(6)).all() for k in ('y','baseline_closed'))
            or a['scenario'].shape!=(n,) or a['scenario'].dtype.kind!='U' or not np.isin(a['scenario'],parent.reference.SCENES).all()):raise ValueError('Visible known labels/scenarios/original baseline required')


def _align(iq,saved,role):
    if not np.array_equal(iq['ids'],saved['ids']):raise ValueError('IQ and frozen features/predictions physical IDs differ')
    for k,v in iq['metadata'].items():
        if k=='domain':
            if not np.array_equal(v,np.searchsorted([1,3,4,6,8],saved['rx_i'])*3+saved['day_i']-1):raise ValueError('IQ public domain differs')
        elif not np.array_equal(v,saved[k]):raise ValueError('IQ and saved public metadata differ: '+k)
    if role=='V' and not np.array_equal(iq['baseline_closed'][:,1],saved['baseline_closed']):raise ValueError('Original same-P1 F0 cache baseline differs')


def _signal_scores(z,states):
    t=time.perf_counter();raw=np.column_stack((_core().score(states[METHODS[1]],z,'known_only'),_core().score(states[METHODS[3]],z,'density_ratio')))
    if raw.shape!=(len(z),2) or raw.dtype!=np.float64 or not np.isfinite(raw).all():raise ValueError('Two finite same-signal density scores required')
    return raw,time.perf_counter()-t


def evaluate(c,device='cuda:0'):
    validate_config(c);out=Path(c['output_root'])
    if out.exists():raise FileExistsError('New immutable signal evaluation root required')
    started=time.perf_counter();model,identity=_backend(c,device);_identity(identity,c)
    if sum(p.numel() for p in model.parameters())!=334721 or any(m.training for m in model.modules()) or any(p.requires_grad or p.dtype!=torch.float32 for p in model.parameters()):raise ValueError('Actual frozen FP32 headless signal+MLP checkpoint required')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    before=_snapshot(model);smoke=_smoke(model,device);_unchanged(model,before)
    if smoke.get('status')!='PASS':raise ValueError('Actual synthetic signal checkpoint smoke must precede source roles')
    out.mkdir(parents=True);_write(out/'smoke.json',smoke);_write(out/'resolved_config.json',c)
    feature=parent._train().load_role(c['feature_root'],'L');fm=feature['manifest'];source=fm['source_feature_manifest']
    pm,controls,old_tau,old_global,pc=_reference(c['reference_prediction_root'],identity,source)
    if pm['feature_manifest']!=fm:raise ValueError('Training features and fixed cross-set control must use the same prepared source handle')
    seen=set();manifests={};fit_seconds=0.
    for role in ('L','OE'):
        if role=='OE':feature=parent._train().load_role(c['feature_root'],role)
        if feature['manifest']!=fm:raise ValueError('Both source teacher roles must share the fixed feature handle')
        iq=None
        try:
            iq=_iq_role(c,role,identity);_align(iq,dict(ids=feature['ids'],**feature['metadata']),role)
            if iq['manifest']!=fm['source_manifests'][role]:raise ValueError('IQ and frozen teacher role handles differ')
            if seen&set(iq['ids'].tolist()):raise ValueError('Source fitting roles overlap')
            seen.update(iq['ids'].tolist());manifests[role]=iq['manifest'];z=_project(model,iq['restored'],device);_unchanged(model,before);meta=iq['metadata'];t=time.perf_counter()
            if role=='L':known,kdiag=_core().fit_known(z,meta['y'],meta['rx_i'],meta['day_i']);known_before=deepcopy(known)
            else:ratio,odiag=_core().fit_background(known,z,meta['rx_i'],meta['day_i'])
            fit_seconds+=time.perf_counter()-t
        finally:
            if iq is not None:_close_role(iq)
    if not _same(known,known_before) or any(not np.array_equal(v,ratio[k]) for k,v in known.items()):raise ValueError('Both new readouts must share exact source-L statistics')
    states={METHODS[0]:controls[METHODS[0]],METHODS[1]:known,METHODS[2]:controls[METHODS[2]],METHODS[3]:ratio};frozen=deepcopy(states)
    def check():
        _unchanged(model,before)
        if not _same(states,frozen):raise ValueError('Frozen signal/control/known/background aggregate changed')
    models=out/'models';models.mkdir()
    for name,state in states.items():_npz(models/(name+'.npz'),state)
    _write(models/'complete.json',dict(status=DENSITY_STATUS,run_id=RUN_ID,code_commit=c['code_commit'],arm=PRIMARY_ARM,method_names=list(METHODS),controls_reused=True,known=kdiag,background=odiag,signal_known_statistics_shared=True,source_V_fit=False,novel_fit=False))
    V=_rows(c['reference_prediction_root'],'known');iq=None
    try:
        iq=_iq_role(c,'V',identity);_align(iq,V,'V')
        if seen&set(iq['ids'].tolist()):raise ValueError('V overlaps fitting records')
        seen.update(iq['ids'].tolist());manifests['V']=iq['manifest'];V['signal_features']=_project(model,iq['restored'],device);check()
    finally:
        if iq is not None:_close_role(iq)
    raw,cpu_known=_signal_scores(V['signal_features'],states);tau=np.empty((4,6),np.float64);glob=np.empty(4,np.float64);methods={};global_controls={}
    for j,oldindex in ((0,0),(2,1)):
        tau[j]=old_tau[oldindex];glob[j]=old_global[oldindex];name=parent.METHODS[oldindex+1]
        methods[METHODS[j]]=deepcopy(pc['methods'][name]);global_controls[METHODS[j]]=deepcopy(pc['global_controls'][name])
    for k,j in enumerate((1,3)):
        s,r=risk.calibrate_known(V['ids'],V['y'],V['closed'],V['baseline_closed'],raw[:,k],V['rx_i'],V['scenario'])
        tau[j]=s['conditional_thresholds'];glob[j]=s['global_threshold'];methods[METHODS[j]]=r['methods']['predicted_class'];global_controls[METHODS[j]]=r['methods']['global_control']
    check();_npz(out/'thresholds.npz',dict(conditional=tau,**{'global':glob}));(out/'calibration').mkdir()
    _write(out/'calibration/complete.json',dict(schema=SCHEMA,status=CALIBRATION_STATUS,run_id=RUN_ID,code_commit=c['code_commit'],arm=PRIMARY_ARM,method_names=list(METHODS),primary_method=PRIMARY_METHOD,
        known_count=COUNTS['V'],group_count=42,known_only=True,unknown_inputs_used=False,source_V_model_fit=False,source_V_threshold_fit=True,target_reads=0,conditional_thresholds=tau.tolist(),global_thresholds=glob.tolist(),methods=methods,global_controls=global_controls,control_calibration_reused=True))
    def combine(a,raw):
        old=a['scores'][:,[1,2]].copy();a['raw_scores']=np.column_stack((a['raw_scores'][:,1],raw[:,0],a['raw_scores'][:,2],raw[:,1]));a['scores']=a['raw_scores']-tau[:,a['closed']].T
        if not np.array_equal(a['scores'][:,[0,2]],old):raise ValueError('Frozen known/ratio controls changed')
    combine(V,raw);_npz(out/'known_predictions.npz',V);check();frozen_tau=tau.copy();print(CALIBRATION_STATUS,flush=True)
    N=_rows(c['reference_prediction_root'],'novel');iq=None
    try:
        iq=_iq_role(c,'novel',identity);_align(iq,N,'novel')
        if seen&set(iq['ids'].tolist()):raise ValueError('Novel overlaps fitting/calibration records')
        manifests['novel']=iq['manifest'];N['signal_features']=_project(model,iq['restored'],device);check()
    finally:
        if iq is not None:_close_role(iq)
    raw,cpu_novel=_signal_scores(N['signal_features'],states);combine(N,raw);check()
    if not np.array_equal(tau,frozen_tau):raise ValueError('Thresholds changed after novel access')
    _npz(out/'novel_predictions.npz',N)
    done=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=c['code_commit'],arm=PRIMARY_ARM,primary_arm=PRIMARY_ARM,config=c,method_names=list(METHODS),primary_method=PRIMARY_METHOD,
        classification_path='P1',score_threshold=0.,model_identity=identity,parent_F0_identity=identity['parent_F0_identity'],dura_identity=identity['dura_identity'],feature_manifest=fm,source_feature_manifest=source,IQ_manifests=manifests,
        reference_run_id=REFERENCE_RUN,reference_code_commit=REFERENCE_CODE,reference_prediction_identity=pm,reference_prediction_root=c['reference_prediction_root'],known_count=COUNTS['V'],novel_count=COUNTS['novel'],source_L_count=COUNTS['L'],source_OE_count=COUNTS['OE'],
        source_only=True,private_truth_reads=0,target_reads=0,original_U_s_reads=0,source_novel_fit_reads=0,real_F_forward_rows=0,synthetic_F_forward_rows=0,G_forward_rows=0,G_checkpoint_loaded=False,
        real_signal_forward_rows=sum(COUNTS.values()),synthetic_signal_forward_rows=2,signal_forward_rows_by_role=dict(COUNTS),eval_neural_updates=0,native_updates=0,open_signal_updates=2220,
        default_head_replaced=False,identity_predictions_unchanged=True,controls_reused_exactly=True,control_refit=False,control_recalibrated=False,models_frozen_before_V=True,all_thresholds_frozen_before_novel=True,
        source_states_unchanged=True,model_parameters_buffers_unchanged=True,source_bank_deployed=False,per_record_source_rows_deployed=0,source_V_model_fit=False,source_V_threshold_fit=True,embedding_training_uses_OE=True,
        signal_known_statistics_shared=True,same_projection_for_signal_methods=True,signal_parameters=334721,signal_feature_parameters=232033,metric_parameters=102688,original_F0_parameters=232993,combined_active_parameters=567714,
        source_aggregate_bytes_by_method={k:sum(a.nbytes for a in s.values()) for k,s in states.items()},threshold_bytes=int(tau.nbytes),cpu_fit_seconds=fit_seconds,cpu_readout_seconds=cpu_known+cpu_novel,cpu_readout_rows=COUNTS['V']+COUNTS['novel'],cpu_readout_score_rows=2*(COUNTS['V']+COUNTS['novel']),
        received_P0_values_read=0,channel_regenerations=0,second_RMS_applied=False,final_phase3_complete=False,full_goal_status='ACTIVE',A_N=None,H_OW=None,seconds=time.perf_counter()-started)
    check();_write(out/'complete.json',done);print(STATUS,flush=True);return done


def main():
    print(json.dumps(dict(event='STAR_SIGNAL_OPEN_EVAL_PROCESS',pid=os.getpid(),cwd=os.getcwd(),argv=list(sys.argv))),flush=True)
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args();evaluate(_read(a.config),a.device)


if __name__=='__main__':main()
