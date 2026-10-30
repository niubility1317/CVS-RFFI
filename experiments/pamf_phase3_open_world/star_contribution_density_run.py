"""Frozen original-F0 four-contribution density with an exactly reused control."""
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

from . import star_consensus_run as base
from . import star_density_run as parent
from . import star_class_risk as risk
from .star_protected_open_cache import load_features

RUN_ID='20261030-phase3-star-contribution-density-s2026092701-r01'
SCHEMA='pamf_phase3_star_contribution_density_prediction_v1'
STATUS='STAR_CONTRIBUTION_DENSITY_PREDICTIONS_COMPLETE'
CALIBRATION_STATUS='STAR_CONTRIBUTION_DENSITY_THRESHOLDS_FROZEN'
PARENT_RUN=parent.RUN_ID
PARENT_CODE='b8163a03e8d3d299127c7914272a2d866f92c89a'
METHODS=('fused160_density_ratio','contribution640_density_ratio')
PRIMARY_METHOD=METHODS[1]
FIELDS=parent.FIELDS.copy()
COUNTS=dict(L=6300,OE=6134,V=27000,novel=3150)
RECEIVERS,SCENES=parent.RECEIVERS,parent.SCENES
KNOWN_FIELDS=parent.KNOWN_FIELDS|{'contribution_features'}
NOVEL_FIELDS=parent.NOVEL_FIELDS|{'contribution_features'}
_backend,_snapshot,_check_models=base._backend,base._snapshot,base._check_models
_read,_write,_npz,_same=parent._read,parent._write,parent._npz,parent._same
_load_role,close_role=base._role,base.close_role


def _core():return parent._core()


def validate_config(c):
    if (not isinstance(c,dict) or set(c)!=FIELDS or c.get('run_id')!=RUN_ID
            or re.fullmatch('[0-9a-f]{40}',str(c.get('code_commit'))) is None
            or not isinstance(c.get('source_config'),dict) or set(c['source_config'])!=parent.SOURCE_FIELDS
            or type(c.get('batch_size')) is not int or c['batch_size']!=128
            or any(not isinstance(c.get(k),str) or not c[k].strip() for k in ('feature_root','reference_prediction_root','output_root'))):
        raise ValueError('Exact seven-field source contribution-density configuration required; no truth or target')
    s=c['source_config'];base.runner.validate_config('calibrate',dict(s,retention=.99))
    if (s['run_id']!=RUN_ID or s['row_id']!='star-contribution-density-s2026092701' or s['code_commit']!=c['code_commit']
            or s['output_root']!=c['output_root'] or s['model_seed']!=2026092701 or s['batch_size']!=128 or s['registered_classes']!=base.native.CLASSES):
        raise ValueError('Fixed original-six F0 and this independent source candidate required')
    out=Path(c['output_root']).resolve()
    if any(out.is_relative_to(Path(p).resolve()) or Path(p).resolve().is_relative_to(out)
            for p in (c['feature_root'],c['reference_prediction_root'],s['source_output'],s['source_cache'])):
        raise ValueError('Output must not overlap immutable source inputs')
    return c


def _encode(reader,values,device):
    """Original hooks only: one features/default-head pass, no installed parts."""
    import torch
    from .source_open_export import _tensor
    x=values.to(device=device,dtype=torch.float32) if torch.is_tensor(values) else _tensor(values,device)
    if x.ndim!=3 or tuple(x.shape[1:])!=(2,256) or not len(x) or not bool(torch.isfinite(x).all()):
        raise ValueError('Finite FP32 cached P1 observations required')
    if any(m.training for m in reader.modules()) or any(p.requires_grad for p in reader.parameters()):
        raise ValueError('The original F0 reader must be frozen in evaluation mode')
    head=reader.core.id_backbone.cls_head;captured={};handles=[];n=len(x)
    def record(name,value):
        if name in captured:raise ValueError('Native contribution captured more than once: '+name)
        captured[name]=value
    def matrix(value,shape):
        if not torch.is_tensor(value) or tuple(value.shape)!=shape or value.dtype!=torch.float32 or not bool(torch.isfinite(value).all()):
            raise ValueError('Finite original FP32 same-pass native contribution required')
        return value
    try:
        targets=(head.base_norm,head.pa_norm,reader.response_projection,reader.response,reader.sequence_pool)
        for name,module in zip(('identity','physical','reference_projection','reference29','signature'),targets):
            handles.append(module.register_forward_hook(lambda m,args,value,key=name:record(key,value)))
        with torch.no_grad(),torch.autocast(device_type=x.device.type,enabled=False):
            f=matrix(reader.features(x),(n,160))
            if set(captured)!={'identity','physical','reference_projection','reference29','signature'}:
                raise ValueError('All same-pass native branch contributions must be captured')
            for key in ('identity','physical','reference_projection','signature'):matrix(captured[key],(n,160))
            reference=matrix(captured['reference29'],(n,29))
            if bool(reference[:,27].ne(0).any()):raise ValueError('Original NoExplicitCFO coordinate 27 must remain zero')
            contributions=torch.stack((captured['identity'],head.gain.tanh()*captured['physical'],
                reader.response_gain.tanh()*captured['reference_projection'],captured['signature']),1)
            logits=matrix(reader.classify_features(f),(n,6))
        return tuple(np.asarray(a.detach().cpu().tolist(),np.float32) for a in (f,contributions,logits))
    finally:
        for handle in handles:handle.remove()


def _smoke(backend,device):
    from .star_open_feature_export import _synthetic
    f,c,z=_encode(backend['reader'],_synthetic(device),device)
    x=_core().contribution_unit(c)
    if f.shape!=(2,160) or z.shape!=(2,6) or x.shape!=(2,640):raise ValueError('Actual original-F0 four-contribution smoke failed')
    return dict(status='PASS',actual_F0_checkpoint=True,contributions_checked=True,synthetic_F_forward_rows=2,
        synthetic_G10_received_rows=2,G_forward_rows=0,real_source_rows=0,target_reads=0,optimizer_updates=0)


def _feature(root,role,identity):
    a=load_features(root,role);n=COUNTS[role];meta=a['metadata'];m=a['manifest']
    if (m.get('parent_F0_identity')!=identity or set(meta)!=({'y','rx_i','day_i','domain'} if role=='L' else {'rx_i','day_i','domain'})
            or a['ids'].shape!=(n,) or len(set(a['ids'].tolist()))!=n
            or any(v.shape!=(n,) or v.dtype!=np.int64 for v in meta.values())
            or a['contributions'].shape!=(n,4,160) or a['contributions'].dtype!=np.float32 or not np.isfinite(a['contributions']).all()):
        raise ValueError('Exact original F0 L/OE contribution features and public source metadata required')
    return a


def _parent(root,identity,fm):
    root=Path(root);m=_read(root/'complete.json')
    expected=dict(run_id=PARENT_RUN,code_commit=PARENT_CODE,schema=parent.SCHEMA,status=parent.STATUS,model_identity=identity,
        source_feature_manifest=fm,dura_identity=fm['dura_identity'],method_names=list(parent.METHODS),primary_method='density_ratio',
        classification_path='P1',known_count=COUNTS['V'],novel_count=COUNTS['novel'],source_L_count=COUNTS['L'],source_OE_count=COUNTS['OE'],
        source_only=True,private_truth_reads=0,target_reads=0,neural_updates=0,identity_predictions_unchanged=True,
        models_frozen_before_V=True,all_thresholds_frozen_before_novel=True,source_states_unchanged=True,model_parameters_buffers_unchanged=True)
    if any(m.get(k)!=v for k,v in expected.items()):raise ValueError('Fixed completed original-F0 density parent and identical source feature handles required')
    with np.load(root/'models/density_ratio.npz',allow_pickle=False) as z:state={k:z[k].copy() for k in z.files}
    _core().validate_state(state,'density_ratio')
    if state['whitener'].shape!=(160,160):raise ValueError('The fixed control must remain the original 160-dimensional density')
    with np.load(root/'thresholds.npz',allow_pickle=False) as z:
        if set(z.files)!={'conditional','global'}:raise ValueError('Exact parent density thresholds required')
        tau=z['conditional'][1].copy();global_tau=float(z['global'][1])
    cal=_read(root/'calibration/complete.json')
    if (cal.get('run_id')!=PARENT_RUN or cal.get('code_commit')!=PARENT_CODE or cal.get('status')!=parent.CALIBRATION_STATUS
            or cal.get('known_only') is not True or cal.get('unknown_inputs_used') is not False
            or cal['methods']['density_ratio']['conditional_thresholds']!=tau.tolist()):
        raise ValueError('Use the exact already-frozen density-ratio control calibration; never fit it again')
    return m,state,tau,global_tau,cal


def _control_rows(root,role):
    fields=parent.KNOWN_FIELDS if role=='known' else parent.NOVEL_FIELDS
    with np.load(Path(root)/(role+'_predictions.npz'),allow_pickle=False) as z:
        if set(z.files)!=fields:raise ValueError('Exact public frozen parent prediction fields required')
        a={k:z[k].copy() for k in z.files}
    parent.validate_rows(a,role)
    a['scores']=a['scores'][:,1].copy();a['raw_scores']=a['raw_scores'][:,1].copy()
    return a


def _predict(reader,data,state,control,device):
    for key in ('ids','rx_i','day_i'):
        value=data['ids'] if key=='ids' else data['metadata'][key]
        if not np.array_equal(value,control[key]):raise ValueError('Fresh P1 cache and frozen-control public IDs/domains differ')
    for key in ('y','scenario'):
        if key in control and not np.array_equal(data['metadata'][key],control[key]):raise ValueError('Known calibration metadata differ')
    if 'baseline_closed' in control and not np.array_equal(data['baseline_closed'][:,1],control['baseline_closed']):
        raise ValueError('Independent same-P1 original F0 baseline differs')
    parts=[];raw=[];cpu=0.
    for start in range(0,len(data['ids']),128):
        f,c,z=_encode(reader,data['restored'][start:start+128],device);end=start+len(f)
        if (not np.array_equal(z,control['default_logits'][start:end])
                or not np.array_equal(z.argmax(1),control['closed'][start:end])
                or not np.array_equal(parent.reference._unit(f),control['unit_features'][start:end])):
            raise ValueError('Fresh original F0 identity/logits/features differ from exact frozen control; no replacement or rerun')
        u=_core().contribution_unit(c);t=time.perf_counter();s=_core().score(state,u,'density_ratio');cpu+=time.perf_counter()-t
        parts.append(u);raw.append(s);print('STAR_CONTRIBUTION_DENSITY_PROGRESS',end,len(data['ids']),flush=True)
    result=dict(control);result['contribution_features']=np.concatenate(parts)
    result['raw_scores']=np.column_stack((control['raw_scores'],np.concatenate(raw)))
    return result,cpu


def run(c,device='cuda:0'):
    validate_config(c);out=Path(c['output_root'])
    if out.exists():raise FileExistsError('New immutable source contribution-density output required')
    started=time.perf_counter();backend,_=_backend(c['source_config'],device);identity=backend['reader_identity']
    parent.reference._identity(identity,c['source_config']);before=_snapshot(backend);smoke=_smoke(backend,device);_check_models(before)
    if smoke.get('status')!='PASS':raise ValueError('Actual original-F0 contribution checkpoint smoke must pass before source inputs')
    out.mkdir(parents=True);_write(out/'smoke.json',smoke);_write(out/'resolved_config.json',c)
    L=_feature(c['feature_root'],'L',identity);fm=L['manifest'];cache_root=Path(fm['config']['L_root']).parent
    pm,control_state,control_tau,control_global,pc=_parent(c['reference_prediction_root'],identity,fm)
    u=_core().contribution_unit(L['contributions']);meta=L['metadata'];t=time.perf_counter()
    state,kdiag=_core().fit_known(u,meta['y'],meta['rx_i'],meta['day_i'],dimension=640);fit_seconds=time.perf_counter()-t
    seen=set(L['ids'].tolist());del L,u
    OE=_feature(c['feature_root'],'OE',identity)
    if OE['manifest']!=fm or seen&set(OE['ids'].tolist()):raise ValueError('L/OE roles must share ancestry and be physically disjoint')
    seen.update(OE['ids'].tolist());u=_core().contribution_unit(OE['contributions']);meta=OE['metadata'];t=time.perf_counter()
    state,odiag=_core().fit_background(state,u,meta['rx_i'],meta['day_i']);fit_seconds+=time.perf_counter()-t;del OE,u
    states=dict(zip(METHODS,(control_state,state)));frozen=deepcopy(states)
    def check():
        _check_models(before)
        if not _same(states,frozen):raise ValueError('Frozen control or contribution-density aggregate changed')
    models=out/'models';models.mkdir()
    for name,value in states.items():_npz(models/(name+'.npz'),value)
    _write(models/'complete.json',dict(run_id=RUN_ID,code_commit=c['code_commit'],status='STAR_CONTRIBUTION_DENSITY_MODELS_FROZEN',
        method_names=list(METHODS),control_reused=True,known=kdiag,background=odiag,source_V_fit=False,novel_fit=False))
    V=novel=None
    try:
        V=_load_role(cache_root/'V','V',identity,fm['dura_identity']);known_control=_control_rows(c['reference_prediction_root'],'known')
        if seen&set(V['ids'].tolist()):raise ValueError('Source fitting and known calibration records overlap')
        seen.update(V['ids'].tolist());known,cpu_known=_predict(backend['reader'],V,state,known_control,device);check()
        new,report=risk.calibrate_known(known['ids'],known['y'],known['closed'],known['baseline_closed'],known['raw_scores'][:,1],known['rx_i'],known['scenario'])
        tau=np.stack((control_tau,new['conditional_thresholds']));global_tau=np.asarray([control_global,new['global_threshold']],np.float64)
        methods={METHODS[0]:deepcopy(pc['methods']['density_ratio']),METHODS[1]:report['methods']['predicted_class']}
        controls={METHODS[0]:deepcopy(pc['global_controls']['density_ratio']),METHODS[1]:report['methods']['global_control']}
        _npz(out/'thresholds.npz',dict(conditional=tau,**{'global':global_tau}));(out/'calibration').mkdir()
        _write(out/'calibration/complete.json',dict(schema=SCHEMA,status=CALIBRATION_STATUS,run_id=RUN_ID,code_commit=c['code_commit'],
            method_names=list(METHODS),primary_method=PRIMARY_METHOD,known_count=COUNTS['V'],group_count=42,known_only=True,unknown_inputs_used=False,
            source_V_model_fit=False,source_V_threshold_fit=True,target_reads=0,conditional_thresholds=tau.tolist(),global_thresholds=global_tau.tolist(),
            methods=methods,global_controls=controls,control_calibration_reused=True))
        known['scores']=known['raw_scores']-tau[:,known['closed']].T
        if not np.array_equal(known['scores'][:,0],known_control['scores']):raise ValueError('Frozen control known margins changed')
        _npz(out/'known_predictions.npz',known);close_role(V);V=None;check();frozen_tau=tau.copy();print(CALIBRATION_STATUS,flush=True)
        # Only now may the candidate or this run open the parent novel arrays.
        novel=_load_role(cache_root/'novel','novel',identity,fm['dura_identity']);novel_control=_control_rows(c['reference_prediction_root'],'novel')
        if seen&set(novel['ids'].tolist()):raise ValueError('Novel source records overlap fit/calibration')
        rows,cpu_novel=_predict(backend['reader'],novel,state,novel_control,device);check()
        rows['scores']=rows['raw_scores']-tau[:,rows['closed']].T
        if not np.array_equal(tau,frozen_tau) or not np.array_equal(rows['scores'][:,0],novel_control['scores']):raise ValueError('Frozen thresholds/control novel decisions changed')
        _npz(out/'novel_predictions.npz',rows);close_role(novel);novel=None
    finally:
        for data in (V,novel):
            if data is not None:close_role(data)
    done=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c,method_names=list(METHODS),primary_method=PRIMARY_METHOD,
        classification_path='P1',score_threshold=0.,model_identity=identity,dura_identity=fm['dura_identity'],source_feature_manifest=fm,
        reference_run_id=PARENT_RUN,reference_code_commit=PARENT_CODE,reference_prediction_identity=pm,reference_prediction_root=c['reference_prediction_root'],
        cache_root=str(cache_root),known_count=COUNTS['V'],novel_count=COUNTS['novel'],source_L_count=COUNTS['L'],source_OE_count=COUNTS['OE'],
        source_only=True,private_truth_reads=0,target_reads=0,original_U_s_reads=0,source_novel_fit_reads=0,OE_TX_or_K_used=False,
        real_F_forward_rows=COUNTS['V']+COUNTS['novel'],synthetic_F_forward_rows=2,G_forward_rows=0,G_checkpoint_loaded=False,
        source_L_forward_rows=0,source_OE_forward_rows=0,received_P0_values_read=0,neural_updates=0,native_updates=0,identity_head_updates=0,G_updates=0,
        default_head_replaced=False,open_parts_installed=False,identity_predictions_unchanged=True,control_reused_exactly=True,control_refit=False,control_recalibrated=False,
        models_frozen_before_V=True,all_thresholds_frozen_before_novel=True,source_states_unchanged=True,model_parameters_buffers_unchanged=True,
        source_bank_deployed=False,per_record_source_rows_deployed=0,source_V_model_fit=False,source_V_threshold_fit=True,
        source_aggregate_bytes_by_method={k:sum(a.nbytes for a in v.values()) for k,v in states.items()},threshold_bytes=int(tau.nbytes),
        cpu_fit_seconds=fit_seconds,cpu_readout_seconds=cpu_known+cpu_novel,cpu_readout_rows=COUNTS['V']+COUNTS['novel'],
        representation='four native branch contributions individually unit normalized, concatenated and globally unit normalized',
        dimensions=dict(zip(METHODS,(160,640))),dimension_dependent_covariance_volume_and_shrinkage=True,
        final_phase3_complete=False,full_goal_status='ACTIVE',A_N=None,H_OW=None,seconds=time.perf_counter()-started)
    _write(out/'complete.json',done);print(STATUS,flush=True);return done


def main():
    print(json.dumps(dict(event='STAR_CONTRIBUTION_DENSITY_PROCESS',pid=os.getpid(),cwd=os.getcwd(),argv=list(sys.argv))),flush=True)
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--device',default='cuda:0')
    a=p.parse_args();run(_read(a.config),a.device)


if __name__=='__main__':main()
