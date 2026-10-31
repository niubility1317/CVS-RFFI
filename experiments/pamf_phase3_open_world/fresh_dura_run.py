"""Frozen fresh-PCD before/after DURA; original DURA predictions are reused."""
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
from . import fresh_world_run as fresh
from . import star_raw_open_train as received
from . import star_density_core as density

RUN_ID='20261031-phase3-fresh-dura-s2026092701-r01'
REFERENCE_CODE='f4b5a0b80cafcf237c3ae9dac8ecd5b24c91bff7'
SCHEMA='pamf_phase3_fresh_dura_prediction_v1'
STATUS='FRESH_DURA_PREDICTIONS_COMPLETE'
MODELS_STATUS='FRESH_DURA_MODELS_FROZEN'
CALIBRATION_STATUS='FRESH_DURA_THRESHOLDS_INHERITED'
MODEL_NAMES=('E0','PCD')
METHODS=('E0_pooled_ratio','E0_predicted_ratio','PCD_pooled_ratio','PCD_predicted_ratio')
PRIMARY_METHOD=METHODS[-1]
PRIMARY_INDEX=3
SOURCE_CONDITIONS=('source_raw','source_DURA')
QUERY_CONDITIONS=('clean_DURA','star_DURA')
CONDITIONS=SOURCE_CONDITIONS+('clean_raw','clean_DURA','star_raw','star_DURA')
COUNTS=dict(fresh.COUNTS)
Q_COUNT=10238
FIELDS={'run_id','code_commit','reference_config','Q_conditions','output_root','batch_size'}
KNOWN_FIELDS,NOVEL_FIELDS,TARGET_FIELDS=fresh.KNOWN_FIELDS,fresh.NOVEL_FIELDS,fresh.TARGET_FIELDS
_read,_write,_same=fresh._read,fresh._write,fresh._same
_save=fresh.best._save


def validate_config(c):
    if (not isinstance(c,dict) or set(c)!=FIELDS or c.get('run_id')!=RUN_ID or re.fullmatch('[0-9a-f]{40}',str(c.get('code_commit'))) is None
            or type(c.get('batch_size')) is not int or c['batch_size']!=128 or not isinstance(c.get('output_root'),str) or not c['output_root'].strip()):
        raise ValueError('Exact frozen fresh-DURA fields required; no truth/U/role overrides')
    p=fresh.validate_config(c['reference_config'])
    if p['code_commit']!=REFERENCE_CODE or c['Q_conditions']!=p['Q_conditions']:raise ValueError('Original completed fresh f4b5 and identical future public handles required')
    source=p['source_control_config']['reference_config']['train_config'];out=Path(c['output_root']).resolve()
    roots=[p['output_root'],p['train_config']['output_root'],p['train_config']['U_public_root'],source['IQ_root'],source['output_root']]
    roots.extend(v['public_root'] for v in c['Q_conditions'].values())
    if any(out.is_relative_to(Path(v).resolve()) or Path(v).resolve().is_relative_to(out) for v in roots):raise ValueError('New immutable output overlaps existing model/data inputs')
    return c


def _reference(c):
    p=c['reference_config'];root=Path(p['output_root']);m=_read(root/'complete.json')
    required=dict(schema=fresh.SCHEMA,status=fresh.STATUS,run_id=fresh.RUN_ID,code_commit=REFERENCE_CODE,config=p,
        all_three_DURA_conditions_complete=True,all_conditions_through_DURA=True,source_models_frozen_before_V=True,
        all_thresholds_frozen_before_novel_and_Q=True,source_states_unchanged=True,frozen_parameters_buffers_unchanged=True,
        classification_unchanged=True,private_truth_reads=0,target_U_reads=0)
    if any(m.get(k)!=v for k,v in required.items()):raise ValueError('Complete original fresh DURA reference required')
    parent,old_states,_,_=fresh._reference(p)
    if m['source_control_prediction_identity']!=parent or m['model_identities']['E0']!=parent['model_identities']['E0']:raise ValueError('Original source E0 ancestry differs')
    fresh._identity(m['model_identities']['TARGET_PCD'],p,parent)
    fit=_read(root/'models/complete.json');old_cal=_read(root/'calibration/complete.json')
    if (fit.get('status')!=fresh.MODELS_STATUS or fit.get('code_commit')!=REFERENCE_CODE or fit.get('identities')!=m['model_identities']
            or fit.get('source_V_fit') is not False or old_cal.get('status')!=fresh.CALIBRATION_STATUS or old_cal.get('code_commit')!=REFERENCE_CODE
            or old_cal.get('known_only') is not True or old_cal.get('unknown_inputs_used') is not False):raise ValueError('Frozen original source density and calibration required')
    states={}
    for name,file in (('E0','E0'),('PCD','TARGET_PCD')):
        with np.load(root/'models'/(file+'.npz'),allow_pickle=False) as z:states[name]={k:z[k].copy() for k in z.files}
        density.validate_state(states[name],'density_ratio')
    if not _same(states['E0'],old_states['E0']):raise ValueError('Original E0 source density changed')
    with np.load(root/'thresholds.npz',allow_pickle=False) as z:
        if set(z.files)!={'conditional'}:raise ValueError('Original exact threshold state required')
        full=z['conditional'].copy()
    if full.shape!=(6,6) or full.dtype!=np.float64 or not np.isfinite(full).all() or old_cal.get('conditional_thresholds')!=full.tolist():raise ValueError('Original six known-only threshold vectors required')
    indices=(0,1,4,5);tau=full[list(indices)].copy();reports={}
    for name,j in zip(METHODS,indices):
        report=old_cal['methods'][fresh.METHODS[j]]
        if report['conditional_thresholds']!=full[j].tolist():raise ValueError('Inherited source threshold report differs')
        reports[name]=deepcopy(report)
    cal=dict(schema=SCHEMA,status=CALIBRATION_STATUS,run_id=RUN_ID,code_commit=c['code_commit'],method_names=list(METHODS),methods=reports,
        conditional_thresholds=tau.tolist(),known_only=True,unknown_inputs_used=False,thresholds_refitted=False,
        parent_calibration_identity={k:old_cal[k] for k in ('run_id','code_commit','status')})
    return m,states,tau,cal


def _backend(c,device):return fresh._backend(c['reference_config'],device)
def _load_model(c,device):return fresh._load_model(c['reference_config'],device)
def _public(c,condition):return fresh._public(c['reference_config'],condition)


def _source_role(c,role):
    if role not in ('V','novel'):raise ValueError('Only original source V/novel received values may be read')
    source=c['reference_config']['source_control_config']['reference_config']['train_config']
    return received.load_received(Path(source['IQ_root'])/role,role)


def _select(a):
    return {k:(v[:,[0,1,4,5]].copy() if k in ('raw_scores','scores') else v[:,[0,2]].copy() if k=='signal_features' else v.copy()) for k,v in a.items()}


def _raw(states,z,closed):
    return np.column_stack([fresh._raw(states[name],z[:,j],closed) for j,name in enumerate(MODEL_NAMES)])


def _validate(a,role):
    n=COUNTS['V' if role=='known' else 'novel'] if role in ('known','novel') else Q_COUNT
    fields=KNOWN_FIELDS if role=='known' else NOVEL_FIELDS if role=='novel' else TARGET_FIELDS
    if (set(a)!=fields or a['ids'].shape!=(n,) or a['ids'].dtype.kind!='U' or len(set(a['ids'].tolist()))!=n or np.any(np.char.strip(a['ids'])=='')
            or a['default_logits'].shape!=(n,6) or a['default_logits'].dtype!=np.float32 or not np.isfinite(a['default_logits']).all()
            or a['closed'].shape!=(n,) or a['closed'].dtype!=np.int64 or not np.array_equal(a['closed'],a['default_logits'].argmax(1))):
        raise ValueError('Complete path-specific original F0 identity and exact public fields required')
    if 'baseline_closed' in fields and (a['baseline_closed'].shape!=(n,) or a['baseline_closed'].dtype!=np.int64 or not np.array_equal(a['baseline_closed'],a['closed'])):raise ValueError('Original F0 baseline must use this same input path')
    z=a['signal_features']
    if (z.shape!=(n,2,160) or z.dtype!=np.float32 or not np.isfinite(z).all() or not np.allclose(np.linalg.norm(z.astype(np.float64),axis=2),1.,rtol=0,atol=2e-6)
            or any(a[k].shape!=(n,4) or a[k].dtype!=np.float64 or not np.isfinite(a[k]).all() for k in ('raw_scores','scores'))):raise ValueError('Two frozen unit160 signals and four finite readouts required')
    if role in ('known','novel'):
        f=a['raw_signal_features']
        if f.shape!=(n,160) or f.dtype!=np.float32 or not np.isfinite(f).all():raise ValueError('Same-forward raw original F0 features required')
        if any(a[k].shape!=(n,) or a[k].dtype!=np.int64 or not np.isin(a[k],allowed).all() for k,allowed in (('rx_i',(1,3,4,6,8)),('day_i',(1,2,3)))):raise ValueError('Original source RX/day metadata required')
        if role=='known' and (a['y'].shape!=(n,) or a['y'].dtype!=np.int64 or not np.isin(a['y'],range(6)).all() or a['scenario'].shape!=(n,) or a['scenario'].dtype.kind!='U' or not np.isin(a['scenario'],fresh.risk.SCENES).all()):raise ValueError('Original known V labels/scenarios required')
    return a


def _source_rows(root,condition,role):
    if condition not in SOURCE_CONDITIONS or role not in ('known','novel'):raise ValueError('Fixed source raw/DURA role required')
    with np.load(Path(root)/'source'/condition/(role+'_predictions.npz'),allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
    return _validate(a,role)


def _target_rows(root,condition):
    if condition not in CONDITIONS[2:]:raise ValueError('Fixed future raw/DURA path required')
    with np.load(Path(root)/'targets'/condition/'predictions.npz',allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
    return _validate(a,'Q')


def _current(b,models,values,device,restore=False):
    a=np.asarray(values)
    if a.dtype!=np.float32 or a.shape!=(len(a),2,256) or not len(a) or not np.isfinite(a).all():raise ValueError('Finite full256 received IQ required')
    with torch.no_grad(),torch.autocast(device_type=torch.device(device).type,enabled=False):
        x=torch.tensor(a.tolist(),device=device,dtype=torch.float32)
        if restore:x=b['dura'][1](x)[0].detach().clone()
        if x.shape!=a.shape or x.dtype!=torch.float32 or not bool(torch.isfinite(x).all()):raise ValueError('Finite same input must reach all frozen readers')
        f=b['reader'].features(x);logits=b['reader'].classify_features(f);values=[models[name](x) for name in MODEL_NAMES]
    f=np.asarray(f.detach().cpu().tolist(),np.float32);logits=np.asarray(logits.detach().cpu().tolist(),np.float32)
    z=np.stack([np.asarray(v.detach().cpu().tolist(),np.float32) for v in values],axis=1)
    if (f.shape!=(len(a),160) or logits.shape!=(len(a),6) or z.shape!=(len(a),2,160) or not all(np.isfinite(v).all() for v in (f,logits,z))
            or not np.allclose(np.linalg.norm(z.astype(np.float64),axis=2),1,rtol=0,atol=2e-6)):raise ValueError('Actual frozen F0/default6 and two unit160 signal outputs required')
    closed=logits.argmax(1).astype(np.int64)
    return dict(default_logits=logits,closed=closed,baseline_closed=closed.copy(),raw_signal_features=f,signal_features=z)


def _smoke(b,models,device):
    from .star_open_feature_export import _synthetic
    values=np.asarray(_synthetic(device).detach().cpu().tolist(),np.float32)
    _current(b,models,values,device,False);_current(b,models,values,device,True)
    return dict(status='PASS',actual_F0_E0_PCD_DURA_checkpoints=True,real_data_reads=0,optimizer_updates=0,
        synthetic_F_forward_rows=4,synthetic_signal_forward_rows=8,G_synthetic_forward_rows=2)


def run(c,device='cuda:0'):
    validate_config(c);out=Path(c['output_root'])
    if out.exists():raise FileExistsError('New immutable fresh-DURA output required')
    pm,states,tau,cal=_reference(c);b=_backend(c,device);pcd,identity=_load_model(c,device)
    identities=dict(E0=pm['model_identities']['E0'],PCD=pm['model_identities']['TARGET_PCD']);models=dict(E0=b['model'],PCD=pcd)
    if (b['model_identity']!=identities['E0'] or identity!=identities['PCD'] or b['reader_identity']!=pm['parent_F0_identity'] or b['dura_identity']!=pm['loaded_dura_identity']):raise ValueError('Actual original F0/source-r03/final-PCD/DURA ancestry differs')
    if any(p.requires_grad or p.dtype!=torch.float32 for m in (b['reader'],b['dura'][0],*models.values()) for p in m.parameters()) or any(v.training for m in (b['reader'],b['dura'][0],*models.values()) for v in m.modules()):raise ValueError('All original models must stay frozen FP32 eval')
    snapshots=[(m,fresh.best._snapshot_model(m)) for m in (b['reader'],b['dura'][0],*models.values())];fixed=deepcopy(states);fixed_tau=tau.copy();tau.setflags(write=False)
    def check():
        for m,before in snapshots:fresh.best._check_model(m,before)
        if not _same(states,fixed) or not np.array_equal(tau,fixed_tau):raise ValueError('Frozen source models or operating points changed')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    smoke=_smoke(b,models,device);check()
    if smoke.get('status')!='PASS' or smoke.get('actual_F0_E0_PCD_DURA_checkpoints') is not True:raise ValueError('Actual checkpoint no-query smoke must precede real inputs')
    out.mkdir();_write(out/'resolved_config.json',c);_write(out/'smoke.json',smoke)
    for name,state in states.items():_save(out/'models'/(name+'.npz'),state)
    _write(out/'models/complete.json',dict(schema=SCHEMA,status=MODELS_STATUS,run_id=RUN_ID,code_commit=c['code_commit'],model_identities=identities,
        parent_F0_identity=pm['parent_F0_identity'],dura_identity=pm['dura_identity'],loaded_dura_identity=b['dura_identity'],models_frozen=True,models_refitted=False,thresholds_refitted=False,source_V_fit=False,novel_fit=False,target_fit=False))
    _save(out/'thresholds.npz',dict(conditional=tau));(out/'calibration').mkdir();_write(out/'calibration/complete.json',cal)
    rows_done={};source_ids=set();source_manifests={};source_count=0;query_count=0;Q_manifests={}
    def score(a):
        a['raw_scores']=_raw(states,a['signal_features'],a['closed']);a['scores']=a['raw_scores']-tau[:,a['closed']].T
        return a
    for role,cache_role in (('known','V'),('novel','novel')):
        old=_select(fresh._source_rows(c['reference_config']['output_root'],'source_DURA',role));_validate(old,role)
        _save(out/'source/source_DURA'/(role+'_predictions.npz'),old);iq=_source_role(c,cache_role)
        try:
            if (iq['manifest']['parent_F0_identity']!=pm['parent_F0_identity'] or iq['manifest']['dura_identity']!=pm['dura_identity']
                    or not np.array_equal(iq['ids'],old['ids']) or any(not np.array_equal(iq['metadata'][k],old[k]) for k in ('rx_i','day_i')+(('y','scenario') if role=='known' else ()))):raise ValueError('Exact original same-record received source role required')
            ids=set(iq['ids'].tolist())
            if ids&source_ids:raise ValueError('Source known/novel roles overlap')
            source_ids.update(ids);blocks=[_current(b,models,iq['received'][j:j+128],device) for j in range(0,len(iq['ids']),128)]
            current={k:np.concatenate([v[k] for v in blocks]) for k in blocks[0]};a={k:v.copy() for k,v in old.items()}
            a.update({k:v for k,v in current.items() if k in a});score(a);_validate(a,role);_save(out/'source/source_raw'/(role+'_predictions.npz'),a)
            source_count+=len(iq['ids']);source_manifests[cache_role]=iq['manifest'];rows_done[role]=len(iq['ids']);check()
            print('FRESH_DURA_SOURCE_COMPLETE',role,len(iq['ids']),flush=True)
        finally:received.close_role(iq)
    paired=None
    for condition in QUERY_CONDITIONS:
        old=_select(fresh._target_rows(c['reference_config']['output_root'],condition));_validate(old,'Q');_save(out/'targets'/condition/'predictions.npz',old)
        iq=None
        try:
            manifest,ids,iq=_public(c,condition)
            if not np.array_equal(ids,old['ids']) or set(ids.tolist())&source_ids:raise ValueError('Exact paired future IDs disjoint from source roles required')
            if paired is not None and not np.array_equal(ids,paired):raise ValueError('Clean/star future canonical order differs')
            paired=ids.copy();blocks=[_current(b,models,iq[j:j+128],device) for j in range(0,len(ids),128)]
            current={k:np.concatenate([v[k] for v in blocks]) for k in blocks[0]};a=dict(ids=ids.copy(),**{k:v for k,v in current.items() if k in TARGET_FIELDS})
            score(a);_validate(a,'Q');_save(out/'targets'/condition.replace('_DURA','_raw')/'predictions.npz',a)
            query_count+=len(ids);Q_manifests[condition]=manifest;check();print('FRESH_DURA_Q_COMPLETE',condition,len(ids),flush=True)
        finally:
            if hasattr(iq,'_mmap'):iq._mmap.close()
    total=source_count+query_count
    done=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c,method_names=list(METHODS),primary_method=PRIMARY_METHOD,model_names=list(MODEL_NAMES),conditions=list(CONDITIONS),
        reference_prediction_identity=pm,model_identities=identities,parent_F0_identity=pm['parent_F0_identity'],dura_identity=pm['dura_identity'],loaded_dura_identity=b['dura_identity'],
        known_count=rows_done['known'],novel_count=rows_done['novel'],Q_counts={name:Q_COUNT for name in CONDITIONS[2:]},Q_manifests=Q_manifests,source_manifests=source_manifests,
        all_six_paths_complete=True,all_DURA_main_conditions_complete=True,all_DURA_controls_reused_exactly=True,models_frozen_before_any_new_input=True,
        all_thresholds_frozen_before_source_raw_and_Q=True,model_parameters_buffers_unchanged=True,source_states_unchanged=True,classification_unchanged=True,
        identity_baseline_path='original_F0_on_each_same_input_path',primary_input_path='DURA',models_refitted=False,thresholds_refitted=False,
        source_IQ_reads=source_count,target_Q_IQ_reads=query_count,source_L_reads=0,source_OE_reads=0,original_U_s_reads=0,source_Us_reads=0,target_U_reads=0,U_ID_metadata_reads=0,
        private_truth_reads=0,U_truth_reads=0,source_novel_truth_reads=0,target_truth_reads=0,new_neural_updates=0,optimizer_updates=0,density_fit_updates=0,threshold_fit_updates=0,
        real_F_forward_rows=total,real_signal_forward_rows=2*total,real_signal_forward_rows_by_model={name:total for name in MODEL_NAMES},G_forward_rows=0,
        source_DURA_rows_reused=source_count,Q_DURA_rows_reused={name:Q_COUNT for name in QUERY_CONDITIONS},threshold_values=24,threshold_bytes=tau.nbytes,
        inherited_PCD_target_U_count=576,PCD_has_target_U_ancestor=True,new_target_DA=False,target_REG=False,formal_target_accessed=False,
        primary_deployed_neural_parameters=834665,full_goal_status='ACTIVE',final_phase3_complete=False,**{k:v for k,v in smoke.items() if k.startswith('synthetic_') or k=='G_synthetic_forward_rows'})
    check();_write(out/'complete.json',done);print(STATUS,flush=True);return done


evaluate=run
def main():
    print(json.dumps(dict(event='FRESH_DURA_PROCESS',pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv)),flush=True)
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args();run(_read(a.config),a.device)


if __name__=='__main__':main()
