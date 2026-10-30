"""Same frozen P1 signal and source operating points before/after DURA."""
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
from . import star_raw_open_eval as reference
from . import star_raw_open_train as received
from . import star_signal_open_train as signal_train
from . import star_signal_open_eval as signal_eval
from . import star_relative_source as relative
from . import star_density_core as density
from .development_transfer import _snapshot_model, _check_model

RUN_ID='20261031-phase3-dura-best-s2026092701-r01'
SCHEMA='pamf_phase3_dura_best_prediction_v1'
STATUS='DURA_BEST_PREDICTIONS_COMPLETE'
MODELS_STATUS='DURA_BEST_MODELS_FROZEN'
REFERENCE_RUN='20261031-phase3-star-raw-open-s2026092701-r01'
REFERENCE_CODE='617681c8b2f24e7d05c1e8f5a48cb74d99a1c8cb'
SIGNAL_RUN='20261030-phase3-star-signal-open-s2026092701-r03'
SIGNAL_CODE='dc01a402582f93e452dbaa3ef59a9fbb44151818'
METHODS=('known_support','density_ratio')
PRIMARY_METHOD='density_ratio'
PRIMARY_INDEX=1
SOURCE_CONDITIONS=('source_raw','source_DURA')
QUERY_CONDITIONS=('clean_DURA','star_DURA')
CONDITIONS=SOURCE_CONDITIONS+('clean_raw','clean_DURA','star_raw','star_DURA')
COUNTS=dict(L=6300,OE=6134,V=27000,novel=3150)
Q_COUNT=864
SCENES=signal_eval.parent.reference.SCENES
FIELDS={'run_id','code_commit','train_config','reference_prediction_root','source_IQ_root','output_root','batch_size','Q_conditions','dura_checkpoint','dura_release'}
TARGET_FIELDS={'ids','default_logits','closed','baseline_closed','signal_features','raw_scores','scores'}
KNOWN_FIELDS=TARGET_FIELDS|{'y','rx_i','day_i','scenario'}
NOVEL_FIELDS=KNOWN_FIELDS-{'y','scenario','baseline_closed'}
_read,_write,_npz,_same=reference._read,reference._write,reference._npz,reference._same


def validate_config(c):
    if (not isinstance(c,dict) or set(c)!=FIELDS or c.get('run_id')!=RUN_ID or re.fullmatch('[0-9a-f]{40}',str(c.get('code_commit'))) is None
            or type(c.get('batch_size')) is not int or c['batch_size']!=128 or not isinstance(c.get('train_config'),dict)
            or any(not isinstance(c.get(k),str) or not c[k].strip() for k in ('reference_prediction_root','source_IQ_root','output_root','dura_checkpoint','dura_release'))):
        raise ValueError('Exact ten-field frozen six-path configuration required; no truth/U/K')
    t=signal_train.validate_config(c['train_config'])
    if (t['run_id'],t['code_commit'],t['arm'])!=(SIGNAL_RUN,SIGNAL_CODE,'E_SIGNAL') or Path(c['source_IQ_root']).resolve()!=Path(t['IQ_root']).resolve():
        raise ValueError('Only original r03 P1 signal and its identical physical source IQ root allowed')
    q=c['Q_conditions']
    if not isinstance(q,dict) or set(q)!=set(QUERY_CONDITIONS):raise ValueError('Both clean_DURA and star_DURA public handles required')
    roots=[]
    for name,family in zip(QUERY_CONDITIONS,('clean_cross_receiver','practical_cross_receiver')):
        s=q[name]
        if (not isinstance(s,dict) or set(s)!={'public_root','capsule_id','split_id','expected_count','view_family'}
                or type(s['expected_count']) is not int or s['expected_count']!=Q_COUNT or s['view_family']!=family
                or any(not isinstance(s[k],str) or not s[k].strip() for k in ('public_root','capsule_id','split_id'))):raise ValueError('Exact separate clean/star future-Q handles required')
        p=Path(s['public_root'])
        if p.name!='public' or p.parent.name!='evaluation':raise ValueError('Only evaluation public may be opened')
        roots.append(p.resolve())
    if roots[0]==roots[1]:raise ValueError('Separate clean and star public inputs required')
    output=Path(c['output_root']).resolve()
    for p in [*roots,Path(c['source_IQ_root']),Path(c['reference_prediction_root']),Path(t['output_root']),Path(c['dura_release']),Path(c['dura_checkpoint'])]:
        p=p.resolve()
        if output.is_relative_to(p) or p.is_relative_to(output):raise ValueError('Immutable output cannot overlap frozen inputs')
    return c


def _reference(root,train_config):
    """Source metadata and original P1 aggregate state; no source rows/P0 CP."""
    root=Path(root);m=_read(root/'complete.json')
    expected=dict(schema=reference.SCHEMA,status=reference.STATUS,run_id=REFERENCE_RUN,code_commit=REFERENCE_CODE,arm='E_RAW',method_names=list(reference.METHODS),
        classification_path='P1',known_count=COUNTS['V'],novel_count=COUNTS['novel'],source_only=True,private_truth_reads=0,target_reads=0,original_U_s_reads=0,
        identity_predictions_unchanged=True,models_frozen_before_V=True,all_thresholds_frozen_before_novel=True,source_states_unchanged=True,model_parameters_buffers_unchanged=True)
    if any(m.get(k)!=v for k,v in expected.items()):raise ValueError('Completed fixed raw-open P1 controls required')
    reference.validate_config(m['config']);sm=m.get('signal_reference_identity',{})
    if (sm.get('run_id')!=SIGNAL_RUN or sm.get('code_commit')!=SIGNAL_CODE or sm.get('config',{}).get('train_config')!=train_config
            or signal_eval._identity(sm['model_identity'],sm['config'])!=m['parent_F0_identity'] or sm['model_identity']['dura_identity']!=m['dura_identity']):
        raise ValueError('Use P1 r03 signal_reference_identity, never raw parent P0 identity')
    with np.load(root/'models/P1_known12.npz',allow_pickle=False) as z:known={k:z[k].copy() for k in z.files}
    with np.load(root/'models/P1_ratio12.npz',allow_pickle=False) as z:state={k:z[k].copy() for k in z.files}
    density.validate_state(known,'known_only');density.validate_state(state,'density_ratio')
    if state['known_means'].shape!=(90,160) or any(not np.array_equal(v,state[k]) for k,v in known.items()):raise ValueError('Shared original D160 ninety-mode P1 known statistics required')
    fit=_read(root/'models/complete.json')
    if any(fit.get(k)!=v for k,v in dict(schema=reference.SCHEMA,status=reference.MODELS_STATUS,run_id=REFERENCE_RUN,code_commit=REFERENCE_CODE,
            P1_models_reused=True,P1_known_statistics_shared=True,source_V_fit=False,novel_fit=False).items()):raise ValueError('Original P1 source model freeze required')
    with np.load(root/'thresholds.npz',allow_pickle=False) as z:
        if set(z.files)!={'conditional'}:raise ValueError('Complete parent conditional thresholds required')
        tau=z['conditional'].copy()
    cal=_read(root/'calibration/complete.json')
    if (tau.shape!=(4,6) or tau.dtype!=np.float64 or not np.isfinite(tau).all()
            or any(cal.get(k)!=v for k,v in dict(schema=reference.SCHEMA,status=reference.CALIBRATION_STATUS,run_id=REFERENCE_RUN,code_commit=REFERENCE_CODE,
                method_names=list(reference.METHODS),known_count=COUNTS['V'],group_count=42,known_only=True,unknown_inputs_used=False,target_reads=0).items())
            or not np.array_equal(tau,cal.get('conditional_thresholds'))):raise ValueError('Exact original P1 two-source operating points required')
    for j,name in enumerate(reference.METHODS[:2]):
        if cal['methods'][name]['conditional_thresholds']!=tau[j].tolist():raise ValueError('Source thresholds/report mismatch')
    for v in state.values():v.setflags(write=False)
    return m,state,tau[:2].copy(),cal


def _validate_rows(a,role):
    fields=KNOWN_FIELDS if role=='known' else NOVEL_FIELDS
    n=COUNTS['V' if role=='known' else 'novel']
    if set(a)!=fields or a['ids'].shape!=(n,) or a['ids'].dtype.kind!='U' or len(set(a['ids'].tolist()))!=n:raise ValueError('Exact compact source fields/unique ID order required; novel truth forbidden')
    if (any(a[k].shape!=(n,) or a[k].dtype!=np.int64 for k in ('rx_i','day_i','closed')) or not np.isin(a['rx_i'],(1,3,4,6,8)).all()
            or not np.isin(a['day_i'],(1,2,3)).all() or not np.isin(a['closed'],np.arange(6)).all()):raise ValueError('Original source public RX/day/default classes required')
    for k,width,dtype in (('default_logits',6,np.float32),('signal_features',160,np.float32),('raw_scores',2,np.float64),('scores',2,np.float64)):
        if a[k].shape!=(n,width) or a[k].dtype!=dtype or not np.isfinite(a[k]).all():raise ValueError('Complete finite compact source predictions required')
    if (not np.array_equal(a['closed'],a['default_logits'].argmax(1)) or not np.allclose(np.linalg.norm(a['signal_features'].astype(np.float64),axis=1),1,atol=2e-6,rtol=0)):
        raise ValueError('Original F0 logits and unit signal feature identity required')
    if role=='known' and (a['y'].shape!=(n,) or a['y'].dtype!=np.int64 or not np.isin(a['y'],np.arange(6)).all()
            or a['scenario'].shape!=(n,) or a['scenario'].dtype.kind!='U' or not np.isin(a['scenario'],SCENES).all()
            or a['baseline_closed'].dtype!=np.int64 or not np.array_equal(a['baseline_closed'],a['closed'])):raise ValueError('Known-only labels/scenarios and same-input F0 baseline required')
    return a


def _rows(root,role):
    if role not in ('known','novel'):raise ValueError('Only source known/novel allowed')
    fields=KNOWN_FIELDS if role=='known' else NOVEL_FIELDS
    with np.load(Path(root)/(role+'_predictions.npz'),allow_pickle=False) as z:
        if set(z.files)!=(reference.KNOWN_FIELDS if role=='known' else reference.NOVEL_FIELDS):raise ValueError('Parent source fields differ; novel labels forbidden')
        a={k:z[k].copy() for k in fields}
    for k in ('raw_scores','scores'):a[k]=a[k][:,:2].copy()
    return _validate_rows(a,role)


def _source_rows(root,condition,role):
    if condition not in SOURCE_CONDITIONS or role not in ('known','novel'):raise ValueError('Declared compact source path required')
    with np.load(Path(root)/'source'/condition/(role+'_predictions.npz'),allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
    return _validate_rows(a,role)


def _source_role(c,role):
    if role not in ('V','novel'):raise ValueError('No new L/OE/Us IQ reads allowed')
    return received.load_received(Path(c['source_IQ_root'])/role,role)


def _backend(c,device):
    from .star_channel import _channel_api
    _channel_api();n=c['train_config']['native_config']
    spec=dict(reader_config={k:n[k] for k in ('base_release','frozen_sources','source_output','model_seed')},source_contract=n['source_contract'],dura_checkpoint=c['dura_checkpoint'],dura_release=c['dura_release'])
    b=relative._backend(spec,device,with_dura=True)
    model,identity=signal_train.load_trained_model(c['train_config'],Path(c['train_config']['output_root'])/'final.pt',device)
    return dict(b,model=model,model_identity=identity)


def _current(b,values,device,restore):
    values=np.asarray(values)
    if values.dtype!=np.float32 or values.shape!=(len(values),2,256) or not np.isfinite(values).all():raise ValueError('Finite FP32 received full256 IQ required')
    with torch.no_grad(),torch.autocast(device_type=torch.device(device).type,enabled=False):
        x=torch.tensor(values.tolist(),dtype=torch.float32,device=device)
        if restore:x=b['dura'][1](x)[0].detach().clone()
        if x.dtype!=torch.float32 or x.shape!=values.shape or not bool(torch.isfinite(x).all()):raise ValueError('Same finite input must reach original F0 and P1 signal')
        logits=b['reader'].classify_features(b['reader'].features(x));z=b['model'](x)
    logits=np.asarray(logits.detach().cpu().tolist(),np.float32);z=np.asarray(z.detach().cpu().tolist(),np.float32)
    if (logits.shape!=(len(values),6) or z.shape!=(len(values),160) or not np.isfinite(logits).all() or not np.isfinite(z).all()
            or not np.allclose(np.linalg.norm(z.astype(np.float64),axis=1),1,atol=2e-6,rtol=0)):raise ValueError('Finite default logits6 and signal unit160 required')
    closed=logits.argmax(1).astype(np.int64)
    return dict(default_logits=logits,closed=closed,baseline_closed=closed.copy(),signal_features=z)


def _smoke(b,device):
    from .star_open_feature_export import _synthetic
    values=np.asarray(_synthetic(device).detach().cpu().tolist(),np.float32);a=_current(b,values,device,True)
    if len(a['closed'])!=2:raise ValueError('Two actual-checkpoint synthetic G10 rows required')
    return dict(status='PASS',actual_F0_signal_DURA_checkpoints=True,G_synthetic_forward_rows=2,synthetic_F_forward_rows=2,synthetic_signal_forward_rows=2,real_data_reads=0,optimizer_updates=0)


def _public(c,name):
    s=c['Q_conditions'][name]
    if name=='clean_DURA':
        from .development_clean import load_clean_public
        m,ids,iq=load_clean_public(s['public_root'])
    else:
        from .star_relative_target import _open
        m,ids,iq=_open(dict(Q_public_root=s['public_root'],Q_capsule_id=s['capsule_id'],Q_split_id=s['split_id'],Q_expected_count=Q_COUNT),'Q')
    if (m.get('capsule_id')!=s['capsule_id'] or m.get('split_id')!=s['split_id'] or m.get('count')!=Q_COUNT
            or m.get('registered_classes')!=signal_train.native.CLASSES or ids.shape!=(Q_COUNT,) or ids.dtype.kind!='U' or len(set(ids.tolist()))!=Q_COUNT):
        if hasattr(iq,'_mmap'):iq._mmap.close()
        raise ValueError('Declared future Q capsule/split/old-six identity differs')
    return m,ids,iq


def _target_rows(root,condition):
    if condition not in CONDITIONS[2:]:raise ValueError('Declared raw/DURA future-Q path required')
    with np.load(Path(root)/'targets'/condition/'predictions.npz',allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
    if set(a)!=TARGET_FIELDS:raise ValueError('Exact public query prediction fields required; no labels/scenes')
    return a


def _scores(state,z):
    known={k:state[k] for k in density.KNOWN_FIELDS}
    return np.column_stack((density.score(known,z,'known_only'),density.score(state,z,'density_ratio')))


def _save(path,a):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);_npz(path,a)


def run(c,device='cuda:0'):
    validate_config(c);out=Path(c['output_root'])
    if out.exists():raise FileExistsError('New immutable six-path output required')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;b=_backend(c,device)
    signal_eval._identity(b['model_identity'],dict(code_commit=SIGNAL_CODE,train_config=c['train_config']))
    snapshots=[(m,_snapshot_model(m)) for m in (b['reader'],b['model'],b['dura'][0])]
    def models_unchanged():
        for model,before in snapshots:_check_model(model,before)
    smoke=_smoke(b,device);models_unchanged()
    if smoke.get('status')!='PASS' or smoke.get('actual_F0_signal_DURA_checkpoints') is not True:raise ValueError('Actual three-checkpoint smoke must precede source/query reads')
    pm,state,tau,pc=_reference(c['reference_prediction_root'],c['train_config']);identity=pm['signal_reference_identity']['model_identity']
    if b['model_identity']!=identity or b['reader_identity']!=identity['parent_F0_identity']:raise ValueError('Actual original F0/P1 signal differs; P0-trained branch excluded')
    actual=b['dura_identity'];expected=identity['dura_identity']
    if any(actual.get(k)!=v or expected.get(k)!=v for k,v in dict(deployment_id='dura_v1',parameters=266951,epoch=20,updates=4940).items()) or any(actual.get(k)!=expected[k] for k in ('checkpoint','arm','training_commit','reuse','graph','native','precision','TF32') if k in expected):raise ValueError('Same frozen DURA v1 computation required')
    frozen=deepcopy(state);frozen_tau=tau.copy();tau.setflags(write=False)
    def check():
        models_unchanged()
        if not _same(state,frozen) or not np.array_equal(tau,frozen_tau):raise ValueError('Frozen source statistics/thresholds changed')
    out.mkdir(parents=True,exist_ok=False);_write(out/'resolved_config.json',c);_write(out/'smoke.json',smoke);_save(out/'models/density.npz',state)
    _write(out/'models/complete.json',dict(schema=SCHEMA,status=MODELS_STATUS,run_id=RUN_ID,code_commit=c['code_commit'],method_names=list(METHODS),models_frozen=True,models_refitted=False,
        thresholds_refitted=False,parent_calibration_identity=pc,model_identity=identity,parent_F0_identity=identity['parent_F0_identity'],dura_identity=identity['dura_identity'],source_V_fit=False,novel_fit=False,target_fit=False))
    _save(out/'thresholds.npz',dict(conditional=tau));(out/'calibration').mkdir();_write(out/'calibration/complete.json',pc)
    source_ids=set();source_manifests={}
    for role,cache_role in (('known','V'),('novel','novel')):
        old=_rows(c['reference_prediction_root'],role)
        if source_ids&set(old['ids'].tolist()):raise ValueError('Known/novel physical records overlap')
        source_ids.update(old['ids'].tolist())
        if not np.allclose(_scores(state,old['signal_features']),old['raw_scores'],rtol=1e-12,atol=1e-9) or not np.array_equal(old['scores'],old['raw_scores']-tau[:,old['closed']].T):raise ValueError('Original source_DURA controls do not reconstruct')
        _save(out/'source/source_DURA'/(role+'_predictions.npz'),old);check();a=None
        try:
            a=_source_role(c,cache_role);meta=a['metadata']
            if (a['manifest']!=pm['IQ_manifests'][cache_role] or not np.array_equal(a['ids'],old['ids'])
                    or any(not np.array_equal(meta[k],old[k]) for k in ('rx_i','day_i')+(('y','scenario') if role=='known' else ()))):raise ValueError('Exact parent source physical IDs/metadata/role handle required')
            blocks=[]
            for start in range(0,len(old['ids']),128):
                batch=_current(b,a['received'][start:start+128],device,False);raw=_scores(state,batch['signal_features'])
                blocks.append(dict(batch,raw_scores=raw,scores=raw-tau[:,batch['closed']].T));check();print('DURA_BEST_SOURCE_PROGRESS',role,start+len(batch['closed']),len(old['ids']),flush=True)
            new=dict(old,**{k:np.concatenate([v[k] for v in blocks]) for k in blocks[0]})
            if role=='novel':new.pop('baseline_closed')
            _validate_rows(new,role);_save(out/'source/source_raw'/(role+'_predictions.npz'),new);source_manifests[cache_role]=a['manifest']
        finally:
            if a is not None:received.close_role(a)
    public_manifests={}
    for name in QUERY_CONDITIONS:
        iq=None
        try:
            manifest,ids,iq=_public(c,name)
            if source_ids&set(ids.tolist()):raise ValueError('Source and query physical roles overlap')
            paths=(name.replace('_DURA','_raw'),name);blocks={p:[] for p in paths}
            for start in range(0,len(ids),128):
                values=iq[start:start+128]
                for path,restore in zip(paths,(False,True)):
                    a=_current(b,values,device,restore);raw=_scores(state,a['signal_features']);blocks[path].append(dict(a,raw_scores=raw,scores=raw-tau[:,a['closed']].T));check()
                print('DURA_BEST_Q_PROGRESS',name,start+len(values),len(ids),flush=True)
            for path in paths:
                result={k:np.concatenate([a[k] for a in blocks[path]]) for k in blocks[path][0]};result['ids']=ids.copy();_save(out/'targets'/path/'predictions.npz',result)
            public_manifests[name]=manifest
        finally:
            if hasattr(iq,'_mmap'):iq._mmap.close()
    check();rows=COUNTS['V']+COUNTS['novel']
    done=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c,method_names=list(METHODS),primary_method=PRIMARY_METHOD,
        classification_path='per_input_original_F0',identity_baseline_path='same_path_original_F0',score_threshold=0.,conditions=list(CONDITIONS),known_count=COUNTS['V'],novel_count=COUNTS['novel'],
        source_known_count=COUNTS['V'],source_novel_count=COUNTS['novel'],Q_counts={name:Q_COUNT for name in CONDITIONS[2:]},Q_manifests=public_manifests,source_IQ_manifests=source_manifests,
        reference_prediction_identity=pm,parent_calibration_identity=pc,model_identity=identity,parent_F0_identity=identity['parent_F0_identity'],dura_identity=identity['dura_identity'],loaded_dura_identity=actual,
        source_only=True,source_received_rows_read=rows,source_L_reads=0,source_OE_reads=0,original_U_s_reads=0,target_U_reads=0,private_truth_reads=0,source_novel_truth_reads=0,target_truth_reads=0,formal_target_accessed=False,
        new_neural_updates=0,native_updates=0,density_fit_updates=0,normalizer_fit_updates=0,threshold_fit_updates=0,
        real_signal_forward_rows=rows+4*Q_COUNT,real_F_forward_rows=rows+4*Q_COUNT,G_forward_rows=2*Q_COUNT,synthetic_signal_forward_rows=2,synthetic_F_forward_rows=2,G_synthetic_forward_rows=2,source_DURA_rows_reused=rows,
        models_frozen_before_any_new_input=True,all_thresholds_frozen_before_source_raw_and_Q=True,all_six_paths_complete=True,all_DURA_main_conditions_complete=True,source_DURA_controls_reused_exactly=True,
        source_states_unchanged=True,model_parameters_buffers_unchanged=True,source_bank_deployed=False,per_record_source_rows_deployed=0,final_phase3_complete=False,full_goal_status='ACTIVE',A_N=None,H_OW=None)
    _write(out/'complete.json',done);print(STATUS,flush=True);return done


evaluate=run


def main(argv=None):
    print(json.dumps(dict(event='DURA_BEST_PROCESS',pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv)),flush=True)
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args(argv);run(_read(a.config),a.device)


if __name__=='__main__':main()
