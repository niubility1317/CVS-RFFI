"""Unlabelled G10 U fits source-bounded shifts; future feature rows only score."""
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

RUN_ID='20261031-phase3-mixture-uda-s2026092701-r01'
SCHEMA='pamf_phase3_mixture_uda_prediction_v1'
SOURCE_STATUS='MIXTURE_UDA_SOURCE_FROZEN'
ADAPTER_STATUS='MIXTURE_UDA_ADAPTERS_FROZEN'
STATUS='MIXTURE_UDA_PREDICTIONS_COMPLETE'
REFERENCE_CODE='9fc23a4cb78111aa96cc72fde0d35c036a876e5d'
METHODS=('DA0_known','DA0_ratio','DA_K_known','DA_K_ratio','DA_M_known','DA_M_ratio')
PRIMARY_METHOD='DA_M_ratio'
PRIMARY_INDEX=5
CONDITIONS=('source_DURA','clean_DURA','star_DURA')
ADAPTERS=('DA_K','DA_M')
COUNTS=dict(L=6300,OE=6134,V=27000,novel=3150)
Q_COUNT=864
U_COUNT=384
U_CAPSULE_ID='p3dev-star-g10-6e1292b62ab09286ebf1a0df'
U_SPLIT_ID='83c5cc1fe2f2a1512dbc1287'
FIELDS={'run_id','code_commit','reference_prediction_root','U_public_root','U_capsule_id','U_split_id','output_root','batch_size'}
KNOWN_FIELDS,NOVEL_FIELDS,TARGET_FIELDS=best.KNOWN_FIELDS,best.NOVEL_FIELDS,best.TARGET_FIELDS
_read,_write,_same=best._read,best._write,best._same


def _core():
    from . import star_mixture_uda_core
    return star_mixture_uda_core


def validate_config(c):
    if (not isinstance(c,dict) or set(c)!=FIELDS or c.get('run_id')!=RUN_ID or re.fullmatch('[0-9a-f]{40}',str(c.get('code_commit'))) is None
            or type(c.get('batch_size')) is not int or c['batch_size']!=128
            or any(not isinstance(c.get(k),str) or not c[k].strip() for k in FIELDS-{'batch_size'})):
        raise ValueError('Exact eight-field U-only frozen mixture configuration required; no truth/K/Q inputs')
    root=Path(c['U_public_root'])
    if root.name!='public' or root.parent.name!='discovery' or c['U_capsule_id']!=U_CAPSULE_ID or c['U_split_id']!=U_SPLIT_ID:
        raise ValueError('Only the fixed star-G10 discovery U capsule/split is allowed; clean is test only')
    output=Path(c['output_root']).resolve()
    for p in (root,Path(c['reference_prediction_root'])):
        p=p.resolve()
        if output.is_relative_to(p) or p.is_relative_to(output):raise ValueError('Immutable output cannot overlap public/source prediction inputs')
    return c


def _reference(root):
    """Metadata/aggregate-only. Never call a parent scorer or open row NPZs."""
    root=Path(root);m=_read(root/'complete.json')
    expected=dict(schema=best.SCHEMA,status=best.STATUS,run_id=best.RUN_ID,code_commit=REFERENCE_CODE,method_names=list(best.METHODS),primary_method=best.PRIMARY_METHOD,
        known_count=COUNTS['V'],novel_count=COUNTS['novel'],classification_path='per_input_original_F0',identity_baseline_path='same_path_original_F0',
        all_six_paths_complete=True,all_DURA_main_conditions_complete=True,models_frozen_before_any_new_input=True,all_thresholds_frozen_before_source_raw_and_Q=True,
        source_DURA_controls_reused_exactly=True,source_states_unchanged=True,model_parameters_buffers_unchanged=True,source_only=True,new_neural_updates=0,density_fit_updates=0,threshold_fit_updates=0,
        private_truth_reads=0,target_U_reads=0,source_novel_truth_reads=0,target_truth_reads=0,formal_target_accessed=False,final_phase3_complete=False)
    if any(m.get(k)!=v for k,v in expected.items()):raise ValueError('Fixed completed DURA-best source-only checkpoint/control identity required')
    c=best.validate_config(m['config'])
    if c['run_id']!=m['run_id'] or c['code_commit']!=m['code_commit']:raise ValueError('Parent config/run identity differs')
    old,state,tau,cal=best._reference(c['reference_prediction_root'],c['train_config'])
    identity=old['signal_reference_identity']['model_identity']
    if (m['reference_prediction_identity']!=old or m['model_identity']!=identity or m['parent_F0_identity']!=identity['parent_F0_identity']
            or m['dura_identity']!=identity['dura_identity'] or m['parent_calibration_identity']!=cal):raise ValueError('Only original P1 r03/F0/DURA ancestry allowed, never P0 theta')
    with np.load(root/'models/density.npz',allow_pickle=False) as z:saved={k:z[k].copy() for k in z.files}
    with np.load(root/'thresholds.npz',allow_pickle=False) as z:
        if set(z.files)!={'conditional'} or not np.array_equal(z['conditional'],tau):raise ValueError('Original two six-class operating points must remain exact')
    fit=_read(root/'models/complete.json')
    if (not _same(saved,state) or any(fit.get(k)!=v for k,v in dict(schema=best.SCHEMA,status=best.MODELS_STATUS,run_id=best.RUN_ID,code_commit=REFERENCE_CODE,
            models_frozen=True,models_refitted=False,thresholds_refitted=False).items()) or fit.get('parent_calibration_identity')!=cal
            or _read(root/'calibration/complete.json')!=cal):raise ValueError('Original density11 and source calibration must be reused without refit/retau')
    return m,state,tau,cal


def _backend(parent,device):return best._backend(parent['config'],device)


def _public(c):
    from .star_relative_target import _open
    m,ids,iq=_open(dict(U_public_root=c['U_public_root'],U_capsule_id=c['U_capsule_id'],U_split_id=c['U_split_id'],U_expected_count=U_COUNT),'U')
    if (m.get('count')!=U_COUNT or m.get('capsule_id')!=c['U_capsule_id'] or m.get('split_id')!=c['U_split_id']
            or m.get('registered_classes')!=best.signal_train.native.CLASSES or ids.shape!=(U_COUNT,) or ids.dtype.kind!='U' or len(set(ids.tolist()))!=U_COUNT):
        if hasattr(iq,'_mmap'):iq._mmap.close()
        raise ValueError('Fixed complete unlabelled star-G10 U records required')
    return m,ids,iq


def _parent_rows(root,role):return best._source_rows(root,'source_DURA',role)
def _parent_target_rows(root,condition):return best._target_rows(root,condition)


def _validate(a,role):
    if role in ('known','novel'):
        best._validate_rows(dict(a,raw_scores=a['raw_scores'][:,:2],scores=a['scores'][:,:2]),role)
        n=COUNTS['V' if role=='known' else 'novel']
    else:
        n=Q_COUNT
        if (set(a)!=TARGET_FIELDS or a['ids'].shape!=(n,) or a['ids'].dtype.kind!='U' or len(set(a['ids'].tolist()))!=n
                or a['closed'].shape!=(n,) or a['closed'].dtype!=np.int64 or not np.array_equal(a['closed'],a['default_logits'].argmax(1))
                or a['baseline_closed'].dtype!=np.int64 or not np.array_equal(a['baseline_closed'],a['closed'])
                or a['default_logits'].shape!=(n,6) or a['default_logits'].dtype!=np.float32 or not np.isfinite(a['default_logits']).all()
                or a['signal_features'].shape!=(n,160) or a['signal_features'].dtype!=np.float32 or not np.isfinite(a['signal_features']).all()
                or not np.allclose(np.linalg.norm(a['signal_features'].astype(np.float64),axis=1),1,atol=2e-6,rtol=0)):raise ValueError('Complete label-free future Q identity/features required')
    if any(a[k].shape!=(n,6) or a[k].dtype!=np.float64 or not np.isfinite(a[k]).all() for k in ('raw_scores','scores')):raise ValueError('Six complete finite mixture readout columns required')
    return a


def _source_rows(root,condition,role):
    if condition!='source_DURA' or role not in ('known','novel'):raise ValueError('Only the fixed source-DURA roles allowed')
    with np.load(Path(root)/'source'/condition/(role+'_predictions.npz'),allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
    return _validate(a,role)


def _target_rows(root,condition):
    if condition not in CONDITIONS[1:]:raise ValueError('Only the paired clean/star DURA future Q allowed')
    with np.load(Path(root)/'targets'/condition/'predictions.npz',allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
    return _validate(a,'Q')


def run(c,device='cuda:0'):
    validate_config(c);out=Path(c['output_root'])
    if out.exists():raise FileExistsError('New immutable U-only adaptation output required')
    pm,state,tau,cal=_reference(c['reference_prediction_root']);identity=pm['model_identity'];core=_core()
    # Also protect the actual checkpoint/cache inputs inherited from the parent.
    for p in (pm['config']['train_config']['output_root'],pm['config']['source_IQ_root'],pm['config']['dura_release'],pm['config']['dura_checkpoint']):
        p=Path(p).resolve();o=out.resolve()
        if o.is_relative_to(p) or p.is_relative_to(o):raise ValueError('Output cannot overlap inherited model or source IQ inputs')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;b=_backend(pm,device)
    if b['model_identity']!=identity or b['reader_identity']!=pm['parent_F0_identity']:raise ValueError('Actual frozen P1 signal and original F0 must equal the completed parent')
    expected=pm['dura_identity'];actual=b['dura_identity']
    if any(actual.get(k)!=v or expected.get(k)!=v for k,v in dict(deployment_id='dura_v1',parameters=266951,epoch=20,updates=4940).items()) or any(actual.get(k)!=expected[k] for k in ('checkpoint','arm','training_commit','reuse','graph','native','precision','TF32') if k in expected):raise ValueError('Same original DURA v1 computation required')
    snapshots=[(m,best._snapshot_model(m)) for m in (b['reader'],b['model'],b['dura'][0])]
    frozen=deepcopy(state);frozen_tau=tau.copy();tau.setflags(write=False)
    def check():
        for m,before in snapshots:best._check_model(m,before)
        if not _same(state,frozen) or not np.array_equal(tau,frozen_tau):raise ValueError('Source model/statistics/thresholds changed')
    smoke=best._smoke(b,device);check()
    if smoke.get('status')!='PASS' or smoke.get('actual_F0_signal_DURA_checkpoints') is not True:raise ValueError('Actual three-checkpoint synthetic smoke must precede U')
    prior=core.build_prior(state);prior_before=deepcopy(prior)
    out.mkdir(parents=True,exist_ok=False);_write(out/'resolved_config.json',c);_write(out/'smoke.json',smoke);best._save(out/'models/density.npz',state);best._save(out/'thresholds.npz',dict(conditional=tau))
    _write(out/'models/complete.json',dict(schema=SCHEMA,status=SOURCE_STATUS,run_id=RUN_ID,code_commit=c['code_commit'],model_identity=identity,models_frozen=True,models_refitted=False,thresholds_refitted=False,
        parent_calibration_identity=cal,source_prior_frozen_before_U=True,U_reads=0,Q_reads=0))
    (out/'calibration').mkdir();_write(out/'calibration/complete.json',cal);best._save(out/'adaptation/source_prior.npz',prior)
    iq=None
    try:
        manifest,ids,iq=_public(c);blocks=[]
        for start in range(0,len(ids),128):
            a=best._current(b,iq[start:start+128],device,True);a.pop('baseline_closed');blocks.append(a);check()
            print('MIXTURE_UDA_U_PROGRESS',start+len(a['closed']),len(ids),flush=True)
        U={k:np.concatenate([a[k] for a in blocks]) for k in blocks[0]};U['ids']=ids.copy()
    finally:
        if hasattr(iq,'_mmap'):iq._mmap.close()
    adapters,diagnostics=core.fit_adapters(state,U['signal_features'],prior);check()
    if set(adapters)!=set(ADAPTERS):raise ValueError('Both preregistered U-only adapters must complete')
    for arm in ADAPTERS:
        core.validate_adapter(adapters[arm],prior)
        for value in adapters[arm].values():value.setflags(write=False)
        best._save(out/'adaptation'/(arm+'.npz'),adapters[arm])
    if not _same(prior,prior_before):raise ValueError('Source nuisance prior changed during U fit')
    adapter_before=deepcopy(adapters);best._save(out/'adaptation/U_features.npz',U);_write(out/'adaptation/fit_diagnostics.json',diagnostics)
    _write(out/'adaptation/complete.json',dict(schema=SCHEMA,status=ADAPTER_STATUS,run_id=RUN_ID,code_commit=c['code_commit'],U_count=U_COUNT,U_manifest=manifest,adapter_names=list(ADAPTERS),
        fit_input='U_signal_features_only',fit_diagnostics_ref='fit_diagnostics.json',iterations_per_adapter=20,source_prior_frozen_before_U=True,query_fit=False,true_U_labels_used=False,true_unknown_count_used=False,
        source_density_refitted=False,thresholds_refitted=False,adapters_frozen=True,Q_reads=0,source_prediction_reads=0,U_bank_deployed=False,private_U_features_ref='U_features.npz'))
    print(ADAPTER_STATUS,flush=True)
    def frozen_check():
        check()
        if not _same(prior,prior_before) or not _same(adapters,adapter_before):raise ValueError('Frozen U adapters changed on source/future rows')
    all_ids=set(ids.tolist());source_ids=set();six_tau=np.tile(tau,(3,1));deltas=[np.zeros(160,np.float64)]+[adapters[arm]['delta'] for arm in ADAPTERS]
    def prediction(parent,is_source):
        raw=parent['raw_scores']
        if not np.array_equal(parent['scores'],raw-tau[:,parent['closed']].T):raise ValueError('Parent operating points changed')
        if is_source:result=dict(parent,raw_scores=np.tile(raw,(1,3)),scores=np.tile(parent['scores'],(1,3)))
        else:
            if not np.allclose(core.scores(state,parent['signal_features'],deltas[0]),raw,rtol=1e-12,atol=1e-9):raise ValueError('Exact DA0 controls must reconstruct')
            six=np.concatenate([raw]+[core.scores(state,parent['signal_features'],delta) for delta in deltas[1:]],axis=1)
            result=dict(parent,raw_scores=six,scores=six-six_tau[:,parent['closed']].T)
            if not np.array_equal(result['scores'][:,:2],parent['scores']):raise ValueError('Exact DA0 margins changed')
        return result
    for role in ('known','novel'):
        a=_parent_rows(c['reference_prediction_root'],role);current=set(a['ids'].tolist())
        if current&(all_ids|source_ids):raise ValueError('Source known/novel and U physical roles overlap')
        source_ids.update(current);result=prediction(a,True);_validate(result,role);best._save(out/'source/source_DURA'/(role+'_predictions.npz'),result);frozen_check()
    for condition in CONDITIONS[1:]:
        a=_parent_target_rows(c['reference_prediction_root'],condition)
        if set(a['ids'].tolist())&(all_ids|source_ids):raise ValueError('Future Q must be physically disjoint from fit/calibration/U')
        result=prediction(a,False);_validate(result,'Q');best._save(out/'targets'/condition/'predictions.npz',result);frozen_check()
    done=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c,method_names=list(METHODS),primary_method=PRIMARY_METHOD,conditions=list(CONDITIONS),
        known_count=COUNTS['V'],novel_count=COUNTS['novel'],U_count=U_COUNT,Q_counts={name:Q_COUNT for name in CONDITIONS[1:]},U_manifest=manifest,Q_manifests=pm['Q_manifests'],
        reference_prediction_identity=pm,parent_calibration_identity=cal,model_identity=identity,parent_F0_identity=pm['parent_F0_identity'],dura_identity=expected,loaded_dura_identity=actual,
        classification_path='parent_original_F0_same_DURA_input',identity_baseline_path='parent_original_F0_same_DURA_input',classification_unchanged=True,score_threshold=0.,
        source_models_frozen_before_U=True,adapters_frozen_before_source_and_Q=True,all_three_DURA_conditions_complete=True,source_controls_reused_exactly=True,DA0_controls_reused_exactly=True,
        all_thresholds_reused_exactly=True,source_states_unchanged=True,frozen_target_adapters_unchanged=True,frozen_parameters_buffers_unchanged=True,
        source_IQ_reads=0,source_L_reads=0,source_OE_reads=0,target_Q_IQ_reads=0,target_U_reads=U_COUNT,original_U_s_reads=0,private_truth_reads=0,U_truth_reads=0,source_novel_truth_reads=0,target_truth_reads=0,
        new_neural_updates=0,native_updates=0,density_fit_updates=0,normalizer_fit_updates=0,threshold_fit_updates=0,adapter_iterations_per_arm=20,
        real_F_forward_rows=U_COUNT,real_signal_forward_rows=U_COUNT,G_forward_rows=U_COUNT,synthetic_F_forward_rows=2,synthetic_signal_forward_rows=2,G_synthetic_forward_rows=2,
        source_bank_deployed=False,U_bank_deployed=False,per_record_source_rows_deployed=0,formal_target_accessed=False,final_phase3_complete=False,full_goal_status='ACTIVE',A_N=None,H_OW=None)
    frozen_check();_write(out/'complete.json',done);print(STATUS,flush=True);return done


evaluate=run


def main(argv=None):
    print(json.dumps(dict(event='MIXTURE_UDA_PROCESS',pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv)),flush=True)
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args(argv);run(_read(a.config),a.device)


if __name__=='__main__':main()
