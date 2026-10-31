"""Source-L prototype identity and own-class-routed frozen PCD density ratios.

No new neural forward on real inputs, density fit, or target adaptation occurs.
Original F0 logits remain evidence/baseline; prototype distances are not logits.
"""
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
from . import pcd_cluster_registry_run as reg
from . import source_cluster_geometry as geometry
from . import star_macro12_risk as risk

RUN_ID='20261031-phase3-prototype-identity-s2026092701-r01'
SCHEMA='pamf_phase3_prototype_identity_prediction_v1'
STATUS='PROTOTYPE_IDENTITY_PREDICTIONS_COMPLETE'
MODELS_STATUS='PROTOTYPE_IDENTITY_MODELS_FROZEN'
CALIBRATION_STATUS='PROTOTYPE_IDENTITY_THRESHOLDS_FROZEN'
PROTOTYPE_SCHEMA='pamf_phase3_prototype_identity_means_v1'
REFERENCE_CODE=reg.REFERENCE_CODE
GEOMETRY_CODE='074a84caa44387e5ad8472ab8f217c92150421fd'
ARMS=('OLD_PCD','PROTO')
METHODS=tuple(arm+'_'+method for arm in ARMS for method in ('pooled_ratio','predicted_ratio'))
PRIMARY_METHOD=METHODS[-1]
PRIMARY_INDEX=3
FIELDS={'run_id','code_commit','reference_config','source_geometry_root','output_root','batch_size'}
COUNTS=dict(reg.COUNTS)
SOURCE_COUNT=6300
Q_COUNT=reg.Q_COUNT
ROW_KEYS=reg.ROW_KEYS
BASE_FIELDS={'ids','default_logits','closed','baseline_closed','signal_features','prototype_distances','raw_scores','scores'}
_read,_write,_same=reg._read,reg._write,reg._same
_reference,_backend,_load_model,_smoke=reg._reference,reg._backend,reg._load_model,reg._smoke
_save,_npz=reg.fresh.best._save,reg._npz


def validate_config(c):
    if (not isinstance(c,dict) or set(c)!=FIELDS or c.get('run_id')!=RUN_ID or re.fullmatch('[0-9a-f]{40}',str(c.get('code_commit'))) is None
            or type(c.get('batch_size')) is not int or c['batch_size']!=128
            or any(not isinstance(c.get(k),str) or not c[k].strip() for k in ('source_geometry_root','output_root'))):raise ValueError('Exact six-field frozen prototype identity configuration required')
    p=reg.fresh.validate_config(c['reference_config'])
    if p['code_commit']!=REFERENCE_CODE:raise ValueError('Exact completed fresh f4b5 PCD reference required')
    roots=[c['source_geometry_root'],p['output_root'],p['train_config']['output_root'],p['train_config']['U_public_root'],p['source_control_config']['output_root'],
        p['source_control_config']['reference_config']['train_config']['IQ_root']]+[v['public_root'] for v in p['Q_conditions'].values()]
    out=Path(c['output_root']).resolve()
    if any(out.is_relative_to(Path(v).resolve()) or Path(v).resolve().is_relative_to(out) for v in roots):raise ValueError('Immutable prototype output overlaps source/model/public inputs')
    return c


def _features(z):
    z=np.asarray(z)
    if z.ndim!=2 or z.shape[1]!=160 or not len(z) or z.dtype!=np.float32 or not np.isfinite(z).all() or not np.allclose(np.linalg.norm(z.astype(np.float64),axis=1),1.,rtol=0,atol=2e-6):raise ValueError('Final PCD finite FP32 unit160 required')
    return z


def validate_prototypes(p):
    if not isinstance(p,dict) or set(p)!={'schema','means'} or np.asarray(p['schema']).shape!=() or np.asarray(p['schema']).dtype.kind!='U' or p['schema'].item()!=PROTOTYPE_SCHEMA:raise ValueError('Exact six-source-mean aggregate required')
    means=reg.core.base._float(p['means'],(6,160),'source PCD unit-feature means')
    if np.any(np.linalg.norm(means,axis=1)>1+2e-6):raise ValueError('Unnormalized arithmetic means of unit source features required')
    return p


def fit_prototypes(features,y):
    z=_features(features).astype(np.float64);y=reg.core.base._closed(y,len(z))
    if set(y.tolist())!=set(range(6)):raise ValueError('All six registered source L classes required')
    return validate_prototypes(dict(schema=np.asarray(PROTOTYPE_SCHEMA),means=np.stack([z[y==c].mean(0) for c in range(6)])))


def classify(features,p):
    validate_prototypes(p);z=reg.core.density.unit(_features(features))
    distances=np.column_stack([np.linalg.norm(z-mean,axis=1) for mean in p['means']])
    return distances.argmin(1).astype(np.int64),distances


def _geometry_header(c,pm):
    root=Path(c['source_geometry_root']);m=_read(root/'complete.json');summary=_read(root/'summary.json')
    required=dict(schema=geometry.SCHEMA,status=geometry.STATUS,run_id=geometry.RUN_ID,code_commit=GEOMETRY_CODE,source_L_reads=SOURCE_COUNT,
        source_only_inputs=True,model_source_only=False,PCD_target_U_ancestor_count=576,original_models_unchanged=True,source_states_unchanged=True,
        private_truth_reads=0,target_U_reads=0,target_Q_reads=0)
    expected=dict(PCD=pm['model_identities']['TARGET_PCD'],F0=pm['parent_F0_identity'],r03=pm['model_identities']['E0'],DURA=pm['dura_identity'])
    if any(m.get(k)!=v for k,v in required.items()) or m.get('config',{}).get('reference_config')!=c['reference_config'] or summary.get('checkpoint_identities')!=expected:raise ValueError('Source geometry must use the exact selected final PCD/F0/r03/G reference')
    return m,summary


def _source(c,pm):
    """Only source public IDs/known metadata and cached final-PCD features, no IQ."""
    header,summary=_geometry_header(c,pm);r=c['reference_config'];folder=Path(r['source_control_config']['reference_config']['train_config']['IQ_root'])/'L'
    ids=np.load(folder/'ids.npy',allow_pickle=False);meta=_npz(folder/'metadata.npz');manifest=_read(folder/'manifest.json')
    if manifest!=summary['source_manifest']:raise ValueError('Source geometry and original source L handles differ')
    geometry._metadata(dict(ids=ids,metadata=meta))
    with np.load(Path(c['source_geometry_root'])/'source_features.npz',allow_pickle=False) as z:
        expected={'ids','rx_i','day_i','F0_closed','F0_unit','r03_unit','PCD_unit','PCD_white'}
        if set(z.files)!=expected:raise ValueError('Only the original source geometry feature artifact is allowed')
        feature_ids=z['ids'];features=z['PCD_unit'];rx=z['rx_i'];day=z['day_i']
    reg.core.base._ids(feature_ids,SOURCE_COUNT);_features(features)
    if set(feature_ids.tolist())!=set(ids.tolist()):raise ValueError('Source feature physical IDs differ from source L')
    lookup={v:i for i,v in enumerate(ids.tolist())};order=np.asarray([lookup[v] for v in feature_ids.tolist()],np.int64)
    if not np.array_equal(rx,meta['rx_i'][order]) or not np.array_equal(day,meta['day_i'][order]):raise ValueError('Source feature RX/day alignment differs')
    return fit_prototypes(features,meta['y'][order]),header,manifest,np.bincount(meta['y'],minlength=6).tolist(),feature_ids


def _parent_rows(c,key):return reg._parent_rows(c,key)


def _pair(old,state,p):
    a=reg._compact(old);features=_features(a['features']);closed,distances=classify(features,p)
    common=dict(ids=a['ids'],default_logits=a['default_logits'],baseline_closed=a['baseline_closed'],signal_features=features,prototype_distances=distances)
    for k in ('y','rx_i','day_i','scenario'):
        if k in a:common[k]=a[k]
    expected=reg.fresh._raw(state,features,a['closed'])
    if not np.allclose(expected,old['raw_scores'][:,4:6],atol=1e-9,rtol=1e-12):raise ValueError('Original PCD source density/control raw scores do not reconstruct')
    return dict(OLD_PCD=dict(common,closed=a['closed'],raw_scores=old['raw_scores'][:,4:6].copy(),scores=old['scores'][:,4:6].copy()),
        PROTO=dict(common,closed=closed,raw_scores=reg.fresh._raw(state,features,closed)))


def _finish(pair,tau):
    for j,arm in enumerate(ARMS):
        a=pair[arm];scores=a['raw_scores']-tau[2*j:2*j+2,a['closed']].T
        if arm=='OLD_PCD' and not np.array_equal(a['scores'],scores):raise ValueError('Original two PCD threshold operating points changed')
        a['scores']=scores
    return pair


def _validate(a,n,arm,role):
    fields=BASE_FIELDS|({'y','rx_i','day_i','scenario'} if role=='known' else {'rx_i','day_i'} if role=='novel' else set())
    if set(a)!=fields:raise ValueError('Exact public/known-source row fields required; no target truth')
    reg.core.base._ids(a['ids'],n);reg.core.base._closed(a['closed'],n);reg.core.base._closed(a['baseline_closed'],n);_features(a['signal_features'])
    if a['signal_features'].shape!=(n,160) or a['default_logits'].shape!=(n,6) or a['default_logits'].dtype!=np.float32 or not np.isfinite(a['default_logits']).all():raise ValueError('Frozen original F0 logits and final PCD features required')
    for k,shape in (('prototype_distances',(n,6)),('raw_scores',(n,2)),('scores',(n,2))):reg.core.base._float(a[k],shape,k)
    if not np.array_equal(a['baseline_closed'],a['default_logits'].argmax(1)) or not np.array_equal(a['closed'],a['baseline_closed'] if arm=='OLD_PCD' else a['prototype_distances'].argmin(1)):raise ValueError('Own-arm identity routing differs from original F0/prototype evidence')
    return a


def _rows(root,role,arm):
    if role not in ('known','novel') or arm not in ARMS:raise ValueError('Fixed source role and identity arm required')
    return _validate(_npz(Path(root)/'source'/arm/(role+'_predictions.npz')),COUNTS['V' if role=='known' else 'novel'],arm,role)


def _target_rows(root,condition,arm):
    if condition not in ('clean_DURA','star_DURA') or arm not in ARMS:raise ValueError('Fixed clean/star-DURA condition and identity arm required')
    return _validate(_npz(Path(root)/'targets'/condition/arm/'predictions.npz'),Q_COUNT,arm,'Q')


def run(c,device='cuda:0'):
    validate_config(c);out=Path(c['output_root'])
    if out.exists():raise FileExistsError('New immutable prototype identity output required')
    pm,state,old_tau,old_cal=_reference(c);b=_backend(c,device);model,identity=_load_model(c,device)
    if identity!=pm['model_identities']['TARGET_PCD'] or b['reader_identity']!=pm['parent_F0_identity'] or b['model_identity']!=pm['model_identities']['E0'] or b['dura_identity']!=pm['loaded_dura_identity']:raise ValueError('Actual F0/r03/PCD/DURA ancestry differs')
    models=(b['reader'],b['model'],model,b['dura'][0]);snapshots=[(m,reg.fresh.best._snapshot_model(m)) for m in models];original=deepcopy(state);fixed_old=old_tau.copy()
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    def check():
        for m,snapshot in snapshots:reg.fresh.best._check_model(m,snapshot)
        if not _same(state,original) or not np.array_equal(old_tau,fixed_old):raise ValueError('Original Gaussian/threshold state changed')
    smoke=_smoke(b,model,device);check()
    if smoke.get('status')!='PASS':raise ValueError('Actual checkpoint no-query smoke must pass first')
    print('PROTOTYPE_IDENTITY_SMOKE_PASS',flush=True);p,geo,manifest,class_counts,Lids=_source(c,pm);check();fixed_p=deepcopy(p)
    out.mkdir(parents=True);_write(out/'resolved_config.json',c);_write(out/'smoke.json',smoke);_save(out/'models/density.npz',state);_save(out/'models/prototypes.npz',p)
    _write(out/'models/complete.json',dict(schema=SCHEMA,status=MODELS_STATUS,run_id=RUN_ID,code_commit=c['code_commit'],model_identity=identity,source_geometry_identity=geo,source_L_manifest=manifest,
        models_frozen=True,original_density_refitted=False,source_prototype_fits=1,source_L_feature_reads=SOURCE_COUNT,class_counts=class_counts,source_prototypes_frozen_before_V=True,
        source_V_fit=False,source_OE_fit=False,novel_fit=False,target_Q_fit=False,prototype_means_normalized=False,prototype_parameters=960,prototype_bytes=p['means'].nbytes))
    print(MODELS_STATUS,flush=True);tau=np.empty((4,6),np.float64);tau[:2]=old_tau
    reports={METHODS[j]:deepcopy(old_cal['methods'][reg.fresh.METHODS[4+j]]) for j in range(2)}
    V=_parent_rows(c,'source_V');pair=_pair(V,state,p);new=pair['PROTO']
    for j in range(2):
        fitted,report=risk.calibrate_known(new['ids'],new['y'],new['closed'],new['baseline_closed'],new['raw_scores'][:,j],new['rx_i'],new['scenario']);tau[2+j]=fitted['conditional_thresholds'];reports[METHODS[2+j]]=report;check()
    frozen_tau=tau.copy();_save(out/'thresholds.npz',dict(conditional=tau))
    cal=dict(schema=SCHEMA,status=CALIBRATION_STATUS,run_id=RUN_ID,code_commit=c['code_commit'],method_names=list(METHODS),primary_method=PRIMARY_METHOD,methods=reports,conditional_thresholds=tau.tolist(),
        known_count=COUNTS['V'],known_only=True,unknown_inputs_used=False,hard_names=list(risk.HARD_NAMES),diagnostic_names=list(risk.DIAGNOSTIC_NAMES),group_count=42,source_V_threshold_fit=True,
        source_V_neural_fit=False,source_V_density_fit=False,identity_routing='own arm predicted class',protection_baseline='original F0 same DURA closed')
    (out/'calibration').mkdir();_write(out/'calibration/complete.json',cal);print(CALIBRATION_STATUS,flush=True)
    seen=set(Lids.tolist());paired_ids=None
    for key in ROW_KEYS:
        old=V if key=='source_V' else _parent_rows(c,key);pair=_finish(pair if key=='source_V' else _pair(old,state,p),tau);ids=pair['OLD_PCD']['ids']
        if set(ids.tolist())&seen:raise ValueError('Source support/calibration/novel and future Q roles overlap')
        if key.startswith('source'):seen.update(ids.tolist())
        elif paired_ids is None:paired_ids=ids.copy()
        elif not np.array_equal(paired_ids,ids):raise ValueError('Same canonical future clean/star order required')
        for arm in ARMS:
            role='known' if key=='source_V' else 'novel' if key=='source_NV' else 'Q';n=COUNTS['V' if role=='known' else 'novel'] if role!='Q' else Q_COUNT;_validate(pair[arm],n,arm,role)
            path=out/'source'/arm/(role+'_predictions.npz') if role!='Q' else out/'targets'/key[2:]/arm/'predictions.npz';_save(path,pair[arm])
        check()
        if not _same(p,fixed_p) or not np.array_equal(tau,frozen_tau):raise ValueError('Frozen prototype/threshold state changed')
        print('PROTOTYPE_IDENTITY_PREDICTIONS_SAVED',key,len(ids),flush=True)
    done=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c,method_names=list(METHODS),primary_method=PRIMARY_METHOD,reference_prediction_identity=pm,
        source_geometry_identity=geo,model_identity=identity,parent_F0_identity=pm['parent_F0_identity'],dura_identity=pm['dura_identity'],source_L_manifest=manifest,known_count=COUNTS['V'],novel_count=COUNTS['novel'],
        Q_counts={name:Q_COUNT for name in ('clean_DURA','star_DURA')},source_L_feature_reads=SOURCE_COUNT,source_prototype_fits=1,new_threshold_sets_fitted=2,new_density_fits=0,optimizer_updates=0,
        real_F_forward_rows=0,real_signal_forward_rows=0,G_forward_rows=0,synthetic_F_forward_rows=smoke['synthetic_F_forward_rows'],synthetic_signal_forward_rows=smoke['synthetic_signal_forward_rows'],G_synthetic_forward_rows=smoke['G_synthetic_forward_rows'],
        source_prototypes_frozen_before_V=True,all_thresholds_frozen_before_novel_and_Q=True,all_three_DURA_conditions_complete=True,all_conditions_through_DURA=True,controls_reused_exactly=True,
        original_models_unchanged=True,source_states_unchanged=True,prototype_state_unchanged=True,classification_unchanged=False,identity_baseline_path='original_F0_same_DURA',
        private_truth_reads=0,U_truth_reads=0,target_U_reads=0,target_truth_reads=0,source_novel_truth_reads=0,source_OE_reads=0,original_U_s_reads=0,new_UDA_updates=0,target_DA_inherited=True,target_REG=False,
        PCD_U_ancestor=576,source_only=False,source_bank_deployed=False,per_record_source_rows_deployed=0,new_neural_updates=0,prototype_bytes=p['means'].nbytes,new_threshold_bytes=tau[2:].nbytes,primary_threshold_bytes=tau[3].nbytes,
        comparison_loaded_parameters=sum(sum(v.numel() for v in m.parameters()) for m in models),minimal_primary_neural_parameters=601672,F0_retained_for_baseline=True,deployment_latency=None,
        formal_target_accessed=False,A_N=None,H_OW=None,final_phase3_complete=False,full_goal_status='ACTIVE')
    check();_write(out/'complete.json',done);print(STATUS,flush=True);return done


def _prediction_gate(root):
    root=Path(root);done=_read(root/'complete.json');c=validate_config(done['config']);pm,state,old_tau,old_cal=_reference(c);geo,_=_geometry_header(c,pm)
    required=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=c['code_commit'],method_names=list(METHODS),primary_method=PRIMARY_METHOD,reference_prediction_identity=pm,source_geometry_identity=geo,
        source_prototypes_frozen_before_V=True,all_thresholds_frozen_before_novel_and_Q=True,all_three_DURA_conditions_complete=True,all_conditions_through_DURA=True,controls_reused_exactly=True,
        original_models_unchanged=True,source_states_unchanged=True,prototype_state_unchanged=True,classification_unchanged=False,source_L_feature_reads=SOURCE_COUNT,source_prototype_fits=1,new_threshold_sets_fitted=2,
        real_F_forward_rows=0,real_signal_forward_rows=0,G_forward_rows=0,optimizer_updates=0,new_density_fits=0,private_truth_reads=0,U_truth_reads=0,target_U_reads=0,new_UDA_updates=0,target_DA_inherited=True,target_REG=False,final_phase3_complete=False)
    if Path(c['output_root']).resolve()!=root.resolve() or any(done.get(k)!=v for k,v in required.items()):raise ValueError('Complete frozen three-condition prototype predictions required before truth')
    if not _same(_npz(root/'models/density.npz'),state):raise ValueError('Original PCD density changed')
    p=validate_prototypes(_npz(root/'models/prototypes.npz'));fit=_read(root/'models/complete.json');tau=_npz(root/'thresholds.npz')['conditional'];cal=_read(root/'calibration/complete.json')
    if fit.get('status')!=MODELS_STATUS or fit.get('source_geometry_identity')!=geo or fit.get('model_identity')!=pm['model_identities']['TARGET_PCD'] or fit.get('source_prototypes_frozen_before_V') is not True:raise ValueError('Source prototype model ancestry/freeze differs')
    if tau.shape!=(4,6) or tau.dtype!=np.float64 or not np.isfinite(tau).all() or not np.array_equal(tau[:2],old_tau) or cal.get('status')!=CALIBRATION_STATUS or cal.get('conditional_thresholds')!=tau.tolist() or cal.get('known_only') is not True or cal.get('unknown_inputs_used') is not False:raise ValueError('Four fixed own-identity threshold vectors required')
    for j,name in enumerate(METHODS):
        report=cal['methods'][name]
        if report['conditional_thresholds']!=tau[j].tolist() or (j<2 and report!=old_cal['methods'][reg.fresh.METHODS[4+j]]):raise ValueError('Original/new calibration identity differs')
    rows={}
    for key in ROW_KEYS:
        old=_parent_rows(c,key);expected=_finish(_pair(old,state,p),tau);pair={}
        for arm in ARMS:
            a=_rows(root,'known' if key=='source_V' else 'novel',arm) if key.startswith('source') else _target_rows(root,key[2:],arm)
            if not _same(a,expected[arm]):raise ValueError('Original controls or prototype identities/density/threshold routing changed')
            pair[arm]=a
        rows[key]=pair
    return done,state,p,tau,cal,rows


evaluate=run


def main():
    print(json.dumps(dict(event='PROTOTYPE_IDENTITY_PROCESS',pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv)),flush=True)
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--device',default='cuda:0');args=parser.parse_args();run(_read(args.config),args.device)


if __name__=='__main__':main()
