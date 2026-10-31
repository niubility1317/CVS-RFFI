"""Source-L MD/RMD readouts over fixed PCD features and original F0 identity."""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import numpy as np
import torch
from . import prototype_identity_run as inherited
from . import pcd_relative_metric_core as core

RUN_ID='20261031-phase3-pcd-relative-metric-s2026092701-r01'
SCHEMA='pamf_phase3_pcd_relative_metric_prediction_v1'
STATUS='PCD_RELATIVE_METRIC_PREDICTIONS_COMPLETE'
MODELS_STATUS='PCD_RELATIVE_METRIC_MODELS_FROZEN'
CALIBRATION_STATUS='PCD_RELATIVE_METRIC_THRESHOLDS_FROZEN'
METHODS=('OLD_PCD_pooled_ratio','OLD_PCD_predicted_ratio',*core.METHODS)
PRIMARY_METHOD=METHODS[-1]
CONDITIONS=('source_DURA','clean_DURA','star_DURA')
COUNTS=dict(inherited.COUNTS)
SOURCE_COUNT=6300
Q_COUNT=10238
ROW_KEYS=inherited.ROW_KEYS
FIELDS=inherited.FIELDS
BASE_FIELDS={'ids','default_logits','closed','baseline_closed','signal_features','raw_scores','scores'}
_read,_write,_same,_save,_npz=inherited._read,inherited._write,inherited._same,inherited._save,inherited._npz
_reference,_backend,_load_model,_smoke=inherited._reference,inherited._backend,inherited._load_model,inherited._smoke
risk=inherited.risk


def validate_config(c):
    if not isinstance(c,dict) or set(c)!=FIELDS or c.get('run_id')!=RUN_ID:raise ValueError('Exact six-field PCD-relative metric configuration required')
    inherited.validate_config(dict(c,run_id=inherited.RUN_ID))
    return c


def _source(c,pm):
    header,summary=inherited._geometry_header(c,pm)
    folder=Path(c['reference_config']['source_control_config']['reference_config']['train_config']['IQ_root'])/'L'
    ids=np.load(folder/'ids.npy',allow_pickle=False);meta=_npz(folder/'metadata.npz');manifest=_read(folder/'manifest.json')
    if manifest!=summary['source_manifest']:raise ValueError('Source geometry and source L handles differ')
    inherited.geometry._metadata(dict(ids=ids,metadata=meta))
    with np.load(Path(c['source_geometry_root'])/'source_features.npz',allow_pickle=False) as z:
        if set(z.files)!={'ids','rx_i','day_i','F0_closed','F0_unit','r03_unit','PCD_unit','PCD_white'}:raise ValueError('Exact cached source geometry fields required')
        fid=z['ids'].copy();features=z['PCD_unit'].copy();rx=z['rx_i'].copy();day=z['day_i'].copy()
    inherited.reg.core.base._ids(fid,SOURCE_COUNT);inherited._features(features)
    if set(fid.tolist())!=set(ids.tolist()):raise ValueError('Source feature physical IDs differ')
    lookup={v:i for i,v in enumerate(ids.tolist())};order=np.asarray([lookup[v] for v in fid.tolist()],np.int64)
    if not np.array_equal(rx,meta['rx_i'][order]) or not np.array_equal(day,meta['day_i'][order]):raise ValueError('Source feature RX/day alignment differs')
    return features,meta['y'][order],fid,header,manifest


def _parent_rows(c,key):return inherited.reg._parent_rows(c,key)


def _row(old,density,metric):
    compact=inherited.reg._compact(old);z=inherited._features(compact.pop('features'));compact.pop('base_margin')
    expected=inherited.reg.fresh._raw(density,z,compact['closed'])
    if not np.allclose(expected,old['raw_scores'][:,4:6],rtol=1e-12,atol=1e-9):raise ValueError('Original PCD density scores do not reconstruct')
    return dict(compact,signal_features=z,raw_scores=np.column_stack((old['raw_scores'][:,4:6],core.metric_scores(z,compact['closed'],metric))))


def _finish(a,tau,old):
    a['scores']=a['raw_scores']-tau[:,a['closed']].T
    if not np.array_equal(a['scores'][:,:2],old['scores'][:,4:6]):raise ValueError('Old PCD threshold decisions changed')
    return a


def _validate(a,role):
    n=COUNTS['V' if role=='known' else 'novel'] if role in ('known','novel') else Q_COUNT
    fields=BASE_FIELDS|({'y','rx_i','day_i','scenario'} if role=='known' else {'rx_i','day_i'} if role=='novel' else set())
    if set(a)!=fields:raise ValueError('Exact public/source-known fields required; no future truth')
    base=inherited.reg.core.base;base._ids(a['ids'],n);base._closed(a['closed'],n);base._closed(a['baseline_closed'],n);inherited._features(a['signal_features'])
    if a['signal_features'].shape!=(n,160) or a['default_logits'].shape!=(n,6) or a['default_logits'].dtype!=np.float32 or not np.isfinite(a['default_logits']).all():raise ValueError('Complete original F0/final PCD evidence required')
    for k in ('raw_scores','scores'):base._float(a[k],(n,6),k)
    if not np.array_equal(a['closed'],a['default_logits'].argmax(1)) or not np.array_equal(a['closed'],a['baseline_closed']):raise ValueError('Original F0 identity must route every readout')
    return a


def _rows(root,role):
    if role not in ('known','novel'):raise ValueError('Fixed source role required')
    return _validate(_npz(Path(root)/'predictions'/('source_V.npz' if role=='known' else 'source_NV.npz')),role)


def _target_rows(root,condition):
    if condition not in CONDITIONS[1:]:raise ValueError('Fixed future DURA condition required')
    return _validate(_npz(Path(root)/'predictions'/('Q_'+condition+'.npz')),'Q')


def run(c,device='cuda:0'):
    validate_config(c);out=Path(c['output_root'])
    if out.exists():raise FileExistsError('New immutable PCD-relative metric root required')
    pm,density,old_tau,old_cal=_reference(c);b=_backend(c,device);model,identity=_load_model(c,device)
    if (identity!=pm['model_identities']['TARGET_PCD'] or b['reader_identity']!=pm['parent_F0_identity']
            or b['model_identity']!=pm['model_identities']['E0'] or b['dura_identity']!=pm['loaded_dura_identity']):raise ValueError('Actual frozen F0/r03/PCD/DURA ancestry differs')
    models=(b['reader'],b['model'],model,b['dura'][0]);snapshots=[(m,inherited.reg.fresh.best._snapshot_model(m)) for m in models]
    fixed_density=deepcopy(density);fixed_old=old_tau.copy()
    def check():
        for m,snapshot in snapshots:inherited.reg.fresh.best._check_model(m,snapshot)
        if not _same(density,fixed_density) or not np.array_equal(old_tau,fixed_old):raise ValueError('Original density or old thresholds changed')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    smoke=_smoke(b,model,device);check()
    if smoke.get('status')!='PASS':raise ValueError('Actual checkpoint smoke required before source features')
    features,y,Lids,geo,manifest=_source(c,pm);metric=core.fit_metric(features,y);core.validate_metric(metric);fixed_metric=deepcopy(metric)
    out.mkdir(parents=True);_write(out/'resolved_config.json',c);_write(out/'smoke.json',smoke)
    _save(out/'models/density.npz',density);_save(out/'models/metric.npz',metric)
    _write(out/'models/complete.json',dict(schema=SCHEMA,status=MODELS_STATUS,run_id=RUN_ID,code_commit=c['code_commit'],model_identity=identity,source_geometry_identity=geo,
        source_L_manifest=manifest,models_frozen=True,models_frozen_before_V=True,source_metric_fits=1,source_L_feature_reads=SOURCE_COUNT,original_density_refitted=False,
        source_V_fit=False,novel_fit=False,target_Q_fit=False,metric_state_bytes=sum(v.nbytes for v in metric.values())))
    V=_parent_rows(c,'source_V');a=_row(V,density,metric);tau=np.empty((6,6),np.float64);tau[:2]=old_tau
    reports={METHODS[j]:deepcopy(old_cal['methods'][inherited.reg.fresh.METHODS[4+j]]) for j in range(2)}
    for j in range(2,6):
        fitted,report=risk.calibrate_known(a['ids'],a['y'],a['closed'],a['baseline_closed'],a['raw_scores'][:,j],a['rx_i'],a['scenario']);tau[j]=fitted['conditional_thresholds'];reports[METHODS[j]]=report
    _save(out/'thresholds.npz',dict(conditional=tau));(out/'calibration').mkdir()
    _write(out/'calibration/complete.json',dict(schema=SCHEMA,status=CALIBRATION_STATUS,run_id=RUN_ID,code_commit=c['code_commit'],method_names=list(METHODS),primary_method=PRIMARY_METHOD,
        methods=reports,conditional_thresholds=tau.tolist(),known_only=True,unknown_inputs_used=False,target_reads=0,source_V_neural_fit=False,source_V_density_fit=False,source_V_threshold_fit=True,
        group_count=42,hard_names=list(risk.HARD_NAMES),diagnostic_names=list(risk.DIAGNOSTIC_NAMES)))
    fixed_tau=tau.copy();used=set(Lids.tolist());paired=None
    for key in ROW_KEYS:
        parent=V if key=='source_V' else _parent_rows(c,key);row=_finish(a if key=='source_V' else _row(parent,density,metric),tau,parent)
        role='known' if key=='source_V' else 'novel' if key=='source_NV' else 'Q';_validate(row,role);ids=set(row['ids'].tolist())
        if ids&used:raise ValueError('Source L/V/NV and future physical roles overlap')
        if key.startswith('source'):used.update(ids)
        elif paired is None:paired=row['ids'].copy()
        elif not np.array_equal(paired,row['ids']):raise ValueError('Paired future clean/star order differs')
        _save(out/'predictions'/(key+'.npz'),row);check()
        if not _same(metric,fixed_metric) or not np.array_equal(tau,fixed_tau):raise ValueError('Frozen source metric or thresholds changed')
    done=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c,method_names=list(METHODS),primary_method=PRIMARY_METHOD,conditions=list(CONDITIONS),
        reference_prediction_identity=pm,source_geometry_identity=geo,model_identity=identity,parent_F0_identity=pm['parent_F0_identity'],dura_identity=pm['dura_identity'],source_L_manifest=manifest,
        known_count=COUNTS['V'],novel_count=COUNTS['novel'],Q_counts={n:Q_COUNT for n in CONDITIONS[1:]},models_frozen_before_V=True,all_thresholds_frozen_before_novel_and_Q=True,
        all_three_DURA_conditions_complete=True,all_conditions_through_DURA=True,classification_unchanged=True,controls_reused_exactly=True,source_metric_fits=1,new_threshold_sets_fitted=4,
        source_L_feature_reads=SOURCE_COUNT,private_truth_reads=0,U_truth_reads=0,target_U_reads=0,target_truth_reads=0,source_novel_truth_reads=0,source_OE_reads=0,original_U_s_reads=0,
        real_signal_forward_rows=0,real_F_forward_rows=0,G_forward_rows=0,optimizer_updates=0,new_neural_updates=0,new_GMM_fits=0,original_models_unchanged=True,source_states_unchanged=True,
        synthetic_F_forward_rows=smoke['synthetic_F_forward_rows'],synthetic_signal_forward_rows=smoke['synthetic_signal_forward_rows'],G_synthetic_forward_rows=smoke['G_synthetic_forward_rows'],
        source_only=False,PCD_U_ancestor=576,new_target_DA=False,target_REG=False,A_N=None,H_OW=None,source_bank_deployed=False,metric_state_bytes=sum(v.nbytes for v in metric.values()),
        threshold_values=36,threshold_bytes=tau.nbytes,new_threshold_values=24,new_threshold_bytes=tau[2:].nbytes,formal_target_accessed=False,final_phase3_complete=False,full_goal_status='ACTIVE')
    _write(out/'complete.json',done);print(STATUS,flush=True);return done


def _prediction_gate(root):
    root=Path(root);done=_read(root/'complete.json');c=validate_config(done['config']);pm,density,old_tau,old_cal=_reference(c);geo,_=inherited._geometry_header(c,pm)
    required=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=c['code_commit'],method_names=list(METHODS),primary_method=PRIMARY_METHOD,reference_prediction_identity=pm,source_geometry_identity=geo,
        models_frozen_before_V=True,all_thresholds_frozen_before_novel_and_Q=True,all_three_DURA_conditions_complete=True,all_conditions_through_DURA=True,classification_unchanged=True,
        controls_reused_exactly=True,source_metric_fits=1,new_threshold_sets_fitted=4,source_L_feature_reads=SOURCE_COUNT,private_truth_reads=0,U_truth_reads=0,target_U_reads=0,
        original_models_unchanged=True,real_signal_forward_rows=0,real_F_forward_rows=0,G_forward_rows=0,optimizer_updates=0,final_phase3_complete=False)
    if Path(c['output_root']).resolve()!=root.resolve() or any(done.get(k)!=v for k,v in required.items()):raise ValueError('Complete frozen metric predictions required before truth')
    if not _same(_npz(root/'models/density.npz'),density):raise ValueError('Old PCD density changed')
    metric=core.validate_metric(_npz(root/'models/metric.npz'));fit=_read(root/'models/complete.json')
    if (fit.get('status')!=MODELS_STATUS or fit.get('model_identity')!=pm['model_identities']['TARGET_PCD'] or fit.get('source_geometry_identity')!=geo
            or fit.get('models_frozen_before_V') is not True or fit.get('source_V_fit') is not False or fit.get('novel_fit') is not False or fit.get('target_Q_fit') is not False):raise ValueError('Source metric ancestry or freeze changed')
    thresholds=_npz(root/'thresholds.npz');tau=thresholds.get('conditional');cal=_read(root/'calibration/complete.json')
    if (set(thresholds)!={'conditional'} or tau.shape!=(6,6) or tau.dtype!=np.float64 or not np.isfinite(tau).all() or not np.array_equal(tau[:2],old_tau)
            or cal.get('status')!=CALIBRATION_STATUS or cal.get('conditional_thresholds')!=tau.tolist() or cal.get('known_only') is not True or cal.get('unknown_inputs_used') is not False):raise ValueError('Six legal known-only thresholds required')
    for j,name in enumerate(METHODS):
        if cal['methods'][name]['conditional_thresholds']!=tau[j].tolist() or (j<2 and cal['methods'][name]!=old_cal['methods'][inherited.reg.fresh.METHODS[4+j]]):raise ValueError('Old/new calibration routing differs')
    rows={}
    for key in ROW_KEYS:
        parent=_parent_rows(c,key);actual=_rows(root,'known' if key=='source_V' else 'novel') if key.startswith('source') else _target_rows(root,key[2:])
        expected=_finish(_row(parent,density,metric),tau,parent)
        for field,value in expected.items():
            same=(np.allclose(actual[field],value,rtol=1e-12,atol=1e-8) if field=='raw_scores'
                else np.array_equal(actual[field][:,:2],value[:,:2]) if field=='scores' else np.array_equal(actual[field],value))
            if not same:raise ValueError('F0 identity, cached controls or MD/RMD reconstruction changed')
        if not np.array_equal(actual['scores'],actual['raw_scores']-tau[:,actual['closed']].T):raise ValueError('Original predicted-class threshold routing changed')
        rows[key]=actual
    return done,cal,rows


evaluate=run


def main():
    print(json.dumps(dict(event='PCD_RELATIVE_METRIC_PROCESS',pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv)),flush=True)
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args();run(_read(a.config),a.device)


if __name__=='__main__':main()
