"""Source-only class-conditional evidence with frozen F0 and cached star IQ.

L alone fits two path geometries. L/OE membership fits two matched CPU models.
Known V calibrates four fixed inference rules; novel inputs first open after
all conditional thresholds freeze. Identity always uses original F0 P1.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from . import star_paired_run as previous
from . import star_ova_evidence as core
from . import star_class_risk as risk

RUN_ID='20261029-phase3-star-ova-evidence-s2026092701-r01'
SCHEMA='pamf_phase3_star_ova_prediction_v1'
STATUS='STAR_OVA_PREDICTIONS_COMPLETE'
CALIBRATION_STATUS='STAR_OVA_THRESHOLDS_FROZEN'
READOUT_SCHEMA='pamf_phase3_star_ova_readouts_v1'
DESCRIPTOR_SCHEMA='class_conditional_paired_branch_evidence_v1'
MODEL_NAMES=('membership','ova_hard')
METHODS=('membership_routed','membership_max','ova_routed','ova_max')
PRIMARY_METHOD='ova_max'
CLASSIFICATION_PATH='P1'
REJECTION_INPUT_PATHS=('P1','P0')
COUNTS=dict(previous.COUNTS)
FIELDS=previous.FIELDS
KNOWN_FIELDS=previous.KNOWN_FIELDS|{'class_scores'}
NOVEL_FIELDS=previous.NOVEL_FIELDS|{'class_scores'}
geometry_core=previous.geometry_core
_backend,_smoke=previous._backend,previous._smoke
_role,_values,_branches=previous._role,previous._values,previous._branches
_snapshot,_check_models,_disjoint=previous._snapshot,previous._check_models,previous._disjoint
_encode,_write,_npz,close_role=previous._encode,previous._write,previous._npz,previous.close_role
_save_geometry=previous._save_geometry


def validate_config(config):
    if (not isinstance(config,dict) or set(config)!=FIELDS or config.get('run_id')!=RUN_ID
            or not isinstance(config.get('source_config'),dict)
            or config['source_config'].get('run_id')!=RUN_ID
            or config['source_config'].get('row_id')!='star-ova-s2026092701'):
        raise ValueError('Only the fixed OVA source run and six-field configuration are allowed')
    # Reuse input validation only; no previous run state or increment is opened.
    mapped=dict(config,run_id=previous.RUN_ID,source_config=dict(config['source_config'],
        run_id=previous.RUN_ID,row_id='star-paired-s2026092701'))
    previous.validate_config(mapped)
    return config


def _training_complete(state):
    core._state(state)
    training=state.get('training',{})
    if (state.get('schema')!=READOUT_SCHEMA or state.get('descriptor_schema')!=DESCRIPTOR_SCHEMA
            or state.get('descriptor_dim')!=106 or state.get('class_count')!=6
            or state.get('model_names')!=list(MODEL_NAMES)
            or training.get('model_seed')!=2026092701 or training.get('steps_per_arm')!=1000
            or training.get('total_optimizer_updates')!=2000
            or training.get('known_count')!=COUNTS['L'] or training.get('oe_count')!=COUNTS['OE']
            or training.get('initial_parameters_equal') is not True
            or training.get('parameter_counts')!={name:3457 for name in MODEL_NAMES}
            or set(training.get('losses',{}))!=set(MODEL_NAMES)
            or any(len(trace)!=1000 or not np.isfinite(trace).all() for trace in training['losses'].values())):
        raise ValueError('Two matched 3457-parameter source models must complete exactly 1000 updates each')


def _save_readouts(path,state):
    core._state(state)
    arrays=dict(schema=np.asarray(state['schema']),descriptor_schema=np.asarray(state['descriptor_schema']),
        descriptor_dim=np.asarray(state['descriptor_dim'],np.int64),class_count=np.asarray(state['class_count'],np.int64),
        model_names=np.asarray(state['model_names']),mean=state['mean'],scale=state['scale'])
    for name,model in state['models'].items():
        for key,value in model.items():arrays['model__'+name+'__'+key]=value
    _npz(path,arrays)


def load_readouts(path,training_json):
    """Load the two source aggregate models without IQ, feature banks or truth."""
    shapes={'mean':(np.float64,(106,)),'scale':(np.float64,(106,))}
    shapes.update({'model__'+name+'__'+key:(np.float32,shape) for name in MODEL_NAMES
        for key,shape in (('weight1',(32,106)),('bias1',(32,)),('weight2',(1,32)),('bias2',(1,)))})
    metadata={'schema','descriptor_schema','descriptor_dim','class_count','model_names'}
    with np.load(path,allow_pickle=False) as z:
        if set(z.files)!=set(shapes)|metadata:raise ValueError('Complete aggregate-only OVA readout fields required')
        for key,value in (('schema',READOUT_SCHEMA),('descriptor_schema',DESCRIPTOR_SCHEMA)):
            if z[key].shape!=() or z[key].dtype.kind!='U' or z[key].item()!=value:
                raise ValueError('OVA state/descriptor schema differs')
        for key,value in (('descriptor_dim',106),('class_count',6)):
            if z[key].shape!=() or z[key].dtype!=np.int64 or z[key].item()!=value:
                raise ValueError('The fixed 106D, six-class OVA descriptor is required')
        if z['model_names'].shape!=(2,) or z['model_names'].dtype.kind!='U' or z['model_names'].tolist()!=list(MODEL_NAMES):
            raise ValueError('Two ordered membership/OVA source models required')
        arrays={key:z[key].copy() for key in shapes}
    if any(a.dtype!=shapes[k][0] or a.shape!=shapes[k][1] or not np.isfinite(a).all() for k,a in arrays.items()):
        raise ValueError('Finite FP64 normalizer and FP32 106x32x1 model arrays required')
    report=previous.previous.runner._read(training_json)
    if not isinstance(report,dict) or not isinstance(report.get('training'),dict):
        raise ValueError('The complete source training trace is required')
    state=dict(schema=READOUT_SCHEMA,descriptor_schema=DESCRIPTOR_SCHEMA,descriptor_dim=106,class_count=6,
        model_names=list(MODEL_NAMES),mean=arrays['mean'],scale=arrays['scale'],models={name:{
            key:arrays['model__'+name+'__'+key] for key in ('weight1','bias1','weight2','bias2')} for name in MODEL_NAMES},
        training=report['training'])
    _training_complete(state)
    return state


def _descriptor(branches,logits,geometries):
    result=core.descriptor(branches['P1'],logits['P1'],geometries['P1'],
        branches['P0'],logits['P0'],geometries['P0'])
    if result.shape!=(len(logits['P1']),6,106) or result.dtype!=np.float32 or not np.isfinite(result).all():
        raise ValueError('Finite row-local six-class 106D paired descriptors required')
    return result


def _training_inputs(reader,role,geometries,device,batch_size,*,fit=False):
    branches={};logits={}
    for path in REJECTION_INPUT_PATHS:
        branches[path],logits[path]=_branches(reader,_values(role,path),device,batch_size)
        if fit:geometries[path]=geometry_core.fit_geometry(branches[path],role['metadata']['y'])
    return _descriptor(branches,logits,geometries),logits['P1'].argmax(axis=1).astype(np.int64)


def _four_scores(values,closed):
    if (values.shape!=(len(closed),2,6) or values.dtype!=np.float64 or not np.isfinite(values).all()
            or closed.dtype!=np.int64 or np.any((closed<0)|(closed>=6))):
        raise ValueError('Two finite six-class evidence vectors and original P1 labels required')
    rows=np.arange(len(closed))
    return np.column_stack((values[rows,0,closed],values[:,0].max(axis=1),
        values[rows,1,closed],values[:,1].max(axis=1)))


def _predict_role(reader,role,geometries,state,device,batch_size,label):
    scores=[];closed=[];class_scores=[]
    for start in range(0,len(role['ids']),batch_size):
        end=min(start+batch_size,len(role['ids']));branches={};logits={}
        for path in REJECTION_INPUT_PATHS:
            branches[path],logits[path]=_encode(reader,_values(role,path)[start:end],device)
        prediction=logits['P1'].argmax(axis=1).astype(np.int64)
        values=core.score(_descriptor(branches,logits,geometries),state)
        scores.append(_four_scores(values,prediction));closed.append(prediction);class_scores.append(values)
        print('STAR_OVA_PREDICTION_PROGRESS',label,end,len(role['ids']),flush=True)
    return np.concatenate(scores),np.concatenate(closed),np.concatenate(class_scores)


def _check_state(geometries,state,before):
    if not previous.previous.previous._same(dict(geometries=geometries,readouts=state),before):
        raise ValueError('Frozen OVA source aggregate state changed during prediction')


def _calibrate(role,closed,baseline,scores):
    conditional=[];global_threshold=[];methods={};controls={}
    for mi,name in enumerate(METHODS):
        state,report=risk.calibrate_known(role['ids'],role['metadata']['y'],closed,baseline,scores[:,mi],
            role['metadata']['rx_i'],role['metadata']['scenario'])
        tau=state['conditional_thresholds'];global_value=state['global_threshold']
        if (tau.shape!=(6,) or tau.dtype!=np.float64 or not np.isfinite(tau).all()
                or not np.isfinite(global_value)):
            raise ValueError('Four finite known-only six-class conditional threshold vectors required')
        conditional.append(tau.copy());global_threshold.append(global_value)
        methods[name]=dict(report['methods']['predicted_class'],conditional_thresholds=tau.tolist())
        controls[name]=dict(report['methods']['global_control'],global_threshold=float(global_value))
        print('STAR_OVA_METHOD_CALIBRATED',name,methods[name]['constraints_passed'],flush=True)
    return np.asarray(conditional,np.float64),np.asarray(global_threshold,np.float64),methods,controls


def _margins(scores,closed,thresholds):
    with np.errstate(over='ignore',invalid='ignore'):
        values=scores-thresholds[:,closed].T
    if values.shape!=(len(closed),4) or not np.isfinite(values).all():raise ValueError('Finite four-method conditional margins required')
    return values


def run(config,device='cuda:0'):
    validate_config(config);output=Path(config['output_root'])
    if output.exists():raise FileExistsError('A new immutable source OVA output root is required')
    started=time.perf_counter();backend,contract=_backend(config['source_config'],device);snapshots=_snapshot(backend)
    smoke=_smoke(backend,device);_check_models(snapshots)
    if smoke.get('status')!='PASS':raise ValueError('Actual frozen F0 synthetic G10 smoke must pass before real cached IQ')
    output.mkdir(parents=True,exist_ok=False);_write(output/'smoke.json',smoke)
    reader=backend['reader'];identity=backend['reader_identity'];handles={};seen=set();L=OE=V=novel=None
    try:
        L=_role(Path(config['cache_root'])/'L','L',identity);_disjoint(L,seen);handles['L']=L['manifest'];dura=L['manifest']['dura_identity']
        geometries={};known_desc,known_closed=_training_inputs(reader,L,geometries,device,config['batch_size'],fit=True)
        known_y=L['metadata']['y'].copy();known_domains=L['metadata']['domain'].copy();close_role(L);L=None
        folder=output/'geometry';folder.mkdir()
        for path in previous.previous.PATHS:_save_geometry(folder/(path+'.npz'),geometries[path])
        OE=_role(Path(config['cache_root'])/'OE','OE',identity,dura);_disjoint(OE,seen);handles['OE']=OE['manifest']
        oe_desc,_=_training_inputs(reader,OE,geometries,device,config['batch_size'])
        print('STAR_OVA_FIT_START',len(known_desc),len(oe_desc),2000,'cpu',flush=True)
        state=core.fit(known_desc,oe_desc,known_y,known_closed,known_domains,OE['metadata']['domain'],model_seed=2026092701)
        _training_complete(state);_save_readouts(output/'readouts.npz',state)
        _write(output/'readouts.json',dict(training=state['training'],schema=READOUT_SCHEMA,descriptor_schema=DESCRIPTOR_SCHEMA,
            run_id=RUN_ID,code_commit=config['code_commit'],source_only=True,target_reads=0,OE_TX_labels_read=0,
            source_V_fit_reads=0,source_novel_fit_reads=0,classification_path=CLASSIFICATION_PATH,
            model_names=list(MODEL_NAMES),method_names=list(METHODS),new_encoder_updates=0,new_identity_head_updates=0))
        del known_desc,oe_desc,known_y,known_closed,known_domains
        close_role(OE);OE=None;before=copy.deepcopy(dict(geometries=geometries,readouts=state));_check_models(snapshots)
        print('STAR_OVA_READOUTS_FROZEN',flush=True)
        V=_role(Path(config['cache_root'])/'V','V',identity,dura);_disjoint(V,seen);handles['V']=V['manifest']
        known_scores,known_closed,known_class_scores=_predict_role(reader,V,geometries,state,device,config['batch_size'],'V')
        baseline=np.array(V['baseline_closed'][:,1],copy=True)
        if not np.array_equal(known_closed,baseline):raise ValueError('The same-input original F0 P1 baseline differs')
        conditional,global_threshold,methods,controls=_calibrate(V,known_closed,baseline,known_scores)
        _check_state(geometries,state,before);_check_models(snapshots)
        _npz(output/'thresholds.npz',{'global':global_threshold,'conditional':conditional})
        calibration=output/'calibration';calibration.mkdir()
        _write(calibration/'complete.json',dict(schema=SCHEMA,status=CALIBRATION_STATUS,run_id=RUN_ID,code_commit=config['code_commit'],
            method_names=list(METHODS),primary_method=PRIMARY_METHOD,model_names=list(MODEL_NAMES),classification_path=CLASSIFICATION_PATH,
            known_only=True,unknown_inputs_used=False,source_only=True,target_reads=0,group_count=42,known_count=COUNTS['V'],
            conditional_thresholds=conditional.tolist(),global_thresholds=global_threshold.tolist(),methods=methods,global_controls=controls,
            constraints_passed=all(row['constraints_passed'] for row in methods.values()),score_threshold=0,
            source_V_neural_fit=False,source_V_normalizer_fit=False,source_V_threshold_fit=True,
            routing='threshold indexed only by original frozen P1 predicted class; complete ties',
            limits=dict(known_macro_acceptance_min=.9,known_macro_frr=.1,original_closed_minus_new_open=.02)))
        frozen_thresholds=conditional.copy();prediction=output/'predict';prediction.mkdir()
        _npz(prediction/'known_predictions.npz',dict(ids=V['ids'],y=V['metadata']['y'],rx_i=V['metadata']['rx_i'],day_i=V['metadata']['day_i'],
            scenario=V['metadata']['scenario'],scores=_margins(known_scores,known_closed,conditional),closed=known_closed,
            baseline_closed=baseline,class_scores=known_class_scores))
        close_role(V);V=None;print(CALIBRATION_STATUS,flush=True)
        # No novel cache or private truth is available before all four thresholds freeze.
        novel=_role(Path(config['cache_root'])/'novel','novel',identity,dura);_disjoint(novel,seen);handles['novel']=novel['manifest']
        novel_scores,novel_closed,novel_class_scores=_predict_role(reader,novel,geometries,state,device,config['batch_size'],'novel')
        _check_state(geometries,state,before);_check_models(snapshots)
        if not np.array_equal(conditional,frozen_thresholds):raise ValueError('Frozen conditional thresholds changed during novel prediction')
        _npz(prediction/'novel_predictions.npz',dict(ids=novel['ids'],rx_i=novel['metadata']['rx_i'],day_i=novel['metadata']['day_i'],
            scores=_margins(novel_scores,novel_closed,conditional),closed=novel_closed,class_scores=novel_class_scores))
        close_role(novel);novel=None
        result=dict(status=STATUS,schema=SCHEMA,run_id=RUN_ID,code_commit=config['code_commit'],config=config,
            method_names=list(METHODS),primary_method=PRIMARY_METHOD,model_names=list(MODEL_NAMES),classification_path=CLASSIFICATION_PATH,
            rejection_input_paths=list(REJECTION_INPUT_PATHS),score_threshold=0,known_count=COUNTS['V'],novel_count=COUNTS['novel'],counts=dict(COUNTS),
            parameters=232993,model_classes=6,parent_F0_identity=identity,dura_identity=dura,cache_handles=handles,
            source_only=True,target_reads=0,source_novel_truth_reads=0,original_U_s_reads=0,clean_IQ_reads=0,
            new_encoder_updates=0,new_identity_head_updates=0,encoder_updates=0,identity_head_updates=0,optimizer_updates_F=0,
            default_head_replaced=False,G_forward_rows=0,G_updates=0,channel_regenerations=0,
            model_parameters_buffers_unchanged=True,L_aggregates_unchanged_during_V=True,all_thresholds_frozen_before_novel=True,
            source_V_metric_fit=False,source_V_readout_fit=False,source_novel_fit_reads=0,
            readout_model_count=2,readout_steps_per_arm=1000,readout_optimizer_updates_total=2000,
            readout_parameter_counts={name:3457 for name in MODEL_NAMES},descriptor_dim=106,descriptor_schema=DESCRIPTOR_SCHEMA,
            normalizer_role_prior=dict(known=.6,oe=.4),loss_role_prior=dict(known=.5,oe=.5),source_domain_count=15,
            F0_forward_rows_by_role={key:2*n for key,n in COUNTS.items()},synthetic_F0_forward_rows=2,
            F0_forward_rows_actual=2*sum(COUNTS.values())+2,synthetic_G10_generated_rows=2,
            derived_path_exposures=2*sum(COUNTS.values()),independent_new_physical_records=0,
            known_predictions_ref='predict/known_predictions.npz',novel_predictions_ref='predict/novel_predictions.npz',
            thresholds_ref='thresholds.npz',calibration_ref='calibration/complete.json',no_query_smoke=True,
            seconds=time.perf_counter()-started,final_phase3_complete=False,
            scope='source G0 falsification on repeated novel development; original P1 identity unchanged')
        _write(output/'complete.json',result);print(STATUS,flush=True);return result
    finally:
        for role in (novel,V,OE,L):
            if role is not None:close_role(role)


def main(argv=None):
    print(json.dumps(dict(event='STAR_OVA_RUN_PROCESS',pid=os.getpid(),cwd=os.getcwd(),
        argv=list(sys.argv) if argv is None else list(argv)),ensure_ascii=False),flush=True)
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    command=sub.add_parser('run');command.add_argument('--config',required=True);command.add_argument('--device',default='cuda:0')
    args=parser.parse_args(argv);done=run(previous.previous.runner._read(args.config),args.device)
    print(json.dumps(dict(status=done['status'])),flush=True);return 0


if __name__=='__main__':raise SystemExit(main())
