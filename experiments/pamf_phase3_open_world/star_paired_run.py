"""Paired source evidence, frozen F0, and unchanged P1 identity predictions.

P1's 100D consensus descriptor precedes P0's auxiliary 100D descriptor.
Only L fits geometry; only L/OE membership fits the paired readers. Known V
calibrates five global thresholds after all learned state is frozen. Novel
inputs are opened only after threshold completion; private truth is absent.
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

from . import star_consensus_run as previous
from . import star_paired_readout as core

RUN_ID='20261029-phase3-star-paired-evidence-s2026092701-r01'
SCHEMA='pamf_phase3_star_paired_prediction_v1'
STATUS='STAR_PAIRED_PREDICTIONS_COMPLETE'
CALIBRATION_STATUS='STAR_PAIRED_THRESHOLDS_FROZEN'
READOUT_SCHEMA='pamf_phase3_star_paired_readouts_v1'
METHODS=('energy','p1_mlp16_step200','paired_mlp16_step200','p1_mlp16_step1000','paired_mlp16_step1000')
LEARNED_METHODS=METHODS[1:]
PRIMARY_METHOD=METHODS[-1]
CLASSIFICATION_PATH='P1'
REJECTION_INPUT_PATHS=('P1','P0')
COUNTS=dict(previous.COUNTS)
FIELDS=previous.FIELDS
KNOWN_FIELDS=previous.KNOWN_FIELDS
NOVEL_FIELDS=previous.NOVEL_FIELDS
geometry_core=previous.core
_backend,_smoke=previous._backend,previous._smoke
_role,_values,_branches=previous._role,previous._values,previous._branches
_snapshot,_check_models,_disjoint=previous._snapshot,previous._check_models,previous._disjoint
_encode,_write,_npz,close_role=previous._encode,previous._write,previous._npz,previous.close_role
_save_geometry=previous.previous._save_geometry


def validate_config(config):
    if (not isinstance(config,dict) or set(config)!=FIELDS or config.get('run_id')!=RUN_ID
            or not isinstance(config.get('source_config'),dict)
            or config['source_config'].get('run_id')!=RUN_ID
            or config['source_config'].get('row_id')!='star-paired-s2026092701'):
        raise ValueError('Only the fixed paired source run and six-field input schema is allowed')
    # Input-schema reuse only; no previous run, checkpoint increment or output is opened.
    mapped=dict(config,run_id=previous.RUN_ID,source_config=dict(config['source_config'],
        run_id=previous.RUN_ID,row_id='star-consensus-s2026092701'))
    previous.validate_config(mapped)
    return config


def _save_readouts(path,state):
    core._state(state)
    arrays=dict(schema=np.asarray(state['schema']),pair_dim=np.asarray(200,np.int64),method_names=np.asarray(state['method_names']))
    for arm,normalizer in state['normalizers'].items():
        for key,value in normalizer.items():arrays['normalizer__'+arm+'__'+key]=value
    for method,model in state['models'].items():
        for key,value in model.items():arrays['model__'+method+'__'+key]=value
    _npz(path,arrays)


def load_readouts(path,training_json):
    """Restore aggregate arrays and the separate complete source training trace."""
    shapes={**{'normalizer__'+arm+'__'+key:(np.float64,(200,)) for arm in ('p1_only','paired') for key in ('mean','scale')},
        **{'model__'+method+'__'+key:(np.float32,shape) for method in LEARNED_METHODS
           for key,shape in (('weight1',(16,200)),('bias1',(16,)),('weight2',(1,16)),('bias2',(1,)))}}
    with np.load(path,allow_pickle=False) as z:
        if (set(z.files)!=set(shapes)|{'schema','pair_dim','method_names'} or z['schema'].shape!=()
                or z['schema'].dtype.kind!='U' or z['schema'].item()!=READOUT_SCHEMA or z['pair_dim'].shape!=()
                or z['pair_dim'].dtype!=np.int64 or z['pair_dim'].item()!=200
                or z['method_names'].dtype.kind!='U' or z['method_names'].shape!=(4,)
                or z['method_names'].tolist()!=list(LEARNED_METHODS)):
            raise ValueError('The complete fixed 200D paired readout aggregate schema is required')
        arrays={key:z[key].copy() for key in shapes}
    if any(value.dtype!=shapes[key][0] or value.shape!=shapes[key][1] or not np.isfinite(value).all() for key,value in arrays.items()):
        raise ValueError('Finite paired normalizer/model array shapes and dtypes required')
    report=previous.runner._read(training_json)
    if not isinstance(report,dict) or not isinstance(report.get('training'),dict):raise ValueError('The source-only paired training trace is required')
    state=dict(schema=READOUT_SCHEMA,pair_dim=200,method_names=list(LEARNED_METHODS),
        normalizers={arm:{key:arrays['normalizer__'+arm+'__'+key] for key in ('mean','scale')} for arm in ('p1_only','paired')},
        models={method:{key:arrays['model__'+method+'__'+key] for key in ('weight1','bias1','weight2','bias2')} for method in LEARNED_METHODS},
        training=report['training'])
    core._state(state)
    return state


def _energy(logits):
    values=np.asarray(logits,np.float64)
    if values.ndim!=2 or values.shape[1]!=6 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Finite full P1 six-class logits required for energy')
    maximum=values.max(axis=1)
    return maximum+np.log(np.exp(values-maximum[:,None]).sum(axis=1))


def _training_pair(reader,role,geometries,device,batch_size,*,fit=False):
    descriptors={}
    for path in previous.PATHS:
        branches,logits=_branches(reader,_values(role,path),device,batch_size)
        if fit:geometries[path]=geometry_core.fit_geometry(branches,role['metadata']['y'])
        descriptors[path]=geometry_core.descriptor(branches,logits,geometries[path])
    return core.pair_descriptors(descriptors['P1'],descriptors['P0'])


def _predict_role(reader,role,geometries,state,device,batch_size,label):
    scores=[];closed=[]
    for start in range(0,len(role['ids']),batch_size):
        end=min(start+batch_size,len(role['ids']));desc={};p1_logits=None
        for path in REJECTION_INPUT_PATHS:
            branches,logits=_encode(reader,_values(role,path)[start:end],device)
            desc[path]=geometry_core.descriptor(branches,logits,geometries[path])
            if path=='P1':p1_logits=logits
        pair=core.pair_descriptors(desc['P1'],desc['P0'])
        result=core.score_compact(pair,_energy(p1_logits),state)
        if result.shape!=(end-start,5) or result.dtype!=np.float64 or not np.isfinite(result).all():
            raise ValueError('Five finite source-frozen paired knownness scores required')
        scores.append(result);closed.append(p1_logits.argmax(axis=1).astype(np.int64))
        print('STAR_PAIRED_PREDICTION_PROGRESS',label,end,len(role['ids']),flush=True)
    return np.concatenate(scores),np.concatenate(closed)


def _check_state(geometries,state,before):
    if not previous.previous._same(dict(geometries=geometries,readouts=state),before):
        raise ValueError('Frozen paired source geometry/readout aggregate changed during prediction')


def _training_complete(state):
    training=state.get('training',{})
    if (state.get('schema')!=READOUT_SCHEMA or state.get('pair_dim')!=200 or state.get('method_names')!=list(LEARNED_METHODS)
            or training.get('steps_per_arm')!=1000 or training.get('total_optimizer_updates')!=2000
            or training.get('checkpoint_steps')!=[200,1000]
            or training.get('checkpoint_updates')!={name:200 if name.endswith('200') else 1000 for name in LEARNED_METHODS}
            or training.get('known_count')!=COUNTS['L'] or training.get('oe_count')!=COUNTS['OE']
            or any(training.get(k) is not True for k in ('initial_parameters_equal','seed_reset_per_arm','fresh_optimizer_per_arm'))
            or training.get('parameter_counts')!={method:3233 for method in LEARNED_METHODS}
            or set(training.get('losses',{}))!={'p1_only','paired'}
            or any(len(trace)!=1000 or not np.isfinite(trace).all() for trace in training['losses'].values())):
        raise ValueError('Two matched fixed 1000-step source readouts and four same-trajectory snapshots must complete')


def run(config,device='cuda:0'):
    validate_config(config);output=Path(config['output_root'])
    if output.exists():raise FileExistsError('A new immutable paired source output root is required')
    started=time.perf_counter();backend,contract=_backend(config['source_config'],device);snapshots=_snapshot(backend)
    smoke=_smoke(backend,device);_check_models(snapshots)
    if smoke.get('status')!='PASS':raise ValueError('Actual frozen F0 synthetic-star smoke must pass before real cached IQ')
    output.mkdir(parents=True,exist_ok=False);_write(output/'smoke.json',smoke)
    reader=backend['reader'];identity=backend['reader_identity'];handles={};seen=set();L=OE=V=novel=None
    try:
        L=_role(Path(config['cache_root'])/'L','L',identity);_disjoint(L,seen);handles['L']=L['manifest'];dura=L['manifest']['dura_identity']
        geometries={};known_pair=_training_pair(reader,L,geometries,device,config['batch_size'],fit=True)
        known_domains=L['metadata']['domain'].copy();close_role(L);L=None
        folder=output/'geometry';folder.mkdir()
        for path in previous.PATHS:_save_geometry(folder/(path+'.npz'),geometries[path])
        OE=_role(Path(config['cache_root'])/'OE','OE',identity,dura);_disjoint(OE,seen);handles['OE']=OE['manifest']
        oe_pair=_training_pair(reader,OE,geometries,device,config['batch_size'])
        print('STAR_PAIRED_FIT_START',len(known_pair),len(oe_pair),2000,'cpu',flush=True)
        state=core.fit_readouts(known_pair,oe_pair,known_domains,OE['metadata']['domain'],model_seed=2026092701)
        _training_complete(state);_save_readouts(output/'readouts.npz',state)
        _write(output/'readouts.json',dict(training=state['training'],run_id=RUN_ID,code_commit=config['code_commit'],
            source_only=True,target_reads=0,source_V_fit_reads=0,source_novel_fit_reads=0,OE_TX_labels_read=0,
            classification_path=CLASSIFICATION_PATH,rejection_input_paths=list(REJECTION_INPUT_PATHS),control_p0_zeroed=True,
            descriptor_schema='head_aligned_pair_P1_then_P0_v1',encoder_updates=0,identity_head_updates=0))
        del known_pair,oe_pair,known_domains
        close_role(OE);OE=None;before=copy.deepcopy(dict(geometries=geometries,readouts=state));_check_models(snapshots)
        print('STAR_PAIRED_READOUTS_FROZEN',flush=True)
        # Neither V nor novel has been opened before the full 2000-update source fit.
        V=_role(Path(config['cache_root'])/'V','V',identity,dura);_disjoint(V,seen);handles['V']=V['manifest']
        known_scores,known_closed=_predict_role(reader,V,geometries,state,device,config['batch_size'],'V')
        baseline=np.array(V['baseline_closed'][:,1],copy=True)
        if not np.array_equal(known_closed,baseline):raise ValueError('Same-input P1 original F0 closed baseline differs')
        thresholds,report=core.calibrate_known(V['ids'],V['metadata']['y'],known_closed,baseline,known_scores,
            V['metadata']['rx_i'],V['metadata']['scenario'])
        if thresholds.shape!=(5,) or thresholds.dtype!=np.float64 or not np.isfinite(thresholds).all():
            raise ValueError('Five finite known-only maximal global thresholds required')
        _check_state(geometries,state,before);_check_models(snapshots)
        with (output/'thresholds.npy').open('xb') as stream:np.save(stream,thresholds,allow_pickle=False)
        calibration=output/'calibration';calibration.mkdir()
        _write(calibration/'complete.json',dict(report,status=CALIBRATION_STATUS,schema=SCHEMA,run_id=RUN_ID,
            code_commit=config['code_commit'],method_names=list(METHODS),group_count=42,known_only=True,
            unknown_inputs_used=False,source_only=True,target_reads=0,classification_path=CLASSIFICATION_PATH))
        prediction=output/'predict';prediction.mkdir()
        _npz(prediction/'known_predictions.npz',dict(ids=V['ids'],y=V['metadata']['y'],rx_i=V['metadata']['rx_i'],day_i=V['metadata']['day_i'],
            scenario=V['metadata']['scenario'],scores=known_scores,closed=known_closed,baseline_closed=baseline))
        close_role(V);V=None;print(CALIBRATION_STATUS,flush=True)
        novel=_role(Path(config['cache_root'])/'novel','novel',identity,dura);_disjoint(novel,seen);handles['novel']=novel['manifest']
        novel_scores,novel_closed=_predict_role(reader,novel,geometries,state,device,config['batch_size'],'novel')
        _check_state(geometries,state,before);_check_models(snapshots)
        _npz(prediction/'novel_predictions.npz',dict(ids=novel['ids'],rx_i=novel['metadata']['rx_i'],day_i=novel['metadata']['day_i'],
            scores=novel_scores,closed=novel_closed))
        close_role(novel);novel=None
        result=dict(status=STATUS,schema=SCHEMA,run_id=RUN_ID,code_commit=config['code_commit'],config=config,
            method_names=list(METHODS),primary_method=PRIMARY_METHOD,classification_path=CLASSIFICATION_PATH,
            rejection_input_paths=list(REJECTION_INPUT_PATHS),control_p0_zeroed=True,known_count=COUNTS['V'],novel_count=COUNTS['novel'],counts=dict(COUNTS),
            parameters=232993,model_classes=6,parent_F0_identity=identity,dura_identity=dura,cache_handles=handles,
            source_only=True,target_reads=0,source_novel_truth_reads=0,original_U_s_reads=0,clean_IQ_reads=0,
            new_encoder_updates=0,new_identity_head_updates=0,encoder_updates=0,identity_head_updates=0,optimizer_updates_F=0,
            default_head_replaced=False,G_forward_rows=0,G_updates=0,channel_regenerations=0,
            model_parameters_buffers_unchanged=True,L_aggregates_unchanged_during_V=True,all_thresholds_frozen_before_novel=True,
            source_V_metric_fit=False,source_V_readout_fit=False,source_novel_fit_reads=0,
            readout_steps_per_arm=1000,readout_optimizer_updates_total=2000,checkpoint_steps=[200,1000],
            snapshot_count=4,readout_parameter_counts={method:3233 for method in LEARNED_METHODS},
            normalizer_role_prior=dict(known=.6,oe=.4),loss_role_prior=dict(known=.5,oe=.5),source_domain_count=15,
            F0_forward_rows_by_role={key:2*n for key,n in COUNTS.items()},synthetic_F0_forward_rows=2,synthetic_G10_generated_rows=2,
            derived_path_exposures=2*sum(COUNTS.values()),independent_new_physical_records=0,
            known_predictions_ref='predict/known_predictions.npz',novel_predictions_ref='predict/novel_predictions.npz',
            thresholds_ref='thresholds.npy',calibration_ref='calibration/complete.json',seconds=time.perf_counter()-started,
            final_phase3_complete=False,scope='source-only repeated novel development; paired P0 evidence never changes the P1 identity head')
        _write(output/'complete.json',result);print(STATUS,flush=True);return result
    finally:
        for role in (novel,V,OE,L):
            if role is not None:close_role(role)


def main(argv=None):
    print(json.dumps(dict(event='STAR_PAIRED_RUN_PROCESS',pid=os.getpid(),cwd=os.getcwd(),
        argv=list(sys.argv) if argv is None else list(argv)),ensure_ascii=False),flush=True)
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    command=sub.add_parser('run');command.add_argument('--config',required=True);command.add_argument('--device',default='cuda:0')
    args=parser.parse_args(argv);done=run(previous.runner._read(args.config),args.device)
    print(json.dumps(dict(status=done['status'])),flush=True);return 0


if __name__=='__main__':raise SystemExit(main())
