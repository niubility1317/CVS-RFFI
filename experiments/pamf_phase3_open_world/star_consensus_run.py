"""Frozen original F0 source G10 consensus readouts on cached P0/P1 inputs.

Only L fits geometry; L/OE membership and source domains fit CPU readouts.
Known V chooses one global threshold per path/method. The novel role is first
opened after both paths' readouts, geometry and thresholds are immutable.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import re
import sys
import time

import numpy as np

from . import runner
from . import source_open_embedding_train as native
from . import star_consensus_readout as core
from . import star_relative_source as previous
from . import star_representation_cache as cache
from .branch_export import DIMENSIONS
from .source_consensus import DESCRIPTOR_SCHEMA
from .source_embedding_readout_run import _save

RUN_ID='20261029-phase3-star-consensus-s2026092701-r01'
SCHEMA='pamf_phase3_star_consensus_prediction_v1'
STATUS='STAR_CONSENSUS_PREDICTIONS_COMPLETE'
CALIBRATION_STATUS='STAR_CONSENSUS_THRESHOLDS_FROZEN'
PARENT_RUN=native.PARENT_RUN
PARENT_ROW=native.PARENT_ROW
PATHS=('P0','P1')
METHODS=('msp','maxlogit','margin','energy','cosine','linear_oe','mlp16_oe')
PRIMARY_PATH='P1'
PRIMARY_METHOD='mlp16_oe'
COUNTS=dict(L=6300,OE=6134,V=27000,novel=3150)
FIELDS={'run_id','code_commit','source_config','cache_root','output_root','batch_size'}
SOURCE_FIELDS=runner.COMMON|{'source_cache'}
KNOWN_FIELDS={'ids','y','rx_i','day_i','scenario','scores','closed','baseline_closed'}
NOVEL_FIELDS={'ids','rx_i','day_i','scores','closed'}
_write,_npz=previous._write,previous._npz
_snapshot,_check_models=previous._snapshot,previous._check_models
load_role,close_role=cache.load_role,cache.close_role


def validate_config(config):
    if (not isinstance(config,dict) or set(config)!=FIELDS or config.get('run_id')!=RUN_ID
            or not isinstance(config.get('code_commit'),str) or re.fullmatch(r'[0-9a-f]{40}',config['code_commit']) is None
            or not isinstance(config.get('source_config'),dict) or set(config['source_config'])!=SOURCE_FIELDS
            or type(config.get('batch_size')) is not int or config['batch_size']!=128
            or any(not isinstance(config[k],str) or not config[k].strip() for k in ('cache_root','output_root'))):
        raise ValueError('Only the fixed six-field source consensus configuration is allowed')
    source=config['source_config']
    runner.validate_config('calibrate',dict(source,retention=.99))
    if (source['run_id']!=RUN_ID or source['row_id']!='star-consensus-s2026092701'
            or source['code_commit']!=config['code_commit'] or source['output_root']!=config['output_root']
            or source['batch_size']!=128 or source['model_seed']!=2026092701 or type(source['model_seed']) is not int
            or source['registered_classes']!=native.CLASSES):
        raise ValueError('Only the original full-six F0 and this independent run/output are allowed')
    output=Path(config['output_root']).resolve()
    if any(output.is_relative_to(Path(p).resolve()) for p in (config['cache_root'],source['source_output'],source['source_cache'])):
        raise ValueError('Output cannot write inside the input cache or original source state')
    return config


def _backend(config,device):
    """Frozen native E50 provenance check; no IQ cache, G or increment loading."""
    import torch
    contract=native._contract(config)
    reader=runner._load_model(config,contract,device)
    original=runner._read(Path(config['source_output'])/'complete.json')['config']
    if original.get('run_id')!=PARENT_RUN or original.get('row_id')!=PARENT_ROW:
        raise ValueError('Only the fixed original Phase1 F0 E50 may generate star consensus evidence')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    identity=dict(model='original_F0_E50',parameters=232993,epoch=50,updates=11100,model_seed=2026092701,
        source_output=config['source_output'],source_contract=config['source_contract'],base_release=config['base_release'])
    return dict(reader=reader,dura=None,reader_identity=identity),contract


def _encode(reader,values,device):
    import torch
    from .source_open_export import _tensor,extract_branches
    with torch.no_grad():
        branches,logits=extract_branches(reader,_tensor(values,device))
    return {k:np.asarray(v.detach().cpu().tolist(),np.float32) for k,v in branches.items()},np.asarray(logits.detach().cpu().tolist(),np.float32)


def _smoke(backend,device):
    from .star_channel import observe
    path=Path(__file__).resolve().parents[2]/'configs/pamf_phase2_practical_mixture_20261028.json'
    mixture=runner._read(path);t=np.arange(256);rows=[]
    for i,w in enumerate((.07,.11)):
        raw=np.stack((np.cos(w*t),np.sin(w*t)))
        value,_=observe(raw,physical_id='star-consensus-synthetic-'+str(i),receiver='1-19',day='2021_03_08',
            episode=dict(id='MIX01',scene_seed=2026103001,channel_seed=2026104001),mixture=mixture)
        rows.append(value)
    branches,logits=_encode(backend['reader'],np.stack(rows),device)
    if set(branches)!=set(DIMENSIONS) or logits.shape!=(2,6) or not np.isfinite(logits).all():
        raise ValueError('Actual frozen F0 synthetic-star checkpoint smoke failed')
    return dict(status='PASS',actual_F0_checkpoint=True,synthetic_G10_received_rows=2,F0_forward_rows=2,
        G_forward_rows=0,real_cache_IQ_reads=0,source_V_reads=0,source_novel_reads=0,target_reads=0,
        optimizer_updates=0,scope='synthetic P0 F0 smoke; cached P1 was verified by the cache builder')


def _role(root,role,identity,expected_dura=None):
    data=load_role(root,role);manifest=data['manifest'];n=COUNTS[role]
    parent=manifest.get('parent_F0_identity');dura=manifest.get('dura_identity')
    required=dict(model='original_F0_E50',parameters=232993,epoch=50,updates=11100,model_seed=2026092701)
    if (manifest.get('role')!=role or manifest.get('count')!=n or manifest.get('source_only') is not True
            or manifest.get('received_input_path')!='P0' or manifest.get('restored_input_path')!='P1'
            or not isinstance(parent,dict) or any(parent.get(k)!=v for k,v in required.items())
            or parent.get('source_output')!=identity.get('source_output')
            or not isinstance(dura,dict) or dura.get('deployment_id')!='dura_v1' or dura.get('parameters')!=266951
            or dura.get('epoch')!=20 or dura.get('updates')!=4940 or (expected_dura is not None and dura!=expected_dura)):
        close_role(data)
        raise ValueError('Only matching original-F0/DURA source cache role handles may be consumed')
    return data


def _values(role,path):
    return role['received' if path=='P0' else 'restored']


def _branches(reader,values,device,batch_size):
    parts={key:[] for key in DIMENSIONS};logits=[]
    for start in range(0,len(values),batch_size):
        b,l=_encode(reader,values[start:start+batch_size],device)
        for key in DIMENSIONS:parts[key].append(b[key])
        logits.append(l)
        print('STAR_CONSENSUS_FIT_FEATURE_PROGRESS',start+len(l),len(values),flush=True)
    return {key:np.concatenate(rows) for key,rows in parts.items()},np.concatenate(logits)


def _score(reader,values,state,device,batch_size,path,role):
    scores=[];closed=[]
    for start in range(0,len(values),batch_size):
        branches,logits=_encode(reader,values[start:start+batch_size],device)
        block=core.all_scores(branches,logits,state['geometry'],state['readouts'])
        if block.shape!=(len(logits),7) or block.dtype!=np.float64 or not np.isfinite(block).all():
            raise ValueError('Seven finite row-local original-head source scores required')
        scores.append(block);closed.append(logits.argmax(axis=1).astype(np.int64))
        print('STAR_CONSENSUS_PREDICT_PROGRESS',role,path,start+len(logits),len(values),flush=True)
    return np.concatenate(scores),np.concatenate(closed)


def _check_states(states):
    for state in states.values():
        if not previous._same(dict(geometry=state['geometry'],readouts=state['readouts']),state['before']):
            raise ValueError('L/OE-frozen source aggregate state changed during prediction')


def _disjoint(role,seen):
    ids=set(role['ids'].tolist())
    if seen&ids:raise ValueError('Declared source L/OE/V/novel physical IDs must be disjoint')
    seen.update(ids)


def run(config,device='cuda:0'):
    validate_config(config);output=Path(config['output_root'])
    if output.exists():raise FileExistsError('A new immutable source consensus output is required')
    started=time.perf_counter();backend,contract=_backend(config['source_config'],device);snapshots=_snapshot(backend)
    smoke=_smoke(backend,device);_check_models(snapshots)
    if smoke.get('status')!='PASS':raise ValueError('Actual frozen F0 no-query smoke must pass before real cache inputs')
    output.mkdir(parents=True,exist_ok=False);_write(output/'smoke.json',smoke)
    reader=backend['reader'];identity=backend['reader_identity'];handles={};seen=set();L=OE=V=novel=None
    try:
        L=_role(Path(config['cache_root'])/'L','L',identity);_disjoint(L,seen);handles['L']=L['manifest']
        dura=L['manifest']['dura_identity']
        OE=_role(Path(config['cache_root'])/'OE','OE',identity,dura);_disjoint(OE,seen);handles['OE']=OE['manifest']
        states={}
        for path in PATHS:
            branches,logits=_branches(reader,_values(L,path),device,config['batch_size'])
            geometry=core.fit_geometry(branches,L['metadata']['y'])
            known_desc=core.descriptor(branches,logits,geometry)
            del branches,logits
            branches,logits=_branches(reader,_values(OE,path),device,config['batch_size'])
            oe_desc=core.descriptor(branches,logits,geometry)
            readouts=core.fit_readouts(known_desc,oe_desc,L['metadata']['domain'],OE['metadata']['domain'],model_seed=2026092701)
            training=readouts['training']
            if (training.get('steps')!=200 or training.get('known_count')!=COUNTS['L'] or training.get('oe_count')!=COUNTS['OE']
                    or training.get('regime')!='source_role_domain_balanced_v1'
                    or training.get('parameter_counts')!={'linear_oe':101,'mlp16_oe':1633}
                    or set(training.get('losses',{}))!={'linear_oe','mlp16_oe'}
                    or any(len(v)!=200 or not np.isfinite(v).all() for v in training['losses'].values())):
                raise ValueError('Both source-domain-balanced readouts must complete exactly 200 CPU updates')
            row=output/'by_path'/path;row.mkdir(parents=True)
            _save(row,geometry,readouts)
            _write(row/'readouts.json',dict(training=training,path=path,descriptor_schema=DESCRIPTOR_SCHEMA,
                source_only=True,target_reads=0,encoder_updates=0,identity_head_updates=0,source_V_fit_reads=0,novel_fit_reads=0))
            states[path]=dict(geometry=geometry,readouts=readouts,before=copy.deepcopy(dict(geometry=geometry,readouts=readouts)))
            del branches,logits,known_desc,oe_desc
            _check_models(snapshots);print('STAR_CONSENSUS_PATH_FIT_FROZEN',path,flush=True)
        close_role(OE);OE=None;close_role(L);L=None
        # Both paths' geometry, normalization and readout weights are now L/OE frozen.
        V=_role(Path(config['cache_root'])/'V','V',identity,dura);_disjoint(V,seen);handles['V']=V['manifest']
        known_scores=np.empty((COUNTS['V'],2,7),np.float64);known_closed=np.empty((COUNTS['V'],2),np.int64)
        baseline=np.array(V['baseline_closed'],copy=True)
        for pi,path in enumerate(PATHS):
            values,closed=_score(reader,_values(V,path),states[path],device,config['batch_size'],path,'V')
            if not np.array_equal(closed,baseline[:,pi]):
                raise ValueError('The same-path original F0 closed baseline and cached predictions differ')
            thresholds,report=core.calibrate_known(V['ids'],V['metadata']['y'],closed,baseline[:,pi],values,
                V['metadata']['rx_i'],V['metadata']['scenario'])
            if thresholds.shape!=(7,) or thresholds.dtype!=np.float64 or not np.isfinite(thresholds).all():
                raise ValueError('Seven known-only frozen global thresholds required')
            _check_states(states);_check_models(snapshots)
            row=output/'by_path'/path;(row/'calibration').mkdir()
            with (row/'thresholds.npy').open('xb') as stream:np.save(stream,thresholds,allow_pickle=False)
            _write(row/'calibration/complete.json',dict(report,status=CALIBRATION_STATUS,schema=SCHEMA,run_id=RUN_ID,
                code_commit=config['code_commit'],path=path,method_names=list(METHODS),group_count=42,known_only=True,
                unknown_inputs_used=False,source_only=True,target_reads=0))
            states[path]['thresholds']=thresholds.copy();known_scores[:,pi]=values;known_closed[:,pi]=closed
            print(CALIBRATION_STATUS,path,flush=True)
        predict=output/'predict';predict.mkdir()
        _npz(predict/'known_predictions.npz',dict(ids=V['ids'],y=V['metadata']['y'],rx_i=V['metadata']['rx_i'],
            day_i=V['metadata']['day_i'],scenario=V['metadata']['scenario'],scores=known_scores,closed=known_closed,baseline_closed=baseline))
        close_role(V);V=None
        # This is the first novel-role open. No private truth is ever available here.
        novel=_role(Path(config['cache_root'])/'novel','novel',identity,dura);_disjoint(novel,seen);handles['novel']=novel['manifest']
        novel_scores=np.empty((COUNTS['novel'],2,7),np.float64);novel_closed=np.empty((COUNTS['novel'],2),np.int64)
        for pi,path in enumerate(PATHS):
            novel_scores[:,pi],novel_closed[:,pi]=_score(reader,_values(novel,path),states[path],device,config['batch_size'],path,'novel')
            _check_states(states);_check_models(snapshots)
        _npz(predict/'novel_predictions.npz',dict(ids=novel['ids'],rx_i=novel['metadata']['rx_i'],day_i=novel['metadata']['day_i'],
            scores=novel_scores,closed=novel_closed))
        close_role(novel);novel=None
        result=dict(status=STATUS,schema=SCHEMA,run_id=RUN_ID,code_commit=config['code_commit'],config=config,
            path_names=list(PATHS),method_names=list(METHODS),primary_path=PRIMARY_PATH,primary_method=PRIMARY_METHOD,
            known_count=COUNTS['V'],novel_count=COUNTS['novel'],counts=dict(COUNTS),parameters=232993,model_classes=6,
            parent_F0_identity=identity,dura_identity=dura,cache_handles=handles,source_only=True,
            target_reads=0,source_novel_truth_reads=0,original_U_s_reads=0,clean_IQ_reads=0,
            encoder_updates=0,identity_head_updates=0,optimizer_updates_F=0,new_encoder_updates=0,new_identity_head_updates=0,
            default_head_replaced=False,readout_steps_per_model=200,readout_parameter_counts={'linear_oe':101,'mlp16_oe':1633},
            readout_training_regime='source_role_domain_balanced_v1',descriptor_schema=DESCRIPTOR_SCHEMA,
            model_parameters_buffers_unchanged=True,L_aggregates_unchanged_during_V=True,all_thresholds_frozen_before_novel=True,
            source_V_metric_fit=False,source_V_readout_fit=False,novel_fit_reads=0,
            G_forward_rows=0,channel_regenerations=0,F0_forward_rows_by_role={k:2*v for k,v in COUNTS.items()},
            synthetic_F0_forward_rows=2,synthetic_G10_generated_rows=2,
            readout_optimizer_updates_by_path={p:400 for p in PATHS},readout_optimizer_updates_total=800,
            derived_path_exposures=2*sum(COUNTS.values()),independent_new_physical_records=0,
            known_predictions_ref='predict/known_predictions.npz',novel_predictions_ref='predict/novel_predictions.npz',
            baseline_scope='same-input same-path original F0 full default head',seconds=time.perf_counter()-started,
            final_phase3_complete=False,scope='source-only repeated heldout novel evaluation; no formal target access')
        _write(output/'complete.json',result);print(STATUS,flush=True);return result
    finally:
        for data in (novel,V,OE,L):
            if data is not None:close_role(data)


def main(argv=None):
    print(json.dumps(dict(event='STAR_CONSENSUS_RUN_PROCESS',pid=os.getpid(),cwd=os.getcwd(),
        argv=list(sys.argv) if argv is None else list(argv)),ensure_ascii=False),flush=True)
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    command=sub.add_parser('run');command.add_argument('--config',required=True);command.add_argument('--device',default='cuda:0')
    args=parser.parse_args(argv);done=run(runner._read(args.config),args.device)
    print(json.dumps(dict(status=done['status'])),flush=True);return 0


if __name__=='__main__':raise SystemExit(main())
