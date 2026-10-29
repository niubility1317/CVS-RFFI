"""Calibration-only predicted-class risks on frozen paired P1 evidence.

Only the parent readout aggregates enter the first synthetic feature smoke.
Known V labels calibrate thresholds; novel score rows are first opened after
the complete thresholds artifact. No IQ, F0, G, geometry fit or private truth
is loaded, and the original P1 head and baseline predictions remain unchanged.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import re
import sys

import numpy as np

from . import star_class_risk as core

RUN_ID='20261029-phase3-star-class-risk-s2026092701-r01'
SCHEMA='pamf_phase3_star_class_risk_prediction_v1'
STATUS='STAR_CLASS_RISK_PREDICTIONS_COMPLETE'
CALIBRATION_STATUS='STAR_CLASS_RISK_THRESHOLDS_FROZEN'
PARENT_RUN='20261029-phase3-star-paired-evidence-s2026092701-r01'
PARENT_COMMIT='bcc873cd9f9273133d74b5bc591c947b37ce16e1'
PARENT_METHODS=('energy','p1_mlp16_step200','paired_mlp16_step200','p1_mlp16_step1000','paired_mlp16_step1000')
METHODS=('global_control','predicted_class')
COUNTS=dict(L=6300,OE=6134,V=27000,novel=3150)
FIELDS={'run_id','code_commit','parent_prediction_root','output_root'}
KNOWN_FIELDS={'ids','y','rx_i','day_i','scenario','scores','closed','baseline_closed'}
NOVEL_FIELDS={'ids','rx_i','day_i','scores','closed'}
RECEIVERS=(1,3,4,6,8)
SCENES=('practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban')


def _read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _write(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')


def _npz(path,arrays):
    with Path(path).open('xb') as stream:np.savez(stream,**arrays)


def _same(a,b):
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and set(a)==set(b) and all(_same(a[k],b[k]) for k in a)
    return a==b


def validate_config(config):
    if (not isinstance(config,dict) or set(config)!=FIELDS or config.get('run_id')!=RUN_ID
            or any(not isinstance(config[k],str) or not config[k].strip() for k in FIELDS)
            or re.fullmatch(r'[0-9a-f]{40}',config['code_commit']) is None):
        raise ValueError('Only the four-field CPU calibration-only configuration is accepted')
    if Path(config['output_root']).resolve().is_relative_to(Path(config['parent_prediction_root']).resolve()):
        raise ValueError('A new immutable root outside the frozen parent output is required')
    return config


def _parent(root):
    marker=_read(Path(root)/'complete.json')
    if (marker.get('status')!='STAR_PAIRED_PREDICTIONS_COMPLETE' or marker.get('schema')!='pamf_phase3_star_paired_prediction_v1'
            or marker.get('run_id')!=PARENT_RUN or marker.get('code_commit')!=PARENT_COMMIT
            or marker.get('method_names')!=list(PARENT_METHODS) or marker.get('primary_method')!=PARENT_METHODS[-1]
            or marker.get('classification_path')!='P1' or marker.get('rejection_input_paths')!=['P1','P0']
            or marker.get('control_p0_zeroed') is not True or marker.get('source_only') is not True
            or marker.get('counts')!=COUNTS or marker.get('known_count')!=COUNTS['V'] or marker.get('novel_count')!=COUNTS['novel']
            or marker.get('parameters')!=232993 or marker.get('model_classes')!=6
            or marker.get('readout_steps_per_arm')!=1000 or marker.get('readout_optimizer_updates_total')!=2000
            or marker.get('checkpoint_steps')!=[200,1000] or marker.get('snapshot_count')!=4
            or marker.get('readout_parameter_counts')!={name:3233 for name in PARENT_METHODS[1:]}
            or marker.get('known_predictions_ref')!='predict/known_predictions.npz'
            or marker.get('novel_predictions_ref')!='predict/novel_predictions.npz'
            or any(marker.get(k)!=0 for k in ('target_reads','source_novel_truth_reads','new_encoder_updates','new_identity_head_updates',
                                               'encoder_updates','identity_head_updates','optimizer_updates_F','G_forward_rows','G_updates'))
            or any(marker.get(k) is not True for k in ('model_parameters_buffers_unchanged','L_aggregates_unchanged_during_V','all_thresholds_frozen_before_novel'))):
        raise ValueError('The complete fixed source-only paired parent and all five methods are required')
    identity=marker.get('parent_F0_identity',{});dura=marker.get('dura_identity',{});handles=marker.get('cache_handles',{})
    if (not isinstance(identity,dict) or any(identity.get(k)!=v for k,v in dict(model='original_F0_E50',parameters=232993,epoch=50,updates=11100,model_seed=2026092701).items())
            or not isinstance(dura,dict) or dura.get('deployment_id')!='dura_v1' or dura.get('parameters')!=266951
            or not isinstance(handles,dict) or set(handles)!=set(COUNTS)
            or any(not isinstance(handles[k],dict) or handles[k].get('role')!=k or handles[k].get('count')!=n or handles[k].get('source_only') is not True for k,n in COUNTS.items())):
        raise ValueError('Original frozen F0/DURA and completed source role handles are required')
    return marker


def _smoke(root,marker):
    # The codec reads only aggregates and training JSON, never parent predictions.
    from .star_paired_run import load_readouts
    from .star_paired_readout import score_readouts
    state=load_readouts(root/'readouts.npz',root/'readouts.json');training=state['training']
    counts={name:sum(value.size for value in model.values()) for name,model in state['models'].items()}
    if (state['method_names']!=list(PARENT_METHODS[1:]) or counts!={name:3233 for name in PARENT_METHODS[1:]}
            or training.get('known_count')!=COUNTS['L'] or training.get('oe_count')!=COUNTS['OE']
            or training.get('steps_per_arm')!=1000 or training.get('total_optimizer_updates')!=2000
            or training.get('checkpoint_steps')!=[200,1000] or training.get('domain_count')!=15
            or training.get('model_seed')!=2026092701):
        raise ValueError('The real frozen four-snapshot source readout checkpoint differs')
    before=copy.deepcopy(state)
    pair=np.stack((np.zeros(200,np.float32),np.full(200,.001,np.float32)))
    scores=score_readouts(pair,state)
    if scores.shape!=(2,4) or scores.dtype!=np.float64 or not np.isfinite(scores).all() or not _same(state,before):
        raise ValueError('Real frozen readout synthetic-feature smoke failed')
    return state,dict(status='PASS',feature_smoke=True,IQ_smoke=False,actual_parent_readout_checkpoint=True,
        synthetic_pair_rows=2,descriptor_width=200,readout_models=4,parameters_per_model=3233,
        historical_readout_optimizer_updates=2000,parent_state_unchanged=True,source_prediction_rows_read=0,
        F0_forward_rows=0,G_forward_rows=0,IQ_reads=0,query_reads=0,target_reads=0,new_neural_updates=0,
        scope='NumPy readout checkpoint feature smoke; not a G10 IQ/F0/G smoke')


def _parent_threshold(root,marker):
    report=_read(root/'calibration/complete.json');values=np.load(root/'thresholds.npy',allow_pickle=False)
    if (report.get('status')!='STAR_PAIRED_THRESHOLDS_FROZEN' or report.get('schema')!='pamf_phase3_star_paired_prediction_v1'
            or report.get('run_id')!=PARENT_RUN or report.get('code_commit')!=PARENT_COMMIT
            or report.get('method_names')!=list(PARENT_METHODS) or report.get('group_count')!=42
            or report.get('known_only') is not True or report.get('unknown_inputs_used') is not False
            or set(report.get('methods',{}))!=set(PARENT_METHODS) or values.shape!=(5,) or values.dtype!=np.float64
            or not np.isfinite(values).all()
            or any(report['methods'][name].get('threshold')!=float(values[j]) for j,name in enumerate(PARENT_METHODS))):
        raise ValueError('All five parent known-only global thresholds must be frozen and aligned')
    return float(values[4])


def _load_prediction(root,marker,known):
    n=COUNTS['V' if known else 'novel'];fields=KNOWN_FIELDS if known else NOVEL_FIELDS
    with np.load(root/marker['known_predictions_ref' if known else 'novel_predictions_ref'],allow_pickle=False) as z:
        if set(z.files)!=fields:raise ValueError('Exact parent prediction fields required; no private novel TX/scene inputs')
        data={k:z[k].copy() for k in z.files}
    ids=data['ids']
    if (ids.shape!=(n,) or ids.dtype.kind!='U' or len(set(ids.tolist()))!=n or np.any(np.char.strip(ids)=='')
            or data['scores'].shape!=(n,5) or data['scores'].dtype!=np.float64 or not np.isfinite(data['scores']).all()
            or data['closed'].shape!=(n,) or data['closed'].dtype!=np.int64 or np.any((data['closed']<0)|(data['closed']>=6))):
        raise ValueError('Complete finite parent five-score rows and P1 identities required')
    for key,allowed in (('rx_i',RECEIVERS),('day_i',(1,2,3))):
        if data[key].shape!=(n,) or data[key].dtype!=np.int64 or not np.isin(data[key],allowed).all():
            raise ValueError('Aligned original source RX/day metadata required')
    if known and (data['y'].shape!=(n,) or data['y'].dtype!=np.int64 or np.any((data['y']<0)|(data['y']>=6))
            or data['scenario'].shape!=(n,) or data['scenario'].dtype.kind!='U' or not np.isin(data['scenario'],SCENES).all()
            or data['baseline_closed'].shape!=(n,) or data['baseline_closed'].dtype!=np.int64 or not np.array_equal(data['baseline_closed'],data['closed'])):
        raise ValueError('Visible source V labels and unchanged original P1 baseline required')
    return data


def _load_known(root,marker):return _load_prediction(root,marker,True)
def _load_novel(root,marker):return _load_prediction(root,marker,False)


def _apply(data,state):
    # The apply API has no true labels, RX, scene, unknown identity or quota.
    margins=core.apply(data['scores'][:,4],data['closed'],state)
    if margins.shape!=(len(data['ids']),2) or margins.dtype!=np.float64 or not np.isfinite(margins).all():
        raise ValueError('Two finite predicted-class-only knownness margins required')
    return {**{k:v for k,v in data.items() if k!='scores'},'scores':margins}


def run(config):
    validate_config(config);output=Path(config['output_root'])
    if output.exists():raise FileExistsError('A new immutable calibration-only output root is required')
    root=Path(config['parent_prediction_root']);parent=_parent(root)
    readout,smoke=_smoke(root,parent);readout_before=copy.deepcopy(readout)
    output.mkdir(parents=True,exist_ok=False);_write(output/'smoke.json',smoke)
    parent_tau=_parent_threshold(root,parent)
    known=_load_known(root,parent)
    state,report=core.calibrate_known(known['ids'],known['y'],known['closed'],known['baseline_closed'],known['scores'][:,4],known['rx_i'],known['scenario'])
    if (not isinstance(state,dict) or state.get('schema')!=core.STATE_SCHEMA or set(state)!={'schema','global_threshold','conditional_thresholds'}
            or not np.isfinite(state['global_threshold']) or state['conditional_thresholds'].shape!=(6,)
            or state['conditional_thresholds'].dtype!=np.float64 or not np.isfinite(state['conditional_thresholds']).all()):
        raise ValueError('One finite global and six frozen conditional thresholds required')
    if state['global_threshold']!=parent_tau:raise ValueError('Recomputed global control threshold differs from the frozen parent')
    frozen=copy.deepcopy(state);known_prediction=_apply(known,state)
    if not _same(state,frozen) or not _same(readout,readout_before):raise ValueError('Frozen source state changed during known prediction')
    _npz(output/'thresholds.npz',{'global':np.asarray([state['global_threshold']],np.float64),'conditional':state['conditional_thresholds']})
    calibration=output/'calibration';calibration.mkdir()
    _write(calibration/'complete.json',dict(report,schema=SCHEMA,status=CALIBRATION_STATUS,run_id=RUN_ID,code_commit=config['code_commit'],
        method_names=list(METHODS),primary_method='predicted_class',group_count=42,known_only=True,unknown_inputs_used=False,
        global_threshold=float(state['global_threshold']),conditional_thresholds=state['conditional_thresholds'].tolist(),
        score_threshold=0.,routing='predicted_class',classification_path='P1',source_only=True,target_reads=0,conformal_guarantee=False))
    predict=output/'predict';predict.mkdir();_npz(predict/'known_predictions.npz',known_prediction)
    print(CALIBRATION_STATUS,flush=True)
    # First novel file open; no novel score or private truth participated in fitting.
    novel=_load_novel(root,parent)
    if set(known['ids'].tolist())&set(novel['ids'].tolist()):raise ValueError('Known and heldout novel prediction IDs must be separate')
    novel_prediction=_apply(novel,state)
    if not _same(state,frozen) or not _same(readout,readout_before):raise ValueError('Frozen source state changed during novel prediction')
    _npz(predict/'novel_predictions.npz',novel_prediction)
    result=dict(status=STATUS,schema=SCHEMA,run_id=RUN_ID,code_commit=config['code_commit'],config=config,method_names=list(METHODS),
        primary_method='predicted_class',classification_path='P1',routing='predicted_class',use_predicted_class_only=True,score_threshold=0.,
        parent_run_id=PARENT_RUN,parent_code_commit=PARENT_COMMIT,parent_prediction_root=str(root),parent_method=PARENT_METHODS[4],parent_score_column=4,
        parent_F0_identity=parent['parent_F0_identity'],dura_identity=parent['dura_identity'],counts=dict(COUNTS),
        source_only=True,target_reads=0,source_novel_truth_reads=0,known_count=COUNTS['V'],novel_count=COUNTS['novel'],
        new_encoder_updates=0,new_identity_head_updates=0,new_readout_updates=0,new_neural_updates=0,
        F0_forward_rows=0,G_forward_rows=0,IQ_reads=0,channel_generations=0,new_independent_physical_records=0,
        historical_readout_optimizer_updates=2000,all_thresholds_frozen_before_novel=True,model_parameters_buffers_unchanged=True,
        parent_readout_state_unchanged=True,global_control_matches_parent=True,source_V_labels_used_for_calibration=True,
        unknown_calibration_scores_read=0,unknown_calibration_labels_read=0,conformal_guarantee=False,
        known_predictions_ref='predict/known_predictions.npz',novel_predictions_ref='predict/novel_predictions.npz',
        thresholds_ref='thresholds.npz',calibration_ref='calibration/complete.json',final_phase3_complete=False,
        scope='source-known empirical class-stratified risk; repeated source-heldout novel evaluation; no independent target confirmation')
    _write(output/'complete.json',result);print(STATUS,flush=True);return result


def main(argv=None):
    print(json.dumps(dict(event='STAR_CLASS_RISK_PROCESS',pid=os.getpid(),cwd=os.getcwd(),argv=list(sys.argv) if argv is None else list(argv)),ensure_ascii=False),flush=True)
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    command=sub.add_parser('run');command.add_argument('--config',required=True)
    args=parser.parse_args(argv);done=run(_read(args.config));print(json.dumps(dict(status=done['status'])),flush=True);return 0


if __name__=='__main__':raise SystemExit(main())
