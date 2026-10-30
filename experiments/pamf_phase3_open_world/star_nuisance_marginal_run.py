"""Marginalize source RX/day uncertainty; preserve selected DURA controls."""
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
from . import star_class_density_run as parent
from . import star_nuisance_marginal_core as core
from .star_local_cov_run import _reference
from . import star_signal_open_eval as signal_eval
from .development_transfer import _snapshot_model, _check_model

RUN_ID='20261031-phase3-nuisance-marginal-s2026092701-r01'
SCHEMA='pamf_phase3_nuisance_marginal_prediction_v1'
STATUS='NUISANCE_MARGINAL_PREDICTIONS_COMPLETE'
MODELS_STATUS='NUISANCE_MARGINAL_MODELS_FROZEN'
CALIBRATION_STATUS='NUISANCE_MARGINAL_THRESHOLDS_FROZEN'
PARENT_CODE='4af0668331682636ee212077b38cd912857a1b31'
METHODS=('pooled_ratio','predicted_ratio','marginal_pooled_ratio','marginal_predicted_ratio')
PRIMARY_METHOD=METHODS[3]
CONDITIONS=parent.CONDITIONS
COUNTS=parent.COUNTS
Q_COUNT=parent.Q_COUNT
FIELDS={'run_id','code_commit','reference_config','output_root','batch_size'}
_read,_write,_npz,_same=parent._read,parent._write,parent._npz,parent._same
_rows,_target_rows=parent._rows,parent._target_rows
risk=parent.risk


def validate_config(c):
    if (not isinstance(c,dict) or set(c)!=FIELDS or c.get('run_id')!=RUN_ID
            or re.fullmatch('[0-9a-f]{40}',str(c.get('code_commit'))) is None
            or type(c.get('batch_size')) is not int or c['batch_size']!=128
            or not isinstance(c.get('output_root'),str) or not c['output_root'].strip()):
        raise ValueError('Exact source-only nuisance marginalization configuration required; no truth/U/K')
    p=parent.validate_config(c['reference_config'])
    if p['code_commit']!=PARENT_CODE:raise ValueError('The selected completed class-density source model is required')
    output=Path(c['output_root']).resolve()
    paths=(p['output_root'],p['reference_prediction_root'],p['train_config']['IQ_root'],p['train_config']['output_root'])
    if any(output.is_relative_to(Path(v).resolve()) or Path(v).resolve().is_relative_to(output) for v in paths):
        raise ValueError('New immutable output cannot overlap source or frozen prediction inputs')
    return c


def _validate_prior_state(state,prior):
    core.validate_prior(prior)
    mu=state['known_means'].reshape(6,15,160)
    offsets=(mu-mu.mean(axis=1,keepdims=True)).mean(axis=0)
    offsets-=offsets.mean(axis=0,keepdims=True)
    covariance=(prior['basis'].T*prior['variances'])@prior['basis']
    if not np.allclose(covariance,offsets.T@offsets/15.,rtol=1e-10,atol=1e-10):
        raise ValueError('The nuisance covariance must be exactly source-domain mean variation')
    return prior


def _raw(state,prior,z,closed):
    controls=parent.core.raw_scores(state,z,closed)[:,[1,3]]
    return np.column_stack((controls,core.raw_scores(state,prior,z,closed)))


def _combine(a,raw,tau):
    result=dict(a,raw_scores=raw,scores=raw-tau[:,a['closed']].T)
    for name in ('raw_scores','scores'):
        if not np.array_equal(result[name][:,:2],a[name][:,[1,3]]):
            raise ValueError('Both selected operating points must be copied exactly')
    return result


def run(c,device='cuda:0'):
    validate_config(c);out=Path(c['output_root'])
    if out.exists():raise FileExistsError('Observe existing output; no replacement or automatic restart')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    p=c['reference_config'];b=parent._backend(p,device)
    snapshots=[(model,_snapshot_model(model)) for model in (b['reader'],b['model'],b['dura'][0])]
    def unchanged():
        for model,before in snapshots:_check_model(model,before)
    smoke=parent._smoke(b,device);unchanged()
    if smoke.get('status')!='PASS' or smoke.get('actual_F0_signal_DURA_checkpoints') is not True:
        raise ValueError('Actual-checkpoint no-query smoke must pass before aggregate derivation or cached source/query access')
    pm,state,old_tau,old_cal=_reference(c);identity=pm['model_identity']
    if b['model_identity']!=identity or b['reader_identity']!=identity['parent_F0_identity']:
        raise ValueError('Actual frozen original F0 and r03 model differ')
    if b['dura_identity']!=pm['loaded_dura_identity']:
        raise ValueError('Actual DURA v1 identity differs from all three inherited tests')
    original=deepcopy(state);out.mkdir(parents=True);_write(out/'resolved_config.json',c);_write(out/'smoke.json',smoke)
    prior,diag=core.build_prior(state);_validate_prior_state(state,prior);unchanged()
    prior_before=deepcopy(prior)
    def check():
        unchanged()
        if not _same(state,original) or not _same(prior,prior_before):raise ValueError('Frozen density or nuisance prior changed')
    check();(out/'models').mkdir();_npz(out/'models/density.npz',state);_npz(out/'models/prior.npz',prior)
    _write(out/'models/complete.json',dict(schema=SCHEMA,status=MODELS_STATUS,run_id=RUN_ID,code_commit=c['code_commit'],
        models_frozen=True,original_density_refitted=False,nuisance_prior_fitted=True,prior_rank=len(prior['basis']),source_L_count=0,
        source_V_fit=False,novel_fit=False,target_fit=False,source_aggregate_only=True,diagnostics=diag))
    tau=np.empty((4,6),np.float64);tau[:2]=old_tau
    reports={METHODS[j]:deepcopy(old_cal['methods'][parent.METHODS[k]]) for j,k in enumerate((1,3))}
    def raw(a):
        result=_raw(state,prior,a['signal_features'],a['closed'])
        if not np.allclose(result[:,:2],a['raw_scores'][:,[1,3]],rtol=1e-12,atol=1e-9):
            raise ValueError('Selected pooled and conditional ratios do not reconstruct')
        result[:,:2]=a['raw_scores'][:,[1,3]];return result
    V=_rows(p['output_root'],'known')
    values=raw(V);check()
    for j in (2,3):
        fitted,report=risk.calibrate_known(V['ids'],V['y'],V['closed'],V['baseline_closed'],values[:,j],V['rx_i'],V['scenario'])
        tau[j]=fitted['conditional_thresholds'];reports[METHODS[j]]=report;check()
    frozen_tau=tau.copy();_npz(out/'thresholds.npz',dict(conditional=tau));(out/'calibration').mkdir()
    _write(out/'calibration/complete.json',dict(schema=SCHEMA,status=CALIBRATION_STATUS,run_id=RUN_ID,code_commit=c['code_commit'],
        method_names=list(METHODS),primary_method=PRIMARY_METHOD,known_count=COUNTS['V'],group_count=42,
        hard_names=list(risk.HARD_NAMES),diagnostic_names=list(risk.DIAGNOSTIC_NAMES),methods=reports,
        conditional_thresholds=tau.tolist(),known_only=True,unknown_inputs_used=False,target_reads=0,
        source_V_neural_fit=False,source_V_density_fit=False,source_V_threshold_fit=True,shared_thresholds_for_all_conditions=True))
    _npz(out/'known_predictions.npz',_combine(V,values,tau));print(CALIBRATION_STATUS,flush=True)
    N=_rows(p['output_root'],'novel');seen=set(V['ids'].tolist())
    if seen&set(N['ids'].tolist()):raise ValueError('Source novel must not overlap fitting/calibration roles')
    _npz(out/'novel_predictions.npz',_combine(N,raw(N),tau));seen.update(N['ids'].tolist());check()
    for name in CONDITIONS:
        a=_target_rows(p['output_root'],name)
        if seen&set(a['ids'].tolist()):raise ValueError('Source fitting/calibration/test cannot overlap target Q')
        folder=out/'targets'/name;folder.mkdir(parents=True);_npz(folder/'predictions.npz',_combine(a,raw(a),tau));check()
        print('NUISANCE_MARGINAL_Q_COMPLETE',name,len(a['ids']),flush=True)
    if not np.array_equal(tau,frozen_tau):raise ValueError('Source thresholds changed after novel/query access')
    done=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c,
        method_names=list(METHODS),primary_method=PRIMARY_METHOD,reference_prediction_identity=pm,
        model_identity=identity,parent_F0_identity=identity['parent_F0_identity'],dura_identity=identity['dura_identity'],
        classification_path='P1',identity_baseline_path='same_DURA_output_P1',score_threshold=0.,known_count=COUNTS['V'],novel_count=COUNTS['novel'],
        Q_counts=pm['Q_counts'],Q_manifests=pm['Q_manifests'],source_only=True,private_truth_reads=0,
        source_novel_truth_reads=0,target_truth_reads=0,target_U_reads=0,original_U_s_reads=0,formal_target_accessed=False,
        neural_updates=0,density_fit_updates=0,nuisance_prior_fitted=True,prior_rank=len(prior['basis']),source_L_reads=0,source_OE_reads=0,
        models_frozen_before_V=True,all_thresholds_frozen_before_novel_and_Q=True,all_three_conditions_complete=True,
        all_conditions_through_DURA=True,source_states_unchanged=True,model_parameters_buffers_unchanged=True,
        controls_reused_exactly=True,identity_predictions_unchanged=True,source_bank_deployed=False,per_record_source_rows_deployed=0,
        source_V_threshold_fit=True,real_signal_forward_rows=0,real_F_forward_rows=0,G_forward_rows=0,
        synthetic_signal_forward_rows=2,synthetic_F_forward_rows=2,G_synthetic_forward_rows=2,
        queries_reused_without_new_neural_forward=True,final_phase3_complete=False,full_goal_status='ACTIVE',A_N=None,H_OW=None)
    _write(out/'complete.json',done);print(STATUS,flush=True);return done


def main():
    print(json.dumps(dict(event='NUISANCE_MARGINAL_PROCESS',pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv)),flush=True)
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    run(_read(a.config),a.device)


if __name__=='__main__':main()
