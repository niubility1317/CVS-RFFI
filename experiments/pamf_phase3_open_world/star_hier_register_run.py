"""Cached source/U providers for the hierarchical anonymous-registration policy."""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import numpy as np
from . import star_signal_register_run as base
from . import star_hier_register_core as core

RUN_ID='20261031-phase3-hier-register-s2026092701-r01'
SCHEMA='pamf_phase3_hier_register_prediction_v1'
STATUS='HIER_REGISTER_PREDICTIONS_COMPLETE'
MODELS_STATUS='HIER_REGISTER_SOURCE_FROZEN'
REGISTRATION_STATUS='HIER_REGISTER_ANONYMOUS_FROZEN'
CALIBRATION_STATUS='HIER_REGISTER_THRESHOLDS_FROZEN'
INPUT_CODE='e48f3b357eccb12e778c1eb348776e5c6b457922'
METHODS,PRIMARY_METHOD,BASE_METHODS=base.METHODS,base.PRIMARY_METHOD,base.BASE_METHODS
CONDITIONS,COUNTS,Q_COUNT,U_COUNT=base.CONDITIONS,base.COUNTS,base.Q_COUNT,base.U_COUNT
CACHED_U_FEATURES=True
FIELDS=base.FIELDS|{'input_reference_root'}
INPUT_U_FIELDS={'ids','default_logits','closed','signal_features','base_scores','assignments','dbscan_labels','rejected'}
parent=base.parent
_rows,_target_rows,_base=base._rows,base._target_rows,base._base
_same,_read,_write,_npz=base._same,base._read,base._write,base._npz


def validate_config(c):
    if (not isinstance(c,dict) or set(c)!=FIELDS or c.get('run_id')!=RUN_ID
            or not isinstance(c.get('input_reference_root'),str) or not c['input_reference_root'].strip()):
        raise ValueError('Exact hierarchical U-only configuration plus completed input cache required; no truth/K')
    old={k:v for k,v in c.items() if k!='input_reference_root'};old['run_id']=base.RUN_ID;base.validate_config(old)
    root=Path(c['input_reference_root']);out=Path(c['output_root']).resolve();resolved=root.resolve()
    if root.name!='predict' or root.parent.name!=base.RUN_ID:
        raise ValueError('Only the original signal-register prediction input cache is allowed')
    if out.is_relative_to(resolved) or resolved.is_relative_to(out):
        raise ValueError('New immutable output cannot overlap the original input cache')
    return c


def _headers(c,identity):
    """Metadata only. No old calibration, source/query predictions, or U values."""
    root=Path(c['input_reference_root']);done=_read(root/'complete.json')
    expected=dict(schema=base.SCHEMA,status=base.STATUS,run_id=base.RUN_ID,code_commit=INPUT_CODE,
        method_names=list(METHODS),primary_method=PRIMARY_METHOD,known_count=COUNTS['V'],novel_count=COUNTS['novel'],U_count=U_COUNT,
        source_states_unchanged=True,model_parameters_buffers_unchanged=True,original_neural_models_source_only=True,
        original_density_refitted=False,neural_updates=0,private_truth_reads=0,U_truth_reads=0,
        all_three_conditions_complete=True,all_conditions_through_DURA=True,source_radius_frozen_before_U=True,anonymous_frozen_before_V_and_Q=True)
    if any(done.get(k)!=v for k,v in expected.items()):raise ValueError('Completed original source-only neural/cache ancestry required')
    old=base.validate_config(done['config'])
    if (old['code_commit']!=INPUT_CODE or Path(old['output_root']).resolve()!=root.resolve()
            or old['reference_config']!=c['reference_config']
            or any(old[k]!=c[k] for k in ('U_public_root','U_capsule_id','U_split_id'))
            or done.get('model_identity')!=identity or done.get('parent_F0_identity')!=identity['parent_F0_identity']
            or done.get('dura_identity')!=identity['dura_identity']):
        raise ValueError('Cached U must retain the exact current F0, r03, DURA and public-source contract')
    prior=done.get('reference_prediction_identity',{})
    if prior.get('config')!=c['reference_config'] or prior.get('model_identity')!=identity:
        raise ValueError('Original selected source-only density ancestry differs')
    manifest=done.get('U_manifest',{});classes=c['reference_config']['train_config']['native_config']['registered_classes']
    if (manifest.get('count')!=U_COUNT or manifest.get('capsule_id')!=c['U_capsule_id'] or manifest.get('split_id')!=c['U_split_id']
            or manifest.get('registered_classes')!=classes):raise ValueError('Fixed complete star-G10 discovery U metadata required')
    fit=_read(root/'models/complete.json');model_expected=dict(schema=base.SCHEMA,status=base.MODELS_STATUS,run_id=base.RUN_ID,code_commit=INPUT_CODE,
        model_identity=identity,source_L_count=COUNTS['L'],source_radius_fitted=True,source_radius_frozen_before_U=True,
        original_density_refitted=False,source_V_fit=False,target_U_reads=0)
    if any(fit.get(k)!=v for k,v in model_expected.items()) or not isinstance(fit.get('source_L_manifest'),dict) or not isinstance(fit.get('diagnostics'),dict):
        raise ValueError('Already frozen original source-L radius and source manifest required')
    registration=_read(root/'registration/complete.json')
    registration_expected=dict(schema=base.SCHEMA,status=base.REGISTRATION_STATUS,run_id=base.RUN_ID,code_commit=INPUT_CODE,U_count=U_COUNT,
        U_manifest=manifest,anonymous_frozen_before_V_and_Q=True,true_U_labels_used=False,true_unknown_count_used=False,query_fit=False,
        source_prediction_reads=0,U_bank_deployed=False)
    if any(registration.get(k)!=v for k,v in registration_expected.items()):raise ValueError('Only the original frozen label-free U feature export may be reused')
    return done,fit


def _source(c,b,identity,device,state):
    _,fit=_headers(c,identity);root=Path(c['input_reference_root'])
    with np.load(root/'models/density.npz',allow_pickle=False) as z:stored={k:z[k].copy() for k in z.files}
    core.base._state(stored)
    if not _same(stored,state):raise ValueError('Cached source density must equal the current original frozen state')
    with np.load(root/'models/source_radius.npz',allow_pickle=False) as z:source={k:z[k].copy() for k in z.files}
    core.validate_source(source)
    diag=deepcopy(fit['diagnostics']);diag.update(source_radius_reused=True,source_radius_used_for_clustering=False,source_radius_new_fits=0)
    return source,diag,np.asarray([],dtype='U'),deepcopy(fit['source_L_manifest']),dict(source_L_reads=0,source_signal_rows=0,source_radius_new_fits=0)


def _reference(c):
    result=base._reference(c);_headers(c,result[0]['model_identity'])
    return result


def _U(c,b,device):
    done,_=_headers(c,b['model_identity']);root=Path(c['input_reference_root'])
    if b['reader_identity']!=done['parent_F0_identity'] or b['dura_identity']!=done['dura_identity']:
        raise ValueError('Cached U and actual frozen F0/DURA identities differ')
    with np.load(root/'registration/U.npz',allow_pickle=False) as z:
        if set(z.files)!=INPUT_U_FIELDS:raise ValueError('Exact original eight-field U cache without private labels required')
        values={key:z[key].copy() for key in ('ids','default_logits','closed','signal_features')}
    ids=core.base._ids(values['ids'],U_COUNT);logits=values['default_logits'];features=values['signal_features'];closed=values['closed']
    if (logits.shape!=(U_COUNT,6) or logits.dtype!=np.float32 or not np.isfinite(logits).all()
            or closed.shape!=(U_COUNT,) or closed.dtype!=np.int64 or not np.array_equal(closed,logits.argmax(1))
            or features.shape!=(U_COUNT,160) or features.dtype!=np.float32 or not np.isfinite(features).all()
            or not np.allclose(np.linalg.norm(features.astype(np.float64),axis=1),1.,atol=2e-6,rtol=0)):
        raise ValueError('Aligned finite original U default logits, unit r03 features and opaque IDs required')
    return values,deepcopy(done['U_manifest'])


def run(c,device='cuda:0'):return base.run(c,device,policy=sys.modules[__name__])
evaluate=run


def main():
    print(json.dumps(dict(event='HIER_REGISTER_PROCESS',pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv)),flush=True)
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--device',default='cuda:0');args=parser.parse_args()
    run(_read(args.config),args.device)


if __name__=='__main__':main()
