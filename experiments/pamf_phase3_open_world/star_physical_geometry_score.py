"""Truth-last frozen source-novel geometry for the two physical-learning arms.

F0 features are reused from their completed export; no model, IQ, target data
or fitting is consumed here. TX truth evaluates five fixed neighbor maps.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys

import numpy as np

from . import star_consensus_score as common
from .star_open_feature_score import (
    CONDITIONS, _nearest_indices, _retrieval, _cosine_gap, _spectrum, _between_ratio,
)


RUN_ID='20261030-phase3-star-physical-geometry-s2026092701-r01'
PARENT_RUN='20261030-phase3-star-physical-open-s2026092701-r01'
PARENT_COMMIT='31fa172946aeaecd2d3f1e784aee7366b7b5bee1'
BASELINE_RUN='20261030-phase3-star-open-feature-diagnostic-s2026092701-r01'
BASELINE_COMMIT='55977eec2cf0c2bd09d7c6856179f919e9f0923d'
BASELINE_SCHEMA='pamf_phase3_open_feature_export_v1'
BASELINE_STATUS='STAR_OPEN_FEATURES_COMPLETE'
EXPORT_SCHEMA='pamf_phase3_physical_geometry_export_v1'
EXPORT_STATUS='STAR_PHYSICAL_GEOMETRY_FEATURES_COMPLETE'
SCHEMA='pamf_phase3_physical_geometry_score_v1'
STATUS='STAR_PHYSICAL_GEOMETRY_SCORES_COMPLETE'
SOURCE_COUNT=3150
ARMS=('P_L','P_LOE')
REPRESENTATIONS={'F0.fused160':dict(file='F0.npz',key='fused160',dimension=160)}|{
    arm+'.'+key:dict(file=arm+'.npz',key=key,dimension=dimension)
    for arm in ARMS for key,dimension in (('fused160',160),('open64',64))}
_read,_write=common._read,common._write


def _matches(value,required):
    return isinstance(value,dict) and all(value.get(k)==v for k,v in required.items())


def _prediction(root):
    root=Path(root);done=_read(root/'complete.json')
    required=dict(schema=EXPORT_SCHEMA,status=EXPORT_STATUS,run_id=RUN_ID,source_count=SOURCE_COUNT,
        optimizer_updates=0,target_reads=0,private_truth_reads=0,model_states_unchanged=True,source_role='novel',
        model_updates=0,threshold_updates=0,source_L_reads=0,source_V_reads=0,source_OE_reads=0,original_U_s_reads=0,
        G_forward_rows=0,input_path='cached_P1_only',received_cache_values_read=0,baseline_F0_features_reused=True,
        actual_frozen_checkpoints=list(ARMS),real_F_forward_rows=2*SOURCE_COUNT,synthetic_F_forward_rows=4)
    if (not _matches(done,required) or not isinstance(done.get('code_commit'),str)
            or re.fullmatch('[0-9a-f]{40}',done['code_commit']) is None or done.get('representations')!=REPRESENTATIONS):
        raise ValueError('All five source-only frozen physical geometry exports must complete before private truth')
    parents=done.get('parent_identities',{})
    if not isinstance(parents,dict) or set(parents)!={'F0',*ARMS}:
        raise ValueError('Reused original F0 and both fixed physical source parent identities required')
    f0=parents['F0'];native_keys=('source_output','source_contract','base_release')
    if (not _matches(f0,dict(model='original_F0_E50',parameters=232993,epoch=50,updates=11100,model_seed=2026092701))
            or any(not isinstance(f0.get(k),str) or not f0[k].strip() for k in native_keys)):
        raise ValueError('The fixed original F0 E50 source identity is required')
    for arm in ARMS:
        identity=parents[arm]
        if (not _matches(identity,dict(run_id=PARENT_RUN,code_commit=PARENT_COMMIT,arm=arm,
                updates=2220,cumulative_updates=13320,identifier_parameters=315106,open_world_parameters=82113,
                training_container_parameters=982835,identity_head_replaced=False,parent_F0_identity=f0,
                input_path='cached_P1_and_same_record_P0',identity_classification_path='P1',reference_mode='constant',
                membership_task='rotating_five',physical_mode='L_only' if arm=='P_L' else 'L_and_OE',
                training_input_paths=['cached_P1','same_record_cached_P0']))
                or not isinstance(identity.get('checkpoint'),str) or not identity['checkpoint'].strip()
                or not _matches(identity.get('original_parent_identity'),dict(parent_epoch=50,parent_updates=11100,
                    model_seed=2026092701,**{k:f0[k] for k in native_keys}))):
            raise ValueError('Each representation must use its declared frozen complete physical-learning source arm')
    manifest=done.get('source_manifest',{})
    if (not _matches(manifest,dict(schema='pamf_phase3_star_representation_cache_role_v1',status='STAR_REPRESENTATION_ROLE_COMPLETE',
            role='novel',count=SOURCE_COUNT,iq_shape=[SOURCE_COUNT,2,256],source_only=True,all_new_IQ_satellite=True,
            restored_input_path='P1',parent_F0_identity=f0))
            or not isinstance(manifest.get('dura_identity'),dict) or not manifest['dura_identity']):
        raise ValueError('Only the matching existing source-novel G10 restored cache may be exported')
    baseline=done.get('baseline_original_export_identity',{})
    if (not _matches(baseline,dict(schema=BASELINE_SCHEMA,status=BASELINE_STATUS,run_id=BASELINE_RUN,
            code_commit=BASELINE_COMMIT,source_count=SOURCE_COUNT,source_role='novel',source_manifest=manifest))
            or not _matches(baseline.get('parent_identities'),dict(F0=f0))):
        raise ValueError('F0 must reuse the exact completed original feature-export identity and source cache')
    metadata=None;features={}
    for arm in ('F0',*ARMS):
        fields={'ids','rx_i','day_i','fused160'}|({'open64'} if arm!='F0' else set())
        with np.load(root/(arm+'.npz'),allow_pickle=False) as saved:
            if set(saved.files)!=fields:raise ValueError('Exact unit feature/public metadata fields required; labels and raw features forbidden')
            data={k:saved[k].copy() for k in saved.files}
        if (not common._ids(data['ids'],SOURCE_COUNT)
                or any(data[k].shape!=(SOURCE_COUNT,) or data[k].dtype!=np.int64 or not np.isin(data[k],roster).all()
                    for k,roster in (('rx_i',common.RECEIVERS),('day_i',(1,2,3))))):
            raise ValueError('Unique opaque IDs and legal aligned source RX/day metadata required')
        current={k:data[k] for k in ('ids','rx_i','day_i')}
        if metadata is not None and any(not np.array_equal(current[k],metadata[k]) for k in current):
            raise ValueError('All five representations must preserve exactly the same ID order and public metadata')
        metadata=current
        for name,spec in REPRESENTATIONS.items():
            if spec['file']!=arm+'.npz':continue
            x=data[spec['key']]
            if x.shape!=(SOURCE_COUNT,spec['dimension']) or x.dtype!=np.float32 or not np.isfinite(x).all():
                raise ValueError('Aligned finite FP32 features with the declared dimensions required')
            norm=np.linalg.norm(x.astype(np.float64),axis=1)
            if not np.all((norm==0)|np.isclose(norm,1.,rtol=0.,atol=2e-6)):
                raise ValueError('Only already unit-normalized features or exact zero vectors are accepted')
            features[name]=x
    return done,metadata,features


def score(feature_root,novel_truth,output):
    output=Path(output)
    if output.exists():raise FileExistsError('New immutable independent physical geometry score root required')
    marker,metadata,features=_prediction(feature_root)
    nearest={}
    for name in REPRESENTATIONS:
        nearest[name]=_nearest_indices(features[name],metadata['ids'],metadata['rx_i'],metadata['day_i'])
        print('STAR_PHYSICAL_GEOMETRY_NEIGHBORS_FROZEN',name,SOURCE_COUNT,flush=True)
    truth=common._truth(novel_truth,metadata)  # Sole private read, after all five label-blind four-mask choices.
    tx=truth['tx'];rx=metadata['rx_i'];day=metadata['day_i'];results={}
    for name in REPRESENTATIONS:
        x=features[name]
        results[name]=dict(dimension=REPRESENTATIONS[name]['dimension'],
            nearest_neighbor=_retrieval(nearest[name],tx,rx,day,common.UNKNOWN_TX),
            cosine_cross_RX_and_day=_cosine_gap(x,tx,rx,day),spectrum=_spectrum(x),
            between_group_variance_ratio=dict(TX=_between_ratio(x,tx),RX=_between_ratio(x,rx),day=_between_ratio(x,day)),
            variance_scope='one-factor between-group sum of squares / total centered sum of squares; not additive or causal; collapse N/A')
        print('STAR_PHYSICAL_GEOMETRY_SCORED',name,flush=True)
    result=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=marker['code_commit'],source_count=SOURCE_COUNT,
        source_role='novel',representations=results,primary_arm='P_LOE',representation_order=list(REPRESENTATIONS),truth_last=True,
        all_neighbors_fixed_before_truth=True,neighbor_selection_TX_labels_used=False,true_K_used_for_neighbor_selection=False,
        parent_run=PARENT_RUN,parent_code_commit=PARENT_COMMIT,baseline_F0_features_reused=True,
        baseline_original_export_run=BASELINE_RUN,baseline_original_export_code_commit=BASELINE_COMMIT,
        actual_frozen_checkpoints=list(ARMS),real_F_forward_rows=marker['real_F_forward_rows'],synthetic_F_forward_rows=4,
        source_only=True,source_RX_seen=True,source_novel_historically_exposed=True,new_blind_confirmation=False,
        required_source_heldout_TX=list(common.UNKNOWN_TX),source_RX=list(common.RECEIVERS),source_days=[1,2,3],
        target_reads=0,target_U_reads=0,target_Q_reads=0,G_forward_rows=0,optimizer_updates=0,trained_parameters=0,
        diagnostic_only=True,not_future_registration_accuracy=True,A_N=None,H_OW=None,
        final_phase3_complete=False,full_goal_passed=False,full_goal_status='ACTIVE',eligible_primary=False,next_step='DIAGNOSTIC_ONLY',
        claim='Frozen source-heldout same-pool geometry only; not unseen RX, blind new TX, unsupervised classification or future anonymous registration')
    output.mkdir(parents=True);_write(output/'results.json',result)
    _write(output/'complete.json',{k:v for k,v in result.items() if k!='representations'})
    print(STATUS,flush=True);return result


def main(argv=None):
    print(json.dumps(dict(event='STAR_PHYSICAL_GEOMETRY_SCORE_PROCESS',pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv)),flush=True)
    p=argparse.ArgumentParser();p.add_argument('--feature-root',required=True);p.add_argument('--novel-truth',required=True)
    p.add_argument('--output',required=True);a=p.parse_args(argv);score(a.feature_root,a.novel_truth,a.output)


if __name__=='__main__':main()
