"""Independent source-only protected160 scores and frozen-neighbor geometry."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from . import star_protected_open_eval as evaluation
from . import star_consensus_score as common
from . import star_open_feature_score as geometry
from .star_physical_open_score import _metrics, _known_pass

RUN_ID=evaluation.RUN_ID
SCHEMA='pamf_phase3_star_protected_scores_v1'
STATUS='STAR_PROTECTED_SCORES_COMPLETE'
_read,_write=common._read,common._write


def _prediction(root):
    root=Path(root);m=_read(root/'complete.json')
    expected=dict(schema=evaluation.SCHEMA,status=evaluation.STATUS,run_id=RUN_ID,method='relation_max',primary_arm='R1',path='P1',
        known_count=evaluation.COUNTS['V'],novel_count=evaluation.COUNTS['novel'],source_L_count=evaluation.COUNTS['L'],source_only=True,
        source_truth_reads=0,target_reads=0,source_OE_reads=0,original_U_s_reads=0,source_V_neural_fit=False,
        native_identity_frozen=True,native_updates=0,eval_neural_updates=0,identity_predictions_unchanged=True,
        aggregates_frozen_before_V=True,thresholds_frozen_before_novel=True,frozen_models_verified=True,source_aggregates_unchanged=True,
        source_roles_disjoint=True,source_bank_deployed=False,per_record_source_rows_deployed=0,
        identifier_parameters=299170,open_world_parameters=66177,training_container_parameters=966899,
        G_forward_rows=0,received_cache_values_read=0,formal_target_accessed=False,novel_features_same_forward=True)
    if m.get('arm') not in evaluation.ARMS or any(m.get(k)!=v for k,v in expected.items()):
        raise ValueError('All frozen source predictions and native identity must complete before private truth')
    c=evaluation.validate_config(m['config'])
    if (any(c.get(k)!=m.get(k) for k in ('run_id','code_commit','arm')) or evaluation._identity(m['model_identity'],c)!=m['parent_F0_identity']
            or m['dura_identity']!=m['model_identity']['dura_identity']):
        raise ValueError('Frozen source config/checkpoint/parent identities differ')
    smoke=_read(root/'smoke.json')
    if any(smoke.get(k)!=v for k,v in dict(status='PASS',actual_checkpoint=True,native_identity_unchanged=True).items()):
        raise ValueError('Actual-checkpoint no-query smoke must pass before completed source predictions')
    with np.load(root/'source_aggregates.npz',allow_pickle=False) as z:
        aggregates={k:z[k].copy() for k in z.files}
    evaluation.validate_aggregates(aggregates)
    if int(aggregates['counts'].sum())!=evaluation.COUNTS['L']:
        raise ValueError('Source aggregate counts must cover exactly L')
    tau=np.load(root/'thresholds.npy',allow_pickle=False);cal=_read(root/'calibration/complete.json')
    if (tau.shape!=(6,) or tau.dtype!=np.float64 or not np.isfinite(tau).all()
            or any(cal.get(k)!=v for k,v in dict(schema=evaluation.SCHEMA,status=evaluation.CALIBRATION_STATUS,
                run_id=RUN_ID,code_commit=m['code_commit'],arm=m['arm'],method='relation_max',
                selected_calibration_method='predicted_class',aggregates_fit_role='L',known_count=evaluation.COUNTS['V'],
                group_count=42,known_only=True,source_V_neural_fit=False,source_V_threshold_fit=True,unknown_inputs_used=False,target_reads=0).items())
            or cal.get('conditional_thresholds')!=tau.tolist()):
        raise ValueError('Known-only completed six predicted-class thresholds required')
    rows={}
    for role,fields,n in (('known',evaluation.KNOWN_FIELDS,evaluation.COUNTS['V']),('novel',evaluation.NOVEL_FIELDS,evaluation.COUNTS['novel'])):
        with np.load(root/(role+'_predictions.npz'),allow_pickle=False) as z:
            if set(z.files)!=fields:raise ValueError('Exact public predictions required; novel truth/TX prohibited')
            a={k:z[k].copy() for k in z.files}
        if (not common._ids(a['ids'],n) or any(a[k].shape!=(n,6) or a[k].dtype!=np.float32 or not np.isfinite(a[k]).all() for k in ('default_logits','relation_logits'))
                or a['closed'].shape!=(n,) or a['closed'].dtype!=np.int64 or not np.array_equal(a['closed'],a['default_logits'].argmax(1))
                or a['scores'].shape!=(n,) or a['scores'].dtype!=np.float64 or not np.isfinite(a['scores']).all()
                or not np.allclose(a['scores'],a['relation_logits'].max(1).astype(np.float64)-tau[a['closed']],rtol=0.,atol=1e-12)
                or any(a[k].shape!=(n,) or a[k].dtype!=np.int64 or not np.isin(a[k],roster).all() for k,roster in (('rx_i',evaluation.RECEIVERS),('day_i',(1,2,3))))):
            raise ValueError('Finite same-P1 default identity and reconstructible relation margin required')
        if role=='known':
            if (a['y'].shape!=(n,) or a['y'].dtype!=np.int64 or not np.isin(a['y'],range(6)).all()
                    or a['scenario'].shape!=(n,) or a['scenario'].dtype.kind!='U' or not np.isin(a['scenario'],evaluation.SCENES).all()
                    or a['baseline_closed'].shape!=(n,) or a['baseline_closed'].dtype!=np.int64 or not np.array_equal(a['closed'],a['baseline_closed'])):
                raise ValueError('Visible V labels and unchanged independent same-P1 original F0 baseline required')
            rebuilt,report=evaluation.risk.calibrate_known(a['ids'],a['y'],a['closed'],a['baseline_closed'],a['relation_logits'].max(1).astype(np.float64),a['rx_i'],a['scenario'])
            if (not np.array_equal(rebuilt['conditional_thresholds'],tau)
                    or any(report[k]!=cal[k] for k in ('methods','coverage','constraints_passed','common_tail_rate'))):
                raise ValueError('Known-risk calibration must reconstruct before private truth')
        rows[role]=a
    if set(rows['known']['ids'].tolist())&set(rows['novel']['ids'].tolist()):raise ValueError('Known/heldout physical IDs must be disjoint')
    with np.load(root/'novel_features.npz',allow_pickle=False) as z:
        if set(z.files)!=evaluation.FEATURE_FIELDS:raise ValueError('Only public same-forward novel geometry fields allowed')
        features={k:z[k].copy() for k in z.files}
    novel=rows['novel'];n=len(novel['ids'])
    if any(not np.array_equal(features[k],novel[k]) for k in ('ids','rx_i','day_i')):
        raise ValueError('Novel geometry and frozen prediction IDs/domains must align exactly')
    for key in ('original_fused160','protected160'):
        x=features[key]
        if x.shape!=(n,160) or x.dtype!=np.float32 or not np.isfinite(x).all():raise ValueError('Finite FP32 source-heldout unit160 geometry required')
        norms=np.linalg.norm(x.astype(np.float64),axis=1)
        if not np.all((norms==0)|np.isclose(norms,1.,rtol=0.,atol=2e-6)):raise ValueError('Source geometry must be unit or zero')
    return m,cal,rows,features


def score(prediction_root,novel_truth,output):
    output=Path(output)
    if output.exists():raise FileExistsError('New immutable independent protected source scoring root required')
    m,cal,rows,features=_prediction(prediction_root);known,novel=rows['known'],rows['novel']
    representations=('original_fused160','protected160')
    nearest={name:geometry._nearest_indices(features[name],novel['ids'],novel['rx_i'],novel['day_i']) for name in representations}
    truth=common._truth(novel_truth,novel)  # Sole private read after ALL arrays/calibration/neighbors are fixed.
    km=dict(common._masks(known['rx_i'],known['scenario']));nm=dict(common._masks(novel['rx_i'],truth['scenario']))
    def metric(k,n,kr=tuple(range(6)),ur=common.UNKNOWN_TX):
        return _metrics(known['y'][k],known['closed'][k],known['scores'][k],truth['tx'][n],novel['scores'][n],known['baseline_closed'][k],kr,ur)
    groups={key:metric(mask,nm[key]) for key,mask in km.items()};allk=np.ones(len(known['ids']),bool);alln=np.ones(len(novel['ids']),bool)
    source=dict(overall=groups['overall'],groups=groups,
        by_RX={str(r):groups['rx/'+str(r)] for r in common.RECEIVERS},by_scene={s:groups['scene/'+s] for s in common.SCENES},
        by_day={str(d):metric(known['day_i']==d,novel['day_i']==d) for d in (1,2,3)},
        by_TX={str(c):metric(known['y']==c,alln,(c,)) for c in range(6)}|{str(c):metric(allk,truth['tx']==c,ur=(c,)) for c in common.UNKNOWN_TX},
        known_only_calibration_passed=bool(cal['methods']['predicted_class']['constraints_passed']),
        known_42_groups_passed=all(_known_pass(r) for r in groups.values()),source_RX_seen=True,source_novel_historically_exposed=True)
    overall=source['overall'];unknown=bool(overall['coverage']['unknown_complete'] and overall['R_U'] is not None and overall['R_U']>=.90-1e-12)
    passed=bool(source['known_42_groups_passed'] and source['known_only_calibration_passed'] and unknown)
    eligible=bool(m['arm']=='R1' and passed)
    source_geometry={}
    for name in representations:
        x=features[name];tx=truth['tx'];rx=novel['rx_i'];day=novel['day_i']
        source_geometry[name]=dict(nearest_neighbor=geometry._retrieval(nearest[name],tx,rx,day,common.UNKNOWN_TX),
            cosine_cross_RX_and_day=geometry._cosine_gap(x,tx,rx,day),spectrum=geometry._spectrum(x),
            between_group_variance_ratio=dict(TX=geometry._between_ratio(x,tx),RX=geometry._between_ratio(x,rx),day=geometry._between_ratio(x,day)))
    result=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=m['code_commit'],arm=m['arm'],primary_arm='R1',primary_method='relation_max',path='P1',
        source=source,source_geometry=source_geometry,source_geometry_diagnostic_only=True,primary_summary=overall,source_goals_passed=passed,
        eligible_for_target_prereg=eligible,next_step='ELIGIBLE_FOR_SEPARATE_CROSS_RX_PREREGISTRATION' if eligible else 'NO_PROMOTION',
        source_only=True,source_RX_seen=True,source_novel_historically_exposed=True,new_blind_confirmation=False,
        truth_last=True,all_geometry_neighbors_before_truth=True,geometry_changes_gate=False,geometry_not_registration_accuracy=True,
        target_reads=0,target_U_reads=0,target_Q_reads=0,formal_target_accessed=False,target_DA=False,target_REG=False,A_N=None,H_OW=None,
        final_phase3_complete=False,full_goal_passed=False,full_goal_status='ACTIVE',native_updates=0,optimizer_updates=0,
        identifier_parameters=299170,open_world_parameters=66177,training_container_parameters=966899,
        costs={k:m[k] for k in ('real_F_forward_rows','synthetic_F_forward_rows','adapter_L_feature_rows','G_forward_rows','source_aggregate_bytes','threshold_bytes')},
        scope='Source eligibility only; seen source RX and historically exposed heldout TX, no cross-RX target/DA/anonymous registration success')
    output.mkdir(parents=True);_write(output/'results.json',result)
    _write(output/'complete.json',{k:v for k,v in result.items() if k not in ('source','source_geometry')})
    print(STATUS,m['arm'],result['next_step'],flush=True);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--prediction-root',required=True);p.add_argument('--novel-truth',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();score(a.prediction_root,a.novel_truth,a.output)


if __name__=='__main__':main()
