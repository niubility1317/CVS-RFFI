"""Independent source-only metrics for frozen metric residual density evidence."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from . import star_metric_residual_eval as evaluation
from . import star_consensus_score as common
from .star_physical_open_score import _metrics,_known_pass
from .star_density_score import _paired_counts

RUN_ID=evaluation.RUN_ID
SCHEMA='pamf_phase3_star_metric_residual_scores_v1'
STATUS='STAR_METRIC_RESIDUAL_SCORES_COMPLETE'
METHODS=evaluation.METHODS
PRIMARY=evaluation.PRIMARY_METHOD
COUNTS=evaluation.COUNTS
_read,_write,_truth=common._read,common._write,common._truth


def _prediction(root):
    root=Path(root);m=_read(root/'complete.json')
    expected=dict(schema=evaluation.SCHEMA,status=evaluation.STATUS,run_id=RUN_ID,primary_arm='M_LOE',method_names=list(METHODS),primary_method=PRIMARY,
        classification_path='P1',score_threshold=0.,known_count=COUNTS['V'],novel_count=COUNTS['novel'],source_L_count=COUNTS['L'],source_OE_count=COUNTS['OE'],
        source_only=True,private_truth_reads=0,target_reads=0,original_U_s_reads=0,source_novel_fit_reads=0,real_F_forward_rows=0,synthetic_F_forward_rows=0,
        G_forward_rows=0,G_checkpoint_loaded=False,synthetic_open_forward_rows=2,open_projection_rows=sum(COUNTS.values()),open_projection_rows_by_role=dict(COUNTS),
        eval_neural_updates=0,native_updates=0,default_head_replaced=False,identity_predictions_unchanged=True,control_reused_exactly=True,control_refit=False,control_recalibrated=False,
        models_frozen_before_V=True,all_thresholds_frozen_before_novel=True,source_states_unchanged=True,model_parameters_buffers_unchanged=True,
        source_bank_deployed=False,per_record_source_rows_deployed=0,source_V_model_fit=False,source_V_threshold_fit=True,density_background_uses_OE=True,
        embedding_training_uses_OE=m.get('arm')=='M_LOE',metric_parameters=102688,original_F0_parameters=232993,combined_active_parameters=335681,
        reference_run_id=evaluation.REFERENCE_RUN,reference_code_commit=evaluation.REFERENCE_CODE,final_phase3_complete=False)
    if m.get('arm') not in evaluation.ARMS or any(m.get(k)!=v for k,v in expected.items()):raise ValueError('Complete frozen source metric evaluation must precede private truth')
    c=evaluation.validate_config(m['config'])
    if any(c.get(k)!=m.get(k) for k in ('run_id','code_commit','arm')):raise ValueError('Actual row identity differs')
    parent=evaluation._identity(m['model_identity'],c)
    if parent!=m['parent_F0_identity'] or m['dura_identity']!=m['model_identity']['dura_identity']:raise ValueError('Frozen original F0/G ancestry differs')
    pm,old_state,old_tau,old_global,pc=evaluation._reference(c['reference_prediction_root'],m['model_identity'],m['source_feature_manifest'])
    if pm!=m['reference_prediction_identity']:raise ValueError('Fixed source-reference marker changed')
    smoke=_read(root/'smoke.json')
    if any(smoke.get(k)!=v for k,v in dict(status='PASS',actual_checkpoint=True,synthetic_open_forward_rows=2,synthetic_F_forward_rows=0,real_F_forward_rows=0,G_forward_rows=0).items()):
        raise ValueError('Actual frozen projection-checkpoint synthetic feature smoke required')
    fit=_read(root/'models/complete.json')
    if any(fit.get(k)!=v for k,v in dict(status='STAR_METRIC_RESIDUAL_DENSITIES_FROZEN',run_id=RUN_ID,code_commit=m['code_commit'],arm=m['arm'],
            method_names=list(METHODS),control_reused=True,source_V_fit=False,novel_fit=False).items()):raise ValueError('Both frozen source density states required')
    states={}
    for name in METHODS:
        with np.load(root/'models'/(name+'.npz'),allow_pickle=False) as z:state={k:z[k].copy() for k in z.files}
        evaluation._core().validate_state(state,'density_ratio')
        if state['whitener'].shape!=(160,160):raise ValueError('Both density models must use the fixed D160 family')
        states[name]=state
    if not evaluation._same(states[METHODS[0]],old_state):raise ValueError('Fixed original F0 control model changed')
    with np.load(root/'thresholds.npz',allow_pickle=False) as z:
        if set(z.files)!={'conditional','global'}:raise ValueError('Exact source-known threshold arrays required')
        tau=z['conditional'].copy();global_tau=z['global'].copy()
    cal=_read(root/'calibration/complete.json')
    if (tau.shape!=(2,6) or tau.dtype!=np.float64 or not np.isfinite(tau).all() or global_tau.shape!=(2,) or global_tau.dtype!=np.float64 or not np.isfinite(global_tau).all()
            or not np.array_equal(tau[0],old_tau) or global_tau[0]!=old_global
            or any(cal.get(k)!=v for k,v in dict(schema=evaluation.SCHEMA,status=evaluation.CALIBRATION_STATUS,run_id=RUN_ID,code_commit=m['code_commit'],arm=m['arm'],
                method_names=list(METHODS),primary_method=PRIMARY,known_count=COUNTS['V'],group_count=42,known_only=True,unknown_inputs_used=False,
                source_V_model_fit=False,source_V_threshold_fit=True,target_reads=0,conditional_thresholds=tau.tolist(),global_thresholds=global_tau.tolist(),control_calibration_reused=True).items())
            or cal.get('methods',{}).get(METHODS[0])!=pc['methods'][METHODS[0]]):
        raise ValueError('Unchanged F0 control and all frozen known-only learned-metric thresholds required')
    for j,name in enumerate(METHODS):
        row=cal['methods'][name]
        if set(row.get('groups',{}))!=set(common.GROUP_NAMES) or row.get('conditional_thresholds')!=tau[j].tolist():raise ValueError('All 42 source risk groups required')
    rows={}
    for role in ('known','novel'):
        fields=evaluation.KNOWN_FIELDS if role=='known' else evaluation.NOVEL_FIELDS
        with np.load(root/(role+'_predictions.npz'),allow_pickle=False) as z:
            if set(z.files)!=fields:raise ValueError('Exact public metric predictions required; private novel labels forbidden')
            a={k:z[k].copy() for k in z.files}
        evaluation.reference.parent.validate_rows({k:v for k,v in a.items() if k not in ('contribution_features','learned_features')},role)
        n=len(a['ids']);z=a['learned_features'];inputs=a['contribution_features']
        if (z.shape!=(n,160) or z.dtype!=np.float32 or not np.isfinite(z).all() or not np.allclose(np.linalg.norm(z.astype(np.float64),axis=1),1.,rtol=0,atol=2e-6)
                or inputs.shape!=(n,640) or inputs.dtype!=np.float32 or not np.isfinite(inputs).all()):raise ValueError('Finite frozen same-row learned unit160 and source contributions required')
        old=evaluation._reference_rows(c['reference_prediction_root'],role)
        for key,value in old.items():
            actual=a[key][:,0] if key in ('scores','raw_scores') else a[key]
            if not np.array_equal(actual,value):raise ValueError('Original F0 control/default head/metadata must remain exact')
        raw=evaluation._core().score(states[PRIMARY],z,'density_ratio')
        if (not np.allclose(a['raw_scores'][:,1],raw,rtol=1e-12,atol=1e-9) or not np.array_equal(a['scores'],a['raw_scores']-tau[:,a['closed']].T)):
            raise ValueError('Density/margin must reconstruct from saved frozen learned features and aggregate statistics')
        rows[role]=a
    if set(rows['known']['ids'].tolist())&set(rows['novel']['ids'].tolist()):raise ValueError('Known and novel source roles overlap')
    return m,cal,rows['known'],rows['novel']


def score(prediction_root,novel_truth,output):
    output=Path(output)
    if output.exists():raise FileExistsError('New immutable independent metric scoring root required')
    m,cal,known,novel=_prediction(prediction_root)
    truth=_truth(novel_truth,novel)  # Sole private truth access after both complete policies.
    km=dict(common._masks(known['rx_i'],known['scenario']));nm=dict(common._masks(novel['rx_i'],truth['scenario']))
    allk=np.ones(len(known['ids']),bool);alln=np.ones(len(novel['ids']),bool);methods={}
    for j,name in enumerate(METHODS):
        def metric(k,n,kr=tuple(range(6)),ur=common.UNKNOWN_TX):
            return _metrics(known['y'][k],known['closed'][k],known['scores'][k,j],truth['tx'][n],novel['scores'][n,j],known['baseline_closed'][k],kr,ur)
        groups={key:metric(mask,nm[key]) for key,mask in km.items()};overall=groups['overall']
        kp=all(_known_pass(v) for v in groups.values());cp=cal['methods'][name]['constraints_passed']
        up=bool(overall['coverage']['unknown_complete'] and overall['R_U'] is not None and overall['R_U']>=.90-1e-12)
        goals=dict(known_42_groups_passed=kp,source_known_calibration_passed=cp,source_overall_unknown_macro_90_passed=up,passed=bool(kp and cp and up),
            gate_scope='source_overall_falsification_only',RX_scene_unknown_diagnostic_only=True)
        methods[name]=dict(overall=overall,groups=groups,goals=goals,diagnostic_only=name!=PRIMARY or m['arm']!='M_LOE',
            by_RX={str(r):groups['rx/'+str(r)] for r in common.RECEIVERS},by_scene={s:groups['scene/'+s] for s in common.SCENES},
            by_day={str(d):metric(known['day_i']==d,novel['day_i']==d) for d in (1,2,3)},
            by_TX={str(c):metric(known['y']==c,alln,(c,)) for c in range(6)}|{tx:metric(allk,truth['tx']==tx,ur=(tx,)) for tx in common.UNKNOWN_TX})
    main=methods[PRIMARY];control=methods[METHODS[0]];passed=main['goals']['passed'];eligible=bool(m['arm']=='M_LOE' and passed)
    keys=('A_K','C_K','R_U','H_score','auroc_known_macro','oscr_auc_macro','original_closed_minus_new_open')
    def delta(a,b):return {k:None if a[k] is None or b[k] is None else a[k]-b[k] for k in keys}
    paired=dict(primary=PRIMARY,control=METHODS[0],same_physical_IDs_and_original_F0_head=True,
        thresholds_scope='F0 control exact reused; learned density uses only its own frozen source-known calibration',
        counts=_paired_counts(known,novel,allk,alln),overall=delta(main['overall'],control['overall']),
        groups={key:dict(delta=delta(main['groups'][key],control['groups'][key]),counts=_paired_counts(known,novel,km[key],nm[key])) for key in km},
        by_day={str(d):_paired_counts(known,novel,known['day_i']==d,novel['day_i']==d) for d in (1,2,3)},
        by_known_TX={str(c):_paired_counts(known,novel,known['y']==c,alln) for c in range(6)},by_unknown_TX={tx:_paired_counts(known,novel,allk,truth['tx']==tx) for tx in common.UNKNOWN_TX})
    result=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=m['code_commit'],arm=m['arm'],primary_arm='M_LOE',method_names=list(METHODS),primary_method=PRIMARY,
        methods=methods,primary_summary=main['overall'],paired_primary_vs_control=paired,control_reused_exactly=True,source_gate_passed=passed,
        eligible_for_target_prereg=eligible,eligible_primary=eligible,gate_scope='source_overall_falsification_only',next_step='SOURCE_ELIGIBLE_FOR_SEPARATE_TARGET_PREREGISTRATION' if eligible else 'NO_PROMOTION',
        source_only=True,truth_last=True,target_reads=0,target_U_reads=0,target_Q_reads=0,formal_target_accessed=False,target_DA=False,target_REG=False,
        source_RX_seen=True,source_novel_historically_exposed=True,new_blind_confirmation=False,A_N=None,H_OW=None,final_phase3_complete=False,full_goal_passed=False,full_goal_status='ACTIVE',
        original_F0_parameters=232993,metric_parameters=102688,combined_active_parameters=335681,existing_restorer_parameters=266951,native_updates=0,
        embedding_training_uses_OE=m['arm']=='M_LOE',density_background_uses_OE=True,source_bank_deployed=False,
        costs={k:m[k] for k in ('real_F_forward_rows','synthetic_F_forward_rows','G_forward_rows','synthetic_open_forward_rows','open_projection_rows',
            'open_projection_rows_by_role','source_aggregate_bytes_by_method','threshold_bytes','cpu_fit_seconds','cpu_readout_seconds','cpu_readout_rows')},
        scope='Seen-source-RX historical-novel early falsification only; no target adaptation, discovery, registration or final-goal success')
    output.mkdir(parents=True);_write(output/'results.json',result);_write(output/'complete.json',{k:v for k,v in result.items() if k not in ('methods','paired_primary_vs_control')})
    print(STATUS,m['arm'],result['next_step'],flush=True);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prediction-root',required=True);p.add_argument('--novel-truth',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();score(a.prediction_root,a.novel_truth,a.output)


if __name__=='__main__':main()
