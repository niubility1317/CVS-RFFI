"""Truth-last centered-evidence scoring with formal CK95/RU95 protection."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from . import ground_tail_score as tail

RUN_ID='20261101-phase3-ground-centered-s2026092701-r01'
SCHEMA='pamf_phase3_ground_centered_scores_v1'
STATUS='GROUND_CENTERED_SCORES_COMPLETE'
METHODS=('DMEAN_energy','DMEAN_centered','CPAIR_energy','CPAIR_centered')
PRIMARY=METHODS[3]
CONDITIONS=tail.CONDITIONS
old,risk=tail.old,tail.risk
common,fresh,previous=tail.common,tail.fresh,tail.previous
_read,_write=tail._read,tail._write
_source_known_truth,_source_truth,_future_truth=tail._source_known_truth,tail._source_truth,tail._future_truth
_strict_path=tail._strict_path


def _raw(logits):
    from .ground_centered_core import centered_energy
    energy=np.column_stack([risk.energy(logits[:,j]) for j in range(4)])
    for j in (1,3):energy[:,j]=centered_energy(logits[:,j])
    return energy


def _reconstruct(a,tau):
    logits=a['default_logits'];closed=a['closed'];n=len(a['ids'])
    if (tau.shape!=(4,6) or tau.dtype!=np.float64 or not np.isfinite(tau).all()
            or logits.shape!=(n,4,6) or logits.dtype!=np.float32 or not np.isfinite(logits).all()
            or closed.shape!=(n,4) or closed.dtype!=np.int64 or not np.array_equal(closed,logits.argmax(2))
            or not np.array_equal(logits[:,0],logits[:,1]) or not np.array_equal(logits[:,2],logits[:,3])
            or a['baseline_logits'].shape!=(n,6) or a['baseline_logits'].dtype!=np.float32 or not np.isfinite(a['baseline_logits']).all()
            or a['baseline_closed'].shape!=(n,) or a['baseline_closed'].dtype!=np.int64 or not np.array_equal(a['baseline_closed'],a['baseline_logits'].argmax(1))):
        raise ValueError('Paired readouts require the same own native logits and a separate original F0 baseline')
    raw=_raw(logits)
    margins=np.column_stack([risk.apply(raw[:,j],closed[:,j],dict(schema=risk.SCHEMA,conditional_thresholds=tau[j])) for j in range(4)])
    if not np.array_equal(a['raw_scores'],raw) or not np.array_equal(a['scores'],margins):
        raise ValueError('Four energy/centered scores and each own-class frozen margin must reconstruct')


def _exact_control(a,parent):
    shared=('ids','baseline_logits','baseline_closed')+tuple(k for k in ('rx_i','day_i') if k in parent)
    if (any(not np.array_equal(a[k],parent[k]) for k in shared)
            or any(not np.array_equal(a[k][:,0],parent[k][:,1]) for k in ('default_logits','closed','raw_scores','scores'))
            or not np.array_equal(a['default_logits'][:,1],parent['default_logits'][:,1])):
        raise ValueError('Old DMEAN energy and the independent canonical F0 baseline must remain exact')


def _decision(primary,cal):
    calibrated=bool(cal['methods'][PRIMARY]['constraints_passed'])
    source=bool(calibrated and primary['source_clean_DURA']['goals']['passed'])
    cross=bool(primary['cross_clean_DURA']['goals']['passed'])
    return source,cross,bool(source and cross),bool(calibrated and all(primary[c]['goals']['passed'] for c in CONDITIONS))


def _prediction(root):
    root=Path(root);m=_read(root/'complete.json')
    if m.get('status')!='GROUND_CENTERED_PREDICTIONS_COMPLETE' or m.get('run_id')!=RUN_ID:
        raise ValueError('All completed sixteen centered predictions required before truth')
    return _prediction_artifacts(root,m)


def _prediction_artifacts(root,m):
    # Metadata and saved predictions only. No training loader, NN, IQ or CAL fit.
    from . import ground_centered_run as evaluation
    required=dict(schema=evaluation.SCHEMA,status=evaluation.STATUS,run_id=RUN_ID,method_names=list(METHODS),primary_method=PRIMARY,conditions=list(CONDITIONS),
        all_16_points_complete=True,all_four_DURA_conditions_complete=True,all_conditions_through_DURA=True,neural_models_frozen_before_CAL=True,all_thresholds_frozen_before_Q=True,
        source_controls_reused_exactly=True,original_F0_baseline_independent_of_method_columns=True,paired_readouts_share_logits=True,original_models_unchanged=True,
        calibration_known_count=evaluation.CAL_COUNT,source_query_count=evaluation.SOURCE_Q_COUNT,cross_query_count=evaluation.TARGET_Q_COUNT,
        source_test_labels_read=0,source_novel_truth_reads=0,target_truth_reads=0,private_truth_reads=0,target_U_reads=0,target_U_GT_reads=0,source_U_GT_reads=0,source_Us_reads=0,
        prediction_optimizer_updates=0,source_density_fit_updates=0,new_threshold_values=18,reused_threshold_values=6,threshold_values=24,threshold_bytes=192,
        real_student_forward_rows=evaluation.CAL_COUNT+2*evaluation.SOURCE_Q_COUNT+2*evaluation.TARGET_Q_COUNT,real_legacy_forward_rows=evaluation.CAL_COUNT,
        real_F_forward_rows=0,G_forward_rows=2*evaluation.TARGET_Q_COUNT,synthetic_r03_forward_rows=0,synthetic_PCD_forward_rows=0,
        source_only=True,ground_clean_training=True,new_target_DA=False,student_heads_jointly_trained=True,
        primary_deployed_neural_parameters=499944,primary_threshold_values=6,formal_target_accessed=False,full_goal_status='ACTIVE',final_phase3_complete=False)
    if any(m.get(k)!=v for k,v in required.items()):raise ValueError('All sixteen native/legacy predictions must freeze before any private truth')
    c=evaluation.validate_config(m['config']);parent,dataset,trained=evaluation._metadata(c)
    expected_identities=dict(DMEAN=parent['model_identities']['DMEAN'],CPAIR=trained['identities']['CPAIR'])
    if (m['code_commit']!=c['code_commit'] or m['reference_prediction_identity']!=parent or m['dataset_identity']!=dataset
            or m['model_identities']!=expected_identities or m['source_tail_reference_identity']!=trained['source_tail_reference_identity']
            or m['source_ground_reference_identity']!=trained['source_ground_reference_identity']
            or m['training_costs']!={k:v for k,v in trained.items() if isinstance(v,(bool,int,float))}
            or m['inherited_data_costs']!={k:v for k,v in dataset.items() if isinstance(v,(bool,int,float))}
            or m['trained_optimizer_updates']!=trained['total_optimizer_updates']):
        raise ValueError('Exact source data, independent CPAIR, frozen DMEAN identities and costs required')
    if trained.get('real_F_forward_rows')!=0 or trained.get('G_forward_rows')!=0:
        raise ValueError('Same-input cached source training must report zero new real F0 and G forwards')
    if m['data_capsule_id']!=dataset['capsule_id'] or m['data_split_id']!=dataset['split_id']:
        raise ValueError('The same once-validated source-clean capsule/split must be reused')
    for key in ('parent_F0_identity','dura_identity','loaded_dura_identity'):
        if m[key]!=dataset[key] or parent[key]!=dataset[key] or trained[key]!=dataset[key]:raise ValueError('Original F0/DURA ancestry differs')
    smoke=_read(root/'smoke.json')
    smoke_required=dict(status='PASS',actual_original_F0_legacy_student_DURA=True,optimizer_updates=0,real_data_reads=0,
        synthetic_student_forward_rows=2,synthetic_legacy_forward_rows=2,synthetic_F_forward_rows=2,G_synthetic_forward_rows=2,
        synthetic_r03_forward_rows=0,synthetic_PCD_forward_rows=0)
    if (any(smoke.get(k)!=v for k,v in smoke_required.items())
            or any(m.get(k)!=smoke.get(k) for k in ('synthetic_student_forward_rows','synthetic_legacy_forward_rows','synthetic_F_forward_rows','G_synthetic_forward_rows'))):
        raise ValueError('Actual frozen original F0/DMEAN/CPAIR/DURA no-query smoke and counters required')
    old_cal,old_tau=evaluation._old_calibration(c);saved=old._arrays(root/'thresholds.npz');tau=saved.get('conditional');cal=_read(root/'calibration/complete.json')
    if (set(saved)!={'conditional'} or not isinstance(tau,np.ndarray) or tau.shape!=(4,6) or tau.dtype!=np.float64 or not np.isfinite(tau).all()
            or not np.array_equal(tau[0],old_tau[1])
            or any(cal.get(k)!=v for k,v in dict(schema=evaluation.SCHEMA,status=evaluation.CALIBRATION_STATUS,run_id=RUN_ID,code_commit=m['code_commit'],
                method_names=list(METHODS),primary_method=PRIMARY,calibration_count=evaluation.CAL_COUNT,calibration_known_count=evaluation.CAL_COUNT,
                calibration_manifest=old_cal['calibration_manifest'],reference_calibration_role=old_cal['calibration_manifest'],known_only=True,unknown_inputs_used=False,test_inputs_used=False,
                target_reads=0,neural_models_frozen_before_CAL=True,group_names=list(risk.GROUP_NAMES),source_controls_reused_exactly=True,
                new_threshold_values=18,reused_threshold_values=6,threshold_values=24,threshold_bytes=192).items())
            or set(cal.get('methods',{}))!=set(METHODS) or cal.get('conditional_thresholds')!=tau.tolist()
            or cal['methods'][METHODS[0]]!=old_cal['methods']['DMEAN_energy']):
        raise ValueError('Six exact old DMEAN thresholds/report and eighteen CAL-only new thresholds required')
    for j,name in enumerate(METHODS):
        report=cal['methods'][name]
        if (any(report.get(k)!=v for k,v in dict(schema=risk.SCHEMA,conditional_thresholds=tau[j].tolist(),calibration_count=evaluation.CAL_COUNT,
                group_names=list(risk.GROUP_NAMES),known_only=True,unknown_inputs_used=False,test_inputs_used=False,U_labels_used=False).items())
                or set(report.get('groups',{}))!=set(risk.GROUP_NAMES) or type(report.get('constraints_passed')) is not bool):
            raise ValueError('Complete known24 calibration reports and own-class frozen thresholds required')
    rows={condition:evaluation._rows(root,condition) for condition in CONDITIONS}
    for condition,a in rows.items():_reconstruct(a,tau);_exact_control(a,evaluation._legacy(c,condition))
    source,target=rows[CONDITIONS[0]],rows[CONDITIONS[2]]
    if (any(not np.array_equal(source[k],rows[CONDITIONS[1]][k]) for k in ('ids','rx_i','day_i'))
            or not np.array_equal(target['ids'],rows[CONDITIONS[3]]['ids']) or set(source['ids'].tolist())&set(target['ids'].tolist())):
        raise ValueError('Source/future role separation and clean/star ordered physical IDs required')
    novel=old._public_novel(c['reference_config']['reference_config'])
    if not common._ids(novel['ids'],dataset['counts']['novel']) or not set(novel['ids'].tolist()).issubset(set(source['ids'].tolist())):
        raise ValueError('Source novel public IDs must align to the complete role-blind query')
    risk.masks(novel['rx_i'],novel['day_i'])
    return m,cal,rows,novel


def _legacy_score(config):
    """The control's scored aggregates are consumed only after four truth joins."""
    return _read(Path(config['reference_config']['output_root']).parent/'score/results.json')


def _score_frozen(m,cal,rows,public_novel,source_known_truth,source_novel_truth,clean_truth,star_truth,out):
    kt=_source_known_truth(source_known_truth);nt=_source_truth(source_novel_truth,public_novel)
    truths={c:old._join_source(rows[c],kt,nt,c=='source_star_DURA') for c in CONDITIONS[:2]}
    for condition,path in zip(CONDITIONS[2:],(clean_truth,star_truth)):
        truths[condition]=_future_truth(path,rows[condition]['ids'])
        if not np.isin(truths[condition]['scenario'],('clean',) if 'clean' in condition else common.SCENES).all():
            raise ValueError('Cross clean/star future scorer scenarios differ')
    if any(not np.array_equal(truths[CONDITIONS[2]][key],truths[CONDITIONS[3]][key]) for key in ('ids','y','tx','receiver','day')):
        raise ValueError('Future clean/star truths must share physical records and full true rosters')
    methods={}
    for j,name in enumerate(METHODS):
        paths={c:old._source(rows[c],truths[c],c,j) for c in CONDITIONS[:2]}
        for c in CONDITIONS[2:]:
            a=dict(rows[c],closed=rows[c]['closed'][:,j]);paths[c]=fresh._target(a,truths[c],c.removeprefix('cross_'),j)
        paths={c:_strict_path(path,c) for c,path in paths.items()}
        calibrated=cal['methods'][name]['constraints_passed'];source=bool(calibrated and paths[CONDITIONS[0]]['goals']['passed'])
        cross=bool(paths[CONDITIONS[2]]['goals']['passed']);frozen=bool(calibrated and all(path['goals']['passed'] for path in paths.values()))
        methods[name]=dict(paths=paths,diagnostic_only=name!=PRIMARY,source_CAL_constraints_passed=calibrated,ground_source_success=source,
            cross_clean_success=cross,ground_stage_success=bool(source and cross),frozen_rejection_success=frozen)
    legacy=_legacy_score(m['config']);reference=m['config']['reference_config']
    if legacy.get('status')!=tail.STATUS or legacy.get('run_id')!=reference['run_id'] or legacy.get('code_commit')!=reference['code_commit']:
        raise ValueError('Previously completed tail DMEAN scores required for the exact control comparison')
    for c in CONDITIONS:
        if methods[METHODS[0]]['paths'][c]['overall']!=legacy['methods']['DMEAN_energy']['paths'][c]['overall']:
            raise ValueError('Every legacy DMEAN overall must remain exact under the same four truths')
    primary=methods[PRIMARY]['paths'];paired={}
    for j,name in enumerate(METHODS[:3]):
        paired[name]={}
        for c in CONDITIONS:
            p=old._paired(rows[c],truths[c],3,j,c in CONDITIONS[:2]);p['delta']=previous._delta(primary[c]['overall'],methods[name]['paths'][c]['overall']);paired[name][c]=p
    source,cross,ground,frozen=_decision(primary,cal)
    result=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=m['code_commit'],method_names=list(METHODS),primary_method=PRIMARY,conditions=list(CONDITIONS),
        methods=methods,primary_summary={c:primary[c]['overall'] for c in CONDITIONS},paired_primary_vs_controls=paired,
        ground_source_success=source,cross_clean_success=cross,ground_stage_success=ground,frozen_rejection_success=frozen,scientific_success=frozen,development_gate_passed=frozen,
        source_star_success=primary['source_star_DURA']['goals']['passed'],cross_star_success=primary['cross_star_DURA']['goals']['passed'],
        next_step='FROZEN_REJECTION_PASSED_NEEDS_UDA_DISCOVERY_REGISTRATION' if frozen else 'NO_PROMOTION',
        truth_last=True,all_four_predictions_before_truth=True,all_16_points_before_truth=True,source_known_truth_reads=1,source_novel_truth_reads=1,target_truth_reads=2,private_truth_reads=4,
        U_truth_reads=0,source_U_GT_reads=0,adaptation_success=None,new_target_DA=False,target_REG=False,registration_success=None,A_N=None,H_KN=None,H_OW=None,
        legacy_energy_overall_exact=True,original_F0_unchanged=True,classification_unchanged=False,identity_baseline_path='same_DURA_output_original_F0',
        source_only=True,source_Us_reads=0,training_target_U_reads=0,new_source_density_models=0,new_source_threshold_sets=3,new_threshold_values=18,reused_threshold_values=6,threshold_values=24,threshold_bytes=192,
        primary_deployed_neural_parameters=499944,primary_threshold_values=6,source_bank_deployed=False,formal_target_accessed=False,full_goal_status='ACTIVE',full_goal_passed=False,final_phase3_complete=False,
        source_CAL_role='known half of historical source V; disjoint TEST labels are not training or calibration inputs',global_physical_history='UNKNOWN',new_blind_confirmation=False,source_novel_historically_exposed=True,
        scope='Matched source-only DMEAN and CPAIR native readers with energy and centered-evidence readouts. Fixed CPAIR_centered requires CK95/RU95/original-F0loss2pp and full rosters on all four conditions, source-clean all24 groups and six star scenes. No Us/target adaptation or anonymous registration')
    result.update(training_costs=m['training_costs'],inherited_data_costs=m['inherited_data_costs'],trained_optimizer_updates=m['trained_optimizer_updates'],
        prediction_costs={k:m[k] for k in ('real_student_forward_rows','real_legacy_forward_rows','real_F_forward_rows','G_forward_rows','synthetic_student_forward_rows','synthetic_legacy_forward_rows','G_synthetic_forward_rows')},
        combined_new_real_student_forward_rows=m['training_costs']['real_student_forward_rows']+m['real_student_forward_rows'],
        combined_new_real_F_forward_rows=m['training_costs']['real_F_forward_rows']+m['real_F_forward_rows'],combined_new_G_forward_rows=m['training_costs']['G_forward_rows']+m['G_forward_rows'],new_data_exports=0,
        cost_scope='One CPAIR student is newly trained; legacy DMEAN only adds CAL forwards and retains its source training ancestry. Existing ground data costs are inherited. Three new CAL threshold sets, no new density, Us or target U learning')
    out.mkdir(parents=True);_write(out/'results.json',result);_write(out/'complete.json',{k:v for k,v in result.items() if k not in ('methods','paired_primary_vs_controls')})
    print(STATUS,result['next_step'],flush=True);return result


def score(prediction_root,source_known_truth,source_novel_truth,clean_truth,star_truth,output):
    out=Path(output)
    if out.exists():raise FileExistsError('New immutable centered-evidence score root required')
    m,cal,rows,public_novel=_prediction(prediction_root)
    return _score_frozen(m,cal,rows,public_novel,source_known_truth,source_novel_truth,clean_truth,star_truth,out)


def main():
    p=argparse.ArgumentParser()
    for flag in ('prediction-root','source-known-truth','source-novel-truth','clean-truth','star-truth','output'):p.add_argument('--'+flag,required=True)
    a=p.parse_args();score(a.prediction_root,a.source_known_truth,a.source_novel_truth,a.clean_truth,a.star_truth,a.output)


if __name__=='__main__':main()
