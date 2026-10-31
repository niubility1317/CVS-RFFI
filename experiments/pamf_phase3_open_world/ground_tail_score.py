"""Truth-last group-tail energy scoring with the formal CK95/RU95 goals."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from . import ground_open_score as old
from . import ground_open_core as risk

RUN_ID='20261101-phase3-ground-tail-s2026092701-r01'
SCHEMA='pamf_phase3_ground_tail_scores_v1'
STATUS='GROUND_TAIL_SCORES_COMPLETE'
METHODS=('OLD_SSL_energy','DMEAN_energy','DTAIL_energy')
PRIMARY=METHODS[2]
CONDITIONS=old.CONDITIONS
common,fresh,previous=old.common,old.fresh,old.previous
_read,_write=old._read,old._write
_source_known_truth,_source_truth,_future_truth=old._source_known_truth,old._source_truth,old._future_truth


def _reconstruct(a,tau):
    logits=a['default_logits'];closed=a['closed'];n=len(a['ids'])
    if (tau.shape!=(3,6) or tau.dtype!=np.float64 or not np.isfinite(tau).all()
            or logits.shape!=(n,3,6) or logits.dtype!=np.float32 or not np.isfinite(logits).all()
            or closed.shape!=(n,3) or closed.dtype!=np.int64 or not np.array_equal(closed,logits.argmax(2))
            or a['baseline_logits'].shape!=(n,6) or a['baseline_logits'].dtype!=np.float32 or not np.isfinite(a['baseline_logits']).all()
            or a['baseline_closed'].shape!=(n,) or a['baseline_closed'].dtype!=np.int64 or not np.array_equal(a['baseline_closed'],a['baseline_logits'].argmax(1))):
        raise ValueError('Native energy heads and the separately inherited original F0 baseline required')
    raw=np.column_stack([risk.energy(logits[:,j]) for j in range(3)])
    margins=np.column_stack([risk.apply(raw[:,j],closed[:,j],dict(schema=risk.SCHEMA,conditional_thresholds=tau[j])) for j in range(3)])
    if not np.array_equal(a['raw_scores'],raw) or not np.array_equal(a['scores'],margins):
        raise ValueError('Every energy and own predicted-class frozen margin must reconstruct')


def _exact_control(a,parent):
    shared=('ids','baseline_closed')+tuple(k for k in ('rx_i','day_i') if k in parent)
    if (any(not np.array_equal(a[k],parent[k]) for k in shared)
            or not np.array_equal(a['baseline_logits'],parent['default_logits'][:,0])
            or any(not np.array_equal(a[k][:,0],parent[k][:,2]) for k in ('default_logits','closed','raw_scores','scores'))):
        raise ValueError('Original F0 baseline and all old SSL public predictions must remain exact')


def _known95(row):
    return bool(row['coverage']['known_complete'] and row['C_K'] is not None and row['C_K']>=.95-1e-12
        and row['original_closed_minus_new_open'] is not None and row['original_closed_minus_new_open']<=.02+1e-12)


def _pass95(row):
    return bool(_known95(row) and row['coverage']['complete'] and row['R_U'] is not None and row['R_U']>=.95-1e-12)


def _strict_path(path,condition):
    """Replace the old helpers' diagnostic90 rules with this fixed formal goal."""
    if condition not in CONDITIONS:raise ValueError('Only four frozen DURA conditions allowed')
    expected_scenes={'clean'} if 'clean' in condition else set(common.SCENES)
    scenes=path['by_scene'];complete_scenes=set(scenes)==expected_scenes
    overall=_pass95(path['overall']);scene_pass=complete_scenes and all(_pass95(r) for r in scenes.values())
    groups=path.get('groups',{});source_clean=condition=='source_clean_DURA'
    source_groups_complete=set(groups)==set(risk.GROUP_NAMES)
    group_pass=source_groups_complete and all(_pass95(r) for r in groups.values())
    path=dict(path)
    # The inherited diagnostic rule is retained only under an explicit name.
    if 'goals' in path:path['legacy_diagnostic_goals']=path['goals']
    path['goals']=dict(known_minimum=.95,unknown_minimum=.95,original_F0_loss_max=.02,overall_passed=overall,
        all_main_scenes_passed=bool(scene_pass),required_scene_coverage_complete=bool(complete_scenes and all(r['coverage']['complete'] for r in scenes.values())),
        all_24_groups_passed=bool(group_pass) if source_clean else None,
        required_group_coverage_complete=bool(source_groups_complete and all(r['coverage']['complete'] for r in groups.values())) if source_clean else None,
        passed=bool(overall and (group_pass if source_clean else scene_pass)))
    if groups:path['known_24_groups_passed']=bool(source_groups_complete and all(_known95(r) for r in groups.values()))
    if 'scene_groups' in path:
        sg=path['scene_groups'];path['known_scene_groups_passed']=all(_known95(r) for r in sg.values())
        star='star' in condition
        path['known_12_scene_groups_passed']=bool(set(common.GROUP_NAMES[:12]).issubset(sg) and all(_known95(sg[g]) for g in common.GROUP_NAMES[:12])) if star else None
        path['known_42_groups_passed']=bool(set(sg)==set(common.GROUP_NAMES) and all(_known95(r) for r in sg.values())) if star else None
    path['formal_rejection_goal']=True
    path['scene_groups_diagnostic_only']=False
    path['source_RX_scene_cross_groups_diagnostic_only']=True if condition.startswith('source') else None
    path['source_star_diagnostic_only']=False if condition=='source_star_DURA' else path.get('source_star_diagnostic_only')
    return path


def _decision(primary,cal):
    calibrated=bool(cal['methods'][PRIMARY]['constraints_passed'])
    source=bool(calibrated and primary['source_clean_DURA']['goals']['passed'])
    cross=bool(primary['cross_clean_DURA']['goals']['passed'])
    ground=bool(source and cross)
    frozen=bool(calibrated and all(primary[c]['goals']['passed'] for c in CONDITIONS))
    return source,cross,ground,frozen


def _prediction(root):
    # Imported only on a real artifact gate so metric helpers remain independent
    # of training/model dependencies; no loader or forward runs on the CPU side.
    root=Path(root);m=_read(root/'complete.json')
    if m.get('status')!='GROUND_TAIL_PREDICTIONS_COMPLETE' or m.get('run_id')!=RUN_ID:
        raise ValueError('All completed ground-tail predictions required before truth')
    from . import ground_tail_run as evaluation
    required=dict(schema=evaluation.SCHEMA,status=evaluation.STATUS,run_id=RUN_ID,method_names=list(METHODS),primary_method=PRIMARY,conditions=list(CONDITIONS),
        all_12_points_complete=True,all_four_DURA_conditions_complete=True,all_conditions_through_DURA=True,neural_models_frozen_before_CAL=True,all_thresholds_frozen_before_Q=True,
        source_controls_reused_exactly=True,original_F0_baseline_independent_of_method_columns=True,calibration_known_count=evaluation.CAL_COUNT,
        source_query_count=evaluation.SOURCE_Q_COUNT,cross_query_count=evaluation.TARGET_Q_COUNT,source_test_labels_read=0,source_novel_truth_reads=0,target_truth_reads=0,private_truth_reads=0,
        target_U_reads=0,target_U_GT_reads=0,source_U_GT_reads=0,source_Us_reads=0,prediction_optimizer_updates=0,source_density_fit_updates=0,
        new_threshold_values=12,reused_threshold_values=6,threshold_values=18,threshold_bytes=144,
        real_student_forward_rows=2*evaluation.CAL_COUNT+4*evaluation.SOURCE_Q_COUNT+4*evaluation.TARGET_Q_COUNT,real_F_forward_rows=0,G_forward_rows=2*evaluation.TARGET_Q_COUNT,
        source_only=True,ground_clean_training=True,new_target_DA=False,original_models_unchanged=True,student_heads_jointly_trained=True,
        primary_deployed_neural_parameters=499944,primary_threshold_values=6,formal_target_accessed=False,full_goal_status='ACTIVE',final_phase3_complete=False)
    if any(m.get(k)!=v for k,v in required.items()):raise ValueError('All twelve native/legacy predictions and independent F0 baseline must freeze before truth')
    c=evaluation.validate_config(m['config']);parent,dataset,trained=evaluation._metadata(c)
    if (m['code_commit']!=c['code_commit'] or m['dataset_identity']!=dataset or m['model_identities']!=trained['identities']
            or m['source_ground_reference_identity']!=trained['source_ground_reference_identity']
            or m['training_costs']!={k:v for k,v in trained.items() if isinstance(v,(bool,int,float))}
            or m['inherited_data_costs']!={k:v for k,v in dataset.items() if isinstance(v,(bool,int,float))}
            or m['trained_optimizer_updates']!=trained['total_optimizer_updates']):
        raise ValueError('Exact old data, new fixed-budget training identities and inherited costs required')
    if trained.get('real_F_forward_rows')!=0 or trained.get('G_forward_rows')!=0:
        raise ValueError('Cached same-input source training must report zero new real F0 and G forwards')
    if m['data_capsule_id']!=dataset['capsule_id'] or m['data_split_id']!=dataset['split_id']:
        raise ValueError('The same once-validated ground r03 capsule/split must be reused')
    for key in ('parent_F0_identity','dura_identity','loaded_dura_identity'):
        if m[key]!=dataset[key] or parent[key]!=dataset[key] or trained[key]!=dataset[key]:raise ValueError('Frozen original F0/DURA ancestry differs')
    train=evaluation._train()
    for arm in ('DMEAN','DTAIL'):
        i=m['model_identities'][arm]
        expected=dict(run_id=RUN_ID,code_commit=m['code_commit'],arm=arm,updates=train.STEPS,reader_parameters=232993,trainable_parameters=232945,default_head_parameters=960,
            frozen_Sinc_parameters=48,Sinc_frozen=True,identity_head_updates=train.STEPS,identity_head_jointly_trained=True,identity_head_replaced=False,projection_updates=0,
            initial_native_logits_equal_original_F0=True,original_parent_updates=11100,cumulative_updates=11100+train.STEPS,
            parent_F0_identity=dataset['parent_F0_identity'],dura_identity=dataset['dura_identity'],loaded_dura_identity=dataset['loaded_dura_identity'],
            source_ground_reference_identity=trained['source_ground_reference_identity'],ground_data_identity=trained['ground_data_identity'],source_handles=trained['source_handles'],
            same_input_teacher=True,source_only=True,ground_clean_training=True,source_U_s_used=False,source_Us_reads=0,source_U_s_reads=0,source_U_s_exposures=0,
            target_U_used=False,target_U_reads=0,target_Q_reads=0,source_U_GT_reads=0,input_path='G(clean)',OE_loss_reduction='equal_public_RX_day_domains',tail_fraction=1. if arm=='DMEAN' else .25)
        if any(i.get(k)!=v for k,v in expected.items()) or set(i['source_handles'])!={'L','OE'}:
            raise ValueError('Both independent source-only original-F0 native mean/tail students required')
        if any(i['source_handles'][role]!=dataset['roles'][role] for role in ('L','OE')):
            raise ValueError('Only original G(clean) source L/OE handles may train new students')
    smoke=_read(root/'smoke.json')
    if (any(smoke.get(k)!=v for k,v in dict(status='PASS',actual_ground_students_DURA=True,optimizer_updates=0,real_data_reads=0,synthetic_student_forward_rows=4,G_synthetic_forward_rows=2).items())
            or any(m.get(k)!=smoke.get(k) for k in ('synthetic_student_forward_rows','G_synthetic_forward_rows'))):
        raise ValueError('Actual frozen two-student/DURA no-query smoke and counters required')
    old_cal,old_tau=evaluation._old_calibration(c);saved=old._arrays(root/'thresholds.npz');tau=saved.get('conditional');cal=_read(root/'calibration/complete.json')
    if (set(saved)!={'conditional'} or not isinstance(tau,np.ndarray) or tau.shape!=(3,6) or tau.dtype!=np.float64 or not np.isfinite(tau).all() or not np.array_equal(tau[0],old_tau[2])
            or any(cal.get(k)!=v for k,v in dict(schema=evaluation.SCHEMA,status=evaluation.CALIBRATION_STATUS,run_id=RUN_ID,code_commit=m['code_commit'],
                method_names=list(METHODS),primary_method=PRIMARY,calibration_count=evaluation.CAL_COUNT,calibration_known_count=evaluation.CAL_COUNT,
                calibration_manifest=old_cal['calibration_manifest'],reference_calibration_role=old_cal['calibration_manifest'],known_only=True,unknown_inputs_used=False,test_inputs_used=False,
                target_reads=0,neural_models_frozen_before_CAL=True,group_names=list(risk.GROUP_NAMES),source_controls_reused_exactly=True,new_threshold_values=12,reused_threshold_values=6,threshold_values=18).items())
            or set(cal.get('methods',{}))!=set(METHODS) or cal.get('conditional_thresholds')!=tau.tolist()
            or cal['methods'][METHODS[0]]!=old_cal['methods']['SSL_energy']):
        raise ValueError('Exact six old SSL thresholds/report and twelve CAL-only new thresholds required')
    for j,name in enumerate(METHODS):
        report=cal['methods'][name]
        if (any(report.get(k)!=v for k,v in dict(schema=risk.SCHEMA,conditional_thresholds=tau[j].tolist(),calibration_count=evaluation.CAL_COUNT,
                group_names=list(risk.GROUP_NAMES),known_only=True,unknown_inputs_used=False,test_inputs_used=False,U_labels_used=False).items())
                or set(report.get('groups',{}))!=set(risk.GROUP_NAMES) or type(report.get('constraints_passed')) is not bool):
            raise ValueError('Complete source known24 calibration reports and own-class thresholds required')
    rows={condition:evaluation._rows(root,condition) for condition in CONDITIONS}
    for condition,a in rows.items():
        _reconstruct(a,tau);_exact_control(a,evaluation._legacy(c,condition))
    source=rows[CONDITIONS[0]];target=rows[CONDITIONS[2]]
    if (any(not np.array_equal(source[k],rows[CONDITIONS[1]][k]) for k in ('ids','rx_i','day_i'))
            or not np.array_equal(target['ids'],rows[CONDITIONS[3]]['ids']) or set(source['ids'].tolist())&set(target['ids'].tolist())):
        raise ValueError('Source/future role separation and clean/star same ordered IDs required')
    novel=old._public_novel(c['reference_config'])
    if not common._ids(novel['ids'],dataset['counts']['novel']) or not set(novel['ids'].tolist()).issubset(set(source['ids'].tolist())):
        raise ValueError('Source novel public IDs must align to the complete mixed query')
    risk.masks(novel['rx_i'],novel['day_i'])
    return m,cal,rows,novel


def _legacy_score(config):
    """Read the immutable control's scored aggregates only after current truth."""
    return _read(Path(config['reference_config']['output_root']).parent/'score/results.json')


def score(prediction_root,source_known_truth,source_novel_truth,clean_truth,star_truth,output):
    out=Path(output)
    if out.exists():raise FileExistsError('New immutable independent ground-tail score root required')
    m,cal,rows,public_novel=_prediction(prediction_root)
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
            a=dict(rows[c],closed=rows[c]['closed'][:,j])
            paths[c]=fresh._target(a,truths[c],c.removeprefix('cross_'),j)
        paths={c:_strict_path(path,c) for c,path in paths.items()}
        calibrated=cal['methods'][name]['constraints_passed'];source=bool(calibrated and paths[CONDITIONS[0]]['goals']['passed'])
        cross=bool(paths[CONDITIONS[2]]['goals']['passed']);frozen=bool(calibrated and all(path['goals']['passed'] for path in paths.values()))
        methods[name]=dict(paths=paths,diagnostic_only=name!=PRIMARY,source_CAL_constraints_passed=calibrated,ground_source_success=source,
            cross_clean_success=cross,ground_training_success=bool(source and cross),frozen_rejection_success=frozen)
    legacy=_legacy_score(m['config'])
    reference=m['config']['reference_config']
    if legacy.get('status')!=old.STATUS or legacy.get('run_id')!=reference['run_id'] or legacy.get('code_commit')!=reference['code_commit']:
        raise ValueError('Previously completed ground SSL scores required for the exact control comparison')
    for c in CONDITIONS:
        if methods[METHODS[0]]['paths'][c]['overall']!=legacy['methods']['SSL_energy']['paths'][c]['overall']:
            raise ValueError('Every legacy SSL overall metric must remain exact under the same four truths')
    primary=methods[PRIMARY]['paths'];paired={}
    for j,name in enumerate(METHODS[:2]):
        paired[name]={}
        for c in CONDITIONS:
            pair=old._paired(rows[c],truths[c],2,j,c in CONDITIONS[:2]);pair['delta']=previous._delta(primary[c]['overall'],methods[name]['paths'][c]['overall']);paired[name][c]=pair
    source,cross,ground,frozen=_decision(primary,cal)
    result=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=m['code_commit'],method_names=list(METHODS),primary_method=PRIMARY,conditions=list(CONDITIONS),
        methods=methods,primary_summary={c:primary[c]['overall'] for c in CONDITIONS},paired_primary_vs_controls=paired,
        ground_source_success=source,cross_clean_success=cross,ground_training_success=ground,frozen_rejection_success=frozen,
        scientific_success=frozen,development_gate_passed=frozen,source_star_success=primary['source_star_DURA']['goals']['passed'],cross_star_success=primary['cross_star_DURA']['goals']['passed'],
        next_step='FROZEN_REJECTION_PASSED_NEEDS_UDA_DISCOVERY_REGISTRATION' if frozen else 'NO_PROMOTION',
        truth_last=True,all_four_predictions_before_truth=True,all_12_points_before_truth=True,source_known_truth_reads=1,source_novel_truth_reads=1,target_truth_reads=2,private_truth_reads=4,
        U_truth_reads=0,source_U_GT_reads=0,adaptation_success=None,new_target_DA=False,target_REG=False,registration_success=None,A_N=None,H_KN=None,H_OW=None,
        legacy_SSL_overall_exact=True,original_F0_unchanged=True,classification_unchanged=False,identity_baseline_path='same_DURA_output_original_F0',
        source_only=True,source_Us_reads=0,training_target_U_reads=0,new_source_density_models=0,new_source_threshold_sets=2,new_threshold_values=12,reused_threshold_values=6,threshold_values=18,threshold_bytes=144,
        primary_deployed_neural_parameters=499944,primary_threshold_values=6,source_bank_deployed=False,formal_target_accessed=False,full_goal_status='ACTIVE',full_goal_passed=False,final_phase3_complete=False,
        source_CAL_role='known calibration half of historical source V; heldout TEST labels are not training or calibration inputs',
        global_physical_history='UNKNOWN',new_blind_confirmation=False,source_novel_historically_exposed=True,
        scope='Two source-only native readers: public-domain equal mean or hardest25percent pair-tail OE; no Us/target adaptation. Fixed DTAIL must satisfy CK95/RU95/original-F0loss2pp and full rosters on all four conditions, with source-clean all24 groups and all six star scenes. Frozen rejection stage cannot complete UDA/discovery/registration')
    result.update(training_costs=m['training_costs'],inherited_data_costs=m['inherited_data_costs'],trained_optimizer_updates=m['trained_optimizer_updates'],
        prediction_costs={k:m[k] for k in ('real_student_forward_rows','real_F_forward_rows','G_forward_rows','synthetic_student_forward_rows','G_synthetic_forward_rows')},
        combined_new_real_student_forward_rows=m['training_costs']['real_student_forward_rows']+m['real_student_forward_rows'],
        combined_new_real_F_forward_rows=m['training_costs']['real_F_forward_rows']+m['real_F_forward_rows'],
        combined_new_G_forward_rows=m['training_costs']['G_forward_rows']+m['G_forward_rows'],new_data_exports=0,
        cost_scope='Old G(clean) capsule is reused; its G/F0 derivation costs are inherited, not newly incurred. No Us/target U, no teacher real forward, no new density fitting; new mean/tail training and two known CAL threshold sets are explicit')
    out.mkdir(parents=True);_write(out/'results.json',result);_write(out/'complete.json',{k:v for k,v in result.items() if k not in ('methods','paired_primary_vs_controls')})
    print(STATUS,result['next_step'],flush=True);return result


def main():
    p=argparse.ArgumentParser()
    for flag in ('prediction-root','source-known-truth','source-novel-truth','clean-truth','star-truth','output'):p.add_argument('--'+flag,required=True)
    a=p.parse_args();score(a.prediction_root,a.source_known_truth,a.source_novel_truth,a.clean_truth,a.star_truth,a.output)


if __name__=='__main__':main()
