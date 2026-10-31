"""Truth-last source-only frozen-F0 ViM and nuisance-complement readouts."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from . import ground_open_score as old
from . import ground_tail_score as goals
from . import ground_open_core as risk

RUN_ID='20261101-phase3-ground-geometry-s2026092701-r01'
SCHEMA='pamf_phase3_ground_geometry_scores_v1'
STATUS='GROUND_GEOMETRY_SCORES_COMPLETE'
METHODS=('F0_energy','OLD_SSL_energy','F0_vim','F0_domain_vim')
PRIMARY=METHODS[3]
CONDITIONS=old.CONDITIONS
common,fresh,previous=old.common,old.fresh,old.previous
_read,_write=old._read,old._write
_source_known_truth,_source_truth,_future_truth=old._source_known_truth,old._source_truth,old._future_truth
_strict_path,_paired=goals._strict_path,old._paired


def _geometry_components(standard,domain,coordinates,logits):
    from . import ground_geometry_core as core
    for state,mode in ((standard,'standard'),(domain,'domain_orthogonal')):
        core.validate_state(state)
        if state['mode'].item()!=mode:raise ValueError('Matching standard/domain-orthogonal source geometry modes required')
    return [core.raw_scores(state,coordinates,logits) for state in (standard,domain)]


def _geometry_raw(standard,domain,coordinates,logits):
    # ViM is column1; residual-only column0 is not either preregistered readout.
    return np.column_stack([v[:,1] for v in _geometry_components(standard,domain,coordinates,logits)])


def _reconstruct(a,tau,states,head):
    values=_geometry_components(states['standard'],states['domain_orthogonal'],a['classifier_coordinates'],a['fresh_F0_logits'])
    raw=np.column_stack((risk.energy(a['default_logits'][:,0]),risk.energy(a['default_logits'][:,1]),values[0][:,1],values[1][:,1]))
    residual=np.column_stack([-v[:,0] for v in values])
    margin=np.column_stack([a['raw_scores'][:,j]-tau[j,a['closed'][:,j]] for j in range(4)])
    # The first two columns are exact legacy controls. Matrix multiplication in
    # FP64 can differ by CPU batch shape; saved margins retain exact routing.
    if (not np.array_equal(a['raw_scores'][:,:2],raw[:,:2]) or not np.allclose(a['raw_scores'][:,2:],raw[:,2:],rtol=1e-12,atol=1e-9)
            or not np.allclose(a['residual_norms'],residual,rtol=1e-12,atol=1e-9) or not np.array_equal(a['scores'],margin)):
        raise ValueError('Both frozen source geometries, residuals and all own-class margins must reconstruct')
    oracle=(a['classifier_coordinates']@head['normalized_weight'].T)*head['scale']
    if not np.allclose(a['fresh_F0_logits'],oracle,atol=1e-4,rtol=1e-5):
        raise ValueError('Saved FP32 classifier coordinates must reconstruct the same frozen default head')


def _exact_controls(a,parent):
    shared=('ids','baseline_closed')+tuple(k for k in ('rx_i','day_i') if k in parent)
    if (any(not np.array_equal(a[k],parent[k]) for k in shared) or not np.array_equal(a['baseline_logits'],parent['default_logits'][:,0])
            or any(not np.array_equal(a[k][:,:2],parent[k][:,[0,2]]) for k in ('default_logits','closed','raw_scores','scores'))):
        raise ValueError('Two legacy energy columns and original F0 same-DURA baseline must remain exact')


def _decision(primary,cal):
    calibrated=bool(cal['methods'][PRIMARY]['constraints_passed'])
    source=bool(calibrated and primary['source_clean_DURA']['goals']['passed'])
    cross=bool(primary['cross_clean_DURA']['goals']['passed'])
    return source,cross,bool(source and cross),bool(calibrated and all(primary[c]['goals']['passed'] for c in CONDITIONS))


def _prediction(root):
    root=Path(root);m=_read(root/'complete.json')
    if (m.get('status')!='GROUND_GEOMETRY_PREDICTIONS_COMPLETE' or m.get('run_id')!=RUN_ID or m.get('all_16_points_complete') is not True):
        raise ValueError('All sixteen frozen geometry/legacy prediction points must precede truth')
    from . import ground_geometry_run as evaluation
    from . import ground_geometry_core as core
    required=dict(schema=evaluation.SCHEMA,status=evaluation.STATUS,run_id=RUN_ID,method_names=list(METHODS),primary_method=PRIMARY,conditions=list(CONDITIONS),
        all_16_points_complete=True,all_four_DURA_conditions_complete=True,all_conditions_through_DURA=True,neural_models_frozen=True,source_subspaces_frozen_before_CAL=True,all_thresholds_frozen_before_Q=True,
        source_controls_reused_exactly=True,original_F0_baseline_independent_of_fresh_replay=True,source_subspace_fit_count=2,nuisance_SVD_fit_count=1,source_subspace_fit_role='Us',source_subspace_fit_rows=evaluation.US_COUNT,
        source_Us_reads=evaluation.US_COUNT,source_Us_ID_reads=evaluation.US_COUNT,source_U_GT_reads=0,source_L_reads=0,source_OE_reads=0,source_test_labels_read=0,source_novel_truth_reads=0,target_truth_reads=0,private_truth_reads=0,
        target_U_reads=0,U_ID_metadata_reads=0,target_U_GT_reads=0,prediction_optimizer_updates=0,optimizer_updates=0,source_density_fit_updates=0,
        calibration_known_count=evaluation.CAL_COUNT,source_query_count=evaluation.SOURCE_Q_COUNT,cross_query_count=evaluation.TARGET_Q_COUNT,new_threshold_values=12,reused_threshold_values=12,threshold_values=24,threshold_bytes=192,
        real_F_forward_rows=evaluation.US_COUNT+evaluation.CAL_COUNT+2*evaluation.SOURCE_Q_COUNT+2*evaluation.TARGET_Q_COUNT,real_SSL_forward_rows=0,real_student_forward_rows=0,G_forward_rows=2*evaluation.TARGET_Q_COUNT,
        original_models_unchanged=True,original_default_head_unchanged=True,exact_native_classifier_coordinates=True,source_only=True,ground_clean_source_fit=True,new_target_DA=False,
        primary_deployed_neural_parameters=499944,primary_threshold_values=6,source_rows_deployed=False,formal_target_accessed=False,full_goal_status='ACTIVE',final_phase3_complete=False)
    if any(m.get(k)!=v for k,v in required.items()):raise ValueError('All sixteen source-frozen geometry and legacy predictions required before truth')
    c=evaluation.validate_config(m['config']);parent,dataset=evaluation._metadata(c)
    if (m['code_commit']!=c['code_commit'] or m['dataset_identity']!=dataset or m['data_capsule_id']!=dataset['capsule_id'] or m['data_split_id']!=dataset['split_id']
            or m['inherited_data_costs']!={k:v for k,v in dataset.items() if isinstance(v,(bool,int,float))}):
        raise ValueError('Exact old ground capsule, source roles and inherited data costs required')
    if any(m[k]!=dataset[k] or parent[k]!=dataset[k] for k in ('parent_F0_identity','dura_identity','loaded_dura_identity')):
        raise ValueError('The unchanged original F0 and same-DURA ancestry must match')
    fit=_read(root/'models/complete.json');states=evaluation._states(root)
    expected=dict(schema=evaluation.SCHEMA,status=evaluation.MODELS_STATUS,run_id=RUN_ID,code_commit=m['code_commit'],modes=list(evaluation.MODES),source_count=evaluation.US_COUNT,
        source_role_manifest=dataset['roles']['Us'],source_TX_labels_used=False,source_L_reads=0,source_OE_reads=0,target_U_reads=0,target_Q_reads=0,optimizer_updates=0,source_only=True)
    if any(fit.get(k)!=v for k,v in expected.items()) or fit.get('diagnostics')!=m['source_fit_diagnostics']:
        raise ValueError('Two source-Us-only frozen subspaces and their exact source metadata required')
    for mode in evaluation.MODES:
        state=states[mode];core.validate_state(state);marker=_read(root/'models'/mode/'complete.json');diag=fit['diagnostics'][mode];r=state['nuisance_basis'].shape[1]
        if (state['mode'].item()!=mode or any(marker.get(k)!=v for k,v in dict(schema=evaluation.SCHEMA,status=evaluation.MODELS_STATUS,run_id=RUN_ID,code_commit=m['code_commit'],mode=mode,
                source_role='Us',source_count=evaluation.US_COUNT,source_role_manifest=dataset['roles']['Us'],parent_F0_identity=dataset['parent_F0_identity'],dura_identity=dataset['dura_identity'],
                diagnostic=diag,optimizer_updates=0,source_TX_labels_used=False,CAL_inputs_used=False,OE_inputs_used=False,target_U_reads=0,target_Q_reads=0,source_rows_deployed=False).items())
                or any(diag.get(k)!=v for k,v in dict(schema=core.SCHEMA,mode=mode,source_role='U_s',source_count=evaluation.US_COUNT,source_Us_count=evaluation.US_COUNT,domain_count=15,
                    feature_dimension=160,nuisance_rank=r,effective_dimension=160-r,principal_dimension=80-r,residual_dimension=80,source_TX_labels_used=False,OE_inputs_used=False,
                    CAL_inputs_used=False,V_inputs_used=False,target_U_reads=0,target_Q_reads=0,neural_updates=0,source_only=True,source_rows_deployed=False,coordinates_renormalized=False).items())
                or len(diag.get('domain_counts',[]))!=15 or any(type(v) is not int or v<=0 for v in diag['domain_counts']) or sum(diag['domain_counts'])!=evaluation.US_COUNT
                or diag.get('alpha')!=float(state['alpha'])):
            raise ValueError('Matching immutable source aggregate mode, nuisance rank and alpha required; no CPU refit')
    smoke=_read(root/'smoke.json')
    if (any(smoke.get(k)!=v for k,v in dict(status='PASS',actual_original_F0_DURA=True,exact_native_classifier_coordinates=True,synthetic_F_forward_rows=2,G_synthetic_forward_rows=2,optimizer_updates=0,real_data_reads=0).items())
            or m.get('synthetic_F_forward_rows')!=smoke['synthetic_F_forward_rows'] or m.get('G_synthetic_forward_rows')!=smoke['G_synthetic_forward_rows']):
        raise ValueError('Actual no-query F0/default-head-coordinate/DURA smoke required')
    head=old._arrays(root/'classifier_head.npz')
    if (set(head)!={'normalized_weight','scale'} or head['normalized_weight'].shape!=(6,160) or head['normalized_weight'].dtype!=np.float32 or not np.isfinite(head['normalized_weight']).all()
            or head['scale'].shape!=() or head['scale'].dtype!=np.float32 or head['scale'].item()!=30.):
        raise ValueError('Exact frozen native normalized default-head audit state required')
    cal=_read(root/'calibration/complete.json');saved=old._arrays(root/'thresholds.npz');tau=saved.get('conditional');oldcal,oldtau=evaluation._old_calibration(c)
    if (set(saved)!={'conditional'} or not isinstance(tau,np.ndarray) or tau.dtype!=np.float64 or tau.shape!=(4,6) or not np.isfinite(tau).all() or not np.array_equal(tau[:2],oldtau[[0,2]])
            or any(cal.get(k)!=v for k,v in dict(schema=evaluation.SCHEMA,status=evaluation.CALIBRATION_STATUS,run_id=RUN_ID,code_commit=m['code_commit'],method_names=list(METHODS),primary_method=PRIMARY,
                calibration_count=evaluation.CAL_COUNT,calibration_manifest=oldcal['calibration_manifest'],known_only=True,unknown_inputs_used=False,test_inputs_used=False,target_reads=0,
                neural_models_frozen=True,source_subspaces_frozen_before_CAL=True,group_names=list(risk.GROUP_NAMES),source_controls_reused_exactly=True,new_threshold_values=12,reused_threshold_values=12,threshold_values=24).items())
            or set(cal.get('methods',{}))!=set(METHODS) or cal.get('conditional_thresholds')!=tau.tolist()
            or any(cal['methods'][METHODS[j]]!=oldcal['methods'][name] for j,name in ((0,'F0_energy'),(1,'SSL_energy')))):
        raise ValueError('Twelve exact inherited and twelve new CAL-only class thresholds required')
    for j,name in enumerate(METHODS):
        report=cal['methods'][name]
        if (any(report.get(k)!=v for k,v in dict(schema=risk.SCHEMA,conditional_thresholds=tau[j].tolist(),calibration_count=evaluation.CAL_COUNT,group_names=list(risk.GROUP_NAMES),
                known_only=True,unknown_inputs_used=False,test_inputs_used=False,U_labels_used=False).items()) or set(report.get('groups',{}))!=set(risk.GROUP_NAMES) or type(report.get('constraints_passed')) is not bool):
            raise ValueError('Complete known24 source calibration reports and matching own-class thresholds required')
    rows={condition:evaluation._rows(root,condition) for condition in CONDITIONS};replay=m['replay_diagnostics']
    if set(replay)!={'CAL',*CONDITIONS}:raise ValueError('CAL and all four actual F0 replay summaries required')
    for condition,a in rows.items():
        _exact_controls(a,evaluation._legacy(c,condition));_reconstruct(a,tau,states,head)
        if replay[condition]!=evaluation._replay(a['fresh_F0_logits'],a['baseline_logits']):raise ValueError('Actual fresh/cached F0 replay differences must be reported exactly')
    if (replay['CAL'].get('count')!=evaluation.CAL_COUNT or any(type(d.get('closed_mismatch_count')) is not int or not 0<=d['closed_mismatch_count']<=d['count']
            or not np.isfinite(d['maximum_absolute_logit_difference']) or d['maximum_absolute_logit_difference']<0 for d in replay.values())
            or m['classification_unchanged_from_cached_F0']!=all(d['closed_mismatch_count']==0 for d in replay.values())):
        raise ValueError('Replay metadata must distinguish unchanged weights from observed classification replay differences')
    source=rows[CONDITIONS[0]];target=rows[CONDITIONS[2]]
    if any(not np.array_equal(source[k],rows[CONDITIONS[1]][k]) for k in ('ids','rx_i','day_i')) or not np.array_equal(target['ids'],rows[CONDITIONS[3]]['ids']) or set(source['ids'].tolist())&set(target['ids'].tolist()):
        raise ValueError('Same ordered clean/star records and disjoint source/future physical roles required')
    novel=old._public_novel(c['reference_config'])
    if not common._ids(novel['ids'],dataset['counts']['novel']) or not set(novel['ids'].tolist()).issubset(set(source['ids'].tolist())):
        raise ValueError('Original source novel public metadata must align to the complete mixed source query')
    risk.masks(novel['rx_i'],novel['day_i']);return m,cal,rows,novel


def score(prediction_root,source_known_truth,source_novel_truth,clean_truth,star_truth,output):
    out=Path(output)
    if out.exists():raise FileExistsError('New immutable independent ground-geometry score root required')
    m,cal,rows,public_novel=_prediction(prediction_root)
    kt=_source_known_truth(source_known_truth);nt=_source_truth(source_novel_truth,public_novel)
    truths={c:old._join_source(rows[c],kt,nt,c=='source_star_DURA') for c in CONDITIONS[:2]}
    for c,path in zip(CONDITIONS[2:],(clean_truth,star_truth)):
        truths[c]=_future_truth(path,rows[c]['ids'])
        if not np.isin(truths[c]['scenario'],('clean',) if 'clean' in c else common.SCENES).all():raise ValueError('Future clean/star scorer scenarios differ')
    if any(not np.array_equal(truths[CONDITIONS[2]][key],truths[CONDITIONS[3]][key]) for key in ('ids','y','tx','receiver','day')):
        raise ValueError('Future clean/star truth physical records and complete TX rosters must align')
    methods={}
    for j,name in enumerate(METHODS):
        paths={c:old._source(rows[c],truths[c],c,j) for c in CONDITIONS[:2]}
        for c in CONDITIONS[2:]:paths[c]=fresh._target(dict(rows[c],closed=rows[c]['closed'][:,j]),truths[c],c.removeprefix('cross_'),j)
        paths={c:_strict_path(path,c) for c,path in paths.items()};source,cross,ground,frozen=_decision(paths,dict(methods={PRIMARY:cal['methods'][name]}))
        methods[name]=dict(paths=paths,diagnostic_only=name!=PRIMARY,classification_model='legacy_SSL' if j==1 else 'original_F0',source_CAL_constraints_passed=cal['methods'][name]['constraints_passed'],
            ground_source_success=source,cross_clean_success=cross,ground_stage_success=ground,frozen_rejection_success=frozen)
    legacy=_legacy_score(m['config']);reference=m['config']['reference_config']
    if legacy.get('status')!=old.STATUS or legacy.get('run_id')!=reference['run_id'] or legacy.get('code_commit')!=reference['code_commit']:
        raise ValueError('Completed immutable old ground score metadata required after current truth')
    for j,oldname in ((0,'F0_energy'),(1,'SSL_energy')):
        for c in CONDITIONS:
            if methods[METHODS[j]]['paths'][c]['overall']!=legacy['methods'][oldname]['paths'][c]['overall']:
                raise ValueError('All eight legacy F0/SSL overall metrics must stay exact on the same four truths')
    primary=methods[PRIMARY]['paths'];paired={}
    for j,name in enumerate(METHODS[:3]):
        paired[name]={}
        for c in CONDITIONS:
            p=_paired(rows[c],truths[c],3,j,c in CONDITIONS[:2]);p['delta']=previous._delta(primary[c]['overall'],methods[name]['paths'][c]['overall']);paired[name][c]=p
    source,cross,ground,frozen=_decision(primary,cal)
    result=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=m['code_commit'],method_names=list(METHODS),primary_method=PRIMARY,conditions=list(CONDITIONS),methods=methods,
        primary_summary={c:primary[c]['overall'] for c in CONDITIONS},paired_primary_vs_controls=paired,ground_source_success=source,cross_clean_success=cross,ground_stage_success=ground,
        frozen_rejection_success=frozen,scientific_success=frozen,development_gate_passed=frozen,source_star_success=primary['source_star_DURA']['goals']['passed'],cross_star_success=primary['cross_star_DURA']['goals']['passed'],
        next_step='FROZEN_REJECTION_PASSED_NEEDS_UDA_DISCOVERY_REGISTRATION' if frozen else 'NO_PROMOTION',truth_last=True,all_four_predictions_before_truth=True,all_16_points_before_truth=True,
        source_known_truth_reads=1,source_novel_truth_reads=1,target_truth_reads=2,private_truth_reads=4,U_truth_reads=0,source_U_GT_reads=0,legacy_energy_overall_exact=True,
        replay_diagnostics=m['replay_diagnostics'],classification_unchanged_from_cached_F0=m['classification_unchanged_from_cached_F0'],original_models_unchanged=True,original_default_head_unchanged=True,
        classification_path='legacy_F0/SSL for controls; the same actual frozen F0 forward for both ViM readouts',identity_baseline_path='same_DURA_output_original_F0_cached_baseline',
        source_only=True,source_fit_role='Us',source_Us_fit_rows=m['source_subspace_fit_rows'],source_TX_labels_used=False,source_TX_labels_before_CAL=0,source_L_reads=0,source_OE_reads=0,
        source_statistical_learning=True,new_neural_updates=0,optimizer_updates=0,
        new_source_subspace_models=2,nuisance_SVD_fit_count=1,new_source_density_models=0,new_source_threshold_sets=2,new_threshold_values=12,reused_threshold_values=12,threshold_values=24,threshold_bytes=192,
        source_fit_diagnostics=m['source_fit_diagnostics'],inherited_data_costs=m['inherited_data_costs'],
        costs={k:m[k] for k in ('real_F_forward_rows','real_SSL_forward_rows','real_student_forward_rows','G_forward_rows','synthetic_F_forward_rows','G_synthetic_forward_rows','source_Us_reads','source_Us_ID_reads')},
        primary_deployed_neural_parameters=499944,primary_deployed_geometry_numeric_values=12961,primary_deployed_geometry_numeric_bytes=103688,primary_threshold_values=6,primary_threshold_bytes=48,
        source_rows_deployed=False,nuisance_basis_audit_only=True,adaptation_success=None,new_target_DA=False,target_U_reads=0,target_REG=False,registration_success=None,A_N=None,H_KN=None,H_OW=None,
        formal_target_accessed=False,global_physical_history='UNKNOWN',source_novel_historically_exposed=True,new_blind_confirmation=False,full_goal_status='ACTIVE',full_goal_passed=False,final_phase3_complete=False,
        source_CAL_role='separate known source calibration half; TEST labels are not PCA/alpha/rank/calibration inputs',
        scope='Frozen original F0 with two source-Us-only uncentered ViM states; domain mode removes the15public-domain mean span before fixed80 residual PCA. No coordinate renormalization, NN update, target adaptation or CPU refit. Fixed domain-ViM must satisfy formal95 rejection/known coverage and original-F0loss2pp; full Phase3 remains active')
    out.mkdir(parents=True);_write(out/'results.json',result);_write(out/'complete.json',{k:v for k,v in result.items() if k not in ('methods','paired_primary_vs_controls')});print(STATUS,result['next_step'],flush=True);return result


def _legacy_score(config):
    return _read(Path(config['reference_config']['output_root']).parent/'score/results.json')


def main():
    p=argparse.ArgumentParser()
    for flag in ('prediction-root','source-known-truth','source-novel-truth','clean-truth','star-truth','output'):p.add_argument('--'+flag,required=True)
    a=p.parse_args();score(a.prediction_root,a.source_known_truth,a.source_novel_truth,a.clean_truth,a.star_truth,a.output)


if __name__=='__main__':main()
