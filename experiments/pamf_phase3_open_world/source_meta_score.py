"""Truth-last assessment of matched source-only ERM/MLDG native/open students."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from . import source_meta_run as evaluation
from . import fresh_world_score as fresh
from . import star_dura_best_score as previous
from . import star_consensus_score as common
from . import star_density_core as density

RUN_ID=evaluation.RUN_ID
SCHEMA='pamf_phase3_source_meta_scores_v1'
STATUS='SOURCE_META_SCORES_COMPLETE'
METHODS=evaluation.METHODS
PRIMARY=METHODS[-1]
ARMS=evaluation.ARMS
CONDITIONS=evaluation.CONDITIONS
COUNTS=dict(evaluation.COUNTS)
Q_COUNT=10238
_read,_write,_source_truth=fresh._read,fresh._write,fresh._source_truth
_future_truth=fresh._future_truth


def _arrays(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def _prediction(root):
    root=Path(root);m=_read(root/'complete.json')
    required=dict(schema=evaluation.SCHEMA,status=evaluation.STATUS,run_id=RUN_ID,method_names=list(METHODS),primary_method=PRIMARY,arm_order=list(ARMS),conditions=list(CONDITIONS),
        known_count=COUNTS['V'],novel_count=COUNTS['novel'],source_models_frozen_before_V=True,all_thresholds_frozen_before_novel_and_Q=True,all_three_DURA_conditions_complete=True,all_conditions_through_DURA=True,
        original_F0_unchanged=True,student_heads_jointly_trained=True,identity_head_replaced=False,classification_unchanged=False,source_control_readouts_reused_exactly=True,source_states_unchanged=True,frozen_parameters_buffers_unchanged=True,
        source_density_models_fitted=2,new_threshold_sets_fitted=4,source_V_neural_fit=False,source_V_density_fit=False,source_V_threshold_fit=True,
        private_truth_reads=0,U_truth_reads=0,source_novel_truth_reads=0,target_truth_reads=0,target_U_reads=0,original_U_s_reads=0,new_neural_updates=0,new_candidate_optimizer_updates=888,
        training_target_U_count=0,U_ID_metadata_reads=0,source_IQ_reads=sum(COUNTS.values()),target_Q_IQ_reads=2*Q_COUNT,real_signal_forward_rows=2*sum(COUNTS.values())+4*Q_COUNT,
        real_F_forward_rows=2*Q_COUNT,G_forward_rows=2*Q_COUNT,threshold_values=48,threshold_bytes=384,source_only=True,target_DA=False,target_REG=False,formal_target_accessed=False,final_phase3_complete=False)
    if any(m.get(k)!=v for k,v in required.items()):raise ValueError('All source-only eight-readout three-condition predictions must complete before truth')
    c=evaluation.validate_config(m['config']);pm,old_states,old_tau,old_cal=evaluation._reference(c)
    if c['code_commit']!=m['code_commit'] or m['reference_prediction_identity']!=pm or any(m[k]!=pm[k] for k in ('parent_F0_identity','dura_identity','loaded_dura_identity')):raise ValueError('Actual execution and fixed fresh source/F0/G identities differ')
    new_identities,Uids,_=evaluation._training_metadata(c,pm)
    identities=dict(E0=pm['model_identities']['E0'],OLD_PCD=pm['model_identities']['TARGET_PCD'],**new_identities)
    fit=_read(root/'models/complete.json');cal=_read(root/'calibration/complete.json')
    if (any(fit.get(k)!=v for k,v in dict(schema=evaluation.SCHEMA,status=evaluation.MODELS_STATUS,run_id=RUN_ID,code_commit=m['code_commit'],identities=identities,
            neural_states_frozen=True,source_fit_inputs_only=True,source_V_fit=False,novel_fit=False,target_Q_fit=False,source_density_models_fitted=2,META_NATIVE_has_target_U_ancestor=False).items()) or m['model_identities']!=identities):
        raise ValueError('Two source-fitted new densities and two unchanged control densities required')
    states={arm:_arrays(root/'models'/(arm+'.npz')) for arm in ARMS}
    for state in states.values():density.validate_state(state,'density_ratio')
    if any(not evaluation._same(states[a],old_states[a]) for a in ARMS[:2]):raise ValueError('Inherited control source density changed')
    saved=_arrays(root/'thresholds.npz');tau=saved.get('conditional')
    if (set(saved)!={'conditional'} or not isinstance(tau,np.ndarray) or tau.dtype!=np.float64 or tau.shape!=(8,6) or not np.isfinite(tau).all() or not np.array_equal(tau[:4],old_tau)
            or any(cal.get(k)!=v for k,v in dict(schema=evaluation.SCHEMA,status=evaluation.CALIBRATION_STATUS,run_id=RUN_ID,code_commit=m['code_commit'],known_only=True,unknown_inputs_used=False,target_reads=0).items())
            or set(cal.get('methods',{}))!=set(METHODS) or cal.get('conditional_thresholds')!=tau.tolist()):raise ValueError('Twenty-four inherited and twenty-four new own-head source thresholds required')
    for j,name in enumerate(METHODS):
        if cal['methods'][name]['conditional_thresholds']!=tau[j].tolist() or (j<4 and cal['methods'][name]!=old_cal['methods'][name]):raise ValueError('Inherited/own-head source calibration report differs')
    used=set(Uids.tolist());rows={name:{} for name in CONDITIONS};common_source={};paired=None
    for condition in CONDITIONS:
        old_source={};old_target=None
        roles=('known','novel') if condition=='source_DURA' else ('Q',)
        for role in roles:
            if role=='Q':old_target=evaluation.fresh._target_rows(c['reference_config']['output_root'],condition)
            else:old_source[role]=evaluation.fresh._source_rows(c['reference_config']['output_root'],'source_DURA',role)
            reference=None
            for index,arm in enumerate(ARMS):
                a=evaluation._target_rows(root,condition,arm) if role=='Q' else evaluation._source_rows(root,arm,role)
                raw=evaluation._raw(states[arm],a['signal_features'],a['closed'])
                if not np.allclose(a['raw_scores'],raw,rtol=1e-12,atol=1e-9) or not np.array_equal(a['scores'],a['raw_scores']-tau[2*index:2*index+2,a['closed']].T):raise ValueError('Own predicted identity must route both density and frozen conditional thresholds')
                if index<2:
                    expected=evaluation._slice(old_target if role=='Q' else old_source[role],arm,role)
                    if not evaluation._same(a,expected):raise ValueError('Every reused E0/OLD_PCD score, feature, logit and metadata must remain exact')
                keys=('ids','baseline_closed')+(() if role=='Q' else ('rx_i','day_i'))+(('y','scenario') if role=='known' else ())
                if reference is None:reference=a
                elif any(not np.array_equal(a[k],reference[k]) for k in keys):raise ValueError('All four arms must share original F0 same-input baseline and ordered metadata')
                if role=='Q':rows[condition][arm]=a
                else:rows[condition].setdefault(arm,{})[role]=a
            ids=set(reference['ids'].tolist())
            if ids&used:raise ValueError('Source, discovery and future records overlap')
            if role=='Q':
                if paired is None:paired=reference['ids'].copy()
                elif not np.array_equal(paired,reference['ids']):raise ValueError('Clean/star future ID order differs')
            else:used.update(ids);common_source[role]=reference
    return m,cal,rows


def _paired(new,old,nn,on,index,km=None,nm=None):
    return previous.paired_counts(new,old,nn,on,index,km,nm)


def score(prediction_root,source_novel_truth,clean_truth,star_truth,output):
    out=Path(output)
    if out.exists():raise FileExistsError('New immutable source-meta score root required')
    m,cal,rows=_prediction(prediction_root)
    nt=_source_truth(source_novel_truth,rows['source_DURA']['E0']['novel'])
    truths={base:_future_truth(path,rows[base+'_DURA']['E0']['ids']) for base,path in (('clean',clean_truth),('star',star_truth))}
    for base,t in truths.items():
        if not np.isin(t['scenario'],('clean',) if base=='clean' else common.SCENES).all():raise ValueError('Private future condition/scenario mismatch')
    if any(not np.array_equal(truths['clean'][k],truths['star'][k]) for k in ('ids','y','tx','receiver','day')):raise ValueError('Clean/star truth must describe the same physical future records')
    methods={}
    for j,name in enumerate(METHODS):
        arm=ARMS[j//2];index=j%2
        paths={'source_DURA':previous._source(rows['source_DURA'][arm],nt,index)}
        paths.update({condition:fresh._target(rows[condition][arm],truths[condition.split('_')[0]],condition,index) for condition in CONDITIONS[1:]})
        methods[name]=dict(paths=paths,classification_model=arm,diagnostic_only=name!=PRIMARY,parent_source_calibration_passed=bool(cal['methods'][name]['constraints_passed']))
    primary=methods[PRIMARY]['paths'];paired={}
    for arm in ARMS[:-1]:
        label=arm+'_predicted_ratio';pairs={}
        for condition in CONDITIONS:
            if condition=='source_DURA':
                new,old=rows[condition]['META_NATIVE']['known'],rows[condition][arm]['known'];nn,on=rows[condition]['META_NATIVE']['novel'],rows[condition][arm]['novel']
                km=dict(common._masks(new['rx_i'],new['scenario']));nm=dict(common._masks(nn['rx_i'],nt['scenario']));uy=nt['tx'];unknowns=common.UNKNOWN_TX
                rx_masks={str(r):(new['rx_i']==r,nn['rx_i']==r) for r in common.RECEIVERS};day_masks={str(d):(new['day_i']==d,nn['day_i']==d) for d in (1,2,3)}
            else:
                t=truths[condition.split('_')[0]];y=t['y'];k=y<6;u=~k;a,b=rows[condition]['META_NATIVE'],rows[condition][arm]
                def known(v):return dict(ids=v['ids'][k],y=y[k],closed=v['closed'][k],scores=v['scores'][k])
                def novel(v):return dict(ids=v['ids'][u],scores=v['scores'][u])
                new,old,nn,on=known(a),known(b),novel(a),novel(b)
                scenes=('clean',) if condition=='clean_DURA' else common.SCENES
                km={s:t['scenario'][k]==s for s in scenes};nm={s:t['scenario'][u]==s for s in scenes};uy=y[u];unknowns=fresh.TARGET_UNKNOWN
                rx_masks={str(r):(t['receiver'][k]==r,t['receiver'][u]==r) for r in np.unique(t['receiver'])};day_masks={d:(t['day'][k]==d,t['day'][u]==d) for d in fresh.DAYS}
            pairs[condition]=dict(counts=_paired(new,old,nn,on,1),groups={g:_paired(new,old,nn,on,1,mask,nm[g]) for g,mask in km.items()},
                delta=previous._delta(primary[condition]['overall'],methods[label]['paths'][condition]['overall']),
                by_RX={g:_paired(new,old,nn,on,1,*masks) for g,masks in rx_masks.items()},by_day={g:_paired(new,old,nn,on,1,*masks) for g,masks in day_masks.items()},
                by_known_TX={str(c):_paired(new,old,nn,on,1,new['y']==c) for c in range(6)},by_unknown_TX={str(c):_paired(new,old,nn,on,1,nm=uy==c) for c in unknowns})
        paired[label]=pairs
    gain={name:paired[name]['star_DURA']['delta']['A_K'] for name in ('ERM_NATIVE_predicted_ratio','OLD_PCD_predicted_ratio')}
    positive=all(v is not None and v>1e-12 for v in gain.values())
    rejection=all(paired[a+'_predicted_ratio']['star_DURA']['delta']['R_U'] is not None and paired[a+'_predicted_ratio']['star_DURA']['delta']['R_U']>=-1e-12 for a in ARMS[:-1])
    known=bool(primary['source_DURA']['known_12_groups_passed'] and cal['methods'][PRIMARY]['constraints_passed'])
    protection=all(primary[c]['goals']['passed'] for c in CONDITIONS[1:]);success=bool(known and positive and rejection and protection)
    result=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=m['code_commit'],method_names=list(METHODS),primary_method=PRIMARY,arm_order=list(ARMS),conditions=list(CONDITIONS),methods=methods,
        primary_summary={c:primary[c]['overall'] for c in CONDITIONS},paired_primary_vs_controls=paired,star_A_K_gain_vs_ERM_NATIVE=gain['ERM_NATIVE_predicted_ratio'],star_A_K_gain_vs_OLD_PCD=gain['OLD_PCD_predicted_ratio'],
        star_A_K_gain_over_both_controls=bool(positive),star_R_U_not_lower_than_all_controls=bool(rejection),source_known_12_passed=known,target_protection_passed=bool(protection),
        scientific_success=success,source_meta_success=success,adaptation_success=None,development_gate_passed=success,eligible_primary=success,
        next_step='SOURCE_META_PASSED' if success else 'NO_PROMOTION',truth_last=True,all_three_predictions_before_truth=True,source_truth_reads=1,target_truth_reads=2,U_truth_reads=0,
        original_F0_unchanged=True,student_heads_jointly_trained=True,identity_head_replaced=False,classification_unchanged=False,classification_path='each own native default head',identity_baseline_path='same_DURA_output_original_F0',source_control_readouts_reused_exactly=True,
        source_V_role='known_threshold_calibration_not_independent_test',
        new_candidate_optimizer_updates=888,new_identity_head_updates_total=888,new_body_updates_total=888,new_projection_updates_total=888,prediction_optimizer_updates=0,Sinc_optimizer_steps=0,
        new_source_density_models=2,new_source_threshold_sets=4,threshold_values=48,threshold_bytes=384,source_bank_deployed=False,U_bank_deployed=False,prototypes_deployed=False,
        source_meta_alpha=.001,ERM_inner_alpha=0.,source_meta_beta=.5,second_order=True,
        costs={k:m[k] for k in ('source_IQ_reads','target_Q_IQ_reads','target_U_reads','U_ID_metadata_reads','training_target_U_count','real_F_forward_rows','G_forward_rows','real_signal_forward_rows','real_signal_forward_rows_by_arm','synthetic_F_forward_rows','G_synthetic_forward_rows','synthetic_signal_forward_rows')},
        training_costs=m['training_costs'],combined_new_real_signal_forward_rows=m['combined_new_real_signal_forward_rows'],combined_new_G_forward_rows=m['combined_new_G_forward_rows'],combined_new_real_F_forward_rows=m['combined_new_real_F_forward_rows'],
        ERM_NATIVE_source_only=True,META_NATIVE_source_only=True,META_NATIVE_target_U_ancestor=False,new_target_DA=False,new_target_UDA_updates=0,training_target_U_reads=0,
        OLD_PCD_has_inherited_target_U_ancestor=True,inherited_PCD_target_U_count=576,formal_target_accessed=False,target_REG=False,registration_success=None,A_N=None,H_OW=None,
        full_goal_status='ACTIVE',full_goal_passed=False,final_phase3_complete=False,original_F0_parameters=232993,joint_student_parameters=335681,existing_restorer_parameters=266951,primary_deployed_neural_parameters=602632,
        source_RX_seen=True,source_novel_historically_exposed=True,fresh_development_RX=34,global_physical_history='UNKNOWN',new_blind_confirmation=False,
        scope='Two source-only complete F0 identity/pure-r03 projection students, with matched held-RX F+0.5G outer objectives and inner alpha0 versus second-order alpha0.001; no new U adaptation. Source V calibrates thresholds; previously exposed source novel and RX34 future Q are frozen evaluation, with no discovery, registration or blind confirmation claim')
    out.mkdir(parents=True);_write(out/'results.json',result);_write(out/'complete.json',{k:v for k,v in result.items() if k not in ('methods','paired_primary_vs_controls')});print(STATUS,result['next_step'],flush=True);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--prediction-root',required=True);p.add_argument('--source-novel-truth',required=True);p.add_argument('--clean-truth',required=True);p.add_argument('--star-truth',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();score(a.prediction_root,a.source_novel_truth,a.clean_truth,a.star_truth,a.output)


if __name__=='__main__':main()
