"""Independent truth-last assessment of frozen-PCD target moment correction."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from . import fresh_world_score as fresh
from . import star_dura_best_score as best
from . import star_consensus_score as common
from . import star_density_core as density
from .fresh_dura_score import _source

RUN_ID='20261031-phase3-pcd-moment-s2026092701-r01'
SCHEMA='pamf_phase3_pcd_moment_scores_v1'
STATUS='PCD_MOMENT_SCORES_COMPLETE'
METHODS=tuple(a+'_'+b+'_ratio' for a in ('E0','PCD','SHIFT','CORAL') for b in ('pooled','predicted'))
PRIMARY=METHODS[-1]
CONDITIONS=('source_DURA','clean_DURA','star_DURA')
COUNTS=dict(L=6300,OE=6134,V=27000,novel=3150)
Q_COUNT=10238
U_COUNT=576
_read,_write,_source_truth,_future_truth=fresh._read,fresh._write,fresh._source_truth,fresh._future_truth


def _evaluation():
    from . import pcd_moment_run
    return pcd_moment_run


def _arrays(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def _prediction(root):
    """Validate complete public artifacts without re-estimating any U statistic."""
    e=_evaluation();root=Path(root);done=_read(root/'complete.json');c=e.validate_config(done['config'])
    pm,states,expected_tau,parent_cal=e._reference(c)
    expected=dict(schema=e.SCHEMA,status=e.STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c,
        method_names=list(METHODS),primary_method=PRIMARY,conditions=list(CONDITIONS),reference_prediction_identity=pm,
        parent_F0_identity=pm['parent_F0_identity'],dura_identity=pm['dura_identity'],
        known_count=COUNTS['V'],novel_count=COUNTS['novel'],Q_counts={name:Q_COUNT for name in CONDITIONS[1:]},Q_manifests=pm['Q_manifests'],
        target_U_reads=U_COUNT,U_ID_metadata_reads=U_COUNT,U_feature_rows_read=U_COUNT,U_statistic_fit_calls=1,
        private_truth_reads=0,U_truth_reads=0,source_novel_truth_reads=0,target_truth_reads=0,
        new_neural_updates=0,optimizer_updates=0,source_density_models_fitted=0,new_threshold_sets_fitted=0,real_F_forward_rows=0,real_signal_forward_rows=0,G_forward_rows=0,
        source_IQ_reads=0,target_Q_IQ_reads=0,source_only=False,target_DA=True,target_REG=False,classification_unchanged=True,
        source_L_reads=0,source_OE_reads=0,source_Us_reads=0,original_U_s_reads=0,
        source_cached_prediction_rows=COUNTS['V']+COUNTS['novel'],target_Q_cached_prediction_rows=2*Q_COUNT,
        synthetic_F_forward_rows=2,synthetic_signal_forward_rows=4,synthetic_r03_forward_rows=2,synthetic_PCD_forward_rows=2,G_synthetic_forward_rows=2,
        threshold_values=48,threshold_bytes=384,unique_source_threshold_values=24,unique_source_threshold_bytes=192,new_threshold_values=0,
        formal_target_accessed=False,final_phase3_complete=False)
    if any(done.get(k)!=v for k,v in expected.items()):raise ValueError('Complete frozen-model eight-readout three-condition moment predictions required before truth')
    frozen=dict(all_three_DURA_conditions_complete=True,all_conditions_through_DURA=True,source_states_unchanged=True,
        model_parameters_buffers_unchanged=True,source_control_readouts_reused_exactly=True,adapters_frozen_before_Q=True,
        all_thresholds_frozen_before_U_and_Q=True,source_V_fit=False,source_NV_fit=False,Q_fit=False)
    if any(done.get(k)!=v for k,v in frozen.items()):raise ValueError('Every condition and all inherited neural/source states must be frozen before truth')
    identities=dict(E0=pm['model_identities']['E0'],PCD=pm['model_identities']['TARGET_PCD'])
    fit=_read(root/'models/complete.json')
    model_header=dict(schema=e.SCHEMA,status='PCD_MOMENT_MODELS_FROZEN',run_id=RUN_ID,code_commit=c['code_commit'],model_identities=identities,
        parent_F0_identity=pm['parent_F0_identity'],dura_identity=pm['dura_identity'],models_frozen=True,models_refitted=False,
        thresholds_refitted=False,source_V_fit=False,novel_fit=False,target_fit=False)
    if done.get('model_identities')!=identities or any(fit.get(k)!=v for k,v in model_header.items()):raise ValueError('Both original source density/model identities must be frozen without refit')
    smoke=_read(root/'smoke.json')
    if any(smoke.get(k)!=v for k,v in dict(status='PASS',actual_F0_r03_DURA_PCD_checkpoints=True,synthetic_F_forward_rows=2,
            synthetic_signal_forward_rows=4,G_synthetic_forward_rows=2,real_data_reads=0,optimizer_updates=0).items()):
        raise ValueError('Actual F0/r03/PCD/DURA no-query smoke must pass before discovery consumption')
    # Source states and thresholds are inherited from f4b5; neither this gate
    # nor the producer's metadata helper may open discovery feature values.
    for name in ('E0','PCD'):
        saved=_arrays(root/'models'/(name+'.npz'));density.validate_state(saved,'density_ratio')
        if not e._same(saved,states[name]):raise ValueError('Inherited '+name+' source density changed')
    if (expected_tau.shape!=(8,6) or expected_tau.dtype!=np.float64 or not np.isfinite(expected_tau).all()
            or not np.array_equal(expected_tau[4:6],expected_tau[2:4]) or not np.array_equal(expected_tau[6:8],expected_tau[2:4])):
        raise ValueError('The target adapters must use the same inherited PCD thresholds')
    saved=_arrays(root/'thresholds.npz');tau=saved.get('conditional')
    if (set(saved)!={'conditional'} or not isinstance(tau,np.ndarray) or tau.dtype!=np.float64 or tau.shape!=(8,6)
            or not np.isfinite(tau).all() or not np.array_equal(tau,expected_tau)):
        raise ValueError('All eight operating points must exactly inherit the four fixed source threshold vectors')
    cal=_read(root/'calibration/complete.json')
    if cal!=parent_cal:raise ValueError('Exact inherited eight-method source calibration reports required without refit')
    adapters={a:_arrays(root/'adapters'/(a+'.npz')) for a in ('SHIFT','CORAL')}
    for name,a in adapters.items():
        e.core.validate_adapter(a)
        if a['kind'].item()!=name:raise ValueError('Named adapter kind differs')
    meta=_read(root/'adapters/complete.json');diag=meta.get('diagnostics',{});status=diag.get('status')
    adapter_header=dict(schema=e.SCHEMA,status='PCD_MOMENT_ADAPTERS_FROZEN',run_id=RUN_ID,code_commit=c['code_commit'],
        model_identity=identities['PCD'],parent_F0_identity=pm['parent_F0_identity'],dura_identity=pm['dura_identity'],
        adapters_frozen_before_Q=True,query_fit=False,U_only=True,U_count=U_COUNT,U_IDs_ref='adapters/U_ids.npy',
        source_state_refitted=False,thresholds_refitted=False,optimizer_updates=0)
    if (any(meta.get(k)!=v for k,v in adapter_header.items()) or status not in ('FITTED','NO_CORE')
            or meta.get('fit_status')!=status or meta.get('adaptation_status')!=status or done.get('adaptation_status')!=status
            or meta.get('fit_succeeded') is not (status=='FITTED') or diag.get('fit_succeeded') is not (status=='FITTED')
            or done.get('fit_succeeded') is not (status=='FITTED')):
        raise ValueError('One explicit fitted or NO_CORE target-only moment state must freeze before all predictions')
    manifest,reference=e._U_header(c,pm)
    if (meta.get('U_manifest')!=manifest or done.get('U_manifest')!=manifest
            or meta.get('U_reference_identity')!=reference or done.get('U_reference_identity')!=reference):
        raise ValueError('Discovery cache capsule/split and frozen-state provenance differ')
    _legal_adapters(e.core,states['PCD'],adapters,status)
    ids=np.load(root/'adapters/U_ids.npy',allow_pickle=False)
    if not common._ids(ids,U_COUNT):raise ValueError('Only complete unique opaque discovery IDs may be read by the CPU scorer')
    used=set(ids.tolist());rows={};paired_ids=None
    for condition in CONDITIONS:
        rows[condition]={} if condition=='source_DURA' else None
        for role in (('known','novel') if condition=='source_DURA' else ('Q',)):
            a=e._source_rows(root,role) if role!='Q' else e._target_rows(root,condition)
            old=fresh.evaluation._source_rows(c['reference_config']['output_root'],'source_DURA',role) if role!='Q' else fresh.evaluation._target_rows(c['reference_config']['output_root'],condition)
            original=e._select(old)
            for key,value in original.items():
                actual=a[key][:,:4] if key in ('scores','raw_scores') else a[key]
                if not np.array_equal(actual,value):raise ValueError('Every original E0/PCD score, unit, logit, identity and metadata must remain exact')
            z=a['signal_features'];closed=a['closed'];raw0=fresh.evaluation._raw(states['E0'],z[:,0],closed);raw1=fresh.evaluation._raw(states['PCD'],z[:,1],closed)
            adapted=[raw1,raw1] if role!='Q' else [e.core.raw_scores(states['PCD'],z[:,1],closed,adapters[n]) for n in ('SHIFT','CORAL')]
            raw=np.concatenate((raw0,raw1,*adapted),axis=1)
            if not np.allclose(a['raw_scores'],raw,rtol=1e-12,atol=1e-9) or not np.array_equal(a['scores'],a['raw_scores']-tau[:,closed].T):
                raise ValueError('Frozen target transforms or original-F0 conditional routing do not reconstruct')
            if role!='Q' and any(not np.array_equal(a[key][:,4:6],a[key][:,2:4]) or not np.array_equal(a[key][:,6:8],a[key][:,2:4]) for key in ('raw_scores','scores')):
                raise ValueError('Source RX must bypass the target adapters and repeat PCD exactly')
            if status=='NO_CORE' and (not np.array_equal(a['raw_scores'][:,4:6],a['raw_scores'][:,2:4])
                    or not np.array_equal(a['raw_scores'][:,6:8],a['raw_scores'][:,2:4])):
                raise ValueError('NO_CORE must preserve the original PCD operating point exactly')
            row_ids=set(a['ids'].tolist())
            if used&row_ids:raise ValueError('Discovery U, source and future query physical records overlap')
            if role=='Q':
                if paired_ids is None:paired_ids=a['ids'].copy()
                elif not np.array_equal(paired_ids,a['ids']):raise ValueError('Clean/star future physical ID order differs')
                rows[condition]=a
            else:used.update(row_ids);rows[condition][role]=a
    return done,cal,rows


def _legal_adapters(core,state,adapters,status):
    if status=='NO_CORE':
        if any(not np.array_equal(a['matrix'],np.eye(160)) or not np.array_equal(a['offset'],np.zeros((6,160))) for a in adapters.values()):
            raise ValueError('NO_CORE requires two exact identity adapters')
        return
    moments=core.source_moments(state);A=adapters['CORAL']['matrix'];delta=-adapters['SHIFT']['offset'][0]
    B=moments['sqrt']@A@moments['inverse_sqrt']
    if not np.allclose(B,B.T,rtol=0,atol=1e-10):raise ValueError('CORAL must be symmetric in source-standardized coordinates')
    eigenvalues=np.linalg.eigvalsh((B+B.T)*.5)
    if np.any(eigenvalues<.5-1e-10) or np.any(eigenvalues>2.+1e-10):raise ValueError('CORAL standardized responses exceed the fixed bounds')
    expected=moments['class_means']-(moments['class_means']+delta)@A
    if not np.allclose(adapters['CORAL']['offset'],expected,rtol=1e-12,atol=1e-10):raise ValueError('CORAL class offsets must share the same source means and SHIFT translation')


def _paired(known,novel,new_index,old_index,km=None,nm=None):
    def select(a,index):return dict(a,scores=a['scores'][:,[index]])
    return best.paired_counts(select(known,new_index),select(known,old_index),select(novel,new_index),select(novel,old_index),0,km,nm)


def _pairs(rows,nt,truths,index,primary,control):
    result={}
    for condition in CONDITIONS:
        if condition=='source_DURA':
            k,n=rows[condition]['known'],rows[condition]['novel'];km=dict(common._masks(k['rx_i'],k['scenario']));nm=dict(common._masks(n['rx_i'],nt['scenario']))
            rx={str(r):(k['rx_i']==r,n['rx_i']==r) for r in common.RECEIVERS};days={str(d):(k['day_i']==d,n['day_i']==d) for d in (1,2,3)}
            uy=nt['tx'];unknowns=common.UNKNOWN_TX
        else:
            a=rows[condition];t=truths[condition.split('_')[0]];y=t['y'];ki=y<6;ni=~ki
            k=dict(ids=a['ids'][ki],y=y[ki],closed=a['closed'][ki],scores=a['scores'][ki]);n=dict(ids=a['ids'][ni],scores=a['scores'][ni])
            scenes=('clean',) if condition=='clean_DURA' else common.SCENES
            km={s:t['scenario'][ki]==s for s in scenes};nm={s:t['scenario'][ni]==s for s in scenes};uy=y[ni];unknowns=fresh.TARGET_UNKNOWN
            rx={str(r):(t['receiver'][ki]==r,t['receiver'][ni]==r) for r in np.unique(t['receiver'])};days={d:(t['day'][ki]==d,t['day'][ni]==d) for d in fresh.DAYS}
        pair=lambda km=None,nm=None:_paired(k,n,7,index,km,nm)
        result[condition]=dict(counts=pair(),groups={g:pair(mask,nm[g]) for g,mask in km.items()},
            by_RX={g:pair(*masks) for g,masks in rx.items()},by_day={g:pair(*masks) for g,masks in days.items()},
            by_known_TX={str(c):pair(k['y']==c) for c in range(6)},by_unknown_TX={str(c):pair(nm=uy==c) for c in unknowns},
            delta=best._delta(primary[condition]['overall'],control[condition]['overall']))
    return result


def score(prediction_root,source_novel_truth,clean_truth,star_truth,output):
    out=Path(output)
    if out.exists():raise FileExistsError('New immutable independent PCD-moment score output required')
    done,cal,rows=_prediction(prediction_root)
    nt=_source_truth(source_novel_truth,rows['source_DURA']['novel'])
    truths={base:_future_truth(path,rows[base+'_DURA']['ids']) for base,path in (('clean',clean_truth),('star',star_truth))}
    for base,t in truths.items():
        if not np.isin(t['scenario'],('clean',) if base=='clean' else common.SCENES).all():raise ValueError('Fixed clean/star future scenario mismatch')
    if any(not np.array_equal(truths['clean'][k],truths['star'][k]) for k in ('ids','y','tx','receiver','day')):raise ValueError('Both conditions must describe the same ordered physical future records')
    methods={}
    for index,name in enumerate(METHODS):
        paths={'source_DURA':_source(rows['source_DURA'],nt,index)}
        paths.update({c:fresh._target(rows[c],truths[c.split('_')[0]],c,index) for c in CONDITIONS[1:]})
        methods[name]=dict(paths=paths,diagnostic_only=name!=PRIMARY,parent_source_calibration_passed=bool(cal['methods'][name]['constraints_passed']))
    primary=methods[PRIMARY]['paths'];paired={name:_pairs(rows,nt,truths,index,primary,methods[name]['paths']) for index,name in ((1,METHODS[1]),(3,METHODS[3]),(5,METHODS[5]))}
    gains={name:paired[name]['star_DURA']['delta']['A_K'] for name in (METHODS[3],METHODS[5])}
    positive=all(v is not None and v>1e-12 for v in gains.values())
    rejection=all(p['star_DURA']['delta']['R_U'] is not None and p['star_DURA']['delta']['R_U']>=-1e-12 for p in paired.values())
    known=bool(primary['source_DURA']['known_12_groups_passed'] and cal['methods'][PRIMARY]['constraints_passed'])
    protection=all(primary[c]['goals']['passed'] for c in CONDITIONS[1:]);success=bool(done['adaptation_status']=='FITTED' and known and positive and rejection and protection)
    result=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=done['code_commit'],method_names=list(METHODS),primary_method=PRIMARY,conditions=list(CONDITIONS),
        methods=methods,primary_summary={c:primary[c]['overall'] for c in CONDITIONS},paired_primary_vs_controls=paired,
        star_A_K_gain_vs_PCD=gains[METHODS[3]],star_A_K_gain_vs_SHIFT=gains[METHODS[5]],star_A_K_gain_over_both_controls=bool(positive),
        star_R_U_not_lower_than_all_controls=bool(rejection),source_known_12_passed=known,source_known_42_passed=bool(primary['source_DURA']['known_42_groups_passed']),
        target_protection_passed=bool(protection),scientific_success=success,adaptation_success=success,development_gate_passed=success,eligible_primary=success,
        next_step='PCD_MOMENT_ADAPTATION_PASSED' if success else 'NO_PROMOTION',truth_last=True,all_three_predictions_before_truth=True,
        source_truth_reads=1,target_truth_reads=2,U_truth_reads=0,scorer_U_ID_metadata_reads=U_COUNT,scorer_U_feature_reads=0,scorer_U_statistics_refitted=False,
        target_statistics_status=done['adaptation_status'],target_statistics_fit_calls=1,target_U_reads=U_COUNT,U_feature_rows_read=U_COUNT,
        new_target_DA=True,new_target_neural_updates=0,neural_updates=0,density_fit_updates=0,threshold_fit_updates=0,
        classification_unchanged=True,classification_path='same_DURA_original_F0_default_head',identity_baseline_path='same_DURA_original_F0',
        inherited_controls_reused_exactly=True,source_thresholds_reused_exactly=True,source_path_bypasses_target_adapter=True,
        source_V_role='inherited known-only calibration; no new validation or target calibration',threshold_values=48,threshold_bytes=384,
        unique_inherited_threshold_values=24,unique_inherited_threshold_bytes=192,
        source_only=False,PCD_has_inherited_target_U_ancestor=True,inherited_PCD_target_U_count=576,adapter_has_target_U_ancestor=True,
        source_bank_deployed=False,U_bank_deployed=False,target_REG=False,registration_success=None,A_N=None,H_OW=None,
        formal_target_accessed=False,full_goal_status='ACTIVE',full_goal_passed=False,final_phase3_complete=False,
        primary_deployed_neural_parameters=834665,primary_adapter_numeric_values=26560,primary_adapter_numeric_bytes=212480,
        costs={k:v for k,v in done.items() if k.endswith('_forward_rows') or k.endswith('_reads')},
        inherited_training_cost_scope='Original F0/r03/final PCD U576/DURA training, source density, calibration and caches remain inherited costs',
        source_RX_seen=True,source_novel_historically_exposed=True,fresh_development_RX=34,global_physical_history='UNKNOWN',new_blind_confirmation=False,
        scope='U576-only frozen-PCD moment correction with fixed source density/thresholds and unchanged original-F0 identity; all fixed source/clean/star predictions precede independent truth, with no discovery or registration')
    out.mkdir(parents=True);_write(out/'results.json',result);_write(out/'complete.json',{k:v for k,v in result.items() if k not in ('methods','paired_primary_vs_controls')})
    print(STATUS,result['next_step'],flush=True);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--prediction-root',required=True);p.add_argument('--source-novel-truth',required=True)
    p.add_argument('--clean-truth',required=True);p.add_argument('--star-truth',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();score(a.prediction_root,a.source_novel_truth,a.clean_truth,a.star_truth,a.output)


if __name__=='__main__':main()
