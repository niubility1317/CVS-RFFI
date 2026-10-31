"""Truth-last raw/DURA diagnostics of frozen E0/PCD on the fresh RX34 pool."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from . import fresh_world_score as fresh
from . import star_dura_best_score as best
from . import star_consensus_score as common
from . import star_density_core as density

RUN_ID='20261031-phase3-fresh-dura-s2026092701-r01'
SCHEMA='pamf_phase3_fresh_dura_scores_v1'
STATUS='FRESH_DURA_SCORES_COMPLETE'
METHODS=('E0_pooled_ratio','E0_predicted_ratio','PCD_pooled_ratio','PCD_predicted_ratio')
PRIMARY=METHODS[-1]
CONDITIONS=('source_raw','source_DURA','clean_raw','clean_DURA','star_raw','star_DURA')
COUNTS=dict(L=6300,OE=6134,V=27000,novel=3150)
Q_COUNT=10238
_read,_write,_source_truth,_future_truth=fresh._read,fresh._write,fresh._source_truth,fresh._future_truth


def _evaluation():
    from . import fresh_dura_run
    return fresh_dura_run


def _arrays(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def _align(a,b,keys):
    if any(not np.array_equal(a[k],b[k]) for k in keys):raise ValueError('Raw/DURA ordered physical IDs and public metadata differ')


def _prediction(root):
    e=_evaluation();root=Path(root);done=_read(root/'complete.json');c=e.validate_config(done['config'])
    pm,states,tau,parent_cal=e._reference(c)
    expected=dict(schema=e.SCHEMA,status=e.STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c,
        method_names=list(METHODS),primary_method=PRIMARY,conditions=list(CONDITIONS),reference_prediction_identity=pm,
        parent_F0_identity=pm['parent_F0_identity'],dura_identity=pm['dura_identity'],loaded_dura_identity=pm['loaded_dura_identity'],
        known_count=COUNTS['V'],novel_count=COUNTS['novel'],target_U_reads=0,U_ID_metadata_reads=0,private_truth_reads=0,
        source_novel_truth_reads=0,target_truth_reads=0,new_neural_updates=0,density_fit_updates=0,threshold_fit_updates=0,
        all_six_paths_complete=True,all_DURA_main_conditions_complete=True,all_DURA_controls_reused_exactly=True,
        models_frozen_before_any_new_input=True,all_thresholds_frozen_before_source_raw_and_Q=True,
        model_parameters_buffers_unchanged=True,source_states_unchanged=True,
        source_IQ_reads=COUNTS['V']+COUNTS['novel'],target_Q_IQ_reads=2*Q_COUNT,
        real_F_forward_rows=COUNTS['V']+COUNTS['novel']+2*Q_COUNT,
        real_signal_forward_rows=2*(COUNTS['V']+COUNTS['novel']+2*Q_COUNT),G_forward_rows=0,
        synthetic_F_forward_rows=4,synthetic_signal_forward_rows=8,G_synthetic_forward_rows=2,
        formal_target_accessed=False,final_phase3_complete=False)
    if any(done.get(k)!=v for k,v in expected.items()):raise ValueError('Complete fixed six-path E0/PCD prediction identity required before truth')
    # The producer supplies source-only inherited state/threshold objects; no
    # fresh-world scorer gate or discovery U IDs are consumed here.
    if tau.shape!=(4,6) or tau.dtype!=np.float64 or not np.isfinite(tau).all():raise ValueError('Four inherited six-class operating points required')
    identities=dict(E0=pm['model_identities']['E0'],PCD=pm['model_identities']['TARGET_PCD'])
    fit=_read(root/'models/complete.json')
    fitted=dict(schema=e.SCHEMA,status=e.MODELS_STATUS,run_id=RUN_ID,code_commit=c['code_commit'],model_identities=identities,
        parent_F0_identity=pm['parent_F0_identity'],dura_identity=pm['dura_identity'],loaded_dura_identity=pm['loaded_dura_identity'],
        models_frozen=True,models_refitted=False,thresholds_refitted=False,source_V_fit=False,novel_fit=False,target_fit=False)
    if done.get('model_identities')!=identities or any(fit.get(k)!=v for k,v in fitted.items()):
        raise ValueError('Two actual inherited E0/PCD source-only fit states must be frozen before every new input')
    for arm in ('E0','PCD'):
        saved=_arrays(root/'models'/(arm+'.npz'));density.validate_state(saved,'density_ratio')
        if not e._same(saved,states[arm]):raise ValueError('Inherited '+arm+' source density changed')
    threshold=_arrays(root/'thresholds.npz')
    if set(threshold)!={'conditional'} or not np.array_equal(threshold['conditional'],tau) or threshold['conditional'].dtype!=np.float64:
        raise ValueError('Exact inherited FP64 thresholds required, without refit')
    cal=_read(root/'calibration/complete.json')
    if cal!=parent_cal:raise ValueError('Exact inherited selected calibration report required')
    rows={};source_ids=set()
    for condition in CONDITIONS:
        if condition.startswith('source_'):
            rows[condition]={}
            for role in ('known','novel'):
                a=e._source_rows(root,condition,role);rows[condition][role]=a
                raw=e._raw(states,a['signal_features'],a['closed'])
                if not np.allclose(a['raw_scores'],raw,rtol=1e-12,atol=1e-9) or not np.array_equal(a['scores'],a['raw_scores']-tau[:,a['closed']].T):
                    raise ValueError('Source raw/margin must reconstruct with own path original-F0 class routing')
                if condition=='source_DURA':
                    old=e.fresh._source_rows(c['reference_config']['output_root'],'source_DURA',role)
                    if not e._same(a,e._select(old)):raise ValueError('Every inherited DURA source value must remain exact')
                if condition=='source_raw':
                    ids=set(a['ids'].tolist())
                    if source_ids&ids:raise ValueError('Source known and novel roles overlap')
                    source_ids.update(ids)
        else:
            a=e._target_rows(root,condition);rows[condition]=a;raw=e._raw(states,a['signal_features'],a['closed'])
            if not np.allclose(a['raw_scores'],raw,rtol=1e-12,atol=1e-9) or not np.array_equal(a['scores'],a['raw_scores']-tau[:,a['closed']].T):
                raise ValueError('Future raw/margin must reconstruct with own path original-F0 class routing')
            if source_ids&set(a['ids'].tolist()):raise ValueError('Source and future physical IDs overlap')
            if condition.endswith('_DURA'):
                old=e.fresh._target_rows(c['reference_config']['output_root'],condition)
                if not e._same(a,e._select(old)):raise ValueError('Every inherited DURA query value must remain exact')
    for role in ('known','novel'):
        _align(rows['source_raw'][role],rows['source_DURA'][role],('ids','rx_i','day_i')+(('y','scenario') if role=='known' else ()))
    for base in ('clean','star'):_align(rows[base+'_raw'],rows[base+'_DURA'],('ids',))
    _align(rows['clean_raw'],rows['star_raw'],('ids',))
    return done,cal,rows


def _target(a,truth,condition,index):
    if condition not in ('clean_raw','clean_DURA','star_raw','star_DURA'):raise ValueError('Fixed fresh raw/DURA target condition required')
    mapped='clean_DURA' if condition.startswith('clean_') else 'star_DURA'
    return fresh._target(a,truth,mapped,index)


def _source(a,truth,index):
    result=best._source(a,truth,index);k,n=a['known'],a['novel']
    def metric(day,scene):
        km=(k['day_i']==day)&(k['scenario']==scene);nm=(n['day_i']==day)&(truth['scenario']==scene)
        return best._metrics(k['y'][km],k['closed'][km],k['scores'][km,index],truth['tx'][nm],n['scores'][nm,index],k['baseline_closed'][km],tuple(range(6)),common.UNKNOWN_TX)
    result['by_day_scene']={str(d):{s:metric(d,s) for s in common.SCENES} for d in (1,2,3)}
    return result


def _pairs(rows,truths,nt,index,paths):
    pairs={}
    for base in ('source','clean','star'):
        if base=='source':
            new,old=rows['source_DURA']['known'],rows['source_raw']['known'];nn,on=rows['source_DURA']['novel'],rows['source_raw']['novel']
            km=dict(common._masks(new['rx_i'],new['scenario']));nm=dict(common._masks(nn['rx_i'],nt['scenario']));uy=nt['tx'];unknowns=common.UNKNOWN_TX
            rx={str(r):(new['rx_i']==r,nn['rx_i']==r) for r in common.RECEIVERS}
            day={str(d):(new['day_i']==d,nn['day_i']==d) for d in (1,2,3)}
        else:
            t=truths[base];y=t['y'];k=y<6;u=~k
            def known(a):return dict(ids=a['ids'][k],closed=a['closed'][k],y=y[k],scores=a['scores'][k])
            def novel(a):return dict(ids=a['ids'][u],scores=a['scores'][u])
            new,old=known(rows[base+'_DURA']),known(rows[base+'_raw']);nn,on=novel(rows[base+'_DURA']),novel(rows[base+'_raw'])
            scenes=('clean',) if base=='clean' else common.SCENES
            km={s:t['scenario'][k]==s for s in scenes};nm={s:t['scenario'][u]==s for s in scenes};uy=y[u];unknowns=fresh.TARGET_UNKNOWN
            rx={str(r):(t['receiver'][k]==r,t['receiver'][u]==r) for r in np.unique(t['receiver'])}
            day={d:(t['day'][k]==d,t['day'][u]==d) for d in fresh.DAYS}
        pair=lambda k=None,u=None:best.paired_counts(new,old,nn,on,index,k,u)
        pairs[base]=dict(counts=pair(),groups={name:pair(mask,nm[name]) for name,mask in km.items()},
            by_RX={name:pair(*mask) for name,mask in rx.items()},by_day={name:pair(*mask) for name,mask in day.items()},
            by_known_TX={str(c):pair(new['y']==c) for c in range(6)},by_unknown_TX={str(c):pair(u=uy==c) for c in unknowns},
            delta=best._delta(paths[base+'_DURA']['overall'],paths[base+'_raw']['overall']),
            identity_correctness='Each path uses its own original-F0 argmax; correctness is recomputed separately before paired counts')
    return pairs


def score(prediction_root,source_novel_truth,clean_truth,star_truth,output):
    out=Path(output)
    if out.exists():raise FileExistsError('New immutable independent fresh raw/DURA score root required')
    done,cal,rows=_prediction(prediction_root)
    nt=_source_truth(source_novel_truth,rows['source_DURA']['novel'])
    truths={base:_future_truth(path,rows[base+'_DURA']['ids']) for base,path in (('clean',clean_truth),('star',star_truth))}
    for base,t in truths.items():
        if not np.isin(t['scenario'],('clean',) if base=='clean' else common.SCENES).all():raise ValueError('Fresh clean/star truth scenario mismatch')
    if any(not np.array_equal(truths['clean'][key],truths['star'][key]) for key in ('ids','y','tx','receiver','day')):
        raise ValueError('Raw/DURA clean/star must describe the same ordered physical future records')
    methods={};paired={}
    for index,name in enumerate(METHODS):
        paths={c:_source(rows[c],nt,index) for c in CONDITIONS[:2]}
        paths.update({c:_target(rows[c],truths[c.split('_')[0]],c,index) for c in CONDITIONS[2:]})
        methods[name]=dict(paths=paths,diagnostic_only=True,parent_source_calibration_passed=bool(cal['methods'][name]['constraints_passed']))
        paired[name]=_pairs(rows,truths,nt,index,paths)
    main=methods[PRIMARY]['paths'];source=main['source_DURA'];known=bool(source['known_12_groups_passed'] and cal['methods'][PRIMARY]['constraints_passed'])
    target=all(main[c]['goals']['passed'] for c in ('clean_DURA','star_DURA'))
    absolute=bool(known and source['source_unknown_90_diagnostic_passed'] and target)
    result=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=done['code_commit'],method_names=list(METHODS),primary_method=PRIMARY,conditions=list(CONDITIONS),
        methods=methods,primary_summary={c:main[c]['overall'] for c in CONDITIONS},paired_DURA_minus_raw=paired,
        diagnostic_completed=True,next_step='NO_PROMOTION',eligible_primary=False,development_gate_passed=False,
        DURA_primary_absolute_goals_passed=absolute,source_known_12_passed=known,source_known_42_passed=source['known_42_groups_passed'],
        source_unknown_90_diagnostic_passed=source['source_unknown_90_diagnostic_passed'],target_protection_passed=bool(target),
        truth_last=True,all_six_predictions_before_truth=True,source_truth_reads=1,target_truth_reads=2,U_truth_reads=0,
        primary_paths=['source_DURA','clean_DURA','star_DURA'],raw_paths_diagnostic_only=True,source_known_calibration_diagnostic_only=True,
        source_unknown_diagnostic_only=True,source_V_role='inherited known-only threshold calibration, not a new independent validation',
        classification_path='per_input_original_F0',identity_baseline_path='same_path_original_F0',DURA_protection_baseline='original_F0_on_same_DURA_P1',
        shared_frozen_models_and_thresholds=True,DURA_reference_outputs_reused_exactly=True,neural_updates=0,density_fit_updates=0,threshold_fit_updates=0,
        adaptation_success=None,new_target_DA=False,new_target_UDA_updates=0,target_U_reads=0,U_ID_metadata_reads=0,target_REG=False,A_N=None,H_OW=None,
        E0_source_only=True,PCD_has_inherited_target_U_ancestor=True,inherited_PCD_target_U_count=576,
        threshold_values=24,threshold_bytes=192,source_bank_deployed=False,U_bank_deployed=False,formal_target_accessed=False,
        full_goal_status='ACTIVE',full_goal_passed=False,final_phase3_complete=False,
        original_F0_parameters=232993,E0_parameters=334721,PCD_parameters=334721,existing_restorer_parameters=266951,
        primary_deployed_neural_parameters=834665,costs={k:v for k,v in done.items() if k.endswith('_forward_rows') or k.endswith('_reads')},
        inherited_training_cost_scope='Original F0, E0/r03, final PCD/U576, DURA and frozen cache construction remain ancestor costs; new training/statistics/threshold fitting are zero',
        source_RX_seen=True,source_novel_historically_exposed=True,fresh_development_RX=34,global_physical_history='UNKNOWN',new_blind_confirmation=False,
        scope='Frozen same-ID raw/DURA diagnostic on source seen RX and fresh RX34 two-date development records; no new candidate promotion, UDA, discovery or registration')
    out.mkdir(parents=True);_write(out/'results.json',result);_write(out/'complete.json',{k:v for k,v in result.items() if k not in ('methods','paired_DURA_minus_raw')})
    print(STATUS,result['next_step'],flush=True);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--prediction-root',required=True);p.add_argument('--source-novel-truth',required=True)
    p.add_argument('--clean-truth',required=True);p.add_argument('--star-truth',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();score(a.prediction_root,a.source_novel_truth,a.clean_truth,a.star_truth,a.output)


if __name__=='__main__':main()
