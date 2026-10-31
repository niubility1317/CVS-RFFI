"""Independent truth-last scoring of frozen receiver input states."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from . import receiver_input_run as evaluation
from . import prototype_identity_score as previous
from . import fresh_world_score as fresh
from . import star_dura_best_score as metrics
from . import star_consensus_score as common

RUN_ID=evaluation.RUN_ID
SCHEMA='pamf_phase3_receiver_input_scores_v1'
STATUS='RECEIVER_INPUT_SCORES_COMPLETE'
ARMS=evaluation.ARMS
METHODS=evaluation.METHODS
PRIMARY=METHODS[-1]
CONDITIONS=evaluation.CONDITIONS
_read,_write,_source_truth,_future_truth=fresh._read,fresh._write,fresh._source_truth,fresh._future_truth


def _prediction(root):
    done,_,_,cal,rows=evaluation._prediction_gate(root)
    if (done.get('run_id')!=RUN_ID or done.get('method_names')!=list(METHODS) or done.get('primary_method')!=PRIMARY
            or set(rows)!=set(evaluation.ROW_KEYS) or any(set(v)!=set(ARMS) for v in rows.values()) or set(cal.get('methods',{}))!=set(METHODS)):
        raise ValueError('All six frozen readouts and three DURA conditions required before truth')
    return done,cal,rows


def _paired(rows,truths,nt,methods,reference):
    paired={};main=methods[PRIMARY]['paths'];ref=reference+'_predicted_ratio'
    for condition in CONDITIONS:
        if condition=='source_DURA':
            new,old=rows['source_V']['TARGET_INPUT'],rows['source_V'][reference];nn,on=rows['source_NV']['TARGET_INPUT'],rows['source_NV'][reference]
            km=dict(common._masks(new['rx_i'],new['scenario']));nm=dict(common._masks(nn['rx_i'],nt['scenario']));uy=nt['tx'];unknowns=common.UNKNOWN_TX
            rx_masks={str(r):(new['rx_i']==r,nn['rx_i']==r) for r in common.RECEIVERS};day_masks={str(d):(new['day_i']==d,nn['day_i']==d) for d in (1,2,3)}
        else:
            t=truths[condition.split('_')[0]];y=t['y'];k=y<6;u=~k;a,b=rows['Q_'+condition]['TARGET_INPUT'],rows['Q_'+condition][reference]
            def known(v):return dict(ids=v['ids'][k],y=y[k],closed=v['closed'][k],scores=v['scores'][k])
            def novel(v):return dict(ids=v['ids'][u],scores=v['scores'][u])
            new,old,nn,on=known(a),known(b),novel(a),novel(b);scenes=('clean',) if condition=='clean_DURA' else common.SCENES
            km={s:t['scenario'][k]==s for s in scenes};nm={s:t['scenario'][u]==s for s in scenes};uy=y[u];unknowns=fresh.TARGET_UNKNOWN
            rx_masks={str(r):(t['receiver'][k]==r,t['receiver'][u]==r) for r in np.unique(t['receiver'])};day_masks={d:(t['day'][k]==d,t['day'][u]==d) for d in fresh.DAYS}
        paired[condition]=dict(counts=previous._pair(new,old,nn,on,1),groups={g:previous._pair(new,old,nn,on,1,mask,nm[g]) for g,mask in km.items()},
            by_RX={g:previous._pair(new,old,nn,on,1,*masks) for g,masks in rx_masks.items()},by_day={g:previous._pair(new,old,nn,on,1,*masks) for g,masks in day_masks.items()},
            by_known_TX={str(c):previous._pair(new,old,nn,on,1,new['y']==c) for c in range(6)},by_unknown_TX={str(c):previous._pair(new,old,nn,on,1,nm=uy==c) for c in unknowns},
            delta=metrics._delta(main[condition]['overall'],methods[ref]['paths'][condition]['overall']))
    return paired


def score(prediction_root,source_novel_truth,clean_truth,star_truth,output):
    out=Path(output)
    if out.exists():raise FileExistsError('New immutable receiver-input score root required')
    done,cal,rows=_prediction(prediction_root)
    nt=_source_truth(source_novel_truth,rows['source_NV']['OLD_PCD'])
    truths={base:_future_truth(path,rows['Q_'+base+'_DURA']['OLD_PCD']['ids']) for base,path in (('clean',clean_truth),('star',star_truth))}
    for base,t in truths.items():
        if not np.isin(t['scenario'],('clean',) if base=='clean' else common.SCENES).all():raise ValueError('Future condition/scenario mismatch')
    if any(not np.array_equal(truths['clean'][key],truths['star'][key]) for key in ('ids','y','tx','receiver','day')):raise ValueError('Clean/star private physical labels and dates must be paired')
    methods={}
    for j,name in enumerate(METHODS):
        arm=ARMS[j//2];index=j%2;source=dict(known=rows['source_V'][arm],novel=rows['source_NV'][arm])
        paths=dict(source_DURA=metrics._source(source,nt,index));paths.update({c:fresh._target(rows['Q_'+c][arm],truths[c.split('_')[0]],c,index) for c in CONDITIONS[1:]})
        methods[name]=dict(paths=paths,classification_model='frozen original F0 with '+arm+' input path',diagnostic_only=name!=PRIMARY,source_calibration_passed=bool(cal['methods'][name]['constraints_passed']))
    main=methods[PRIMARY]['paths'];paired={arm:_paired(rows,truths,nt,methods,arm) for arm in ('OLD_PCD','SOURCE_INPUT')}
    controls=[methods[a+'_predicted_ratio']['paths']['star_DURA']['overall'] for a in ('OLD_PCD','SOURCE_INPUT')]
    decisions=[previous._control_gate(main['star_DURA']['overall'],control) for control in controls]
    positive=all(v[0] for v in decisions);kept=all(v[1] for v in decisions)
    known=bool(main['source_DURA']['known_12_groups_passed'] and cal['methods'][PRIMARY]['constraints_passed']);protection=all(main[c]['goals']['passed'] for c in CONDITIONS[1:]);success=bool(known and positive and kept and protection)
    result=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=done['code_commit'],method_names=list(METHODS),primary_method=PRIMARY,arm_order=list(ARMS),conditions=list(CONDITIONS),methods=methods,
        primary_summary={c:main[c]['overall'] for c in CONDITIONS},paired_primary_vs_OLD_PCD_predicted_ratio=paired['OLD_PCD'],paired_primary_vs_SOURCE_INPUT_predicted_ratio=paired['SOURCE_INPUT'],
        star_A_K_gain_over_both_controls=bool(positive),star_R_U_not_lower_than_both_controls=bool(kept),source_known_12_passed=known,target_protection_passed=bool(protection),
        adaptation_success=success,development_gate_passed=success,eligible_primary=success,next_step='DEVELOPMENT_INPUT_ADAPTATION_PASSED_REQUIRES_SEPARATE_CONFIRMATION' if success else 'NO_PROMOTION',
        truth_last=True,all_three_predictions_before_truth=True,source_truth_reads=1,target_truth_reads=2,U_truth_reads=0,classification_unchanged=False,original_F0_weights_unchanged=True,original_default_head_unchanged=True,
        identity_baseline_path='same_DURA_output_original_F0_without_input_adapter',candidate_identity_path='same_original_F0_head_after_own_input_adapter',old_density_and_thresholds_unchanged=True,
        prediction_costs={k:v for k,v in done.items() if k.endswith('_reads') or k.endswith('_forward_rows')},new_density_fits=0,new_threshold_sets_fitted=0,new_neural_weight_updates=0,
        training_run_id=done['training_run_id'],training_code_commit=done['training_code_commit'],training_reused=done['training_reused'],training_identity=done['training_identity'],
        inherited_training_costs={k:v for k,v in done['training_identity'].items() if k.endswith('_forward_rows') or k.endswith('_reads')},
        inherited_adapter_optimizer_updates=done['inherited_adapter_optimizer_updates'],new_adapter_optimizer_updates=0,evaluation_only_recovery=True,
        adapter_parameters_per_arm=16,primary_adapter_bytes=64,threshold_values=36,threshold_bytes=288,unique_threshold_values=12,
        PCD_has_inherited_target_U_ancestor=True,inherited_PCD_target_U_count=576,SOURCE_INPUT_adapter_source_only=True,SOURCE_INPUT_system_source_only=False,TARGET_INPUT_uses_target_U=True,
        new_target_DA=False,target_DA_inherited=True,source_only=False,target_REG=False,registration_success=None,A_N=None,H_OW=None,source_bank_deployed=False,U_bank_deployed=False,
        formal_target_accessed=False,full_goal_status='ACTIVE',full_goal_passed=False,final_phase3_complete=False,primary_deployed_neural_parameters=834681,
        source_RX_seen=True,source_novel_historically_exposed=True,fresh_development_RX=34,global_physical_history='UNKNOWN',new_blind_confirmation=False,
        scope='Evaluation-only recovery reuses exact r01 trained input adapters and source-before-U evidence. No new optimization or U IQ reads; source rows are inherited calibration diagnostics, clean/Q are test only, no discovery or registration')
    out.mkdir(parents=True);_write(out/'results.json',result);_write(out/'complete.json',{k:v for k,v in result.items() if k not in ('methods','paired_primary_vs_OLD_PCD_predicted_ratio','paired_primary_vs_SOURCE_INPUT_predicted_ratio')})
    print(STATUS,result['next_step'],flush=True);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--prediction-root',required=True);p.add_argument('--source-novel-truth',required=True);p.add_argument('--clean-truth',required=True);p.add_argument('--star-truth',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();score(a.prediction_root,a.source_novel_truth,a.clean_truth,a.star_truth,a.output)


if __name__=='__main__':main()
