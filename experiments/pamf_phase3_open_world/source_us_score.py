"""Independent truth-last scoring of source-Us regularization, with fixed controls."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from . import fresh_world_score as fresh
from . import star_dura_best_score as previous
from . import star_consensus_score as common
from . import source_us_run as evaluation

RUN_ID=evaluation.RUN_ID
SCHEMA='pamf_phase3_source_us_replay_scores_v1'
STATUS='SOURCE_US_REPLAY_SCORES_COMPLETE'
METHODS=evaluation.METHODS
PRIMARY=METHODS[-1]
CONDITIONS=evaluation.CONDITIONS
COUNTS=dict(evaluation.COUNTS)
Q_COUNT=evaluation.Q_COUNT
_read,_write,_source_truth=fresh._read,fresh._write,fresh._source_truth
_future_truth=fresh._future_truth


def _prediction(root):
    done,cal,rows=evaluation._prediction_gate(root)
    if (done.get('run_id')!=RUN_ID or done.get('method_names')!=list(METHODS) or done.get('primary_method')!=PRIMARY
            or set(rows)!={'source','clean_DURA','star_DURA'} or set(rows['source'])!={'known','novel'}):
        raise ValueError('All six readouts and three complete conditions required before truth')
    return done,cal,rows


def _control_gate(main,controls):
    gain=all(main['A_K'] is not None and c['A_K'] is not None and main['A_K']>c['A_K']+1e-12 for c in controls)
    kept=all(main['R_U'] is not None and c['R_U'] is not None and main['R_U']>=c['R_U']-1e-12 for c in controls)
    return bool(gain),bool(kept)


def _paired(rows,truths,nt,reference):
    pairs={}
    for condition in CONDITIONS:
        if condition=='source_DURA':
            k,n=rows['source']['known'],rows['source']['novel'];uy=nt['tx'];unknowns=common.UNKNOWN_TX
            km=dict(common._masks(k['rx_i'],k['scenario']));nm=dict(common._masks(n['rx_i'],nt['scenario']))
            rx={str(v):(k['rx_i']==v,n['rx_i']==v) for v in common.RECEIVERS}
            days={str(v):(k['day_i']==v,n['day_i']==v) for v in (1,2,3)}
        else:
            t=truths[condition];y=t['y'];known=y<6;a=rows[condition]
            k=dict(y=y[known],closed=a['closed'][known],scores=a['scores'][known]);n=dict(scores=a['scores'][~known]);uy=y[~known];unknowns=fresh.TARGET_UNKNOWN
            scenes=('clean',) if condition=='clean_DURA' else common.SCENES
            km={s:t['scenario'][known]==s for s in scenes};nm={s:t['scenario'][~known]==s for s in scenes}
            rx={str(v):(t['receiver'][known]==v,t['receiver'][~known]==v) for v in np.unique(t['receiver'])}
            days={v:(t['day'][known]==v,t['day'][~known]==v) for v in fresh.DAYS}
        pair=lambda ka=None,na=None:fresh._pair(k,n,5,reference,ka,na)
        pairs[condition]=dict(counts=pair(),groups={s:pair(mask,nm[s]) for s,mask in km.items()},
            by_RX={s:pair(*mask) for s,mask in rx.items()},by_day={s:pair(*mask) for s,mask in days.items()},
            by_known_TX={str(c):pair(k['y']==c) for c in range(6)},by_unknown_TX={str(c):pair(na=uy==c) for c in unknowns})
    return pairs


def score(prediction_root,source_novel_truth,clean_truth,star_truth,output):
    out=Path(output)
    if out.exists():raise FileExistsError('New immutable source-Us score output required')
    done,cal,rows=_prediction(prediction_root)
    nt=_source_truth(source_novel_truth,rows['source']['novel'])
    truths={c:_future_truth(path,rows[c]['ids']) for c,path in [('clean_DURA',clean_truth),('star_DURA',star_truth)]}
    for c,t in truths.items():
        if not np.isin(t['scenario'],('clean',) if c=='clean_DURA' else common.SCENES).all():raise ValueError('Private condition/scenario mismatch')
    if any(not np.array_equal(truths['clean_DURA'][k],truths['star_DURA'][k]) for k in ('ids','y','tx','receiver','day')):
        raise ValueError('Clean/star physical truth must be paired')
    methods={}
    for index,name in enumerate(METHODS):
        paths={'source_DURA':previous._source(rows['source'],nt,index)}
        paths.update({c:fresh._target(rows[c],truths[c],c,index) for c in CONDITIONS[1:]})
        methods[name]=dict(paths=paths,diagnostic_only=name!=PRIMARY,source_calibration_passed=bool(cal['methods'][name]['constraints_passed']))
    main=methods[PRIMARY]['paths'];paired={}
    for index in (1,3):
        name=METHODS[index];paired[name]=_paired(rows,truths,nt,index)
        for c in CONDITIONS:paired[name][c]['delta']=previous._delta(main[c]['overall'],methods[name]['paths'][c]['overall'])
    control_names=(METHODS[1],METHODS[3]);positive,kept=_control_gate(main['star_DURA']['overall'],[methods[n]['paths']['star_DURA']['overall'] for n in control_names])
    known=bool(main['source_DURA']['known_12_groups_passed'] and cal['methods'][PRIMARY]['constraints_passed'])
    protection=all(main[c]['goals']['passed'] for c in CONDITIONS[1:]);success=bool(known and positive and kept and protection)
    result=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=done['code_commit'],method_names=list(METHODS),primary_method=PRIMARY,conditions=list(CONDITIONS),methods=methods,
        primary_summary={c:main[c]['overall'] for c in CONDITIONS},paired_primary_vs_SOURCE_predicted_ratio=paired[METHODS[1]],paired_primary_vs_OLD_PCD_predicted_ratio=paired[METHODS[3]],
        star_A_K_gain_vs_SOURCE=paired[METHODS[1]]['star_DURA']['delta']['A_K'],star_A_K_gain_vs_OLD_PCD=paired[METHODS[3]]['star_DURA']['delta']['A_K'],
        star_A_K_gain_over_both_controls=positive,star_R_U_not_lower_than_both_controls=kept,source_known_12_passed=known,target_protection_passed=bool(protection),
        source_US_regularization_success=success,development_gate_passed=success,eligible_primary=success,adaptation_success=None,
        next_step='SOURCE_US_REGULARIZATION_PASSED_REQUIRES_SEPARATE_CONFIRMATION' if success else 'NO_PROMOTION',
        truth_last=True,all_three_predictions_before_truth=True,source_truth_reads=1,target_truth_reads=2,Us_truth_reads=0,U_truth_reads=0,
        classification_unchanged=True,identity_baseline_path='same_DURA_output_original_F0',source_control_readouts_reused_exactly=True,
        SOURCE_weights_source_only=True,SOURCE_US_weights_source_only=True,SOURCE_US_original_Us_used=True,OLD_PCD_weights_source_only=False,OLD_PCD_inherited_target_U_count=576,
        source_Us_extra_training_data=True,controls_are_budget_matched=False,new_target_UDA_updates=0,target_REG=False,registration_success=None,A_N=None,H_OW=None,
        new_source_density_models=1,new_source_threshold_sets=2,threshold_values=36,threshold_bytes=288,source_bank_deployed=False,U_bank_deployed=False,
        prediction_costs={k:v for k,v in done.items() if k.endswith('_reads') or k.endswith('_forward_rows')},training_costs=done.get('training_costs',{}),
        formal_target_accessed=False,full_goal_status='ACTIVE',full_goal_passed=False,final_phase3_complete=False,source_RX_seen=True,source_novel_historically_exposed=True,
        fresh_development_RX=34,global_physical_history='UNKNOWN',new_blind_confirmation=False,
        scope='Additional source-Us P1 replay regularization, original F0 identity, frozen SOURCE/PCD controls; no new target adaptation, unknown registration or blind confirmation')
    out.mkdir(parents=True);_write(out/'results.json',result);_write(out/'complete.json',{k:v for k,v in result.items() if k not in ('methods','paired_primary_vs_SOURCE_predicted_ratio','paired_primary_vs_OLD_PCD_predicted_ratio')})
    print(STATUS,result['next_step'],flush=True);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--prediction-root',required=True);p.add_argument('--source-novel-truth',required=True);p.add_argument('--clean-truth',required=True);p.add_argument('--star-truth',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();score(a.prediction_root,a.source_novel_truth,a.clean_truth,a.star_truth,a.output)


if __name__=='__main__':main()
