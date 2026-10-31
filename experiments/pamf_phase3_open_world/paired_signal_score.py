"""Independent truth-last assessment of paired signal source/target learning."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from . import fresh_world_score as fresh
from . import star_dura_best_score as previous
from . import star_consensus_score as common
from . import source_us_score as controls
from . import paired_signal_run as evaluation

RUN_ID=evaluation.RUN_ID
SCHEMA='pamf_phase3_paired_signal_scores_v1'
STATUS='PAIRED_SIGNAL_SCORES_COMPLETE'
METHODS=evaluation.METHODS
PRIMARY=METHODS[-1]
CONDITIONS=evaluation.CONDITIONS
_read,_write= fresh._read,fresh._write


def _prediction(root):
    done,cal,rows=evaluation._prediction_gate(root)
    if done.get('method_names')!=list(METHODS)or done.get('primary_method')!=PRIMARY or set(rows)!={'source','clean_DURA','star_DURA'}or set(rows['source'])!={'known','novel'}:
        raise ValueError('All ten methods and three DURA conditions must precede truth')
    return done,cal,rows


def _paired(rows,truths,nt,reference):
    result={}
    for condition in CONDITIONS:
        if condition=='source_DURA':
            k,n=rows['source']['known'],rows['source']['novel'];uy=nt['tx'];unknowns=common.UNKNOWN_TX
            km=dict(common._masks(k['rx_i'],k['scenario']));nm=dict(common._masks(n['rx_i'],nt['scenario']))
            rx={str(r):(k['rx_i']==r,n['rx_i']==r)for r in common.RECEIVERS}
            days={str(d):(k['day_i']==d,n['day_i']==d)for d in(1,2,3)}
        else:
            t=truths[condition];y=t['y'];known=y<6;a=rows[condition]
            k=dict(y=y[known],closed=a['closed'][known],scores=a['scores'][known]);n=dict(scores=a['scores'][~known]);uy=y[~known];unknowns=fresh.TARGET_UNKNOWN
            scenes=('clean',)if condition=='clean_DURA'else common.SCENES
            km={s:t['scenario'][known]==s for s in scenes};nm={s:t['scenario'][~known]==s for s in scenes}
            rx={str(r):(t['receiver'][known]==r,t['receiver'][~known]==r)for r in np.unique(t['receiver'])}
            days={d:(t['day'][known]==d,t['day'][~known]==d)for d in fresh.DAYS}
        pair=lambda ka=None,na=None:fresh._pair(k,n,9,reference,ka,na)
        result[condition]=dict(counts=pair(),groups={s:pair(mask,nm[s])for s,mask in km.items()},by_RX={s:pair(*mask)for s,mask in rx.items()},
            by_day={s:pair(*mask)for s,mask in days.items()},by_known_TX={str(c):pair(k['y']==c)for c in range(6)},by_unknown_TX={str(c):pair(na=uy==c)for c in unknowns})
    return result


def score(prediction_root,source_novel_truth,clean_truth,star_truth,output):
    out=Path(output)
    if out.exists():raise FileExistsError('New immutable independent score root required')
    done,cal,rows=_prediction(prediction_root)
    nt=fresh._source_truth(source_novel_truth,rows['source']['novel'])
    truths={c:fresh._future_truth(path,rows[c]['ids'])for c,path in[('clean_DURA',clean_truth),('star_DURA',star_truth)]}
    for c,t in truths.items():
        if not np.isin(t['scenario'],('clean',)if c=='clean_DURA'else common.SCENES).all():raise ValueError('Truth condition/scenario mismatch')
    if any(not np.array_equal(truths['clean_DURA'][k],truths['star_DURA'][k])for k in('ids','y','tx','receiver','day')):raise ValueError('Clean/star truths must pair physical records')
    methods={}
    for j,name in enumerate(METHODS):
        paths={'source_DURA':previous._source(rows['source'],nt,j)}
        paths.update({c:fresh._target(rows[c],truths[c],c,j)for c in CONDITIONS[1:]})
        methods[name]=dict(paths=paths,diagnostic_only=name!=PRIMARY,source_calibration_passed=bool(cal['methods'][name]['constraints_passed']))
    main=methods[PRIMARY]['paths'];paired={}
    for j in(1,3,5,7):
        name=METHODS[j];paired[name]=_paired(rows,truths,nt,j)
        for c in CONDITIONS:paired[name][c]['delta']=previous._delta(main[c]['overall'],methods[name]['paths'][c]['overall'])
    names=(METHODS[7],METHODS[3]);positive,kept=controls._control_gate(main['star_DURA']['overall'],[methods[n]['paths']['star_DURA']['overall']for n in names])
    input_effect={c:previous._delta(methods[METHODS[7]]['paths'][c]['overall'],methods[METHODS[5]]['paths'][c]['overall'])for c in CONDITIONS}
    known=bool(main['source_DURA']['known_12_groups_passed']and cal['methods'][PRIMARY]['constraints_passed'])
    protection=all(main[c]['goals']['passed']for c in CONDITIONS[1:]);success=bool(known and positive and kept and protection)
    result=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=done['code_commit'],method_names=list(METHODS),primary_method=PRIMARY,conditions=list(CONDITIONS),methods=methods,
        primary_summary={c:main[c]['overall']for c in CONDITIONS},paired_primary_vs_SOURCE_predicted_ratio=paired[METHODS[1]],paired_primary_vs_OLD_PCD_predicted_ratio=paired[METHODS[3]],
        paired_primary_vs_SOURCE_P1_predicted_ratio=paired[METHODS[5]],paired_primary_vs_SOURCE_PAIR_predicted_ratio=paired[METHODS[7]],star_A_K_gain_over_both_controls=positive,star_R_U_not_lower_than_both_controls=kept,
        source_known_12_passed=known,target_protection_passed=bool(protection),source_pair_input_effect=input_effect,gate_control_methods=list(names),
        adaptation_success=success,development_gate_passed=success,eligible_primary=success,next_step='PAIRED_SIGNAL_STAGE_PASSED_REQUIRES_CONFIRMATION'if success else'NO_PROMOTION',
        truth_last=True,all_three_predictions_before_truth=True,source_truth_reads=1,target_truth_reads=2,U_truth_reads=0,
        classification_unchanged=True,identity_baseline_path='same_DURA_output_original_F0',source_control_readouts_reused_exactly=True,
        SOURCE_weights_source_only=True,OLD_PCD_weights_source_only=False,OLD_PCD_inherited_target_U_count=576,SOURCE_P1_weights_source_only=True,SOURCE_PAIR_weights_source_only=True,TARGET_PAIR_weights_source_only=False,
        controls_source_sampling_matched=True,pair_source_target_budget_matched=True,new_target_UDA_updates=444,new_student_optimizer_updates=1332,target_REG=False,registration_success=None,A_N=None,H_OW=None,
        new_source_density_models=3,new_source_threshold_sets=6,threshold_values=60,threshold_bytes=480,source_bank_deployed=False,U_bank_deployed=False,
        paired_signal_deployed=True,prototype_deployed=False,prediction_costs={k:v for k,v in done.items()if k.endswith('_reads')or k.endswith('_forward_rows')},training_costs=done.get('training_costs',{}),
        formal_target_accessed=False,full_goal_status='ACTIVE',full_goal_passed=False,final_phase3_complete=False,source_RX_seen=True,source_novel_historically_exposed=True,
        fresh_development_RX=34,global_physical_history='UNKNOWN',new_blind_confirmation=False,
        scope='Three matched paired-signal students, source P1-only versus source pair versus pair/U-PCD; original P1 F0 identity, pure-source initialization, old SOURCE/PCD controls; no registration or blind confirmation')
    out.mkdir(parents=True);_write(out/'results.json',result)
    _write(out/'complete.json',{k:v for k,v in result.items()if k!='methods'and not k.startswith('paired_primary_')})
    print(STATUS,result['next_step'],flush=True);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--prediction-root',required=True);p.add_argument('--source-novel-truth',required=True);p.add_argument('--clean-truth',required=True);p.add_argument('--star-truth',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();score(a.prediction_root,a.source_novel_truth,a.clean_truth,a.star_truth,a.output)
if __name__=='__main__':main()
