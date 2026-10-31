"""Truth-last MD/RMD source-metric readout scoring; F0 identity is unchanged."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from . import pcd_relative_metric_run as evaluation
from . import fresh_world_score as fresh
from . import star_dura_best_score as previous
from . import star_consensus_score as common
from .source_us_score import _paired

RUN_ID=evaluation.RUN_ID
SCHEMA='pamf_phase3_pcd_relative_metric_scores_v1'
STATUS='PCD_RELATIVE_METRIC_SCORES_COMPLETE'
METHODS=evaluation.METHODS
PRIMARY=METHODS[-1]
CONDITIONS=evaluation.CONDITIONS
_read,_write,_source_truth=fresh._read,fresh._write,fresh._source_truth
_future_truth=fresh._future_truth


def _prediction(root):
    done,cal,flat=evaluation._prediction_gate(root)
    if done.get('method_names')!=list(METHODS) or done.get('primary_method')!=PRIMARY or set(flat)!=set(evaluation.ROW_KEYS):raise ValueError('All six metric decisions must freeze before three truths')
    rows=dict(source=dict(known=flat['source_V'],novel=flat['source_NV']),clean_DURA=flat['Q_clean_DURA'],star_DURA=flat['Q_star_DURA'])
    return done,cal,rows


def _control_gate(main,old):
    gain=main['A_K'] is not None and old['A_K'] is not None and main['A_K']>old['A_K']+1e-12
    kept=main['R_U'] is not None and old['R_U'] is not None and main['R_U']>=old['R_U']-1e-12
    return bool(gain),bool(kept)


def score(prediction_root,source_novel_truth,clean_truth,star_truth,output):
    out=Path(output)
    if out.exists():raise FileExistsError('New immutable PCD-relative metric score root required')
    done,cal,rows=_prediction(prediction_root)
    nt=_source_truth(source_novel_truth,rows['source']['novel'])
    truths={c:_future_truth(path,rows[c]['ids']) for c,path in [('clean_DURA',clean_truth),('star_DURA',star_truth)]}
    for c,t in truths.items():
        if not np.isin(t['scenario'],('clean',) if c=='clean_DURA' else common.SCENES).all():raise ValueError('Future condition and truth scenario differ')
    if any(not np.array_equal(truths['clean_DURA'][k],truths['star_DURA'][k]) for k in ('ids','y','tx','receiver','day')):raise ValueError('Paired future canonical truth differs')
    methods={}
    for index,name in enumerate(METHODS):
        paths={'source_DURA':previous._source(rows['source'],nt,index)}
        paths.update({c:fresh._target(rows[c],truths[c],c,index) for c in CONDITIONS[1:]})
        methods[name]=dict(paths=paths,diagnostic_only=name!=PRIMARY,source_calibration_passed=bool(cal['methods'][name]['constraints_passed']))
    main=methods[PRIMARY]['paths'];paired={}
    for reference in (1,3):
        pairs=_paired(rows,truths,nt,reference)
        for condition in CONDITIONS:pairs[condition]['delta']=previous._delta(main[condition]['overall'],methods[METHODS[reference]]['paths'][condition]['overall'])
        paired[METHODS[reference]]=pairs
    positive,kept=_control_gate(main['star_DURA']['overall'],methods[METHODS[1]]['paths']['star_DURA']['overall'])
    known=bool(main['source_DURA']['known_12_groups_passed'] and cal['methods'][PRIMARY]['constraints_passed'])
    protection=all(main[c]['goals']['passed'] for c in CONDITIONS[1:]);success=bool(known and positive and kept and protection)
    result=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=done['code_commit'],method_names=list(METHODS),primary_method=PRIMARY,conditions=list(CONDITIONS),methods=methods,
        primary_summary={c:main[c]['overall'] for c in CONDITIONS},paired_primary_vs_OLD_PCD_predicted_ratio=paired[METHODS[1]],paired_primary_vs_MD_predicted=paired[METHODS[3]],
        star_A_K_gain_vs_OLD_PCD=paired[METHODS[1]]['star_DURA']['delta']['A_K'],star_A_K_strictly_higher_than_OLD_PCD=positive,star_R_U_not_lower_than_OLD_PCD=kept,
        source_known_12_passed=known,target_protection_passed=bool(protection),metric_readout_success=success,development_gate_passed=success,eligible_primary=success,adaptation_success=None,
        next_step='SOURCE_METRIC_READOUT_PASSED_REQUIRES_SEPARATE_CONFIRMATION' if success else 'NO_PROMOTION',truth_last=True,all_three_predictions_before_truth=True,
        source_truth_reads=1,target_truth_reads=2,U_truth_reads=0,classification_unchanged=True,controls_reused_exactly=True,original_F0_unchanged=True,
        source_metric_fits=1,new_threshold_sets_fitted=4,new_threshold_values=24,threshold_values=36,threshold_bytes=288,new_neural_updates=0,new_GMM_fits=0,
        source_statistic_inputs_only=True,model_source_only=False,PCD_U_ancestor=576,new_target_DA=False,target_REG=False,registration_success=None,A_N=None,H_OW=None,
        metric_state_bytes=done['metric_state_bytes'],prediction_costs={k:v for k,v in done.items() if k.endswith('_reads') or k.endswith('_forward_rows')},
        source_bank_deployed=False,formal_target_accessed=False,full_goal_status='ACTIVE',full_goal_passed=False,final_phase3_complete=False,
        source_RX_seen=True,source_V_is_calibration_not_independent_test=True,source_novel_historically_exposed=True,fresh_development_RX=34,
        global_physical_history='UNKNOWN',new_blind_confirmation=False,
        scope='Frozen F0 identity and inherited U576-adapted PCD; source-L class/global covariance scores, known-V calibration, no new target adaptation or registration')
    out.mkdir(parents=True);_write(out/'results.json',result);_write(out/'complete.json',{k:v for k,v in result.items() if k not in ('methods','paired_primary_vs_OLD_PCD_predicted_ratio','paired_primary_vs_MD_predicted')})
    print(STATUS,result['next_step'],flush=True);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--prediction-root',required=True);p.add_argument('--source-novel-truth',required=True);p.add_argument('--clean-truth',required=True);p.add_argument('--star-truth',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();score(a.prediction_root,a.source_novel_truth,a.clean_truth,a.star_truth,a.output)


if __name__=='__main__':main()
