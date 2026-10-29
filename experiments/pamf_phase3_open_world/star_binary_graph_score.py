"""Independent truth-last scoring of frozen binary graph and anonymous states."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

from . import star_reciprocal_score as previous
from . import unlabeled_registration_score as prior
from .scoring import _truth

RUN_ID='20261030-phase3-star-binary-graph-s2026092701-r01'
SCHEMA='pamf_phase3_star_binary_graph_scores_v1'
STATUS='STAR_BINARY_GRAPH_SCORES_COMPLETE'
STATES=previous.STATES
PRIMARY_STATE='DA1_REG1'
SCENES,OLD_LABELS,UNKNOWN_LABELS=previous.SCENES,previous.OLD_LABELS,previous.UNKNOWN_LABELS
_metrics,_goals,_counts,_difference=previous._metrics,previous._goals,previous._counts,previous._difference
_targets,_write=previous._targets,previous._write


def _prediction_gate(root):
    from .star_binary_graph_run import _predictions
    return _predictions(root)


def score(predictions_root,U_truth,Q_truth,targets,output):
    output=Path(output)
    if output.exists():raise FileExistsError('New immutable binary graph score output required')
    done,p,diag,states=_prediction_gate(predictions_root)
    spec=_targets(targets)
    u=_truth(U_truth,p['U_ids'])
    if (not np.all(u['receiver']==26) or np.any((u['y']<0)|(u['y']>7)) or not np.isin(u['scenario'],SCENES).all()):
        raise ValueError('Only the fixed RX24-1 U truth may form scorer-side maps')
    arrivals=sorted(int(v) for v in np.unique(u['y']) if v>=6)
    later=sorted(set(UNKNOWN_LABELS)-set(arrivals))
    mappings,clusters,partitions={},{},{}
    for da in ('DA0','DA1'):
        d=diag[da];k=len(states[da]['centers'])
        mappings[da],clusters[da]=prior._matching(u['y'],d['assignments'],d['candidate_mask'],k,arrivals)
        partitions[da]=previous._components(u['y'],d)
    # All U maps and diagnostics freeze before Q truth; Q cannot rematch slots.
    future=_truth(Q_truth,p['ids']);y=future['y'];scenes=future['scenario']
    if (not np.all(future['receiver']==26) or not np.isin(y,OLD_LABELS+UNKNOWN_LABELS).all() or not np.isin(scenes,SCENES).all()):
        raise ValueError('Only the fixed development future-Q truth may be scored')
    masks=dict(overall=np.ones(len(y),bool),**{'scene/'+s:scenes==s for s in SCENES})
    rows,mapped_states={},{}
    for si,name in enumerate(STATES):
        registered=si>=2;da='DA'+str(si%2);mapping=mappings[da];raw=p['prediction'][si]
        mapped=prior._mapped(raw,mapping) if registered else raw.copy();mapped_states[name]=mapped
        joint=p['joint_margins_'+name];choice=joint.argmax(1).astype(np.int64)
        choice=prior._mapped(choice,mapping) if registered else p['closed_prediction'][si%2]
        actual=sorted(set(UNKNOWN_LABELS)-(set(mapping.values()) if registered else set()))
        def calculate(mask,complete=True):
            return _metrics(y[mask],raw[mask],mapped[mask],p['baseline_closed_prediction'][mask],p['knownness'][si,mask],
                choice[mask],registered,arrivals,later,actual,complete=complete)
        rows[name]=dict(overall=calculate(masks['overall']),by_scenario={s:calculate(masks['scene/'+s]) for s in SCENES},
            by_transmitter={str(c):calculate(y==c,False) for c in OLD_LABELS+UNKNOWN_LABELS},
            by_receiver={'26':calculate(masks['overall'])},by_day={done['Q_day']:calculate(masks['overall'])},
            by_receiver_day={'26/'+done['Q_day']:calculate(masks['overall'])},registered=registered,
            anonymous_count=len(states[da]['centers']) if registered else 0,
            matched_arrival_labels=sorted(mapping.values()) if registered else [],actual_unregistered_labels=actual,
            unknown_rejection_rule='Only raw -1 rejects; unmatched anonymous -2 is accepted wrong',
            new_macro_denominator_rule='All true U arrivals including undiscovered/unmatched; missing Q class N/A; REG0 N/A')
    previous._retention(rows)
    first,second=rows['DA0_REG0']['overall'],rows['DA1_REG0']['overall']
    gain=None if first['A_K'] is None or second['A_K'] is None else second['A_K']-first['A_K']
    checks=dict(positive_old_macro_open_accuracy_gain=gain is not None and gain>0.,DA1_absolute_v3_constraints=rows['DA1_REG0']['goals']['passed'])
    adaptation=dict(checks=checks,old_macro_open_accuracy_gain=gain,passed=all(checks.values()))
    registration={da:rows[da+'_REG1']['goals'] for da in ('DA0','DA1')}
    comparisons=(('DA1_REG0_minus_DA0_REG0','DA1_REG0','DA0_REG0'),('DA1_REG1_minus_DA0_REG1','DA1_REG1','DA0_REG1'),
        ('DA0_REG1_minus_DA0_REG0','DA0_REG1','DA0_REG0'),('DA1_REG1_minus_DA1_REG0','DA1_REG1','DA1_REG0'))
    deltas={key:dict(previous._compare(y,scenes,p,STATES.index(left),STATES.index(right),rows[left],rows[right],
        mapped_states[left],mapped_states[right],later),left_state=left,right_state=right) for key,left,right in comparisons}
    passed=bool(adaptation['passed'] and registration['DA1']['passed'])
    result=dict(status=STATUS,schema=SCHEMA,run_id=RUN_ID,code_commit=done['code_commit'],condition='P1',states=rows,
        primary_state=PRIMARY_STATE,primary_summary=rows[PRIMARY_STATE]['overall'],primary_pass=passed,
        adaptation_goals=adaptation,registration_goals=registration,mappings=clusters,component_diagnostics=partitions,deltas=deltas,
        frozen_rejection_stage_passed=rows['DA0_REG0']['goals']['passed'],
        next_step='READY_FOR_SEPARATE_CONFIRMATION' if passed else 'NO_PROMOTION',
        required_main_scenes=list(SCENES),observed_scenes=sorted(set(scenes.tolist())),future_unknown_labels=list(UNKNOWN_LABELS),
        U_true_arrival_labels=arrivals,planned_later_labels=later,targets_schema=spec['schema'],condition_unknown_rejection_min=.90,
        intended_registered_CK_min=.90,new_accuracy_min=.80,original_F0_loss_max=.02,old_registration_loss_max=.02,
        primary_metrics='True-TX macro A_K, state-specific R_U and harmonic(A_K,R_U); A_N/H_OW separate',
        unknown_denominator_rule='REG0 all [6,7,8]; REG1 planned later outside U; fixed-cohort comparisons also reported',
        development_only=True,truth_last=True,all_U_mappings_frozen_before_Q_truth=True,Q_truth_used_for_cluster_mapping=False,
        algorithm_target_label_inputs=0,true_unknown_K_inputs=0,future_Q_fitting=0,formal_target_reads=0,
        source_parameters_updated=0,identity_head_updates=0,target_graph_fitted_on_U_only=True,
        target_neural_optimizer_updates=0,source_thresholds_updated=0,anonymous_K_may_be_zero=True,anonymous_K_may_differ_by_DA=True,
        deployment_unlabeled_U_memory=True,source_per_record_bank=False,costs=done['costs'],all_new_IQ_satellite=True,
        final_phase3_complete=False,full_goal_status='ACTIVE',automatic_formal_or_multiseed_promotion=False,
        missing_group_policy='N/A and coverage false; no empty or incomplete scene certification',
        U_purity_is_not_future_registration_accuracy=True,day_grouping='Q immutable role date; truth has no per-row day')
    output.mkdir(parents=True,exist_ok=False);_write(output/'scores.json',result)
    _write(output/'complete.json',{k:result[k] for k in ('status','schema','run_id','code_commit','primary_state',
        'primary_summary','primary_pass','next_step','truth_last','full_goal_status','final_phase3_complete')})
    return result


def main(argv=None):
    print(json.dumps(dict(event='STAR_BINARY_GRAPH_SCORE_PROCESS',pid=os.getpid(),cwd=os.getcwd(),argv=list(sys.argv) if argv is None else list(argv))),flush=True)
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--predictions-root',required=True)
    p.add_argument('--U-truth',required=True);p.add_argument('--Q-truth',required=True);p.add_argument('--targets',required=True)
    p.add_argument('--output',required=True);a=p.parse_args(argv)
    result=score(a.predictions_root,a.U_truth,a.Q_truth,a.targets,a.output)
    print(json.dumps(dict(status=result['status'],primary_pass=result['primary_pass'])),flush=True);return 0


if __name__=='__main__':raise SystemExit(main())
