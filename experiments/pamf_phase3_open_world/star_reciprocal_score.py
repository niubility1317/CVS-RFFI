"""Truth-last reciprocal adaptation and anonymous-registration diagnostics.

The producer validates all public artifacts first. Each arm/DA gets its own
U-only Hungarian map; all four maps freeze before any future-Q truth is read.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

from . import star_neighborhood_score as base
from . import unlabeled_registration_score as prior
from .scoring import _truth

RUN_ID='20261030-phase3-star-reciprocal-adapter-s2026092701-r01'
SCHEMA='pamf_phase3_star_reciprocal_scores_v1'
STATUS='STAR_RECIPROCAL_SCORES_COMPLETE'
ARMS=('self_pl','reciprocal')
STATES=base.STATES
PRIMARY_ARM='reciprocal'
PRIMARY_STATE='DA1_REG1'
SCENES,OLD_LABELS,UNKNOWN_LABELS=base.SCENES,base.OLD_LABELS,base.UNKNOWN_LABELS
_metrics,_goals,_counts,_difference=base._metrics,base._goals,base._counts,base._difference
_targets,_write=base._targets,base._write


def _prediction_gate(root):
    from .star_reciprocal_run import _predictions
    return _predictions(root)


def _components(y,diag):
    registered={int(c):i for i,c in enumerate(diag['registered_component_indices'])};rows=[]
    for c,n in enumerate(diag['component_sizes']):
        mask=diag['component_labels']==c;labels,counts=np.unique(y[mask],return_counts=True)
        old=int(np.sum(mask&(y<6)));n=int(n)
        rows.append(dict(component_index=c,count=n,composition={str(int(k)):int(v) for k,v in zip(labels,counts)},
            old_count=old,old_contamination=old/n if n else None,purity=int(counts.max())/n if n else None,
            registered_anonymous_index=registered.get(c),public_component=diag['components'][c]))
    return dict(component_count=len(rows),components=rows,unassigned_component_rows=int(np.sum(diag['component_labels']<0)),
        truth_side_only=True,not_true_K_input=True,future_registration_accuracy=False)


def _pseudo(y,pseudo):
    result={}
    for key in ('selected','trusted','expanded'):
        mask=pseudo[key];n=int(mask.sum());old=mask&(y<6);unknown=mask&(y>=6);no=int(old.sum());nu=int(unknown.sum())
        correct=int(np.sum(old&(pseudo['labels']==y)))
        result[key]=dict(count=n,old_count=no,unknown_count=nu,old_fraction=no/n if n else None,
            unknown_fraction=nu/n if n else None,old_correct_pseudo_labels=correct,
            old_label_accuracy=correct/no if no else None,correct_old_fraction_of_selected=correct/n if n else None)
    return dict(result,truth_side_only=True,used_for_algorithm=False,future_registration_accuracy=False)


def _retention(rows):
    for da in ('DA0','DA1'):
        current,previous=rows[da+'_REG1'],rows[da+'_REG0']
        for group in ('overall','by_scenario','by_transmitter','by_receiver','by_day','by_receiver_day'):
            pairs=[(current[group],previous[group])] if group=='overall' else [(v,previous[group][key]) for key,v in current[group].items()]
            for a,b in pairs:
                a['old_loss_from_same_DA_REG0']=None if a['A_K'] is None or b['A_K'] is None else b['A_K']-a['A_K']
        previous['goals']=_goals(previous,False);current['goals']=_goals(current,True)


def _compare(y,scenes,p,left_index,right_index,left,right,mapped_left,mapped_right,later):
    a,b=p['prediction'][left_index],p['prediction'][right_index]
    counts=_counts(y,a,b,mapped_left,mapped_right,UNKNOWN_LABELS)
    same=left['overall']['R_U_labels']==right['overall']['R_U_labels']
    actual=_counts(y,a,b,mapped_left,mapped_right,left['overall']['R_U_labels'])
    return dict(_difference(left['overall'],right['overall']),counts=counts,
        unknown_rejected_net=actual['unknown_rejected_net'] if same else None,
        unknown_rejected_net_scope='same main R_U roster' if same else 'N/A: REG changes R_U roster; use fixed-cohort counts',
        R_U_rosters=dict(left=left['overall']['R_U_labels'],right=right['overall']['R_U_labels']),primary_R_U_and_H_comparable=same,
        fixed_cohort_counts=dict(all_future_nonold=counts,planned_later=_counts(y,a,b,mapped_left,mapped_right,later)),
        by_scenario={s:dict(delta=_difference(left['by_scenario'][s],right['by_scenario'][s]),
            counts=_counts(y[scenes==s],a[scenes==s],b[scenes==s],mapped_left[scenes==s],mapped_right[scenes==s],UNKNOWN_LABELS)) for s in SCENES})


def score(predictions_root,U_truth,Q_truth,targets,output):
    output=Path(output)
    if output.exists():raise FileExistsError('New immutable reciprocal score output required; never overwrite')
    done,p,diag,states,pseudo=_prediction_gate(predictions_root)
    spec=_targets(targets)
    # Public validation is owned by the producer's complete artifact loader.
    # Do not inherit neighborhood K equality, fusion-applied or partition gates.
    u=_truth(U_truth,p['U_ids'])
    if (not np.all(u['receiver']==26) or np.any((u['y']<0)|(u['y']>7)) or not np.isin(u['scenario'],SCENES).all()):
        raise ValueError('Only fixed RX24-1 U truth may fit independent scorer-side anonymous maps')
    arrivals=sorted(int(v) for v in np.unique(u['y']) if v>=6)
    later=sorted(set(UNKNOWN_LABELS)-set(arrivals));mappings={};clusters={};partitions={}
    for arm in ARMS:
        mappings[arm]={};clusters[arm]={};partitions[arm]={}
        for da in ('DA0','DA1'):
            k=len(states[arm][da]['centers']);d=diag[arm][da]
            mappings[arm][da],clusters[arm][da]=prior._matching(u['y'],d['assignments'],d['candidate_mask'],k,arrivals)
            partitions[arm][da]=_components(u['y'],d)
    pseudo_diagnostics={arm:_pseudo(u['y'],pseudo[arm]) for arm in ARMS}
    # No future labels enter any of the four already-fixed U assignments/maps.
    future=_truth(Q_truth,p['ids']);y=future['y'];scenes=future['scenario']
    if (not np.all(future['receiver']==26) or not np.isin(y,OLD_LABELS+UNKNOWN_LABELS).all() or not np.isin(scenes,SCENES).all()):
        raise ValueError('Only fixed RX24-1 future-Q truth is allowed; formal target is excluded')
    masks=dict(overall=np.ones(len(y),bool),**{'scene/'+s:scenes==s for s in SCENES})
    arms={};mapped_states={}
    comparisons=(('DA1_REG0_minus_DA0_REG0','DA1_REG0','DA0_REG0'),('DA1_REG1_minus_DA0_REG1','DA1_REG1','DA0_REG1'),
        ('DA0_REG1_minus_DA0_REG0','DA0_REG1','DA0_REG0'),('DA1_REG1_minus_DA1_REG0','DA1_REG1','DA1_REG0'))
    for ai,arm in enumerate(ARMS):
        rows={};mapped_states[arm]={}
        for si,name in enumerate(STATES):
            registered=si>=2;da='DA'+str(si%2);mapping=mappings[arm][da];raw=p['prediction'][ai,si]
            mapped=prior._mapped(raw,mapping) if registered else raw.copy();mapped_states[arm][name]=mapped
            joint=p['joint_'+arm+'_'+name];choice=joint.argmax(1).astype(np.int64)
            choice=prior._mapped(choice,mapping) if registered else p['closed_prediction'][ai,si%2]
            actual=sorted(set(UNKNOWN_LABELS)-(set(mapping.values()) if registered else set()))
            def calculate(mask,complete=True):
                return _metrics(y[mask],raw[mask],mapped[mask],p['baseline_closed_prediction'][mask],p['knownness'][ai,si,mask],
                    choice[mask],registered,arrivals,later,actual,complete=complete)
            rows[name]=dict(overall=calculate(masks['overall']),by_scenario={s:calculate(masks['scene/'+s]) for s in SCENES},
                by_transmitter={str(c):calculate(y==c,False) for c in OLD_LABELS+UNKNOWN_LABELS},
                by_receiver={'26':calculate(masks['overall'])},by_day={done['Q_day']:calculate(masks['overall'])},
                by_receiver_day={'26/'+done['Q_day']:calculate(masks['overall'])},registered=registered,anonymous_count=len(states[arm][da]['centers']) if registered else 0,
                matched_arrival_labels=sorted(mapping.values()) if registered else [],actual_unregistered_labels=actual,
                unknown_rejection_rule='Only raw -1 rejects; unmatched anonymous -2 is accepted wrong',
                new_macro_denominator_rule='All true U arrivals including undiscovered/unmatched; missing Q class is N/A; REG0 N/A')
        _retention(rows)
        first,second=rows['DA0_REG0']['overall'],rows['DA1_REG0']['overall']
        gain=None if first['A_K'] is None or second['A_K'] is None else second['A_K']-first['A_K']
        checks=dict(positive_old_macro_open_accuracy_gain=gain is not None and gain>0.,
            DA1_absolute_v3_constraints=rows['DA1_REG0']['goals']['passed'])
        adaptation=dict(checks=checks,old_macro_open_accuracy_gain=gain,passed=all(checks.values()))
        registration={da:rows[da+'_REG1']['goals'] for da in ('DA0','DA1')}
        deltas={key:dict(_compare(y,scenes,p,(ai,STATES.index(left)),(ai,STATES.index(right)),rows[left],rows[right],
            mapped_states[arm][left],mapped_states[arm][right],later),left_state=left,right_state=right) for key,left,right in comparisons}
        arms[arm]=dict(states=rows,adaptation_goals=adaptation,registration_goals=registration,
            frozen_rejection_stage_passed=rows['DA0_REG0']['goals']['passed'],
            joint_pass=bool(adaptation['passed'] and registration['DA1']['passed']),mappings=clusters[arm],
            component_diagnostics=partitions[arm],pseudo_label_diagnostics=pseudo_diagnostics[arm],deltas=deltas)
    main=arms[PRIMARY_ARM];passed=main['joint_pass']
    paired={name:dict(_compare(y,scenes,p,(1,STATES.index(name)),(0,STATES.index(name)),arms['reciprocal']['states'][name],
        arms['self_pl']['states'][name],mapped_states['reciprocal'][name],mapped_states['self_pl'][name],later),
        left_arm='reciprocal',right_arm='self_pl',state=name) for name in ('DA1_REG0','DA1_REG1')}
    result=dict(status=STATUS,schema=SCHEMA,run_id=RUN_ID,code_commit=done['code_commit'],condition='P1',arms=arms,
        primary_arm=PRIMARY_ARM,primary_state=PRIMARY_STATE,primary_summary=main['states'][PRIMARY_STATE]['overall'],primary_pass=passed,
        frozen_rejection_stage_passed=main['frozen_rejection_stage_passed'],
        next_step='READY_FOR_SEPARATE_CONFIRMATION' if passed else 'NO_PROMOTION',paired_main_minus_control=paired,
        required_main_scenes=list(SCENES),observed_scenes=sorted(set(scenes.tolist())),future_unknown_labels=list(UNKNOWN_LABELS),
        U_true_arrival_labels=arrivals,planned_later_labels=later,targets_schema=spec['schema'],condition_unknown_rejection_min=.90,
        intended_registered_CK_min=.90,new_accuracy_min=.80,original_F0_loss_max=.02,old_registration_loss_max=.02,
        primary_metrics='True-TX macro A_K, state-specific R_U and harmonic(A_K,R_U); A_N/H_OW separately',
        unknown_denominator_rule='REG0 all [6,7,8]; REG1 planned later outside U, with actual-unregistered diagnostics',
        curve_correctness='Actual current DA old head for REG0; U-mapped joint winner for REG1; original F0 only protection baseline',
        development_only=True,truth_last=True,all_four_U_mappings_frozen_before_Q_truth=True,Q_truth_used_for_cluster_mapping=False,
        algorithm_target_label_inputs=0,true_unknown_K_inputs=0,future_Q_fitting=0,formal_target_reads=0,
        source_parameters_updated=0,identity_head_updates=0,target_adapter_fitted_on_U_only=True,
        source_thresholds_updated=0,anonymous_K_may_be_zero=True,anonymous_K_may_differ_by_DA=True,
        costs=done['costs'],all_new_IQ_satellite=True,final_phase3_complete=False,full_goal_status='ACTIVE',
        automatic_formal_or_multiseed_promotion=False,missing_group_policy='N/A and coverage false; no empty or incomplete scene certification',
        U_purity_is_not_future_registration_accuracy=True,day_grouping='Q immutable role date; truth has no per-row day')
    output.mkdir(parents=True,exist_ok=False);_write(output/'scores.json',result)
    _write(output/'complete.json',{k:result[k] for k in ('status','schema','run_id','code_commit','primary_arm','primary_state',
        'primary_summary','primary_pass','next_step','truth_last','full_goal_status','final_phase3_complete')})
    return result


def main(argv=None):
    print(json.dumps(dict(event='STAR_RECIPROCAL_SCORE_PROCESS',pid=os.getpid(),cwd=os.getcwd(),argv=list(sys.argv) if argv is None else list(argv))),flush=True)
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--predictions-root',required=True)
    parser.add_argument('--U-truth',required=True);parser.add_argument('--Q-truth',required=True)
    parser.add_argument('--targets',required=True);parser.add_argument('--output',required=True);a=parser.parse_args(argv)
    result=score(a.predictions_root,a.U_truth,a.Q_truth,a.targets,a.output)
    print(json.dumps(dict(status=result['status'],primary_pass=result['primary_pass'])),flush=True);return 0


if __name__=='__main__':raise SystemExit(main())
