"""Synthetic fixed scores only: registration mechanism counterexamples."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from d92_registration_score_diagnostics import diagnose_registration


def case():
    return dict(b_scores=[[3.,1.],[0.,2.]],b_classes=['old-b','old-d'],b_ids=['o1','o2'],
        c_scores=[[3.,1.,4.],[0.,2.,1.]],c_classes=['old-b','old-d','new-a'],c_ids=['o1','o2'],
        held_labels=dict(o1='old-b',o2='old-d'),old_classes=['old-b','old-d'])


@pytest.mark.parametrize('b,c,expected',[
    ([[3,1],[0,2]],[[3,1,4],[0,2,1]],(1.,1.,.5,0.,.5)),  # Frozen old columns, new competition.
    ([[3,1],[0,2]],[[1,3,-1],[0,2,-1]],(1.,.5,.5,.5,0.)), # Old ordering only.
    ([[3,1],[0,2]],[[1,3,-1],[0,2,4]],(1.,.5,0.,.5,.5)),  # Both mechanisms.
    ([[1,3],[0,2]],[[3,1,4],[0,2,-1]],(.5,1.,.5,-.5,.5)), # Refit recovery hidden by competition.
    ([[1,3],[0,2]],[[3,1,-1],[0,2,-1]],(.5,1.,1.,-.5,0.)),# Net improvement, negative old-order loss.
])
def test_exact_decision_decomposition_counterexamples(b,c,expected):
    args=case();args.update(b_scores=b,c_scores=c);result=diagnose_registration(**args)
    ba,oa,ca,ordering,competition=expected
    assert (result['B_acc'],result['C_old_columns_acc'],result['C_full_acc'])==(ba,oa,ca)
    decomposition=result['decomposition']
    assert decomposition['old_order_change']['accuracy_fraction']==ordering
    assert decomposition['new_competition_loss']['accuracy_fraction']==competition>=0
    assert decomposition['total_old_accuracy_drop']['accuracy_fraction']==ba-ca
    assert decomposition['old_order_change']['percentage_points']==100*ordering
    assert decomposition['exact_count_identity'] is True
    left=decomposition['old_order_change']['correct_count_difference']+decomposition['new_competition_loss']['correct_count_difference']
    assert left==decomposition['total_old_accuracy_drop']['correct_count_difference']
    assert result['transitions']['new_competition']['incorrect_to_correct']==0
    for record in result['old_records']:
        assert record['old_order_correct_loss']+record['new_competition_correct_loss']==record['total_correct_loss']
    json.dumps(result,allow_nan=False)


def test_transition_counts_do_not_conflate_changed_wrong_predictions_with_accuracy_loss():
    args=case();args['b_scores']=[[1,3],[0,2]];args['c_scores']=[[1,3,5],[0,2,-1]]
    result=diagnose_registration(**args)
    assert result['old_winner_displaced_by_new_count']==1
    assert result['decomposition']['new_competition_loss']['correct_count_difference']==0
    assert result['transitions']['new_competition']['incorrect_to_incorrect']==1
    assert result['old_records'][0]['C_full_predicted_class']=='new-a'


def test_new_records_explicitly_reported_and_not_used_as_old_denominator():
    args=case();args['c_ids']+=['n1','n2'];args['c_scores']+=[[0,1,5],[4,0,2]]
    args['held_labels'].update(n1='new-a',n2='new-a')
    result=diagnose_registration(**args)
    assert result['old_held_count']==2 and result['additional_new_held_count']==2
    assert result['additional_new_held_explicitly_present'] is True
    assert result['Cnew_acc']==result['C_harmonic_mean']==.5
    assert result['Cnew_correct_count']==1
    assert [r['physical_id'] for r in result['new_records']]==['n1','n2']
    assert result['C_full_acc']==.5


def test_physical_id_and_class_column_permutations_preserve_complete_result():
    args=case();args['c_ids']+=['n1'];args['c_scores']+=[[0,1,5]];args['held_labels']['n1']='new-a'
    expected=diagnose_registration(**args);changed=deepcopy(args)
    changed['b_ids']=args['b_ids'][::-1];changed['b_scores']=[r[::-1] for r in args['b_scores'][::-1]]
    changed['b_classes']=args['b_classes'][::-1];changed['old_classes']=args['old_classes'][::-1]
    order=[2,0,1];columns=[1,2,0]
    changed['c_ids']=[args['c_ids'][i] for i in order]
    changed['c_classes']=[args['c_classes'][i] for i in columns]
    changed['c_scores']=[[args['c_scores'][i][j] for j in columns] for i in order]
    assert diagnose_registration(**changed)==expected


def test_exact_ties_follow_physical_lexical_policy_for_all_three_decisions():
    result=diagnose_registration(b_scores=[[1,1]],b_classes=['z-old','b-old'],b_ids=['id'],
        c_scores=[[1,1,1]],c_classes=['z-old','a-new','b-old'],c_ids=['id'],
        held_labels={'id':'b-old'},old_classes=['z-old','b-old'])
    record=result['old_records'][0]
    assert record['B_predicted_class']==record['C_old_columns_predicted_class']=='b-old'
    assert record['C_full_predicted_class']=='a-new'
    assert record['C_full_top_tie_classes']==['a-new','b-old','z-old']
    assert record['margins']['C_true_old_minus_max_new']==0
    assert record['margins']['C_max_old_minus_max_new']==0
    assert result['decomposition']['new_competition_loss']['percentage_points']==100
    assert all(result['tie_counts'][k]==1 for k in ('B_old_top_tie_records','C_old_columns_top_tie_records',
        'C_full_old_top_tie_records','C_old_new_max_tie_records','C_true_old_max_new_tie_records'))


def test_tie_with_lexically_later_new_class_does_not_displace_old_winner():
    args=case();args['c_scores']=[[3,1,3],[0,2,2]];args['c_classes'][-1]='z-new'
    result=diagnose_registration(**args)
    assert result['C_old_columns_acc']==result['C_full_acc']==1
    assert result['tie_counts']['C_old_new_max_tie_records']==2
    assert result['old_winner_displaced_by_new_count']==0


def test_near_ties_are_not_promoted_to_exact_ties():
    args=case();args['c_scores']=[[3,1,3+1e-12],[0,2,2-1e-12]]
    result=diagnose_registration(**args)
    assert result['C_full_acc']==.5 and result['tie_counts']['C_full_old_top_tie_records']==0


def test_non_tie_class_renaming_preserves_decisions_after_inverse_mapping():
    args=case();before=diagnose_registration(**args);renaming={'old-b':'z','old-d':'a','new-a':'m'}
    for name in ('old_classes','b_classes','c_classes'):args[name]=[renaming[c] for c in args[name]]
    args['held_labels']={i:renaming[c] for i,c in args['held_labels'].items()}
    after=diagnose_registration(**args)
    assert after['decomposition']==before['decomposition']
    for left,right in zip(before['old_records'],after['old_records']):
        for field in ('B_predicted_class','C_old_columns_predicted_class','C_full_predicted_class'):
            assert right[field]==renaming[left[field]]


def test_heads_with_unrelated_positive_affine_scales_compare_only_argmax():
    args=case();before=diagnose_registration(**args)
    args['b_scores']=[[1000*v+500 for v in row] for row in args['b_scores']]
    args['c_scores']=[[.01*v-9 for v in row] for row in args['c_scores']]
    after=diagnose_registration(**args)
    assert after['decomposition']==before['decomposition'] and after['transitions']==before['transitions']
    assert after['old_records'][0]['margins']['C_true_old_minus_max_new']==pytest.approx(-.01)
    assert 'never subtracted' in after['margin_scope']


def test_n0_same_old_registry_has_exact_zero_competition_even_with_refit():
    args=case();args['c_classes']=['old-d','old-b'];args['c_scores']=[[3,1],[2,0]]
    result=diagnose_registration(**args)
    assert result['new_classes']==[] and result['Cnew_acc'] is result['C_harmonic_mean'] is None
    assert result['C_old_columns_acc']==result['C_full_acc']==.5
    assert result['decomposition']['new_competition_loss']['correct_count_difference']==0
    for row in result['old_records']:
        assert row['C_old_columns_predicted_class']==row['C_full_predicted_class']
        assert row['margins']['C_true_old_minus_max_new'] is None
        assert row['margin_unavailable_reasons']['C_true_old_minus_max_new']=='NO_COMPARISON_CLASS'


def test_one_old_class_has_no_other_old_margin_and_correct_new_competition():
    result=diagnose_registration(b_scores=[[2]],b_classes=['old'],b_ids=['o'],
        c_scores=[[2,3]],c_classes=['old','new'],c_ids=['o'],held_labels={'o':'old'},old_classes=['old'])
    assert result['B_acc']==result['C_old_columns_acc']==1 and result['C_full_acc']==0
    assert result['old_records'][0]['margins']['B_true_minus_max_other_old'] is None


@pytest.mark.parametrize('fault',['missing_b_id','missing_c_id','duplicate_b_id','duplicate_c_id','missing_label',
    'extra_label','unknown_label','b_new_label','missing_c_old','b_extra_class','duplicate_class',
    'nan','infinity','wrong_rows','wrong_columns','scalar_scores','string_score','boolean_score','complex_score',
    'empty_old','empty_b','empty_c'])
def test_invalid_inputs_fail_without_silent_drops_or_zero_fills(fault):
    args=case()
    if fault=='missing_b_id':args['b_ids']=args['b_ids'][:1];args['b_scores']=args['b_scores'][:1]
    elif fault=='missing_c_id':args['c_ids'][0]='different';args['held_labels']['different']=args['held_labels'].pop('o1')
    elif fault=='duplicate_b_id':args['b_ids'][1]=args['b_ids'][0]
    elif fault=='duplicate_c_id':args['c_ids'][1]=args['c_ids'][0]
    elif fault=='missing_label':args['held_labels'].pop('o1')
    elif fault=='extra_label':args['held_labels']['extra']='old-b'
    elif fault=='unknown_label':args['held_labels']['o1']='absent'
    elif fault=='b_new_label':args['held_labels']['o1']='new-a'
    elif fault=='missing_c_old':args['c_classes'][0]='other'
    elif fault=='b_extra_class':args['b_classes'].append('new-a')
    elif fault=='duplicate_class':args['c_classes'][1]=args['c_classes'][0]
    elif fault=='nan':args['b_scores'][0][0]=float('nan')
    elif fault=='infinity':args['c_scores'][0][0]=float('inf')
    elif fault=='wrong_rows':args['c_scores'].pop()
    elif fault=='wrong_columns':args['b_scores'][0].pop()
    elif fault=='scalar_scores':args['c_scores']=1
    elif fault=='string_score':args['c_scores'][0][0]='1'
    elif fault=='boolean_score':args['c_scores'][0][0]=True
    elif fault=='complex_score':args['c_scores'][0][0]=1j
    elif fault=='empty_old':args['old_classes']=[]
    elif fault=='empty_b':args['b_ids']=[];args['b_scores']=[]
    elif fault=='empty_c':args['c_ids']=[];args['c_scores']=[];args['held_labels']={}
    with pytest.raises(ValueError):diagnose_registration(**args)


def test_extreme_finite_scores_preserve_decision_and_explicit_margin_overflow():
    args=case();args['b_scores'][0]=[1e308,-1e308];args['c_scores'][0]=[1e308,-1e308,-1e308]
    result=diagnose_registration(**args);row=result['old_records'][0]
    assert row['B_correct'] and row['C_full_correct']
    assert row['margins']['C_true_old_minus_max_new'] is None
    assert row['margin_unavailable_reasons']['C_true_old_minus_max_new']=='DIFFERENCE_NOT_REPRESENTABLE_AS_FINITE_FLOAT'
    json.dumps(result,allow_nan=False)


def test_inputs_unchanged_and_analysis_boundary_is_explicit():
    args=case();before=deepcopy(args);result=diagnose_registration(**args)
    assert args==before
    assert result['fit_performed'] is False and result['analysis_only'] is True
    assert result['query_role_mask_for_deployment_allowed'] is False
    assert result['accuracy_unit']=='fraction' and 'offline' in result['limitation'].lower()
    assert 'causal' in result['limitation']
