import copy
import pytest
from tools.r17_compact_acceptance import target_acceptance,VIEWS


def metrics(correct=134400,n=168000):
    return dict(role='target',count=n,views={v:dict(count=n,correct=correct,accuracy=100*correct/n) for v in VIEWS})


def test_one_percentage_point_is_inclusive_and_distinct_from_one_percent_relative():
    result=target_acceptance(metrics(132720),metrics())
    assert result['accepted'] and not result['zero_drop']
    assert result['maximum_correct_loss']==1680
    assert result['views']['clean']['accuracy_delta_pp']==-1.
    # 80 -> 79 is permitted; it would fail a 1% relative accuracy rule.
    assert 79<80*.99


def test_one_more_error_in_one_view_fails_despite_improvement_in_other_views():
    original=metrics();candidate=metrics(150000)
    candidate['views']['practical_low_urban']=metrics(132719)['views']['practical_low_urban']
    result=target_acceptance(candidate,original)
    assert not result['accepted'] and not result['views']['practical_low_urban']['accepted']
    assert result['views']['clean']['accepted']


def test_fractional_sample_allowance_uses_integer_counts_without_rounding():
    original=metrics(900,1001)
    assert target_acceptance(metrics(890,1001),original)['accepted']
    assert not target_acceptance(metrics(889,1001),original)['accepted']


@pytest.mark.parametrize('bad',['source','missing_view','wrong_count','rounded_accuracy','noninteger_correct'])
def test_incomplete_or_inconsistent_target_evidence_is_rejected(bad):
    original=metrics();candidate=copy.deepcopy(original)
    if bad=='source':candidate['role']='source'
    elif bad=='missing_view':candidate['views'].pop('clean')
    elif bad=='wrong_count':candidate['views']['clean']['count']-=1
    elif bad=='rounded_accuracy':candidate['views']['clean']['accuracy']=79.999
    else:candidate['views']['clean']['correct']=134400.0
    with pytest.raises(ValueError):target_acceptance(candidate,original)
