import math
import pytest
from experiments.pamf_ssl_research.source_cap_counterfactual import group_counterfactual,analyze_segment,pool


def values(L=1.,U=3.,A=1.,before=1.5,after=.5,capped=1,tiny=0):
    return dict(L_norm=L,U_norm=U,applied_U_norm=A,dot_before=before,dot_after=after,
        conflict=int(before<0),capped=capped,tiny_L=tiny,conflicts=int(before<0),caps=capped)


def test_aligned_capped_gradient_keeps_its_direction_and_doubles_only_the_bound():
    q=group_counterfactual(values())
    assert q['cap1_applied_U_norm']==1 and q['cap2_applied_U_norm']==2
    assert q['cap2_dot_after']==1 and q['cap2_combined_gradient_norm_squared']==7
    assert q['cap1_capped']==q['cap2_capped']==1


def test_conflicting_component_is_still_removed_before_both_caps():
    q=group_counterfactual(values(U=math.sqrt(13),before=-2,after=0))
    assert q['cap2_applied_U_norm']==2 and q['cap2_dot_after']==0
    assert q['cap2_combined_gradient_norm_squared']==5


def test_uncapped_or_antiparallel_gradients_are_not_amplified():
    for v in (values(U=math.sqrt(.2),A=math.sqrt(.2),before=.2,after=.2,capped=0),
              values(U=2,A=0,before=-2,after=0,capped=0)):
        q=group_counterfactual(v)
        assert q['cap2_applied_U_norm']==q['cap1_applied_U_norm']


def test_new_bound_can_become_uncapped_without_forcing_a_factor_two():
    q=group_counterfactual(values(U=1.5,before=.75,after=.5))
    assert q['cap2_applied_U_norm']==1.5 and q['cap2_capped']==0 and q['cap2_dot_after']==.75


def test_tiny_L_remains_zero_U_under_the_same_exact_boundary():
    q=group_counterfactual(values(L=1e-7,U=3,A=0,before=0,after=0,capped=0,tiny=1))
    assert q['cap2_applied_U_norm']==0


@pytest.mark.parametrize('change',[dict(U_norm=float('nan')),dict(applied_U_norm=2),dict(tiny_L=1),dict(dot_after=-1)])
def test_nonfinite_or_incompatible_logged_guard_facts_are_rejected(change):
    v=values();v.update(change)
    with pytest.raises(ValueError):group_counterfactual(v)


def test_missing_native_records_and_wrong_budget_cannot_form_complete_source_evidence():
    with pytest.raises(ValueError,match='Complete closed'):analyze_segment([],2220,20)
    with pytest.raises(ValueError,match='Complete originalE20'):analyze_segment([],4440,20)
    with pytest.raises(ValueError,match='Both complete'):pool([])
