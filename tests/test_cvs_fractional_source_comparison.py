import copy
import pytest
from experiments.cvs_fractional_identity.compare_prior_source import paired_summary,SEEDS,VARIANTS,PRIOR_VARIANT

def fixture():
    prior=[dict(variant=PRIOR_VARIANT,seed=s,parameters=202553,accuracy=.9,worst_rx=.8) for s in sorted(SEEDS)]
    new=[dict(source_record=dict(variant=v,seed=s,parameters=202553+int(v=='fractional_learned'),accuracy=.91,worst_rx=.82),epochs=[dict(alignment_strength=.5)]) for v in VARIANTS for s in sorted(SEEDS)]
    return new,prior

def test_paired_comparison_preserves_input_and_percentage_point_units():
    new,prior=fixture();before=copy.deepcopy((new,prior));rows,summary=paired_summary(new,prior)
    assert len(rows)==8 and len(summary)==2 and (new,prior)==before
    assert all(r['V_delta_pp_mean']==pytest.approx(1.) and r['worst_RX_delta_pp_mean']==pytest.approx(2.) for r in summary)

@pytest.mark.parametrize('corruption',['missing_old','duplicate_new','wrong_core_count'])
def test_unmatched_source_comparison_is_rejected(corruption):
    new,prior=fixture()
    if corruption=='missing_old':prior.pop()
    elif corruption=='duplicate_new':new[-1]=copy.deepcopy(new[0])
    else:prior[0]['parameters']+=1
    with pytest.raises(ValueError):paired_summary(new,prior)
