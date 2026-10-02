import copy
import pytest
from experiments.cvs_energy_identity.compare_prior_source import paired_summary,SEEDS,VARIANTS,PRIOR_VARIANT
def fixture():
    prior=[dict(variant=PRIOR_VARIANT,seed=s,parameters=202553,accuracy=.9,worst_rx=.8) for s in sorted(SEEDS)]
    new=[dict(source_record=dict(variant=v,seed=s,parameters=202553,accuracy=.91,worst_rx=.82),epochs=[dict(alignment_strength=.5 if v=='energy_half' else 0.)]) for v in VARIANTS for s in sorted(SEEDS)]
    return new,prior
def test_matched_half_summary_preserves_inputs_and_percentage_points():
    new,prior=fixture();before=copy.deepcopy((new,prior));rows,summary=paired_summary(new,prior)
    assert len(rows)==4 and len(summary)==1 and (new,prior)==before
    assert all(r['alignment_strength']==r['prior_alignment_strength']==.5 for r in rows)
    assert summary[0]['V_delta_pp_mean']==pytest.approx(1.) and summary[0]['worst_RX_delta_pp_mean']==pytest.approx(2.)
@pytest.mark.parametrize('corruption',['missing_old','duplicate_new','wrong_core_count','wrong_alpha'])
def test_unmatched_normalization_comparison_rejected(corruption):
    new,prior=fixture()
    if corruption=='missing_old':prior.pop()
    elif corruption=='duplicate_new':new[-1]=copy.deepcopy(new[0])
    elif corruption=='wrong_core_count':prior[0]['parameters']+=1
    else:next(r for r in new if r['source_record']['variant']=='energy_half')['epochs'][-1]['alignment_strength']=.4
    with pytest.raises(ValueError):paired_summary(new,prior)
