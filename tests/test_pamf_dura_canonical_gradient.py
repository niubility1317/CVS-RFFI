import pytest
import torch

from experiments.pamf_dura_efficiency.canonical_gradient import gradients,decision


def test_changed_feature_reference_reaches_IQ_without_reference_or_model_gradients():
    class Reader:
        def features(self,x):return torch.cat((x.flatten(1),.3*x.flatten(1).square()),1)
    generator=torch.Generator().manual_seed(31)
    clean=torch.randn(3,2,8,generator=generator);restored=clean+.03*torch.roll(clean,1,-1)
    reader=Reader();reference=reader.features(clean).requires_grad_(True)
    canonical=(reference.detach()+.04).requires_grad_(True)
    terms,gi,gc,go=gradients(restored,clean,reader,reference,canonical)
    assert all(g.shape==clean.shape and torch.isfinite(g).all() and g.norm()>0 for g in (gi,gc,go))
    assert not torch.equal(gc,go) and reference.grad is None and canonical.grad is None
    assert set(terms)=={'iq_nmse','original_feature_error','canonical_feature_error'}


def rows():return [dict(view=v,ratio=4.,cosine=.2,gradient_delta_to_original=.2) for v in (1,20) for _ in range(16)]


def test_recipe_needs_measurable_difference_from_failed_original_Fcos_in_both_views():
    value=rows();result=decision(value)
    assert result['RX_canonical_training_study_eligible'] and result['alpha']==.0625
    for row in value[16:]:row['gradient_delta_to_original']=.09
    assert not decision(value)['RX_canonical_training_study_eligible']
    value=rows()
    for row in value[:16]:row['cosine']=-.1
    assert not decision(value)['RX_canonical_training_study_eligible']
    with pytest.raises(ValueError):decision(rows()[:-1])
