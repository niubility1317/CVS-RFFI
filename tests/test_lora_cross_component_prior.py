import torch

from experiments.pamf_lora_main.cross_component_prior import (
    Regression, component_features, context_center, contextual_private,
    fit_regression, ordered_context, private_spectrum, shrink_order_features,
)


def test_ordered_context_holds_out_each_chirp():
    value=torch.randn(2,6,16,dtype=torch.complex64)
    original=ordered_context(value)
    changed=value.clone()
    changed[:,2]+=4+3j
    torch.testing.assert_close(ordered_context(changed)[:,2],original[:,2])
    assert (ordered_context(changed)[:,3]-original[:,3]).abs().mean()>.001


def test_ordered_context_finds_linear_trend_without_common_term():
    index=torch.arange(6,dtype=torch.float32)-2.5
    wave=torch.tensor([1+2j,3+4j,1-3j,2j])
    packet=(index[:,None]*wave[None]+(7-4j))[None]
    expected=(index[:,None]*wave[None])[None]
    torch.testing.assert_close(ordered_context(packet),expected)


def test_contextual_prior_preserves_shared_spectrum_and_keeps_private_zero_sum():
    received=torch.randn(3,6,16,dtype=torch.complex64)
    base=torch.randn_like(received)
    center=private_spectrum(torch.randn(3,6,6,dtype=torch.complex64))
    support=torch.arange(16)<6
    gain=torch.rand(3,6,6)
    recovered=contextual_private(received,base,support,center,gain)
    torch.testing.assert_close(recovered.mean(1),base.mean(1),atol=2e-7,rtol=1e-5)
    torch.testing.assert_close(recovered[...,~support],base[...,~support])


def test_fixed_ridge_can_predict_spectrum_without_using_holdout_data():
    gen=torch.Generator().manual_seed(621)
    features=torch.randn(120,6,8,generator=gen)
    transform=torch.randn(6,8,5,generator=gen)
    targets=(features.transpose(0,1)@transform).transpose(0,1)
    model=fit_regression(features[:100],targets[:100],ridge=.01)
    recovered=model.predict(features[100:])
    assert (recovered-targets[100:]).square().mean()<.01


def test_order_observation_cannot_reenter_its_own_prior_via_other_chirp_predictions():
    spectrum=torch.zeros((2,6,4),dtype=torch.complex64)
    mask=torch.ones(4,dtype=torch.bool)
    base=spectrum.clone()
    changed=spectrum.clone()
    changed[:,2]=8+3j
    clean_features=component_features(base,mask,mask,order_spectrum=base)[1]
    disturbed_features=component_features(base,mask,mask,order_spectrum=changed)[1]
    coefficient=torch.randn((6,16,8),generator=torch.Generator().manual_seed(21))
    predictor=Regression(torch.zeros((6,16)),torch.ones((6,16)),torch.zeros((6,8)),coefficient)
    before=context_center(predictor,clean_features)
    after=context_center(predictor,disturbed_features)
    torch.testing.assert_close(after[:,2],before[:,2])
    assert after.abs().max()>.1


def test_rank_deficient_wide_ridge_does_not_require_batched_lu(monkeypatch):
    def unavailable_lu(*_args, **_kwargs):
        raise RuntimeError('Batched LU unavailable')

    monkeypatch.setattr(torch.linalg, 'solve', unavailable_lu)
    gen=torch.Generator().manual_seed(81)
    x=torch.randn((25,6,384),generator=gen)
    y=torch.randn((25,6,128),generator=gen)
    estimator=fit_regression(x,y,ridge=.10)
    prediction=estimator.predict(x)
    assert torch.isfinite(prediction).all()
    assert (prediction-y).square().mean()<.001


def test_order_confidence_uses_other_chirp_noise_and_preserves_public_context():
    features=torch.ones((3,6,10))
    estimator=Regression(torch.zeros((6,10)),torch.ones((6,10)),torch.zeros((6,4)),torch.empty((6,10,4)))
    quiet=torch.zeros((3,6,1))
    torch.testing.assert_close(shrink_order_features(features,estimator,quiet,order_width=4),features)
    noisy=quiet.clone()
    noisy[:,2]=2.
    result=shrink_order_features(features,estimator,noisy,order_width=4)
    torch.testing.assert_close(result[:,:,:6],features[:,:,:6])
    torch.testing.assert_close(result[:,2],features[:,2])
    assert result[:,0,6:].max()<1.
    assert result.min()>=0.
