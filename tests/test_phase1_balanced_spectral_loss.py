import torch
import numpy as np
import pytest
from experiments.phase1_dura_orthogonal.balanced_spectral_loss import balanced_spectral_loss


def test_bands_are_balanced_and_loss_is_scale_invariant():
    torch.manual_seed(80)
    clean=torch.randn(3,2,256)
    output=(clean+.1*torch.randn_like(clean)).requires_grad_()
    loss=balanced_spectral_loss(output,clean)
    assert loss > 0
    torch.testing.assert_close(loss,balanced_spectral_loss(output*3,clean*3))
    assert balanced_spectral_loss(clean,clean).item() == 0
    loss.backward()
    assert torch.isfinite(output.grad).all() and output.grad.abs().sum() > 0


def test_empty_band_has_bounded_denominator():
    clean=torch.zeros(1,2,256)
    clean[:,0]=1
    output=clean.clone()
    output[0,0,0]+=.1
    assert torch.isfinite(balanced_spectral_loss(output,clean))


def test_frozen_evaluation_requires_matching_source_objective(monkeypatch, tmp_path):
    from experiments.phase1_dura_orthogonal import ground
    from experiments.phase1_dura_orthogonal.multiscale_spectral import DuraMultiscaleSpectral
    model = DuraMultiscaleSpectral(width=32,taps=9)
    common = dict(commit='source-commit',epochs=5,architecture='dura_multiscale',
        objective='wide_balanced_spectral_guard',TX_labels_read=0,F_weights_read=0,F_outputs_read=0,
        V_reads=0,target_reads=0,DURA_frozen=True,source_fit_size=32768,DURA_training_reads=1,
        source_spectral_weight=.25,source_spectral_floor_fraction=.01,
        source_spectral_bands=[[0.,.125],[.125,.25],[.25,.375],[.375,.501]],
        adaptive_filter_taps=9,noise_gate_low=ground.GATE_LOG10_LOW,
        noise_gate_high=ground.GATE_LOG10_HIGH,spectral_freq_bins=17)
    summary=dict(status='SOURCE_TRAINED',fit_ids=32768,**common)
    state=dict(schema=ground.SCHEMA,fit_ids=np.arange(32768),width=32,model=model.state_dict(),**common)
    monkeypatch.setattr(ground.core,'read',lambda _:summary)
    monkeypatch.setattr(ground.torch,'load',lambda *args,**kwargs:state)
    monkeypatch.setattr(ground,'frozen_dura',lambda *args:model)
    monkeypatch.setattr(ground.core,'teacher',lambda *args:(model,None))
    loaded,_,_=ground.load_fixed(tmp_path,'source-run','source-commit')
    assert isinstance(loaded,DuraMultiscaleSpectral)
    state['source_spectral_floor_fraction']=.02
    with pytest.raises(ValueError,match='spectral'):
        ground.load_fixed(tmp_path,'source-run','source-commit')
