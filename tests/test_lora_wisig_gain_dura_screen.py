import numpy as np
import torch

from experiments.pamf_lora_main.gain_dura_screen import (
    COUNT, MODES, SCREEN_VIEWS, WiSigRealGain, score_panel, screen_rows,
)
from experiments.pamf_lora_main.gain_dura_fusion import fuse


class Identity:
    def __call__(self, x):
        return x

    adjoint = __call__


def test_fusion_uses_the_companion_in_a_new_direction_and_caps_edit():
    observed=torch.zeros(2,16,dtype=torch.complex64)
    anchor=observed.clone()
    anchor[0,0]=1+1j
    companion=anchor.clone()
    companion[:,0]+=3j
    companion[:,1]=4
    estimate=fuse(observed,anchor,companion,Identity(),torch.ones(2))
    delta=estimate-anchor
    assert 0<delta[0,1].real<4 and delta[0,0].abs()<1e-6
    assert delta[0].norm()<=.25*anchor[0].norm()+1e-6
    assert torch.equal(estimate[1],anchor[1])


def test_single_receiver_panel_is_stable_and_does_not_select_excluded_rows():
    ids=np.array([str(i) for i in range(3*COUNT)])
    rx=np.repeat((1,2,3),COUNT)
    rows=screen_rows(ids,rx,2,excluded=[COUNT],limit=COUNT-1)
    assert len(rows)==COUNT-1 and np.all(rx[rows]==2) and COUNT not in rows
    assert np.array_equal(rows,screen_rows(ids,rx,2,excluded=[COUNT],limit=COUNT-1))


def test_clean_noninferiority_is_required_even_when_paired_gain_is_clear():
    labels=np.zeros(COUNT,dtype=np.int16)
    predictions=np.zeros((len(SCREEN_VIEWS),len(MODES),COUNT),dtype=np.int16)
    predictions[1,:-1,:24]=1
    assert score_panel(predictions,labels)['promote']
    predictions[0,-1,0]=1
    assert not score_panel(predictions,labels)['promote']


def test_wisig_real_gain_uses_the_same_shape_as_its_observation():
    x=torch.randn(2,256,dtype=torch.complex64)
    model=WiSigRealGain()
    p,r=model(x)
    torch.testing.assert_close(r,p,rtol=0,atol=0)
    assert r.shape==x.shape and r.is_complex()
