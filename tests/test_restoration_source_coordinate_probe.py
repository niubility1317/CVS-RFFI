import numpy as np
import pytest
import torch

from experiments.pamf_phase3_restoration.source_coordinate_probe import (
    counterfactual_readout,
    rotate_iq,
    select_indices,
    validate_baseline,
)


def test_quarter_rotations_form_exact_group_and_invert_without_iq_loss():
    x=torch.randn(3,2,256)
    for a in range(4):
        torch.testing.assert_close(rotate_iq(rotate_iq(x,a),(-a)%4),x,rtol=0,atol=0)
        for b in range(4):
            torch.testing.assert_close(rotate_iq(rotate_iq(x,a),b),rotate_iq(x,(a+b)%4),rtol=0,atol=0)


def test_counterfactual_maps_identity_back_to_original_frame():
    x=torch.randn(2,2,256)
    for k in range(4):
        torch.testing.assert_close(counterfactual_readout(lambda y:y,x,k),x,rtol=0,atol=0)


def test_directional_bias_is_detected_without_averaging_or_angle_selection():
    x=torch.zeros(2,2,256)
    bias=torch.zeros_like(x);bias[:,0]=1.
    model=lambda y:y+bias
    for k in range(4):
        torch.testing.assert_close(counterfactual_readout(model,x,k),rotate_iq(bias,(-k)%4),rtol=0,atol=0)
    assert not torch.equal(counterfactual_readout(model,x,0),counterfactual_readout(model,x,1))


def test_fixed_panel_covers_all_source_rx_day_cells_without_identity_labels():
    rx=np.repeat([1,3,4,6,8],360)
    day=np.tile(np.repeat([1,2,3],120),5)
    chosen=select_indices(rx,day)
    assert len(chosen)==len(set(chosen.tolist()))==480
    for r in np.unique(rx):
        for d in np.unique(day):assert ((rx[chosen]==r)&(day[chosen]==d)).sum()==32


def test_wrong_frozen_producer_is_rejected_before_baseline_iq_access():
    with pytest.raises(ValueError,match='Exact frozen source V baseline'):
        validate_baseline({'status':'SOURCE_V_PREDICTIONS_COMPLETE','code_commit':'wrong'}, {})
