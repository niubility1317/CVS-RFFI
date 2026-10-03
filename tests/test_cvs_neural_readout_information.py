"""A constructive distinction between local readout capacity and invariance."""
import torch
from experiments.cvs_neural_readout_identity.information import verify


def test_feature_counterexample_and_no_rng_side_effect():
    torch.manual_seed(42)
    before = torch.get_rng_state().clone()
    result = verify()
    assert torch.equal(before, torch.get_rng_state())
    assert result['optimization_steps'] == 0
    assert result['formal_data_access'] is False
    assert result['constructed_mixing_independent_phase_difference'] > .1
