import pytest
import torch
from torch import nn

from experiments.pamf_phase2_practical.group2_b_state import load_b_state, save_b_state
from experiments.pamf_phase2_practical.vendor_native.comparison_suite.registration import RegistrationState


def test_actual_b_and_anchor_round_trip_with_binding_and_exclusive_output(tmp_path):
    model = nn.Sequential(nn.Linear(2, 3), nn.BatchNorm1d(3)).eval()
    anchor = RegistrationState('mopc_hr', nn.Linear(3, 2), (0, 1), 2, prototypes=torch.randn(2, 3))
    job = dict(key='proto-s1-c1', profile='protonet_cda', seed=1, capsule='capsules/P/M/R')
    path = tmp_path/'B-K5.pt'
    native_prototypes = torch.randn(2, 3)
    rng = torch.get_rng_state().clone()
    binding = save_b_state(path, job, 5, 'old-split', 17, model, anchor, native_prototypes)
    assert torch.equal(rng, torch.get_rng_state())
    restored = load_b_state(path, job, 5, 'old-split', 17, 'cpu')
    assert binding == restored['binding']
    assert all(torch.equal(v, restored['model'].state_dict()[k]) for k, v in model.state_dict().items())
    assert torch.equal(anchor.prototypes, restored['anchor'].prototypes)
    assert torch.equal(native_prototypes, restored['old_prototypes'])
    with pytest.raises(ValueError, match='binding differs'):
        load_b_state(path, job, 5, 'different-split', 17, 'cpu')
    with pytest.raises(FileExistsError):
        save_b_state(path, job, 5, 'old-split', 17, model, anchor, native_prototypes)
