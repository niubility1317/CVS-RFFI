import pytest
import torch
from torch import nn

from experiments.pamf_phase2_practical.group2_b_state import load_b_state, save_b_state
from experiments.pamf_phase2_practical.group2_cil_bridge import AdaptedClassifier, make_anchor
from experiments.pamf_phase2_practical.group2_cross_fit import register_c
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


def test_mopc_from_reloaded_actual_anchor_matches_in_memory(tmp_path):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(12)
        model = nn.Linear(3, 4).eval()
        bridged = AdaptedClassifier(model, nn.Linear(4, 2), None).eval()
        old_x, old_y = torch.randn(4, 3), torch.tensor([0, 0, 1, 1])
        new_x, new_y = torch.randn(4, 3), torch.tensor([2, 2, 3, 3])
        anchor = make_anchor(bridged, old_x, old_y, method='mopc_hr')
        expected = register_c({'mopc_hr': anchor}, new_x, new_y, seed=42, methods=('mopc_hr',))['mopc_hr']
        job = dict(key='d-s1-c1', profile='dadda_sda', seed=1, capsule='capsules/P/M/R')
        path = tmp_path/'B-K2.pt'
        save_b_state(path, job, 2, 'old-split', 17, model, anchor)
        restored = load_b_state(path, job, 2, 'old-split', 17, 'cpu')['anchor']
        actual = register_c({'mopc_hr': restored}, new_x, new_y, seed=42, methods=('mopc_hr',))['mopc_hr']
        assert all(torch.equal(v, actual.model.state_dict()[k]) for k, v in expected.model.state_dict().items())
        assert torch.equal(expected.prototypes, actual.prototypes)
    finally:
        torch.set_num_threads(previous)
