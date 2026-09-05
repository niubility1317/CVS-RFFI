from types import SimpleNamespace
import pytest
import torch

from cvsrffi.pair_failure import build_pair_failure_payload


def test_failure_payload_detaches_groups_and_keeps_evidence_boundary():
    value = torch.tensor(float('nan'), requires_grad=True)
    packet = build_pair_failure_payload(epoch=2, batch=7, loss=value,
        labeled_components={'orbit_z': value}, unlabeled_components={'orbit_logit': torch.tensor(1.)},
        first_nonfinite_gradient={'parameter_name': 'backbone.weight'},
        args=SimpleNamespace(pair_reform='safe'),
        source_physical_ids={'labeled': [(1, 2, 3, 4, 5)], 'unlabeled': [(2, 3, 4, 5, 6)]},
        gradscale=1024., rng_state={'torch': torch.tensor([1], dtype=torch.uint8)},
        model_state={'weight': torch.ones(2, requires_grad=True)})
    assert packet['claim_boundary'] == 'OBSERVATION_ONLY_NOT_P5_ROOT_CAUSE'
    assert not packet['finite']['loss']
    assert not packet['finite']['labeled_components']['orbit_z']
    assert packet['finite']['unlabeled_components']['orbit_logit']
    assert not packet['loss'].requires_grad and packet['loss'].device.type == 'cpu'
    assert not packet['model_state']['weight'].requires_grad
    packet['model_state']['weight'][0] = 3
    assert packet['gradscale'] == 1024.


def test_failure_packet_rejects_truth_fields_even_nested():
    kwargs = dict(epoch=1, batch=1, loss=torch.tensor(0.), labeled_components={},
        unlabeled_components={}, first_nonfinite_gradient=None, args={}, source_physical_ids={})
    with pytest.raises(ValueError, match='truth'):
        build_pair_failure_payload(**kwargs, diagnostics={'nested': {'true_tx_i': torch.tensor([1])}})
    with pytest.raises(ValueError, match='physical'):
        build_pair_failure_payload(**{**kwargs, 'source_physical_ids': {'u': {'tx_i': [1]}}})


def test_snapshot_does_not_alias_live_state_and_handles_nested_containers():
    weight = torch.tensor([1.], requires_grad=True)
    packet = build_pair_failure_payload(epoch=1, batch=1, loss=weight.sum(),
        labeled_components={'nested': [weight]}, unlabeled_components={},
        first_nonfinite_gradient=None, args={}, source_physical_ids={'u': [1, 2]},
        optimizer_state={'state': {0: {'momentum_buffer': weight}}})
    with torch.no_grad():
        weight.fill_(9)
    assert packet['optimizer_state']['state'][0]['momentum_buffer'].item() == 1
