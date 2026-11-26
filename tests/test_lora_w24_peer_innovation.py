from types import SimpleNamespace

import torch

from experiments.pamf_lora_main.w24_peer_innovation import innovation_fields, peer_moments, restorer
from experiments.pamf_lora_main.w24_formal import FULL, cases
from experiments.pamf_lora_main.w24_third_prior import fit_inputs, trial


WIDE = 384


class DuraCapacity:
    def __init__(self, variant, *, samples, condition_fields):
        self.shape = variant, samples, condition_fields

    def stages(self, y, quality, rms, *, condition=()):
        return condition


class DuraW24(DuraCapacity):
    def propose(self, y, quality, rms, edges):
        return self.stages(y, quality, rms, condition=edges)


def test_receiver_reference_excludes_the_chirp_being_restored():
    y = torch.arange(24, dtype=torch.float32).view(1, 8, 3).to(torch.complex64) * (1 + 2j)
    mean, variance = peer_moments(y)
    for j in range(8):
        others = y[:, [k for k in range(8) if k != j]]
        expected_mean = others.mean(1)
        expected_variance = (others - expected_mean[:, None]).abs().square().mean(1)
        torch.testing.assert_close(mean[:, j], expected_mean)
        torch.testing.assert_close(variance[:, j], expected_variance)


def test_condition_fields_match_and_constant_preamble_is_finite():
    y = torch.ones(2, 8, 384, dtype=torch.complex64) * (1 - 1j)
    for use_peers in (False, True):
        fields = innovation_fields(y, use_peers=use_peers)
        assert len(fields) == 3
        assert all(field.shape == y.shape and torch.isfinite(field).all() for field in fields)
        assert all(torch.count_nonzero(field) == 0 for field in fields)


def test_both_arms_add_three_fields_at_the_same_solver_interface():
    y = torch.rand(2, 8, WIDE, dtype=torch.complex64)
    edges = tuple(torch.full_like(y.real[:, :, :1], x) for x in range(6))
    archive = SimpleNamespace(DuraW24=DuraW24)
    local = restorer(archive, use_peers=False)
    peer = restorer(archive, use_peers=True)
    assert local.shape == peer.shape == ('expanded', WIDE, 9)
    for model, use_peers in ((local, False), (peer, True)):
        fields = model.propose(y, None, None, edges)
        assert all(x is old for x, old in zip(fields[:6], edges, strict=True))
        for got, expected in zip(fields[6:], innovation_fields(y, use_peers=use_peers), strict=True):
            torch.testing.assert_close(got, expected)


def test_peer_trial_is_full_only_and_fit_paths_are_source_only():
    spec = trial('peer')
    assert spec.arms == ('innovation_local', 'innovation_peer')
    assert spec.source_modes[-2:] == ('innovation_local_full', 'innovation_peer_full')
    assert set(cases('peer')) == {f'{FULL}-{i}' for i in range(3)}
    cfg = dict(source_root='/U', run_root='/run', classifier_root='/F', native_target_root='/target')
    assert fit_inputs(cfg, 'innovation_peer') == dict(source_root='/U', run_root='/run/innovation_peer')
