import json

import numpy as np
import pytest
import torch

from experiments.pamf_lora_main import w24_formal as formal
from experiments.pamf_lora_main import w24_peer_spectral_direction_source as source
from experiments.pamf_lora_main.w24_component_directions import component_angle, component_norm, components, matched_component_edit


def test_matched_direction_preserves_each_parent_component():
    rng = torch.Generator().manual_seed(814)
    shape = (2, 8, 384)
    base, proposed = (torch.randn(shape, dtype=torch.complex64, generator=rng) for _ in range(2))
    phase = torch.exp(1j * torch.randn(shape, generator=rng))
    result = matched_component_edit(base, proposed, phase)
    original, estimate, actual = (components(value, phase)[0] for value in (base, proposed, result))
    torch.testing.assert_close(component_norm(actual), component_norm(original))
    torch.testing.assert_close(actual / component_norm(actual), estimate / component_norm(estimate))
    torch.testing.assert_close(matched_component_edit(base, torch.zeros_like(proposed), phase), base)


def test_component_angle_finite_gradient_and_zero_reference():
    gen = torch.Generator().manual_seed(815)
    target = torch.randn((2, 8, 384), dtype=torch.complex64, generator=gen)
    phase = torch.ones_like(target)
    proposal = (target * .5j).requires_grad_()
    loss, cos = component_angle(proposal, target, phase)
    torch.testing.assert_close(loss, torch.tensor(1.))
    torch.testing.assert_close(cos, torch.zeros(4))
    loss.backward()
    assert proposal.grad is not None and torch.isfinite(proposal.grad).all() and proposal.grad.abs().max() > 0
    identity, _ = component_angle(proposal, torch.zeros_like(target), phase)
    assert identity.item() == 0


def test_component_fit_only_accepts_bound_source_paths(tmp_path):
    rt = dict(project=str(tmp_path), run_root=str(tmp_path/'runs'/source.COMPONENT_RUN),
              source_root=str(tmp_path/'releases'/f'{source.DATA_RUN}-release'/'data'/'source'),
              practical_cache=str(tmp_path/'runs'/source.PRACTICAL_CACHE_RUN/'source_practical'),
              training_release_root=str(tmp_path/'releases'/f'{source.TRAIN_RUN}-release'))
    source.fit_config(rt)
    with pytest.raises(ValueError, match='source-only'):
        source.fit_config(dict(rt, target_root='unexpected'))


def test_formal_component_matrix_starts_from_frozen_fit_without_source_scores(tmp_path):
    group = tmp_path/'runs'/source.COMPONENT_RUN
    group.mkdir(parents=True)
    rt = dict(project=str(tmp_path), run_id=formal.RUN_COMPONENT_DIRECTIONS_CPU,
              run_root=str(group.parent/formal.RUN_COMPONENT_DIRECTIONS_CPU), candidate_root=str(group),
              third_prior=formal.COMPONENT_DIRECTIONS_KIND)
    assert formal.cpu_formal(rt) and not formal.component_directions_ready(rt)
    (group/'fit-complete.json').write_text(json.dumps(dict(status='SOURCE_FROZEN', steps=source.STEPS,
        component_angle_weight=source.COMPONENT_WEIGHT, **source.ISOLATION)), encoding='utf-8')
    (group/'direction.pt').touch()
    assert formal.component_directions_ready(rt)
    matrix = formal.cases(formal.COMPONENT_DIRECTIONS_KIND)
    assert list(matrix.values()) == [formal.FULL] * 3
    for case in matrix:
        proof = formal.receipt(case, True, third_prior=formal.COMPONENT_DIRECTIONS_KIND)
        assert proof['modes'][-2:] == ['peer_grouped', 'trained_grouped']
        assert proof['per_receiver'] == 2000 and proof['target_truth_read'] is False


def test_formal_component_forward_keeps_archived_baseline_slots(monkeypatch):
    monkeypatch.setattr(source, 'receive_grouped_readouts', lambda *args: (1, 2, 3, 4))
    monkeypatch.setattr(formal, 'classify_aperture', lambda head, view: np.full(2, view, dtype=np.int16))
    output, activity = formal.component_direction_labels([object()], (None,) * 4, np.empty((2, 8192)))
    assert output.tolist() == [[[1, 1], [3, 3], [4, 4]]] and activity == 2
