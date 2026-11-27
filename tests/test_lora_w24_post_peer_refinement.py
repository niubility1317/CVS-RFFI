import json

import numpy as np
import pytest
import torch

from experiments.pamf_lora_main.w24_component_directions import component_norm, components
from experiments.pamf_lora_main.w24_post_peer_refinement import PostPeerRefiner, endpoint_risk


def example():
    rng = torch.Generator().manual_seed(31)
    y = torch.randn(3, 8, 384, dtype=torch.complex64, generator=rng)
    prior = torch.randn(3, 8, 384, dtype=torch.complex64, generator=rng) * 0.12
    phase = torch.exp(1j * torch.randn(3, 8, 384, generator=rng))
    quality = torch.rand(3, 8, 1, generator=rng)
    return y, prior, phase, quality


@pytest.mark.parametrize('peer_first', [False, True])
def test_refinement_preserves_component_norm_and_empty_edit(peer_first):
    y, prior, phase, quality = example()
    prior[0] = 0
    model = PostPeerRefiner(peer_first=peer_first)
    edit = model(y, prior, phase, quality)
    assert edit.dtype == torch.complex64 and edit.shape == y.shape
    assert torch.equal(edit[0], prior[0])
    torch.testing.assert_close(component_norm(components(edit, phase)[0]),
                               component_norm(components(prior, phase)[0]), rtol=2e-5, atol=1e-6)


def test_ablations_have_identical_parameters_and_distinct_order():
    y, prior, phase, quality = example()
    before = PostPeerRefiner(peer_first=False)
    after = PostPeerRefiner(peer_first=True)
    after.load_state_dict(before.state_dict(), strict=True)
    assert set(before.state_dict()) == set(after.state_dict())
    reference = before(y, prior, phase, quality)
    candidate = after(y, prior, phase, quality)
    assert torch.linalg.vector_norm(candidate - reference) > 1e-4


def test_endpoint_source_loss_uses_actual_correction_and_preserves_identity():
    y, prior, phase, quality = example()
    known = torch.tensor([True, True, False])
    source = y + prior
    exact, terms = endpoint_risk(y, source, prior, prior, phase, known)
    assert torch.isfinite(exact) and terms['identity'] == 0
    assert terms['weak_spectrum'] < 1e-8 and terms['chirp_difference'] < 1e-8
    model = PostPeerRefiner(peer_first=True)
    proposal = model(y, prior, phase, quality)
    loss, terms = endpoint_risk(y, source, prior, proposal, phase, known)
    assert loss > exact
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_full_release_accepts_freeze_without_source_score_and_rejects_contamination(tmp_path):
    from experiments.pamf_lora_main.w24_formal import (
        FULL, POST_PEER_KIND, RUN_POST_PEER_CPU, cases, cpu_formal, evaluation_modes, post_peer_ready,
    )
    from experiments.pamf_lora_main.w24_peer_spectral_direction_source import COMPONENT_RUN
    from experiments.pamf_lora_main.w24_post_peer_refinement_source import ARMS, BATCH, ISOLATION, RUN, SEED, STEPS

    source = tmp_path/'runs'/RUN
    runtime = dict(project=str(tmp_path), run_id=RUN_POST_PEER_CPU, third_prior=POST_PEER_KIND,
                   run_root=str(tmp_path/'runs'/RUN_POST_PEER_CPU), candidate_root=str(source))
    assert cpu_formal(runtime)
    assert set(cases(POST_PEER_KIND).values()) == {FULL}
    assert evaluation_modes(FULL, True, third_prior=POST_PEER_KIND)[-2:] == ARMS
    assert not post_peer_ready(runtime)
    source.mkdir(parents=True)
    evidence = dict(status='SOURCE_FROZEN', base=COMPONENT_RUN, arms=list(ARMS), steps=STEPS,
                    model_seed=SEED, batch=BATCH, fit=20000, held=500, **ISOLATION)
    (source/'fit-complete.json').write_text(json.dumps(evidence), encoding='utf-8')
    (source/'refinement.pt').touch()
    assert post_peer_ready(runtime)
    evidence['target_reads'] = 1
    (source/'fit-complete.json').write_text(json.dumps(evidence), encoding='utf-8')
    with pytest.raises(ValueError, match='source-only'):
        post_peer_ready(runtime)


def test_native_readout_keeps_direct_unchanged_and_reuses_native_edges(monkeypatch):
    from experiments.pamf_lora_main import w24_post_peer_refinement_source as recipe
    from experiments.pamf_lora_main.w24_aperture_noise_audit import CENTER, center_rms, model_iq
    from experiments.pamf_lora_main.w24_fixed_dose_direction_model import ApertureDirection

    class Parent:
        def restore_native(self, raw):
            return raw - 0.15 * raw.roll(1, dims=-1)

    class Tracker:
        def track(self, iq):
            z = torch.view_as_complex(iq.transpose(-1, -2).contiguous())
            return z, torch.ones_like(z), z.real.new_ones(z.shape[:-1])

    def direction(raw, edit):
        return edit, edit + 0.07 * edit.roll(1, dims=-1)

    wave = torch.randn(2, 8192, dtype=torch.complex64, generator=torch.Generator().manual_seed(4))
    aperture = ApertureDirection()
    inputs = aperture, Parent(), Tracker(), direction

    def native_aperture(raw):
        # The production NumPy extractor lives in the archived W24 release.
        return aperture.project_regions(torch.from_numpy(np.array(raw, copy=True))).numpy()

    monkeypatch.setattr(recipe, 'native_aperture', native_aperture)
    models = tuple(PostPeerRefiner(peer_first=order).eval() for order in (False, True))
    frame, half = recipe.prepare(*inputs, wave)
    y = native_aperture(wave.numpy())
    edge_parent = native_aperture(half.numpy())
    direct, retained, *actual = recipe.receive(inputs, models, wave.numpy())
    np.testing.assert_array_equal(direct.numpy(), model_iq(y, center_rms(y)))
    np.testing.assert_array_equal(retained.numpy(), model_iq(edge_parent, center_rms(edge_parent)))
    for arm, output in zip(models, actual, strict=True):
        expected = edge_parent.copy()
        with torch.no_grad():
            expected[..., :CENTER] = (frame[0] + arm(*frame)).numpy()
        np.testing.assert_array_equal(output.numpy(), model_iq(expected, center_rms(expected)))
