import numpy as np
import pytest
from scipy.signal import resample_poly
import torch

from experiments.pamf_lora_main.w24_native_center import NativeCenterProjection, direction_terms
from experiments.pamf_lora_main import w24_formal, w24_third_prior
from experiments.cric_dura_comparison.io import write
from experiments.pamf_lora_main.w24_packet_dura import PACKET, NativePacketDura, native_additive_pair, source_objective
from experiments.pamf_lora_main.w24_third_prior import NATIVE_CENTER_ARMS, fit_inputs, trial


def test_native_projection_matches_fixed_rational_filter_with_capture_boundaries():
    rng = np.random.default_rng(202)
    raw = (rng.normal(size=(3, 8192)) + 1j * rng.normal(size=(3, 8192))).astype(np.complex64)
    starts = np.arange(8)[:, None] * 1024 + 12 - 125
    crop = starts + np.arange(1250)
    crop = np.where(crop < 0, -crop, crop)
    crop = np.where(crop >= 8192, 2 * 8192 - 2 - crop, crop)
    expected = resample_poly(raw[:, crop], up=48, down=125, axis=-1)[..., 48:432]
    projection = NativeCenterProjection()
    np.testing.assert_allclose(projection(torch.from_numpy(raw)).numpy(), expected, rtol=2e-6, atol=4e-7)
    assert not list(projection.parameters())


def test_center_angle_keeps_direction_separate_from_strength_and_handles_clean():
    rng = torch.Generator().manual_seed(211)
    oracle = torch.randn((3, PACKET), generator=rng, dtype=torch.complex64)
    oracle[-1].zero_()
    projector = NativeCenterProjection()
    aligned, metrics = direction_terms(projector, -.4 * oracle, oracle)
    rotated, _ = direction_terms(projector, (1j * .6 - 1) * oracle, oracle)
    opposed, _ = direction_terms(projector, -1.6 * oracle, oracle)
    assert aligned < 1e-5 < rotated < opposed
    torch.testing.assert_close(metrics[2], torch.tensor(.6), atol=1e-5, rtol=0.)
    zero, diagnostics = direction_terms(projector, oracle[-1:], oracle[-1:])
    assert zero == 0 and torch.isfinite(diagnostics).all()


def test_native_center_pair_changes_only_gradient_and_is_full_only():
    spec = trial('native-center-direction')
    assert spec.steps == 24000 and spec.arms == spec.fit_arms == NATIVE_CENTER_ARMS
    assert spec.routes == ('full_native',)
    context = dict(source_root='src', run_root='out', native_target_root='blocked', classifier_root='blocked')
    for arm in spec.arms:
        assert fit_inputs(context, arm) == dict(source_root='src', run_root=f'out/{arm}')
    model = NativePacketDura()
    projector = NativeCenterProjection()
    raw = torch.randn(4, PACKET, dtype=torch.complex64)
    received, references = native_additive_pair(raw, torch.Generator().manual_seed(30), with_identity=True)

    def calculate(project=None, weight=0.):
        return source_objective(model, received, references[1], with_identity=True, audit_references=references,
                                center_project=project, center_weight=weight)

    legacy, _ = calculate()
    control, ref = calculate(projector)
    candidate, treated = calculate(projector, .20)
    assert legacy == control and candidate > control and ref.shape == (13,)
    torch.testing.assert_close(treated, ref)
    p = model.priors[-1].read[-1].weight
    g0 = torch.autograd.grad(control, p)[0]
    g1 = torch.autograd.grad(candidate, p)[0]
    assert torch.isfinite(g1).all() and not torch.allclose(g0, g1)


def test_requested_formal_evaluation_is_bound_to_both_frozen_native_center_rows(tmp_path, monkeypatch):
    kind = w24_formal.NATIVE_CENTER_KIND
    spec = trial(kind)
    candidate = tmp_path / 'runs' / spec.run
    candidate.mkdir(parents=True)
    write(candidate/'state.json', dict(status='SOURCE_SCORED', promote=False, arms=list(spec.arms), steps=spec.steps))
    calls = []
    monkeypatch.setattr(w24_third_prior, 'frozen_source_results', lambda *args: calls.append(args))
    monkeypatch.setattr(w24_third_prior, 'qualified_source', lambda *args: pytest.fail('Unexpected promotion check'))
    runtime = dict(project=tmp_path, candidate_root=candidate, third_prior=kind,
                   run_root=candidate.parent / w24_formal.RUN_NATIVE_CENTER, run_id=w24_formal.RUN_NATIVE_CENTER)
    w24_formal.verify_matched_source(runtime)
    assert calls == [(candidate, kind)]
    assert list(w24_formal.cases(kind)) == [f'full_native-{i}' for i in range(3)]
    for altered in (dict(third_prior='native-band-risk'), dict(candidate_root=tmp_path), dict(run_root=tmp_path)):
        with pytest.raises(ValueError, match='Unexpected frozen parent'):
            w24_formal.verify_matched_source(dict(runtime, **altered))
    monkeypatch.setattr(w24_third_prior, 'qualified_source', lambda *args: calls.append('normal-policy'))
    w24_formal.verify_matched_source(dict(runtime, run_id='another-run'))
    assert calls[-1] == 'normal-policy'
