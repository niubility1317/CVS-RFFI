import torch
from experiments.pamf_phase3_restoration.mixture_affine_model import affine_blend


def test_identity_and_signal_when_baseline_probabilities_saturate():
    components = torch.tensor([[[2., -1., 3., 5.], [1., 2., -4., 0.]]], dtype=torch.float64)
    baseline = components[:, :, 0]
    adjustment = torch.zeros(1, 4, dtype=torch.float32, requires_grad=True)
    output = affine_blend(baseline, components, adjustment)
    torch.testing.assert_close(output, baseline, rtol=0, atol=0)
    output.square().sum().backward()
    assert torch.isfinite(adjustment.grad).all()
    assert adjustment.grad.abs().sum() > 1e-4
    assert abs(adjustment.grad.sum().item()) < 1e-6


def test_affine_adds_no_response_to_common_logits_shift():
    torch.manual_seed(7)
    components = torch.randn(3, 5, 4, dtype=torch.float64)
    baseline = torch.randn(3, 5, dtype=torch.float64)
    adjustment = torch.randn(3, 4, dtype=torch.float32)
    shift = torch.full_like(adjustment, 3.)
    torch.testing.assert_close(affine_blend(baseline, components, adjustment),
                               affine_blend(baseline, components, adjustment+shift),
                               rtol=1e-6, atol=1e-6)


def test_full_model_matched_initialization_and_branch_gradients():
    from experiments.pamf_phase3_restoration.mixture_affine_model import scratch_affine_mixture
    from experiments.pamf_phase3_restoration.mixture_gate_model import scratch_mixture_gate
    torch.set_num_threads(2)
    original, affine = scratch_mixture_gate(19), scratch_affine_mixture(19)
    assert set(original.state_dict()) == set(affine.state_dict())
    assert all(torch.equal(v, affine.state_dict()[k]) for k, v in original.state_dict().items())
    offsets = torch.zeros(4, 512)
    offsets[0, 0], offsets[1, 0] = .4, -.4
    offsets[2, 1], offsets[3, 1] = .4, -.4
    weights = torch.full((4,), .25)
    cov = torch.eye(512)
    within = cov - (offsets.T*weights) @ offsets
    for model in (original, affine):
        model.physical.bind_prior(torch.zeros(512), cov)
        model.physical.bind_mixture(within, offsets, weights)
    x = torch.randn(2, 2, 256) * .3
    old, _ = original(x, return_aux=True)
    new, aux = affine(x, return_aux=True)
    torch.testing.assert_close(old, new, rtol=0, atol=0)
    torch.testing.assert_close(new, aux['base_iq'], rtol=0, atol=0)
    new.square().mean().backward()
    assert torch.isfinite(affine.detail.output.weight.grad).all()
    assert affine.detail.output.weight.grad.abs().sum() > 0
    assert affine.detail.refiner.output.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in affine.physical.parameters())


def test_matched_scientific_config_and_queue(tmp_path):
    import json
    from experiments.pamf_phase3_restoration import mixture_affine_trial as affine
    from experiments.pamf_phase3_restoration import mixture_gate_trial as parent
    baseline, candidate = parent.configuration(parent.ARM), affine.configuration(affine.ARM)
    changes = {'run_id', 'arm', 'architecture', 'scientific_change', 'matched_control',
               'fusion', 'source_evidence', 'resource_policy'}
    for key in set(baseline) | set(candidate):
        if key not in changes:
            assert baseline.get(key) == candidate.get(key), key
    assert not affine.previous_gpu_stages_finished(tmp_path)
    folder = tmp_path/'runs'/parent.RUN
    folder.mkdir(parents=True)
    (folder/f'terminal-{parent.ARM}-train.json').write_text(json.dumps({'exit_code':0}))
    assert not affine.previous_gpu_stages_finished(tmp_path)
    (folder/f'terminal-{parent.ARM}-cross_both.json').write_text(json.dumps({'exit_code':0}))
    assert affine.previous_gpu_stages_finished(tmp_path)
