import torch
from experiments.phase1_joint_particle.particle import JointParticleRecovery
from experiments.phase1_joint_wide import train, ground


def test_wide_entrypoints_apply_same_registered_width(monkeypatch):
    calls = []
    monkeypatch.setattr(train.joint_train, 'WIDTH', 192)
    monkeypatch.setattr(train.joint_train, 'main', lambda: calls.append(('train', train.joint_train.WIDTH)))
    monkeypatch.setattr(ground.joint_ground, 'WIDTH', 192)
    monkeypatch.setattr(ground.joint_ground, 'main', lambda: calls.append(('ground', ground.joint_ground.WIDTH)))
    train.main()
    ground.main()
    assert calls == [('train', 384), ('ground', 384)]


def test_wide_initial_output_is_dura_and_has_expected_capacity():
    class IdentityAction:
        def __init__(self, kernel, main, image):
            self.kernel, self.main, self.image = kernel, main, image
        def __call__(self, x):
            return x
        def adjoint(self, x):
            return x
    torch.manual_seed(3)
    model = JointParticleRecovery(width=384).eval()
    assert sum(p.numel() for p in model.parameters()) == 7120514
    x = torch.randn(2, 2, 256)
    base = torch.randn_like(x)
    direction = torch.randn_like(x)
    variance = torch.full((2,), .05)
    action = IdentityAction(torch.ones(2, 1), torch.ones(2, 256), torch.zeros(2, 256))
    with torch.no_grad():
        output = model(x, base, direction, variance, action)
    assert torch.equal(output, base)


def test_wide_ground_rejects_narrow_checkpoint(monkeypatch, tmp_path):
    import pytest
    from experiments.phase1_joint_particle import ground as g
    monkeypatch.setattr(g, 'WIDTH', 384)
    narrow = dict(schema=g.SCHEMA, commit='fixed', epochs=20, fit_ids=32768,
                  views_per_record=3, width=192, steps=g.STEPS, inference_particles=g.PARTICLES,
                  training_particles=1, TX_labels_read=0, F_weights_read=0,
                  F_outputs_read=0, V_reads=0, target_reads=0, DURA_frozen=True)
    monkeypatch.setattr(g.core, 'read', lambda *a: dict(narrow, status='SOURCE_TRAINED'))
    monkeypatch.setattr(g.torch, 'load', lambda *a, **k: narrow)
    with pytest.raises(ValueError, match='checkpoint'):
        g.load_fixed(tmp_path, 'source', 'fixed', 'cpu')


def test_wide_gpu_training_step_has_finite_gradients():
    import pytest
    if not torch.cuda.is_available():
        pytest.skip('CUDA GPU required')
    class IdentityAction:
        def __init__(self, kernel, main, image):
            self.kernel, self.main, self.image = kernel, main, image
        def __call__(self, value):
            return value
        def adjoint(self, value):
            return value
    model = JointParticleRecovery(width=384).cuda()
    x = torch.randn(2, 2, 256, device='cuda')
    base = torch.randn_like(x)
    direction = torch.randn_like(x)
    variance = torch.full((2,), .05, device='cuda')
    action = IdentityAction(torch.ones(2, 1, device='cuda'), torch.ones(2, 256, device='cuda'),
                            torch.zeros(2, 256, device='cuda'))
    latent = torch.randn(2, 1, 2, 256, device='cuda')
    output = model(x, base, direction, variance, action, latent=latent)
    (output-x).square().mean().backward()
    assert model.output.weight.grad is not None
    assert torch.isfinite(model.output.weight.grad).all()
    assert model.output.weight.grad.abs().sum() > 0
