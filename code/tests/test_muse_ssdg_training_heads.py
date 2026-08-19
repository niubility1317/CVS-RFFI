import torch

from cvsrffi.muse_ssdg import MUSETrainingHeads


def test_training_heads_are_finite_and_not_deployable():
    heads = MUSETrainingHeads(160, 32, 6, 20, 6)
    zid = torch.randn(8, 160)
    zdom = torch.randn(8, 32)
    domains = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])

    local = heads.local_prob(zid, domains)

    assert local.shape == (8, 6)
    assert torch.isfinite(local).all()
    loss = heads.nuisance_loss(
        zdom,
        torch.randn(8, 6),
        torch.ones(8, dtype=torch.bool),
    )
    assert torch.isfinite(loss)
    assert heads.deployment_state_dict() == {}


def test_self_supervised_loss_is_symmetric_and_backpropagates_to_both_views():
    heads = MUSETrainingHeads(160, 32, 6, 20, 6)
    z_id_a = torch.randn(5, 160, requires_grad=True)
    z_id_b = torch.randn(5, 160, requires_grad=True)

    loss_ab = heads.self_supervised_loss(z_id_a, z_id_b)
    loss_ba = heads.self_supervised_loss(z_id_b, z_id_a)

    assert torch.isfinite(loss_ab)
    assert torch.allclose(loss_ab, loss_ba, atol=1e-6)
    loss_ab.backward()
    assert z_id_a.grad is not None
    assert z_id_b.grad is not None
    assert torch.isfinite(z_id_a.grad).all()
    assert torch.isfinite(z_id_b.grad).all()


def test_nuisance_empty_mask_returns_zero_with_zero_gradient():
    heads = MUSETrainingHeads(160, 32, 6, 20, 6)
    z_dom = torch.randn(4, 32, requires_grad=True)
    targets = torch.randn(4, 6)

    loss = heads.nuisance_loss(z_dom, targets, torch.zeros(4, dtype=torch.bool))

    assert loss.requires_grad
    assert loss.item() == 0.0
    loss.backward()
    assert torch.equal(z_dom.grad, torch.zeros_like(z_dom))


def test_training_state_contains_trainable_heads_but_no_deployment_state():
    heads = MUSETrainingHeads(160, 32, 6, 20, 6)

    state = heads.training_state_dict()

    assert state
    assert all(torch.isfinite(value).all() for value in state.values())
    assert heads.deployment_state_dict() == {}
