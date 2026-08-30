from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from cvsrffi.phase1_hcfdg.losses import (
    content_conditioned_lodo_loss,
    counterfactual_losses,
    compose_hcfdg_loss,
    hierarchical_dro_loss,
    lodo_prototype_loss,
)
from cvsrffi.phase1_hcfdg.model import HCFDGOutput


def _episode_embeddings():
    z = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.1, 0.9, 0.0],
            [0.8, 0.2, 0.0],
            [0.2, 0.8, 0.0],
        ],
        requires_grad=True,
    )
    y = torch.tensor([0, 1, 0, 1, 0, 1])
    domain = torch.tensor([0, 0, 1, 1, 9, 9])
    return z, y, domain


def test_lodo_prototypes_exclude_every_query_domain_row():
    z, y, domain = _episode_embeddings()

    loss, info = lodo_prototype_loss(z, y, domain, query_domain=9, temperature=0.10)

    assert 9 not in info.support_domains
    assert info.query_count == int((domain == 9).sum())
    assert info.prototype_counts == {0: 2, 1: 2}
    assert torch.isfinite(loss)


def test_content_conditioning_falls_back_without_close_support():
    z, y, _ = _episode_embeddings()
    domains = torch.tensor([0, 0, 1, 1, 2, 2])
    keys = torch.tensor(
        [[0.0, 0.0], [0.1, 0.0], [1.0, 0.0], [1.1, 0.0], [4.0, 0.0], [4.1, 0.0]]
    )

    _, info = content_conditioned_lodo_loss(
        z,
        y,
        domains,
        keys,
        query_domain=2,
        max_distance=0.01,
    )

    assert info.fallback_classes == frozenset({0, 1})


def test_content_conditioning_uses_soft_weights_when_support_is_close():
    z, y, domain = _episode_embeddings()
    keys = torch.tensor(
        [[0.00], [1.00], [0.02], [1.02], [0.01], [1.01]], dtype=torch.float32
    )

    loss, info = content_conditioned_lodo_loss(
        z,
        y,
        domain,
        keys,
        query_domain=9,
        max_distance=0.05,
    )

    assert info.fallback_classes == frozenset()
    assert info.weighted_classes == frozenset({0, 1})
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(z.grad).all()


def test_counterfactual_loss_reports_id_inv_env_and_style_terms():
    labels = torch.tensor([0, 1])
    z_id = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    cf_z_id = torch.tensor([[0.9, 0.1], [0.1, 0.9]], requires_grad=True)
    cf_logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]], requires_grad=True)
    cf_env_logits = torch.tensor([[0.0, 2.0], [2.0, 0.0]], requires_grad=True)
    target_env = torch.tensor([1, 0])
    h = torch.tensor([[1.0, 2.0], [2.0, 1.0]])
    cf_h = torch.tensor([[1.2, 1.8], [1.8, 1.2]], requires_grad=True)

    result = counterfactual_losses(
        cf_logits,
        labels,
        cf_z_id=cf_z_id,
        z_id=z_id,
        cf_env_logits=cf_env_logits,
        target_env=target_env,
        cf_h=cf_h,
        target_h=h,
    )

    assert result.total.ndim == 0
    assert result.cf_id.item() > 0.0
    assert result.cf_inv.item() > 0.0
    assert result.cf_env.item() > 0.0
    assert result.style.item() > 0.0
    assert torch.isfinite(result.total)
    result.total.backward()
    assert torch.isfinite(cf_z_id.grad).all()
    assert torch.isfinite(cf_logits.grad).all()
    assert torch.isfinite(cf_env_logits.grad).all()
    assert torch.isfinite(cf_h.grad).all()


def test_hdro_shrinks_small_child_group_to_parent():
    per_sample_loss = torch.tensor([0.1, 0.2, 0.3, 0.2, 0.9, 0.1], requires_grad=True)
    groups = {
        "tx:0": torch.tensor([True, True, True, True, True, False]),
        "tx_rx:0:1": torch.tensor([False, False, False, False, True, False]),
    }

    loss, info = hierarchical_dro_loss(
        per_sample_loss,
        groups,
        kappa=8.0,
        tau=0.25,
        min_group=4,
    )

    assert info.shrunk_risks["tx_rx:0:1"] < info.raw_risks["tx_rx:0:1"]
    loss.backward()
    assert torch.isfinite(per_sample_loss.grad).all()


def test_compose_uses_frozen_component_weights_and_zeroes_disabled_terms():
    z, y, domain = _episode_embeddings()
    common_logits = torch.tensor(
        [[2.0, 0.0], [0.0, 2.0], [2.0, 0.0], [0.0, 2.0], [2.0, 0.0], [0.0, 2.0]],
        requires_grad=True,
    )
    output = SimpleNamespace(common_logits=common_logits, z_id=z)

    result = compose_hcfdg_loss(
        output,
        labels=y,
        domain=domain,
        query_domain=9,
        use_lodo=True,
        use_counterfactual=False,
        use_hdro=False,
        use_csd=False,
        use_fac=False,
    )

    assert torch.isclose(result.total, result.id_loss + 0.40 * result.lodo_loss)
    assert result.cf_loss.item() == 0.0
    assert result.hdro_loss.item() == 0.0
    assert result.csd_loss.item() == 0.0
    assert result.fac_loss.item() == 0.0
    assert result.cf_loss.device == common_logits.device


def test_csd_is_exactly_specific_head_cross_entropy():
    labels = torch.tensor([0, 1, 1])
    common_logits = torch.tensor(
        [[4.0, -1.0], [3.0, -2.0], [-3.0, 2.0]], requires_grad=True
    )
    specific_logits = torch.tensor(
        [[0.4, -0.2], [-0.7, 1.3], [1.1, -0.5]], requires_grad=True
    )
    output = SimpleNamespace(
        common_logits=common_logits,
        specific_logits=specific_logits,
    )

    result = compose_hcfdg_loss(
        output,
        labels=labels,
        use_lodo=False,
        use_counterfactual=False,
        use_hdro=False,
        use_csd=True,
        use_fac=False,
    )

    expected = F.cross_entropy(specific_logits, labels)
    torch.testing.assert_close(result.csd_loss, expected, rtol=0.0, atol=0.0)


def test_enabled_lodo_fails_closed_without_query_definition():
    output = SimpleNamespace(
        common_logits=torch.randn(4, 2),
        z_id=torch.randn(4, 3),
    )

    with pytest.raises(ValueError, match=r"^LODO requires query_domain or query_mask$"):
        compose_hcfdg_loss(
            output,
            labels=torch.tensor([0, 1, 0, 1]),
            domain=torch.tensor([0, 0, 1, 1]),
            use_lodo=True,
            use_counterfactual=False,
            use_hdro=False,
            use_csd=False,
            use_fac=False,
        )


def test_enabled_counterfactual_fails_closed_without_cf_tensors():
    output = SimpleNamespace(
        common_logits=torch.randn(2, 2),
        z_id=torch.randn(2, 3),
    )

    with pytest.raises(
        ValueError,
        match=(
            r"^counterfactual requires cf_logits, cf_z_id, cf_env_logits, "
            r"target_env, cf_h, and target_h$"
        ),
    ):
        compose_hcfdg_loss(
            output,
            labels=torch.tensor([0, 1]),
            use_lodo=False,
            use_counterfactual=True,
            use_hdro=False,
            use_csd=False,
            use_fac=False,
        )


def test_enabled_hdro_fails_closed_without_complete_groups():
    output = SimpleNamespace(common_logits=torch.randn(4, 2))

    with pytest.raises(
        ValueError,
        match=r"^HDRO requires groups or receiver, day, and channel labels$",
    ):
        compose_hcfdg_loss(
            output,
            labels=torch.tensor([0, 1, 0, 1]),
            use_lodo=False,
            use_counterfactual=False,
            use_hdro=True,
            use_csd=False,
            use_fac=False,
        )


def test_enabled_csd_fails_closed_without_specific_logits():
    output = SimpleNamespace(common_logits=torch.randn(2, 2))

    with pytest.raises(ValueError, match=r"^CSD requires specific_logits and labels$"):
        compose_hcfdg_loss(
            output,
            labels=torch.tensor([0, 1]),
            use_lodo=False,
            use_counterfactual=False,
            use_hdro=False,
            use_csd=True,
            use_fac=False,
        )


def test_enabled_fac_fails_closed_without_auxiliary_logits_and_targets():
    output = SimpleNamespace(common_logits=torch.randn(2, 2))

    with pytest.raises(
        ValueError,
        match=(
            r"^FAC requires logits and targets for receiver, day, channel, "
            r"conditional_receiver, and tx_from_env$"
        ),
    ):
        compose_hcfdg_loss(
            output,
            labels=torch.tensor([0, 1]),
            use_lodo=False,
            use_counterfactual=False,
            use_hdro=False,
            use_csd=False,
            use_fac=True,
        )


def test_task4_output_composes_all_enabled_loss_families():
    labels = torch.tensor([0, 1, 0, 1, 0, 1])
    receiver = torch.tensor([0, 0, 1, 1, 2, 2])
    day = torch.tensor([0, 1, 0, 1, 0, 1])
    channel = torch.tensor([0, 0, 1, 1, 0, 0])
    z_id = torch.randn(6, 4, requires_grad=True)
    common_logits = torch.randn(6, 2, requires_grad=True)
    specific_logits = torch.randn(6, 2, requires_grad=True)
    output = HCFDGOutput(
        common_logits=common_logits,
        specific_logits=specific_logits,
        z_id=z_id,
        z_rx=torch.randn(6, 2),
        z_day=torch.randn(6, 2),
        z_channel=torch.randn(6, 2),
        z_env=torch.randn(6, 6),
        receiver_logits=torch.randn(6, 3, requires_grad=True),
        day_logits=torch.randn(6, 2, requires_grad=True),
        channel_logits=torch.randn(6, 2, requires_grad=True),
        tx_from_env_logits=torch.randn(6, 2, requires_grad=True),
        conditional_receiver_logits=torch.randn(6, 3, requires_grad=True),
        fused_feature=torch.randn(6, 4),
    )

    result = compose_hcfdg_loss(
        output,
        labels=labels,
        domain=receiver,
        receiver=receiver,
        day=day,
        channel=channel,
        query_domain=2,
        use_lodo=True,
        use_counterfactual=True,
        use_hdro=True,
        use_csd=True,
        use_fac=True,
        cf_logits=torch.randn(6, 2, requires_grad=True),
        cf_z_id=torch.randn(6, 4, requires_grad=True),
        z_id_for_cf=z_id,
        cf_env_logits=torch.randn(6, 3, requires_grad=True),
        target_env=receiver,
        cf_h=torch.randn(6, 4, requires_grad=True),
        target_h=torch.randn(6, 4),
    )

    assert result.info.active_components == frozenset(
        {"id", "lodo", "counterfactual", "hdro", "csd", "fac"}
    )
    assert all(
        torch.isfinite(value)
        for value in (
            result.total,
            result.id_loss,
            result.lodo_loss,
            result.cf_loss,
            result.hdro_loss,
            result.csd_loss,
            result.fac_loss,
        )
    )
    result.total.backward()
    assert torch.isfinite(common_logits.grad).all()
    assert torch.isfinite(specific_logits.grad).all()
    assert torch.isfinite(z_id.grad).all()


def test_counterfactual_subset_uses_its_own_same_tx_labels() -> None:
    base = SimpleNamespace(
        common_logits=torch.randn(4, 2, requires_grad=True),
        z_id=torch.randn(4, 3, requires_grad=True),
    )
    counterfactual = SimpleNamespace(
        cf_logits=torch.randn(2, 2, requires_grad=True),
        cf_z_id=torch.randn(2, 3, requires_grad=True),
        z_id=torch.randn(2, 3, requires_grad=True),
        cf_env_logits=torch.randn(2, 3, requires_grad=True),
        target_env=torch.tensor([1, 2]),
        cf_h=torch.randn(2, 4, requires_grad=True),
        target_h=torch.randn(2, 4),
        labels=torch.tensor([0, 1]),
    )

    result = compose_hcfdg_loss(
        base,
        labels=torch.tensor([0, 1, 0, 1]),
        use_lodo=False,
        use_counterfactual=True,
        use_hdro=False,
        use_csd=False,
        use_fac=False,
        counterfactual_output=counterfactual,
    )

    assert torch.isfinite(result.total)
