from __future__ import annotations

import torch

from model_dual_cvsincnet import build_dual_model


def _count(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def test_single_embedding_exactly_matches_dual_parameter_count() -> None:
    common = {
        "num_classes": 6,
        "num_domains": 8,
        "model_size": "M",
        "dataset": "wisig",
        "input_len": 256,
        "model_variant": "lite_d",
        "branch_ablation": "no_dac",
        "domain_branch_ablation": "no_stats",
        "domain_enhancer": "rcn_stats",
        "domain_enhancer_strength": 0.35,
        "arch_family": "cvsincnet",
    }
    dual = build_dual_model(**common, representation_mode="dual")
    single = build_dual_model(
        **common,
        representation_mode="single_parameter_matched",
    )
    assert _count(single) == _count(dual)
    assert single.dom_backbone is None
    assert single.dom_head is None
    assert single.adv_head is None
    assert single.identity_capacity is not None


def test_single_embedding_forward_has_no_separate_domain_representation() -> None:
    model = build_dual_model(
        num_classes=4,
        num_domains=3,
        input_len=128,
        arch_family="cvcnn",
        representation_mode="single_parameter_matched",
    )
    x = torch.randn(2, 2, 128)
    y = torch.tensor([0, 1], dtype=torch.long)
    out = model(x, y_tx=y, return_aux=True)
    assert out["tx_logits"].shape == (2, 4)
    assert out["z_id"].shape == out["z_dom"].shape
    assert out["z_id"].data_ptr() == out["z_dom"].data_ptr()
    assert out["aux_dom"] == {}
    assert out["domain_branch_ablation"] == "not_present"
    loss = torch.nn.functional.cross_entropy(out["tx_logits"], y)
    loss.backward()
    assert all(
        parameter.grad is not None
        for parameter in model.identity_capacity.parameters()
    )
