import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from model_dual_cvsincnet import build_dual_model  # noqa: E402


def _tiny_model(**kwargs):
    return build_dual_model(
        num_classes=3,
        num_domains=2,
        model_size="S",
        dataset="wisig",
        input_len=64,
        sample_rate_hz=25e6,
        model_variant="lite_h",
        id_feature_key="feat_joint",
        dom_feature_key="feat_imp",
        **kwargs,
    )


def test_crra_is_only_enabled_on_identity_backbone():
    model = _tiny_model(use_crra=True, crra_rank=4, crra_alpha_max=0.25)
    out = model(torch.randn(2, 2, 64), return_aux=True)
    assert out["aux_id"]["crra_enabled"] is True
    assert out["aux_dom"].get("crra_enabled", False) is False
    assert "crra_correction_energy" in out
    assert out["crra_condition_tx_adv_logits"].shape == (2, 3)


def test_crra_does_not_replace_pa_features():
    model = _tiny_model(use_crra=True, crra_rank=4)
    out = model(torch.randn(2, 2, 64), return_aux=True)
    assert out["id_feat_pa"].shape == out["id_feat_joint"].shape
    assert out["aux_id"]["crra_pa_bypass"] is True


def test_crra_can_be_disabled_without_changing_builder_contract():
    model = _tiny_model(use_crra=False)
    out = model(torch.randn(2, 2, 64), return_aux=True)
    assert out["aux_id"].get("crra_enabled", False) is False


def test_crra_reliability_fuses_time_frequency_and_pa_without_freq_reconstruction():
    model = _tiny_model(use_crra=True, crra_rank=4)
    out = model(
        torch.randn(2, 2, 64),
        return_aux=True,
        domain_labels=torch.tensor([0, 1]),
        crra_epoch=47,
    )
    reliability = out["crra_branch_reliability"]
    assert reliability.shape == (2, 3)
    assert torch.allclose(reliability.sum(dim=1), torch.ones(2), atol=1e-6)
    assert torch.isfinite(reliability).all()
    assert model.id_backbone.crra_freq is None
    assert out["aux_id"]["crra_reliability_pa"].shape == (2,)


def test_crra_support_centres_receive_source_domains_only():
    model = _tiny_model(use_crra=True, crra_rank=4)
    model.train()
    model(
        torch.randn(2, 2, 64),
        return_aux=True,
        domain_labels=torch.tensor([0, 1]),
        update_crra_support=True,
        crra_epoch=1,
    )
    counts = model.id_backbone.crra_time.support.counts
    assert counts.tolist() == [1, 1]
    assert model.dom_backbone.crra_time is None
