import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from model_dual_cvsincnet import build_dual_model  # noqa: E402
from ntrs import ntrs_safe_fuse_logits  # noqa: E402


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


def test_ntrs_is_identity_only_and_exposes_factorized_safe_outputs():
    model = _tiny_model(
        use_ntrs=True,
        ntrs_rank=8,
        ntrs_alpha_max=0.20,
        ntrs_q_dim=16,
        ntrs_fast_dim=8,
        ntrs_slow_dim=8,
        ntrs_metadata_dim=3,
    )
    model.train()
    out = model(
        torch.randn(2, 2, 64),
        y_tx=torch.tensor([0, 1]),
        return_aux=True,
        domain_labels=torch.tensor([0, 1]),
        ntrs_epoch=68,
        update_ntrs_source=True,
        ntrs_metadata=torch.randn(2, 3),
        ntrs_metadata_valid=torch.ones(2, dtype=torch.bool),
    )

    assert out["ntrs_enabled"] is True
    assert out["ntrs_raw_logits"].shape == (2, 3)
    assert out["ntrs_robust_logits"].shape == (2, 3)
    assert out["tx_logits"].shape == (2, 3)
    assert out["ntrs_z_anchor"].shape == out["ntrs_z_rob"].shape == out["z_id"].shape
    assert out["ntrs_receiver_logits"].shape == (2, 2)
    assert out["ntrs_day_logits"].shape == (2, 2)
    assert out["ntrs_channel_logits"].shape == (2, 2)
    assert out["ntrs_context_tx_adv_logits"].shape == (2, 3)
    assert out["ntrs_safe_gate"].shape == (2,)
    assert out["aux_id"].get("ntrs_physical_view", False) is False
    assert out["aux_phys"].get("ntrs_physical_view", False) is True
    assert out["aux_dom"].get("ntrs_physical_view", False) is False


def test_ntrs_robust_head_is_independent_but_initialized_from_raw_cosface():
    model = _tiny_model(use_ntrs=True, ntrs_q_dim=16, ntrs_fast_dim=8, ntrs_slow_dim=8)
    raw_weight = model.id_backbone.cls_head.head.weight
    robust_weight = model.ntrs_robust_head.weight

    assert raw_weight.data_ptr() != robust_weight.data_ptr()
    assert torch.allclose(raw_weight, robust_weight)


def test_ntrs_v2_identity_bypass_is_bitwise_raw_and_skips_all_ntrs_modules():
    model = _tiny_model(
        use_ntrs=True,
        ntrs_variant="v2_min",
        ntrs_identity_bypass=True,
        ntrs_q_dim=16,
        ntrs_fast_dim=8,
    ).eval()
    x = torch.randn(2, 2, 64)
    labels = torch.tensor([0, 1])
    with torch.no_grad():
        out = model(x, y_tx=labels, return_aux=True, ntrs_epoch=200)

    assert torch.equal(out["tx_logits"], out["ntrs_raw_logits"])
    assert torch.equal(out["z_id"], out["ntrs_z_anchor"])
    assert torch.equal(out["ntrs_z_rob"], out["ntrs_z_anchor"])
    assert torch.count_nonzero(out["ntrs_correction"]) == 0
    assert out["ntrs_identity_bypass"] is True
    assert out["aux_phys"] == {}


def test_ntrs_v2_identity_bypass_does_not_shift_core_initialization_rng():
    torch.manual_seed(392034)
    control = _tiny_model(use_ntrs=False)
    control_core = {
        name: value.detach().clone()
        for name, value in control.state_dict().items()
        if "ntrs_" not in name
    }
    torch.manual_seed(392034)
    bypass = _tiny_model(
        use_ntrs=True,
        ntrs_variant="v2_min",
        ntrs_identity_bypass=True,
        ntrs_q_dim=16,
        ntrs_fast_dim=8,
    )
    bypass_core = {
        name: value.detach().clone()
        for name, value in bypass.state_dict().items()
        if "ntrs_" not in name
    }
    assert control_core.keys() == bypass_core.keys()
    assert all(torch.equal(control_core[name], bypass_core[name]) for name in control_core)


def test_ntrs_v2_min_uses_shared_head_no_layernorm_and_one_identity_forward():
    model = _tiny_model(
        use_ntrs=True,
        ntrs_variant="v2_min",
        ntrs_q_dim=16,
        ntrs_fast_dim=8,
    )
    assert model.ntrs_robust_head is None
    assert not any(isinstance(module, torch.nn.LayerNorm) for module in model.ntrs_context.modules())
    assert not any(isinstance(module, torch.nn.LayerNorm) for module in model.ntrs_robustifier.modules())

    identity_calls = []
    hook = model.id_backbone.register_forward_pre_hook(lambda _module, _args: identity_calls.append(1))
    try:
        out_frozen = model(
            torch.randn(2, 2, 64),
            y_tx=torch.tensor([0, 1]),
            return_aux=True,
            ntrs_epoch=90,
        )
        out_active = model(
            torch.randn(2, 2, 64),
            y_tx=torch.tensor([0, 1]),
            return_aux=True,
            ntrs_epoch=130,
        )
    finally:
        hook.remove()

    assert len(identity_calls) == 2
    assert torch.equal(out_frozen["ntrs_z_rob"], out_frozen["ntrs_z_anchor"])
    assert torch.equal(out_frozen["ntrs_robust_logits"], out_frozen["ntrs_raw_logits"])
    assert out_active["ntrs_shared_head"] is True


def test_ntrs_safe_fusion_falls_back_to_raw_on_disagreement_or_excess_energy():
    raw = torch.tensor([[4.0, 1.0, 0.0], [4.0, 1.0, 0.0], [4.0, 1.0, 0.0]])
    robust = torch.tensor([[1.0, 5.0, 0.0], [5.0, 2.0, 0.0], [5.0, 2.0, 0.0]])
    gate = torch.ones(3)
    energy = torch.tensor([0.01, 0.01, 0.30])

    fused, safe_gate, agreement = ntrs_safe_fuse_logits(
        raw,
        robust,
        gate,
        correction_energy=energy,
        energy_threshold=0.10,
        unknown_rescue=False,
    )

    assert agreement.tolist() == [False, True, True]
    assert safe_gate.tolist() == [0.0, 1.0, 0.0]
    assert torch.equal(fused[0], raw[0])
    assert torch.equal(fused[1], robust[1])
    assert torch.equal(fused[2], raw[2])


def test_ntrs_physical_view_keeps_original_iq_for_pa_and_domain_paths():
    model = _tiny_model(
        use_ntrs=True,
        ntrs_q_dim=16,
        ntrs_fast_dim=8,
        ntrs_slow_dim=8,
        ntrs_metadata_dim=3,
    )
    with torch.no_grad():
        model.ntrs_corrector.parameter_head.bias.fill_(0.30)
        model.ntrs_corrector.gate_head.bias.fill_(3.0)
    pa_inputs = []
    domain_inputs = []

    def _capture_pa(_module, args):
        pa_inputs.append(args[0].detach().clone())

    def _capture_domain(_module, args):
        domain_inputs.append(args[0].detach().clone())

    pa_hook = model.id_backbone.pa_lift.register_forward_pre_hook(_capture_pa)
    dom_hook = model.dom_backbone.register_forward_pre_hook(_capture_domain)
    x = torch.randn(2, 2, 64)
    try:
        out = model(
            x,
            y_tx=torch.tensor([0, 1]),
            return_aux=True,
            domain_labels=torch.tensor([0, 1]),
            ntrs_epoch=68,
            ntrs_metadata=torch.randn(2, 3),
            ntrs_metadata_valid=torch.ones(2, dtype=torch.bool),
        )
    finally:
        pa_hook.remove()
        dom_hook.remove()

    assert len(pa_inputs) == 2
    assert all(torch.allclose(value, x) for value in pa_inputs)
    assert len(domain_inputs) == 1
    assert torch.allclose(domain_inputs[0], x)
    assert torch.all(out["ntrs_physical_correction_energy"] > 0.0)


def test_frequency_dual_view_changes_frequency_embedding_without_changing_pa_embedding():
    model = _tiny_model(use_ntrs=True, ntrs_q_dim=16, ntrs_fast_dim=8, ntrs_slow_dim=8)
    backbone = model.id_backbone.eval()
    raw = torch.randn(2, 2, 64)
    # A circular shift preserves FFT magnitude and can legitimately leave the
    # magnitude-only frequency branch unchanged. Use a deterministic spectral
    # shaping perturbation to verify the dual-view selector instead.
    envelope = torch.linspace(0.25, 1.75, raw.size(-1)).view(1, 1, -1)
    corrected = raw * envelope

    with torch.no_grad():
        raw_mix = backbone(
            corrected,
            return_aux=True,
            original_iq=raw,
            frequency_dual_mix=torch.zeros(2),
        )
        corrected_mix = backbone(
            corrected,
            return_aux=True,
            original_iq=raw,
            frequency_dual_mix=torch.ones(2),
        )

    assert raw_mix["ntrs_physical_view"] is True
    assert corrected_mix["ntrs_frequency_dual_view"] is True
    assert not torch.allclose(raw_mix["f_emb"], corrected_mix["f_emb"])
    assert torch.allclose(raw_mix["pa_local"], corrected_mix["pa_local"])
