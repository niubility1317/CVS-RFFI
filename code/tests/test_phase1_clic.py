from __future__ import annotations

"""RED contract tests for the frozen Phase1 CLIC token operators."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.phase1_clic import (  # noqa: E402
    CLIC_EMBED_DIM,
    CLIC_EXTRA_PARAMETER_COUNT,
    CLIC_INIT_SEED,
    CLIC_INPUT_LENGTH,
    CLIC_LAGS,
    CLICConfig,
    CLICConfigError,
    CLICForwardResult,
    CLICFusion,
    CLICRuntimeError,
    CLICTokenBatch,
    clic_state_sha256,
    initialize_clic_module_,
    totalized_clic_tokens,
)
from model import build_model  # noqa: E402
from model_dual_cvsincnet import build_dual_model  # noqa: E402
from post_stage_common import build_baseline_model, merge_checkpoint_args  # noqa: E402


def _lite_backbone_kwargs(**overrides):
    """Construct the smallest real CVSincNet with the frozen d=160 head."""

    kwargs = {
        "num_classes": 4,
        "model_size": "S",
        "dataset": "wisig",
        "input_len": 256,
        "sample_rate_hz": 25e6,
        "model_variant": "lite_d",
        "branch_ablation": "time_only",
    }
    kwargs.update(overrides)
    return kwargs


def build_lite_clic_backbone(
    *,
    operator_mode: str = "raw_phase_control",
    frozen_mode: bool = True,
    **overrides,
):
    kwargs = _lite_backbone_kwargs(**overrides)
    kwargs.update(
        phase1_clic_frozen_mode=bool(frozen_mode),
        phase1_clic_operator_mode=str(operator_mode),
    )
    return build_model(**kwargs)


def build_lite_dual_clic_model(
    *,
    operator_mode: str = "raw_phase_control",
    frozen_mode: bool = True,
    **overrides,
):
    kwargs = _lite_backbone_kwargs(**overrides)
    kwargs.pop("num_classes", None)
    kwargs.pop("sample_rate_hz", None)
    return build_dual_model(
        num_classes=4,
        num_domains=2,
        model_size=kwargs.pop("model_size"),
        dataset=kwargs.pop("dataset"),
        input_len=kwargs.pop("input_len"),
        sample_rate_hz=25e6,
        domain_enhancer="off",
        fast_infer_when_no_aux=False,
        phase1_clic_frozen_mode=bool(frozen_mode),
        phase1_clic_operator_mode=str(operator_mode),
        **kwargs,
    )


def _nonzero_iq(*, batch: int = 2, length: int = CLIC_INPUT_LENGTH) -> torch.Tensor:
    t = torch.arange(length, dtype=torch.float32)
    phase = 0.17 * t.square() / length + 0.031 * t
    phase = phase.unsqueeze(0) + torch.arange(batch, dtype=torch.float32).unsqueeze(1) * 0.23
    return torch.stack((phase.cos(), phase.sin()), dim=1)


def _apply_complex_gain_phase_cfo(
    x: torch.Tensor,
    *,
    magnitude: float,
    phase: float,
    omega: float,
) -> torch.Tensor:
    t = torch.arange(x.shape[-1], dtype=x.dtype, device=x.device)
    z = torch.complex(x[:, 0], x[:, 1])
    transform = magnitude * torch.exp(1j * (phase + omega * t))
    transformed = z * transform
    return torch.stack((transformed.real, transformed.imag), dim=1)


def _cuda_rng_snapshot() -> list[torch.Tensor]:
    """Capture every currently available CUDA generator without skipping."""

    return [state.clone() for state in torch.cuda.get_rng_state_all()]


def _assert_rng_snapshot_unchanged(
    cpu_before: torch.Tensor,
    cpu_after: torch.Tensor,
    cuda_before: list[torch.Tensor],
    cuda_after: list[torch.Tensor],
) -> None:
    assert torch.equal(cpu_before, cpu_after)
    assert len(cuda_before) == len(cuda_after)
    for before, after in zip(cuda_before, cuda_after):
        assert torch.equal(before, after)


def _assert_finite_nonzero_gradient(tensor: torch.Tensor) -> None:
    assert tensor.grad is not None
    assert bool(torch.isfinite(tensor.grad).all().item())
    assert int(torch.count_nonzero(tensor.grad).item()) > 0


def test_clic_public_api_imports():
    assert CLIC_LAGS == (1, 2, 4, 8)
    assert CLIC_INPUT_LENGTH == 256
    assert CLIC_EMBED_DIM == 160
    assert CLIC_INIT_SEED == 7281164
    assert CLICConfig is not None
    assert CLICTokenBatch is not None
    assert callable(totalized_clic_tokens)


def test_clic_fusion_public_api_imports():
    assert CLIC_EXTRA_PARAMETER_COUNT == 32529
    assert CLICForwardResult is not None
    assert CLICFusion is not None
    assert callable(initialize_clic_module_)
    assert callable(clic_state_sha256)


def test_clic_token_shape_and_fixed_lags():
    x = torch.randn(3, 2, 256, dtype=torch.float32)
    c = totalized_clic_tokens(x, operator_mode="raw_phase_control")
    g = totalized_clic_tokens(x, operator_mode="complex_local_invariant_curvature")

    assert CLIC_LAGS == (1, 2, 4, 8)
    assert c.tokens.shape == g.tokens.shape == (3, 16, 256)
    assert c.valid_mask.shape == g.valid_mask.shape == (3, 4, 256)
    assert c.reliability.shape == g.reliability.shape == (3, 4, 256)
    assert c.valid_mask.dtype is torch.bool
    assert g.valid_mask.dtype is torch.bool
    assert torch.all((c.reliability >= 0) & (c.reliability <= 1))
    assert torch.all((g.reliability >= 0) & (g.reliability <= 1))


def test_clic_zero_domain_is_totalized_and_nonfinite_fails_closed():
    zero = torch.zeros(2, 2, 256)
    out = totalized_clic_tokens(zero, operator_mode="complex_local_invariant_curvature")

    assert torch.count_nonzero(out.tokens) == 0
    assert torch.count_nonzero(out.valid_mask) == 0
    assert torch.count_nonzero(out.reliability) == 0
    assert torch.count_nonzero(out.valid_fraction) == 0
    assert torch.count_nonzero(out.reliability_mean) == 0

    bad = zero.clone()
    bad[0, 0, 9] = float("nan")
    with pytest.raises(CLICRuntimeError, match="non-finite"):
        totalized_clic_tokens(bad, operator_mode="complex_local_invariant_curvature")


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
def test_clic_any_nonfinite_received_iq_fails_closed(bad_value: float):
    bad = torch.ones(1, 2, 256, dtype=torch.float32)
    bad[0, 1, 9] = bad_value
    with pytest.raises(CLICRuntimeError, match="non-finite"):
        totalized_clic_tokens(bad, operator_mode="raw_phase_control")


def test_g_operator_is_invariant_to_complex_gain_phase_and_linear_cfo():
    x = _nonzero_iq()
    transformed = _apply_complex_gain_phase_cfo(
        x,
        magnitude=1.7,
        phase=0.4,
        omega=0.03,
    )

    before = totalized_clic_tokens(x, operator_mode="complex_local_invariant_curvature")
    after = totalized_clic_tokens(
        transformed,
        operator_mode="complex_local_invariant_curvature",
    )

    torch.testing.assert_close(before.tokens, after.tokens, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(before.valid_mask, after.valid_mask)
    torch.testing.assert_close(before.reliability, after.reliability, rtol=2e-5, atol=2e-6)


def test_short_input_and_unknown_operator_fail_closed():
    with pytest.raises(CLICConfigError):
        totalized_clic_tokens(torch.zeros(1, 2, 16), operator_mode="raw_phase_control")
    with pytest.raises(CLICConfigError):
        totalized_clic_tokens(torch.zeros(1, 2, 256), operator_mode="other")


def test_clic_fusion_parameter_count_state_sha_and_rng_restore():
    outer_cpu = torch.random.get_rng_state().clone()
    outer_cuda = _cuda_rng_snapshot()
    try:
        torch.manual_seed(99)
        caller_cpu_before = torch.random.get_rng_state().clone()
        caller_cuda_before = _cuda_rng_snapshot()
        c = CLICFusion(embed_dim=160, input_length=256)
        caller_cpu_after = torch.random.get_rng_state().clone()
        caller_cuda_after = _cuda_rng_snapshot()
        _assert_rng_snapshot_unchanged(
            caller_cpu_before,
            caller_cpu_after,
            caller_cuda_before,
            caller_cuda_after,
        )
        assert sum(parameter.numel() for parameter in c.parameters()) == 32529

        torch.manual_seed(123)
        caller_cpu_before_g = torch.random.get_rng_state().clone()
        caller_cuda_before_g = _cuda_rng_snapshot()
        g = CLICFusion(embed_dim=160, input_length=256)
        caller_cpu_after_g = torch.random.get_rng_state().clone()
        caller_cuda_after_g = _cuda_rng_snapshot()
        _assert_rng_snapshot_unchanged(
            caller_cpu_before_g,
            caller_cpu_after_g,
            caller_cuda_before_g,
            caller_cuda_after_g,
        )
        assert sum(parameter.numel() for parameter in g.parameters()) == 32529
        assert clic_state_sha256(c) == clic_state_sha256(g)
    finally:
        torch.random.set_rng_state(outer_cpu)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(outer_cuda)


def test_c_and_g_share_module_shape_and_emit_quality():
    module = CLICFusion(embed_dim=160, input_length=256)
    x = _nonzero_iq(batch=4, length=256)
    z = torch.randn(4, 160)
    outputs = []
    for mode in ("raw_phase_control", "complex_local_invariant_curvature"):
        out = module(x, z, operator_mode=mode)
        assert isinstance(out, CLICForwardResult)
        assert out.z_id.shape == (4, 160)
        assert out.q_clic.shape == (4, 4)
        assert out.token_batch.tokens.shape == (4, 16, 256)
        assert out.token_batch.valid_mask.shape == (4, 4, 256)
        outputs.append(out)
    assert outputs[0].z_id.shape == outputs[1].z_id.shape
    assert outputs[0].q_clic.shape == outputs[1].q_clic.shape


def test_all_domain_outside_positions_return_bit_exact_base_and_full_fallback():
    module = CLICFusion(embed_dim=160, input_length=256)
    z = torch.randn(2, 160)
    out = module(
        torch.zeros(2, 2, 256),
        z,
        operator_mode="complex_local_invariant_curvature",
    )
    expected_q_clic = torch.tensor(
        [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=z.dtype,
    )
    assert torch.equal(out.z_id, z)
    assert torch.equal(out.q_clic, expected_q_clic)
    assert torch.equal(out.token_batch.valid_mask, torch.zeros_like(out.token_batch.valid_mask))


def test_valid_clic_path_is_not_an_always_base_identity():
    module = CLICFusion(embed_dim=160, input_length=256)
    x = _nonzero_iq(batch=2, length=256)
    z = torch.randn(2, 160)
    out = module(x, z, operator_mode="complex_local_invariant_curvature")
    assert torch.all(out.q_clic[:, 0] > 0)
    assert not torch.equal(out.z_id, z)


def test_fusion_has_live_token_encoder_correction_gate_and_base_gradients():
    module = CLICFusion(embed_dim=160, input_length=256)
    x = _nonzero_iq(batch=4, length=256)
    z = torch.randn(4, 160, requires_grad=True)
    result = module(x, z, operator_mode="complex_local_invariant_curvature")
    loss = result.z_id.square().mean() + result.q_clic[:, 0].square().mean()
    loss.backward()

    _assert_finite_nonzero_gradient(module.depthwise.weight)
    _assert_finite_nonzero_gradient(module.pointwise.weight)
    _assert_finite_nonzero_gradient(module.embed.weight)
    _assert_finite_nonzero_gradient(module.correction.weight)
    _assert_finite_nonzero_gradient(module.gate.weight)
    _assert_finite_nonzero_gradient(z)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
def test_clic_fusion_nonfinite_inputs_fail_closed(bad_value: float):
    module = CLICFusion(embed_dim=160, input_length=256)
    base_x = _nonzero_iq(batch=1, length=256)
    base_z = torch.zeros(1, 160)
    for mode in ("raw_phase_control", "complex_local_invariant_curvature"):
        bad_x = base_x.clone()
        bad_x[0, 0, 9] = bad_value
        with pytest.raises(CLICRuntimeError, match="non-finite"):
            module(bad_x, base_z, operator_mode=mode)

        bad_z = base_z.clone()
        bad_z[0, 0] = bad_value
        with pytest.raises(CLICRuntimeError, match="non-finite"):
            module(base_x, bad_z, operator_mode=mode)


def test_clic_fusion_nonfinite_intermediate_fails_closed():
    module = CLICFusion(embed_dim=160, input_length=256)
    with torch.no_grad():
        module.depthwise.weight[0, 0, 0] = float("nan")
    with pytest.raises(CLICRuntimeError, match="non-finite"):
        module(
            _nonzero_iq(batch=1, length=256),
            torch.zeros(1, 160),
            operator_mode="complex_local_invariant_curvature",
        )


def test_clic_runs_before_the_only_exact_identity_head():
    model = build_lite_dual_clic_model(
        operator_mode="complex_local_invariant_curvature",
    )
    model.eval()
    events = []
    clic = getattr(model.id_backbone, "clic", None)
    assert isinstance(clic, CLICFusion)
    assert getattr(model.dom_backbone, "clic", None) is None
    clic_hook = clic.register_forward_pre_hook(lambda *_: events.append("clic"))
    head_hook = model.id_backbone.cls_head.head.register_forward_pre_hook(
        lambda *_: events.append("head")
    )
    try:
        with torch.no_grad():
            out = model(
                _nonzero_iq(batch=2),
                y_tx=torch.tensor([0, 1]),
                return_aux=True,
            )
    finally:
        clic_hook.remove()
        head_hook.remove()

    assert events == ["clic", "head"]
    assert out["z_id"].shape == (2, 160)
    assert out["z_dom"].shape == (2, 160)
    assert out["q_clic"].shape == (2, 4)
    assert out["tx_logits"].shape == (2, 4)
    assert out["aux_id"]["feat_joint_base"].shape == (2, 160)
    assert out["aux_id"]["feat_joint"].shape == (2, 160)
    assert out["aux_id"]["z_id"].shape == (2, 160)
    assert out["aux_id"]["q_clic"].shape == (2, 4)


def test_fast_and_aux_logits_share_the_same_clic_prehead_path():
    model = build_lite_clic_backbone(operator_mode="raw_phase_control")
    model.eval()
    x = _nonzero_iq(batch=2)
    with torch.no_grad():
        fast = model(x, return_aux=False)
        aux = model(x, return_aux=True)
    torch.testing.assert_close(fast, aux["logits"], rtol=0, atol=0)
    torch.testing.assert_close(aux["feat_joint"], aux["z_id"], rtol=0, atol=0)
    assert aux["feat_joint_base"].shape == (2, 160)
    assert aux["q_clic"].shape == (2, 4)


def test_domain_backbone_is_features_only_and_never_owns_or_calls_a_head():
    model = build_lite_dual_clic_model(operator_mode="raw_phase_control")
    model.eval()
    calls = []
    assert getattr(model.dom_backbone, "clic", None) is None
    handle = model.dom_backbone.cls_head.head.register_forward_hook(
        lambda *_: calls.append("domain-head")
    )
    try:
        with torch.no_grad():
            out = model(_nonzero_iq(batch=2), return_aux=True)
    finally:
        handle.remove()
    assert calls == []
    assert out["z_dom"].shape == (2, 160)
    assert "logits" not in out["aux_dom"]


def test_disabled_clic_preserves_the_legacy_backbone_contract():
    model = build_lite_clic_backbone(frozen_mode=False)
    model.eval()
    assert getattr(model, "clic", None) is None
    x = _nonzero_iq(batch=2)
    with torch.no_grad():
        fast = model(x, return_aux=False)
        aux = model(x, return_aux=True)
    torch.testing.assert_close(fast, aux["logits"], rtol=0, atol=0)
    for key in ("feat_cls", "feat_imp", "feat_dac", "feat_pa", "feat_con", "feat_joint"):
        assert key in aux
    for key in ("feat_joint_base", "z_id", "q_clic"):
        assert key not in aux


def test_disabled_clic_preserves_the_legacy_dual_contract():
    model = build_lite_dual_clic_model(frozen_mode=False)
    model.eval()
    assert getattr(model.id_backbone, "clic", None) is None
    with torch.no_grad():
        out = model(_nonzero_iq(batch=2), return_aux=True)
    assert out["tx_logits"].shape == (2, 4)
    assert out["z_id"].shape == (2, 160)
    assert out["z_dom"].shape == (2, 160)
    assert "q_clic" not in out


def test_clic_rejects_single_parameter_matched_representation():
    with pytest.raises(ValueError, match="single_parameter_matched|CLIC"):
        build_lite_dual_clic_model(
            representation_mode="single_parameter_matched",
        )


def test_clic_rejects_a_domain_backbone_clic_module():
    model = build_lite_dual_clic_model(frozen_mode=False)
    model.dom_backbone.clic = CLICFusion(embed_dim=160, input_length=256)
    with pytest.raises(ValueError, match="domain.*CLIC|CLIC.*domain"):
        model(_nonzero_iq(batch=2), return_aux=True)


def test_clic_rejects_wrong_shape_identity_head():
    with pytest.raises(ValueError, match="160|head"):
        build_lite_clic_backbone(model_variant="lite_c")


def test_clic_rejects_a_non_trainable_exact_identity_head():
    model = build_lite_clic_backbone()
    model.eval()
    model.cls_head.head.weight.requires_grad_(False)
    with pytest.raises(ValueError, match="trainable|requires_grad|head"):
        model(_nonzero_iq(batch=2), return_aux=False)


def test_clic_rejects_an_invalid_operator_mode_fail_closed():
    with pytest.raises(ValueError, match="operator|mode"):
        build_lite_clic_backbone(operator_mode="not-a-clic-operator")


def test_checkpoint_and_model_kwargs_reach_the_clic_constructor():
    ckpt = {
        "args": {
            "dataset": "wisig",
            "model_variant": "lite_d",
        }
    }
    cli_args = SimpleNamespace(
        phase1_clic_frozen_mode=True,
        phase1_clic_operator_mode="complex_local_invariant_curvature",
        model_variant="lite_d",
    )
    model_args = merge_checkpoint_args(
        ckpt,
        cli_args,
        input_len=256,
        num_domains=2,
    )
    assert model_args.phase1_clic_frozen_mode is True
    assert model_args.phase1_clic_operator_mode == "complex_local_invariant_curvature"
    model = build_baseline_model(model_args, torch.device("cpu"))
    assert isinstance(model.id_backbone.clic, CLICFusion)
    assert model.id_backbone.clic_operator_mode == "complex_local_invariant_curvature"
    assert getattr(model.dom_backbone, "clic", None) is None
