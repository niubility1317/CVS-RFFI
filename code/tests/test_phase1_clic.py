from __future__ import annotations

"""RED contract tests for the frozen Phase1 CLIC token operators."""

import sys
import gc
import inspect
import json
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn


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
    CLICTerminalError,
    CLICTokenBatch,
    CLICWarmStartError,
    FORMAL_LEO_WEAK_SCENARIOS,
    clic_state_sha256,
    clic_raw_unscaled_vjp_audit,
    clic_scaled_backward_and_classify,
    initialize_clic_module_,
    new_clic_receipt,
    release_clic_retained_graph_roots,
    strict_clic_warm_start,
    totalized_clic_tokens,
    update_clic_amp_receipt,
    update_clic_common_binding_receipt,
    update_clic_resource_receipt,
    validate_clic_terminal_receipt,
    write_clic_failure_receipt,
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


# ---------------------------------------------------------------------------
# Task 4 RED contracts.  These fixtures intentionally use a tiny synthetic
# model/receipt; they exercise the strict helper interfaces without copying
# any HNCCD/HSCF scientific identity or making C/G a control-vs-active pair.
# ---------------------------------------------------------------------------


class _ClicContractModel(nn.Module):
    """Small real module tree with the frozen ``id_backbone.clic`` path."""

    def __init__(self) -> None:
        super().__init__()
        self.id_backbone = nn.Module()
        self.id_backbone.encoder = nn.Linear(CLIC_EMBED_DIM, CLIC_EMBED_DIM, bias=False)
        self.id_backbone.clic = CLICFusion(embed_dim=CLIC_EMBED_DIM, input_length=CLIC_INPUT_LENGTH)
        self.id_backbone.cls_head = nn.Module()
        self.id_backbone.cls_head.head = nn.Linear(CLIC_EMBED_DIM, 4, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.id_backbone.encoder(x)
        return self.id_backbone.cls_head.head(z)


class _TinyAmpModel(nn.Module):
    """One-parameter model for deterministic finite/overflow AMP paths."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([[1.0]], dtype=torch.float32))
        self.forward_calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.forward_calls += 1
        return x @ self.weight


class _RecoveringScaler:
    """CPU GradScaler double with one skip/backoff and observable call counts."""

    def __init__(self, scale: float = 1.0e5) -> None:
        self._scale = float(scale)
        self._found_nonfinite = False
        self.scale_calls = 0
        self.unscale_calls = 0
        self.step_calls = 0
        self.update_calls = 0

    def get_scale(self) -> float:
        return self._scale

    def scale(self, value: torch.Tensor) -> torch.Tensor:
        self.scale_calls += 1
        return value * self._scale

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        self.unscale_calls += 1
        self._found_nonfinite = False
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                parameter.grad.div_(self._scale)
                if not bool(torch.isfinite(parameter.grad.detach()).all().item()):
                    self._found_nonfinite = True

    def step(self, optimizer: torch.optim.Optimizer):
        self.step_calls += 1
        if self._found_nonfinite:
            return None
        return optimizer.step()

    def update(self) -> None:
        self.update_calls += 1
        if self._found_nonfinite:
            self._scale *= 0.5


class _SavedTensorToken:
    __slots__ = ("tensor", "__weakref__")

    def __init__(self, tensor: torch.Tensor) -> None:
        self.tensor = tensor.detach()


class _FiniteForwardNonfiniteBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value: torch.Tensor) -> torch.Tensor:
        return value.clone()

    @staticmethod
    def backward(ctx, gradient: torch.Tensor) -> torch.Tensor:
        return torch.full_like(gradient, float("nan"))


def _state_without_clic(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
        if not key.startswith("id_backbone.clic.")
    }


def _tensor_state_bytes(value: torch.Tensor) -> bytes:
    return value.detach().cpu().contiguous().numpy().tobytes(order="C")


def _first_existing_key(state: dict[str, torch.Tensor]) -> str:
    return next(iter(state))


def _receipt_base(arm: str) -> dict[str, object]:
    """Canonical scalar/count/SHA-only starting receipt for either active arm."""

    receipt = new_clic_receipt(arm=arm)
    receipt.update(
        {
            "schema": "cvs.phase1.clic_receipt.v1",
            "method": "P1_CLIC",
            "arm": arm,
            "operator_id": "C_RAW_PHASE_CONTROL" if arm == "C" else "G_INVARIANT_CURVATURE",
            "batch_size": 128,
            "input_length": CLIC_INPUT_LENGTH,
            "embed_dim": CLIC_EMBED_DIM,
            "local_class_count": 4,
            "lags": [1, 2, 4, 8],
            "clip": 8,
            "checkpoint_sha256": "a" * 64,
            "final_checkpoint_sha256": "f" * 64,
            "existing_state_sha256": "b" * 64,
            "clic_init_state_sha256": "c" * 64,
            "clic_parameter_count": CLIC_EXTRA_PARAMETER_COUNT,
            "existing_state_unchanged": True,
            "optimizer_state_restored": False,
            "rng_state_restored": False,
            "new_adamw": True,
            "source_split_count": 4,
            "source_split_sha256": "1" * 64,
            "class_order_count": 4,
            "class_order_sha256": "2" * 64,
            "physical_order_count": 128,
            "physical_order_sha256": "3" * 64,
            "common_batch_sequence_sha256": "4" * 64,
            "common_batch_sequence_batches": 0,
            "common_batch_sequence_rows": 0,
            "common_scenario_batches": {scene: 0 for scene in FORMAL_LEO_WEAK_SCENARIOS},
            "use_target": False,
            "use_proxy": False,
            "use_held": False,
            "use_u": False,
            "query_truth_access": False,
            "query_role_access": False,
            "new_view_count": 0,
            "second_forward_count": 0,
            "state_feedback_count": 0,
            "legacy_method_identity": False,
            "scene_audits": {},
            "resource_observations": [],
            "amp_attempts": 0,
            "scaled_backward_count": 0,
            "unscale_count": 0,
            "optimizer_step_attempts": 0,
            "effective_optimizer_steps": 0,
            "raw_finite_overflow_skips": 0,
            "scale_decrease_count": 0,
            "optimizer_unchanged_count": 0,
            "raw_nonfinite_count": 0,
            "material_nonfinite_count": 0,
            "consecutive_overflow_skips": 0,
            "max_consecutive_overflow_skips": 0,
            "persistent_overflow": False,
            "graph_release_count": 0,
            "head_path": "id_backbone.cls_head.head",
            "completed": False,
        }
    )
    return receipt


def _scene_audit() -> dict[str, object]:
    def group() -> dict[str, object]:
        return {"count": 1, "norm": 1.0, "finite": True, "nonzero": True}

    return {
        "valid_token_coverage": 1.0,
        "gate_or_correction_nonzero": True,
        "raw_unscaled": True,
        "diagnostic_only": True,
        "touches_amp_optimizer_rng": False,
        "completed": True,
        "token": group(),
        "clic": group(),
        "base": group(),
        "head": group(),
    }


def _complete_receipt(arm: str) -> dict[str, object]:
    """Build one three-scene C/G receipt through the new scalar helpers."""

    receipt = _receipt_base(arm)
    for index, scene in enumerate(FORMAL_LEO_WEAK_SCENARIOS, start=1):
        receipt = update_clic_common_binding_receipt(
            receipt,
            binding={
                "scene": scene,
                "batch_index": index,
                "rows": 128,
                "source_split_count": 4,
                "source_split_sha256": "1" * 64,
                "class_order_count": 4,
                "class_order_sha256": "2" * 64,
                "physical_order_count": 128,
                "physical_order_sha256": "3" * 64,
                "common_batch_sequence_sha256": f"{index:064x}"[-64:],
            },
        )
        receipt.setdefault("scene_audits", {})[scene] = _scene_audit()
        receipt = update_clic_resource_receipt(
            receipt,
            observation={
                "scene": scene,
                "batch_index": index,
                "peak_memory_bytes": 2_097_152,
                "step_time_seconds": 0.01 * index,
                "selection_feedback": False,
            },
        )
        receipt = update_clic_amp_receipt(
            receipt,
            event={
                "scene": scene,
                "batch_index": index,
                "amp_overflow_detected": False,
                "scaled_backward_count": 1,
                "unscale_count": 1,
                "optimizer_step_attempted": True,
                "effective_optimizer_step": True,
                "raw_finite": True,
                "scale_decreased": False,
                "optimizer_state_unchanged": False,
                "raw_nonfinite": False,
                "material_nonfinite": False,
            },
        )
    receipt["common_batch_sequence_batches"] = 3
    receipt["common_batch_sequence_rows"] = 3 * 128
    receipt["common_scenario_batches"] = {scene: 1 for scene in FORMAL_LEO_WEAK_SCENARIOS}
    receipt["graph_release_count"] = 3
    receipt["completed"] = True
    return receipt


def _walk_keys(value: object, prefix: str = ""):
    if isinstance(value, dict):
        for key, child in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            yield name
            yield from _walk_keys(child, name)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from _walk_keys(child, f"{prefix}[{index}]")


def test_strict_clic_warm_start_preserves_existing_bytes_and_clic_init_state() -> None:
    model = _ClicContractModel()
    baseline_model = _ClicContractModel()
    baseline = _state_without_clic(baseline_model)
    clic_before = clic_state_sha256(model.id_backbone.clic)
    receipt = strict_clic_warm_start(model, baseline, checkpoint_sha256="a" * 64)

    assert receipt["existing_state_unchanged"] is True
    assert receipt["optimizer_state_restored"] is False
    assert receipt["rng_state_restored"] is False
    assert receipt["clic_parameter_count"] == CLIC_EXTRA_PARAMETER_COUNT
    assert receipt["clic_init_state_sha256"] == clic_state_sha256(model.id_backbone.clic)
    assert receipt["clic_init_state_sha256"] == clic_before
    old_state_sha = receipt["existing_state_sha256"]
    assert isinstance(old_state_sha, str)
    assert old_state_sha == old_state_sha.lower()
    assert len(old_state_sha) == 64
    assert all(character in "0123456789abcdef" for character in old_state_sha)
    for key, value in baseline.items():
        assert _tensor_state_bytes(model.state_dict()[key]) == _tensor_state_bytes(value), key


@pytest.mark.parametrize(
    "mutation",
    ("missing_old", "extra_old", "shape", "dtype", "nonfinite", "checkpoint_clic_key"),
)
def test_strict_clic_warm_start_rejects_key_shape_dtype_and_nonfinite_mutations(mutation: str) -> None:
    model = _ClicContractModel()
    baseline = _state_without_clic(_ClicContractModel())
    key = _first_existing_key(baseline)
    if mutation == "missing_old":
        baseline.pop(key)
    elif mutation == "extra_old":
        baseline["id_backbone.encoder.unexpected"] = torch.zeros_like(baseline[key])
    elif mutation == "shape":
        baseline[key] = baseline[key].reshape(-1)[:1]
    elif mutation == "dtype":
        baseline[key] = baseline[key].double()
    elif mutation == "nonfinite":
        baseline[key] = baseline[key].clone()
        baseline[key].view(-1)[0] = float("nan")
    elif mutation == "checkpoint_clic_key":
        baseline["id_backbone.clic.depthwise.weight"] = model.id_backbone.clic.depthwise.weight.detach().clone()
    with pytest.raises(CLICWarmStartError):
        strict_clic_warm_start(model, baseline, checkpoint_sha256="b" * 64)


@pytest.mark.parametrize("checkpoint_sha256", ("A" * 64, "0" * 63, "g" * 64))
def test_strict_clic_warm_start_requires_lowercase_sha256(checkpoint_sha256: str) -> None:
    with pytest.raises(CLICWarmStartError):
        strict_clic_warm_start(
            _ClicContractModel(),
            _state_without_clic(_ClicContractModel()),
            checkpoint_sha256=checkpoint_sha256,
        )


def test_strict_clic_warm_start_isolated_from_caller_mapping_mutation() -> None:
    model = _ClicContractModel()
    baseline = _state_without_clic(_ClicContractModel())
    strict_clic_warm_start(model, baseline, checkpoint_sha256="a" * 64)
    loaded_bytes = {
        key: _tensor_state_bytes(value)
        for key, value in model.state_dict().items()
        if not key.startswith("id_backbone.clic.")
    }
    key = _first_existing_key(baseline)
    baseline[key].zero_()
    assert _tensor_state_bytes(model.state_dict()[key]) == loaded_bytes[key]
    for name, expected in loaded_bytes.items():
        assert _tensor_state_bytes(model.state_dict()[name]) == expected


def test_clic_raw_unscaled_vjp_audits_token_all_clic_base_and_exact_head_groups() -> None:
    model = _ClicContractModel()
    token = torch.randn(2, 16, CLIC_INPUT_LENGTH, requires_grad=True)
    base_parameter = nn.Parameter(torch.tensor(1.0))
    head_weight = model.id_backbone.cls_head.head.weight
    loss = token.square().mean() + base_parameter.square() + head_weight.square().mean()
    loss = loss + sum(parameter.square().mean() for parameter in model.id_backbone.clic.parameters())
    audit = clic_raw_unscaled_vjp_audit(
        loss,
        token,
        model.id_backbone.clic,
        (base_parameter,),
        head_weight,
    )

    assert audit["raw_unscaled"] is True
    assert audit["diagnostic_only"] is True
    assert audit["touches_amp_optimizer_rng"] is False
    for name in ("token", "clic", "base", "head"):
        summary = audit[name]
        assert summary["count"] > 0
        assert summary["norm"] > 0.0
        assert summary["finite"] is True
        assert summary["nonzero"] is True


@pytest.mark.parametrize(
    "scene,group",
    [(scene, group) for scene in FORMAL_LEO_WEAK_SCENARIOS for group in ("token", "clic", "base", "head")],
)
def test_terminal_rejects_missing_or_zero_scene_vjp_group(scene: str, group: str) -> None:
    receipt = _complete_receipt("G")
    receipt["scene_audits"][scene][group]["norm"] = 0.0
    with pytest.raises(CLICTerminalError, match="VJP|audit|scene"):
        validate_clic_terminal_receipt(receipt, arm="G")


def test_c_and_g_share_one_common_binding_but_both_have_active_scene_audits() -> None:
    c = _complete_receipt("C")
    g = _complete_receipt("G")
    assert c["common_batch_sequence_sha256"] == g["common_batch_sequence_sha256"]
    assert c["common_scenario_batches"] == g["common_scenario_batches"]
    for receipt in (c, g):
        assert set(receipt["scene_audits"]) == set(FORMAL_LEO_WEAK_SCENARIOS)
        assert all(receipt["scene_audits"][scene]["completed"] for scene in FORMAL_LEO_WEAK_SCENARIOS)
        assert all(observation["selection_feedback"] is False for observation in receipt["resource_observations"])
        assert validate_clic_terminal_receipt(receipt, arm=receipt["arm"])["completed"] is True


@pytest.mark.parametrize(
    "field",
    ("source_split_count", "source_split_sha256", "class_order_count", "class_order_sha256", "physical_order_count", "physical_order_sha256"),
)
def test_common_binding_count_or_sha_drift_fails_closed(field: str) -> None:
    receipt = _complete_receipt("C")
    receipt[field] = 0 if field.endswith("count") else "z" * 64
    with pytest.raises(CLICTerminalError, match="binding|source|class|physical|SHA"):
        validate_clic_terminal_receipt(receipt, arm="C")


@pytest.mark.parametrize(
    "mutation",
    ("empty", "missing_one", "bool_peak", "negative_peak", "nan_step", "negative_step", "selection_feedback"),
)
def test_terminal_requires_exact_resource_observation_count_and_rejects_seven_mutations(mutation: str) -> None:
    receipt = _complete_receipt("G")
    observations = [dict(observation) for observation in receipt["resource_observations"]]
    if mutation == "empty":
        observations = []
    elif mutation == "missing_one":
        observations = observations[:-1]
    elif mutation == "bool_peak":
        observations[0]["peak_memory_bytes"] = True
    elif mutation == "negative_peak":
        observations[0]["peak_memory_bytes"] = -1
    elif mutation == "nan_step":
        observations[0]["step_time_seconds"] = float("nan")
    elif mutation == "negative_step":
        observations[0]["step_time_seconds"] = -0.01
    elif mutation == "selection_feedback":
        observations[0]["selection_feedback"] = True
    receipt["resource_observations"] = observations
    with pytest.raises(CLICTerminalError, match="resource|peak|step|selection"):
        validate_clic_terminal_receipt(receipt, arm="G")


def test_terminal_revalidates_contract_after_fake_completed_flag() -> None:
    receipt = _complete_receipt("C")
    receipt["completed"] = True
    receipt["common_batch_sequence_batches"] = 99
    receipt["resource_observations"] = receipt["resource_observations"][:1]
    with pytest.raises(CLICTerminalError, match="common|resource|terminal|count"):
        validate_clic_terminal_receipt(receipt, arm="C")


@pytest.mark.parametrize(
    "forbidden",
    (
        "raw_iq",
        "feature_tensor",
        "logits",
        "physical_id",
        "member_id",
        "receiver_token",
        "target_metric",
        "target_label",
        "role",
        "scorer_path",
        "query_row",
        "legacy_method_identity",
    ),
)
def test_terminal_recursively_rejects_nested_forbidden_receipt_fields(forbidden: str) -> None:
    receipt = _complete_receipt("G")
    receipt["nested"] = {"deep": {forbidden: torch.zeros(1) if forbidden in {"raw_iq", "feature_tensor", "logits"} else "forbidden"}}
    with pytest.raises(CLICTerminalError, match="forbidden|raw|feature|logit|target|query|legacy|ID|receiver|scorer"):
        validate_clic_terminal_receipt(receipt, arm="G")


def _run_clic_amp_batch(*, scaled_overflow: bool, raw_nonfinite: bool = False):
    model = _TinyAmpModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    scaler = _RecoveringScaler(scale=1.0e5)
    output = model(torch.ones(4, 1))
    if raw_nonfinite:
        loss = _FiniteForwardNonfiniteBackward.apply(output).sum()
    elif scaled_overflow:
        loss = output.sum() * 1.0e34
    else:
        loss = output.square().mean()
    before = [parameter.detach().clone() for parameter in model.parameters()]
    info = clic_scaled_backward_and_classify(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        loss=loss,
    )
    return model, optimizer, scaler, output, loss, before, info


def test_clic_amp_finite_path_has_one_scaled_backward_unscale_and_effective_step() -> None:
    model, optimizer, scaler, _output, _loss, _before, info = _run_clic_amp_batch(scaled_overflow=False)
    scaler.step(optimizer)
    scaler.update()
    assert info["amp_overflow_detected"] is False
    assert info["scaled_backward_count"] == 1
    assert info["optimizer_unscale_count"] == 1
    assert scaler.scale_calls == scaler.unscale_calls == 1
    assert scaler.step_calls == scaler.update_calls == 1
    assert model.weight.grad is not None


def test_clic_amp_raw_finite_scaled_overflow_skips_once_without_optimizer_drift() -> None:
    model, optimizer, scaler, _output, _loss, before, info = _run_clic_amp_batch(scaled_overflow=True)
    scaler.step(optimizer)
    scaler.update()
    assert info["amp_overflow_detected"] is True
    assert info["amp_overflow_recoverable"] is True
    assert info["amp_overflow_kind"] == "COMBINED_SCALED_OVERFLOW_RAW_FINITE"
    assert info["scaled_backward_count"] == 1
    assert info["optimizer_unscale_count"] == 1
    assert scaler.scale_calls == scaler.unscale_calls == 1
    assert scaler.step_calls == scaler.update_calls == 1
    assert scaler.get_scale() < 1.0e5
    assert all(torch.equal(before_value, parameter.detach()) for before_value, parameter in zip(before, model.parameters()))


def test_clic_amp_raw_nonfinite_and_material_nonfinite_fail_closed() -> None:
    with pytest.raises(CLICRuntimeError, match="non-finite|raw|material"):
        _run_clic_amp_batch(scaled_overflow=False, raw_nonfinite=True)
    model = _TinyAmpModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    with pytest.raises(CLICRuntimeError, match="non-finite|loss"):
        clic_scaled_backward_and_classify(
            model=model,
            optimizer=optimizer,
            scaler=_RecoveringScaler(scale=1.0),
            loss=torch.tensor(float("nan"), requires_grad=True),
        )


def test_clic_amp_receipt_closes_counts_and_records_overflow_skip() -> None:
    receipt = _receipt_base("G")
    finite = {
        "amp_overflow_detected": False,
        "scaled_backward_count": 1,
        "unscale_count": 1,
        "optimizer_step_attempted": True,
        "effective_optimizer_step": True,
        "raw_finite": True,
        "scale_decreased": False,
        "optimizer_state_unchanged": False,
        "raw_nonfinite": False,
        "material_nonfinite": False,
    }
    receipt = update_clic_amp_receipt(receipt, event=finite)
    overflow = {
        **finite,
        "amp_overflow_detected": True,
        "effective_optimizer_step": False,
        "raw_finite": True,
        "scale_decreased": True,
        "optimizer_state_unchanged": True,
    }
    receipt = update_clic_amp_receipt(receipt, event=overflow)
    assert receipt["amp_attempts"] == 2
    assert receipt["scaled_backward_count"] == 2
    assert receipt["unscale_count"] == 2
    assert receipt["raw_finite_overflow_skips"] == 1
    assert receipt["optimizer_unchanged_count"] == 1
    assert receipt["effective_optimizer_steps"] == 1


def test_terminal_rejects_persistent_overflow_and_zero_effective_optimizer_steps() -> None:
    persistent = _complete_receipt("G")
    persistent["consecutive_overflow_skips"] = 2
    persistent["max_consecutive_overflow_skips"] = 2
    persistent["persistent_overflow"] = True
    with pytest.raises(CLICTerminalError, match="overflow|consecutive|effective"):
        validate_clic_terminal_receipt(persistent, arm="G")

    zero_step = _complete_receipt("G")
    zero_step["effective_optimizer_steps"] = 0
    with pytest.raises(CLICTerminalError, match="effective|optimizer|step"):
        validate_clic_terminal_receipt(zero_step, arm="G")


@pytest.mark.parametrize("scaled_overflow", [False, True])
def test_clic_saved_tensor_graph_roots_release_before_next_forward_without_gc(
    scaled_overflow: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    saved_tokens: list[weakref.ReferenceType[_SavedTensorToken]] = []

    def pack(tensor: torch.Tensor) -> _SavedTensorToken:
        token = _SavedTensorToken(tensor)
        saved_tokens.append(weakref.ref(token))
        return token

    def unpack(token: _SavedTensorToken) -> torch.Tensor:
        return token.tensor

    model = _TinyAmpModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    scaler = _RecoveringScaler(scale=1.0e5 if scaled_overflow else 1.0)
    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        output = model(torch.ones(4, 1))
        loss = output.sum() * 1.0e34 if scaled_overflow else output.square().mean()
        info = clic_scaled_backward_and_classify(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            loss=loss,
        )
        scaler.step(optimizer)
        scaler.update()
    roots = {"output": output, "loss": loss, "alias": output}
    del output, loss
    assert saved_tokens
    assert info["amp_overflow_detected"] is scaled_overflow
    if not scaled_overflow:
        assert any(reference() is not None for reference in saved_tokens)
    release_clic_retained_graph_roots(roots)
    assert roots == {}
    assert all(reference() is None for reference in saved_tokens)
    assert model.forward_calls == 1
    assert scaler.unscale_calls == 1
    source = inspect.getsource(release_clic_retained_graph_roots)
    assert "gc.collect" not in source and "empty_cache" not in source
    gc.collect()
    assert all(reference() is None for reference in saved_tokens)


def test_clic_vjp_and_amp_helpers_do_not_add_a_second_backward_or_unscale() -> None:
    assert inspect.getsource(clic_scaled_backward_and_classify).count(".backward(") == 1
    assert inspect.getsource(clic_scaled_backward_and_classify).count(".unscale_(") == 1
    assert inspect.getsource(clic_raw_unscaled_vjp_audit).count("autograd.grad(") == 1


def test_clic_failure_receipt_is_data_free_and_records_stage_error_and_message_digest(tmp_path: Path) -> None:
    target = write_clic_failure_receipt(
        tmp_path,
        candidate_id="P1_CLIC_G",
        run_id="phase1_clic_g_20260812_v1",
        receipt=_receipt_base("G"),
        error=CLICRuntimeError("raw/material non-finite at common batch 1"),
        failure_stage="amp_classification",
    )
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["schema"] == "cvs.phase1.clic_failure_receipt.v1"
    assert payload["candidate_id"] == "P1_CLIC_G"
    assert payload["run_id"] == "phase1_clic_g_20260812_v1"
    assert payload["failure_stage"] == "amp_classification"
    assert payload["exception_type"] == "CLICRuntimeError"
    assert len(payload["message_digest"]) == 64
    assert not any(
        any(token in name.lower() for token in ("raw_iq", "feature", "logit", "physical_id", "receiver", "target", "query", "scorer"))
        for name in _walk_keys(payload["receipt"])
    )
    assert not any(isinstance(value, torch.Tensor) for value in payload["receipt"].values())
