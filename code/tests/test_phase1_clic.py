from __future__ import annotations

"""RED contract tests for the frozen Phase1 CLIC token operators."""

import sys
import ast
import gc
import inspect
import json
import re
import shlex
import subprocess
import textwrap
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
import cvsrffi.phase1_clic as phase1_clic_module  # noqa: E402
import SSDG.train_ssdg as train_ssdg  # noqa: E402


TRAIN_SCRIPT = CODE_ROOT / "SSDG" / "train_ssdg.py"
CLIC_LAUNCHER = CODE_ROOT / "scripts" / "launch_phase1_clic12_20260811.sh"


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

    clic_groups = {
        name: group()
        for name in ("depthwise", "pointwise", "embed", "correction", "gate")
    }
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
        "clic_groups": clic_groups,
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


@pytest.mark.parametrize("scene", FORMAL_LEO_WEAK_SCENARIOS)
def test_terminal_rejects_scene_without_clic_group_map(scene: str) -> None:
    receipt = _complete_receipt("G")
    del receipt["scene_audits"][scene]["clic_groups"]
    with pytest.raises(CLICTerminalError, match="CLIC VJP group|audit|scene"):
        validate_clic_terminal_receipt(receipt, arm="G")


@pytest.mark.parametrize(
    "scene,group",
    [
        (scene, group)
        for scene in FORMAL_LEO_WEAK_SCENARIOS
        for group in ("depthwise", "pointwise", "embed", "correction", "gate")
    ],
)
def test_terminal_rejects_scene_missing_one_clic_group(scene: str, group: str) -> None:
    receipt = _complete_receipt("G")
    del receipt["scene_audits"][scene]["clic_groups"][group]
    with pytest.raises(CLICTerminalError, match="CLIC VJP group|audit|scene"):
        validate_clic_terminal_receipt(receipt, arm="G")


@pytest.mark.parametrize(
    "scene,group",
    [
        (scene, group)
        for scene in FORMAL_LEO_WEAK_SCENARIOS
        for group in ("depthwise", "pointwise", "embed", "correction", "gate")
    ],
)
def test_terminal_rejects_zero_scene_clic_group(scene: str, group: str) -> None:
    receipt = _complete_receipt("G")
    receipt["scene_audits"][scene]["clic_groups"][group]["norm"] = 0.0
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
        "token_tensor",
        "token_payload",
        "source_label_values",
        "truth",
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


@pytest.mark.parametrize(
    "forbidden",
    ("token_tensor", "token_payload", "source_label_values", "target_metric", "truth"),
)
def test_clic_failure_receipt_rejects_nested_forbidden_fields_before_projection(
    tmp_path: Path, forbidden: str
) -> None:
    receipt = _receipt_base("G")
    receipt["nested"] = {"deep": {forbidden: "forbidden"}}
    with pytest.raises(CLICTerminalError, match="forbidden|receipt|target|truth|token|label"):
        write_clic_failure_receipt(
            tmp_path,
            candidate_id="P1_CLIC_G",
            run_id=f"phase1_clic_g_forbidden_{forbidden}",
            receipt=receipt,
            error=CLICRuntimeError("receipt field policy violation"),
            failure_stage="receipt_validation",
        )


# ---------------------------------------------------------------------------
# Task 5 RED contracts: source-only trainer integration and immutable launcher.
# These tests intentionally fail until Terra wires the frozen trainer path and
# the model exposes the live token tensor from the already executed CLIC seam.
# ---------------------------------------------------------------------------


def _tensor_paths(value: object, target: torch.Tensor, prefix: str = "") -> list[str]:
    paths: list[str] = []
    if torch.is_tensor(value):
        same_object = value is target
        same_storage = (
            value.untyped_storage().data_ptr() == target.untyped_storage().data_ptr()
            and tuple(value.shape) == tuple(target.shape)
        )
        if same_object or same_storage:
            paths.append(prefix or "<root>")
        return paths
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            paths.extend(_tensor_paths(child, target, child_prefix))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            paths.extend(_tensor_paths(child, target, f"{prefix}[{index}]"))
    return paths


def test_train_parser_exposes_frozen_clic_flags_defaults_choices_and_rejection() -> None:
    parser = train_ssdg.build_arg_parser()
    defaults = parser.parse_args(["--output_dir", "clic-red-default"])
    assert defaults.phase1_clic_frozen_mode is False
    assert defaults.phase1_clic_operator_mode == "raw_phase_control"

    args = parser.parse_args(
        [
            "--output_dir",
            "clic-red",
            "--phase1_clic_frozen_mode",
            "true",
            "--phase1_clic_operator_mode",
            "complex_local_invariant_curvature",
        ]
    )
    assert args.phase1_clic_frozen_mode is True
    assert args.phase1_clic_operator_mode == "complex_local_invariant_curvature"
    action = next(
        action
        for action in parser._actions
        if "--phase1_clic_operator_mode" in action.option_strings
    )
    assert tuple(action.choices) == (
        "raw_phase_control",
        "complex_local_invariant_curvature",
    )
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--output_dir",
                "clic-red-invalid",
                "--phase1_clic_operator_mode",
                "not-a-clic-operator",
            ]
        )


def _clic_validation_args(tmp_path: Path):
    """Build parser args that reach CLIC config validation without data access."""

    parser = train_ssdg.build_arg_parser()
    args = parser.parse_args(
        [
            "--output_dir",
            str(tmp_path / "clic-config"),
            "--baseline_ckpt",
            "baseline-for-config-only.pth",
            "--from_scratch",
            "false",
            "--freeze_backbone",
            "false",
            "--amp",
            "true",
            "--epochs",
            "40",
            "--label_epochs",
            "40",
            "--pseudo_epochs",
            "0",
            "--batch_size",
            "128",
            "--checkpoint_selection",
            "final_only",
            "--phase1_clic_frozen_mode",
            "true",
            "--phase1_clic_operator_mode",
            "raw_phase_control",
            "--use_sat_consistency",
            "--lambda_sat_cons",
            "0.10",
            "--lambda_sat_cls",
            "0",
            "--sat_cons_start_epoch",
            "1",
            "--sat_train_scenarios",
            ",".join(FORMAL_LEO_WEAK_SCENARIOS),
            "--sat_view_prob",
            "1.0",
        ]
    )
    # The trainer's CLIC gate consumes this legacy input-length attribute even
    # though the public parser does not expose it as a CLI switch.
    args.wisig_out_len = CLIC_INPUT_LENGTH
    for action in parser._actions:
        if action.dest.startswith("lambda_"):
            setattr(args, action.dest, 0.0)
    args.lambda_sat_cons = 0.10
    args.lambda_sat_cls = 0.0
    for field in (
        "phase1_ccpc_leo_frozen_mode",
        "phase1_ccpc_leo_enabled",
        "phase1_pamr_frozen_mode",
        "phase1_pamr_enabled",
        "phase1_cb_sfce_frozen_mode",
        "phase1_cb_sfce_enabled",
        "phase1_gd_proto_nll_frozen_mode",
        "phase1_gd_proto_nll_enabled",
        "phase1_icmt_frozen_mode",
        "phase1_icmt_enabled",
        "phase1_cagm_frozen_mode",
        "phase1_cagm_enabled",
        "phase1_rcrmd_frozen_mode",
        "phase1_rcrmd_enabled",
        "phase1_rcat_frozen_mode",
        "phase1_rcat_enabled",
        "phase1_rcmmc_frozen_mode",
        "phase1_rcmmc_enabled",
        "phase1_hscf_frozen_mode",
        "phase1_hscf_enabled",
        "phase1_hnccd_frozen_mode",
        "phase1_hnccd_enabled",
        "phase1_recte_frozen_mode",
        "phase1_recte_enabled",
        "phase1_cp_sfce_frozen_mode",
        "phase1_cp_sfce_enabled",
        "manytx_real_oe_enabled",
        "manytx_real_oe_protocol_enabled",
        "use_unlabeled",
        "use_ema_teacher",
        "use_concat_sat_channel_aug",
        "use_aug",
        "use_mixstyle",
        "use_phase2_ground_prototypes",
        "pseudo_domain_gate",
        "pseudo_temporal_gate",
        "use_proto_memory",
        "use_feature_masks",
        "use_txrx_geometry_losses",
        "reject_head",
    ):
        if hasattr(args, field):
            setattr(args, field, False)
    args.sat_view_schedule = ""
    args.phase1_source_proxy_unknown_tx_ids = ""
    return args


def test_clic_zero_gate_covers_every_parser_lambda_except_satellite_consistency() -> None:
    parser = train_ssdg.build_arg_parser()
    parser_lambda_fields = {
        action.dest
        for action in parser._actions
        if action.dest.startswith("lambda_")
    }
    source = inspect.getsource(train_ssdg.train)
    tree = ast.parse(textwrap.dedent(source))
    clic_scopes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and any(
            isinstance(name, ast.Name) and name.id == "clic_frozen_mode_active"
            for name in ast.walk(node.test)
        )
    ]
    config_scopes = [
        scope
        for scope in clic_scopes
        if any(
            isinstance(node, ast.Call) and _ast_call_name(node) == "CLICConfig"
            for node in ast.walk(scope)
        )
    ]
    assert len(config_scopes) == 1, "CLIC config gate scope is absent or duplicated"
    gate_literals = {
        node.value
        for node in ast.walk(config_scopes[0])
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    missing = sorted(parser_lambda_fields - {"lambda_sat_cons"} - gate_literals)
    assert not missing, f"CLIC config zero gate misses parser lambda fields: {missing}"


@pytest.mark.parametrize(
    "lambda_field",
    (
        "lambda_energy_in",
        "lambda_energy_out",
        "lambda_reject_neg",
        "lambda_inter_neg",
        "lambda_shell_neg",
        "lambda_tail_outward_neg",
        "lambda_bridge_neg",
        "lambda_tail_cvar",
        "lambda_overflow_cap",
        "lambda_risk_energy_out",
    ),
)
def test_clic_config_dynamically_rejects_nonzero_parser_lambda(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    lambda_field: str,
) -> None:
    args = _clic_validation_args(tmp_path)
    setattr(args, lambda_field, 1.0)

    def data_must_not_be_reached(*_args, **_kwargs):
        pytest.fail(f"CLIC config accepted nonzero {lambda_field} before data construction")

    monkeypatch.setattr(train_ssdg, "_build_ssdg_wisig_data", data_must_not_be_reached)
    with pytest.raises(CLICConfigError, match=lambda_field):
        train_ssdg.train(args)


def test_clic_config_accepts_declared_proxy_tx_before_source_l_only_data_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A proxy TX role declaration is not permission to load its rows."""

    args = _clic_validation_args(tmp_path)
    args.phase1_source_train_tx_ids = "20-15,20-19,6-15,8-20"
    args.phase1_source_known_validation_tx_ids = "14-7"
    args.phase1_source_proxy_unknown_tx_ids = "14-10"

    class DataBoundaryReached(RuntimeError):
        pass

    def stop_at_source_l_builder(*_args, **_kwargs):
        raise DataBoundaryReached("CLIC config accepted the TX-role manifest")

    monkeypatch.setattr(train_ssdg, "_build_ssdg_wisig_data", stop_at_source_l_builder)
    with pytest.raises(DataBoundaryReached, match="accepted the TX-role manifest"):
        train_ssdg.train(args)


def test_clic_source_l_data_build_rejects_any_loaded_proxy_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _clic_validation_args(tmp_path)
    args.phase1_source_train_tx_ids = "20-15,20-19,6-15,8-20"
    args.phase1_source_known_validation_tx_ids = "14-7"
    args.phase1_source_proxy_unknown_tx_ids = "14-10"

    monkeypatch.setattr(
        train_ssdg,
        "_build_ssdg_wisig_data",
        lambda *_args, **_kwargs: {
            "split_info": {
                "tx_partition_receipt": {
                    "enabled": True,
                    "held_tx_loaded_by_training": True,
                }
            }
        },
    )
    with pytest.raises(CLICConfigError, match="held/proxy TX loaded by training"):
        train_ssdg.train(args)


def test_clic_reads_physical_rows_from_the_real_move_batch_extra_tuple() -> None:
    labels = torch.arange(128, dtype=torch.long) % 4
    domains = torch.arange(128, dtype=torch.long) % 7
    metadata = {
        "base_index": torch.arange(1000, 1128, dtype=torch.long),
        "sig_i": torch.arange(2000, 2128, dtype=torch.long),
    }

    rows = train_ssdg._clic_source_batch_physical_rows(
        (domains, metadata),
        labels,
        expected_rows=128,
    )

    assert rows[0] == (1000, 2000, 0)
    assert rows[-1] == (1127, 2127, 3)
    assert len(rows) == 128


@pytest.mark.parametrize(
    "extra",
    (
        (),
        (torch.zeros(128, dtype=torch.long),),
        (torch.zeros(128, dtype=torch.long), {"base_index": torch.arange(128)}),
    ),
)
def test_clic_physical_row_binding_rejects_missing_real_batch_metadata(extra) -> None:
    with pytest.raises(CLICRuntimeError, match="metadata|binding"):
        train_ssdg._clic_source_batch_physical_rows(
            extra,
            torch.arange(128, dtype=torch.long) % 4,
            expected_rows=128,
        )


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    (
        ("base_index", True),
        ("base_index", 0.5),
        ("base_index", "1"),
        ("base_index", float("nan")),
        ("base_index", -1),
        ("sig_i", False),
        ("sig_i", 1.5),
        ("sig_i", "2"),
        ("sig_i", float("inf")),
        ("sig_i", -1),
        ("label", True),
        ("label", 1.0),
        ("label", "1"),
        ("label", -1),
        ("label", 4),
    ),
)
def test_clic_physical_row_binding_rejects_nonintegral_or_out_of_range_values(
    field: str,
    invalid_value: object,
) -> None:
    base_indices = list(range(1000, 1128))
    signal_indices = list(range(2000, 2128))
    labels = [index % 4 for index in range(128)]
    if field == "base_index":
        base_indices[0] = invalid_value
    elif field == "sig_i":
        signal_indices[0] = invalid_value
    else:
        labels[0] = invalid_value
    extra = (
        torch.zeros(128, dtype=torch.long),
        {"base_index": base_indices, "sig_i": signal_indices},
    )

    with pytest.raises(CLICRuntimeError, match="malformed"):
        train_ssdg._clic_source_batch_physical_rows(
            extra,
            labels,
            expected_rows=128,
        )


def test_clic_parser_exposes_only_zero_or_three_batch_technical_smoke() -> None:
    parser = train_ssdg.build_arg_parser()
    args = parser.parse_args(
        [
            "--output_dir",
            "unused",
            "--phase1_clic_technical_smoke_batches",
            "3",
        ]
    )
    assert args.phase1_clic_technical_smoke_batches == 3
    source = inspect.getsource(train_ssdg.train)
    assert "phase1_clic_technical_smoke_receipt.json" in source
    assert source.index("release_clic_retained_graph_roots") < source.index(
        "phase1_clic_technical_smoke_receipt.json"
    )


@pytest.mark.parametrize("batches", (-1, 1, 2, 4))
def test_clic_config_rejects_any_partial_technical_smoke_batch_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    batches: int,
) -> None:
    args = _clic_validation_args(tmp_path)
    args.phase1_clic_technical_smoke_batches = batches

    monkeypatch.setattr(
        train_ssdg,
        "_build_ssdg_wisig_data",
        lambda *_args, **_kwargs: pytest.fail("invalid smoke count reached data"),
    )
    with pytest.raises(CLICConfigError, match="zero or three"):
        train_ssdg.train(args)


@pytest.mark.parametrize(
    "operator_mode",
    ("raw_phase_control", "complex_local_invariant_curvature"),
)
def test_clic_forward_exposes_the_exact_live_token_tensor_as_training_aux(operator_mode: str) -> None:
    model = build_lite_clic_backbone(operator_mode=operator_mode)
    model.train()
    clic = model.clic
    calls: list[CLICForwardResult] = []
    original_forward = clic.forward

    def wrapped_forward(received_i: torch.Tensor, z_base: torch.Tensor, **kwargs):
        result = original_forward(received_i, z_base, **kwargs)
        calls.append(result)
        return result

    clic.forward = wrapped_forward
    try:
        output = model(_nonzero_iq(batch=2).requires_grad_(True), return_aux=True)
    finally:
        clic.forward = original_forward

    assert len(calls) == 1, "one model forward must execute one CLIC forward"
    live_tokens = calls[0].token_batch.tokens
    assert live_tokens.requires_grad
    aliases = _tensor_paths(output, live_tokens)
    assert aliases, "training-only aux must expose result.token_batch.tokens"
    assert any("token" in path.lower() for path in aliases)
    loss = output["logits"].square().mean()
    token_grad = torch.autograd.grad(loss, live_tokens, retain_graph=True, allow_unused=True)[0]
    assert token_grad is not None
    assert torch.isfinite(token_grad).all()
    assert torch.count_nonzero(token_grad) > 0

    disabled = build_lite_clic_backbone(frozen_mode=False)
    disabled.eval()
    with torch.no_grad():
        disabled_output = disabled(_nonzero_iq(batch=2), return_aux=True)
    assert not _tensor_paths(disabled_output, live_tokens)


def test_clic_launcher_has_the_frozen_12_arm_matrix_and_no_target_or_unknown_training_args() -> None:
    launcher_text = CLIC_LAUNCHER.read_text(encoding="utf-8")
    assert launcher_text.startswith("#!/usr/bin/env bash\n")
    assert 'RUN_ID="${RUN_ID:-phase1_clic12_20260812_v3}"' in launcher_text
    assert "--phase1_clic_frozen_mode true" in launcher_text
    assert "--epochs 40" in launcher_text
    assert "--batch_size 128" in launcher_text
    assert "--lambda_sat_cls 0" in launcher_text
    assert "--lambda_sat_cons 0.10" in launcher_text
    assert "--sat_cons_start_epoch 1" in launcher_text
    assert not re.search(r"--(?:lambda_clic|clic_loss|loss_clic)(?:\s|=)", launcher_text, re.I)
    assert 'C) operator="raw_phase_control"' in launcher_text
    assert 'G) operator="complex_local_invariant_curvature"' in launcher_text
    calls = re.findall(r"^launch_arm (\d) ([CG]) (\d)$", launcher_text, flags=re.MULTILINE)
    assert len(calls) == 12
    assert sum(arm == "C" for _, arm, _ in calls) == 6
    assert sum(arm == "G" for _, arm, _ in calls) == 6
    assert [gpu for _, _, gpu in calls] == ["0", "0", "1", "1", "2", "2", "3", "3", "4", "5", "6", "7"]
    for flag in (
        "ccpc_leo",
        "pamr",
        "cb_sfce",
        "gd_proto_nll",
        "icmt",
        "cagm",
        "rcrmd",
        "rcat",
        "rcmmc",
        "hscf",
        "hnccd",
        "recte",
        "cp_sfce",
    ):
        assert f"--phase1_{flag}_enabled false" in launcher_text
        assert f"--lambda_{flag} 0" in launcher_text
    forbidden_training_flags = re.findall(
        r"--[^\s=]*(?:target|proxy|unknown)[^\s=]*", launcher_text, re.I
    )
    assert forbidden_training_flags == ["--phase1_source_proxy_unknown_tx_ids"]

    relative = f"scripts/{CLIC_LAUNCHER.name}"
    syntax = subprocess.run(["bash", "-n", relative], cwd=str(CODE_ROOT), text=True, capture_output=True)
    assert syntax.returncode == 0, syntax.stderr
    dry = subprocess.run(["bash", relative, "--dry-run"], cwd=str(CODE_ROOT), text=True, capture_output=True)
    assert dry.returncode == 0, dry.stderr
    rows = [line for line in dry.stdout.splitlines() if "[DRY-RUN]" in line]
    assert len(rows) == 12
    assert sum("--phase1_clic_operator_mode raw_phase_control" in line for line in rows) == 6
    assert sum("--phase1_clic_operator_mode complex_local_invariant_curvature" in line for line in rows) == 6
    assert all("phase1_clic12_20260812_v3" in line for line in rows)
    assert all("--epochs 40" in line and "--batch_size 128" in line for line in rows)
    assert all("--seed 7281164" in line for line in rows)
    assert all("--lambda_sat_cls 0" in line for line in rows)
    assert all("--lambda_sat_cons 0.10" in line for line in rows)
    assert all("--sat_cons_start_epoch 1" in line for line in rows)
    assert not re.search(r"--(?:lambda_clic|clic_loss|loss_clic)(?:\s|=)", "\n".join(rows), re.I)
    assert all(
        re.findall(r"--[^\s=]*(?:target|proxy|unknown)[^\s=]*", row, re.I)
        == ["--phase1_source_proxy_unknown_tx_ids"]
        for row in rows
    )

    # A textual dry-run is insufficient: an argparse action such as
    # ``store_true`` rejects a trailing literal ``true`` only at process start.
    # Parse every emitted child command through the real training parser so the
    # launcher cannot pass local checks and then fail all remote arms before
    # entering training.
    parser = train_ssdg.build_arg_parser()
    frozen_tx_universe = {"14-10", "14-7", "20-15", "20-19", "6-15", "8-20"}
    for row in rows:
        tokens = shlex.split(row)
        script_index = next(
            index for index, token in enumerate(tokens) if token.endswith("SSDG/train_ssdg.py")
        )
        parsed = parser.parse_args(tokens[script_index + 1 :])
        assert parsed.phase1_clic_frozen_mode is True
        train_tx = {value for value in parsed.phase1_source_train_tx_ids.split(",") if value}
        validation_tx = {
            value for value in parsed.phase1_source_known_validation_tx_ids.split(",") if value
        }
        proxy_tx = {
            value for value in parsed.phase1_source_proxy_unknown_tx_ids.split(",") if value
        }
        assert len(train_tx) == 4
        assert len(validation_tx) == len(proxy_tx) == 1
        assert not train_tx & validation_tx
        assert not train_tx & proxy_tx
        assert not validation_tx & proxy_tx
        assert train_tx | validation_tx | proxy_tx == frozen_tx_universe


def _ast_call_name(node: ast.Call) -> str:
    function = node.func
    if isinstance(function, ast.Name):
        return function.id
    if isinstance(function, ast.Attribute):
        return function.attr
    return ""


def test_train_clic_static_path_orders_config_warm_start_and_task4_lifecycle() -> None:
    """RED contract for the source-only trainer's single CLIC lifecycle."""

    source = inspect.getsource(train_ssdg.train)
    tree = ast.parse(textwrap.dedent(source))

    def calls(name: str) -> list[ast.Call]:
        return [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and _ast_call_name(node) == name
        ]

    required = (
        "new_clic_receipt",
        "strict_clic_warm_start",
        "update_clic_common_binding_receipt",
        "clic_raw_unscaled_vjp_audit",
        "clic_scaled_backward_and_classify",
        "update_clic_amp_receipt",
        "update_clic_resource_receipt",
        "release_clic_retained_graph_roots",
        "validate_clic_terminal_receipt",
        "write_clic_failure_receipt",
    )
    line: dict[str, int] = {}
    for name in required:
        found = calls(name)
        assert found, f"CLIC train path does not call {name}"
        line[name] = min(node.lineno for node in found)

    assert len(calls("clic_raw_unscaled_vjp_audit")) == 1
    assert len(calls("clic_scaled_backward_and_classify")) == 1
    assert len(calls("validate_clic_terminal_receipt")) == 1
    assert (
        line["update_clic_common_binding_receipt"]
        < line["clic_raw_unscaled_vjp_audit"]
        < line["clic_scaled_backward_and_classify"]
        < line["update_clic_amp_receipt"]
        < line["update_clic_resource_receipt"]
        < line["release_clic_retained_graph_roots"]
        < line["validate_clic_terminal_receipt"]
    )

    config_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and _ast_call_name(node)
        in {"CLICConfig", "validate_clic_config", "validate_clic_args", "validate_clic_configuration"}
    ]
    assert config_calls, "CLIC config validation/construction is absent from train()"
    config_line = min(node.lineno for node in config_calls)
    data_line = min(node.lineno for node in calls("_build_ssdg_wisig_data"))
    model_line = min(node.lineno for node in calls("build_baseline_model"))
    assert config_line < data_line
    assert config_line < model_line

    adamw_lines = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _ast_call_name(node) == "AdamW"
    ]
    assert adamw_lines
    assert line["strict_clic_warm_start"] < min(adamw_lines)

    assert "phase1_clic_config_receipt.json" in source
    assert "phase1_clic_terminal_receipt.json" in source
    assert "raw_phase_control" in source
    assert "complex_local_invariant_curvature" in source

    clic_source = source[source.find("new_clic_receipt") :]
    assert clic_source
    for marker in (
        "source_l_only",
        "use_u",
        "use_v",
        "use_proxy",
        "use_held",
        "use_target",
        "registered",
        "unknown",
        "state_feedback_count",
        "second_forward_count",
    ):
        assert marker in clic_source, f"CLIC zero-access marker missing: {marker}"
    assert not re.search(r"(?:clic_.*loss|loss_.*clic)", clic_source, re.I)

    batch_loops = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.For, ast.AsyncFor))
        and node.lineno <= line["clic_raw_unscaled_vjp_audit"]
        and int(getattr(node, "end_lineno", node.lineno)) >= line["release_clic_retained_graph_roots"]
    ]
    assert batch_loops, "CLIC retained-graph release is not inside a batch loop"
    batch_loop = min(batch_loops, key=lambda node: int(getattr(node, "end_lineno", node.lineno)) - node.lineno)
    assert line["validate_clic_terminal_receipt"] > int(getattr(batch_loop, "end_lineno", batch_loop.lineno))


def test_train_clic_graph_root_map_is_deleted_and_released_after_resource_telemetry() -> None:
    """RED graph-lifetime contract: no CLIC root survives its release."""

    source = inspect.getsource(train_ssdg.train)
    tree = ast.parse(textwrap.dedent(source))

    def calls(name: str) -> list[ast.Call]:
        return [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and _ast_call_name(node) == name
        ]

    assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "clic_retained_graph_roots"
            for target in node.targets
        )
    ]
    assert assignments, "CLIC retained graph root mapping is absent"
    root_assignment = assignments[0]
    assert isinstance(root_assignment.value, ast.Dict)
    release_calls = calls("release_clic_retained_graph_roots")
    assert len(release_calls) == 1
    release_call = release_calls[0]
    assert root_assignment.lineno < release_call.lineno

    mapped_roots = {
        key.value
        for key in root_assignment.value.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }
    assert mapped_roots
    deleted_roots = {
        name.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Delete) and root_assignment.lineno < node.lineno < release_call.lineno
        for target in node.targets
        for name in ast.walk(target)
        if isinstance(name, ast.Name)
    }
    assert mapped_roots == deleted_roots

    enclosing_loops = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.For, ast.AsyncFor))
        and node.lineno <= root_assignment.lineno
        and int(getattr(node, "end_lineno", node.lineno)) >= release_call.lineno
    ]
    assert enclosing_loops
    batch_loop = min(
        enclosing_loops,
        key=lambda node: int(getattr(node, "end_lineno", node.lineno)) - node.lineno,
    )
    post_release_loads = {
        node.id
        for node in ast.walk(batch_loop)
        if isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and release_call.lineno < node.lineno <= int(getattr(batch_loop, "end_lineno", batch_loop.lineno))
    }
    assert not mapped_roots.intersection(post_release_loads)
    resource_calls = calls("update_clic_resource_receipt")
    assert resource_calls and max(node.lineno for node in resource_calls) < release_call.lineno

    release_source = inspect.getsource(release_clic_retained_graph_roots)
    for forbidden in ("gc.collect", "empty_cache", ".backward(", ".unscale_(", "forward("):
        assert forbidden not in release_source


def _ast_target_name(node: ast.AST) -> str:
    """Return a simple assignment target name without source-text slicing."""

    return node.id if isinstance(node, ast.Name) else ""


def _ast_subscript_field(node: ast.AST, base: str) -> str:
    """Return a literal mapping field for ``base[field]`` assignments."""

    if not isinstance(node, ast.Subscript) or not isinstance(node.value, ast.Name):
        return ""
    if node.value.id != base:
        return ""
    key = node.slice
    return key.value if isinstance(key, ast.Constant) and isinstance(key.value, str) else ""


def _ast_dict_fields(node: ast.AST) -> dict[str, ast.AST]:
    """Collect literal keys from a mapping expression for local AST checks."""

    if not isinstance(node, ast.Dict):
        return {}
    fields: dict[str, ast.AST] = {}
    for key, value in zip(node.keys, node.values):
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            fields[key.value] = value
    return fields


def _ast_mapping_field_writes(tree: ast.AST, base: str) -> list[tuple[int, str, ast.AST]]:
    """Find direct mapping writes while leaving unrelated mechanism receipts alone."""

    writes: list[tuple[int, str, ast.AST]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            value = node.value
            for target in targets:
                field = _ast_subscript_field(target, base)
                if field:
                    writes.append((node.lineno, field, value))
                elif _ast_target_name(target) == base:
                    writes.extend((node.lineno, key, child) for key, child in _ast_dict_fields(value).items())
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr != "update" or not isinstance(node.func.value, ast.Name) or node.func.value.id != base:
                continue
            if node.args:
                writes.extend((node.lineno, key, child) for key, child in _ast_dict_fields(node.args[0]).items())
    return writes


def _ast_is_name(node: ast.AST, name: str) -> bool:
    return isinstance(node, ast.Name) and node.id == name


def _ast_is_literal(node: ast.AST, value: object) -> bool:
    return isinstance(node, ast.Constant) and node.value == value


def _ast_call_with_selected_checkpoint_sha(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call) or _ast_call_name(node) != "_sha256_file":
        return False
    return bool(node.args) and _ast_is_name(node.args[0], "selected_checkpoint")


def _ast_clic_terminal_scope(tree: ast.AST) -> ast.If:
    """Select the explicit frozen-mode branch, excluding legacy saves."""

    scopes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and any(
            isinstance(name, ast.Name) and name.id == "phase1_clic_frozen_mode"
            for name in ast.walk(node.test)
        )
    ]
    assert len(scopes) == 1, "CLIC terminal scope must be one explicit frozen-mode branch"
    return scopes[0]


def test_train_clic_checkpoint_receipt_binds_external_sha_without_checkpoint_rewrite() -> None:
    """CLIC terminal closure must be one-way: save, hash, bind, validate, then write."""

    source = inspect.getsource(train_ssdg.train)
    tree = ast.parse(textwrap.dedent(source))
    clic_scope = _ast_clic_terminal_scope(tree)

    save_calls = [
        node
        for node in ast.walk(clic_scope)
        if isinstance(node, ast.Call)
        and _ast_call_name(node) == "save_payload"
        and len(node.args) >= 2
        and _ast_is_name(node.args[0], "selected_checkpoint")
        and _ast_is_name(node.args[1], "final_payload")
    ]
    assert len(save_calls) == 1, "CLIC selected checkpoint must be saved exactly once"
    save_line = save_calls[0].lineno

    precheckpoint_fields = _ast_mapping_field_writes(clic_scope, "clic_receipt_precheckpoint")
    assert precheckpoint_fields, "CLIC precheckpoint receipt snapshot is absent"
    completed_before_save = [
        value
        for line, field, value in precheckpoint_fields
        if field == "completed" and line < save_line
    ]
    contract_before_save = [
        value
        for line, field, value in precheckpoint_fields
        if field == "terminal_contract" and line < save_line
    ]
    assert any(_ast_is_literal(value, False) for value in completed_before_save)
    assert any(
        _ast_is_literal(value, "AWAITING_EXTERNAL_CHECKPOINT_SHA")
        for value in contract_before_save
    )
    # A precheckpoint snapshot may inherit the empty schema field, but it must not
    # write a fake/non-empty final SHA before the file itself has been hashed.
    for line, field, value in precheckpoint_fields:
        if line < save_line and field == "final_checkpoint_sha256":
            assert _ast_is_literal(value, ""), "precheckpoint must not claim a placeholder final SHA"

    final_payload_dicts = [
        node
        for node in ast.walk(clic_scope)
        if isinstance(node, ast.Assign)
        and any(_ast_target_name(target) == "final_payload" for target in node.targets)
        and "clic_receipt_precheckpoint" in _ast_dict_fields(node.value)
    ]
    assert len(final_payload_dicts) == 1, "CLIC final payload must carry the named precheckpoint receipt"

    sha_assignments = [
        node
        for node in ast.walk(clic_scope)
        if isinstance(node, ast.Assign)
        and any(_ast_target_name(target) == "selected_checkpoint_sha256" for target in node.targets)
        and _ast_call_with_selected_checkpoint_sha(node.value)
        and node.lineno > save_line
    ]
    assert len(sha_assignments) == 1, "CLIC must hash selected_checkpoint exactly after its sole save"
    sha_line = sha_assignments[0].lineno

    terminal_writes = _ast_mapping_field_writes(clic_scope, "clic_terminal_receipt")
    final_sha_writes = [
        (line, value)
        for line, field, value in terminal_writes
        if field == "final_checkpoint_sha256" and line > sha_line
    ]
    assert final_sha_writes, "external CLIC receipt must bind the final checkpoint SHA"
    assert any(_ast_is_name(value, "selected_checkpoint_sha256") for _, value in final_sha_writes)
    assert not any(
        field in {"selected_checkpoint_path", "selected_checkpoint_sha256"}
        for _, field, _ in terminal_writes
    ), "strict CLIC core must not be extended with envelope path/hash fields"

    completed_after_sha = [
        value
        for line, field, value in terminal_writes
        if field == "completed" and line > sha_line
    ]
    assert any(_ast_is_literal(value, True) for value in completed_after_sha)

    validation_calls = [
        node
        for node in ast.walk(clic_scope)
        if isinstance(node, ast.Call)
        and _ast_call_name(node) == "validate_clic_terminal_receipt"
        and node.args
        and _ast_is_name(node.args[0], "clic_terminal_receipt")
    ]
    assert len(validation_calls) == 1
    validation_line = validation_calls[0].lineno
    assert validation_line > max(line for line, value in [(line, value) for line, field, value in terminal_writes if field == "completed" and line > sha_line])

    envelope_writes = _ast_mapping_field_writes(clic_scope, "clic_terminal_envelope")
    assert envelope_writes, "versioned CLIC terminal envelope construction is absent"
    envelope_fields = {field: (line, value) for line, field, value in envelope_writes}
    assert _ast_is_literal(envelope_fields["schema"][1], "cvs.phase1.clic_terminal_envelope.v1")
    assert _ast_is_literal(envelope_fields["method"][1], "P1_CLIC")
    assert _ast_is_name(envelope_fields["strict_core"][1], "clic_terminal_receipt")
    path_line, path_value = envelope_fields["selected_checkpoint_path"]
    hash_line, hash_value = envelope_fields["selected_checkpoint_sha256"]
    assert path_line > validation_line and hash_line > validation_line
    assert any(_ast_is_name(child, "selected_checkpoint") for child in ast.walk(path_value))
    assert _ast_is_name(hash_value, "selected_checkpoint_sha256")

    envelope_validation_calls = [
        node
        for node in ast.walk(clic_scope)
        if isinstance(node, ast.Call)
        and _ast_call_name(node) == "validate_clic_terminal_envelope"
        and node.args
        and _ast_is_name(node.args[0], "clic_terminal_envelope")
    ]
    assert len(envelope_validation_calls) == 1
    envelope_validation_line = envelope_validation_calls[0].lineno
    assert envelope_validation_line > max(path_line, hash_line)

    external_writes = [
        node
        for node in ast.walk(clic_scope)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "write_text"
        and any(
            isinstance(const, ast.Constant)
            and const.value == "phase1_clic_terminal_receipt.json"
            for const in ast.walk(node)
        )
    ]
    assert len(external_writes) == 1
    assert external_writes[0].lineno > envelope_validation_line
    assert any(
        _ast_is_name(child, "clic_terminal_envelope")
        for child in ast.walk(external_writes[0])
    )

    # Restrict the no-overwrite assertion to the CLIC-named payload call above;
    # legacy Phase1 mechanism checkpoints use different payload/path variables.
    later_selected_checkpoint_saves = [
        node
        for node in ast.walk(clic_scope)
        if isinstance(node, ast.Call)
        and _ast_call_name(node) == "save_payload"
        and node.lineno > sha_line
        and node.args
        and _ast_is_name(node.args[0], "selected_checkpoint")
    ]
    assert not later_selected_checkpoint_saves, "selected_checkpoint must not be overwritten after SHA binding"


def _valid_clic_terminal_envelope() -> dict[str, object]:
    strict_core = validate_clic_terminal_receipt(_complete_receipt("G"), arm="G")
    return {
        "schema": "cvs.phase1.clic_terminal_envelope.v1",
        "method": "P1_CLIC",
        "strict_core": strict_core,
        "selected_checkpoint_path": "runs/phase1_clic/F6G/final_ssdg.pth",
        "selected_checkpoint_sha256": strict_core["final_checkpoint_sha256"],
    }


def _validate_clic_terminal_envelope(envelope: dict[str, object]) -> dict[str, object]:
    validator = getattr(phase1_clic_module, "validate_clic_terminal_envelope", None)
    assert callable(validator), "CLIC terminal envelope validator API is absent"
    return validator(envelope)


@pytest.mark.parametrize("field", ("selected_checkpoint_path", "selected_checkpoint_sha256"))
def test_strict_clic_core_rejects_external_envelope_fields(field: str) -> None:
    strict_core = _complete_receipt("G")
    strict_core[field] = "runs/final_ssdg.pth" if field.endswith("path") else "f" * 64
    with pytest.raises(CLICTerminalError):
        validate_clic_terminal_receipt(strict_core, arm="G")


def test_valid_clic_terminal_envelope_revalidates_strict_core() -> None:
    envelope = _valid_clic_terminal_envelope()
    validated = _validate_clic_terminal_envelope(envelope)
    assert validated["schema"] == "cvs.phase1.clic_terminal_envelope.v1"
    assert validated["method"] == "P1_CLIC"
    strict_core = validated["strict_core"]
    assert strict_core["final_checkpoint_sha256"] == validated["selected_checkpoint_sha256"]
    revalidated_core = validate_clic_terminal_receipt(strict_core, arm=str(strict_core["arm"]))
    assert revalidated_core["final_checkpoint_sha256"] == validated["selected_checkpoint_sha256"]


@pytest.mark.parametrize(
    "mutation",
    (
        "missing_schema",
        "missing_method",
        "missing_strict_core",
        "missing_path",
        "missing_hash",
        "wrong_schema",
        "wrong_method",
        "empty_path",
        "invalid_hash",
        "hash_mismatch",
        "strict_core_drift",
        "extra_key",
    ),
)
def test_clic_terminal_envelope_rejects_binding_and_schema_drift(mutation: str) -> None:
    envelope = _valid_clic_terminal_envelope()
    if mutation == "missing_schema":
        envelope.pop("schema")
    elif mutation == "missing_method":
        envelope.pop("method")
    elif mutation == "missing_strict_core":
        envelope.pop("strict_core")
    elif mutation == "missing_path":
        envelope.pop("selected_checkpoint_path")
    elif mutation == "missing_hash":
        envelope.pop("selected_checkpoint_sha256")
    elif mutation == "wrong_schema":
        envelope["schema"] = "cvs.phase1.clic_terminal_envelope.v0"
    elif mutation == "wrong_method":
        envelope["method"] = "OTHER"
    elif mutation == "empty_path":
        envelope["selected_checkpoint_path"] = ""
    elif mutation == "invalid_hash":
        envelope["selected_checkpoint_sha256"] = "g" * 64
    elif mutation == "hash_mismatch":
        envelope["selected_checkpoint_sha256"] = "a" * 64
    elif mutation == "strict_core_drift":
        envelope["strict_core"]["final_checkpoint_sha256"] = "a" * 64
    elif mutation == "extra_key":
        envelope["unexpected"] = False
    with pytest.raises(CLICTerminalError):
        _validate_clic_terminal_envelope(envelope)
