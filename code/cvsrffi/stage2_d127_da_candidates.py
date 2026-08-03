"""Frozen pure-tensor cores for the three D127 DA candidates.

This module deliberately has no hook platform, runner, scorer, serialization,
or D92-Lite implementation.  It only realizes the frozen support-conditioned
rank-two states and exposes query paths with no label or update arguments.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor


RANK = 2
SUMMARY_DIM = 5
JOINT_PROJ_INPUT_DIM = 320
RELATIVE_RESIDUAL_BUDGET = 0.05
LAYER_NORM_EPS = 1.0e-5

CANDIDATE_A = "DA-A-FSRG-time_fuse"
CANDIDATE_B = "DA-B-FSRG-t2norm"
CANDIDATE_C = "DA-C-RDHA-joint_proj"

TAP_A = "id_backbone.time_fuse.1:pre_relu"
TAP_B = "id_backbone.t2.norm:pre_relu"
TAP_C = "id_backbone.cls_head.joint_proj.0:input"
_FSRG_TAPS = {CANDIDATE_A: TAP_A, CANDIDATE_B: TAP_B}


class D127DACandidateError(ValueError):
    """Raised when a frozen D127 contract is violated."""


def _matrix(value: Tensor, *, name: str, width: int | None = None) -> Tensor:
    if (
        not torch.is_tensor(value)
        or not value.dtype.is_floating_point
        or value.ndim != 2
        or int(value.shape[0]) < 1
        or (width is not None and int(value.shape[1]) != width)
        or not bool(torch.isfinite(value).all().item())
    ):
        expected = "[N,D]" if width is None else f"[N,{width}]"
        raise D127DACandidateError(f"{name} must be finite floating {expected}")
    return value


def _taps(value: Tensor, *, name: str, width: int) -> Tensor:
    """Validate a vector tap [N,d] or a channel-time tap [N,d,T]."""

    if (
        not torch.is_tensor(value)
        or not value.dtype.is_floating_point
        or value.ndim not in (2, 3)
        or int(value.shape[0]) < 1
        or int(value.shape[1]) != width
        or (value.ndim == 3 and int(value.shape[2]) < 1)
        or not bool(torch.isfinite(value).all().item())
    ):
        raise D127DACandidateError(
            f"{name} must be finite floating [N,{width}] or [N,{width},T]"
        )
    return value


def _vector(value: Tensor, *, name: str, width: int) -> Tensor:
    if (
        not torch.is_tensor(value)
        or not value.dtype.is_floating_point
        or value.ndim != 1
        or int(value.shape[0]) != width
        or not bool(torch.isfinite(value).all().item())
    ):
        raise D127DACandidateError(f"{name} must be finite floating [{width}]")
    return value


def _copy(value: Tensor, *, keep_grad: bool) -> Tensor:
    copied = value.detach().clone().contiguous()
    if keep_grad:
        copied.requires_grad_(bool(value.requires_grad))
    return copied


def _same_context(left: Tensor, right: Tensor, *, name: str) -> None:
    if left.device != right.device or left.dtype != right.dtype:
        raise D127DACandidateError(f"{name} must share tensor dtype and device")


def _positive(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise D127DACandidateError(f"{name} must be finite and positive")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise D127DACandidateError(f"{name} must be finite and positive")
    return result


def _labels(value: Tensor, *, rows: int, device: torch.device) -> Tensor:
    if (
        not torch.is_tensor(value)
        or value.ndim != 1
        or int(value.shape[0]) != rows
        or value.dtype.is_floating_point
        or value.dtype == torch.bool
        or value.device != device
    ):
        raise D127DACandidateError(
            "support_labels must be integral [N] on the support tensor device"
        )
    return value


def _groups(labels: Tensor, *, require_equal_k: bool = True) -> tuple[Tensor, ...]:
    """Class groups in first-row order, independent of class-ID values."""

    unique, inverse, counts = torch.unique(
        labels, sorted=True, return_inverse=True, return_counts=True
    )
    if int(unique.numel()) < 2:
        raise D127DACandidateError("at least two registered classes are required")
    if require_equal_k and not bool(torch.all(counts == counts[0]).item()):
        raise D127DACandidateError("all registered classes must have the same K-shot count")
    ordered: list[tuple[int, Tensor]] = []
    for index in range(int(unique.numel())):
        members = torch.nonzero(inverse == index, as_tuple=False).reshape(-1)
        ordered.append((int(members[0].item()), members))
    ordered.sort(key=lambda item: item[0])
    return tuple(members for _first, members in ordered)


@dataclass(frozen=True, slots=True)
class D127ResourceReceipt:
    """Compact, inspectable resource and access counters for one DA state."""

    candidate_id: str
    tap_name: str
    state_source: str
    dimension: int
    rank: int
    support_rows: int
    registered_class_count: int
    support_gradient_calls: int
    phase2_backward_calls: int
    phase2_optimizer_steps: int
    query_rows_used_for_fit: int
    query_state_updates: int
    query_selection_count: int
    query_gradient_calls: int
    truth_role_quota_inputs: int
    global_reassignment_calls: int
    source_rows_used_at_phase2: int
    clean_rows_used_at_phase2: int
    adapter_macs_per_sample: int
    relative_budget: float

    def __post_init__(self) -> None:
        values = (
            self.dimension,
            self.rank,
            self.support_rows,
            self.registered_class_count,
            self.support_gradient_calls,
            self.phase2_backward_calls,
            self.phase2_optimizer_steps,
            self.query_rows_used_for_fit,
            self.query_state_updates,
            self.query_selection_count,
            self.query_gradient_calls,
            self.truth_role_quota_inputs,
            self.global_reassignment_calls,
            self.source_rows_used_at_phase2,
            self.clean_rows_used_at_phase2,
            self.adapter_macs_per_sample,
        )
        if (
            self.candidate_id not in {CANDIDATE_A, CANDIDATE_B, CANDIDATE_C}
            or not self.tap_name
            or not self.state_source
            or self.rank != RANK
            or any(type(value) is not int or value < 0 for value in values)
            or not math.isfinite(float(self.relative_budget))
            or float(self.relative_budget) <= 0.0
        ):
            raise D127DACandidateError("D127 resource receipt drift")

    @property
    def protocol_closed(self) -> bool:
        return (
            self.query_rows_used_for_fit == 0
            and self.query_state_updates == 0
            and self.query_selection_count == 0
            and self.query_gradient_calls == 0
            and self.truth_role_quota_inputs == 0
            and self.global_reassignment_calls == 0
            and self.source_rows_used_at_phase2 == 0
            and self.clean_rows_used_at_phase2 == 0
        )

    def as_dict(self) -> dict[str, int | float | str | bool]:
        return {
            "candidate_id": self.candidate_id,
            "tap_name": self.tap_name,
            "state_source": self.state_source,
            "dimension": self.dimension,
            "rank": self.rank,
            "support_rows": self.support_rows,
            "registered_class_count": self.registered_class_count,
            "support_gradient_calls": self.support_gradient_calls,
            "phase2_backward_calls": self.phase2_backward_calls,
            "phase2_optimizer_steps": self.phase2_optimizer_steps,
            "query_rows_used_for_fit": self.query_rows_used_for_fit,
            "query_state_updates": self.query_state_updates,
            "query_selection_count": self.query_selection_count,
            "query_gradient_calls": self.query_gradient_calls,
            "truth_role_quota_inputs": self.truth_role_quota_inputs,
            "global_reassignment_calls": self.global_reassignment_calls,
            "source_rows_used_at_phase2": self.source_rows_used_at_phase2,
            "clean_rows_used_at_phase2": self.clean_rows_used_at_phase2,
            "adapter_macs_per_sample": self.adapter_macs_per_sample,
            "relative_budget": self.relative_budget,
            "protocol_closed": self.protocol_closed,
        }


@dataclass(frozen=True, slots=True)
class FSRGAsset:
    """Phase1-sealed rank-two parameters for exactly frozen candidate A or B."""

    candidate_id: str
    tap_name: str
    U: Tensor
    V: Tensor
    d_f_diag: Tensor
    rho: float

    def __post_init__(self) -> None:
        if _FSRG_TAPS.get(self.candidate_id) != self.tap_name:
            raise D127DACandidateError("FSRG candidate/tap binding is not frozen")
        U = _matrix(self.U, name="U")
        V = _matrix(self.V, name="V")
        d_f_diag = _vector(self.d_f_diag, name="d_f_diag", width=RANK)
        if (
            int(U.shape[1]) != RANK
            or int(V.shape[0]) != RANK
            or int(U.shape[0]) != int(V.shape[1])
        ):
            raise D127DACandidateError("FSRG U/V must be [d,2] and [2,d]")
        _same_context(U, V, name="FSRG U/V")
        _same_context(U, d_f_diag, name="FSRG d_f_diag")
        if not bool(torch.all(d_f_diag > 0.0).item()):
            raise D127DACandidateError("FSRG d_f_diag must be strictly positive")
        object.__setattr__(self, "U", _copy(U, keep_grad=True))
        object.__setattr__(self, "V", _copy(V, keep_grad=True))
        object.__setattr__(self, "d_f_diag", _copy(d_f_diag, keep_grad=False))
        object.__setattr__(self, "rho", _positive(self.rho, name="rho"))

    @property
    def dimension(self) -> int:
        return int(self.U.shape[0])

    @property
    def a_max(self) -> float:
        return self.rho / math.sqrt(2.0)


@dataclass(frozen=True, slots=True)
class FSRGState:
    """Support-only, detached two-coefficient state for candidate A or B."""

    candidate_id: str
    tap_name: str
    a: Tensor
    support_gradient: Tensor
    rho: float
    a_max: float
    receipt: D127ResourceReceipt

    def __post_init__(self) -> None:
        if (
            _FSRG_TAPS.get(self.candidate_id) != self.tap_name
            or self.receipt.candidate_id != self.candidate_id
            or self.receipt.tap_name != self.tap_name
            or self.receipt.state_source != "support_gradient"
            or not self.receipt.protocol_closed
        ):
            raise D127DACandidateError("FSRG state receipt/binding drift")
        rho = _positive(self.rho, name="rho")
        a_max = _positive(self.a_max, name="a_max")
        if not math.isclose(a_max, rho / math.sqrt(2.0), abs_tol=1.0e-12):
            raise D127DACandidateError("FSRG a_max must be rho/sqrt(2)")
        a = _vector(self.a, name="a", width=RANK)
        gradient = _vector(
            self.support_gradient, name="support_gradient", width=RANK
        )
        tol = max(1.0e-12, float(torch.finfo(a.dtype).eps) * 32.0)
        if (
            float(torch.linalg.vector_norm(a).item()) > rho + tol
            or bool(torch.any(torch.abs(a) > a_max + tol).item())
            or not bool(torch.any(torch.abs(gradient) > 0.0).item())
        ):
            raise D127DACandidateError("FSRG projected state drift")
        object.__setattr__(self, "a", _copy(a, keep_grad=False))
        object.__setattr__(
            self, "support_gradient", _copy(gradient, keep_grad=False)
        )
        object.__setattr__(self, "rho", rho)
        object.__setattr__(self, "a_max", a_max)


@dataclass(frozen=True, slots=True)
class RDHAAsset:
    """Phase1-sealed Q/b/mean/std asset for C's fixed 320D hidden input."""

    U: Tensor
    V: Tensor
    Q: Tensor
    b: Tensor
    mean_p1: Tensor
    std_p1: Tensor
    a_max: float
    candidate_id: str = CANDIDATE_C
    tap_name: str = TAP_C

    def __post_init__(self) -> None:
        if self.candidate_id != CANDIDATE_C or self.tap_name != TAP_C:
            raise D127DACandidateError("RDHA candidate/tap binding is not frozen")
        U = _matrix(self.U, name="U", width=RANK)
        V = _matrix(self.V, name="V", width=JOINT_PROJ_INPUT_DIM)
        Q = _matrix(self.Q, name="Q", width=SUMMARY_DIM)
        b = _vector(self.b, name="b", width=RANK)
        mean_p1 = _vector(self.mean_p1, name="mean_p1", width=SUMMARY_DIM)
        std_p1 = _vector(self.std_p1, name="std_p1", width=SUMMARY_DIM)
        if int(U.shape[0]) != JOINT_PROJ_INPUT_DIM or int(Q.shape[0]) != RANK:
            raise D127DACandidateError("RDHA U/V/Q shapes must be [320,2], [2,320], [2,5]")
        for name, value in (
            ("V", V),
            ("Q", Q),
            ("b", b),
            ("mean_p1", mean_p1),
            ("std_p1", std_p1),
        ):
            _same_context(U, value, name=f"RDHA U/{name}")
        if not bool(torch.all(std_p1 > 0.0).item()):
            raise D127DACandidateError("RDHA std_p1 must be strictly positive")
        object.__setattr__(self, "U", _copy(U, keep_grad=True))
        object.__setattr__(self, "V", _copy(V, keep_grad=True))
        object.__setattr__(self, "Q", _copy(Q, keep_grad=True))
        object.__setattr__(self, "b", _copy(b, keep_grad=True))
        object.__setattr__(self, "mean_p1", _copy(mean_p1, keep_grad=False))
        object.__setattr__(self, "std_p1", _copy(std_p1, keep_grad=False))
        object.__setattr__(self, "a_max", _positive(self.a_max, name="a_max"))


@dataclass(frozen=True, slots=True)
class RDHAState:
    """Class-symmetric support state for C; Phase2 has no backward path."""

    a: Tensor
    summary: Tensor
    standardized_summary: Tensor
    receipt: D127ResourceReceipt
    candidate_id: str = CANDIDATE_C
    tap_name: str = TAP_C

    def __post_init__(self) -> None:
        if (
            self.candidate_id != CANDIDATE_C
            or self.tap_name != TAP_C
            or self.receipt.candidate_id != CANDIDATE_C
            or self.receipt.tap_name != TAP_C
            or self.receipt.state_source != "class_symmetric_summary"
            or self.receipt.phase2_backward_calls != 0
            or self.receipt.phase2_optimizer_steps != 0
            or not self.receipt.protocol_closed
        ):
            raise D127DACandidateError("RDHA state receipt/binding drift")
        a = _vector(self.a, name="a", width=RANK)
        summary = _vector(self.summary, name="summary", width=SUMMARY_DIM)
        standardized = _vector(
            self.standardized_summary, name="standardized_summary", width=SUMMARY_DIM
        )
        object.__setattr__(self, "a", _copy(a, keep_grad=False))
        object.__setattr__(self, "summary", _copy(summary, keep_grad=False))
        object.__setattr__(
            self, "standardized_summary", _copy(standardized, keep_grad=False)
        )


@dataclass(frozen=True, slots=True)
class RDHAOuterResult:
    """Differentiable Phase1-only C episode outputs.

    This is intentionally not a Phase2 state or query API.  The same
    support-derived coefficient vector is applied to both physically separate
    Phase1 support and outer-query hidden tensors so an episodic qKNN loss can
    update U/V/Q/b directly.
    """

    adapted_support: Tensor
    adapted_query: Tensor
    a: Tensor
    summary: Tensor
    standardized_summary: Tensor


def class_balanced_support_loss(
    per_sample_loss: Tensor, support_labels: Tensor
) -> Tensor:
    """Equal-class support loss required by the frozen A/B gradient rule."""

    if (
        not torch.is_tensor(per_sample_loss)
        or not per_sample_loss.dtype.is_floating_point
        or per_sample_loss.ndim != 1
        or int(per_sample_loss.numel()) < 1
        or not bool(torch.isfinite(per_sample_loss.detach()).all().item())
    ):
        raise D127DACandidateError("per_sample_loss must be finite floating [N]")
    labels = _labels(
        support_labels,
        rows=int(per_sample_loss.numel()),
        device=per_sample_loss.device,
    )
    return torch.stack(
        [per_sample_loss.index_select(0, group).mean() for group in _groups(labels)]
    ).mean()


def project_rank2_coefficients(raw: Tensor, rho: float) -> Tensor:
    """Apply the frozen L2 and coordinate projection to one rank-two vector."""

    raw = _vector(raw, name="raw", width=RANK)
    rho = _positive(rho, name="rho")
    a_max = rho / math.sqrt(2.0)
    projected = torch.clamp(raw, min=-a_max, max=a_max)
    norm = torch.linalg.vector_norm(projected)
    if float(norm.item()) > rho:
        projected = projected * (rho / norm)
    return projected


def derive_phase1_fsrg_rho(inner_taps: Tensor, *, dimension: int) -> float:
    """Derive the sealed A/B budget from Phase1 inner taps only.

    Time-domain taps are flattened per physical support row before the median,
    so a [N,d,T] hook cannot change the definition of the five-percent budget.
    This helper is for Phase1 asset construction; Phase2 fit never calls it.
    """

    taps = _taps(inner_taps, name="inner_taps", width=dimension)
    flattened = taps.detach().to(dtype=torch.float64).reshape(int(taps.shape[0]), -1)
    return _positive(
        RELATIVE_RESIDUAL_BUDGET
        * float(torch.quantile(torch.linalg.vector_norm(flattened, dim=1), 0.5).item()),
        name="rho",
    )


def _residual(taps: Tensor, U: Tensor, V: Tensor, a: Tensor) -> Tensor:
    if taps.ndim == 2:
        response = torch.tanh(taps @ V.transpose(0, 1))
        return taps + ((response * a) @ U.transpose(0, 1))
    response = torch.tanh(torch.einsum("rd,ndt->nrt", V, taps))
    residual = torch.einsum(
        "dr,nrt->ndt", U, response * a.reshape(1, RANK, 1)
    )
    return taps + residual


def _fsrg_adapter_macs_per_sample(taps: Tensor, *, dimension: int) -> int:
    time_positions = int(taps.shape[2]) if taps.ndim == 3 else 1
    return 4 * dimension * time_positions


def _check_fsrg_pair(asset: FSRGAsset, state: FSRGState) -> None:
    if (
        state.candidate_id != asset.candidate_id
        or state.tap_name != asset.tap_name
        or state.receipt.dimension != asset.dimension
    ):
        raise D127DACandidateError("FSRG asset/state pairing drift")


def fit_fsrg_support_state(
    support_taps: Tensor,
    support_labels: Tensor,
    asset: FSRGAsset,
    per_sample_loss: Callable[[Tensor], Tensor],
) -> FSRGState:
    """Fit A/B's one support-gradient state; only a has Phase2 gradients."""

    if not isinstance(asset, FSRGAsset) or not callable(per_sample_loss):
        raise D127DACandidateError("FSRG fit requires a sealed asset and loss callable")
    taps = _taps(support_taps, name="support_taps", width=asset.dimension)
    _same_context(taps, asset.U, name="support taps/FSRG asset")
    labels = _labels(
        support_labels, rows=int(taps.shape[0]), device=taps.device
    )
    groups = _groups(labels)

    base = taps.detach()
    a0 = torch.zeros(RANK, dtype=base.dtype, device=base.device, requires_grad=True)
    adapted = _residual(base, asset.U.detach(), asset.V.detach(), a0)
    losses = per_sample_loss(adapted)
    if (
        not torch.is_tensor(losses)
        or losses.device != base.device
        or losses.dtype != base.dtype
        or losses.ndim != 1
        or int(losses.shape[0]) != int(base.shape[0])
    ):
        raise D127DACandidateError("FSRG loss callable must return aligned [N] values")
    support_loss = class_balanced_support_loss(losses, labels)
    try:
        gradient = torch.autograd.grad(
            support_loss, a0, create_graph=False, retain_graph=False, allow_unused=False
        )[0]
    except RuntimeError as exc:
        raise D127DACandidateError(
            "FSRG loss callable is not differentiably bound to the rank-two state"
        ) from exc
    if (
        gradient is None
        or not bool(torch.isfinite(gradient).all().item())
        or not bool(torch.any(torch.abs(gradient) > 0.0).item())
    ):
        raise D127DACandidateError("FSRG support gradient is zero or non-finite")
    a = project_rank2_coefficients(
        -gradient.detach() / torch.sqrt(asset.d_f_diag.detach()), asset.rho
    ).detach()
    if not bool(torch.any(torch.abs(a) > 0.0).item()):
        raise D127DACandidateError("FSRG projected state is zero; identity fallback is forbidden")
    receipt = D127ResourceReceipt(
        candidate_id=asset.candidate_id,
        tap_name=asset.tap_name,
        state_source="support_gradient",
        dimension=asset.dimension,
        rank=RANK,
        support_rows=int(base.shape[0]),
        registered_class_count=len(groups),
        support_gradient_calls=1,
        phase2_backward_calls=1,
        phase2_optimizer_steps=0,
        query_rows_used_for_fit=0,
        query_state_updates=0,
        query_selection_count=0,
        query_gradient_calls=0,
        truth_role_quota_inputs=0,
        global_reassignment_calls=0,
        source_rows_used_at_phase2=0,
        clean_rows_used_at_phase2=0,
        adapter_macs_per_sample=_fsrg_adapter_macs_per_sample(
            base, dimension=asset.dimension
        ),
        relative_budget=RELATIVE_RESIDUAL_BUDGET,
    )
    return FSRGState(
        candidate_id=asset.candidate_id,
        tap_name=asset.tap_name,
        a=a,
        support_gradient=gradient.detach(),
        rho=asset.rho,
        a_max=asset.a_max,
        receipt=receipt,
    )


def apply_fsrg_outer(taps: Tensor, asset: FSRGAsset, state: FSRGState) -> Tensor:
    """Differentiable Phase1 outer path; it is not a Phase2 query API."""

    _check_fsrg_pair(asset, state)
    rows = _taps(taps, name="taps", width=asset.dimension)
    _same_context(rows, asset.U, name="outer taps/FSRG asset")
    return _residual(rows, asset.U, asset.V, state.a)


def _apply_fsrg_phase2(taps: Tensor, asset: FSRGAsset, state: FSRGState) -> Tensor:
    _check_fsrg_pair(asset, state)
    rows = _taps(taps, name="taps", width=asset.dimension)
    _same_context(rows, asset.U, name="phase2 taps/FSRG asset")
    with torch.no_grad():
        adapted = _residual(rows, asset.U.detach(), asset.V.detach(), state.a)
    return adapted.detach()


def adapt_fsrg_support(taps: Tensor, asset: FSRGAsset, state: FSRGState) -> Tensor:
    """Recompute support features under an already frozen A/B state."""

    return _apply_fsrg_phase2(taps, asset, state)


def adapt_fsrg_query(taps: Tensor, asset: FSRGAsset, state: FSRGState) -> Tensor:
    """Query-only A/B forward with no labels, fit, update, or gradients."""

    return _apply_fsrg_phase2(taps, asset, state)


def _norm_hidden(hidden: Tensor) -> Tensor:
    return F.layer_norm(
        hidden,
        normalized_shape=(JOINT_PROJ_INPUT_DIM,),
        weight=None,
        bias=None,
        eps=LAYER_NORM_EPS,
    )


def _rdah_support_summary(
    support_hidden: Tensor, support_labels: Tensor, V: Tensor
) -> Tensor:
    groups = _groups(support_labels)
    response = torch.tanh(_norm_hidden(support_hidden) @ V.transpose(0, 1))
    means = torch.stack(
        [response.index_select(0, group).mean(dim=0) for group in groups]
    )
    mean = means.mean(dim=0)
    centered = means - mean
    covariance = centered.transpose(0, 1) @ centered / float(len(groups))
    summary = torch.stack(
        (mean[0], mean[1], covariance[0, 0], covariance[0, 1], covariance[1, 1])
    )
    if not bool(torch.isfinite(summary.detach()).all().item()):
        raise D127DACandidateError("RDHA support summary is non-finite")
    return summary


def _rdah_adapt(hidden: Tensor, U: Tensor, V: Tensor, a: Tensor) -> Tensor:
    normalized = _norm_hidden(hidden)
    residual = (torch.tanh(normalized @ V.transpose(0, 1)) * a) @ U.transpose(0, 1)
    return hidden + residual


def build_rdah_support_summary(
    support_hidden: Tensor, support_labels: Tensor, asset: RDHAAsset
) -> Tensor:
    """C's Phase2 summary: support-only and explicitly non-differentiable."""

    if not isinstance(asset, RDHAAsset):
        raise D127DACandidateError("RDHA summary requires a sealed asset")
    hidden = _matrix(
        support_hidden, name="support_hidden", width=JOINT_PROJ_INPUT_DIM
    )
    _same_context(hidden, asset.U, name="support hidden/RDHA asset")
    labels = _labels(
        support_labels, rows=int(hidden.shape[0]), device=hidden.device
    )
    with torch.no_grad():
        summary = _rdah_support_summary(
            hidden.detach(), labels, asset.V.detach()
        )
    return summary.detach()


def apply_rdah_outer(
    support_hidden: Tensor,
    support_labels: Tensor,
    query_hidden: Tensor,
    asset: RDHAAsset,
) -> RDHAOuterResult:
    """Differentiable Phase1 episodic path for C; never call in Phase2.

    The state is derived from support only.  query_hidden has no labels or
    selection surface and receives exactly the same support-derived a.
    The episode must supply separate support and query tensors; physical-ID
    disjointness remains a builder/runner authority check.
    """

    if not isinstance(asset, RDHAAsset):
        raise D127DACandidateError("RDHA outer path requires a sealed asset")
    support = _matrix(
        support_hidden, name="support_hidden", width=JOINT_PROJ_INPUT_DIM
    )
    query = _matrix(query_hidden, name="query_hidden", width=JOINT_PROJ_INPUT_DIM)
    if support.data_ptr() == query.data_ptr():
        raise D127DACandidateError(
            "RDHA Phase1 outer support and query must be separate tensors"
        )
    _same_context(support, asset.U, name="support hidden/RDHA asset")
    _same_context(query, asset.U, name="query hidden/RDHA asset")
    labels = _labels(
        support_labels, rows=int(support.shape[0]), device=support.device
    )
    summary = _rdah_support_summary(support, labels, asset.V)
    standardized = (summary - asset.mean_p1.detach()) / asset.std_p1.detach()
    a = asset.a_max * torch.tanh(asset.Q @ standardized + asset.b)
    return RDHAOuterResult(
        adapted_support=_rdah_adapt(support, asset.U, asset.V, a),
        adapted_query=_rdah_adapt(query, asset.U, asset.V, a),
        a=a,
        summary=summary,
        standardized_summary=standardized,
    )


def fit_rdah_support_state(
    support_hidden: Tensor, support_labels: Tensor, asset: RDHAAsset
) -> RDHAState:
    """Fit C from support only, with no Phase2 backward or optimizer step."""

    summary = build_rdah_support_summary(support_hidden, support_labels, asset)
    with torch.no_grad():
        standardized = (summary - asset.mean_p1.detach()) / asset.std_p1.detach()
        a = asset.a_max * torch.tanh(
            asset.Q.detach() @ standardized + asset.b.detach()
        )
    budget = math.sqrt(2.0) * asset.a_max
    tol = max(1.0e-12, float(torch.finfo(a.dtype).eps) * 32.0)
    if (
        not bool(torch.isfinite(a).all().item())
        or float(torch.linalg.vector_norm(a).item()) > budget + tol
        or bool(torch.any(torch.abs(a) > asset.a_max + tol).item())
    ):
        raise D127DACandidateError("RDHA state exceeds the frozen residual budget")
    hidden = _matrix(
        support_hidden, name="support_hidden", width=JOINT_PROJ_INPUT_DIM
    )
    labels = _labels(
        support_labels, rows=int(hidden.shape[0]), device=hidden.device
    )
    receipt = D127ResourceReceipt(
        candidate_id=CANDIDATE_C,
        tap_name=TAP_C,
        state_source="class_symmetric_summary",
        dimension=JOINT_PROJ_INPUT_DIM,
        rank=RANK,
        support_rows=int(hidden.shape[0]),
        registered_class_count=len(_groups(labels)),
        support_gradient_calls=0,
        phase2_backward_calls=0,
        phase2_optimizer_steps=0,
        query_rows_used_for_fit=0,
        query_state_updates=0,
        query_selection_count=0,
        query_gradient_calls=0,
        truth_role_quota_inputs=0,
        global_reassignment_calls=0,
        source_rows_used_at_phase2=0,
        clean_rows_used_at_phase2=0,
        adapter_macs_per_sample=4 * JOINT_PROJ_INPUT_DIM,
        relative_budget=RELATIVE_RESIDUAL_BUDGET,
    )
    return RDHAState(
        a=a.detach(),
        summary=summary,
        standardized_summary=standardized.detach(),
        receipt=receipt,
    )


def _check_rdah_pair(asset: RDHAAsset, state: RDHAState) -> None:
    if (
        state.candidate_id != CANDIDATE_C
        or state.tap_name != TAP_C
        or state.receipt.dimension != JOINT_PROJ_INPUT_DIM
    ):
        raise D127DACandidateError("RDHA asset/state pairing drift")


def _apply_rdah_phase2(hidden: Tensor, asset: RDHAAsset, state: RDHAState) -> Tensor:
    _check_rdah_pair(asset, state)
    rows = _matrix(hidden, name="hidden", width=JOINT_PROJ_INPUT_DIM)
    _same_context(rows, asset.U, name="phase2 hidden/RDHA asset")
    with torch.no_grad():
        adapted = _rdah_adapt(
            rows, asset.U.detach(), asset.V.detach(), state.a
        )
    return adapted.detach()


def adapt_rdah_support(hidden: Tensor, asset: RDHAAsset, state: RDHAState) -> Tensor:
    """Recompute C support rows under its single frozen state."""

    return _apply_rdah_phase2(hidden, asset, state)


def adapt_rdah_query(hidden: Tensor, asset: RDHAAsset, state: RDHAState) -> Tensor:
    """Query-only C forward with no label, fit, update, or gradient surface."""

    return _apply_rdah_phase2(hidden, asset, state)


__all__ = [
    "CANDIDATE_A",
    "CANDIDATE_B",
    "CANDIDATE_C",
    "D127DACandidateError",
    "D127ResourceReceipt",
    "FSRGAsset",
    "FSRGState",
    "JOINT_PROJ_INPUT_DIM",
    "LAYER_NORM_EPS",
    "RANK",
    "RDHAAsset",
    "RDHAOuterResult",
    "RDHAState",
    "RELATIVE_RESIDUAL_BUDGET",
    "SUMMARY_DIM",
    "TAP_A",
    "TAP_B",
    "TAP_C",
    "adapt_fsrg_query",
    "adapt_fsrg_support",
    "adapt_rdah_query",
    "adapt_rdah_support",
    "apply_rdah_outer",
    "apply_fsrg_outer",
    "build_rdah_support_summary",
    "class_balanced_support_loss",
    "derive_phase1_fsrg_rho",
    "fit_fsrg_support_state",
    "fit_rdah_support_state",
    "project_rank2_coefficients",
]
