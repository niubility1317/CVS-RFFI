"""D36 compiled joint int8 adaptation and registration head.

Only labeled, already-admitted LEO_weak support enters ``fit``.  The learned
diagonal/rank-2 operator is compiled into one symmetric-int8 weight per class;
the query path stores no optimizer or FP32 target prototype and makes one
independent all-registered-class decision per row.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F


FEATURE_DIM = 288
Z_DIM = 160
ALLOWED_NEW_CLASS_COUNTS = (2, 5, 10, 20)
SCHEMA = "cvs.phase2.d36_compiled_joint_int8.v1"
LOGIT_SCALE = 18.0
RANK = 2


class D36CompiledJointInt8Error(ValueError):
    pass


def _readonly(value: np.ndarray, dtype: Any) -> np.ndarray:
    contiguous = np.ascontiguousarray(value, dtype=dtype)
    result = np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(
        contiguous.shape
    )
    result.setflags(write=False)
    return result


def _unit_rows(value: np.ndarray, name: str) -> np.ndarray:
    rows = np.asarray(value)
    if (
        rows.dtype != np.float32
        or rows.ndim != 2
        or rows.shape[1] != FEATURE_DIM
        or len(rows) < 1
        or not np.isfinite(rows).all()
    ):
        raise D36CompiledJointInt8Error(
            f"{name} must be finite float32 [N,{FEATURE_DIM}]"
        )
    norms = np.linalg.norm(rows, axis=1)
    if not np.all(np.abs(norms - 1.0) <= 1.0e-4):
        raise D36CompiledJointInt8Error(f"{name} must be upstream unit rows")
    return np.ascontiguousarray(rows)


def _support(
    features: np.ndarray,
    labels: Sequence[str],
    classes: Sequence[str],
    name: str,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], int]:
    rows = _unit_rows(features, f"{name} features")
    y = np.asarray(tuple(str(v) for v in labels))
    registry = tuple(str(v) for v in classes)
    if (
        len(y) != len(rows)
        or not registry
        or len(set(registry)) != len(registry)
        or set(y.tolist()) != set(registry)
    ):
        raise D36CompiledJointInt8Error(f"{name} registry drift")
    counts = [int(np.sum(y == name)) for name in registry]
    if min(counts) < 1 or len(set(counts)) != 1:
        raise D36CompiledJointInt8Error(f"{name} must be symmetric K-shot")
    targets = np.asarray([registry.index(str(v)) for v in y], dtype=np.int64)
    return rows, targets, registry, counts[0]


@dataclass(frozen=True)
class D36CompiledJointConfig:
    arm: str = "C"
    learning_rate: float = 0.03

    def __post_init__(self) -> None:
        arm = str(self.arm).upper()
        if arm.startswith("D36-"):
            arm = arm[4:]
        if arm not in {"A", "B", "C"} or float(self.learning_rate) != 0.03:
            raise D36CompiledJointInt8Error("D36 fixed arm/lr lock drift")
        object.__setattr__(self, "arm", arm)

    @property
    def rank(self) -> int:
        return 0 if self.arm == "A" else RANK


@dataclass(frozen=True)
class D36CompiledJointState:
    schema: str
    classes: tuple[str, ...]
    old_class_count: int
    compiled_qint8: np.ndarray
    compiled_scales_fp16: np.ndarray
    compiled_inverse_norms_fp16: np.ndarray
    radii_fp16: np.ndarray
    calibration_kind: str
    calibration_fp16: np.ndarray
    arm: str

    def __post_init__(self) -> None:
        count = len(self.classes)
        if (
            self.schema != SCHEMA
            or not 2 <= self.old_class_count <= 6
            or len(set(self.classes)) != count
            or any(not str(value) for value in self.classes)
            or (
                count - self.old_class_count != 0
                and count - self.old_class_count not in ALLOWED_NEW_CLASS_COUNTS
            )
            or self.compiled_qint8.shape != (count, FEATURE_DIM)
            or self.compiled_qint8.dtype != np.int8
            or self.compiled_scales_fp16.shape != (count,)
            or self.compiled_scales_fp16.dtype != np.float16
            or self.compiled_inverse_norms_fp16.shape != (count,)
            or self.compiled_inverse_norms_fp16.dtype != np.float16
            or self.radii_fp16.shape != (count,)
            or self.radii_fp16.dtype != np.float16
            or self.calibration_fp16.dtype != np.float16
            or self.calibration_kind not in {"none", "constant", "margin6_irls"}
            or (
                self.calibration_kind == "none"
                and self.calibration_fp16.shape != (0,)
            )
            or (
                self.calibration_kind == "constant"
                and self.calibration_fp16.shape != (1,)
            )
            or (
                self.calibration_kind == "margin6_irls"
                and self.calibration_fp16.shape != (6,)
            )
            or (
                count == self.old_class_count
                and self.calibration_kind != "none"
            )
            or not np.isfinite(self.compiled_scales_fp16).all()
            or not np.isfinite(self.compiled_inverse_norms_fp16).all()
            or not np.isfinite(self.radii_fp16).all()
            or not np.isfinite(self.calibration_fp16).all()
            or bool(np.any(self.compiled_scales_fp16 <= 0))
            or bool(np.any(self.compiled_inverse_norms_fp16 <= 0))
            or bool(np.any(self.radii_fp16 <= 0))
        ):
            raise D36CompiledJointInt8Error("D36 compiled state drift")
        for field, dtype in (
            ("compiled_qint8", np.int8),
            ("compiled_scales_fp16", np.float16),
            ("compiled_inverse_norms_fp16", np.float16),
            ("radii_fp16", np.float16),
            ("calibration_fp16", np.float16),
        ):
            object.__setattr__(self, field, _readonly(getattr(self, field), dtype))

    @property
    def persistent_state_bytes(self) -> int:
        return int(
            self.compiled_qint8.nbytes
            + self.compiled_scales_fp16.nbytes
            + self.compiled_inverse_norms_fp16.nbytes
            + self.radii_fp16.nbytes
            + self.calibration_fp16.nbytes
        )


@dataclass(frozen=True)
class D36CompiledJointResult:
    before_state: D36CompiledJointState
    state: D36CompiledJointState
    training_trace: tuple[dict[str, Any], ...]
    geometry_audit: dict[str, Any]
    resource_audit: dict[str, Any]


def _transform(
    x: torch.Tensor,
    d: torch.Tensor,
    u: torch.Tensor | None,
    v: torch.Tensor | None,
    shrink: float,
) -> torch.Tensor:
    z = x * torch.exp(d)[None, :]
    if u is not None and v is not None and shrink > 0.0:
        z = z + shrink * ((x @ v) @ u.T)
    return F.normalize(z, dim=1)


def _centroids(z: torch.Tensor, targets: torch.Tensor, count: int) -> torch.Tensor:
    return torch.stack(
        [F.normalize(z[targets == i].mean(dim=0), dim=0) for i in range(count)]
    )


def _class_cvar(losses: torch.Tensor, targets: torch.Tensor, count: int) -> torch.Tensor:
    values = torch.stack([losses[targets == i].mean() for i in range(count)])
    return torch.topk(values, k=min(2, count)).values.mean()


def _torch_copy(value: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
    """Copy small support arrays without the Torch/NumPy C-ABI bridge.

    N607 currently pairs Torch 2.1 with NumPy 2.x, where ``torch.from_numpy``
    rejects otherwise valid ``numpy.ndarray`` instances.  D36 support tensors
    are tiny, so a Python-list copy is deterministic and avoids that ABI path.
    """

    return torch.tensor(np.asarray(value).tolist(), dtype=dtype)


def _margin(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    true = logits[torch.arange(len(logits)), targets]
    other = logits.clone()
    other[torch.arange(len(logits)), targets] = -1.0e9
    return true - other.max(dim=1).values


def _robust_prototype(rows: np.ndarray) -> tuple[np.ndarray, float, float]:
    center = rows.mean(axis=0).astype(np.float32)
    center /= np.linalg.norm(center)
    for _ in range(3):
        distance = np.clip(1.0 - rows @ center, 0.0, 2.0)
        delta = max(float(np.median(distance)), 1.0e-4)
        weights = np.minimum(1.0, delta / np.maximum(distance, 1.0e-6))
        center = np.sum(rows * weights[:, None], axis=0)
        center /= np.linalg.norm(center)
    similarity = rows @ rows.T
    medoid = rows[int(np.argmax(np.mean(similarity, axis=1)))]
    radius = max(float(np.median(np.clip(1.0 - rows @ center, 0.0, 2.0))), 1.0e-4)
    alpha = 0.0 if len(rows) == 1 else min(0.5, radius / (radius + 0.1))
    prototype = (1.0 - alpha) * center + alpha * medoid
    prototype /= np.linalg.norm(prototype)
    return prototype.astype(np.float32), radius, alpha


def _quantize(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    peak = np.max(np.abs(rows), axis=1).astype(np.float32)
    scale = peak / np.float32(127.0)
    q = np.clip(np.rint(rows / scale[:, None]), -127, 127).astype(np.int8)
    inverse = 1.0 / np.linalg.norm(q.astype(np.float32), axis=1)
    return q, scale.astype(np.float16), inverse.astype(np.float16)


def _fuse_ground_z(
    prototypes: np.ndarray,
    radii: np.ndarray,
    k_shot: int,
    old_count: int,
    anchors: np.ndarray | None,
    ground_radii: np.ndarray | None,
) -> np.ndarray:
    result = prototypes.copy()
    if anchors is None:
        return result
    anchor_values = np.asarray(anchors, dtype=np.float32).copy()
    if anchor_values.shape != (old_count, Z_DIM) or not np.isfinite(anchor_values).all():
        raise D36CompiledJointInt8Error("ground_anchor_z shape drift")
    anchor_norm = np.linalg.norm(anchor_values, axis=1, keepdims=True)
    if np.any(anchor_norm <= 1.0e-12):
        raise D36CompiledJointInt8Error("ground_anchor_z zero norm")
    anchor_values /= anchor_norm
    ground_r = (
        np.full(old_count, 0.1, dtype=np.float32)
        if ground_radii is None
        else np.asarray(ground_radii, dtype=np.float32)
    )
    if ground_r.shape != (old_count,) or np.any(ground_r <= 0) or not np.isfinite(ground_r).all():
        raise D36CompiledJointInt8Error("ground_anchor_radius drift")
    for c in range(old_count):
        target_z = result[c, :Z_DIM].copy()
        target_z_norm = float(np.linalg.norm(target_z))
        if target_z_norm <= 1.0e-12:
            raise D36CompiledJointInt8Error("target z block zero norm")
        u_t = radii[c] + 1.0 / np.sqrt(k_shot)
        weight = np.clip(0.25 * u_t / (u_t + ground_r[c]), 0.0, 0.20)
        scaled_anchor = anchor_values[c] * np.float32(target_z_norm)
        result[c, :Z_DIM] = (1.0 - weight) * target_z + weight * scaled_anchor
        result[c] /= np.linalg.norm(result[c])
    return result


def _compile_prototypes(
    prototypes: np.ndarray,
    d: np.ndarray,
    u: np.ndarray | None,
    v: np.ndarray | None,
    shrink: float,
) -> np.ndarray:
    compiled = prototypes * np.exp(d)[None, :]
    if u is not None and v is not None:
        compiled += shrink * ((prototypes @ u) @ v.T)
    compiled /= np.linalg.norm(compiled, axis=1, keepdims=True)
    return compiled.astype(np.float32)


def _base_scores(
    rows: np.ndarray, q: np.ndarray, inverse: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    q_fp32 = q.astype(np.float32)
    # Row-local execution makes the deployed score independent of query-batch
    # composition and does not construct a query-query surface.
    cosine = np.stack(
        [np.asarray(row @ q_fp32.T, dtype=np.float32) for row in rows]
    )
    cosine *= inverse.astype(np.float32)[None, :]
    return np.asarray(LOGIT_SCALE * cosine, dtype=np.float32), cosine


def _psi(
    logits: np.ndarray,
    cosine: np.ndarray,
    radii: np.ndarray,
    old_count: int,
) -> np.ndarray:
    old = logits[:, :old_count]
    new = logits[:, old_count:]
    old_order = np.sort(old, axis=1)
    new_order = np.sort(new, axis=1)
    old_best = np.argmax(old, axis=1)
    new_best = np.argmax(new, axis=1)
    rho_o = (1.0 - cosine[np.arange(len(logits)), old_best]) / radii[old_best]
    rho_n = (1.0 - cosine[np.arange(len(logits)), old_count + new_best]) / radii[
        old_count + new_best
    ]
    return np.stack(
        [
            np.ones(len(logits), dtype=np.float32),
            new_order[:, -1] - old_order[:, -1],
            old_order[:, -1] - old_order[:, -2],
            new_order[:, -1] - new_order[:, -2],
            rho_o - rho_n,
            np.minimum(rho_o, rho_n),
        ],
        axis=1,
    ).astype(np.float32)


def _fit_irls(psi: np.ndarray, role: np.ndarray) -> tuple[np.ndarray, list[dict[str, Any]]]:
    w = np.zeros(6, dtype=np.float64)
    class_weight = np.where(
        role == 1,
        0.5 / max(float(np.mean(role == 1)), 1.0e-6),
        0.5 / max(float(np.mean(role == 0)), 1.0e-6),
    )
    trace = []
    x = psi.astype(np.float64)
    y = role.astype(np.float64)
    for step in range(5):
        probability = 1.0 / (1.0 + np.exp(-np.clip(x @ w, -20.0, 20.0)))
        gradient = x.T @ (class_weight * (probability - y)) + 0.1 * w
        curvature = class_weight * probability * (1.0 - probability)
        hessian = (x.T * curvature) @ x + 0.1 * np.eye(6)
        update = np.linalg.solve(hessian, gradient)
        w -= update
        trace.append({"irls_step": step + 1, "update_norm": float(np.linalg.norm(update))})
    return w.astype(np.float32), trace


def fit_d36_compiled_joint_int8(
    old_support_features: np.ndarray,
    old_support_labels: Sequence[str],
    old_registered_classes: Sequence[str],
    new_support_features: np.ndarray,
    new_support_labels: Sequence[str],
    new_registered_classes: Sequence[str],
    fisher_log_diag: np.ndarray,
    *,
    config: D36CompiledJointConfig | None = None,
    ground_anchor_z: np.ndarray | None = None,
    ground_anchor_radius: np.ndarray | None = None,
) -> D36CompiledJointResult:
    locked = config or D36CompiledJointConfig()
    old_x, old_y, old_classes, old_k = _support(
        old_support_features, old_support_labels, old_registered_classes, "D36 old support"
    )
    new_x, new_y0, new_classes, new_k = _support(
        new_support_features, new_support_labels, new_registered_classes, "D36 new support"
    )
    if (
        not 2 <= len(old_classes) <= 6
        or len(new_classes) not in ALLOWED_NEW_CLASS_COUNTS
        or old_k != new_k
        or set(old_classes) & set(new_classes)
    ):
        raise D36CompiledJointInt8Error("D36 matched disjoint K-shot registry drift")
    fisher = np.asarray(fisher_log_diag)
    if fisher.dtype != np.float32 or fisher.shape != (FEATURE_DIM,) or not np.isfinite(fisher).all():
        raise D36CompiledJointInt8Error("fisher_log_diag must be finite float32[288]")
    device = torch.device("cpu")
    torch.manual_seed(36)
    x_old = _torch_copy(old_x, torch.float32).to(device)
    x_new = _torch_copy(new_x, torch.float32).to(device)
    y_old = _torch_copy(old_y, torch.int64).to(device)
    y_new = _torch_copy(new_y0 + len(old_classes), torch.int64).to(device)
    d = _torch_copy(fisher, torch.float32).requires_grad_()
    shrink = float((old_k - 1) / (old_k + 3))
    if locked.rank:
        basis = _torch_copy(
            np.concatenate((old_x, new_x), axis=0)[:RANK].T.copy(),
            torch.float32,
        )
        u = (1.0e-3 * basis).requires_grad_()
        v = (1.0e-3 * torch.flip(basis, dims=[1])).requires_grad_()
    else:
        u = v = None
    params = [d] + ([] if u is None else [u, v])
    optimizer = torch.optim.SGD(params, lr=locked.learning_rate)
    trace: list[dict[str, Any]] = []
    for step in range(6):
        optimizer.zero_grad()
        z = _transform(x_old, d, u, v, shrink)
        proto = _centroids(z, y_old, len(old_classes))
        logits = LOGIT_SCALE * (z @ proto.T)
        per_row = F.cross_entropy(logits, y_old, reduction="none")
        cvar = _class_cvar(per_row, y_old, len(old_classes))
        loss = per_row.mean() + 0.5 * cvar
        loss = loss + 0.1 * torch.mean((d - _torch_copy(fisher, torch.float32)) ** 2)
        if u is not None:
            loss = loss + 1.0e-3 * (torch.mean(u * u) + torch.mean(v * v))
        loss.backward()
        grad_norm = float(torch.sqrt(sum(torch.sum(p.grad * p.grad) for p in params)))
        optimizer.step()
        trace.append(
            {
                "phase": "Stage2-B",
                "step": step + 1,
                "loss": float(loss.detach()),
                "cvar_top2": float(cvar.detach()),
                "support_accuracy": float(torch.mean((torch.argmax(logits, dim=1) == y_old).float()).detach()),
                "gradient_norm": grad_norm,
            }
        )
    d_b = d.detach().clone()
    u_b = None if u is None else u.detach().clone()
    v_b = None if v is None else v.detach().clone()
    with torch.no_grad():
        z_b = _transform(x_old, d, u, v, shrink)
        p_b = _centroids(z_b, y_old, len(old_classes))
        margin_b = _margin(LOGIT_SCALE * (z_b @ p_b.T), y_old)
    x_all = torch.cat((x_old, x_new), dim=0)
    y_all = torch.cat((y_old, y_new), dim=0)
    class_count = len(old_classes) + len(new_classes)
    for step in range(6):
        optimizer.zero_grad()
        z = _transform(x_all, d, u, v, shrink)
        proto = _centroids(z, y_all, class_count)
        logits = LOGIT_SCALE * (z @ proto.T)
        per_row = F.cross_entropy(logits, y_all, reduction="none")
        old_ce = per_row[: len(old_x)].mean()
        new_ce = per_row[len(old_x) :].mean()
        cvar = _class_cvar(per_row, y_all, class_count)
        current_old_margin = _margin(logits[: len(old_x)], y_old)
        preserve = F.relu(margin_b - current_old_margin + 0.1).mean()
        penalty = torch.mean((d - d_b) ** 2)
        if u is not None:
            penalty = penalty + torch.mean((u - u_b) ** 2) + torch.mean((v - v_b) ** 2)
        loss = 0.5 * old_ce + 0.5 * new_ce + 0.5 * cvar + preserve + 1.0e-3 * penalty
        loss.backward()
        grad_norm = float(torch.sqrt(sum(torch.sum(p.grad * p.grad) for p in params)))
        optimizer.step()
        trace.append(
            {
                "phase": "Stage2-C",
                "step": step + 1,
                "loss": float(loss.detach()),
                "old_ce": float(old_ce.detach()),
                "new_ce": float(new_ce.detach()),
                "cvar_top2": float(cvar.detach()),
                "preserve_loss": float(preserve.detach()),
                "old_support_accuracy": float(torch.mean((torch.argmax(logits[: len(old_x)], dim=1) == y_old).float()).detach()),
                "new_support_accuracy": float(torch.mean((torch.argmax(logits[len(old_x) :], dim=1) == y_new).float()).detach()),
                "gradient_norm": grad_norm,
            }
        )
    with torch.no_grad():
        transformed = _transform(x_all, d, u, v, shrink).cpu().numpy().astype(np.float32)
    before_transformed = z_b.cpu().numpy().astype(np.float32)
    before_proto = []
    before_radii = []
    for c in range(len(old_classes)):
        proto, radius, _ = _robust_prototype(before_transformed[old_y == c])
        before_proto.append(proto)
        before_radii.append(radius)
    before_proto = np.stack(before_proto).astype(np.float32)
    before_radii_np = np.maximum(np.asarray(before_radii, dtype=np.float32), 1.0e-4)
    if locked.arm in {"B", "C"}:
        before_proto = _fuse_ground_z(
            before_proto,
            before_radii_np,
            old_k,
            len(old_classes),
            ground_anchor_z,
            ground_anchor_radius,
        )
    before_compiled = _compile_prototypes(
        before_proto,
        d_b.cpu().numpy().astype(np.float32),
        None if u_b is None else u_b.cpu().numpy().astype(np.float32),
        None if v_b is None else v_b.cpu().numpy().astype(np.float32),
        shrink,
    )
    before_q, before_scales, before_inverse = _quantize(before_compiled)
    before_state = D36CompiledJointState(
        schema=SCHEMA,
        classes=old_classes,
        old_class_count=len(old_classes),
        compiled_qint8=before_q,
        compiled_scales_fp16=before_scales,
        compiled_inverse_norms_fp16=before_inverse,
        radii_fp16=before_radii_np.astype(np.float16),
        calibration_kind="none",
        calibration_fp16=np.zeros(0, dtype=np.float16),
        arm=locked.arm,
    )
    target_proto = []
    radii = []
    alphas = []
    for c in range(class_count):
        proto, radius, alpha = _robust_prototype(transformed[y_all.cpu().numpy() == c])
        target_proto.append(proto)
        radii.append(radius)
        alphas.append(alpha)
    target_proto = np.stack(target_proto).astype(np.float32)
    radii_np = np.maximum(np.asarray(radii, dtype=np.float32), 1.0e-4)
    anchor_used = locked.arm in {"B", "C"} and ground_anchor_z is not None
    if locked.arm in {"B", "C"}:
        target_proto = _fuse_ground_z(
            target_proto,
            radii_np,
            old_k,
            len(old_classes),
            ground_anchor_z,
            ground_anchor_radius,
        )
    d_np = d.detach().cpu().numpy().astype(np.float32)
    compiled = _compile_prototypes(
        target_proto,
        d_np,
        None if u is None else u.detach().cpu().numpy().astype(np.float32),
        None if v is None else v.detach().cpu().numpy().astype(np.float32),
        shrink,
    )
    q, scales, inverse = _quantize(compiled)
    restored = q.astype(np.float32) * inverse.astype(np.float32)[:, None]
    quantization_error = np.linalg.norm(compiled - restored, axis=1)
    logits, cosine = _base_scores(np.concatenate((old_x, new_x)), q, inverse)
    role = np.concatenate((np.zeros(len(old_x)), np.ones(len(new_x)))).astype(np.int64)
    irls_trace: list[dict[str, Any]] = []
    if locked.arm == "A":
        calibration_kind = "none"
        calibration = np.zeros(0, dtype=np.float16)
    elif locked.arm == "B":
        diff = np.max(logits[:, len(old_classes) :], axis=1) - np.max(
            logits[:, : len(old_classes)], axis=1
        )
        offset = np.clip(-0.5 * (np.median(diff[role == 0]) + np.median(diff[role == 1])), -2.0, 2.0)
        calibration_kind = "constant"
        calibration = np.asarray([offset], dtype=np.float16)
    else:
        features = _psi(logits, cosine, radii_np, len(old_classes))
        weights, irls_trace = _fit_irls(features, role)
        calibration_kind = "margin6_irls"
        calibration = weights.astype(np.float16)
    state = D36CompiledJointState(
        schema=SCHEMA,
        classes=old_classes + new_classes,
        old_class_count=len(old_classes),
        compiled_qint8=q,
        compiled_scales_fp16=scales,
        compiled_inverse_norms_fp16=inverse,
        radii_fp16=radii_np.astype(np.float16),
        calibration_kind=calibration_kind,
        calibration_fp16=calibration,
        arm=locked.arm,
    )
    adapter_parameters = FEATURE_DIM if locked.rank == 0 else FEATURE_DIM + 2 * FEATURE_DIM * RANK
    transform_macs_per_row = FEATURE_DIM + (0 if locked.rank == 0 else 2 * FEATURE_DIM * RANK)
    stage2b_macs = int(
        6 * (len(old_x) * transform_macs_per_row + len(old_x) * len(old_classes) * FEATURE_DIM)
    )
    stage2c_macs = int(
        6 * (len(x_all) * transform_macs_per_row + len(x_all) * class_count * FEATURE_DIM)
    )
    compile_macs = int(
        len(x_all) * FEATURE_DIM
        + class_count * transform_macs_per_row
        + class_count * FEATURE_DIM
    )
    resource = {
        "schema": "cvs.phase2.d36_compiled_joint_int8_resource.v1",
        "active_adapter_parameters": adapter_parameters,
        "adapter_parameter_cap": 50_000,
        "active_parameters_under_50k": adapter_parameters + len(calibration) < 50_000,
        "adaptation_epochs": 12,
        "adaptation_epoch_cap": 20,
        "adaptation_epoch_cap_pass": True,
        "optimizer_steps": 12,
        "optimizer_persistent_state_bytes": 0,
        "estimated_stage2b_adaptation_macs": stage2b_macs,
        "estimated_stage2c_adaptation_macs": stage2c_macs,
        "estimated_total_adaptation_macs": stage2b_macs + stage2c_macs,
        "estimated_compile_macs": compile_macs,
        "persistent_state_bytes": state.persistent_state_bytes,
        "persistent_state_cap_bytes": 50_000,
        "persistent_state_cap_pass": state.persistent_state_bytes <= 50_000,
        "resident_fp32_target_prototype_count": 0,
        "target_old_int8_prototype_count": len(old_classes),
        "target_new_int8_prototype_count": len(new_classes),
        "query_dot_macs": class_count * FEATURE_DIM,
        "query_scalar_ops": class_count * 3 + (6 if locked.arm == "C" else 1 if locked.arm == "B" else 0),
        "dense_query_graph_bytes": 0,
        "query_rows_used_for_fit": 0,
        "query_labels_used_for_fit": False,
        "query_role_oracle_access": False,
        "query_true_batch_class_count_access": False,
        "query_class_quota_access": False,
        "query_batch_global_assignment": False,
        "phase2_query_role_oracle_access": False,
        "phase2_query_true_batch_class_count_access": False,
        "phase2_query_class_quota_access": False,
        "phase2_query_batch_global_assignment": False,
        "clean_sample_access": False,
        "clean_derived_signal_access": False,
        "source_sample_access": False,
        "source_derived_signal_access": False,
        "int8_component_update_access": False,
        "phase2_sample_view_policy": "leo_weak_only_no_clean_access",
        "phase2_query_decision_policy": "per_sample_all_registered_classes",
    }
    geometry = {
        "schema": "cvs.phase2.d36_compiled_joint_int8_geometry.v1",
        "arm": locked.arm,
        "rank": locked.rank,
        "shot_shrinkage": shrink,
        "fisher_log_diag_initialization": True,
        "ground_anchor_z_used": anchor_used,
        "target_prototype_quantization": "symmetric_int8_fp16_scale_inverse_norm_radius",
        "huber_medoid_alpha_by_class": alphas,
        "compiled_int8_quantization_error_mean": float(np.mean(quantization_error)),
        "compiled_int8_quantization_error_max": float(np.max(quantization_error)),
        "calibration_kind": calibration_kind,
        "calibration_fit_scope": "support_self_fit_runner_must_override_with_oof_calibration",
        "irls_trace": irls_trace,
        "all_registered_classes_finite_per_sample": True,
    }
    return D36CompiledJointResult(before_state, state, tuple(trace), geometry, resource)


def base_score_d36_compiled_joint_int8(
    state: D36CompiledJointState, features: np.ndarray
) -> np.ndarray:
    """Return immutable uncalibrated compiled logits for row-local crossfit."""

    rows = _unit_rows(features, "D36 base scoring features")
    logits, _ = _base_scores(
        rows, state.compiled_qint8, state.compiled_inverse_norms_fp16
    )
    return _readonly(logits, np.float32)


def margin_features_d36_compiled_joint_int8(
    state: D36CompiledJointState, features: np.ndarray
) -> np.ndarray:
    """Build the six fixed margin features without applying calibration."""

    if len(state.classes) - state.old_class_count < 2 or state.old_class_count < 2:
        raise D36CompiledJointInt8Error(
            "margin6 requires at least two registered old and new classes"
        )
    rows = _unit_rows(features, "D36 margin feature rows")
    logits, cosine = _base_scores(
        rows, state.compiled_qint8, state.compiled_inverse_norms_fp16
    )
    return _readonly(
        _psi(
            logits,
            cosine,
            state.radii_fp16.astype(np.float32),
            state.old_class_count,
        ),
        np.float32,
    )


def with_oof_calibration_d36_compiled_joint_int8(
    state: D36CompiledJointState,
    margin_features: np.ndarray,
    old_new_roles: np.ndarray,
) -> D36CompiledJointState:
    """Return a copied state whose calibration is fit only from caller OOF rows."""

    psi = np.asarray(margin_features)
    roles = np.asarray(old_new_roles)
    if (
        psi.dtype != np.float32
        or psi.ndim != 2
        or psi.shape[1] != 6
        or roles.shape != (len(psi),)
        or not np.isfinite(psi).all()
        or set(np.asarray(roles, dtype=np.int64).tolist()) != {0, 1}
    ):
        raise D36CompiledJointInt8Error("OOF calibration rows/roles drift")
    roles = roles.astype(np.int64)
    if state.arm == "A":
        kind = "none"
        calibration = np.zeros(0, dtype=np.float16)
    elif state.arm == "B":
        diff = psi[:, 1]
        offset = np.clip(
            -0.5 * (np.median(diff[roles == 0]) + np.median(diff[roles == 1])),
            -2.0,
            2.0,
        )
        kind = "constant"
        calibration = np.asarray([offset], dtype=np.float16)
    else:
        weights, _ = _fit_irls(psi, roles)
        kind = "margin6_irls"
        calibration = weights.astype(np.float16)
    return D36CompiledJointState(
        schema=state.schema,
        classes=state.classes,
        old_class_count=state.old_class_count,
        compiled_qint8=state.compiled_qint8,
        compiled_scales_fp16=state.compiled_scales_fp16,
        compiled_inverse_norms_fp16=state.compiled_inverse_norms_fp16,
        radii_fp16=state.radii_fp16,
        calibration_kind=kind,
        calibration_fp16=calibration,
        arm=state.arm,
    )


def score_d36_compiled_joint_int8(
    state: D36CompiledJointState, features: np.ndarray
) -> np.ndarray:
    rows = _unit_rows(features, "D36 scoring features")
    logits, cosine = _base_scores(
        rows, state.compiled_qint8, state.compiled_inverse_norms_fp16
    )
    if state.calibration_kind == "constant":
        logits[:, state.old_class_count :] += float(state.calibration_fp16[0])
    elif state.calibration_kind == "margin6_irls":
        psi = _psi(
            logits,
            cosine,
            state.radii_fp16.astype(np.float32),
            state.old_class_count,
        )
        delta = np.clip(
            psi @ state.calibration_fp16.astype(np.float32), -2.0, 2.0
        )
        logits[:, state.old_class_count :] += delta[:, None]
    if not np.isfinite(logits).all():
        raise D36CompiledJointInt8Error("non-finite compiled score")
    return _readonly(logits, np.float32)


def predict_d36_compiled_joint_int8(
    state: D36CompiledJointState, features: np.ndarray
) -> np.ndarray:
    scores = score_d36_compiled_joint_int8(state, features)
    return np.asarray(state.classes)[np.argmax(scores, axis=1)]


__all__ = [
    "base_score_d36_compiled_joint_int8",
    "D36CompiledJointConfig",
    "D36CompiledJointInt8Error",
    "D36CompiledJointResult",
    "D36CompiledJointState",
    "fit_d36_compiled_joint_int8",
    "margin_features_d36_compiled_joint_int8",
    "predict_d36_compiled_joint_int8",
    "score_d36_compiled_joint_int8",
    "with_oof_calibration_d36_compiled_joint_int8",
]
