"""Narrow D105-checkpoint bridge for the frozen NEXT-R1 Phase1 asset/runtime.

This module owns no experiment matrix, scorer, truth reader, or tuning logic.
It joins the two pinned Phase1 archives without pickle, reconstructs the exact
D105 model, captures the signed ``joint_proj.0`` linear output, and supplies
the already-frozen NEXT-R1 asset and row-runtime interfaces.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from . import stage2_next_r1_assets as assets
from . import stage2_next_r1_fabr as fabr
from . import stage2_next_r1_matrix as matrix
from . import stage2_next_r1_runtime as runtime
from . import stage2_next_r1_tsl as tsl
from . import stage2_d129_joint6_heads as d129_heads
from . import stage2_zid_student_t_qknn as qknn
from .grb_jp4_cfm_phase1_held_builder import build_phase1_qknn_locks
from .stage2_d105_phase1_bundle import (
    build_d105_exact_model_from_checkpoint,
    load_d105_exact_sha_bound_checkpoint,
)


SCHEMA = "cvs.phase1.next_r1.real_bridge.v1"
JOIN_SCHEMA = "cvs.phase1.next_r1.real_join.v1"
SMOKE_SCHEMA = "cvs.phase1.next_r1.real_k1_k5_smoke.v1"
SELECTED_RECEIPT_SCHEMA = "cvs.phase1.d106.ls_received_iq_receipt.v1"
SELECTED_MEMBERS = (
    "received_iq",
    "receiver_ids",
    "day_ids",
    "physical_ids",
    "scenario_names",
    "observation_ids",
)
LABEL_MEMBERS = (
    "z_dom",
    "pre_relu",
    "receiver_ids",
    "day_ids",
    "tx_labels",
    "physical_ids",
)
ROW_COUNT = 588


class NextR1RealError(ValueError):
    """The exact real-checkpoint bridge contract did not close."""


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _require_sha(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or value.lower() != value:
        raise NextR1RealError(f"{name} must be a lowercase SHA256")
    try:
        int(value, 16)
    except ValueError as error:
        raise NextR1RealError(f"{name} must be a lowercase SHA256") from error
    return value


def _read_pinned(path: str | Path, expected_sha256: str, name: str) -> bytes:
    source = Path(path)
    expected = _require_sha(expected_sha256, f"{name}_sha256")
    if source.is_symlink() or not source.is_file():
        raise NextR1RealError(f"{name} must be a regular non-symlink file")
    value = source.read_bytes()
    if _sha(value) != expected:
        raise NextR1RealError(f"{name} SHA256 drift")
    return value


def _strings(value: np.ndarray, *, name: str, rows: int) -> tuple[str, ...]:
    array = np.asarray(value)
    if array.ndim != 1 or array.shape[0] != rows or array.dtype.kind not in "US":
        raise NextR1RealError(f"{name} must be a non-object string [{rows}] array")
    result = tuple(str(item) for item in array.tolist())
    if any(not item for item in result):
        raise NextR1RealError(f"{name} contains an empty value")
    return result


def _load_npz(value: bytes, *, members: tuple[str, ...], name: str) -> dict[str, np.ndarray]:
    try:
        with np.load(io.BytesIO(value), allow_pickle=False) as archive:
            if tuple(archive.files) != members:
                raise NextR1RealError(f"{name} member/order drift")
            result = {member: np.asarray(archive[member]) for member in members}
    except (OSError, ValueError) as error:
        raise NextR1RealError(f"{name} is not a no-pickle NPZ") from error
    if any(array.dtype.hasobject for array in result.values()):
        raise NextR1RealError(f"{name} contains an object array")
    return result


@dataclass(frozen=True, slots=True)
class NextR1RealRows:
    received_iq: np.ndarray
    receiver_ids: tuple[str, ...]
    day_ids: tuple[str, ...]
    tx_labels: tuple[str, ...]
    physical_ids: tuple[str, ...]
    scenario_names: tuple[str, ...]
    observation_ids: tuple[str, ...]
    receiver_registry: tuple[str, ...]
    class_registry: tuple[str, ...]
    receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        iq = np.asarray(self.received_iq)
        if iq.dtype != np.dtype("<f4") or iq.ndim != 3 or iq.shape[0] != ROW_COUNT or iq.shape[1] != 2:
            raise NextR1RealError("received_iq must be little-endian float32 [588,2,T]")
        if not np.isfinite(iq).all() or iq.shape[2] < 1:
            raise NextR1RealError("received_iq must be finite and nonempty")
        if len(set(self.physical_ids)) != ROW_COUNT or len(set(self.observation_ids)) != ROW_COUNT:
            raise NextR1RealError("physical and observation IDs must be globally unique")
        if len(self.receiver_registry) != 7 or len(self.class_registry) != 6:
            raise NextR1RealError("real bridge requires the frozen 7x6 registry")
        counts: dict[tuple[str, str], int] = {}
        for receiver, label in zip(self.receiver_ids, self.tx_labels):
            counts[(receiver, label)] = counts.get((receiver, label), 0) + 1
        if set(counts) != set((r, c) for r in self.receiver_registry for c in self.class_registry):
            raise NextR1RealError("Phase1 rows do not form the complete receiver/class grid")
        if any(value != 14 for value in counts.values()):
            raise NextR1RealError("every Phase1 receiver/class cell must contain 14 physical rows")
        frozen = np.array(iq, dtype=np.float32, copy=True, order="C")
        frozen.setflags(write=False)
        object.__setattr__(self, "received_iq", frozen)
        object.__setattr__(self, "receipt", MappingProxyType(dict(self.receipt)))


def load_next_r1_real_rows(
    *,
    selected_iq_archive: str | Path,
    selected_iq_archive_sha256: str,
    selected_iq_receipt: str | Path,
    selected_iq_receipt_sha256: str,
    ls_label_join_archive: str | Path,
    ls_label_join_archive_sha256: str,
) -> NextR1RealRows:
    """Load and exactly join the two caller-pinned 588-row Phase1 archives."""

    iq_bytes = _read_pinned(selected_iq_archive, selected_iq_archive_sha256, "selected_iq_archive")
    label_bytes = _read_pinned(ls_label_join_archive, ls_label_join_archive_sha256, "ls_label_join_archive")
    receipt_bytes = _read_pinned(selected_iq_receipt, selected_iq_receipt_sha256, "selected_iq_receipt")
    try:
        selected_receipt = json.loads(receipt_bytes.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise NextR1RealError("selected_iq_receipt must be UTF-8 JSON") from error
    if not isinstance(selected_receipt, dict):
        raise NextR1RealError("selected_iq_receipt must be a JSON object")
    required = {
        "schema": SELECTED_RECEIPT_SCHEMA,
        "archive_sha256": selected_iq_archive_sha256,
        "row_count": ROW_COUNT,
        "contains_only_selected_ls_rows": True,
        "source_pool_labels_persisted": False,
        "clean_iq_access": False,
        "target_access": False,
        "formal_query_access": False,
    }
    if any(selected_receipt.get(key) != expected for key, expected in required.items()):
        raise NextR1RealError("selected_iq_receipt legality or archive binding drift")
    iq_npz = _load_npz(iq_bytes, members=SELECTED_MEMBERS, name="selected_iq_archive")
    label_npz = _load_npz(label_bytes, members=LABEL_MEMBERS, name="ls_label_join_archive")
    iq = np.asarray(iq_npz["received_iq"])
    if iq.dtype != np.dtype("<f4") or iq.ndim != 3 or iq.shape[:2] != (ROW_COUNT, 2):
        raise NextR1RealError("selected received_iq shape/dtype drift")
    iq_receivers = _strings(iq_npz["receiver_ids"], name="selected.receiver_ids", rows=ROW_COUNT)
    iq_days = _strings(iq_npz["day_ids"], name="selected.day_ids", rows=ROW_COUNT)
    iq_physical = _strings(iq_npz["physical_ids"], name="selected.physical_ids", rows=ROW_COUNT)
    label_receivers = _strings(label_npz["receiver_ids"], name="labels.receiver_ids", rows=ROW_COUNT)
    label_days = _strings(label_npz["day_ids"], name="labels.day_ids", rows=ROW_COUNT)
    label_physical = _strings(label_npz["physical_ids"], name="labels.physical_ids", rows=ROW_COUNT)
    if (iq_receivers, iq_days, iq_physical) != (label_receivers, label_days, label_physical):
        raise NextR1RealError("selected IQ and L_s label rows do not join exactly")
    labels = _strings(label_npz["tx_labels"], name="labels.tx_labels", rows=ROW_COUNT)
    receivers = tuple(sorted(set(iq_receivers)))
    classes = tuple(sorted(set(labels)))
    join_receipt = {
        "schema": JOIN_SCHEMA,
        "selected_iq_archive_sha256": selected_iq_archive_sha256,
        "selected_iq_receipt_sha256": selected_iq_receipt_sha256,
        "ls_label_join_archive_sha256": ls_label_join_archive_sha256,
        "row_count": ROW_COUNT,
        "physical_id_root_sha256": _sha("\n".join(iq_physical).encode()),
        "label_join_only": True,
        "historical_features_consumed": False,
        "pickle_allowed": False,
    }
    return NextR1RealRows(
        received_iq=iq,
        receiver_ids=iq_receivers,
        day_ids=iq_days,
        tx_labels=labels,
        physical_ids=iq_physical,
        scenario_names=_strings(iq_npz["scenario_names"], name="scenario_names", rows=ROW_COUNT),
        observation_ids=_strings(iq_npz["observation_ids"], name="observation_ids", rows=ROW_COUNT),
        receiver_registry=receivers,
        class_registry=classes,
        receipt=join_receipt,
    )


class NextR1RealModelBridge:
    """No-write functional perturbations around one frozen D105 model."""

    def __init__(self, model: Any, rows: NextR1RealRows, checkpoint_sha256: str, device: Any) -> None:
        try:
            import torch
        except ImportError as error:  # pragma: no cover
            raise NextR1RealError("PyTorch is required for the real bridge") from error
        self.model = model.eval()
        self.rows = rows
        self.checkpoint_sha256 = _require_sha(checkpoint_sha256, "checkpoint_sha256")
        self.device = torch.device(device)
        self._named = dict(self.model.id_backbone.named_parameters())
        for block_id in fabr.BLOCK_TIE_ORDER:
            keys = fabr.canonical_parameter_keys(block_id)
            if any(key not in self._named for key in keys):
                raise NextR1RealError(f"model is missing frozen FABR block {block_id}")
            if sum(int(self._named[key].numel()) for key in keys) != fabr.BLOCK_DIMENSIONS[block_id]:
                raise NextR1RealError(f"model FABR block dimension drift: {block_id}")
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def _linear(self) -> Any:
        try:
            return self.model.id_backbone.cls_head.joint_proj[0]
        except (AttributeError, IndexError, TypeError) as error:
            raise NextR1RealError("joint_proj.0 linear module is unavailable") from error

    def _forward(
        self,
        iq: Any,
        *,
        grad: bool,
        parameter_overrides: Mapping[str, Any] | None = None,
    ) -> tuple[Any, Any]:
        import torch
        from torch.func import functional_call

        captured: list[Any] = []
        handle = self._linear().register_forward_hook(lambda _m, _i, output: captured.append(output))
        try:
            context = torch.enable_grad() if grad else torch.no_grad()
            with context:
                if parameter_overrides is None:
                    output = self.model.id_backbone(
                        iq, y=None, return_aux=True, domain_labels=None
                    )
                else:
                    output = functional_call(
                        self.model.id_backbone,
                        dict(parameter_overrides),
                        (iq,),
                        {"y": None, "return_aux": True, "domain_labels": None},
                        strict=False,
                    )
        finally:
            handle.remove()
        if not isinstance(output, Mapping) or "logits" not in output or len(captured) != 1:
            raise NextR1RealError("D105 id_backbone output/tap contract drift")
        logits, pre = output["logits"], captured[0]
        if pre.ndim != 2 or int(pre.shape[1]) != fabr.Z_DIM:
            raise NextR1RealError("joint_proj.0 must produce [N,160]")
        aux = output.get("feat_joint")
        if aux is not None and (aux.shape != pre.shape or not torch.equal(torch.relu(pre), aux)):
            raise NextR1RealError("joint_proj.0 pre-ReLU tap does not totalize feat_joint")
        return logits, pre

    def _indices_tensor(self, indices: Sequence[int]) -> Any:
        import torch

        values = np.asarray(tuple(indices), dtype=np.int64)
        if values.ndim != 1 or values.size < 1 or np.any(values < 0) or np.any(values >= ROW_COUNT):
            raise NextR1RealError("row indices are outside the pinned Phase1 archive")
        array = np.ascontiguousarray(
            self.rows.received_iq[values], dtype=np.float32
        )
        # N607's PyTorch/NumPy pair rejects ``torch.from_numpy`` at the C-API
        # boundary even for an exact numpy.ndarray.  The standard writable
        # buffer protocol is ABI-independent; clone gives the tensor owned
        # storage before the local array leaves scope.
        tensor = torch.frombuffer(
            memoryview(array), dtype=torch.float32, count=int(array.size)
        ).reshape(tuple(int(value) for value in array.shape)).clone()
        return tensor.to(self.device)

    def forward_indices(
        self,
        indices: Sequence[int],
        *,
        block_id: str | None = None,
        basis: np.ndarray | None = None,
        coefficient: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        import torch

        overrides: dict[str, Any] | None = None
        if block_id is not None:
            if basis is None or coefficient is None:
                raise NextR1RealError("perturbed forward requires basis and coefficient")
            keys = fabr.canonical_parameter_keys(block_id)
            direction = np.asarray(basis, dtype=np.float64) @ np.asarray(
                coefficient, dtype=np.float64
            )
            if direction.shape != (fabr.BLOCK_DIMENSIONS[block_id],) or not np.isfinite(direction).all():
                raise NextR1RealError("FABR perturbation shape/value drift")
            overrides = {}
            offset = 0
            for key in keys:
                parameter = self._named[key]
                count = int(parameter.numel())
                delta = torch.as_tensor(
                    direction[offset : offset + count],
                    device=parameter.device,
                    dtype=parameter.dtype,
                ).reshape(parameter.shape)
                overrides[key] = parameter.detach() + delta
                offset += count
        logits, pre = self._forward(
            self._indices_tensor(indices), grad=False, parameter_overrides=overrides
        )
        z160 = fabr.signed_pre_relu160(
            pre.detach().cpu().numpy().astype(np.float32, copy=False)
        )
        return logits.detach().cpu().numpy().astype(np.float32, copy=False), z160

    def gradient_blocks(self, indices: Sequence[int], *, microbatch_size: int = 8) -> tuple[assets.Phase1GradientBlock, ...]:
        """Compute actual per-sample TX-CE gradients for all four frozen blocks."""

        import torch
        import torch.nn.functional as F

        ordered = tuple(int(item) for item in indices)
        if len(set(ordered)) != len(ordered) or microbatch_size < 1:
            raise NextR1RealError("gradient rows must be unique and microbatch_size positive")
        class_index = {label: index for index, label in enumerate(self.rows.class_registry)}
        results: list[assets.Phase1GradientBlock] = []
        for block_id in fabr.BLOCK_TIE_ORDER:
            params = tuple(self._named[key] for key in fabr.canonical_parameter_keys(block_id))
            for parameter in params:
                parameter.requires_grad_(True)
            rows: list[np.ndarray] = []
            try:
                for start in range(0, len(ordered), microbatch_size):
                    batch_indices = ordered[start : start + microbatch_size]
                    logits, _pre = self._forward(self._indices_tensor(batch_indices), grad=True)
                    if logits.ndim != 2 or int(logits.shape[1]) != len(self.rows.class_registry):
                        raise NextR1RealError("TX-logit/class-registry dimension drift")
                    targets = torch.as_tensor(
                        [class_index[self.rows.tx_labels[index]] for index in batch_indices],
                        dtype=torch.long,
                        device=logits.device,
                    )
                    losses = F.cross_entropy(logits, targets, reduction="none")
                    for local in range(len(batch_indices)):
                        grads = torch.autograd.grad(
                            losses[local], params, retain_graph=local + 1 < len(batch_indices), create_graph=False
                        )
                        flat = torch.cat(tuple(value.detach().reshape(-1) for value in grads))
                        rows.append(flat.cpu().numpy().astype(np.float32, copy=False))
            finally:
                for parameter in params:
                    parameter.requires_grad_(False)
            matrix_value = np.ascontiguousarray(np.stack(rows), dtype=np.float32)
            results.append(
                assets.Phase1GradientBlock(
                    block_id=block_id,
                    gradients=matrix_value,
                    phase1_receiver_ids=tuple(self.rows.receiver_ids[index] for index in ordered),
                    phase1_physical_ids=tuple(self.rows.physical_ids[index] for index in ordered),
                )
            )
        return tuple(results)


def load_next_r1_real_model(
    rows: NextR1RealRows, *, checkpoint_path: str | Path, checkpoint_sha256: str, device: Any
) -> tuple[NextR1RealModelBridge, Mapping[str, Any]]:
    checkpoint, load_receipt = load_d105_exact_sha_bound_checkpoint(checkpoint_path, checkpoint_sha256)
    model, build_receipt = build_d105_exact_model_from_checkpoint(
        checkpoint, input_len=int(rows.received_iq.shape[2]), device=device
    )
    bridge = NextR1RealModelBridge(model, rows, checkpoint_sha256, device)
    return bridge, MappingProxyType({"load": dict(load_receipt), "build": dict(build_receipt)})


def _fit_indices(rows: NextR1RealRows, held_receiver: str, held_class: str) -> tuple[int, ...]:
    if held_receiver not in rows.receiver_registry or held_class not in rows.class_registry:
        raise NextR1RealError("held receiver/class is outside the frozen registry")
    return tuple(
        index
        for index, (receiver, label) in enumerate(zip(rows.receiver_ids, rows.tx_labels))
        if receiver != held_receiver and label != held_class
    )


def _strict_phase1_predictions(logits: np.ndarray, *, name: str) -> np.ndarray:
    try:
        return fabr.strict_top1_predictions(
            np.ascontiguousarray(logits, dtype=np.float32)
        )
    except fabr.FABRTieError as error:
        raise NextR1RealError(
            f"Phase1 {name} directional validation has an unresolved exact tie"
        ) from error


def _cells_and_loo(
    bridge: NextR1RealModelBridge, fit_indices: tuple[int, ...], held_receiver: str, held_class: str
) -> tuple[tuple[tsl.Phase1Cell, ...], tuple[assets.TSLPhysicalLOOBinding, ...]]:
    _logits, z_all = bridge.forward_indices(fit_indices)
    by_pair: dict[tuple[str, str], list[tuple[str, np.ndarray]]] = {}
    for position, index in enumerate(fit_indices):
        by_pair.setdefault((bridge.rows.receiver_ids[index], bridge.rows.tx_labels[index]), []).append(
            (bridge.rows.physical_ids[index], z_all[position])
        )
    cells: list[tsl.Phase1Cell] = []
    for (receiver, label), values in sorted(by_pair.items()):
        ordered = sorted(values, key=lambda item: item[0])
        cells.append(tsl.Phase1Cell(receiver, label, tuple(item[0] for item in ordered), np.stack([item[1] for item in ordered])))
    active_receivers = tuple(item for item in bridge.rows.receiver_registry if item != held_receiver)
    active_classes = tuple(item for item in bridge.rows.class_registry if item != held_class)
    bindings: list[assets.TSLPhysicalLOOBinding] = []
    for validation_receiver in active_receivers:
        support_rows: list[np.ndarray] = []
        support_labels: list[str] = []
        support_ids: list[str] = []
        for label in active_classes:
            candidates = sorted(
                (pid, z) for (receiver, cell_label), values in by_pair.items()
                if receiver != validation_receiver and cell_label == label for pid, z in values
            )
            for physical_id, z160 in candidates[:5]:
                support_ids.append(physical_id); support_labels.append(label); support_rows.append(z160)
        for validation_class in active_classes:
            validation = sorted(by_pair[(validation_receiver, validation_class)], key=lambda item: item[0])
            fold = tsl.Phase1PhysicalLOOFold(
                fold_id=f"{validation_receiver}__{validation_class}",
                support_z160=np.stack(support_rows),
                support_labels=tuple(support_labels),
                registered_classes=active_classes,
                support_physical_ids=tuple(support_ids),
                validation_z160=np.stack([item[1] for item in validation]),
                validation_labels=tuple(validation_class for _ in validation),
                validation_physical_ids=tuple(item[0] for item in validation),
            )
            bindings.append(assets.TSLPhysicalLOOBinding(validation_receiver, validation_class, fold))
    return tuple(cells), tuple(bindings)


def build_next_r1_real_asset(
    bridge: NextR1RealModelBridge,
    *,
    held_receiver: str,
    held_class: str,
    row_phase1_seal_sha256: str,
    microbatch_size: int = 8,
) -> assets.NextR1Phase1AssetBundle:
    """Build one actual 420-row FABR+30-cell TSL asset for a frozen LOCO pair."""

    fit_indices = _fit_indices(bridge.rows, held_receiver, held_class)
    if len(fit_indices) != 420:
        raise NextR1RealError("held receiver/class must leave exactly 420 fit rows")
    blocks = bridge.gradient_blocks(fit_indices, microbatch_size=microbatch_size)
    labels = tuple(bridge.rows.tx_labels[index] for index in fit_indices)
    receivers = tuple(bridge.rows.receiver_ids[index] for index in fit_indices)
    physical_ids = tuple(bridge.rows.physical_ids[index] for index in fit_indices)
    cells, loo = _cells_and_loo(bridge, fit_indices, held_receiver, held_class)
    fit_root = assets.phase1_fit_physical_id_root(receivers, labels, physical_ids)
    cell_root = tsl.phase1_cell_physical_id_root(cells)
    row_seal = _require_sha(row_phase1_seal_sha256, "row_phase1_seal_sha256")
    fold_seal = assets.Phase1FoldSeal(
        fold_id=f"next-r1-real__{held_receiver}__{held_class}",
        held_receiver=held_receiver,
        held_class=held_class,
        checkpoint_sha256=bridge.checkpoint_sha256,
        representation_rule_sha256=fabr.REPRESENTATION_RULE_SHA256,
        row_phase1_seal_sha256=row_seal,
        phase1_fit_physical_id_root_sha256=fit_root,
        phase1_cell_physical_id_root_sha256=cell_root,
    )
    class_index = {label: index for index, label in enumerate(bridge.rows.class_registry)}
    baseline_logits, _ = bridge.forward_indices(fit_indices)
    baseline_predictions = _strict_phase1_predictions(
        baseline_logits, name="baseline"
    )
    truth = np.asarray([class_index[label] for label in labels], dtype=np.int64)
    baseline_correct = {label: int(np.sum((np.asarray(labels) == label) & (baseline_predictions == class_index[label]))) for label in sorted(set(labels))}
    totals = {label: labels.count(label) for label in sorted(set(labels))}
    validation_seal = _sha(_canonical({"schema": SCHEMA, "row_seal": row_seal, "validation": "phase1_tx"}))

    def validate(block_id: str, basis: np.ndarray, coefficient: np.ndarray, callback_labels: tuple[str, ...]) -> assets.Phase1DirectionalValidation:
        if callback_labels != labels:
            raise NextR1RealError("directional validation label order drift")
        perturbed_logits, _ = bridge.forward_indices(fit_indices, block_id=block_id, basis=basis, coefficient=coefficient)
        repeated_logits, _ = bridge.forward_indices(fit_indices)
        predicted = _strict_phase1_predictions(
            perturbed_logits, name="perturbed"
        )
        perturbed_correct = {label: int(np.sum((np.asarray(labels) == label) & (predicted == class_index[label]))) for label in totals}
        return assets.Phase1DirectionalValidation(
            basis_sha256=_sha(np.ascontiguousarray(basis, dtype=np.float64).tobytes()),
            coefficient_sha256=_sha(np.ascontiguousarray(coefficient, dtype=np.float32).tobytes()),
            baseline_total_correct=int(np.sum(baseline_predictions == truth)),
            perturbed_total_correct=int(np.sum(predicted == truth)),
            baseline_per_class_correct=baseline_correct,
            perturbed_per_class_correct=perturbed_correct,
            per_class_total=totals,
            forward_action_max_abs_delta=float(np.max(np.abs(perturbed_logits - baseline_logits))),
            repeated_forward_jitter_max_abs_delta=float(np.max(np.abs(repeated_logits - baseline_logits))),
            validation_seal_sha256=validation_seal,
        )

    return assets.build_next_r1_phase1_assets(
        blocks, labels, fold_seal, bridge.rows.receiver_registry, bridge.rows.class_registry,
        cells, loo, validate,
    )


def verified_checkpoint_smoke(
    bridge: NextR1RealModelBridge,
    bundle: assets.NextR1Phase1AssetBundle,
    *,
    checkpoint_path: str | Path,
    checkpoint_sha256: str,
    smoke_indices: Sequence[int],
) -> runtime.NextR1VerifiedCheckpointSmoke:
    """Execute and seal one no-label real-checkpoint representation smoke."""

    if type(bridge) is not NextR1RealModelBridge:
        raise NextR1RealError("checkpoint smoke requires an exact real model bridge")
    if type(bundle) is not assets.NextR1Phase1AssetBundle:
        raise NextR1RealError("checkpoint smoke requires an exact Phase1 asset bundle")
    expected = _require_sha(checkpoint_sha256, "checkpoint_sha256")
    _read_pinned(checkpoint_path, expected, "checkpoint")
    if (
        bridge.checkpoint_sha256 != expected
        or bundle.receipt["checkpoint_sha256"] != expected
    ):
        raise NextR1RealError("checkpoint smoke bridge/bundle/file SHA256 drift")
    ordered = tuple(int(value) for value in smoke_indices)
    if not ordered or len(set(ordered)) != len(ordered):
        raise NextR1RealError("checkpoint smoke indices must be nonempty and unique")
    logits, z160 = bridge.forward_indices(ordered)
    repeated_logits, repeated_z160 = bridge.forward_indices(ordered)
    if (
        logits.dtype != np.float32
        or logits.ndim != 2
        or logits.shape[0] != len(ordered)
        or logits.shape[1] != len(bridge.rows.class_registry)
        or z160.dtype != np.float32
        or z160.shape != (len(ordered), fabr.Z_DIM)
        or not np.isfinite(logits).all()
        or not np.isfinite(z160).all()
        or not np.array_equal(logits, repeated_logits)
        or not np.array_equal(z160, repeated_z160)
    ):
        raise NextR1RealError("real checkpoint smoke forward is nonfinite or non-repeatable")
    payload = {
        "schema": runtime.CHECKPOINT_SMOKE_SCHEMA,
        "completed": True,
        "actual_checkpoint_sha256": bundle.receipt["checkpoint_sha256"],
        "representation_rule_sha256": bundle.receipt["representation_rule_sha256"],
        "builder_bundle_sha256": bundle.bundle_sha256,
        "row_phase1_seal_sha256": bundle.receipt["row_phase1_seal_sha256"],
        "verification_source": "pinned_checkpoint_real_forward",
        "real_forward_rows": len(ordered),
        "real_forward_logits_sha256": _sha(np.ascontiguousarray(logits).tobytes()),
        "real_forward_z160_sha256": _sha(np.ascontiguousarray(z160).tobytes()),
        "real_forward_repeat_exact": True,
    }
    payload["checkpoint_smoke_receipt_sha256"] = _sha(_canonical(payload))
    return runtime.verify_next_r1_checkpoint_smoke(
        payload, bundle=bundle, row_phase1_seal_sha256=bundle.receipt["row_phase1_seal_sha256"]
    )


def make_forward_callback(bridge: NextR1RealModelBridge, indices: Sequence[int], block_id: str, basis: np.ndarray):
    physical_ids = tuple(bridge.rows.physical_ids[index] for index in indices)

    def callback(_token: object, coefficient: np.ndarray) -> fabr.FABRForwardBatch:
        _logits, z160 = bridge.forward_indices(indices, block_id=block_id, basis=basis, coefficient=coefficient)
        return fabr.FABRForwardBatch(z160, physical_ids)

    return callback


def no_truth_head(context: runtime.NextR1ArmContext) -> np.ndarray:
    """Support-only cosine head; query labels/truth/roles are not accepted."""

    logits = np.empty((len(context.query.z160), len(context.registered_classes)), dtype=np.float32)
    for column, label in enumerate(context.registered_classes):
        prototype = np.mean(context.support.z160[np.asarray(context.support_labels) == label], axis=0)
        norm = float(np.linalg.norm(prototype))
        if not np.isfinite(norm) or norm <= 0.0:
            raise NextR1RealError("support-only prototype is zero/non-finite")
        logits[:, column] = context.query.z160 @ (prototype / norm).astype(np.float32)
    return np.ascontiguousarray(logits, dtype=np.float32)


def frozen_qknn_head(context: runtime.NextR1ArmContext) -> np.ndarray:
    """Run the already-frozen Phase1 qKNN lock for the context's active K.

    The callback sees only the support labels/representations and the query
    representation exposed by :class:`NextR1ArmContext`.  It has no scorer,
    truth, receiver role, class quota, or cross-query assignment input.
    """

    if type(context) is not runtime.NextR1ArmContext:
        raise NextR1RealError("frozen qKNN requires an exact NEXT-R1 arm context")
    counts = tuple(context.support_labels.count(value) for value in context.registered_classes)
    if len(set(counts)) != 1 or counts[0] not in matrix.K_VALUES:
        raise NextR1RealError("frozen qKNN context does not contain balanced K1/K5 support")
    lock = build_phase1_qknn_locks()[counts[0]]
    bank = qknn.build_typed_zid_support_bank(
        context.support.z160,
        context.support_labels,
        context.registered_classes,
        config=lock,
    )
    metric = qknn.identity_shared_psd_metric(config=lock)
    logits = qknn.score_zid_student_t_logits(
        bank, context.query.z160, metric=metric
    )
    return np.ascontiguousarray(logits, dtype=np.float32)


def frozen_d92_full160_head(context: runtime.NextR1ArmContext) -> np.ndarray:
    """Fit and score the unchanged historical D92 Full160 K5 mechanism.

    NEXT-R1's LOCO registry is frozen as five retained classes followed by the
    held seen-class proxy.  This directional partition is consumed only by the
    historical comparison head; FABR and TSL do not receive it.  K1 never calls
    this function because the runtime enforces an exact Q/F/L alias.
    """

    if type(context) is not runtime.NextR1ArmContext:
        raise NextR1RealError("historical D92 requires an exact NEXT-R1 arm context")
    counts = tuple(context.support_labels.count(value) for value in context.registered_classes)
    if len(set(counts)) != 1 or counts[0] != 5:
        raise NextR1RealError("historical D92 Full160 callback is frozen to K5")
    fit = d129_heads.fit_d92_full160(
        context.support.z160,
        context.support_labels,
        context.registered_classes,
        old_class_count=matrix.CLASS_COUNT - 1,
    )
    if type(fit.state) is not d129_heads.D129AffineHeadState:
        raise NextR1RealError("historical D92 Full160 did not produce an affine K5 state")
    logits = d129_heads.score_d129_affine_head(fit.state, context.query.z160)
    return np.ascontiguousarray(logits, dtype=np.float32)


__all__ = [
    "NextR1RealError", "NextR1RealRows", "NextR1RealModelBridge",
    "load_next_r1_real_rows", "load_next_r1_real_model", "build_next_r1_real_asset",
    "verified_checkpoint_smoke", "make_forward_callback", "no_truth_head",
    "frozen_qknn_head", "frozen_d92_full160_head",
]
