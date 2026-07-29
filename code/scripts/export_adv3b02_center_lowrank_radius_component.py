"""Export the fixed ADV3B02 Phase1 labeled-train geometry into the v2 codec.

Only normalized ``z_id`` aggregates cross the streaming boundary.  The script
never serializes a full-precision centroid, sample feature, count, or source
path; the sole output is the strict three-file v2 deployment component.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import inspect
import json
from pathlib import Path
import sys
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.checkpoint_loading import build_exact_ssdg_model_from_checkpoint
from cvsrffi.phase1_center_lowrank_prototype_bundle import (
    ALLOWED_NPZ_MEMBERS,
    FEATURE_DIM,
    FEATURE_SCHEMA,
    MANIFEST_NAME,
    MANIFEST_SHA_NAME,
    NPZ_NAME,
    PENDING_OUTER_JOINT_SEAL,
    V1_ALLOWED_MEMBERS,
    build_center_lowrank_component,
    radius_generation_proof_sha256,
    save_center_lowrank_component,
    sha256_file,
    validate_center_lowrank_component,
    v1_payload_sha256,
)
from cvsrffi.phase1_geometry_streaming import (
    Phase1GeometryInMemory,
    Phase1GeometryStreaming,
)


FILE_MEMBER_ALLOWLIST = {NPZ_NAME, MANIFEST_NAME, MANIFEST_SHA_NAME}
CLASS_BINDING_SCHEMA = "phase1_tx_class_handle_binding_v1"
PROVENANCE_STATUS = PENDING_OUTER_JOINT_SEAL
STREAM_HASH_SCHEMA = "phase1_normalized_z_id_class_domain_ordered_stream_v1"
PHASE1_LABELED_RATIO = 0.07
PHASE1_UNLABELED_RATIO = 0.63
PHASE1_SOURCE_VAL_RATIO = 0.30


class Phase1ExportError(ValueError):
    """Raised when an export input or output violates the sealed contract."""


BatchAdapter = Callable[
    [Any, torch.device], tuple[torch.Tensor, torch.Tensor, torch.Tensor]
]


@dataclass(frozen=True)
class Phase1AggregateExport:
    """Aggregate geometry plus the matching two-pass ordered stream proof."""

    geometry: Phase1GeometryInMemory
    phase1_stream_sha256: str

    @property
    def domain_class_centroids(self) -> torch.Tensor:
        return self.geometry.domain_class_centroids

    @property
    def radius_p90_cosine_distance(self) -> torch.Tensor:
        return self.geometry.radius_p90_cosine_distance

    @property
    def active_cell_mask(self) -> torch.Tensor:
        return self.geometry.active_cell_mask

    @property
    def domain_class_counts(self) -> torch.Tensor:
        return self.geometry.domain_class_counts


def _new_stream_digest() -> "hashlib._Hash":
    digest = hashlib.sha256()
    digest.update(STREAM_HASH_SCHEMA.encode("ascii") + b"\0")
    return digest


def _update_stream_digest(
    digest: Any,
    normalized_z_id: torch.Tensor,
    class_index: torch.Tensor,
    domain_index: torch.Tensor,
) -> None:
    features = np.ascontiguousarray(
        normalized_z_id.detach().cpu().numpy(), dtype="<f4"
    )
    classes = np.ascontiguousarray(class_index.detach().cpu().numpy(), dtype="<i8")
    domains = np.ascontiguousarray(domain_index.detach().cpu().numpy(), dtype="<i8")
    for row, class_value, domain_value in zip(features, classes, domains):
        digest.update(np.asarray(class_value, dtype=">i8").tobytes())
        digest.update(np.asarray(domain_value, dtype=">i8").tobytes())
        digest.update(row.tobytes(order="C"))


def _validate_sha256(value: str, field: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 64 or any(ch not in "0123456789abcdef" for ch in normalized):
        raise Phase1ExportError(f"{field} must be a lowercase SHA256 hex digest")
    return normalized


def verify_file_sha256(
    path: str | Path, expected_sha256: str, *, field: str
) -> str:
    """Hash a regular file and fail closed before its contents are consumed."""

    candidate = Path(path)
    if not candidate.is_file():
        raise Phase1ExportError(f"{field} must reference an existing regular file")
    expected = _validate_sha256(expected_sha256, f"expected_{field}_sha256")
    actual = sha256_file(candidate)
    if actual != expected:
        raise Phase1ExportError(
            f"{field} SHA256 mismatch: expected={expected} actual={actual}"
        )
    return actual


def class_handle_binding_sha256(class_registry: Sequence[str]) -> str:
    handles = tuple(str(value) for value in class_registry)
    if not handles or len(set(handles)) != len(handles) or any(not item for item in handles):
        raise Phase1ExportError("class registry handles must be non-empty and unique")
    encoded = json.dumps(
        {
            "schema": CLASS_BINDING_SCHEMA,
            "class_id_to_handle": [
                {"class_index": index, "class_handle": handle}
                for index, handle in enumerate(handles)
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def class_handles_from_binding_source(
    path: str | Path,
    *,
    expected_phase1_txs: Sequence[str],
) -> tuple[str, ...]:
    """Reuse an existing old-class handle registry after checking TX order only."""

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Phase1ExportError("class binding source is unreadable") from exc
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") != "cvs.phase2.d19_adv3b02_class_binding.v1"
        or set(payload) != {"schema", "checkpoint_sha256", "entries", "evidence"}
        or not isinstance(payload.get("entries"), list)
    ):
        raise Phase1ExportError("class binding source schema drift")
    expected = tuple(str(value) for value in expected_phase1_txs)
    entries = payload["entries"]
    if len(entries) != len(expected):
        raise Phase1ExportError("class binding source class-count drift")
    handles: list[str] = []
    for index, (entry, expected_tx) in enumerate(zip(entries, expected)):
        if (
            not isinstance(entry, Mapping)
            or set(entry)
            != {"class_index", "phase1_tx", "registered_class_handle"}
            or int(entry.get("class_index", -1)) != index
            or str(entry.get("phase1_tx", "")) != expected_tx
            or not str(entry.get("registered_class_handle", "")).strip()
        ):
            raise Phase1ExportError("class binding source TX/order drift")
        handles.append(str(entry["registered_class_handle"]))
    if len(set(handles)) != len(handles):
        raise Phase1ExportError("class binding source handles are not unique")
    return tuple(handles)


def _default_batch_adapter(
    batch: Any, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not isinstance(batch, (tuple, list)) or len(batch) != 4:
        raise Phase1ExportError(
            "Phase1 export loader must yield SSDG batches (x,y,d,meta)"
        )
    x, y, domain, _meta = batch
    if not torch.is_tensor(x) or not torch.is_tensor(y) or not torch.is_tensor(domain):
        raise Phase1ExportError("SSDG batch x, y, and d entries must be tensors")
    return (
        x.to(device=device, non_blocking=True),
        y.to(device=device, non_blocking=True).view(-1).long(),
        domain.to(device=device, non_blocking=True).view(-1).long(),
    )


def _normalized_z_id(
    model: torch.nn.Module,
    x: torch.Tensor,
    class_index: torch.Tensor,
    domain_index: torch.Tensor,
    *,
    feature_dim: int,
    num_classes: int,
) -> torch.Tensor:
    output = model(
        x,
        y_tx=class_index,
        grl_lambda=0.0,
        return_aux=True,
        domain_labels=domain_index,
    )
    if not isinstance(output, Mapping) or not torch.is_tensor(output.get("z_id")):
        raise Phase1ExportError("checkpoint model must return tensor output['z_id']")
    tx_logits = output.get("tx_logits")
    if not torch.is_tensor(tx_logits) or tuple(tx_logits.shape) != (
        int(class_index.numel()),
        int(num_classes),
    ):
        raise Phase1ExportError(
            "checkpoint TX-logit width does not match the Phase1 class registry"
        )
    z_id = output["z_id"].detach().float()
    if z_id.ndim != 2 or tuple(z_id.shape) != (int(class_index.numel()), feature_dim):
        raise Phase1ExportError(
            f"z_id must have shape [N,{feature_dim}], got {tuple(z_id.shape)}"
        )
    if not bool(torch.isfinite(z_id).all()):
        raise Phase1ExportError("z_id contains non-finite values")
    norms = torch.linalg.vector_norm(z_id, dim=1)
    if bool(torch.any(norms <= 1.0e-12)):
        raise Phase1ExportError("z_id contains a zero-norm row")
    return F.normalize(z_id, p=2.0, dim=1)


def export_from_loader(
    model: torch.nn.Module,
    loader: Iterable[Any],
    *,
    device: torch.device | str,
    num_domains: int,
    num_classes: int,
    feature_dim: int = FEATURE_DIM,
    required_cell_mask: torch.Tensor | None = None,
    min_samples_per_cell: int = 2,
    radius_histogram_bins: int = 4096,
    batch_adapter: BatchAdapter | None = None,
) -> Phase1AggregateExport:
    """Run two no-gradient passes and return aggregate-only geometry in memory."""

    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise Phase1ExportError("CUDA export was requested but CUDA is unavailable")
    if int(feature_dim) != FEATURE_DIM:
        raise Phase1ExportError(f"ADV3B02 export requires feature_dim={FEATURE_DIM}")
    adapter = batch_adapter or _default_batch_adapter
    stream_parameters = inspect.signature(Phase1GeometryStreaming).parameters
    dimension_field = (
        "feature_dim" if "feature_dim" in stream_parameters else "embedding_dim"
    )
    stream = Phase1GeometryStreaming(
        **{
            "num_domains": int(num_domains),
            "num_classes": int(num_classes),
            dimension_field: int(feature_dim),
            "required_cell_mask": required_cell_mask,
            "min_samples_per_cell": int(min_samples_per_cell),
            "radius_histogram_bins": int(radius_histogram_bins),
        }
    )
    model.to(resolved_device)
    model.eval()

    pass_stream_hashes: list[str] = []
    for pass_index in (1, 2):
        batch_count = 0
        stream_digest = _new_stream_digest()
        with torch.no_grad():
            for batch in loader:
                x, classes, domains = adapter(batch, resolved_device)
                if classes.ndim != 1 or domains.ndim != 1 or classes.shape != domains.shape:
                    raise Phase1ExportError("class/domain indices must be aligned [N]")
                if int(x.shape[0]) != int(classes.numel()):
                    raise Phase1ExportError("input and class/domain batch sizes differ")
                z_id = _normalized_z_id(
                    model,
                    x,
                    classes,
                    domains,
                    feature_dim=int(feature_dim),
                    num_classes=int(num_classes),
                )
                _update_stream_digest(stream_digest, z_id, classes, domains)
                if pass_index == 1:
                    update = stream.update_first_pass
                else:
                    update = stream.update_second_pass
                update_parameters = inspect.signature(update).parameters
                if "normalized_z_id" in update_parameters:
                    update(
                        normalized_z_id=z_id,
                        class_index=classes,
                        domain_index=domains,
                    )
                else:
                    update(
                        z_id=z_id,
                        class_index=classes,
                        domain_index=domains,
                    )
                batch_count += 1
        if batch_count == 0:
            raise Phase1ExportError("Phase1 labeled-train loader is empty")
        pass_stream_hashes.append(stream_digest.hexdigest())
        if pass_index == 1:
            stream.begin_second_pass()
    if pass_stream_hashes[0] != pass_stream_hashes[1]:
        raise Phase1ExportError(
            "ordered normalized z_id/class/domain stream differs between Phase1 passes"
        )
    return Phase1AggregateExport(
        geometry=stream.finalize(),
        phase1_stream_sha256=pass_stream_hashes[0],
    )


def _tensor_to_float32_numpy(value: torch.Tensor) -> np.ndarray:
    """Torch 2.1 / NumPy 2.x-safe aggregate conversion without sample buffers."""

    return np.asarray(value.detach().cpu().tolist(), dtype=np.float32)


def _quantize_centroids(centroids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vectors = np.asarray(centroids, dtype=np.float32)
    if vectors.ndim != 3 or vectors.shape[2] != FEATURE_DIM:
        raise Phase1ExportError(f"centroids must be [D,C,{FEATURE_DIM}]")
    if not np.isfinite(vectors).all():
        raise Phase1ExportError("centroids must be finite")
    max_abs = np.max(np.abs(vectors), axis=2)
    scale32 = np.where(max_abs > 0.0, max_abs / 127.0, 1.0).astype(np.float32)
    scale16 = scale32.astype(np.float16)
    if not np.isfinite(scale16).all() or bool(np.any(scale16 <= 0.0)):
        raise Phase1ExportError("centroid FP16 quantization scale is invalid")
    quantized = np.clip(
        np.rint(vectors / scale32[:, :, None]), -127, 127
    ).astype(np.int8)
    if bool(np.any(quantized == -128)):
        raise Phase1ExportError("centroid quantization emitted forbidden -128")
    return quantized, scale16


def build_v1_aggregate_payload(
    geometry: Phase1GeometryInMemory,
    *,
    domain_registry: Sequence[str],
    class_registry: Sequence[str],
) -> dict[str, np.ndarray]:
    """Build the codec's strict v1 aggregate payload without writing it."""

    centroids = _tensor_to_float32_numpy(geometry.domain_class_centroids)
    active = np.asarray(geometry.active_cell_mask.detach().cpu().tolist(), dtype=np.uint8)
    domains = tuple(str(value) for value in domain_registry)
    classes = tuple(str(value) for value in class_registry)
    if centroids.shape[:2] != (len(domains), len(classes)):
        raise Phase1ExportError("registries do not match aggregate geometry shape")
    if len(set(domains)) != len(domains) or any(not item for item in domains):
        raise Phase1ExportError("domain registry must contain unique non-empty handles")
    if len(set(classes)) != len(classes) or any(not item for item in classes):
        raise Phase1ExportError("class registry must contain unique non-empty handles")
    quantized, scale = _quantize_centroids(centroids)
    quantized[active == 0] = 0
    scale[active == 0] = np.float16(1.0)
    payload = {
        "domain_class_q": quantized,
        "domain_class_scale": scale,
        "domain_class_mask": active,
        "domain_registry": np.asarray(domains, dtype=np.str_),
        "class_registry": np.asarray(classes, dtype=np.str_),
        "feature_schema": np.asarray(FEATURE_SCHEMA, dtype=np.str_),
    }
    if set(payload) != V1_ALLOWED_MEMBERS:
        raise AssertionError("internal v1 aggregate allowlist drift")
    return payload


def v1_aggregate_sha256(payload: Mapping[str, np.ndarray]) -> str:
    return v1_payload_sha256(payload)


def _assert_empty_output(output_dir: str | Path) -> Path:
    root = Path(output_dir)
    if root.exists() and not root.is_dir():
        raise Phase1ExportError("output must be a directory path")
    if root.is_dir() and any(root.iterdir()):
        raise Phase1ExportError("output directory must be absent or empty")
    return root


def save_aggregate_component(
    output_dir: str | Path,
    geometry: Phase1AggregateExport,
    *,
    domain_registry: Sequence[str],
    class_registry: Sequence[str],
    checkpoint_sha256: str,
    class_handle_binding_sha256_value: str,
    generation_code_sha256: str,
    generation_config_sha256: str,
) -> dict[str, Any]:
    """Quantize in memory, invoke the v2 codec, and enforce its file allowlist."""

    root = _assert_empty_output(output_dir)
    v1_payload = build_v1_aggregate_payload(
        geometry,
        domain_registry=domain_registry,
        class_registry=class_registry,
    )
    v1_sha = v1_aggregate_sha256(v1_payload)
    radius = _tensor_to_float32_numpy(geometry.radius_p90_cosine_distance)
    generation_proof_sha = radius_generation_proof_sha256(
        v1_payload,
        radius,
        phase1_stream_sha256=geometry.phase1_stream_sha256,
        checkpoint_sha256=checkpoint_sha256,
        class_handle_binding_sha256=class_handle_binding_sha256_value,
        generation_code_sha256=generation_code_sha256,
        generation_config_sha256=generation_config_sha256,
    )
    payload, manifest = build_center_lowrank_component(
        v1_payload,
        radius_p90_cosine_distance=radius,
        phase1_stream_sha256=geometry.phase1_stream_sha256,
        radius_generation_proof_sha256_value=generation_proof_sha,
        checkpoint_sha256=_validate_sha256(checkpoint_sha256, "checkpoint_sha256"),
        class_handle_binding_sha256=_validate_sha256(
            class_handle_binding_sha256_value, "class_handle_binding_sha256"
        ),
        generation_code_sha256=_validate_sha256(
            generation_code_sha256, "generation_code_sha256"
        ),
        generation_config_sha256=_validate_sha256(
            generation_config_sha256, "generation_config_sha256"
        ),
        provenance_status=PROVENANCE_STATUS,
        formal_phase2_eligible=False,
    )
    saved = save_center_lowrank_component(root, payload, manifest)
    members = {item.name for item in root.iterdir()}
    if members != FILE_MEMBER_ALLOWLIST:
        raise Phase1ExportError("saved component directory violates file allowlist")
    validation = validate_center_lowrank_component(
        root,
        expected_checkpoint_sha256=checkpoint_sha256,
        expected_class_handle_binding_sha256=class_handle_binding_sha256_value,
    )
    return {
        "members": sorted(members),
        "v1_component_sha256": v1_sha,
        "aggregate_generation_proof_sha256": generation_proof_sha,
        "pre_sign_content_root_sha256": saved["pre_sign_content_root_sha256"],
        "component_npz_sha256": saved["component_npz_sha256"],
        "manifest_sha256": saved["manifest_sha256"],
        "validation": validation,
    }


def _build_checkpoint_data_context(
    checkpoint: Mapping[str, Any],
    *,
    wisig_pkl: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
):
    from SSDG import train_ssdg as ssdg

    parser = ssdg.build_arg_parser()
    data_args = parser.parse_args(
        ["--output_dir", str(Path(".phase1_export_parser_only"))]
    )
    checkpoint_args = checkpoint.get("args")
    if not isinstance(checkpoint_args, Mapping):
        raise Phase1ExportError("checkpoint args must be a mapping")
    for key, value in checkpoint_args.items():
        setattr(data_args, str(key), value)
    # The frozen checkpoint contributes model state and its source-domain
    # registry, but a newly generated Phase1 deployment component must obey the
    # current protocol rather than inherit the historical 0.10/0.70/0.20 split
    # (whose train-only label ratio is 0.125).  These three values are locked by
    # 项目.md to 0.07/0.63/0.30, yielding rho_label=0.1 exactly.
    data_args.labeled_ratio = PHASE1_LABELED_RATIO
    data_args.unlabeled_ratio = PHASE1_UNLABELED_RATIO
    data_args.source_val_ratio = PHASE1_SOURCE_VAL_RATIO
    data_args.wisig_pkl = str(wisig_pkl)
    data_args.device = str(device)
    data_args.eval_batch_size = int(batch_size)
    data_args.num_workers = int(num_workers)
    data_ctx = ssdg._build_ssdg_wisig_data(data_args, device)
    split_info = dict(data_ctx.get("split_info") or {})
    if float(split_info.get("rho_label", 1.0)) > 0.1 + 1.0e-8:
        raise Phase1ExportError("checkpoint Phase1 labeled split violates rho_label<=0.1")
    if int(split_info.get("labeled_size", 0)) <= 0:
        raise Phase1ExportError("checkpoint Phase1 labeled split is empty")
    return ssdg, data_args, data_ctx


def _ssdg_batch_adapter(ssdg, domain_label_map: Mapping[int, int]) -> BatchAdapter:
    def adapt(
        batch: Any, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y, extra = ssdg.move_batch(batch, device)
        domain = ssdg.domain_from_extra(extra, domain_label_map, device)
        if domain is None:
            raise Phase1ExportError("SSDG labeled batch has no domain labels")
        return x, y.view(-1).long(), domain.view(-1).long()

    return adapt


def _domain_registry(
    domain_label_map: Mapping[int, int], *, domain_kind: str
) -> tuple[str, ...]:
    if not domain_label_map:
        raise Phase1ExportError("Phase1 domain label map is empty")
    result = [""] * len(domain_label_map)
    for raw, compact in domain_label_map.items():
        index = int(compact)
        if index < 0 or index >= len(result) or result[index]:
            raise Phase1ExportError("Phase1 domain label map is not a compact bijection")
        result[index] = f"{str(domain_kind)}:{int(raw)}"
    if any(not item for item in result):
        raise Phase1ExportError("Phase1 domain label map has a registry gap")
    return tuple(result)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export the fixed ADV3B02 Phase1 labeled-train center-lowrank component."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--wisig-pkl", required=True)
    parser.add_argument("--output", required=True, help="Absent or empty output directory")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-wisig-sha256", required=True)
    parser.add_argument("--expected-class-handle-binding-sha256", required=True)
    parser.add_argument(
        "--class-binding-source",
        default="",
        help=(
            "Optional existing old-class TX-to-registered-handle registry. "
            "Its historical checkpoint field is not used; current TX order is checked."
        ),
    )
    parser.add_argument("--generation-config", required=True)
    parser.add_argument("--expected-generation-config-sha256", required=True)
    parser.add_argument("--expected-generation-code-sha256", default="")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--min-samples-per-cell", type=int, default=2)
    parser.add_argument("--radius-histogram-bins", type=int, default=4096)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if int(args.batch_size) <= 0 or int(args.num_workers) < 0:
        raise Phase1ExportError("batch-size must be positive and num-workers non-negative")
    output = _assert_empty_output(args.output)
    checkpoint_path = Path(args.checkpoint).resolve()
    wisig_path = Path(args.wisig_pkl).resolve()
    generation_config_path = Path(args.generation_config).resolve()

    checkpoint_sha = verify_file_sha256(
        checkpoint_path, args.expected_checkpoint_sha256, field="checkpoint"
    )
    verify_file_sha256(wisig_path, args.expected_wisig_sha256, field="wisig_pkl")
    generation_config_sha = verify_file_sha256(
        generation_config_path,
        args.expected_generation_config_sha256,
        field="generation_config",
    )
    generation_code_sha = sha256_file(Path(__file__).resolve())
    if str(args.expected_generation_code_sha256).strip():
        expected_code = _validate_sha256(
            args.expected_generation_code_sha256, "expected_generation_code_sha256"
        )
        if generation_code_sha != expected_code:
            raise Phase1ExportError(
                "generation code SHA256 does not match the expected binding"
            )

    resolved_device = torch.device(args.device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise Phase1ExportError("CUDA export was requested but CUDA is unavailable")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise Phase1ExportError("checkpoint root must be a mapping")
    ssdg, data_args, data_ctx = _build_checkpoint_data_context(
        checkpoint,
        wisig_pkl=wisig_path,
        device=resolved_device,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
    )
    phase1_txs = tuple(str(value) for value in data_ctx["class_id_to_tx"])
    class_registry = (
        class_handles_from_binding_source(
            args.class_binding_source,
            expected_phase1_txs=phase1_txs,
        )
        if str(args.class_binding_source).strip()
        else phase1_txs
    )
    actual_binding_sha = class_handle_binding_sha256(class_registry)
    expected_binding_sha = _validate_sha256(
        args.expected_class_handle_binding_sha256,
        "expected_class_handle_binding_sha256",
    )
    if actual_binding_sha != expected_binding_sha:
        raise Phase1ExportError(
            "checkpoint Phase1 TX-to-class-handle binding SHA256 mismatch"
        )
    domain_registry = _domain_registry(
        data_ctx["domain_label_map"], domain_kind=str(data_args.wisig_domain)
    )
    model, checkpoint_audit = build_exact_ssdg_model_from_checkpoint(
        checkpoint,
        input_len=int(data_ctx["input_len"]),
        device=resolved_device,
        ssdg_module=ssdg,
    )
    if int(checkpoint_audit["num_domains_from_state"]) != len(domain_registry):
        raise Phase1ExportError(
            "checkpoint domain-head width does not match the Phase1 domain registry"
        )
    geometry = export_from_loader(
        model,
        data_ctx["probe_train_loader"],
        device=resolved_device,
        num_domains=len(domain_registry),
        num_classes=len(class_registry),
        feature_dim=FEATURE_DIM,
        min_samples_per_cell=int(args.min_samples_per_cell),
        radius_histogram_bins=int(args.radius_histogram_bins),
        batch_adapter=_ssdg_batch_adapter(ssdg, data_ctx["domain_label_map"]),
    )
    saved = save_aggregate_component(
        output,
        geometry,
        domain_registry=domain_registry,
        class_registry=class_registry,
        checkpoint_sha256=checkpoint_sha,
        class_handle_binding_sha256_value=actual_binding_sha,
        generation_code_sha256=generation_code_sha,
        generation_config_sha256=generation_config_sha,
    )
    print(
        json.dumps(
            {
                "status": PENDING_OUTER_JOINT_SEAL,
                "output_members": saved["members"],
                "checkpoint_sha256": checkpoint_sha,
                "class_handle_binding_sha256": actual_binding_sha,
                "aggregate_generation_proof_sha256": saved[
                    "aggregate_generation_proof_sha256"
                ],
                "pre_sign_content_root_sha256": saved[
                    "pre_sign_content_root_sha256"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
