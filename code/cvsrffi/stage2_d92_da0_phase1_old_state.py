"""Source-only Phase1 export of an immutable D92 DA0 old-class state.

The exporter surface accepts only already-received source LEO_weak IQ, a
sealed Phase1 runtime, and source provenance.  It materializes the exact D42
old-prefix affine state in locked joint288 coordinates, then persists only the
predictive codec arrays and sealing receipts.  It never accepts target support,
target packages, query rows, or truth.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from cvsrffi.leo_weak_cache import FORMAL_LEO_WEAK_SCENARIOS, sha256_file
from cvsrffi.stage2_d42_unified_shrinkage_lda import (
    D42OldOnlyFitResult,
    D42UnifiedShrinkageLDAError,
    D42UnifiedShrinkageLDAState,
    FEATURE_DIM,
    fit_d42_old_only,
)
from cvsrffi.stage2_diag_cosine_exploration import forward_zid160, registered_feature


SCHEMA = "cvs.phase1.d92_da0_old_state.v1"
JOINT288_TRANSFORM_SCHEMA = "cvs.phase1.d92_da0.joint288_z160_fft96_rf32.v1"
_STATE_ARRAY_NAMES = (
    "log_diag_fp32",
    "coef1_qint8",
    "coef2_qint8",
    "scale1_fp16",
    "scale2_fp16",
    "intercept_fp16",
    "coef_fp32",
    "intercept_fp32",
)
_STATE_METADATA_NAMES = ("schema", "classes", "old_class_count", "covariance_policy")


class D92DA0Phase1OldStateError(ValueError):
    """Raised when a source-only old-state export or its seal drifts."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: Any, *, field: str) -> str:
    result = str(value).lower()
    if len(result) != 64 or any(char not in "0123456789abcdef" for char in result):
        raise D92DA0Phase1OldStateError(f"{field} must be a lowercase SHA256")
    return result


def _scalar_string(value: np.ndarray, *, field: str) -> str:
    array = np.asarray(value)
    if array.size != 1:
        raise D92DA0Phase1OldStateError(f"sealed state scalar drift: {field}")
    return str(array.reshape(-1)[0])


def _scalar_int(value: np.ndarray, *, field: str) -> int:
    array = np.asarray(value)
    if array.size != 1:
        raise D92DA0Phase1OldStateError(f"sealed state scalar drift: {field}")
    return int(array.reshape(-1)[0])


def _state_sha256(state: D42UnifiedShrinkageLDAState) -> str:
    digest = hashlib.sha256(b"cvs.phase2.d42.old_only_state.v1\0")
    digest.update(
        _canonical_bytes(
            {
                "schema": state.schema,
                "classes": list(state.classes),
                "old_class_count": int(state.old_class_count),
                "covariance_policy": state.covariance_policy,
            }
        )
    )
    for name in _STATE_ARRAY_NAMES:
        value = np.ascontiguousarray(np.asarray(getattr(state, name)))
        digest.update(value.tobytes())
    return digest.hexdigest()


def _state_arrays(state: D42UnifiedShrinkageLDAState) -> dict[str, np.ndarray]:
    if not isinstance(state, D42UnifiedShrinkageLDAState):
        raise D92DA0Phase1OldStateError("D92 DA0 requires a D42 predictive state")
    if int(state.old_class_count) != len(state.classes):
        raise D92DA0Phase1OldStateError("D92 DA0 state must be old-only")
    arrays = {
        "schema": np.asarray(state.schema),
        "classes": np.asarray(state.classes, dtype=str),
        "old_class_count": np.asarray(int(state.old_class_count), dtype=np.int64),
        "covariance_policy": np.asarray(state.covariance_policy),
    }
    arrays.update(
        {
            name: np.ascontiguousarray(np.asarray(getattr(state, name)))
            for name in _STATE_ARRAY_NAMES
        }
    )
    return arrays


def _state_from_archive(path: Path) -> D42UnifiedShrinkageLDAState:
    with np.load(path, allow_pickle=False) as archive:
        expected = set(_STATE_METADATA_NAMES) | set(_STATE_ARRAY_NAMES)
        if set(archive.files) != expected:
            raise D92DA0Phase1OldStateError("sealed state member allowlist drift")
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    try:
        return D42UnifiedShrinkageLDAState(
            schema=_scalar_string(arrays["schema"], field="schema"),
            classes=tuple(np.asarray(arrays["classes"]).astype(str).tolist()),
            old_class_count=_scalar_int(
                arrays["old_class_count"], field="old_class_count"
            ),
            log_diag_fp32=np.asarray(arrays["log_diag_fp32"], dtype=np.float32),
            coef1_qint8=np.asarray(arrays["coef1_qint8"], dtype=np.int8),
            coef2_qint8=np.asarray(arrays["coef2_qint8"], dtype=np.int8),
            scale1_fp16=np.asarray(arrays["scale1_fp16"], dtype=np.float16),
            scale2_fp16=np.asarray(arrays["scale2_fp16"], dtype=np.float16),
            intercept_fp16=np.asarray(arrays["intercept_fp16"], dtype=np.float16),
            coef_fp32=np.asarray(arrays["coef_fp32"], dtype=np.float32),
            intercept_fp32=np.asarray(arrays["intercept_fp32"], dtype=np.float32),
            covariance_policy=_scalar_string(
                arrays["covariance_policy"], field="covariance_policy"
            ),
        )
    except (D42UnifiedShrinkageLDAError, TypeError, ValueError) as exc:
        raise D92DA0Phase1OldStateError("sealed D42 state is invalid") from exc


def _validate_provenance(provenance: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "checkpoint_sha256",
        "runtime_sha256",
        "source_cache_set_manifest_sha256",
        "source_dataset_sha256",
        "class_handle_binding_sha256",
        "source_cache_member_sha256_by_scenario",
        "source_cache_physical_id_root_by_scenario",
    }
    if set(provenance) != required:
        raise D92DA0Phase1OldStateError("D92 DA0 provenance exact-schema drift")
    result = {
        name: _sha256(provenance[name], field=name)
        for name in required
        if name
        not in {
            "source_cache_member_sha256_by_scenario",
            "source_cache_physical_id_root_by_scenario",
        }
    }
    for name in (
        "source_cache_member_sha256_by_scenario",
        "source_cache_physical_id_root_by_scenario",
    ):
        mapping = dict(provenance[name])
        if tuple(mapping) != FORMAL_LEO_WEAK_SCENARIOS:
            raise D92DA0Phase1OldStateError(f"{name} scenario order drift")
        result[name] = {
            scenario: _sha256(mapping[scenario], field=f"{name}[{scenario}]")
            for scenario in FORMAL_LEO_WEAK_SCENARIOS
        }
    return result


def build_source_only_joint288(
    runtime: torch.nn.Module,
    source_leo_weak_iq: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Build D92's fixed joint288 view from source-only received IQ."""

    iq = np.asarray(source_leo_weak_iq, dtype=np.float32)
    if (
        iq.ndim != 3
        or iq.shape[1] != 2
        or len(iq) < 1
        or not np.isfinite(iq).all()
        or int(batch_size) < 1
    ):
        raise D92DA0Phase1OldStateError("source LEO_weak IQ input drift")
    zid160 = forward_zid160(runtime, iq, device=device, batch_size=int(batch_size))
    features = registered_feature(iq, zid160)
    if (
        features.dtype != np.float32
        or features.shape != (len(iq), FEATURE_DIM)
        or not np.isfinite(features).all()
        or not np.allclose(np.linalg.norm(features, axis=1), 1.0, atol=1.0e-5)
    ):
        raise D92DA0Phase1OldStateError("source joint288 construction drift")
    return np.ascontiguousarray(features, dtype=np.float32)


def fit_source_only_old_state(
    source_joint288: np.ndarray,
    source_labels: Sequence[str],
    old_class_registry: Sequence[str],
    *,
    seed: int,
    device: torch.device | str = "cpu",
) -> D42OldOnlyFitResult:
    """Fit the unmodified D42 pre-registration state from source-only joint288."""

    features = np.asarray(source_joint288)
    if (
        features.dtype != np.float32
        or features.ndim != 2
        or features.shape[1] != FEATURE_DIM
        or not np.isfinite(features).all()
    ):
        raise D92DA0Phase1OldStateError("source joint288 feature schema drift")
    return fit_d42_old_only(
        np.ascontiguousarray(features),
        source_labels,
        old_class_registry,
        seed=int(seed),
        device=device,
    )


def seal_source_only_old_state(
    output_dir: str | Path,
    *,
    states_by_scenario: Mapping[str, D42OldOnlyFitResult],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Persist D42 codec-only source states with an immutable source-only seal."""

    states = dict(states_by_scenario)
    if tuple(states) != FORMAL_LEO_WEAK_SCENARIOS:
        raise D92DA0Phase1OldStateError("D92 DA0 state scenario order drift")
    checked_provenance = _validate_provenance(provenance)
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise D92DA0Phase1OldStateError("D92 DA0 output must be absent or empty")
    output.mkdir(parents=True, exist_ok=True)
    states_dir = output / "states"
    states_dir.mkdir()
    members: list[dict[str, Any]] = []
    state_rows: dict[str, Any] = {}
    expected_classes: tuple[str, ...] | None = None
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        fit = states[scenario]
        if not isinstance(fit, D42OldOnlyFitResult):
            raise D92DA0Phase1OldStateError("D92 DA0 requires old-only fit results")
        state = fit.state
        if int(state.old_class_count) != len(state.classes):
            raise D92DA0Phase1OldStateError("D92 DA0 state unexpectedly contains new classes")
        if expected_classes is None:
            expected_classes = state.classes
        elif state.classes != expected_classes:
            raise D92DA0Phase1OldStateError("D92 DA0 old registry differs by scenario")
        destination = states_dir / f"{scenario}.npz"
        np.savez(destination, **_state_arrays(state))
        reloaded = _state_from_archive(destination)
        if _state_sha256(reloaded) != fit.state_sha256:
            raise D92DA0Phase1OldStateError("D92 DA0 persisted state byte drift")
        descriptor = {
            "relative_path": str(destination.relative_to(output)).replace("\\", "/"),
            "sha256": sha256_file(destination),
            "size_bytes": int(destination.stat().st_size),
        }
        members.append(descriptor)
        state_rows[scenario] = {
            **descriptor,
            "predictive_state_sha256": fit.state_sha256,
            "old_class_count": int(state.old_class_count),
            "old_k_shot": int(fit.old_k_shot),
            "covariance_policy": state.covariance_policy,
        }
    if expected_classes is None:
        raise D92DA0Phase1OldStateError("D92 DA0 requires source states")
    content_root = hashlib.sha256(_canonical_bytes(members)).hexdigest()
    manifest = {
        "schema": SCHEMA,
        "phase1_source_only": True,
        "old_state_bitwise_frozen": True,
        "joint288_transform_schema": JOINT288_TRANSFORM_SCHEMA,
        "joint288_transform": "registered_feature(z_id160,fft96,rf32)_from_same_received_iq",
        "joint288_dimension": FEATURE_DIM,
        "z160_centroid_padding_used": False,
        "phase2_da_adaptation_used": False,
        "checkpoint_sha256": checked_provenance["checkpoint_sha256"],
        "runtime_sha256": checked_provenance["runtime_sha256"],
        "source_cache_set_manifest_sha256": checked_provenance[
            "source_cache_set_manifest_sha256"
        ],
        "source_dataset_sha256": checked_provenance["source_dataset_sha256"],
        "old_registry_binding_sha256": checked_provenance[
            "class_handle_binding_sha256"
        ],
        "old_class_registry": list(expected_classes),
        "source_cache_member_sha256_by_scenario": checked_provenance[
            "source_cache_member_sha256_by_scenario"
        ],
        "source_cache_physical_id_root_by_scenario": checked_provenance[
            "source_cache_physical_id_root_by_scenario"
        ],
        "state_members_by_scenario": state_rows,
        "members": members,
        "content_root_sha256": content_root,
        "target_receiver_old_support_opened": False,
        "target_receiver_new_support_opened": False,
        "target_query_opened": False,
        "target_package_somph_head_opened": False,
        "query_truth_opened": False,
        "phase2_source_runtime_access_required": False,
        "source_iq_or_feature_rows_persisted": False,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    loaded = load_sealed_source_only_old_state(output)
    if loaded["manifest"]["content_root_sha256"] != content_root:
        raise D92DA0Phase1OldStateError("D92 DA0 manifest readback drift")
    return {
        "schema": SCHEMA,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "content_root_sha256": content_root,
        "state_members_by_scenario": state_rows,
    }


def load_sealed_source_only_old_state(output_dir: str | Path) -> dict[str, Any]:
    """Verify and materialize only the sealed D42 old states for a later scorer."""

    output = Path(output_dir)
    manifest_path = output / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise D92DA0Phase1OldStateError("D92 DA0 manifest is unreadable") from exc
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA:
        raise D92DA0Phase1OldStateError("D92 DA0 manifest schema drift")
    expected_false = (
        "target_receiver_old_support_opened",
        "target_receiver_new_support_opened",
        "target_query_opened",
        "target_package_somph_head_opened",
        "query_truth_opened",
    )
    if any(manifest.get(name) is not False for name in expected_false):
        raise D92DA0Phase1OldStateError("D92 DA0 target-unopened receipt drift")
    if (
        manifest.get("joint288_transform_schema") != JOINT288_TRANSFORM_SCHEMA
        or int(manifest.get("joint288_dimension", -1)) != FEATURE_DIM
        or manifest.get("old_state_bitwise_frozen") is not True
        or manifest.get("source_iq_or_feature_rows_persisted") is not False
    ):
        raise D92DA0Phase1OldStateError("D92 DA0 transform or persistence drift")
    members = manifest.get("members")
    by_scenario = manifest.get("state_members_by_scenario")
    if not isinstance(members, list) or not isinstance(by_scenario, dict):
        raise D92DA0Phase1OldStateError("D92 DA0 state member manifest drift")
    if tuple(by_scenario) != FORMAL_LEO_WEAK_SCENARIOS:
        raise D92DA0Phase1OldStateError("D92 DA0 manifest scenario order drift")
    if hashlib.sha256(_canonical_bytes(members)).hexdigest() != manifest.get(
        "content_root_sha256"
    ):
        raise D92DA0Phase1OldStateError("D92 DA0 content root drift")
    states: dict[str, D42UnifiedShrinkageLDAState] = {}
    expected_classes = tuple(str(value) for value in manifest.get("old_class_registry", []))
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        descriptor = by_scenario[scenario]
        if not isinstance(descriptor, dict):
            raise D92DA0Phase1OldStateError("D92 DA0 state descriptor drift")
        relative = Path(str(descriptor.get("relative_path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise D92DA0Phase1OldStateError("D92 DA0 state path is not confined")
        state_path = output / relative
        if (
            not state_path.is_file()
            or sha256_file(state_path) != descriptor.get("sha256")
            or int(state_path.stat().st_size) != int(descriptor.get("size_bytes", -1))
        ):
            raise D92DA0Phase1OldStateError("D92 DA0 state member digest drift")
        state = _state_from_archive(state_path)
        if (
            state.classes != expected_classes
            or int(state.old_class_count) != len(expected_classes)
            or _state_sha256(state) != descriptor.get("predictive_state_sha256")
        ):
            raise D92DA0Phase1OldStateError("D92 DA0 predictive state closure drift")
        states[scenario] = state
    return {"manifest": manifest, "states_by_scenario": states}


__all__ = [
    "D92DA0Phase1OldStateError",
    "JOINT288_TRANSFORM_SCHEMA",
    "SCHEMA",
    "build_source_only_joint288",
    "fit_source_only_old_state",
    "load_sealed_source_only_old_state",
    "seal_source_only_old_state",
]
