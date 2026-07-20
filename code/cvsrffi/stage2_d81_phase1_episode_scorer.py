"""Deterministic Phase1-only D81 episode scorer for receiver LODO.

The scorer fits the same D81-before head from labeled support and returns logits
for caller-supplied rows.  It has no query labels, receiver identity, target
state, clean sample, or mutable adaptation cache input.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import json
import platform
from pathlib import Path
import threading
from types import MappingProxyType
from typing import Any, Sequence

import numpy as np
import scipy
import torch

from cvsrffi import stage2_d42_unified_shrinkage_lda as d42
from cvsrffi.phase2_candidate_capsule import BASE_CHECKPOINT_SHA256


SCHEMA = "cvs.phase1.d81.episode_scorer.v1"
FEATURE_DIM = 288
DEFAULT_METRIC_SEED = 713101
ALLOWED_SKLEARN_RUNTIME_VERSIONS = ("1.7.0", "1.7.2")
_FIT_LOCK = threading.RLock()

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEPENDENCY_PATHS = {
    "scorer": Path(__file__).resolve(),
    "d42_core": _REPO_ROOT / "code/cvsrffi/stage2_d42_unified_shrinkage_lda.py",
    "d81_core": _REPO_ROOT / "code/cvsrffi/stage2_d81_ground_nuisance_cauchy_center.py",
    "d81_probe": _REPO_ROOT / "code/scripts/probe_d81_ground_nuisance_cauchy_center.py",
    "d80_core": _REPO_ROOT / "code/cvsrffi/stage2_d80_ground_commonmode_denoiser.py",
    "d80_probe": _REPO_ROOT / "code/scripts/probe_d80_ground_commonmode_covariance_denoiser.py",
    "d66_probe": _REPO_ROOT / "code/scripts/probe_d66_ground_domain_reliability_residual.py",
    "d62_probe": _REPO_ROOT / "code/scripts/probe_d62_crossfitted_fisher_row_splice.py",
    "d61_probe": _REPO_ROOT / "code/scripts/probe_d61_identity_primary_fisher_residual.py",
    "d46_probe": _REPO_ROOT / "code/scripts/probe_d46_classwise_loo_reliability_fusion.py",
    "d45_probe": _REPO_ROOT / "code/scripts/probe_d45_inner_loo_reliability_fusion.py",
    "d44_probe": _REPO_ROOT / "code/scripts/probe_d44_full_block_rms_fusion.py",
    "d43_probe": _REPO_ROOT / "code/scripts/probe_d43_structured_covariance.py",
}


class D81Phase1EpisodeScorerError(ValueError):
    """Raised when D81 Phase1 scorer provenance or episode input drifts."""


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise D81Phase1EpisodeScorerError(
        f"D81 ground audit contains noncanonical value: {type(value).__name__}"
    )


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, MappingProxyType):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return copy.deepcopy(value)


def _dependency_hashes() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name, path in _DEPENDENCY_PATHS.items():
        if not path.is_file():
            raise D81Phase1EpisodeScorerError(
                f"D81 dependency source is missing: {name}"
            )
        hashes[name] = _sha256_file(path)
    return hashes


def _require_sha(value: str, name: str) -> str:
    normalized = str(value).lower()
    if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
        raise D81Phase1EpisodeScorerError(f"{name} must be lowercase SHA256")
    return normalized


def raw_concat_to_d81_registered_feature(value: np.ndarray) -> np.ndarray:
    """Recover D81's historical registered-feature geometry from raw concat288."""

    rows = np.asarray(value, dtype=np.float32)
    if (
        rows.ndim != 2
        or rows.shape[1] != FEATURE_DIM
        or len(rows) == 0
        or not np.isfinite(rows).all()
    ):
        raise D81Phase1EpisodeScorerError(
            "D81 raw concat must be finite float32 [N,288]"
        )
    primary = rows[:, :160].astype(np.float64)
    auxiliary = rows[:, 160:].astype(np.float64)
    primary_norm = np.linalg.norm(primary, axis=1, keepdims=True)
    auxiliary_norm = np.linalg.norm(auxiliary, axis=1, keepdims=True)
    if (
        np.any(primary_norm <= 0.0)
        or np.any(auxiliary_norm <= 0.0)
        or not np.isfinite(primary_norm).all()
        or not np.isfinite(auxiliary_norm).all()
    ):
        raise D81Phase1EpisodeScorerError("D81 registered-feature block is degenerate")
    combined = np.concatenate(
        [primary / primary_norm, 4.0 * auxiliary / auxiliary_norm], axis=1
    )
    combined_norm = np.linalg.norm(combined, axis=1, keepdims=True)
    if np.any(combined_norm <= 0.0) or not np.isfinite(combined_norm).all():
        raise D81Phase1EpisodeScorerError("D81 registered-feature concat is degenerate")
    return np.ascontiguousarray(combined / combined_norm, dtype=np.float32)


@dataclass(frozen=True)
class D81Phase1EpisodeScorer:
    nuisance_basis_fp64: np.ndarray
    spectral_weights_fp64: np.ndarray
    ground_manifest_sha256: str
    ground_component_npz_sha256: str
    ground_audit: dict[str, Any]
    device: str = "cpu"
    metric_seed: int = DEFAULT_METRIC_SEED
    phase1_checkpoint_sha256: str = BASE_CHECKPOINT_SHA256
    sklearn_runtime_version: str = str(d42.sklearn.__version__)
    python_runtime_version: str = platform.python_version()
    numpy_runtime_version: str = str(np.__version__)
    scipy_runtime_version: str = str(scipy.__version__)
    torch_runtime_version: str = str(torch.__version__)

    def __post_init__(self) -> None:
        basis = np.asarray(self.nuisance_basis_fp64, dtype=np.float64)
        weights = np.asarray(self.spectral_weights_fp64, dtype=np.float64)
        if (
            basis.ndim != 2
            or basis.shape[0] != 160
            or not 1 <= basis.shape[1] <= 160
            or weights.shape != (basis.shape[1],)
            or not np.isfinite(basis).all()
            or not np.isfinite(weights).all()
            or np.any(weights <= 0.0)
            or not np.isclose(np.sum(weights), 1.0, atol=1e-12)
            or not np.allclose(basis.T @ basis, np.eye(basis.shape[1]), atol=2e-8)
        ):
            raise D81Phase1EpisodeScorerError("D81 nuisance spectrum drift")
        _require_sha(self.ground_manifest_sha256, "ground manifest")
        _require_sha(self.ground_component_npz_sha256, "ground component NPZ")
        if not isinstance(self.ground_audit, dict) or not self.ground_audit:
            raise D81Phase1EpisodeScorerError("D81 ground audit is missing")
        checkpoint = _require_sha(
            self.phase1_checkpoint_sha256, "Phase1 checkpoint"
        )
        if checkpoint != BASE_CHECKPOINT_SHA256:
            raise D81Phase1EpisodeScorerError("D81 Phase1 checkpoint identity drift")
        sklearn_version = str(self.sklearn_runtime_version)
        if (
            sklearn_version not in ALLOWED_SKLEARN_RUNTIME_VERSIONS
            or sklearn_version != str(d42.sklearn.__version__)
        ):
            raise D81Phase1EpisodeScorerError("D81 sklearn runtime version drift")
        runtime_versions = {
            "python_runtime_version": (self.python_runtime_version, platform.python_version()),
            "numpy_runtime_version": (self.numpy_runtime_version, str(np.__version__)),
            "scipy_runtime_version": (self.scipy_runtime_version, str(scipy.__version__)),
            "torch_runtime_version": (self.torch_runtime_version, str(torch.__version__)),
        }
        if any(str(sealed) != current for sealed, current in runtime_versions.values()):
            raise D81Phase1EpisodeScorerError("D81 numerical runtime version drift")
        seed = int(self.metric_seed)
        if seed < 0 or seed > 0x7FFFFFFF:
            raise D81Phase1EpisodeScorerError("D81 metric seed is out of range")
        frozen_audit = _freeze(_json_safe(copy.deepcopy(self.ground_audit)))
        object.__setattr__(self, "nuisance_basis_fp64", basis.copy())
        object.__setattr__(self, "spectral_weights_fp64", weights.copy())
        object.__setattr__(self, "ground_audit", frozen_audit)
        object.__setattr__(self, "metric_seed", seed)
        object.__setattr__(self, "phase1_checkpoint_sha256", checkpoint)
        object.__setattr__(self, "sklearn_runtime_version", sklearn_version)
        for name, (sealed, _current) in runtime_versions.items():
            object.__setattr__(self, name, str(sealed))
        self.nuisance_basis_fp64.setflags(write=False)
        self.spectral_weights_fp64.setflags(write=False)

    @classmethod
    def from_component(
        cls,
        component_dir: str | Path,
        manifest_sha256: str,
        *,
        device: str = "cpu",
        metric_seed: int = DEFAULT_METRIC_SEED,
        phase1_checkpoint_sha256: str = BASE_CHECKPOINT_SHA256,
    ) -> "D81Phase1EpisodeScorer":
        from scripts import probe_d81_ground_nuisance_cauchy_center as probe

        root = Path(component_dir).resolve()
        expected_manifest = _require_sha(manifest_sha256, "ground manifest")
        manifest_path = root / "manifest.json"
        if _sha256_file(manifest_path) != expected_manifest:
            raise D81Phase1EpisodeScorerError("D81 ground manifest SHA mismatch")
        basis, weights, audit = probe.load_ground_basis(root, expected_manifest, FEATURE_DIM)
        component_sha = _require_sha(
            str(audit.get("component_npz_sha256", "")), "ground component NPZ"
        )
        return cls(
            nuisance_basis_fp64=basis,
            spectral_weights_fp64=weights,
            ground_manifest_sha256=expected_manifest,
            ground_component_npz_sha256=component_sha,
            ground_audit=dict(audit),
            device=str(device),
            metric_seed=int(metric_seed),
            phase1_checkpoint_sha256=phase1_checkpoint_sha256,
        )

    @property
    def scorer_id(self) -> str:
        return _canonical_sha256(self.receipt)

    @property
    def receipt(self) -> dict[str, Any]:
        dependencies = _dependency_hashes()
        return {
            "schema": SCHEMA,
            "formula": "D81_before_support_fitted_D62_D42_int8",
            "ground_manifest_sha256": self.ground_manifest_sha256,
            "ground_component_npz_sha256": self.ground_component_npz_sha256,
            "basis_sha256": _sha256_array(self.nuisance_basis_fp64),
            "spectral_weights_sha256": _sha256_array(self.spectral_weights_fp64),
            "ground_audit_sha256": _canonical_sha256(_thaw(self.ground_audit)),
            "phase1_checkpoint_sha256": self.phase1_checkpoint_sha256,
            "metric_seed": int(self.metric_seed),
            "sklearn_runtime_version": self.sklearn_runtime_version,
            "python_runtime_version": self.python_runtime_version,
            "numpy_runtime_version": self.numpy_runtime_version,
            "scipy_runtime_version": self.scipy_runtime_version,
            "torch_runtime_version": self.torch_runtime_version,
            "dependency_code_sha256": dependencies,
            "dependency_closure_sha256": _canonical_sha256(dependencies),
            "device": str(self.device),
            "query_labels_input": False,
            "receiver_or_role_input": False,
            "mutable_fit_cache": False,
        }

    def __call__(
        self,
        support_features: np.ndarray,
        support_labels: np.ndarray,
        query_features: np.ndarray,
        class_ids: np.ndarray,
    ) -> np.ndarray:
        current_versions = (
            str(d42.sklearn.__version__),
            platform.python_version(),
            str(np.__version__),
            str(scipy.__version__),
            str(torch.__version__),
        )
        sealed_versions = (
            self.sklearn_runtime_version,
            self.python_runtime_version,
            self.numpy_runtime_version,
            self.scipy_runtime_version,
            self.torch_runtime_version,
        )
        if current_versions != sealed_versions:
            raise D81Phase1EpisodeScorerError("D81 numerical runtime changed after lock")
        support = np.array(support_features, dtype=np.float32, copy=True)
        query = np.array(query_features, dtype=np.float32, copy=True)
        labels = np.array(support_labels, copy=True)
        class_array = np.array(class_ids, copy=True)
        classes = tuple(str(value) for value in class_array.tolist())
        if (
            support.ndim != 2
            or support.shape[1] != FEATURE_DIM
            or len(support) == 0
            or query.ndim != 2
            or query.shape[1] != FEATURE_DIM
            or len(query) == 0
            or labels.shape != (len(support),)
            or class_array.ndim != 1
            or len(classes) < 2
            or len(set(classes)) != len(classes)
            or not np.isfinite(support).all()
            or not np.isfinite(query).all()
        ):
            raise D81Phase1EpisodeScorerError("D81 episode feature/class shape drift")
        string_labels = np.asarray([str(value) for value in labels.tolist()])
        if set(string_labels.tolist()) != set(classes):
            raise D81Phase1EpisodeScorerError("D81 episode class closure drift")
        counts = np.asarray([np.sum(string_labels == value) for value in classes], dtype=np.int64)
        if np.any(counts <= 0) or len(np.unique(counts)) != 1:
            raise D81Phase1EpisodeScorerError("D81 episode requires balanced K-shot support")
        lookup = {value: index for index, value in enumerate(classes)}
        targets = np.asarray([lookup[value] for value in string_labels], dtype=np.int64)
        d81_support = raw_concat_to_d81_registered_feature(support)
        d81_query = raw_concat_to_d81_registered_feature(query)
        from scripts import probe_d81_ground_nuisance_cauchy_center as probe
        with _FIT_LOCK:
            d81_fit, _call_records, _transform_records = probe.build_d81_fit(
                d42,
                self.nuisance_basis_fp64,
                self.spectral_weights_fp64,
                _thaw(self.ground_audit),
            )
            runtime_device = torch.device(self.device)
            log_diag, trace, _resource = d42._fit_old_only_b3_metric(
                d81_support,
                targets,
                len(classes),
                seed=self.metric_seed,
                device=runtime_device,
            )
            if len(trace) != d42.METRIC_EPOCHS:
                raise D81Phase1EpisodeScorerError(
                    "D81 episode metric lifecycle drift"
                )
            transformed = d42._transform(d81_support, log_diag)
            coefficient, intercept, lda_audit = d81_fit(
                transformed,
                targets,
                len(classes),
                int(counts[0]),
            )
            state, _quantization = d42._compile_state(
                classes,
                len(classes),
                log_diag,
                coefficient,
                intercept,
                str(lda_audit["covariance_policy"]),
                precision="int8",
            )
        scores = d42.score_d42_unified_shrinkage_lda(state, d81_query)
        if scores.shape != (len(query), len(classes)) or not np.isfinite(scores).all():
            raise D81Phase1EpisodeScorerError("D81 episode scorer output drift")
        return np.asarray(scores, dtype=np.float32)


__all__ = [
    "D81Phase1EpisodeScorer",
    "D81Phase1EpisodeScorerError",
    "FEATURE_DIM",
    "SCHEMA",
    "raw_concat_to_d81_registered_feature",
]
