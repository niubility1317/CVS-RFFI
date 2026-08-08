"""Immutable Phase1 dual-readout bundle and truth-free local evidence runtime."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

import numpy as np
import torch


BUNDLE_SCHEMA = "cvs.phase1.dualreadout_openworld_bundle.v2"
CALIBRATION_SCHEMA = "cvs.phase1.dualreadout_source_calibration.v2"
FORMULA_ID = "robust_class_angular_js_continuous_v2"
MANIFEST_NAME = "manifest.json"
MEMBER_PATHS = (
    "runtimes/angular.ts",
    "runtimes/robust.ts",
    "calibration/calibration.npz",
    "calibration/receipt.json",
)


class DualReadoutBundleError(ValueError):
    pass


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _content_root(members: Sequence[Mapping[str, Any]]) -> str:
    projection = [
        {"relative_path": row["relative_path"], "sha256": row["sha256"], "size_bytes": row["size_bytes"]}
        for row in sorted(members, key=lambda item: str(item["relative_path"]))
    ]
    return hashlib.sha256(canonical_json(projection).encode("utf-8")).hexdigest()


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise DualReadoutBundleError("feature arrays must be finite rank-2 matrices")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise DualReadoutBundleError("feature arrays contain zero-norm rows")
    return array / norms


def _softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2 or not np.isfinite(values).all():
        raise DualReadoutBundleError("logits must be a finite rank-2 matrix with C>=2")
    shifted = values - values.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _js(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    p = np.clip(_softmax(left), 1e-12, 1.0)
    q = np.clip(_softmax(right), 1e-12, 1.0)
    middle = np.clip(0.5 * (p + q), 1e-12, 1.0)
    return 0.5 * ((p * (np.log(p) - np.log(middle))).sum(1) + (q * (np.log(q) - np.log(middle))).sum(1))


def _centers_and_radii(
    features: np.ndarray,
    groups: Sequence[str],
    registry: Sequence[str],
    *,
    radius_quantile: float,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    normalized = _normalize_rows(features)
    centers: list[np.ndarray] = []
    radii: list[float] = []
    counts: list[int] = []
    group_array = np.asarray([str(value) for value in groups])
    for handle in registry:
        selected = normalized[group_array == str(handle)]
        if len(selected) < 2:
            raise DualReadoutBundleError(f"group {handle!r} has fewer than two source physical samples")
        center = selected.mean(axis=0)
        norm = float(np.linalg.norm(center))
        if norm <= 1e-12:
            raise DualReadoutBundleError(f"group {handle!r} has a degenerate center")
        center = center / norm
        angles = np.arccos(np.clip(selected @ center, -1.0, 1.0))
        radius = max(float(np.quantile(angles, radius_quantile)), 1e-6)
        centers.append(center)
        radii.append(radius)
        counts.append(len(selected))
    return np.asarray(centers, dtype=np.float32), np.asarray(radii, dtype=np.float32), counts


def _signals(
    calibration: Mapping[str, np.ndarray],
    *,
    angular_logits: np.ndarray,
    robust_z_id: np.ndarray,
    robust_z_dom: np.ndarray,
    robust_logits: np.ndarray,
) -> dict[str, np.ndarray]:
    class_handles = np.asarray(calibration["class_handles"]).astype(str)
    class_centers = _normalize_rows(calibration["class_centers"])
    domain_centers = _normalize_rows(calibration["domain_centers"])
    zid = _normalize_rows(robust_z_id)
    zdom = _normalize_rows(robust_z_dom)
    robust_prob = _softmax(robust_logits)
    if robust_prob.shape[1] != len(class_handles) or np.asarray(angular_logits).shape != np.asarray(robust_logits).shape:
        raise DualReadoutBundleError("runtime class/logit shape differs from calibration")
    class_radius = np.asarray(calibration["class_radii"], dtype=np.float64).reshape(1, -1)
    domain_radius = np.asarray(calibration["domain_radii"], dtype=np.float64).reshape(1, -1)
    class_distance = np.arccos(np.clip(zid @ class_centers.T, -1.0, 1.0)) / np.maximum(class_radius, 1e-8)
    domain_distance = np.arccos(np.clip(zdom @ domain_centers.T, -1.0, 1.0)) / np.maximum(domain_radius, 1e-8)
    class_affinity = np.exp(-0.5 * np.square(class_distance))
    domain_affinity = np.exp(-0.5 * np.square(domain_distance)).max(axis=1)
    js_scale = max(float(np.asarray(calibration["js_scale"]).reshape(())), 1e-8)
    js_value = _js(angular_logits, robust_logits)
    js_normalized = np.clip(js_value / js_scale, 0.0, 1.0)
    geometric_unknown = 1.0 - class_affinity.max(axis=1)
    e_unknown = 1.0 - (1.0 - geometric_unknown) * (1.0 - js_normalized)
    entropy = -(np.clip(robust_prob, 1e-12, 1.0) * np.log(np.clip(robust_prob, 1e-12, 1.0))).sum(axis=1)
    entropy_quality = np.clip(1.0 - entropy / math.log(robust_prob.shape[1]), 0.0, 1.0)
    quality = np.sqrt(np.clip(domain_affinity * entropy_quality, 0.0, 1.0))
    ordered = np.sort(robust_prob, axis=1)
    margin = ordered[:, -1] - ordered[:, -2]
    p_local = np.concatenate(
        [robust_prob * (1.0 - e_unknown[:, None]), e_unknown[:, None]], axis=1
    )
    p_local = p_local / p_local.sum(axis=1, keepdims=True)
    return {
        "d_class": class_distance.astype(np.float32),
        "e_unknown": e_unknown.astype(np.float32),
        "q": quality.astype(np.float32),
        "p_local": p_local.astype(np.float32),
        "robust_margin": margin.astype(np.float32),
        "js_disagreement": js_value.astype(np.float32),
        "robust_pred": robust_prob.argmax(axis=1).astype(np.int64),
    }


def fit_source_calibration(
    *,
    angular_logits: np.ndarray,
    robust_z_id: np.ndarray,
    robust_z_dom: np.ndarray,
    robust_logits: np.ndarray,
    tx_ids: Sequence[str],
    roles: Sequence[str],
    rx_ids: Sequence[str],
    day_ids: Sequence[str],
    physical_ids: Sequence[str],
    class_handles: Sequence[str],
    calibration_roles: Sequence[str] = ("source",),
    radius_quantile: float = 0.95,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    count = len(tx_ids)
    fields = (roles, rx_ids, day_ids, physical_ids)
    if any(len(field) != count for field in fields):
        raise DualReadoutBundleError("calibration metadata lengths differ")
    handles = tuple(str(value) for value in class_handles)
    if len(handles) < 2 or len(set(handles)) != len(handles):
        raise DualReadoutBundleError("class_handles must be unique with C>=2")
    role_allow = {str(value) for value in calibration_roles}
    mask = np.asarray([str(role) in role_allow and str(tx) in handles for role, tx in zip(roles, tx_ids)])
    if not bool(mask.any()):
        raise DualReadoutBundleError("no source-known calibration rows")
    selected_physical = [str(value) for value, keep in zip(physical_ids, mask) if keep]
    if len(set(selected_physical)) != len(selected_physical) or any(not value for value in selected_physical):
        raise DualReadoutBundleError("source calibration requires unique non-empty physical_ids")
    a_logits = np.asarray(angular_logits)[mask]
    r_logits = np.asarray(robust_logits)[mask]
    zid = np.asarray(robust_z_id)[mask]
    zdom = np.asarray(robust_z_dom)[mask]
    tx = np.asarray([str(value) for value in tx_ids])[mask]
    domain = np.asarray([f"rx:{rx}|day:{day}" for rx, day in zip(rx_ids, day_ids)])[mask]
    if a_logits.shape != r_logits.shape or a_logits.shape[0] != len(tx) or a_logits.shape[1] != len(handles):
        raise DualReadoutBundleError("calibration tensor shape mismatch")
    class_centers, class_radii, class_counts = _centers_and_radii(
        zid, tx, handles, radius_quantile=radius_quantile
    )
    domain_handles = sorted(set(domain.tolist()))
    domain_centers, domain_radii, domain_counts = _centers_and_radii(
        zdom, domain, domain_handles, radius_quantile=radius_quantile
    )
    class_index = {handle: index for index, handle in enumerate(handles)}
    truth = np.asarray([class_index[value] for value in tx], dtype=np.int64)
    joint_correct = (np.asarray(a_logits).argmax(1) == truth) & (np.asarray(r_logits).argmax(1) == truth)
    if int(joint_correct.sum()) < max(8, len(handles) * 2):
        raise DualReadoutBundleError("too few jointly correct source rows for calibration")
    js_scale = max(float(np.quantile(_js(a_logits[joint_correct], r_logits[joint_correct]), 0.99)), 1e-8)
    provisional = {
        "class_handles": np.asarray(handles),
        "class_centers": class_centers,
        "class_radii": class_radii,
        "domain_handles": np.asarray(domain_handles),
        "domain_centers": domain_centers,
        "domain_radii": domain_radii,
        "js_scale": np.asarray(js_scale, dtype=np.float32),
    }
    signals = _signals(
        provisional,
        angular_logits=a_logits,
        robust_z_id=zid,
        robust_z_dom=zdom,
        robust_logits=r_logits,
    )
    selected = joint_correct
    tau_q = float(np.quantile(signals["q"][selected], 0.01))
    tau_margin = float(np.quantile(signals["robust_margin"][selected], 0.01))
    tau_unknown_low = float(np.quantile(signals["e_unknown"][selected], 0.99))
    tau_unknown_high = min(
        1.0,
        max(float(np.quantile(signals["e_unknown"][selected], 0.999)), tau_unknown_low + 0.05),
    )
    calibration = {
        **provisional,
        "tau_q": np.asarray(tau_q, dtype=np.float32),
        "tau_margin": np.asarray(tau_margin, dtype=np.float32),
        "tau_unknown_low": np.asarray(tau_unknown_low, dtype=np.float32),
        "tau_unknown_high": np.asarray(tau_unknown_high, dtype=np.float32),
    }
    physical_digest = hashlib.sha256(
        canonical_json(sorted(selected_physical)).encode("utf-8")
    ).hexdigest()
    receipt = {
        "schema": CALIBRATION_SCHEMA,
        "formula_id": FORMULA_ID,
        "threshold_scope": "source_joint_correct_only_no_proxy_or_target_tuning",
        "class_handles": list(handles),
        "domain_handles": domain_handles,
        "source_known_count": int(mask.sum()),
        "source_joint_correct_count": int(joint_correct.sum()),
        "class_counts": class_counts,
        "domain_counts": domain_counts,
        "source_physical_id_set_sha256": physical_digest,
        "radius_quantile": radius_quantile,
        "js_quantile": 0.99,
        "threshold_quantiles": {"q": 0.01, "margin": 0.01, "unknown_low": 0.99, "unknown_high": 0.999},
        "excluded_non_calibration_row_count": int(count - mask.sum()),
        "raw_iq_persisted": False,
        "physical_ids_persisted": False,
    }
    return calibration, receipt


def build_bundle(
    output_dir: str | Path,
    *,
    angular_runtime: str | Path,
    robust_runtime: str | Path,
    calibration: Mapping[str, np.ndarray],
    calibration_receipt: Mapping[str, Any],
    angular_checkpoint_sha256: str,
    robust_checkpoint_sha256: str,
) -> dict[str, Any]:
    root = Path(output_dir)
    if root.exists():
        raise FileExistsError("refusing to reuse Phase1 bundle output directory")
    for path in (angular_runtime, robust_runtime):
        source = Path(path)
        if source.suffix.lower() == ".pth" or not source.is_file():
            raise DualReadoutBundleError("bundle runtime must be a regular non-checkpoint file")
        torch.jit.load(str(source), map_location="cpu")
    root.mkdir(parents=True)
    destinations = {
        "runtimes/angular.ts": Path(angular_runtime),
        "runtimes/robust.ts": Path(robust_runtime),
    }
    for relative, source in destinations.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    calibration_path = root / "calibration/calibration.npz"
    calibration_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(calibration_path, **{key: np.asarray(value) for key, value in calibration.items()})
    receipt = dict(calibration_receipt)
    if receipt.get("schema") != CALIBRATION_SCHEMA or receipt.get("formula_id") != FORMULA_ID:
        raise DualReadoutBundleError("calibration receipt schema/formula drift")
    receipt_path = root / "calibration/receipt.json"
    receipt_path.write_text(canonical_json(receipt) + "\n", encoding="utf-8")
    members = []
    for relative in MEMBER_PATHS:
        path = root / relative
        members.append({"relative_path": relative, "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    manifest = {
        "schema": BUNDLE_SCHEMA,
        "artifact_stage": "phase1_ground_openworld_ready_dualreadout_bundle",
        "evidence_state": "TECHNICAL_BUNDLE_NOT_PERFORMANCE_PROMOTED",
        "formula_id": FORMULA_ID,
        "angular_checkpoint_sha256": str(angular_checkpoint_sha256).lower(),
        "robust_checkpoint_sha256": str(robust_checkpoint_sha256).lower(),
        "class_handles": [str(value) for value in np.asarray(calibration["class_handles"]).tolist()],
        "members": members,
        "content_root_sha256": _content_root(members),
        "raw_training_checkpoint_included": False,
        "raw_iq_included": False,
        "role_or_truth_included": False,
        "phase3_local_evidence_supported": True,
    }
    (root / MANIFEST_NAME).write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    return manifest


@dataclass
class LoadedDualReadoutBundle:
    root: Path
    manifest: dict[str, Any]
    calibration: dict[str, np.ndarray]
    angular_runtime: torch.jit.ScriptModule
    robust_runtime: torch.jit.ScriptModule

    @torch.no_grad()
    def forward_iq(self, rows: torch.Tensor) -> dict[str, np.ndarray]:
        if rows.ndim != 3 or rows.shape[1] != 2 or not torch.isfinite(rows).all():
            raise DualReadoutBundleError("received IQ must be finite [B,2,L]")
        angular = self.angular_runtime(rows)
        robust = self.robust_runtime(rows)
        if not isinstance(angular, (tuple, list)) or not isinstance(robust, (tuple, list)) or len(angular) != 3 or len(robust) != 3:
            raise DualReadoutBundleError("runtime output must be (z_id,z_dom,tx_logits)")
        return self.evaluate_arrays(
            angular_logits=angular[2].detach().cpu().numpy(),
            robust_z_id=robust[0].detach().cpu().numpy(),
            robust_z_dom=robust[1].detach().cpu().numpy(),
            robust_logits=robust[2].detach().cpu().numpy(),
        )

    def evaluate_arrays(self, **arrays: np.ndarray) -> dict[str, np.ndarray]:
        signals = _signals(self.calibration, **arrays)
        handles = list(self.manifest["class_handles"])
        decisions: list[str] = []
        labels: list[str | None] = []
        reasons: list[str] = []
        tq = float(np.asarray(self.calibration["tau_q"]).reshape(()))
        tm = float(np.asarray(self.calibration["tau_margin"]).reshape(()))
        tul = float(np.asarray(self.calibration["tau_unknown_low"]).reshape(()))
        tuh = float(np.asarray(self.calibration["tau_unknown_high"]).reshape(()))
        for index in range(len(signals["q"])):
            if signals["q"][index] >= tq and signals["robust_margin"][index] >= tm and signals["e_unknown"][index] <= tul:
                decisions.append("registered")
                labels.append(handles[int(signals["robust_pred"][index])])
                reasons.append("P1_SOURCE_CALIBRATED_REGISTERED")
            elif signals["q"][index] >= tq and signals["e_unknown"][index] >= tuh:
                decisions.append("unknown")
                labels.append(None)
                reasons.append("P1_SOURCE_CALIBRATED_UNKNOWN")
            else:
                decisions.append("defer")
                labels.append(None)
                reasons.append("P1_INSUFFICIENT_LOCAL_EVIDENCE")
        return {**signals, "local_decision": np.asarray(decisions), "local_label": np.asarray(labels, dtype=object), "reason_code": np.asarray(reasons)}


def load_bundle(
    root: str | Path,
    *,
    expected_content_root_sha256: str,
) -> LoadedDualReadoutBundle:
    package = Path(root)
    actual = sorted(str(path.relative_to(package)).replace("\\", "/") for path in package.rglob("*") if path.is_file())
    expected = sorted([MANIFEST_NAME, *MEMBER_PATHS])
    if actual != expected:
        raise DualReadoutBundleError("bundle exact member allowlist mismatch")
    manifest = json.loads((package / MANIFEST_NAME).read_text(encoding="utf-8"))
    if manifest.get("schema") != BUNDLE_SCHEMA or manifest.get("formula_id") != FORMULA_ID:
        raise DualReadoutBundleError("bundle manifest schema/formula drift")
    expected_manifest_keys = {
        "schema", "artifact_stage", "evidence_state", "formula_id",
        "angular_checkpoint_sha256", "robust_checkpoint_sha256", "class_handles",
        "members", "content_root_sha256", "raw_training_checkpoint_included",
        "raw_iq_included", "role_or_truth_included", "phase3_local_evidence_supported",
    }
    if set(manifest) != expected_manifest_keys:
        raise DualReadoutBundleError("bundle manifest field allowlist mismatch")
    if (
        manifest.get("raw_training_checkpoint_included") is not False
        or manifest.get("raw_iq_included") is not False
        or manifest.get("role_or_truth_included") is not False
        or manifest.get("phase3_local_evidence_supported") is not True
    ):
        raise DualReadoutBundleError("bundle safety/capability flags drift")
    members = manifest.get("members")
    if not isinstance(members, list) or [row.get("relative_path") for row in members] != list(MEMBER_PATHS):
        raise DualReadoutBundleError("bundle member descriptor order drift")
    for row in members:
        path = package / row["relative_path"]
        if sha256_file(path) != row["sha256"] or path.stat().st_size != int(row["size_bytes"]):
            raise DualReadoutBundleError(f"bundle member hash/size mismatch: {row['relative_path']}")
    if _content_root(members) != manifest.get("content_root_sha256"):
        raise DualReadoutBundleError("bundle content root mismatch")
    if str(expected_content_root_sha256).lower() != manifest.get("content_root_sha256"):
        raise DualReadoutBundleError("bundle external content root binding mismatch")
    with np.load(package / "calibration/calibration.npz", allow_pickle=False) as archive:
        calibration = {key: np.array(archive[key], copy=True) for key in archive.files}
    required = {
        "class_handles", "class_centers", "class_radii", "domain_handles", "domain_centers", "domain_radii",
        "js_scale", "tau_q", "tau_margin", "tau_unknown_low", "tau_unknown_high",
    }
    if set(calibration) != required:
        raise DualReadoutBundleError("calibration member allowlist mismatch")
    if list(np.asarray(calibration["class_handles"]).astype(str)) != list(manifest.get("class_handles", [])):
        raise DualReadoutBundleError("class handle binding drift")
    receipt = json.loads((package / "calibration/receipt.json").read_text(encoding="utf-8"))
    forbidden = {"role", "true_label", "proxy_rows", "physical_ids", "raw_iq"}
    if receipt.get("schema") != CALIBRATION_SCHEMA or forbidden.intersection(receipt):
        raise DualReadoutBundleError("calibration receipt schema/forbidden field drift")
    angular = torch.jit.load(str(package / "runtimes/angular.ts"), map_location="cpu").eval()
    robust = torch.jit.load(str(package / "runtimes/robust.ts"), map_location="cpu").eval()
    return LoadedDualReadoutBundle(package, manifest, calibration, angular, robust)


__all__ = [
    "BUNDLE_SCHEMA", "CALIBRATION_SCHEMA", "FORMULA_ID", "DualReadoutBundleError",
    "LoadedDualReadoutBundle", "fit_source_calibration", "build_bundle", "load_bundle", "sha256_file",
]
