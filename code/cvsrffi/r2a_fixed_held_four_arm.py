"""Frozen R2A pseudo-new held-receiver four-arm proxy runner.

This module is intentionally a development-only, non-promotable proxy.  The
builder writes a model packet with no query labels; query labels live only in
the separately sealed truth sidecar consumed by :func:`score_packet`.
"""
from __future__ import annotations

import argparse
import base64
import dataclasses
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from cvsrffi.stage2_bayesian_predictive_head import Phase1BayesianPredictiveHeadLock
from cvsrffi.stage2_joint_rchm_bpp import (
    audit_joint_rchm_bpp_resources, build_joint_rchm_bpp_state,
    deserialize_joint_rchm_bpp_wire, score_joint_rchm_bpp_arm,
    serialize_joint_rchm_bpp_state,
)
from cvsrffi.stage2_receiver_context_hypermetric import Phase1ReceiverContextHypermetricLock
from cvsrffi.stage2_zid_student_t_qknn import (
    Phase1ZIDStudentTLock, build_typed_zid_support_bank,
)
from scripts.export_phase1_singleobs_dual_feature_archive import (
    MEMBERS as DUAL_ARCHIVE_MEMBERS,
    SCHEMA as DUAL_ARCHIVE_SCHEMA,
    verify_phase1_singleobs_dual_feature_archive,
)

SCOPE = "PHASE1_HELD_PROXY_NON_PROMOTABLE"
SCHEMA = "cvs.r2a.fixed-xcov-bpp-k5.v1"
CANDIDATE_REVISION = "v1.1"
K = 5
SCENES = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
ARMS = ("M0", "M_DA", "M_HEAD", "M_JOINT")
Z_DIM = 160
_MEMBERS = ("z_id", "z_dom", "labels", "receiver_ids", "day_ids", "physical_ids", "scenario_names", "class_ids")
COVERAGE_SCHEMA = "cvs.phase1.singleobs_dual_feature_coverage_receipt.v1"
COVERAGE_STATUS = "DESCRIPTIVE_ONLY_NO_HELD_FOLD_DECISION"
REAL_CLASS_IDS = ("14-10", "14-7", "20-15", "20-19", "6-15", "8-20")
_COVERAGE_METADATA = ("labels", "receiver_ids", "day_ids", "physical_ids", "scenario_names", "class_ids", "observation_ids")
_COVERAGE_KEYS = {
    "schema", "status", "artifact_stage", "archive_sha256", "manifest_sha256",
    "metadata_arrays_read", "feature_arrays_read", "row_count",
    "physical_id_unique_count", "observation_id_unique_count", "class_ids",
    "receiver_ids", "day_ids", "scenario_names", "counts_by_class",
    "counts_by_receiver", "counts_by_day", "counts_by_scenario",
    "counts_by_receiver_day_class", "receiver_day_class_cell_count",
    "receiver_day_class_zero_cell_count", "receiver_day_class_min_count",
    "receiver_day_class_max_count", "pre_registered_coverage_gate_passed",
    "k_values_described_only", "min_rows_remaining_after_support_by_k",
    "target_access", "query_access", "held_fold_selected",
}
_ARTIFACT_BINDING_KEYS = {
    "archive_schema", "coverage_schema", "archive_sha256", "manifest_sha256", "coverage_sha256",
}


class R2AFixedHeldError(ValueError):
    pass


def _canon(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


def _sha(value: Any) -> str:
    return hashlib.sha256(value if isinstance(value, bytes) else _canon(value)).hexdigest()


def _sha_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha_text(value: str, name: str = "sha256") -> str:
    if type(value) is not str or len(value) != 64 or value != value.lower() or any(c not in "0123456789abcdef" for c in value):
        raise R2AFixedHeldError(f"{name} must be a lowercase SHA256")
    return value


def _write_new(path: str | Path, data: bytes) -> None:
    target = Path(path)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"refusing to overwrite {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise R2AFixedHeldError("JSON root must be an object")
    return value


def _exact_keys(value: Any, expected: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise R2AFixedHeldError(f"{name} schema drift")
    return value


def _artifact_binding(value: Mapping[str, Any], coverage_sha256: str) -> dict[str, str]:
    binding = _exact_keys(value, _ARTIFACT_BINDING_KEYS, "input artifact binding")
    out = {key: str(binding[key]) for key in _ARTIFACT_BINDING_KEYS}
    if out["archive_schema"] != DUAL_ARCHIVE_SCHEMA or out["coverage_schema"] != COVERAGE_SCHEMA:
        raise R2AFixedHeldError("input artifact schema binding drift")
    for key in ("archive_sha256", "manifest_sha256", "coverage_sha256"):
        out[key] = _sha_text(out[key], key)
    if out["coverage_sha256"] != _sha_text(coverage_sha256, "coverage_sha256"):
        raise R2AFixedHeldError("coverage artifact binding drift")
    return out


def _blas_fingerprint() -> list[dict[str, Any]]:
    fields = ("user_api", "internal_api", "prefix", "version", "threading_layer", "architecture", "num_threads")
    info = [
        {key: item.get(key) for key in fields}
        for item in threadpool_info()
        if item.get("user_api") == "blas"
    ]
    info.sort(key=lambda item: tuple(str(item.get(key)) for key in fields))
    if not info or any(item.get("num_threads") != 1 for item in info):
        raise R2AFixedHeldError("BLAS single-thread execution contract drift")
    return info


def _row_key(coverage: str, held: str, scene: str, physical: str) -> bytes:
    return bytes.fromhex(coverage) + b"\0" + held.encode("utf-8") + b"\0" + scene.encode("utf-8") + b"\0" + physical.encode("utf-8")


def _encode_array(value: np.ndarray) -> dict[str, Any]:
    a = np.ascontiguousarray(value)
    return {"dtype": a.dtype.str, "shape": list(a.shape), "b64": base64.b64encode(a.tobytes()).decode("ascii")}


def _decode_array(value: Mapping[str, Any]) -> np.ndarray:
    if set(value) != {"dtype", "shape", "b64"} or type(value["shape"]) is not list:
        raise R2AFixedHeldError("packet array schema drift")
    dtype = np.dtype(str(value["dtype"])); shape = tuple(value["shape"])
    if any(type(x) is not int or x < 0 for x in shape):
        raise R2AFixedHeldError("packet array shape drift")
    raw = base64.b64decode(str(value["b64"]), validate=True)
    if len(raw) != int(np.prod(shape, dtype=np.int64)) * dtype.itemsize:
        raise R2AFixedHeldError("packet array size drift")
    return np.frombuffer(raw, dtype=dtype).copy().reshape(shape)


def _norm(rows: np.ndarray) -> np.ndarray:
    x = np.asarray(rows, dtype=np.float64)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(~np.isfinite(x)) or np.any(n <= 0):
        raise R2AFixedHeldError("z_id normalization requires finite nonzero rows")
    return x / n


def _q_rows(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    x = np.asarray(rows, dtype=np.float64)
    scale = np.maximum(np.max(np.abs(x), axis=1) / 127.0, np.finfo(np.float16).tiny).astype(np.float16)
    code = np.clip(np.rint(x / scale.astype(np.float64)[:, None]), -127, 127).astype(np.int8)
    replay = code.astype(np.float64) * scale.astype(np.float64)[:, None]
    return code, scale, float(np.max(np.abs(x - replay)))


def _q_vector(value: np.ndarray, *, positive: bool = False) -> tuple[np.ndarray, np.ndarray, float]:
    x = np.asarray(value, dtype=np.float64)
    if positive and np.any(x <= 0):
        raise R2AFixedHeldError("positive quantized vector received a nonpositive value")
    if positive:
        code = np.full(x.shape, 127, dtype=np.int8)
        scale = np.maximum(x / 127.0, np.finfo(np.float16).tiny).astype(np.float16)
    else:
        code = np.sign(x).astype(np.int8) * 127
        scale = np.maximum(np.abs(x) / 127.0, np.finfo(np.float16).tiny).astype(np.float16)
        code[x == 0] = 0
    replay = code.astype(np.float64) * scale.astype(np.float64)
    return code, scale, float(np.max(np.abs(x - replay)))


def _receipt(tag: str, payload: Mapping[str, Any]) -> str:
    return _sha({"tag": tag, "payload": payload})


def _validate_archive(archive: Mapping[str, Any]) -> dict[str, np.ndarray]:
    if not set(_MEMBERS).issubset(archive) or any(name.lower().startswith(("clean", "raw")) for name in archive):
        raise R2AFixedHeldError("archive field allowlist/clean-source boundary drift")
    out = {name: np.asarray(archive[name]) for name in _MEMBERS}
    n = len(out["labels"])
    if n < 1 or out["z_id"].dtype != np.float32 or out["z_dom"].dtype != np.float32 or out["z_id"].shape != (n, Z_DIM) or out["z_dom"].shape != (n, Z_DIM):
        raise R2AFixedHeldError("archive feature shape/dtype drift")
    if not np.isfinite(out["z_id"]).all() or not np.isfinite(out["z_dom"]).all():
        raise R2AFixedHeldError("archive feature finite drift")
    for name in ("labels", "receiver_ids", "day_ids", "physical_ids", "scenario_names", "class_ids"):
        if out[name].ndim != 1 or (name != "class_ids" and len(out[name]) != n):
            raise R2AFixedHeldError(f"archive {name} shape drift")
    for name in ("labels", "receiver_ids", "day_ids", "physical_ids", "scenario_names"):
        out[name] = out[name].astype(str)
        if any(not x for x in out[name]):
            raise R2AFixedHeldError(f"archive {name} empty value")
    out["class_ids"] = out["class_ids"].astype(str)
    if len(out["physical_ids"]) != len(set(out["physical_ids"].tolist())):
        raise R2AFixedHeldError("physical IDs must be globally unique")
    if set(out["scenario_names"].tolist()) != set(SCENES):
        raise R2AFixedHeldError("R2A requires exactly the three fixed weak LEO scenes")
    classes = tuple(sorted(out["class_ids"].tolist()))
    if len(classes) != 6 or len(set(classes)) != 6 or set(out["labels"].tolist()) != set(classes):
        raise R2AFixedHeldError("R2A requires exactly six registered pseudo-new classes")
    if len(set(out["receiver_ids"].tolist())) != 7:
        raise R2AFixedHeldError("R2A requires exactly seven coverage receivers")
    return out


def _coverage_receiver(receivers: Sequence[str], coverage_sha256: str) -> str:
    coverage = _sha_text(coverage_sha256, "coverage_sha256")
    ordered = tuple(sorted(receivers))
    return ordered[int.from_bytes(bytes.fromhex(coverage)[:8], "big", signed=False) % len(ordered)]


def _cell_lock_inputs(a: Mapping[str, np.ndarray], held_receiver: str, held_class: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, dict[str, Any]]:
    keep = (a["receiver_ids"] != held_receiver) & (a["labels"] != held_class)
    if not np.any(keep):
        raise R2AFixedHeldError("B_h exclusion left no Phase1 proxy rows")
    zid, zdom = a["z_id"][keep], a["z_dom"][keep]
    labels, receivers, days, scenes = (a[name][keep] for name in ("labels", "receiver_ids", "day_ids", "scenario_names"))
    center = zdom.astype(np.float64).mean(0); scale = zdom.astype(np.float64).std(0, ddof=1)
    if np.any(~np.isfinite(scale)) or np.any(scale <= 0):
        raise R2AFixedHeldError("B_h z_dom sample-std degeneracy")
    cells: dict[tuple[str, str, str, str], list[int]] = {}
    for i, key in enumerate(zip(receivers, labels, days, scenes)):
        cells.setdefault(tuple(key), []).append(i)
    if any(len(index) < 2 for index in cells.values()):
        raise R2AFixedHeldError("B_h requires n>=2 in every observed receiver/class/day/scene cell")
    grouped: dict[tuple[str, str, str], list[tuple[str, np.ndarray, np.ndarray]]] = {}
    cell_var: dict[str, list[float]] = {c: [] for c in sorted(set(labels.tolist()))}
    for (receiver, label, day, scene), index in sorted(cells.items()):
        ids = np.asarray(index, dtype=np.int64)
        x = ((zdom[ids].astype(np.float64) - center) / scale).mean(0)
        y_rows = _norm(zid[ids])
        y = y_rows.mean(0)
        y /= np.linalg.norm(y)
        grouped.setdefault((label, day, scene), []).append((receiver, x, y))
        cell_var[label].append(float(np.var(y_rows, axis=0, ddof=1).mean()))
    terms: list[np.ndarray] = []
    for key, values in sorted(grouped.items()):
        if len(values) < 2:
            raise R2AFixedHeldError("B_h class/day/scene group lacks receiver contrast")
        xs = np.asarray([v[1] for v in values], dtype=np.float64); ys = np.asarray([v[2] for v in values], dtype=np.float64)
        xs -= xs.mean(0); ys -= ys.mean(0)
        terms.extend(np.outer(x, y) for x, y in zip(xs, ys))
    if not terms:
        raise R2AFixedHeldError("B_h contrast matrix is empty")
    cmat = np.mean(np.asarray(terms, dtype=np.float64), axis=0)
    # NumPy is the sealed local LAPACK surface in ssr-gpu.  The calculation is
    # CPU float64 and has no randomized/iterative rank selection; q=4 is fixed
    # below.  The exact linked LAPACK surface is carried in the packet receipt.
    with threadpool_limits(limits=1, user_api="blas"):
        u, singular, vt = np.linalg.svd(cmat, full_matrices=False)
        blas = _blas_fingerprint()
    tol = max(float(singular[0]) * np.finfo(np.float64).eps * Z_DIM, 1.0e-12)
    if len(singular) < 5 or singular[3] <= tol or singular[3] - singular[4] <= tol:
        raise R2AFixedHeldError("B_h gesvd sigma4/singular-gap degeneracy")
    u, vt = u[:, :4], vt[:4, :]
    for i in range(4):
        j = int(np.argmax(np.abs(vt[i])))
        if vt[i, j] < 0:
            u[:, i] *= -1.0; vt[i] *= -1.0
    b0 = float(np.mean([np.mean(cell_var[c]) for c in sorted(cell_var)]))
    if not np.isfinite(b0) or b0 <= 0:
        raise R2AFixedHeldError("B_h C5 cell unbiased variance degeneracy")
    audit = {"excluded_receiver": held_receiver, "excluded_pseudo_new_class": held_class, "cell_count": len(cells), "contrast_terms": len(terms), "sigma4": float(singular[3]), "sigma4_gap": float(singular[3] - singular[4]), "lapack_driver": "numpy.linalg.svd", "numpy_version": np.__version__, "blas_lapack": blas, "blas_threads": sorted({int(item["num_threads"]) for item in blas})}
    return center, scale, np.vstack((u.T, vt)), b0, audit


def _locks(a: Mapping[str, np.ndarray], held_receiver: str, held_class: str, coverage_sha256: str) -> tuple[Phase1ZIDStudentTLock, Phase1ReceiverContextHypermetricLock, Phase1BayesianPredictiveHeadLock, dict[str, Any]]:
    center, scale, vectors, b0, audit = _cell_lock_inputs(a, held_receiver, held_class)
    ccode, cscale, ce = _q_vector(center); scode, sscale, se = _q_vector(scale, positive=True)
    pcode, pscale, pe = _q_rows(vectors[:4]); bcode, bscale, be = _q_rows(vectors[4:])
    # Fixed eta=1/8 Walsh cross-map keeps all four frozen SVD coordinates
    # observable without selecting a data-dependent map.
    cross_code = np.asarray(((1, 1, 1, 1), (1, -1, 1, -1), (1, 1, -1, -1), (1, -1, -1, 1)), dtype=np.int8) * 127
    cross_scale = np.full(4, np.float16(1.0 / (8.0 * 127.0)), dtype=np.float16)
    quant = {"center": ce, "scale": se, "receiver_projection": pe, "zid_basis": be, "eta": 0.125}
    provenance = {"coverage_sha256": coverage_sha256, "B_h": audit, "quantization": quant, "rounding": "IEEE-754 round-to-nearest-even via numpy.rint", "persistent": "INT8 codes + FP16 scales"}
    qlock = Phase1ZIDStudentTLock(K, 3.0, Z_DIM, 1.0, 0.2, 2.0, 0.5, 2.0, 1.0, _receipt("r2a-qknn-lodo", provenance), _receipt("r2a-qknn-quant", provenance))
    # C5 is intentionally identity by D_eff=5; C6 is the sole count clone.
    rchm = Phase1ReceiverContextHypermetricLock(K, 5, ccode, cscale, scode, sscale, pcode, pscale, cross_code, cross_scale, bcode, bscale, -0.6, -0.1, 1.0, 1.0e6, 1.0e6, 2.0, 1.0, qlock.lock_digest, __import__("cvsrffi.stage2_zid_student_t_qknn", fromlist=["identity_shared_psd_metric"]).identity_shared_psd_metric(config=qlock).metric_receipt_sha256, _receipt("r2a-rchm-lodo", provenance), _receipt("r2a-rchm-coverage", provenance), _receipt("r2a-rchm-crossmap", provenance), _receipt("r2a-rchm-quant", provenance))
    bpp = Phase1BayesianPredictiveHeadLock(K, 5, 2.0, float(max(b0, np.finfo(np.float16).tiny)), 1.0, qlock.lock_digest, _receipt("r2a-bpp-lodo", provenance), _receipt("r2a-bpp-quant", provenance), 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, "0" * 64, "0" * 64)
    audit["quantization_replay_receipt_sha256"] = _receipt("r2a-lock-replay", provenance)
    return qlock, rchm, bpp, audit


def _lock_wire(lock: Any) -> dict[str, Any]:
    value = dataclasses.asdict(lock)
    return {key: _encode_array(item) if isinstance(item, np.ndarray) else item for key, item in value.items()}


def _lock_unwire(value: Mapping[str, Any], cls: Any) -> Any:
    return cls(**{key: _decode_array(item) if isinstance(item, Mapping) and set(item) == {"dtype", "shape", "b64"} else item for key, item in value.items()})


def _support_indices(a: Mapping[str, np.ndarray], held_receiver: str, scene: str, classes: Sequence[str], coverage: str) -> tuple[np.ndarray, list[str], list[str]]:
    out: list[int] = []; labels: list[str] = []; query: list[str] = []
    for label in classes:
        ids = np.flatnonzero((a["receiver_ids"] == held_receiver) & (a["scenario_names"] == scene) & (a["labels"] == label))
        ordered = sorted(ids.tolist(), key=lambda i: hashlib.sha256(_row_key(coverage, label, scene, str(a["physical_ids"][i]))).digest())
        if len(ordered) <= K:
            raise R2AFixedHeldError("each held receiver/class/scene requires five support plus query")
        out.extend(ordered[:K]); labels.extend([label] * K); query.extend(str(a["physical_ids"][i]) for i in ordered[K:])
    return np.asarray(out, dtype=np.int64), labels, query


def _build_row(a: Mapping[str, np.ndarray], held_receiver: str, held: str, scene: str, coverage: str, locks: tuple[Any, Any, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    qlock, rchm5, bpp5 = locks
    classes = tuple(sorted(a["class_ids"].tolist())); old = tuple(c for c in classes if c != held)
    idx5, labels5, _ = _support_indices(a, held_receiver, scene, old, coverage)
    idx6, labels6, query_ids = _support_indices(a, held_receiver, scene, classes, coverage)
    rchm6, bpp6 = replace(rchm5, registered_class_count=6), replace(bpp5, registered_class_count=6)
    def build(indices: np.ndarray, labels: list[str], registered: tuple[str, ...], rlock: Any, hlock: Any) -> Any:
        bank = build_typed_zid_support_bank(a["z_id"][indices], labels, registered, config=qlock)
        receipt = _sha({"support_physical_ids": [str(a["physical_ids"][i]) for i in indices], "registered": list(registered)})
        return build_joint_rchm_bpp_state(a["z_dom"][indices].astype(np.float32), labels, registered, bank=bank, qknn_config=qlock, rchm_lock=rlock, bpp_lock=hlock, support_receipt_sha256=receipt)
    c5, c6 = build(idx5, labels5, old, rchm5, bpp5), build(idx6, labels6, classes, rchm6, bpp6)
    c5audit, c6audit = audit_joint_rchm_bpp_resources(c5), audit_joint_rchm_bpp_resources(c6)
    if c5.rchm.metric.effective_rank != 0 or c5.rchm.audit.fallback_reason != "effective_class_identity":
        raise R2AFixedHeldError("C5 must remain the D_eff=5 identity arm")
    if c6.rchm.metric.effective_rank != 1 or c6audit.support_build_mac != 5936 or c6audit.production_postprocess_mac_per_query != 1126:
        raise R2AFixedHeldError("C6 RCHM/BPP frozen resource/rank gate failed")
    row_id = _sha({"coverage": coverage, "held_receiver": held_receiver, "pseudo_new": held, "scene": scene})
    query_labels = {str(a["physical_ids"][i]): str(a["labels"][i]) for i in np.flatnonzero((a["receiver_ids"] == held_receiver) & (a["scenario_names"] == scene)) if str(a["physical_ids"][i]) in set(query_ids)}
    return ({"row_id": row_id, "pseudo_new": held, "scene": scene, "old_classes": list(old), "query_ids": query_ids, "c5_wire_b64": base64.b64encode(serialize_joint_rchm_bpp_state(c5)).decode("ascii"), "c5_wire_sha256": _sha(serialize_joint_rchm_bpp_state(c5)), "c6_wire_b64": base64.b64encode(serialize_joint_rchm_bpp_state(c6)).decode("ascii"), "c6_wire_sha256": _sha(serialize_joint_rchm_bpp_state(c6)), "resource": {"c5": dataclasses.asdict(c5audit), "c6": dataclasses.asdict(c6audit), "optimizer_steps": 0}}, {"row_id": row_id, "query_labels": query_labels})


def build_packet(archive: Mapping[str, Any], *, coverage_sha256: str, artifact_binding: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    a = _validate_archive(archive); coverage = _sha_text(coverage_sha256, "coverage_sha256")
    binding = _artifact_binding(artifact_binding, coverage)
    receivers = tuple(sorted(set(a["receiver_ids"].tolist()))); held_receiver = _coverage_receiver(receivers, coverage)
    lock_groups: dict[str, Any] = {}; rows: list[dict[str, Any]] = []; truth_rows: list[dict[str, Any]] = []
    for held in sorted(a["class_ids"].tolist()):
        qlock, rlock, hlock, audit = _locks(a, held_receiver, held, coverage)
        lock_groups[held] = {"qknn": _lock_wire(qlock), "rchm_c5": _lock_wire(rlock), "bpp_c5": _lock_wire(hlock), "audit": audit}
        for scene in SCENES:
            row, truth = _build_row(a, held_receiver, held, scene, coverage, (qlock, rlock, hlock))
            rows.append(row); truth_rows.append(truth)
    packet = {"schema": SCHEMA, "candidate_revision": CANDIDATE_REVISION, "evaluation_scope": SCOPE, "pseudo_new": True, "coverage_sha256": coverage, "input_artifact_binding": binding, "held_receiver": held_receiver, "receivers": list(receivers), "classes": sorted(a["class_ids"].tolist()), "K": K, "scenes": list(SCENES), "lock_groups": lock_groups, "rows": rows, "resource_contract": {"effective_rank": 1, "build_mac": 5936, "query_mac": 1126, "optimizer_steps": 0}, "sealed_execution": {"endianness": "coverage SHA first 8 bytes interpreted u64 big-endian", "blas_lapack": "numpy.linalg.svd on CPU float64 under verified threadpoolctl BLAS limit=1", "numpy_version": np.__version__, "singular_gaps": "sigma4>tol and sigma4-sigma5>tol", "bpp_compiled_error_caps": 2.0, "bpp_held_receipts": "zero SHA defaults"}}
    packet["packet_sha256"] = _sha(packet)
    truth = {"schema": SCHEMA + ".truth.v1", "candidate_revision": CANDIDATE_REVISION, "evaluation_scope": SCOPE, "packet_sha256": packet["packet_sha256"], "pseudo_new": True, "rows": truth_rows}
    truth["truth_sha256"] = _sha(truth)
    return packet, truth


def _verify_packet(packet: Mapping[str, Any]) -> None:
    if packet.get("schema") != SCHEMA or packet.get("candidate_revision") != CANDIDATE_REVISION or packet.get("evaluation_scope") != SCOPE or packet.get("pseudo_new") is not True:
        raise R2AFixedHeldError("packet scope/schema drift")
    expected = dict(packet); digest = expected.pop("packet_sha256", None)
    rows = packet.get("rows")
    if _sha(expected) != digest or type(rows) is not list or len(rows) != 18:
        raise R2AFixedHeldError("packet receipt/18-row drift")
    coverage = _sha_text(str(packet.get("coverage_sha256")), "coverage_sha256")
    _artifact_binding(packet.get("input_artifact_binding", {}), coverage)
    classes = packet.get("classes"); receivers = packet.get("receivers")
    if type(classes) is not list or classes != sorted(classes) or len(classes) != 6 or len(set(classes)) != 6:
        raise R2AFixedHeldError("packet class registry drift")
    if type(receivers) is not list or receivers != sorted(receivers) or len(receivers) != 7 or len(set(receivers)) != 7 or packet.get("held_receiver") not in receivers:
        raise R2AFixedHeldError("packet receiver registry drift")
    if packet.get("K") != K or packet.get("scenes") != list(SCENES) or set(packet.get("lock_groups", {})) != set(classes):
        raise R2AFixedHeldError("packet fixed K/scene/lock registry drift")
    expected_pairs = [(held, scene) for held in classes for scene in SCENES]
    for row, (held, scene) in zip(rows, expected_pairs):
        _exact_keys(row, {"row_id", "pseudo_new", "scene", "old_classes", "query_ids", "c5_wire_b64", "c5_wire_sha256", "c6_wire_b64", "c6_wire_sha256", "resource"}, "packet row")
        if row["pseudo_new"] != held or row["scene"] != scene or row["old_classes"] != [c for c in classes if c != held] or row["row_id"] != _sha({"coverage": coverage, "held_receiver": packet["held_receiver"], "pseudo_new": held, "scene": scene}):
            raise R2AFixedHeldError("packet row identity/order drift")
        if type(row["query_ids"]) is not list or not row["query_ids"] or any(type(value) is not str or not value for value in row["query_ids"]) or len(row["query_ids"]) != len(set(row["query_ids"])):
            raise R2AFixedHeldError("packet row opaque query ID drift")
        for prefix in ("c5", "c6"):
            wire_sha = _sha_text(str(row[f"{prefix}_wire_sha256"]), f"{prefix}_wire_sha256")
            try:
                wire = base64.b64decode(str(row[f"{prefix}_wire_b64"]), validate=True)
            except (ValueError, TypeError) as exc:
                raise R2AFixedHeldError("packet state wire base64 drift") from exc
            if _sha(wire) != wire_sha:
                raise R2AFixedHeldError("packet state wire SHA drift")


def _verify_truth(packet: Mapping[str, Any], truth: Mapping[str, Any], expected_truth_sha256: str) -> list[Mapping[str, Any]]:
    _exact_keys(truth, {"schema", "candidate_revision", "evaluation_scope", "packet_sha256", "pseudo_new", "rows", "truth_sha256"}, "truth sidecar")
    expected_digest = _sha_text(expected_truth_sha256, "truth_sha256")
    signed = dict(truth); actual = signed.pop("truth_sha256")
    if actual != expected_digest or _sha(signed) != actual:
        raise R2AFixedHeldError("truth sidecar SHA drift")
    if truth["schema"] != SCHEMA + ".truth.v1" or truth["candidate_revision"] != CANDIDATE_REVISION or truth["evaluation_scope"] != SCOPE or truth["pseudo_new"] is not True or truth["packet_sha256"] != packet["packet_sha256"]:
        raise R2AFixedHeldError("truth sidecar packet/scope drift")
    rows = truth["rows"]
    if type(rows) is not list or len(rows) != 18:
        raise R2AFixedHeldError("truth sidecar 18-row drift")
    for packet_row, row in zip(packet["rows"], rows):
        _exact_keys(row, {"row_id", "query_labels"}, "truth row")
        labels = row["query_labels"]
        if row["row_id"] != packet_row["row_id"] or not isinstance(labels, Mapping) or set(labels) != set(packet_row["query_ids"]) or len(labels) != len(packet_row["query_ids"]):
            raise R2AFixedHeldError("truth row/query order binding drift")
        if any(type(value) is not str or value not in packet["classes"] for value in labels.values()) or set(labels.values()) != set(packet["classes"]):
            raise R2AFixedHeldError("truth row class coverage drift")
    return rows


def _verify_prediction(packet: Mapping[str, Any], prediction: Mapping[str, Any], commit: str) -> list[Mapping[str, Any]]:
    _exact_keys(prediction, {"schema", "candidate_revision", "evaluation_scope", "packet_sha256", "rows", "COMMIT"}, "prediction")
    expected_commit = _sha_text(commit, "COMMIT")
    signed = dict(prediction); actual = signed.pop("COMMIT")
    if actual != expected_commit or _sha(signed) != actual:
        raise R2AFixedHeldError("prediction COMMIT drift")
    if prediction["schema"] != SCHEMA + ".prediction.v1" or prediction["candidate_revision"] != CANDIDATE_REVISION or prediction["evaluation_scope"] != SCOPE or prediction["packet_sha256"] != packet["packet_sha256"]:
        raise R2AFixedHeldError("prediction packet/scope drift")
    rows = prediction["rows"]
    if type(rows) is not list or len(rows) != 18:
        raise R2AFixedHeldError("prediction 18-row drift")
    for packet_row, row in zip(packet["rows"], rows):
        _exact_keys(row, {"row_id", "query_ids", "before", "after"}, "prediction row")
        if row["row_id"] != packet_row["row_id"] or row["query_ids"] != packet_row["query_ids"]:
            raise R2AFixedHeldError("prediction row/query identity drift")
        for stage, expected_classes in (("before", packet_row["old_classes"]), ("after", packet["classes"])):
            arms = row[stage]
            if not isinstance(arms, Mapping) or tuple(arms) != ARMS:
                raise R2AFixedHeldError("prediction four-arm order/schema drift")
            for arm in ARMS:
                payload = _exact_keys(arms[arm], {"classes", "logits", "prediction"}, "prediction arm")
                if payload["classes"] != expected_classes or type(payload["prediction"]) is not list or len(payload["prediction"]) != len(row["query_ids"]) or any(type(value) is not str or value not in expected_classes for value in payload["prediction"]):
                    raise R2AFixedHeldError("prediction arm class/prediction drift")
                if not isinstance(payload["logits"], Mapping):
                    raise R2AFixedHeldError("prediction arm logits schema drift")
                logits = _decode_array(payload["logits"])
                if logits.dtype not in (np.dtype("float32"), np.dtype("float64")) or logits.shape != (len(row["query_ids"]), len(expected_classes)) or not np.isfinite(logits).all():
                    raise R2AFixedHeldError("prediction arm logits shape/dtype/finite drift")
                expected_prediction = [expected_classes[index] for index in np.argmax(logits, axis=1).tolist()]
                if payload["prediction"] != expected_prediction:
                    raise R2AFixedHeldError("prediction arm argmax/logit binding drift")
    return rows


def predict_packet(packet: Mapping[str, Any], query_ids: Sequence[str], query_zid: np.ndarray) -> dict[str, Any]:
    _verify_packet(packet)
    ids = [str(x) for x in query_ids]; z = np.asarray(query_zid)
    if z.dtype != np.float32 or z.shape != (len(ids), Z_DIM) or len(ids) != len(set(ids)) or not np.isfinite(z).all():
        raise R2AFixedHeldError("predict accepts only unique opaque IDs and finite float32 z_id")
    lookup = {key: z[i] for i, key in enumerate(ids)}; output_rows: list[dict[str, Any]] = []
    for row in packet["rows"]:
        group = packet["lock_groups"][row["pseudo_new"]]
        qlock = _lock_unwire(group["qknn"], Phase1ZIDStudentTLock); r5 = _lock_unwire(group["rchm_c5"], Phase1ReceiverContextHypermetricLock); h5 = _lock_unwire(group["bpp_c5"], Phase1BayesianPredictiveHeadLock)
        r6, h6 = replace(r5, registered_class_count=6), replace(h5, registered_class_count=6)
        wanted = row["query_ids"]
        if any(key not in lookup for key in wanted):
            raise R2AFixedHeldError("predict query feature set is missing packet opaque IDs")
        q = np.asarray([lookup[key] for key in wanted], dtype=np.float32)
        c5 = deserialize_joint_rchm_bpp_wire(base64.b64decode(row["c5_wire_b64"]), row["c5_wire_sha256"], qlock, r5, h5)
        c6 = deserialize_joint_rchm_bpp_wire(base64.b64decode(row["c6_wire_b64"]), row["c6_wire_sha256"], qlock, r6, h6)
        def arms(state: Any, lock: Any) -> dict[str, Any]:
            answer = {}
            for arm in ARMS:
                logits = score_joint_rchm_bpp_arm(state, q, arm=arm, qknn_config=qlock, bpp_lock=lock)
                classes = list(state.bank.classes); prediction = [classes[i] for i in np.argmax(logits, axis=1).tolist()]
                answer[arm] = {"classes": classes, "logits": _encode_array(logits), "prediction": prediction}
            return answer
        output_rows.append({"row_id": row["row_id"], "query_ids": wanted, "before": arms(c5, h5), "after": arms(c6, h6)})
    result = {"schema": SCHEMA + ".prediction.v1", "candidate_revision": CANDIDATE_REVISION, "evaluation_scope": SCOPE, "packet_sha256": packet["packet_sha256"], "rows": output_rows}
    result["COMMIT"] = _sha(result)
    return result


def _acc(pred: Sequence[str], truth: Sequence[str], classes: Sequence[str]) -> tuple[float, dict[str, float]]:
    per = {c: float(np.mean([p == y for p, y in zip(pred, truth) if y == c])) for c in classes}
    return float(np.mean(list(per.values()))), per


def score_packet(packet: Mapping[str, Any], prediction: Mapping[str, Any], truth: Mapping[str, Any], *, commit: str, truth_sha256: str) -> list[dict[str, Any]]:
    _verify_packet(packet)
    prediction_rows = _verify_prediction(packet, prediction, commit)
    truth_rows = _verify_truth(packet, truth, truth_sha256)
    metrics: list[dict[str, Any]] = []
    for packet_row, row, truth_row in zip(packet["rows"], prediction_rows, truth_rows):
        labels = truth_row["query_labels"]
        y = [str(labels[key]) for key in row["query_ids"]]; held = packet_row["pseudo_new"]; old = packet_row["old_classes"]
        old_mask = [i for i, value in enumerate(y) if value != held]; new_mask = [i for i, value in enumerate(y) if value == held]
        quartet: dict[str, dict[str, Any]] = {}
        for arm in ARMS:
            before = row["before"][arm]["prediction"]; after = row["after"][arm]["prediction"]
            old_before = float(np.mean([before[i] == y[i] for i in old_mask])); old_after = float(np.mean([after[i] == y[i] for i in old_mask])); seen_new = float(np.mean([after[i] == held for i in new_mask]))
            _, per = _acc(after, y, packet["classes"]); ba = float(np.mean(list(per.values()))); floor = float(min(per.values())); h = 0.0 if old_after + seen_new == 0 else float(2 * old_after * seen_new / (old_after + seen_new))
            quartet[arm] = {"row_id": row["row_id"], "held_receiver": packet["held_receiver"], "pseudo_new": held, "scene": packet_row["scene"], "K": K, "selection_coverage_sha256": packet["coverage_sha256"], "arm": arm, "query_count": len(y), "old_before": old_before, "old_after": old_after, "old_adaptation_gain": old_after - old_before, "seen_new": seen_new, "H_old_new": h, "BA": ba, "floor": floor, "min_old": float(min(per[c] for c in old)), "min_new": float(per[held]), "forgetting": old_before - old_after, "old_to_new": float(np.mean([after[i] == held for i in old_mask])), "new_to_old": float(np.mean([after[i] in old for i in new_mask])), "per_class": per, "resource": packet_row["resource"]}
        i_syn = quartet["M_JOINT"]["H_old_new"] - quartet["M_DA"]["H_old_new"] - quartet["M_HEAD"]["H_old_new"] + quartet["M0"]["H_old_new"]
        for arm in ARMS:
            quartet[arm]["I_syn"] = float(i_syn)
            metrics.append(quartet[arm])
    if len(metrics) != 18 * 4:
        raise R2AFixedHeldError("scoring must preserve all 18x4 negative/positive rows")
    return metrics


def _count_map(receipt: Mapping[str, Any], name: str, keys: Sequence[str]) -> dict[str, int]:
    value = receipt.get(name)
    if not isinstance(value, Mapping) or set(value) != set(keys):
        raise R2AFixedHeldError(f"coverage {name} key drift")
    out = dict(value)
    if any(type(item) is not int or item < 0 for item in out.values()):
        raise R2AFixedHeldError(f"coverage {name} count drift")
    return out


def _validate_coverage_receipt(receipt: Mapping[str, Any], *, archive_sha256: str, manifest_sha256: str, coverage_sha256: str) -> dict[str, str]:
    _exact_keys(receipt, _COVERAGE_KEYS, "coverage receipt")
    if receipt["schema"] != COVERAGE_SCHEMA or receipt["status"] != COVERAGE_STATUS or receipt["artifact_stage"] != "phase1_offline_before_target_access":
        raise R2AFixedHeldError("coverage receipt scope/schema drift")
    if _sha_text(str(receipt["archive_sha256"]), "coverage archive_sha256") != archive_sha256 or _sha_text(str(receipt["manifest_sha256"]), "coverage manifest_sha256") != manifest_sha256:
        raise R2AFixedHeldError("coverage archive/manifest binding drift")
    if receipt["metadata_arrays_read"] != list(_COVERAGE_METADATA) or receipt["feature_arrays_read"] != [] or receipt["target_access"] is not False or receipt["query_access"] is not False or receipt["held_fold_selected"] is not False:
        raise R2AFixedHeldError("coverage metadata-only access boundary drift")
    classes, receivers, days, scenes = (receipt[name] for name in ("class_ids", "receiver_ids", "day_ids", "scenario_names"))
    if classes != list(REAL_CLASS_IDS) or scenes != list(SCENES):
        raise R2AFixedHeldError("coverage class/scene registry drift")
    for name, values, count in (("receiver", receivers, 7), ("day", days, 4)):
        if type(values) is not list or values != sorted(values) or len(values) != count or len(set(values)) != count or any(type(value) is not str or not value or "|" in value for value in values):
            raise R2AFixedHeldError(f"coverage {name} registry drift")
    if receipt["row_count"] != 8400 or receipt["physical_id_unique_count"] != 8400 or receipt["observation_id_unique_count"] != 8400:
        raise R2AFixedHeldError("coverage row/physical/observation count drift")
    by_class = _count_map(receipt, "counts_by_class", classes)
    by_receiver = _count_map(receipt, "counts_by_receiver", receivers)
    by_day = _count_map(receipt, "counts_by_day", days)
    by_scene = _count_map(receipt, "counts_by_scenario", scenes)
    cell_keys = [f"{receiver}|{day}|{label}" for receiver in receivers for day in days for label in classes]
    cells = _count_map(receipt, "counts_by_receiver_day_class", cell_keys)
    values = list(cells.values())
    if any(sum(counts.values()) != 8400 for counts in (by_class, by_receiver, by_day, by_scene)) or sum(values) != 8400:
        raise R2AFixedHeldError("coverage aggregate row-count drift")
    for label in classes:
        if by_class[label] != sum(cells[f"{receiver}|{day}|{label}"] for receiver in receivers for day in days):
            raise R2AFixedHeldError("coverage class/cell aggregate drift")
    for receiver in receivers:
        if by_receiver[receiver] != sum(cells[f"{receiver}|{day}|{label}"] for day in days for label in classes):
            raise R2AFixedHeldError("coverage receiver/cell aggregate drift")
    for day in days:
        if by_day[day] != sum(cells[f"{receiver}|{day}|{label}"] for receiver in receivers for label in classes):
            raise R2AFixedHeldError("coverage day/cell aggregate drift")
    minimum, maximum = min(values), max(values)
    if receipt["receiver_day_class_cell_count"] != 168 or receipt["receiver_day_class_zero_cell_count"] != 0 or receipt["receiver_day_class_min_count"] != minimum or receipt["receiver_day_class_max_count"] != maximum or minimum <= 10 or receipt["pre_registered_coverage_gate_passed"] is not True:
        raise R2AFixedHeldError("coverage receiver-day-class gate drift")
    if receipt["k_values_described_only"] != [1, 5, 10] or receipt["min_rows_remaining_after_support_by_k"] != {str(k): minimum - k for k in (1, 5, 10)}:
        raise R2AFixedHeldError("coverage K remainder drift")
    return {"archive_schema": DUAL_ARCHIVE_SCHEMA, "coverage_schema": COVERAGE_SCHEMA, "archive_sha256": archive_sha256, "manifest_sha256": manifest_sha256, "coverage_sha256": _sha_text(coverage_sha256, "coverage_sha256")}


def _load_archive(path: str | Path, manifest: str | Path, coverage: str | Path, coverage_sha256: str) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    raw_paths = tuple(Path(value) for value in (path, manifest, coverage))
    for name, value in zip(("archive", "manifest", "coverage"), raw_paths):
        if value.is_symlink() or not value.is_file():
            raise R2AFixedHeldError(f"{name} must be a regular non-symlink file")
    archive_path, manifest_path, coverage_path = (value.resolve() for value in raw_paths)
    m = _read_json(manifest_path)
    try:
        verify_phase1_singleobs_dual_feature_archive(archive_path, m)
    except (OSError, TypeError, ValueError, KeyError) as exc:
        raise R2AFixedHeldError("dual archive verifier rejected input") from exc
    archive_sha, manifest_sha = _sha_file(archive_path), _sha_file(manifest_path)
    expected_coverage = _sha_text(coverage_sha256, "coverage_sha256")
    if _sha_file(coverage_path) != expected_coverage:
        raise R2AFixedHeldError("coverage path/SHA256 drift")
    binding = _validate_coverage_receipt(_read_json(coverage_path), archive_sha256=archive_sha, manifest_sha256=manifest_sha, coverage_sha256=expected_coverage)
    with np.load(archive_path, allow_pickle=False) as data:
        if tuple(data.files) != DUAL_ARCHIVE_MEMBERS:
            raise R2AFixedHeldError("dual archive member order drift after verification")
        arrays = {name: np.asarray(data[name]) for name in _MEMBERS}
    return arrays, binding


def _query_arrays(packet: Mapping[str, Any], archive: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    wanted = sorted({value for row in packet["rows"] for value in row["query_ids"]})
    index = {str(value): i for i, value in enumerate(archive["physical_ids"].tolist())}
    if any(value not in index for value in wanted):
        raise R2AFixedHeldError("packet query IDs missing from verified archive")
    return np.asarray(wanted, dtype=np.str_), np.asarray([archive["z_id"][index[value]] for value in wanted], dtype=np.float32)


def _write_query_new(path: str | Path, query_ids: np.ndarray, z_id: np.ndarray) -> None:
    target = Path(path)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"refusing to overwrite {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as handle:
        np.savez_compressed(handle, query_ids=query_ids, z_id=z_id)
        handle.flush(); os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--archive", required=True); b.add_argument("--manifest", required=True); b.add_argument("--coverage", required=True); b.add_argument("--coverage-sha256", required=True); b.add_argument("--packet", required=True); b.add_argument("--truth", required=True); b.add_argument("--query", required=True)
    p = sub.add_parser("predict"); p.add_argument("--packet", required=True); p.add_argument("--query", required=True); p.add_argument("--output", required=True)
    s = sub.add_parser("score"); s.add_argument("--packet", required=True); s.add_argument("--prediction", required=True); s.add_argument("--truth", required=True); s.add_argument("--truth-sha256", required=True); s.add_argument("--commit", required=True); s.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.cmd == "build":
        archive, binding = _load_archive(args.archive, args.manifest, args.coverage, args.coverage_sha256)
        packet, truth = build_packet(archive, coverage_sha256=args.coverage_sha256, artifact_binding=binding)
        query_ids, query_zid = _query_arrays(packet, archive)
        _write_new(args.packet, _canon(packet) + b"\n"); _write_new(args.truth, _canon(truth) + b"\n"); _write_query_new(args.query, query_ids, query_zid)
    elif args.cmd == "predict":
        packet = _read_json(args.packet)
        with np.load(args.query, allow_pickle=False) as query:
            if tuple(query.files) != ("query_ids", "z_id"):
                raise R2AFixedHeldError("predict query file must contain only opaque query_ids and z_id")
            result = predict_packet(packet, query["query_ids"].astype(str).tolist(), np.asarray(query["z_id"]))
        _write_new(args.output, _canon(result) + b"\n")
    else:
        metrics = score_packet(_read_json(args.packet), _read_json(args.prediction), _read_json(args.truth), commit=args.commit, truth_sha256=args.truth_sha256); _write_new(args.output, _canon({"schema": SCHEMA + ".score.v1", "candidate_revision": CANDIDATE_REVISION, "evaluation_scope": SCOPE, "pseudo_new": True, "COMMIT": args.commit, "truth_sha256": args.truth_sha256, "metrics": metrics}) + b"\n")


if __name__ == "__main__":
    main()
