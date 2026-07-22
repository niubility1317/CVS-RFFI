"""K=5 held falsifier for frozen ``SVRN-qKNN-BCRR/r2``.

``build`` alone reads the immutable r8 archive and writes a separately sealed
truth sidecar.  ``predict`` accepts only opaque query IDs and z_id.  ``score``
is the sole truth consumer and emits 72 same-row arm metrics plus the frozen
S18/S19 stopping decision.
"""
from __future__ import annotations

import argparse
import base64
import dataclasses
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from cvsrffi.r2a_fixed_held_four_arm import (
    COVERAGE_SCHEMA,
    DUAL_ARCHIVE_SCHEMA,
    REAL_CLASS_IDS,
    SCENES,
    _artifact_binding,
    _coverage_receiver,
    _decode_array,
    _encode_array,
    _load_archive,
    _query_arrays,
    _read_json,
    _sha_text,
    _support_indices,
    _validate_archive,
    _write_new,
    _write_query_new,
)
from cvsrffi.stage2_svrn_bcr import (
    ETA_GRID,
    KAPPA,
    LAMBDA0,
    MASK_MODULUS,
    MASK_RESIDUES,
    MASK_RETENTION,
    MAX_STATE_BYTES,
    SVRNBranchState,
    build_branch_state,
    deserialize_branch_state,
    qknn_neighbor_receipt,
    score_branch_logits,
    select_svrn_eta,
    serialize_branch_state,
)
from cvsrffi.stage2_zid_student_t_qknn import Phase1ZIDStudentTLock, Z_DIM

CANDIDATE_REVISION = "SVRN-qKNN-BCRR/r2-held"
SCOPE = "PHASE1_HELD_PROXY_NON_PROMOTABLE"
SCHEMA = "cvs.stage2.svrn_bcrr.fixed_held.v2"
ARMS = ("M0", "M_DA", "M_OTHER", "M_JOINT")
K = 5
# The sealed r8 archive stores the physical receiver ID without an ``rx``
# presentation prefix.  Keep the wire value byte-exact with that artifact.
HELD_RECEIVER = "1-1"
MAX_BUILD_NS = 30_000_000_000
MAX_PREDICT_NS = 5_000_000_000
MAX_BUILD_MAC = 100_000_000
MAX_PREDICT_MAC = 20_000_000


class SVRNBCRFixedHeldError(ValueError):
    pass


def _canon(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


def _digest(value: Any) -> str:
    return hashlib.sha256(value if isinstance(value, bytes) else _canon(value)).hexdigest()


def _qknn_lock(coverage_sha256: str) -> Phase1ZIDStudentTLock:
    coverage = _sha_text(coverage_sha256, "coverage_sha256")
    authority = {
        "candidate": CANDIDATE_REVISION,
        "design_commit": "407144dd714270bd8305595176dfcf921246b75d",
        "coverage_sha256": coverage,
        "K": K,
        "student_nu": 3.0,
        "kernel_effective_dim": Z_DIM,
        "kernel_volume_gamma": 1.0,
        "shared_h0": 0.2,
        "scale_prior_strength": 2.0,
        "scale_min_ratio": 0.5,
        "scale_max_ratio": 2.0,
        "temperature": 1.0,
    }
    return Phase1ZIDStudentTLock(
        K, 3.0, Z_DIM, 1.0, 0.2, 2.0, 0.5, 2.0, 1.0,
        _digest({"kind": "svrn-bcrr-fixed-qknn", **authority}),
        _digest({"kind": "svrn-bcrr-fixed-int8", **authority}),
    )


def _lock_wire(lock: Phase1ZIDStudentTLock) -> dict[str, Any]:
    return dataclasses.asdict(lock)


def _lock_unwire(value: Mapping[str, Any]) -> Phase1ZIDStudentTLock:
    if not isinstance(value, Mapping):
        raise SVRNBCRFixedHeldError("qKNN lock wire drift")
    try:
        return Phase1ZIDStudentTLock(**dict(value))
    except Exception as exc:
        raise SVRNBCRFixedHeldError("qKNN lock wire invalid") from exc


def _support_receipt(ids: Sequence[str], classes: Sequence[str]) -> str:
    return _digest({"support_physical_ids": list(map(str, ids)), "registered": list(map(str, classes))})


def _branch_bindings(raw: SVRNBranchState, svrn: SVRNBranchState) -> dict[str, Any]:
    eta_sha = svrn.eta_receipt["receipt_sha256"] if svrn.eta_receipt is not None else None
    return {
        "M0": {"branch": "raw", "qknn_state_receipt_sha256": raw.receipt_sha256},
        "M_DA": {"branch": "svrn", "qknn_state_receipt_sha256": svrn.receipt_sha256, "eta_receipt_sha256": eta_sha},
        "M_OTHER": {"branch": "raw", "qknn_state_receipt_sha256": raw.receipt_sha256, "bcrr_receipt_sha256": raw.bcrr_receipt["receipt_sha256"]},
        "M_JOINT": {"branch": "svrn", "qknn_state_receipt_sha256": svrn.receipt_sha256, "bcrr_receipt_sha256": svrn.bcrr_receipt["receipt_sha256"], "eta_receipt_sha256": eta_sha},
    }


def _fit_state(
    z: np.ndarray,
    labels: list[str],
    classes: tuple[str, ...],
    ids: list[str],
    qlock: Phase1ZIDStudentTLock,
) -> dict[str, Any]:
    started = time.perf_counter_ns()
    eta = select_svrn_eta(z, labels, classes, ids, active_k=K)
    raw = build_branch_state(z, labels, classes, ids, qknn_config=qlock, branch="raw")
    svrn = build_branch_state(z, labels, classes, ids, qknn_config=qlock, branch="svrn", eta_receipt=eta)
    raw_wire, svrn_wire = serialize_branch_state(raw), serialize_branch_state(svrn)
    total = len(raw_wire) + len(svrn_wire)
    elapsed = int(time.perf_counter_ns() - started)
    build_mac = int(raw.resource["build_mac"] + svrn.resource["build_mac"])
    query_mac = int(raw.resource["query_mac_per_sample"] + svrn.resource["query_mac_per_sample"])
    if total > MAX_STATE_BYTES or elapsed > MAX_BUILD_NS or build_mac > MAX_BUILD_MAC or query_mac > MAX_PREDICT_MAC:
        raise SVRNBCRFixedHeldError("SVRN-qKNN-BCRR state resource budget exceeded")
    resource = {
        "support_receipt_sha256": _support_receipt(ids, classes),
        "eta_receipt_sha256": eta["receipt_sha256"],
        "selected_eta": float(eta["selected_eta"]),
        "eta_fallback": eta["fallback"],
        "omega_raw": float(raw.bcrr_receipt["omega_q"]),
        "omega_raw_fallback": raw.bcrr_receipt["fallback"],
        "omega_svrn": float(svrn.bcrr_receipt["omega_q"]),
        "omega_svrn_fallback": svrn.bcrr_receipt["fallback"],
        "wire_state_bytes": {"raw": len(raw_wire), "svrn": len(svrn_wire), "total": total},
        "mac_ledger": {
            "raw_build_mac": int(raw.resource["build_mac"]),
            "svrn_build_mac": int(svrn.resource["build_mac"]),
            "build_total_mac": build_mac,
            "raw_query_mac_per_sample": int(raw.resource["query_mac_per_sample"]),
            "svrn_query_mac_per_sample": int(svrn.resource["query_mac_per_sample"]),
            "four_arm_query_per_sample_mac": query_mac,
            "bcr_factorizations_per_branch": 1,
            "bcr_loo_full_d3_count": 0,
            "formula": "shared Qraw/Qsvrn plus branch-local continuous BCRR d*C residual",
        },
        "quantization": {"raw": dict(raw.quantization_audit), "svrn": dict(svrn.quantization_audit)},
        "optimizer_steps": 0,
        "persistent_fp32_sidecar_bytes": 0,
        "build_elapsed_ns": elapsed,
        "backend": {"name": "numpy_cpu", "cuda_tensor_count": 0, "peak_vram_bytes": 0},
    }
    return {
        "raw_wire_b64": base64.b64encode(raw_wire).decode("ascii"),
        "raw_wire_sha256": _digest(raw_wire),
        "svrn_wire_b64": base64.b64encode(svrn_wire).decode("ascii"),
        "svrn_wire_sha256": _digest(svrn_wire),
        "eta_receipt": dict(eta),
        "branch_bindings": _branch_bindings(raw, svrn),
        "resource": resource,
    }


def _row(
    archive: Mapping[str, np.ndarray], held_receiver: str, held: str,
    scene: str, coverage: str, qlock: Phase1ZIDStudentTLock,
) -> tuple[dict[str, Any], dict[str, Any]]:
    classes = tuple(sorted(archive["class_ids"].astype(str).tolist()))
    old = tuple(label for label in classes if label != held)
    index5, labels5, _ = _support_indices(archive, held_receiver, scene, old, coverage)
    index6, labels6, query_ids = _support_indices(archive, held_receiver, scene, classes, coverage)
    ids5 = [str(archive["physical_ids"][i]) for i in index5]
    ids6 = [str(archive["physical_ids"][i]) for i in index6]
    c5 = _fit_state(archive["z_id"][index5].astype(np.float32), labels5, old, ids5, qlock)
    c6 = _fit_state(archive["z_id"][index6].astype(np.float32), labels6, classes, ids6, qlock)
    row_id = _digest({"coverage": coverage, "held_receiver": held_receiver, "pseudo_new": held, "scene": scene})
    query_set = set(query_ids)
    truth = {
        str(archive["physical_ids"][i]): str(archive["labels"][i])
        for i in np.flatnonzero((archive["receiver_ids"].astype(str) == held_receiver) & (archive["scenario_names"].astype(str) == scene))
        if str(archive["physical_ids"][i]) in query_set
    }
    supports = {"C5": ids5, "C6": ids6}
    return (
        {
            "row_id": row_id, "pseudo_new": held, "scene": scene,
            "old_classes": list(old), "support_physical_ids": supports,
            "support_receipt_sha256": _digest(supports),
            "query_ids": query_ids, "query_ids_sha256": _digest(query_ids),
            "c5": c5, "c6": c6,
        },
        {"row_id": row_id, "query_labels": truth},
    )


def build_packet(
    archive: Mapping[str, Any], *, coverage_sha256: str,
    artifact_binding: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    data = _validate_archive(archive)
    coverage = _sha_text(coverage_sha256, "coverage_sha256")
    binding = _artifact_binding(artifact_binding, coverage)
    receivers = tuple(sorted(set(data["receiver_ids"].astype(str).tolist())))
    held_receiver = _coverage_receiver(receivers, coverage)
    if held_receiver != HELD_RECEIVER:
        raise SVRNBCRFixedHeldError("frozen r8 coverage must select receiver 1-1")
    classes = tuple(sorted(data["class_ids"].astype(str).tolist()))
    if classes != tuple(REAL_CLASS_IDS):
        raise SVRNBCRFixedHeldError("frozen six-class registry drift")
    qlock = _qknn_lock(coverage)
    rows: list[dict[str, Any]] = []
    truths: list[dict[str, Any]] = []
    for held in classes:
        for scene in SCENES:
            row, truth = _row(data, held_receiver, held, scene, coverage, qlock)
            rows.append(row); truths.append(truth)
    build_ns = [row[stage]["resource"]["build_elapsed_ns"] for row in rows for stage in ("c5", "c6")]
    build_mac = [row[stage]["resource"]["mac_ledger"]["build_total_mac"] for row in rows for stage in ("c5", "c6")]
    packet = {
        "schema": SCHEMA, "candidate_revision": CANDIDATE_REVISION,
        "evaluation_scope": SCOPE, "pseudo_new": True,
        "coverage_sha256": coverage, "input_artifact_binding": binding,
        "held_receiver": held_receiver, "receivers": list(receivers),
        "classes": list(classes), "K": K, "scenes": list(SCENES),
        "qknn_lock": _lock_wire(qlock), "rows": rows,
        "contract": {
            "arms": list(ARMS), "design_commit": "407144dd714270bd8305595176dfcf921246b75d",
            "kappa": KAPPA, "eta_grid": list(ETA_GRID),
            "mask_modulus": MASK_MODULUS, "mask_residues": list(MASK_RESIDUES),
            "mask_retention": MASK_RETENTION, "lambda0": LAMBDA0,
            "query_fit_rows": 0, "optimizer_steps": 0,
            "persistent_fp32_sidecar_bytes": 0,
            "fixed_budgets": {
                "max_state_bytes": MAX_STATE_BYTES, "max_build_ns": MAX_BUILD_NS,
                "max_predict_ns": MAX_PREDICT_NS, "max_build_mac": MAX_BUILD_MAC,
                "max_predict_mac": MAX_PREDICT_MAC,
            },
            "build_18_row": {
                "state_count": len(build_ns), "mean_ns": float(np.mean(build_ns)),
                "p95_ns": float(np.percentile(build_ns, 95)),
                "max_build_mac": int(max(build_mac)), "backend": "numpy_cpu",
                "cuda_tensor_count": 0, "peak_vram_bytes": 0,
            },
        },
    }
    packet["packet_sha256"] = _digest(packet)
    truth = {
        "schema": SCHEMA + ".truth.v1", "candidate_revision": CANDIDATE_REVISION,
        "evaluation_scope": SCOPE, "packet_sha256": packet["packet_sha256"],
        "pseudo_new": True, "rows": truths,
    }
    truth["truth_sha256"] = _digest(truth)
    return packet, truth


def _decode_pair(value: Mapping[str, Any]) -> tuple[SVRNBranchState, SVRNBranchState]:
    required = {"raw_wire_b64", "raw_wire_sha256", "svrn_wire_b64", "svrn_wire_sha256", "eta_receipt", "branch_bindings", "resource"}
    if not isinstance(value, Mapping) or set(value) != required:
        raise SVRNBCRFixedHeldError("SVRN-qKNN-BCRR state schema drift")
    wires = {}
    for branch in ("raw", "svrn"):
        try:
            raw = base64.b64decode(value[f"{branch}_wire_b64"], validate=True)
        except Exception as exc:
            raise SVRNBCRFixedHeldError("branch wire base64 drift") from exc
        if _digest(raw) != value[f"{branch}_wire_sha256"]:
            raise SVRNBCRFixedHeldError("branch wire SHA drift")
        wires[branch] = deserialize_branch_state(raw)
    raw, svrn = wires["raw"], wires["svrn"]
    if raw.branch != "raw" or svrn.branch != "svrn" or dict(svrn.eta_receipt or {}) != dict(value["eta_receipt"]):
        raise SVRNBCRFixedHeldError("branch/eta state binding drift")
    if value["branch_bindings"] != _branch_bindings(raw, svrn):
        raise SVRNBCRFixedHeldError("four-arm branch-local SHA binding drift")
    return raw, svrn


def _verify_packet(packet: Mapping[str, Any]) -> None:
    required = {
        "schema", "candidate_revision", "evaluation_scope", "pseudo_new",
        "coverage_sha256", "input_artifact_binding", "held_receiver", "receivers",
        "classes", "K", "scenes", "qknn_lock", "rows", "contract", "packet_sha256",
    }
    if not isinstance(packet, Mapping) or set(packet) != required:
        raise SVRNBCRFixedHeldError("packet schema drift")
    signed = dict(packet); actual = signed.pop("packet_sha256")
    if actual != _digest(signed):
        raise SVRNBCRFixedHeldError("packet SHA drift")
    coverage = _sha_text(packet["coverage_sha256"], "coverage_sha256")
    _artifact_binding(packet["input_artifact_binding"], coverage)
    if (
        packet["schema"] != SCHEMA or packet["candidate_revision"] != CANDIDATE_REVISION
        or packet["evaluation_scope"] != SCOPE or packet["pseudo_new"] is not True
        or packet["held_receiver"] != HELD_RECEIVER or tuple(packet["classes"]) != tuple(REAL_CLASS_IDS)
        or packet["K"] != K or tuple(packet["scenes"]) != SCENES
        or len(packet["receivers"]) != 7 or len(set(packet["receivers"])) != 7
        or packet["held_receiver"] != _coverage_receiver(packet["receivers"], coverage)
        or len(packet["rows"]) != 18
    ):
        raise SVRNBCRFixedHeldError("packet frozen matrix drift")
    qlock = _lock_unwire(packet["qknn_lock"])
    if qlock != _qknn_lock(coverage):
        raise SVRNBCRFixedHeldError("packet fixed qKNN lock drift")
    contract = packet["contract"]
    expected_budgets = {
        "max_state_bytes": MAX_STATE_BYTES, "max_build_ns": MAX_BUILD_NS,
        "max_predict_ns": MAX_PREDICT_NS, "max_build_mac": MAX_BUILD_MAC,
        "max_predict_mac": MAX_PREDICT_MAC,
    }
    if (
        contract.get("arms") != list(ARMS)
        or contract.get("design_commit") != "407144dd714270bd8305595176dfcf921246b75d"
        or contract.get("kappa") != KAPPA
        or tuple(contract.get("eta_grid", ())) != ETA_GRID
        or contract.get("mask_modulus") != MASK_MODULUS
        or tuple(contract.get("mask_residues", ())) != MASK_RESIDUES
        or contract.get("mask_retention") != MASK_RETENTION or contract.get("lambda0") != LAMBDA0
        or contract.get("query_fit_rows") != 0 or contract.get("optimizer_steps") != 0
        or contract.get("persistent_fp32_sidecar_bytes") != 0
        or contract.get("fixed_budgets") != expected_budgets
    ):
        raise SVRNBCRFixedHeldError("packet frozen mechanism contract drift")
    expected_order = [(held, scene) for held in packet["classes"] for scene in SCENES]
    if [(row.get("pseudo_new"), row.get("scene")) for row in packet["rows"]] != expected_order:
        raise SVRNBCRFixedHeldError("packet row order drift")
    for row in packet["rows"]:
        row_required = {"row_id", "pseudo_new", "scene", "old_classes", "support_physical_ids", "support_receipt_sha256", "query_ids", "query_ids_sha256", "c5", "c6"}
        if set(row) != row_required:
            raise SVRNBCRFixedHeldError("packet row schema drift")
        expected_id = _digest({"coverage": coverage, "held_receiver": HELD_RECEIVER, "pseudo_new": row["pseudo_new"], "scene": row["scene"]})
        if (
            row["row_id"] != expected_id or row["query_ids_sha256"] != _digest(row["query_ids"])
            or row["support_receipt_sha256"] != _digest(row["support_physical_ids"])
            or len(row["query_ids"]) != len(set(row["query_ids"]))
            or tuple(row["old_classes"]) != tuple(label for label in packet["classes"] if label != row["pseudo_new"])
            or len(row["support_physical_ids"]["C5"]) != 5 * K
            or len(row["support_physical_ids"]["C6"]) != 6 * K
            or (set(row["support_physical_ids"]["C5"]) | set(row["support_physical_ids"]["C6"])) & set(row["query_ids"])
        ):
            raise SVRNBCRFixedHeldError("packet support/query/class binding drift")
        for stage, classes, key in (("c5", tuple(row["old_classes"]), "C5"), ("c6", tuple(packet["classes"]), "C6")):
            value = row[stage]; raw, svrn = _decode_pair(value)
            expected_support = _support_receipt(row["support_physical_ids"][key], classes)
            if (
                raw.support_receipt_sha256 != expected_support or svrn.support_receipt_sha256 != expected_support
                or value["resource"]["support_receipt_sha256"] != expected_support
                or tuple(deserialize_branch_state(base64.b64decode(value["raw_wire_b64"])).bcrr_receipt["directional_class_loss_qknn"]["0_to_1"]) != classes
                or tuple(deserialize_branch_state(base64.b64decode(value["svrn_wire_b64"])).bcrr_receipt["directional_class_loss_qknn"]["0_to_1"]) != classes
            ):
                raise SVRNBCRFixedHeldError("branch support/class receipt drift")
            resource = value["resource"]; ledger = resource.get("mac_ledger", {})
            if (
                resource.get("eta_receipt_sha256") != value["eta_receipt"].get("receipt_sha256")
                or resource.get("selected_eta") != svrn.eta
                or resource.get("omega_raw") != raw.bcrr_receipt["omega_q"]
                or resource.get("omega_svrn") != svrn.bcrr_receipt["omega_q"]
                or resource.get("persistent_fp32_sidecar_bytes") != 0
                or resource.get("optimizer_steps") != 0
                or resource.get("wire_state_bytes", {}).get("total", MAX_STATE_BYTES + 1) > MAX_STATE_BYTES
                or ledger.get("build_total_mac", MAX_BUILD_MAC + 1) > MAX_BUILD_MAC
                or ledger.get("four_arm_query_per_sample_mac", MAX_PREDICT_MAC + 1) > MAX_PREDICT_MAC
                or ledger.get("bcr_factorizations_per_branch") != 1
                or ledger.get("bcr_loo_full_d3_count") != 0
                or type(resource.get("build_elapsed_ns")) is not int
                or not 0 <= resource["build_elapsed_ns"] <= MAX_BUILD_NS
                or resource.get("backend") != {"name": "numpy_cpu", "cuda_tensor_count": 0, "peak_vram_bytes": 0}
            ):
                raise SVRNBCRFixedHeldError("branch receipt/resource contract drift")
    ns = [row[stage]["resource"]["build_elapsed_ns"] for row in packet["rows"] for stage in ("c5", "c6")]
    mac = [row[stage]["resource"]["mac_ledger"]["build_total_mac"] for row in packet["rows"] for stage in ("c5", "c6")]
    expected_summary = {
        "state_count": 36, "mean_ns": float(np.mean(ns)), "p95_ns": float(np.percentile(ns, 95)),
        "max_build_mac": int(max(mac)), "backend": "numpy_cpu", "cuda_tensor_count": 0, "peak_vram_bytes": 0,
    }
    if contract.get("build_18_row") != expected_summary:
        raise SVRNBCRFixedHeldError("packet build summary drift")


def _neighbor_change_count(raw: Mapping[str, Any], svrn: Mapping[str, Any]) -> int:
    if raw["classes"] != svrn["classes"] or raw["query_count"] != svrn["query_count"]:
        raise SVRNBCRFixedHeldError("qKNN neighbor evidence class/query drift")
    return int(sum(
        raw_order != svrn_order
        for raw_query, svrn_query in zip(raw["orders"], svrn["orders"])
        for raw_order, svrn_order in zip(raw_query, svrn_query)
    ))


def _score_state(value: Mapping[str, Any], query: np.ndarray) -> dict[str, dict[str, Any]]:
    raw, svrn = _decode_pair(value)
    qraw, raw_bcrr = score_branch_logits(raw, query)
    qsvrn, svrn_bcrr = score_branch_logits(svrn, query)
    logits = {"M0": qraw, "M_DA": qsvrn, "M_OTHER": raw_bcrr, "M_JOINT": svrn_bcrr}
    raw_neighbors = dict(qknn_neighbor_receipt(raw, query)); svrn_neighbors = dict(qknn_neighbor_receipt(svrn, query))
    da_neighbor_changes = _neighbor_change_count(raw_neighbors, svrn_neighbors)
    classes = deserialize_branch_state(base64.b64decode(value["raw_wire_b64"])).bcrr_receipt["directional_class_loss_qknn"]["0_to_1"].keys()
    class_list = list(classes)
    out = {}
    for arm in ARMS:
        score = np.asarray(logits[arm], np.float32)
        if score.shape != (len(query), len(class_list)) or not np.isfinite(score).all():
            raise SVRNBCRFixedHeldError("four-arm logit shape/finite drift")
        out[arm] = {
            "classes": class_list, "logits": _encode_array(score),
            "prediction": [class_list[index] for index in np.argmax(score, axis=1).tolist()],
            "neighbor_receipt": dict(raw_neighbors if arm in ("M0", "M_OTHER") else svrn_neighbors),
            "bcrr_neighbor_order_changes": 0,
            "da_neighbor_order_changes": 0 if arm in ("M0", "M_OTHER") else da_neighbor_changes,
        }
    return out


def predict_packet(packet: Mapping[str, Any], query_ids: Sequence[str], query_zid: np.ndarray) -> dict[str, Any]:
    _verify_packet(packet)
    ids = list(map(str, query_ids)); z = np.asarray(query_zid)
    if z.dtype != np.float32 or z.shape != (len(ids), Z_DIM) or len(ids) != len(set(ids)) or not np.isfinite(z).all():
        raise SVRNBCRFixedHeldError("predict accepts only opaque unique IDs and finite float32 z_id")
    lookup = {value: z[index] for index, value in enumerate(ids)}
    rows = []
    for packet_row in packet["rows"]:
        started = time.perf_counter_ns()
        if any(value not in lookup for value in packet_row["query_ids"]):
            raise SVRNBCRFixedHeldError("predict query ID missing")
        query = np.asarray([lookup[value] for value in packet_row["query_ids"]], np.float32)
        row = {
            "row_id": packet_row["row_id"], "query_ids": packet_row["query_ids"],
            "before": _score_state(packet_row["c5"], query),
            "after": _score_state(packet_row["c6"], query),
            "_predict_elapsed_ns": int(time.perf_counter_ns() - started),
        }
        rows.append(row)
    timings = [int(row.pop("_predict_elapsed_ns")) for row in rows]
    if any(value > MAX_PREDICT_NS for value in timings):
        raise SVRNBCRFixedHeldError("fixed prediction time budget exceeded")
    query_mac = sum(
        len(packet["rows"][index]["query_ids"]) * packet["rows"][index]["c6"]["resource"]["mac_ledger"]["four_arm_query_per_sample_mac"]
        for index in range(18)
    )
    prediction = {
        "schema": SCHEMA + ".prediction.v1", "candidate_revision": CANDIDATE_REVISION,
        "evaluation_scope": SCOPE, "packet_sha256": packet["packet_sha256"], "rows": rows,
        "performance": {
            "backend": "numpy_cpu", "cuda_tensor_count": 0, "peak_vram_bytes": 0,
            "row_predict_ns": timings, "mean_ns": float(np.mean(timings)),
            "p95_ns": float(np.percentile(timings, 95)),
            "aggregate_four_arm_mac": int(query_mac), "max_row_budget_ns": MAX_PREDICT_NS,
        },
    }
    prediction["COMMIT"] = _digest(prediction)
    return prediction


def _verify_truth(packet: Mapping[str, Any], truth: Mapping[str, Any], expected: str) -> dict[str, Mapping[str, Any]]:
    signed = dict(truth); actual = signed.pop("truth_sha256", None)
    required = {"schema", "candidate_revision", "evaluation_scope", "packet_sha256", "pseudo_new", "rows", "truth_sha256"}
    if (
        set(truth) != required or actual != expected or actual != _digest(signed)
        or truth["schema"] != SCHEMA + ".truth.v1" or truth["candidate_revision"] != CANDIDATE_REVISION
        or truth["evaluation_scope"] != SCOPE or truth["packet_sha256"] != packet["packet_sha256"]
        or truth["pseudo_new"] is not True or len(truth["rows"]) != 18
    ):
        raise SVRNBCRFixedHeldError("truth seal/schema drift")
    expected_rows = {row["row_id"]: row for row in packet["rows"]}; found = {}
    for row in truth["rows"]:
        if (
            set(row) != {"row_id", "query_labels"} or row["row_id"] in found
            or row["row_id"] not in expected_rows
            or set(row["query_labels"]) != set(expected_rows[row["row_id"]]["query_ids"])
            or any(type(value) is not str or value not in packet["classes"] for value in row["query_labels"].values())
        ):
            raise SVRNBCRFixedHeldError("truth row/key binding drift")
        found[row["row_id"]] = row
    if set(found) != set(expected_rows): raise SVRNBCRFixedHeldError("truth row coverage drift")
    return found


def _verify_prediction(packet: Mapping[str, Any], prediction: Mapping[str, Any], commit: str) -> dict[str, Mapping[str, Any]]:
    signed = dict(prediction); actual = signed.pop("COMMIT", None)
    required = {"schema", "candidate_revision", "evaluation_scope", "packet_sha256", "rows", "performance", "COMMIT"}
    if (
        set(prediction) != required or actual != commit or actual != _digest(signed)
        or prediction["schema"] != SCHEMA + ".prediction.v1"
        or prediction["candidate_revision"] != CANDIDATE_REVISION
        or prediction["evaluation_scope"] != SCOPE or prediction["packet_sha256"] != packet["packet_sha256"]
        or len(prediction["rows"]) != 18
    ):
        raise SVRNBCRFixedHeldError("prediction commit/schema drift")
    perf = prediction["performance"]
    perf_keys = {"backend", "cuda_tensor_count", "peak_vram_bytes", "row_predict_ns", "mean_ns", "p95_ns", "aggregate_four_arm_mac", "max_row_budget_ns"}
    if (
        set(perf) != perf_keys or perf["backend"] != "numpy_cpu" or perf["cuda_tensor_count"] != 0
        or perf["peak_vram_bytes"] != 0 or perf["max_row_budget_ns"] != MAX_PREDICT_NS
        or len(perf["row_predict_ns"]) != 18 or any(type(value) is not int or not 0 <= value <= MAX_PREDICT_NS for value in perf["row_predict_ns"])
        or perf["mean_ns"] != float(np.mean(perf["row_predict_ns"]))
        or perf["p95_ns"] != float(np.percentile(perf["row_predict_ns"], 95))
    ):
        raise SVRNBCRFixedHeldError("prediction resource receipt drift")
    expected_rows = {row["row_id"]: row for row in packet["rows"]}; found = {}
    if [row.get("row_id") for row in prediction["rows"]] != [row["row_id"] for row in packet["rows"]]:
        raise SVRNBCRFixedHeldError("prediction row order drift")
    for row in prediction["rows"]:
        packet_row = expected_rows.get(row.get("row_id"))
        if set(row) != {"row_id", "query_ids", "before", "after"} or packet_row is None or row["row_id"] in found or row["query_ids"] != packet_row["query_ids"]:
            raise SVRNBCRFixedHeldError("prediction row/query binding drift")
        for stage, classes in (("before", packet_row["old_classes"]), ("after", packet["classes"])):
            if tuple(row[stage]) != ARMS: raise SVRNBCRFixedHeldError("prediction four-arm order drift")
            neighbor_by_arm = {}
            for arm in ARMS:
                value = row[stage][arm]
                if set(value) != {"classes", "logits", "prediction", "neighbor_receipt", "bcrr_neighbor_order_changes", "da_neighbor_order_changes"} or value["classes"] != classes:
                    raise SVRNBCRFixedHeldError("prediction arm schema/classes drift")
                logits = _decode_array(value["logits"])
                expected_prediction = [classes[index] for index in np.argmax(logits, axis=1).tolist()]
                if logits.dtype != np.float32 or logits.shape != (len(row["query_ids"]), len(classes)) or not np.isfinite(logits).all() or value["prediction"] != expected_prediction:
                    raise SVRNBCRFixedHeldError("prediction logits/argmax drift")
                neighbor = value["neighbor_receipt"]
                required_neighbor = {"schema", "branch", "qknn_state_receipt_sha256", "qknn_bank_receipt_sha256", "canonical_support_physical_ids_sha256", "classes", "query_count", "orders", "query_rows_used_for_fit", "receipt_sha256"}
                expected_branch = "raw" if arm in ("M0", "M_OTHER") else "svrn"
                expected_state = packet_row["c5" if stage == "before" else "c6"]["branch_bindings"][arm]["qknn_state_receipt_sha256"]
                if not isinstance(neighbor, Mapping) or set(neighbor) != required_neighbor:
                    raise SVRNBCRFixedHeldError("prediction qKNN neighbor evidence schema drift")
                body = {key: neighbor[key] for key in neighbor if key != "receipt_sha256"}
                if (
                    neighbor["schema"] != "cvs.stage2.svrn_bcr.qknn_neighbor_receipt.v1"
                    or neighbor["branch"] != expected_branch
                    or neighbor["qknn_state_receipt_sha256"] != expected_state
                    or neighbor["classes"] != classes or neighbor["query_count"] != len(row["query_ids"])
                    or neighbor["query_rows_used_for_fit"] != 0 or neighbor["receipt_sha256"] != _digest(body)
                    or any(len(per_class) != len(classes) or any(len(order) != K for order in per_class) for per_class in neighbor["orders"])
                    or value["bcrr_neighbor_order_changes"] != 0
                    or type(value["da_neighbor_order_changes"]) is not int or value["da_neighbor_order_changes"] < 0
                ):
                    raise SVRNBCRFixedHeldError("prediction qKNN neighbor evidence drift")
                neighbor_by_arm[arm] = neighbor
            if (
                neighbor_by_arm["M0"] != neighbor_by_arm["M_OTHER"]
                or neighbor_by_arm["M_DA"] != neighbor_by_arm["M_JOINT"]
                or row[stage]["M0"]["da_neighbor_order_changes"] != 0
                or row[stage]["M_OTHER"]["da_neighbor_order_changes"] != 0
                or row[stage]["M_DA"]["da_neighbor_order_changes"] != row[stage]["M_JOINT"]["da_neighbor_order_changes"]
            ):
                raise SVRNBCRFixedHeldError("BCRR/DA qKNN neighbor isolation drift")
        found[row["row_id"]] = row
    expected_mac = sum(len(packet["rows"][i]["query_ids"]) * packet["rows"][i]["c6"]["resource"]["mac_ledger"]["four_arm_query_per_sample_mac"] for i in range(18))
    if perf["aggregate_four_arm_mac"] != expected_mac: raise SVRNBCRFixedHeldError("prediction aggregate MAC drift")
    return found


def _accuracy(prediction: Sequence[str], truth: Sequence[str], classes: Sequence[str]) -> dict[str, float]:
    return {label: float(np.mean([pred == actual for pred, actual in zip(prediction, truth) if actual == label])) for label in classes}


def _transition(base: Sequence[str], candidate: Sequence[str], truth: Sequence[str], held: str) -> dict[str, int]:
    out = {
        "changed": int(sum(a != b for a, b in zip(base, candidate))),
        "wrong_to_correct": int(sum(a != y and b == y for a, b, y in zip(base, candidate, truth))),
        "correct_to_wrong": int(sum(a == y and b != y for a, b, y in zip(base, candidate, truth))),
        "wrong_to_wrong": int(sum(a != y and b != y and a != b for a, b, y in zip(base, candidate, truth))),
    }
    for role, predicate in (("old", lambda y: y != held), ("new", lambda y: y == held)):
        out[f"{role}_wrong_to_correct"] = int(sum(a != y and b == y and predicate(y) for a, b, y in zip(base, candidate, truth)))
        out[f"{role}_correct_to_wrong"] = int(sum(a == y and b != y and predicate(y) for a, b, y in zip(base, candidate, truth)))
    return out


def score_packet(
    packet: Mapping[str, Any], prediction: Mapping[str, Any], truth: Mapping[str, Any],
    *, commit: str, truth_sha256: str,
) -> list[dict[str, Any]]:
    _verify_packet(packet); predicted = _verify_prediction(packet, prediction, commit); truths = _verify_truth(packet, truth, truth_sha256)
    output = []
    for packet_row in packet["rows"]:
        row = predicted[packet_row["row_id"]]; truth_row = truths[packet_row["row_id"]]
        y = [truth_row["query_labels"][value] for value in row["query_ids"]]
        held = packet_row["pseudo_new"]; old = packet_row["old_classes"]
        old_index = [i for i, value in enumerate(y) if value != held]; new_index = [i for i, value in enumerate(y) if value == held]
        quartet = {}
        base_after = row["after"]["M0"]["prediction"]
        mechanism = {
            "c5": {key: packet_row["c5"]["resource"][key] for key in ("selected_eta", "eta_fallback", "omega_raw", "omega_raw_fallback", "omega_svrn", "omega_svrn_fallback")},
            "c6": {key: packet_row["c6"]["resource"][key] for key in ("selected_eta", "eta_fallback", "omega_raw", "omega_raw_fallback", "omega_svrn", "omega_svrn_fallback")},
        }
        for arm in ARMS:
            before = row["before"][arm]["prediction"]; after = row["after"][arm]["prediction"]
            old_before = float(np.mean([before[i] == y[i] for i in old_index])); old_after = float(np.mean([after[i] == y[i] for i in old_index]))
            seen_new = float(np.mean([after[i] == held for i in new_index])); per_class = _accuracy(after, y, packet["classes"])
            harmonic = 0.0 if old_after + seen_new == 0 else float(2 * old_after * seen_new / (old_after + seen_new))
            quartet[arm] = {
                "row_id": row["row_id"], "held_receiver": packet["held_receiver"],
                "pseudo_new": held, "scene": packet_row["scene"], "K": K,
                "selection_coverage_sha256": packet["coverage_sha256"], "arm": arm,
                "query_count": len(y), "old_before": old_before, "old_after": old_after,
                "old_adaptation_gain": old_after - old_before, "seen_new": seen_new,
                "H_old_new": harmonic, "BA": float(np.mean(list(per_class.values()))),
                "floor": float(min(per_class.values())), "min_old": float(min(per_class[label] for label in old)),
                "min_new": float(per_class[held]), "forgetting": old_before - old_after,
                "old_to_new": float(np.mean([after[i] == held for i in old_index])),
                "new_to_old": float(np.mean([after[i] in old for i in new_index])),
                "per_class": per_class, "mechanism": mechanism,
                "transition_vs_M0": _transition(base_after, after, y, held),
                "neighbor_evidence": {
                    "DA_neighbor_order_changes": int(row["after"]["M_DA"]["da_neighbor_order_changes"]),
                    "BCRR_neighbor_order_changes": int(row["after"][arm]["bcrr_neighbor_order_changes"]),
                    "raw_neighbor_receipt_sha256": row["after"]["M0"]["neighbor_receipt"]["receipt_sha256"],
                    "svrn_neighbor_receipt_sha256": row["after"]["M_DA"]["neighbor_receipt"]["receipt_sha256"],
                },
                "resource": {"c5": packet_row["c5"]["resource"], "c6": packet_row["c6"]["resource"]},
            }
        synergy = quartet["M_JOINT"]["H_old_new"] - quartet["M_DA"]["H_old_new"] - quartet["M_OTHER"]["H_old_new"] + quartet["M0"]["H_old_new"]
        for arm in ARMS:
            quartet[arm]["I_syn"] = float(synergy); output.append(quartet[arm])
    if len(output) != 72: raise SVRNBCRFixedHeldError("72 same-row metrics drift")
    return output


def evaluate_stop_gates(metrics: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(metrics) != 72:
        raise SVRNBCRFixedHeldError("stop gate requires 72 metric rows")
    by_arm = {arm: [row for row in metrics if row["arm"] == arm] for arm in ARMS}
    if any(len(rows) != 18 for rows in by_arm.values()): raise SVRNBCRFixedHeldError("stop gate arm coverage drift")
    protection_up = ("old_before", "old_after", "old_adaptation_gain", "seen_new", "H_old_new", "BA", "floor", "min_old", "min_new")
    protection_down = ("forgetting", "old_to_new", "new_to_old")
    keys = (*protection_up, *protection_down)
    means = {arm: {key: float(np.mean([row[key] for row in rows])) for key in keys} for arm, rows in by_arm.items()}
    mean_syn = float(np.mean([row["I_syn"] for row in by_arm["M_JOINT"]]))
    eta_nonzero = any(row["mechanism"]["c6"]["selected_eta"] != 0.0 for row in by_arm["M0"])
    da_changes = sum(row["transition_vs_M0"]["changed"] for row in by_arm["M_DA"])
    other_rescue = sum(row["transition_vs_M0"]["wrong_to_correct"] for row in by_arm["M_OTHER"])
    da_net = sum(row["transition_vs_M0"]["wrong_to_correct"] - row["transition_vs_M0"]["correct_to_wrong"] for row in by_arm["M_DA"])
    da_old_net = sum(row["transition_vs_M0"]["old_wrong_to_correct"] - row["transition_vs_M0"]["old_correct_to_wrong"] for row in by_arm["M_DA"])
    da_new_net = sum(row["transition_vs_M0"]["new_wrong_to_correct"] - row["transition_vs_M0"]["new_correct_to_wrong"] for row in by_arm["M_DA"])
    other_net = sum(row["transition_vs_M0"]["wrong_to_correct"] - row["transition_vs_M0"]["correct_to_wrong"] for row in by_arm["M_OTHER"])
    da_neighbor_changes = sum(row["neighbor_evidence"]["DA_neighbor_order_changes"] for row in by_arm["M_DA"])
    bcrr_neighbor_changes = sum(row["neighbor_evidence"]["BCRR_neighbor_order_changes"] for row in metrics)
    omegas = [row["mechanism"][stage][name] for row in by_arm["M0"] for stage in ("c5", "c6") for name in ("omega_raw", "omega_svrn")]
    component_harm = []
    for arm, base in (("M_DA", "M0"), ("M_OTHER", "M0")):
        for key in protection_up:
            if means[arm][key] < means[base][key]: component_harm.append(f"{arm}:{key}")
        for key in protection_down:
            if means[arm][key] > means[base][key]: component_harm.append(f"{arm}:{key}")
    s18_failures = []
    if not eta_nonzero: s18_failures.append("eta_all_identity")
    if da_changes == 0: s18_failures.append("DA_zero_decision_change")
    if da_net <= 0: s18_failures.append("DA_nonpositive_net_correct")
    if da_old_net < 0 or da_new_net < 0: s18_failures.append("DA_old_or_new_net_negative")
    if not any(omegas): s18_failures.append("BCRR_all_zero")
    if other_rescue == 0 or other_net <= 0: s18_failures.append("OTHER_no_independent_positive_gain")
    if component_harm: s18_failures.append("single_component_protection_harm")
    joint_fail = []
    for comparator in ("M_DA", "M_OTHER"):
        for key in protection_up:
            if means["M_JOINT"][key] < means[comparator][key]: joint_fail.append(f"joint_below_{comparator}:{key}")
        for key in protection_down:
            if means["M_JOINT"][key] > means[comparator][key]: joint_fail.append(f"joint_below_{comparator}:{key}")
    if means["M_JOINT"]["H_old_new"] <= max(means["M_DA"]["H_old_new"], means["M_OTHER"]["H_old_new"]):
        joint_fail.append("joint_H_not_strictly_above_single_components")
    if mean_syn <= 0.0: joint_fail.append("mean_I_syn_not_positive")
    positive_rows = [row for row in by_arm["M_JOINT"] if row["I_syn"] > 0.0]
    positive_classes = sorted({row["pseudo_new"] for row in positive_rows})
    scene_mean_syn = {scene: float(np.mean([row["I_syn"] for row in by_arm["M_JOINT"] if row["scene"] == scene])) for scene in SCENES}
    positive_scenes = sorted(scene for scene, value in scene_mean_syn.items() if value > 0.0)
    if len(positive_rows) < 9: joint_fail.append("joint_positive_slices_below_9")
    if len(positive_scenes) < 2: joint_fail.append("joint_positive_scene_means_below_2_of_3")
    eligible = not s18_failures and not joint_fail
    return {
        "schema": SCHEMA + ".decision.v1", "S18_pass": not s18_failures,
        "S18_failures": s18_failures, "component_harm": component_harm,
        "S19_pass": not joint_fail, "S19_failures": joint_fail,
        "mean_by_arm": means, "mean_I_syn": mean_syn,
        "eta_nonzero_rows": int(sum(row["mechanism"]["c6"]["selected_eta"] != 0.0 for row in by_arm["M0"])),
        "DA_decision_changes": int(da_changes), "DA_net_correct": int(da_net),
        "DA_old_net_correct": int(da_old_net), "DA_new_net_correct": int(da_new_net),
        "OTHER_wrong_to_correct_rescue": int(other_rescue), "OTHER_net_correct": int(other_net),
        "DA_neighbor_order_changes": int(da_neighbor_changes), "BCRR_neighbor_order_changes": int(bcrr_neighbor_changes),
        "positive_joint_classes": positive_classes, "positive_joint_scenes": positive_scenes,
        "positive_I_syn_slice_count": int(len(positive_rows)),
        "scene_mean_I_syn": scene_mean_syn,
        "verdict": "ELIGIBLE_FOR_125_STABILITY_SCREEN" if eligible else "COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="cmd", required=True)
    build = sub.add_parser("build"); [build.add_argument(name, required=True) for name in ("--archive", "--manifest", "--coverage", "--coverage-sha256", "--packet", "--truth", "--query")]
    predict = sub.add_parser("predict"); [predict.add_argument(name, required=True) for name in ("--packet", "--query", "--output")]
    score = sub.add_parser("score"); [score.add_argument(name, required=True) for name in ("--packet", "--prediction", "--truth", "--truth-sha256", "--commit", "--output")]
    args = parser.parse_args()
    if args.cmd == "build":
        archive, binding = _load_archive(args.archive, args.manifest, args.coverage, args.coverage_sha256)
        packet, truth = build_packet(archive, coverage_sha256=args.coverage_sha256, artifact_binding=binding)
        ids, zid = _query_arrays(packet, archive)
        _write_new(args.packet, _canon(packet) + b"\n"); _write_new(args.truth, _canon(truth) + b"\n"); _write_query_new(args.query, ids, zid)
    elif args.cmd == "predict":
        packet = _read_json(args.packet)
        with np.load(args.query, allow_pickle=False) as data:
            if tuple(data.files) != ("query_ids", "z_id"):
                raise SVRNBCRFixedHeldError("query must contain only opaque IDs and z_id")
            prediction = predict_packet(packet, data["query_ids"].astype(str).tolist(), np.asarray(data["z_id"]))
        _write_new(args.output, _canon(prediction) + b"\n")
    else:
        packet = _read_json(args.packet); prediction = _read_json(args.prediction); truth = _read_json(args.truth)
        metrics = score_packet(packet, prediction, truth, commit=args.commit, truth_sha256=args.truth_sha256)
        output = {
            "schema": SCHEMA + ".score.v1", "candidate_revision": CANDIDATE_REVISION,
            "evaluation_scope": SCOPE, "COMMIT": args.commit, "truth_sha256": args.truth_sha256,
            "metrics": metrics, "decision": evaluate_stop_gates(metrics),
        }
        _write_new(args.output, _canon(output) + b"\n")


if __name__ == "__main__":
    main()
