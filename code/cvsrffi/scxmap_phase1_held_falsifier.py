"""Phase1 held-receiver SCXMAP falsifier with sealed prediction artifacts.

This 54-row proxy is development-only and non-promotable.  It holds out one
receiver, rotates each of six Phase1 classes as pseudo-new, covers three weak
LEO scenes and K={1,5,10}, and compares identity Student-t qKNN with the same
head after SCXMAP.  Query truth is stored only in a separately sealed sidecar.
"""

from __future__ import annotations

import argparse
import base64
import dataclasses
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from cvsrffi import r2a_fixed_held_four_arm as r2
from cvsrffi.stage2_scxmap_transform import (
    FittedSCXMapState,
    Phase1SCXMapLock,
    audit_scxmap_resources,
    build_phase1_scxmap_lock,
    fit_scxmap_state,
    transform_scxmap_rows,
)
from cvsrffi.stage2_zid_student_t_qknn import (
    Phase1ZIDStudentTLock,
    build_typed_zid_support_bank,
    deserialize_typed_zid_runtime_state,
    identity_shared_psd_metric,
    score_zid_student_t_logits,
    serialize_typed_zid_runtime_state,
)
from scripts.export_phase1_singleobs_dual_feature_archive import (
    MEMBERS as DUAL_ARCHIVE_MEMBERS,
    _array_sha256 as _exporter_array_sha256,
)


SCOPE = "PHASE1_HELD_PROXY_NON_PROMOTABLE"
SCHEMA = "cvs.phase1.scxmap-held-falsifier.v1"
CANDIDATE = "C-DOM-SCXMAP-D92-GLF/r1"
K_VALUES = (1, 5, 10)
ARMS = ("M0", "M_DA")
SCENES = r2.SCENES
Z_DIM = 160
ROW_COUNT = 6 * len(SCENES) * len(K_VALUES)
BUILD_RECEIPT_SCHEMA = SCHEMA + ".build-receipt.v1"


class SCXMapHeldError(ValueError):
    """Raised when the held proxy artifact closure drifts."""


def _canon(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")


def _sha(value: Any) -> str:
    return hashlib.sha256(value if isinstance(value, bytes) else _canon(value)).hexdigest()


def _sha_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _query_binding(
    query_ids: Sequence[str], query_zid: np.ndarray, query_zdom: np.ndarray
) -> str:
    ids = tuple(str(value) for value in query_ids)
    zid = np.asarray(query_zid)
    zdom = np.asarray(query_zdom)
    if (
        zid.dtype != np.float32
        or zdom.dtype != np.float32
        or zid.shape != (len(ids), Z_DIM)
        or zdom.shape != (len(ids), Z_DIM)
        or len(set(ids)) != len(ids)
        or not np.isfinite(zid).all()
        or not np.isfinite(zdom).all()
    ):
        raise SCXMapHeldError("query binding requires unique paired float32 rows")
    order = np.argsort(np.asarray(ids, dtype=np.str_))
    ordered_ids = [ids[int(index)] for index in order]
    ordered_zid = np.ascontiguousarray(zid[order])
    ordered_zdom = np.ascontiguousarray(zdom[order])
    return _sha(
        {
            "query_ids": ordered_ids,
            "z_id_dtype": ordered_zid.dtype.str,
            "z_id_shape": list(ordered_zid.shape),
            "z_id_sha256": hashlib.sha256(ordered_zid.tobytes()).hexdigest(),
            "z_dom_dtype": ordered_zdom.dtype.str,
            "z_dom_shape": list(ordered_zdom.shape),
            "z_dom_sha256": hashlib.sha256(ordered_zdom.tobytes()).hexdigest(),
        }
    )


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
        raise SCXMapHeldError("JSON root must be an object")
    return value


def _verify_build_receipt(
    receipt: Mapping[str, Any],
    *,
    receipt_file_sha256: str,
    packet: Mapping[str, Any],
    packet_file_sha256: str,
    query_file_sha256: str,
    truth_file_sha256: str | None = None,
) -> None:
    expected_keys = {
        "schema",
        "candidate",
        "evaluation_scope",
        "formal_phase2_eligible",
        "bundle_created",
        "target25_release_authorized",
        "packet_file_sha256",
        "truth_file_sha256",
        "query_file_sha256",
        "packet_sha256",
        "packet_core_sha256",
        "truth_commitment_sha256",
        "query_binding_sha256",
        "receipt_sha256",
    }
    signed = dict(receipt)
    self_digest = signed.pop("receipt_sha256", None)
    if (
        set(receipt) != expected_keys
        or _require_file_sha(receipt_file_sha256, "build receipt file SHA256")
        != receipt_file_sha256
        or receipt.get("schema") != BUILD_RECEIPT_SCHEMA
        or receipt.get("candidate") != CANDIDATE
        or receipt.get("evaluation_scope") != SCOPE
        or receipt.get("formal_phase2_eligible") is not False
        or receipt.get("bundle_created") is not False
        or receipt.get("target25_release_authorized") is not False
        or self_digest != _sha(signed)
        or receipt.get("packet_file_sha256") != packet_file_sha256
        or receipt.get("query_file_sha256") != query_file_sha256
        or (
            truth_file_sha256 is not None
            and receipt.get("truth_file_sha256") != truth_file_sha256
        )
        or receipt.get("packet_sha256") != packet.get("packet_sha256")
        or receipt.get("packet_core_sha256") != packet.get("packet_core_sha256")
        or receipt.get("truth_commitment_sha256")
        != packet.get("truth_commitment_sha256")
        or receipt.get("query_binding_sha256")
        != packet.get("query_binding_sha256")
    ):
        raise SCXMapHeldError("external build receipt binding drift")


def _require_file_sha(value: Any, name: str) -> str:
    if type(value) is not str or len(value) != 64 or value != value.lower():
        raise SCXMapHeldError(f"{name} must be a lowercase SHA256")
    try:
        bytes.fromhex(value)
    except ValueError as exc:
        raise SCXMapHeldError(f"{name} must be hexadecimal") from exc
    return value


def _read_query_file(
    path: str | Path,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as query:
        if tuple(query.files) != ("query_ids", "z_id", "z_dom"):
            raise SCXMapHeldError("query file exact member allowlist drift")
        ids = query["query_ids"].astype(str).tolist()
        zid = np.asarray(query["z_id"])
        zdom = np.asarray(query["z_dom"])
    return ids, zid, zdom


def _receipt(tag: str, payload: Mapping[str, Any]) -> str:
    return _sha({"tag": tag, "payload": payload})


def _qknn_lock(k_shot: int, source: Mapping[str, Any]) -> Phase1ZIDStudentTLock:
    return Phase1ZIDStudentTLock(
        int(k_shot),
        3.0,
        Z_DIM,
        1.0,
        0.2,
        2.0,
        0.5,
        2.0,
        1.0,
        _receipt("scxmap-held-qknn-lodo", source),
        _receipt("scxmap-held-qknn-quant", source),
    )


def _ground_anchors(
    archive: Mapping[str, np.ndarray],
    held_receiver: str,
    ground_classes: Sequence[str],
) -> np.ndarray:
    anchors: list[np.ndarray] = []
    for class_id in ground_classes:
        mask = (archive["receiver_ids"] != held_receiver) & (
            archive["labels"] == class_id
        )
        rows = r2._norm(archive["z_id"][mask])
        if len(rows) < 2:
            raise SCXMapHeldError("ground anchor Phase1 exclusion coverage drift")
        anchor = rows.mean(axis=0)
        norm = float(np.linalg.norm(anchor))
        if not np.isfinite(norm) or norm <= 0.0:
            raise SCXMapHeldError("ground anchor degeneracy")
        anchors.append(np.asarray(anchor / norm, dtype=np.float32))
    return np.stack(anchors).astype(np.float32)


def _phase1_lock(
    archive: Mapping[str, np.ndarray],
    held_receiver: str,
    pseudo_new: str,
    coverage_sha256: str,
) -> tuple[Phase1SCXMapLock, dict[str, Any]]:
    center, scale, vectors, _, cross_audit = r2._cell_lock_inputs(
        archive, held_receiver, pseudo_new
    )
    ground = tuple(
        value for value in sorted(archive["class_ids"].tolist()) if value != pseudo_new
    )
    ground_anchors = _ground_anchors(archive, held_receiver, ground)
    anchor_index = {value: index for index, value in enumerate(ground)}
    phase1_mask = (archive["receiver_ids"] != held_receiver) & np.isin(
        archive["labels"], np.asarray(ground)
    )
    phase1_zdom = archive["z_dom"][phase1_mask].astype(np.float64)
    phase1_zid = r2._norm(archive["z_id"][phase1_mask])
    phase1_labels = archive["labels"][phase1_mask].tolist()
    context = (
        (phase1_zdom - center[None, :]) / scale[None, :]
    ) @ np.asarray(vectors[:4], dtype=np.float64).T
    # Remove every ground class's context mean before fitting the shared map so
    # a TX-only z_dom component cannot masquerade as a receiver correction.
    for class_id in ground:
        mask = np.asarray([value == class_id for value in phase1_labels])
        context[mask] -= context[mask].mean(axis=0, keepdims=True)
    residual = phase1_zid - np.asarray(
        [ground_anchors[anchor_index[value]] for value in phase1_labels],
        dtype=np.float64,
    )
    residual_coordinates = residual @ np.asarray(
        vectors[4:], dtype=np.float64
    ).T
    ridge = 0.05 * len(context)
    with r2.threadpool_limits(limits=1, user_api="blas"):
        cross = np.linalg.solve(
            context.T @ context + ridge * np.eye(4, dtype=np.float64),
            context.T @ residual_coordinates,
        )
        cross_blas = r2._blas_fingerprint()
    if cross.shape != (4, 4) or not np.isfinite(cross).all():
        raise SCXMapHeldError("Phase1 learned cross-map solve drift")
    cross = np.asarray(cross, dtype=np.float32)
    provenance = {
        "coverage_sha256": coverage_sha256,
        "held_receiver": held_receiver,
        "pseudo_new": pseudo_new,
        "ground_classes": list(ground),
        "cross_audit": cross_audit,
        "constants": {
            "rank": 4,
            "ridge_per_row": 0.05,
            "shrink_tau": 6.0,
            "beta_max": 2.0,
            "cross_map": "class_centered_phase1_ridge_4x4",
            "cross_map_ridge": ridge,
            "cross_map_phase1_rows": int(len(context)),
            "cross_map_blas_lapack": cross_blas,
        },
    }
    lock, quant = build_phase1_scxmap_lock(
        ground_classes=ground,
        zdom_center=np.asarray(center, dtype=np.float32),
        zdom_scale=np.asarray(scale, dtype=np.float32),
        receiver_projection=np.asarray(vectors[:4], dtype=np.float32),
        context_to_shift=cross,
        zid_basis=np.asarray(vectors[4:], dtype=np.float32),
        ground_anchors=ground_anchors,
        ridge_per_row=0.05,
        shrink_tau=6.0,
        beta_max=2.0,
        source_receipt_sha256=_receipt("scxmap-held-phase1-lock", provenance),
    )
    replay = lock.ground_anchor_qint8.astype(np.float64) * (
        lock.ground_anchor_scales_fp16.astype(np.float64)[:, None]
    )
    replay /= np.linalg.norm(replay, axis=1, keepdims=True)
    validation_mask = (archive["receiver_ids"] != held_receiver) & np.isin(
        archive["labels"], np.asarray(ground)
    )
    validation = r2._norm(archive["z_id"][validation_mask])
    full_logits = validation @ ground_anchors.astype(np.float64).T
    int8_logits = validation @ replay.T
    full_top1 = np.argmax(full_logits, axis=1)
    int8_top1 = np.argmax(int8_logits, axis=1)
    agreement = float(np.mean(full_top1 == int8_top1))
    partitioned = np.partition(full_logits, -2, axis=1)
    full_margin = partitioned[:, -1] - partitioned[:, -2]
    large_margin_flips = int(
        np.sum((full_margin >= 0.05) & (full_top1 != int8_top1))
    )
    if agreement < 0.995 or large_margin_flips != 0:
        raise SCXMapHeldError("ground anchor INT8 replay promotion gate failed")
    quant["ground_anchor_top1_agreement"] = agreement
    quant["ground_anchor_large_margin_threshold"] = 0.05
    quant["ground_anchor_large_margin_flips"] = large_margin_flips
    quant["ground_anchor_validation_rows"] = int(len(validation))
    cross_replay = lock.context_to_shift_qint8.astype(np.float64) * (
        lock.context_to_shift_scales_fp16.astype(np.float64)[:, None]
    )
    quant["cross_map_replay_max_abs_error"] = float(
        np.max(np.abs(cross_replay - cross.astype(np.float64)))
    )
    quant["cross_map_replay_relative_frobenius_error"] = float(
        np.linalg.norm(cross_replay - cross.astype(np.float64))
        / max(np.linalg.norm(cross.astype(np.float64)), np.finfo(np.float64).tiny)
    )
    return lock, {"cross": cross_audit, "quantization": quant, "provenance": provenance}


def _support_indices(
    archive: Mapping[str, np.ndarray],
    held_receiver: str,
    scene: str,
    classes: Sequence[str],
    k_shot: int,
    coverage_sha256: str,
) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...]]:
    support: list[int] = []
    labels: list[str] = []
    query_ids: list[str] = []
    for class_id in classes:
        indices = np.flatnonzero(
            (archive["receiver_ids"] == held_receiver)
            & (archive["scenario_names"] == scene)
            & (archive["labels"] == class_id)
        )
        ordered = sorted(
            indices.tolist(),
            key=lambda index: hashlib.sha256(
                bytes.fromhex(coverage_sha256)
                + b"\0"
                + scene.encode("utf-8")
                + b"\0"
                + str(archive["physical_ids"][index]).encode("utf-8")
            ).digest(),
        )
        if len(ordered) <= k_shot:
            raise SCXMapHeldError("held cell lacks K support plus query")
        support.extend(ordered[:k_shot])
        labels.extend([class_id] * k_shot)
        query_ids.extend(str(archive["physical_ids"][index]) for index in ordered[k_shot:])
    return (
        np.asarray(support, dtype=np.int64),
        tuple(labels),
        tuple(query_ids),
    )


def _wire_runtime(
    zid: np.ndarray,
    labels: Sequence[str],
    classes: Sequence[str],
    qknn: Phase1ZIDStudentTLock,
) -> tuple[str, str]:
    bank = build_typed_zid_support_bank(zid, labels, classes, config=qknn)
    metric = identity_shared_psd_metric(config=qknn)
    wire = serialize_typed_zid_runtime_state(bank, metric)
    return base64.b64encode(wire).decode("ascii"), _sha(wire)


def _class_tie_tokens(
    archive: Mapping[str, np.ndarray],
    support_indices: np.ndarray,
    support_labels: Sequence[str],
    classes: Sequence[str],
    coverage_sha256: str,
) -> dict[str, str]:
    labels = tuple(str(value) for value in support_labels)
    indices = np.asarray(support_indices)
    if indices.ndim != 1 or len(indices) != len(labels):
        raise SCXMapHeldError("class tie token support alignment drift")
    tokens: dict[str, str] = {}
    for class_id in classes:
        physical_ids = sorted(
            str(archive["physical_ids"][int(index)])
            for index, label in zip(indices.tolist(), labels)
            if label == class_id
        )
        if not physical_ids:
            raise SCXMapHeldError("class tie token lacks support")
        # The display class label is intentionally excluded.  A synchronous
        # class rename therefore carries the same opaque token with the same
        # physical support identity.
        tokens[class_id] = _receipt(
            "scxmap-held-class-tie-token",
            {
                "coverage_sha256": coverage_sha256,
                "support_physical_ids": physical_ids,
            },
        )
    if len(set(tokens.values())) != len(tokens):
        raise SCXMapHeldError("class tie token collision")
    return tokens


def _stable_argmax_predictions(
    logits: np.ndarray,
    classes: Sequence[str],
    class_tie_tokens: Mapping[str, str],
) -> tuple[list[str], int]:
    scores = np.asarray(logits)
    registry = tuple(str(value) for value in classes)
    if (
        scores.ndim != 2
        or scores.shape[1] != len(registry)
        or not np.isfinite(scores).all()
        or set(class_tie_tokens) != set(registry)
    ):
        raise SCXMapHeldError("stable argmax input drift")
    for value in class_tie_tokens.values():
        _require_file_sha(value, "class tie token")
    predictions: list[str] = []
    tie_count = 0
    for row in scores:
        winner_score = np.max(row)
        candidates = np.flatnonzero(row == winner_score).tolist()
        if len(candidates) > 1:
            tie_count += 1
        winner = min(
            candidates,
            key=lambda index: class_tie_tokens[registry[int(index)]],
        )
        predictions.append(registry[int(winner)])
    return predictions, tie_count


def build_packet(
    archive: Mapping[str, Any],
    *,
    coverage_sha256: str,
    artifact_binding: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    a = r2._validate_archive(archive)
    coverage = r2._sha_text(coverage_sha256, "coverage_sha256")
    binding = r2._artifact_binding(artifact_binding, coverage)
    receivers = tuple(sorted(set(a["receiver_ids"].tolist())))
    held_receiver = r2._coverage_receiver(receivers, coverage)
    classes = tuple(sorted(a["class_ids"].tolist()))
    rows: list[dict[str, Any]] = []
    truth_rows: list[dict[str, Any]] = []
    lock_groups: dict[str, Any] = {}
    for pseudo_new in classes:
        lock, lock_audit = _phase1_lock(a, held_receiver, pseudo_new, coverage)
        lock_groups[pseudo_new] = {
            "scxmap": r2._lock_wire(lock),
            "audit": lock_audit,
        }
        old = tuple(value for value in classes if value != pseudo_new)
        for scene in SCENES:
            for k_shot in K_VALUES:
                old_idx, old_labels, _ = _support_indices(
                    a, held_receiver, scene, old, k_shot, coverage
                )
                all_idx, all_labels, query_ids = _support_indices(
                    a, held_receiver, scene, classes, k_shot, coverage
                )
                support_receipt = _receipt(
                    "scxmap-held-old-support",
                    {
                        "physical_ids": [
                            str(a["physical_ids"][index]) for index in old_idx
                        ],
                        "classes": list(old),
                        "K": k_shot,
                    },
                )
                state = fit_scxmap_state(
                    lock,
                    a["z_id"][old_idx].astype(np.float32),
                    a["z_dom"][old_idx].astype(np.float32),
                    old_labels,
                    support_receipt_sha256=support_receipt,
                )
                qknn = _qknn_lock(
                    k_shot,
                    {
                        "coverage_sha256": coverage,
                        "pseudo_new": pseudo_new,
                        "scene": scene,
                        "K": k_shot,
                    },
                )
                m0_before = _wire_runtime(a["z_id"][old_idx], old_labels, old, qknn)
                m0_after = _wire_runtime(
                    a["z_id"][all_idx], all_labels, classes, qknn
                )
                transformed_old = transform_scxmap_rows(
                    lock,
                    state,
                    a["z_id"][old_idx].astype(np.float32),
                    a["z_dom"][old_idx].astype(np.float32),
                )
                transformed_all = transform_scxmap_rows(
                    lock,
                    state,
                    a["z_id"][all_idx].astype(np.float32),
                    a["z_dom"][all_idx].astype(np.float32),
                )
                mda_before = _wire_runtime(transformed_old, old_labels, old, qknn)
                mda_after = _wire_runtime(transformed_all, all_labels, classes, qknn)
                row_id = _sha(
                    {
                        "coverage_sha256": coverage,
                        "held_receiver": held_receiver,
                        "pseudo_new": pseudo_new,
                        "scene": scene,
                        "K": k_shot,
                    }
                )
                rows.append(
                    {
                        "row_id": row_id,
                        "pseudo_new": pseudo_new,
                        "scene": scene,
                        "K": k_shot,
                        "old_classes": list(old),
                        "class_tie_tokens": _class_tie_tokens(
                            a,
                            all_idx,
                            all_labels,
                            classes,
                            coverage,
                        ),
                        "query_ids": list(query_ids),
                        "qknn": r2._lock_wire(qknn),
                        "scxmap_state": dataclasses.asdict(state),
                        "runtime": {
                            "M0_before_b64": m0_before[0],
                            "M0_before_sha256": m0_before[1],
                            "M0_after_b64": m0_after[0],
                            "M0_after_sha256": m0_after[1],
                            "M_DA_before_b64": mda_before[0],
                            "M_DA_before_sha256": mda_before[1],
                            "M_DA_after_b64": mda_after[0],
                            "M_DA_after_sha256": mda_after[1],
                        },
                        "resource": audit_scxmap_resources(lock, state),
                    }
                )
                query_set = set(query_ids)
                truth_rows.append(
                    {
                        "row_id": row_id,
                        "query_labels": {
                            str(a["physical_ids"][index]): str(a["labels"][index])
                            for index in np.flatnonzero(
                                (a["receiver_ids"] == held_receiver)
                                & (a["scenario_names"] == scene)
                            )
                            if str(a["physical_ids"][index]) in query_set
                        },
                    }
                )
    query_ids, query_zid, query_zdom = _query_arrays({"rows": rows}, a)
    packet_core = {
        "schema": SCHEMA,
        "candidate": CANDIDATE,
        "evaluation_scope": SCOPE,
        "formal_phase2_eligible": False,
        "bundle_created": False,
        "target25_release_authorized": False,
        "coverage_sha256": coverage,
        "input_artifact_binding": binding,
        "query_binding_sha256": _query_binding(
            query_ids.tolist(), query_zid, query_zdom
        ),
        "held_receiver": held_receiver,
        "receivers": list(receivers),
        "classes": list(classes),
        "K_values": list(K_VALUES),
        "scenes": list(SCENES),
        "lock_groups": lock_groups,
        "rows": rows,
    }
    packet_core_sha256 = _sha(packet_core)
    truth = {
        "schema": SCHEMA + ".truth.v1",
        "candidate": CANDIDATE,
        "evaluation_scope": SCOPE,
        "packet_core_sha256": packet_core_sha256,
        "rows": truth_rows,
    }
    truth["truth_sha256"] = _sha(truth)
    packet = {
        **packet_core,
        "packet_core_sha256": packet_core_sha256,
        "truth_commitment_sha256": truth["truth_sha256"],
    }
    packet["packet_sha256"] = _sha(packet)
    _verify_packet(packet)
    return packet, truth


def _verify_packet(packet: Mapping[str, Any]) -> None:
    expected_final = dict(packet)
    digest = expected_final.pop("packet_sha256", None)
    expected_core = dict(expected_final)
    core_digest = expected_core.pop("packet_core_sha256", None)
    expected_core.pop("truth_commitment_sha256", None)
    if (
        packet.get("schema") != SCHEMA
        or packet.get("candidate") != CANDIDATE
        or packet.get("evaluation_scope") != SCOPE
        or _sha(expected_final) != digest
        or _sha(expected_core) != core_digest
        or packet.get("formal_phase2_eligible") is not False
        or packet.get("bundle_created") is not False
        or packet.get("target25_release_authorized") is not False
        or type(packet.get("rows")) is not list
        or len(packet["rows"]) != ROW_COUNT
        or packet.get("K_values") != list(K_VALUES)
        or packet.get("scenes") != list(SCENES)
    ):
        raise SCXMapHeldError("SCXMAP held packet schema/receipt/row-count drift")
    for name in (
        "packet_sha256",
        "packet_core_sha256",
        "truth_commitment_sha256",
        "query_binding_sha256",
        "coverage_sha256",
    ):
        _require_file_sha(packet.get(name), f"packet {name}")
    row_ids = [row.get("row_id") for row in packet["rows"]]
    expected_cells = {
        (pseudo_new, scene, k_shot)
        for pseudo_new in packet["classes"]
        for scene in SCENES
        for k_shot in K_VALUES
    }
    observed_cells = {
        (row.get("pseudo_new"), row.get("scene"), row.get("K"))
        for row in packet["rows"]
    }
    if (
        len(set(row_ids)) != ROW_COUNT
        or observed_cells != expected_cells
        or any(
            row.get("old_classes")
            != [value for value in packet["classes"] if value != row.get("pseudo_new")]
            or not isinstance(row.get("class_tie_tokens"), Mapping)
            or set(row["class_tie_tokens"]) != set(packet["classes"])
            or len(set(row["class_tie_tokens"].values())) != len(packet["classes"])
            or type(row.get("query_ids")) is not list
            or len(row["query_ids"]) != len(set(row["query_ids"]))
            for row in packet["rows"]
        )
    ):
        raise SCXMapHeldError("SCXMAP held row bijection drift")


def _runtime(packet_row: Mapping[str, Any], arm: str, stage: str):
    runtime = packet_row["runtime"]
    prefix = f"{arm}_{stage}"
    wire = base64.b64decode(str(runtime[prefix + "_b64"]), validate=True)
    if _sha(wire) != runtime[prefix + "_sha256"]:
        raise SCXMapHeldError("SCXMAP held runtime wire receipt drift")
    return deserialize_typed_zid_runtime_state(wire)


def predict_packet(
    packet: Mapping[str, Any],
    query_ids: Sequence[str],
    query_zid: np.ndarray,
    query_zdom: np.ndarray,
) -> dict[str, Any]:
    _verify_packet(packet)
    ids = tuple(str(value) for value in query_ids)
    zid = np.asarray(query_zid)
    zdom = np.asarray(query_zdom)
    if (
        zid.dtype != np.float32
        or zdom.dtype != np.float32
        or zid.shape != (len(ids), Z_DIM)
        or zdom.shape != (len(ids), Z_DIM)
        or len(set(ids)) != len(ids)
        or not np.isfinite(zid).all()
        or not np.isfinite(zdom).all()
    ):
        raise SCXMapHeldError("predict requires unique IDs and finite paired float32 z_id/z_dom")
    if _query_binding(ids, zid, zdom) != packet["query_binding_sha256"]:
        raise SCXMapHeldError("query feature bytes are not bound to the build packet")
    lookup = {value: index for index, value in enumerate(ids)}
    output_rows: list[dict[str, Any]] = []
    for row in packet["rows"]:
        wanted = row["query_ids"]
        if any(value not in lookup for value in wanted):
            raise SCXMapHeldError("query feature set misses packet IDs")
        indices = np.asarray([lookup[value] for value in wanted], dtype=np.int64)
        qzid, qzdom = zid[indices], zdom[indices]
        lock = r2._lock_unwire(
            packet["lock_groups"][row["pseudo_new"]]["scxmap"], Phase1SCXMapLock
        )
        state = FittedSCXMapState(**row["scxmap_state"])
        transformed = transform_scxmap_rows(lock, state, qzid, qzdom)
        stages: dict[str, Any] = {}
        for stage in ("before", "after"):
            arms: dict[str, Any] = {}
            for arm, features in (("M0", qzid), ("M_DA", transformed)):
                bank, metric = _runtime(row, arm, stage)
                logits = score_zid_student_t_logits(bank, features, metric=metric)
                classes = list(bank.classes)
                prediction, tie_count = _stable_argmax_predictions(
                    logits,
                    classes,
                    {
                        class_id: row["class_tie_tokens"][class_id]
                        for class_id in classes
                    },
                )
                arms[arm] = {
                    "classes": classes,
                    "prediction": prediction,
                    "logits": r2._encode_array(logits),
                    "top_score_tie_rows": tie_count,
                }
            stages[stage] = arms
        output_rows.append(
            {
                "row_id": row["row_id"],
                "query_ids": wanted,
                "before": stages["before"],
                "after": stages["after"],
            }
        )
    result = {
        "schema": SCHEMA + ".prediction.v1",
        "candidate": CANDIDATE,
        "evaluation_scope": SCOPE,
        "packet_sha256": packet["packet_sha256"],
        "rows": output_rows,
    }
    result["COMMIT"] = _sha(result)
    return result


def score_packet(
    packet: Mapping[str, Any],
    prediction: Mapping[str, Any],
    truth: Mapping[str, Any],
    *,
    commit: str,
    truth_sha256: str,
) -> dict[str, Any]:
    _verify_packet(packet)
    pred_signed = dict(prediction)
    actual_commit = pred_signed.pop("COMMIT", None)
    truth_signed = dict(truth)
    actual_truth = truth_signed.pop("truth_sha256", None)
    if (
        set(prediction)
        != {
            "schema",
            "candidate",
            "evaluation_scope",
            "packet_sha256",
            "rows",
            "COMMIT",
        }
        or prediction.get("schema") != SCHEMA + ".prediction.v1"
        or prediction.get("candidate") != CANDIDATE
        or prediction.get("evaluation_scope") != SCOPE
        or set(truth)
        != {
            "schema",
            "candidate",
            "evaluation_scope",
            "packet_core_sha256",
            "rows",
            "truth_sha256",
        }
        or truth.get("schema") != SCHEMA + ".truth.v1"
        or truth.get("candidate") != CANDIDATE
        or truth.get("evaluation_scope") != SCOPE
        or actual_commit != commit
        or _sha(pred_signed) != commit
        or actual_truth != truth_sha256
        or _sha(truth_signed) != truth_sha256
        or truth_sha256 != packet["truth_commitment_sha256"]
        or prediction.get("packet_sha256") != packet["packet_sha256"]
        or truth.get("packet_core_sha256") != packet["packet_core_sha256"]
        or len(prediction.get("rows", [])) != ROW_COUNT
        or len(truth.get("rows", [])) != ROW_COUNT
    ):
        raise SCXMapHeldError("prediction/truth seal or row-count drift")
    metrics: list[dict[str, Any]] = []
    aggregates: dict[int, dict[str, Any]] = {}
    for packet_row, pred_row, truth_row in zip(
        packet["rows"], prediction["rows"], truth["rows"]
    ):
        if (
            set(pred_row) != {"row_id", "query_ids", "before", "after"}
            or set(truth_row) != {"row_id", "query_labels"}
            or pred_row.get("row_id") != packet_row["row_id"]
            or truth_row.get("row_id") != packet_row["row_id"]
            or pred_row.get("query_ids") != packet_row["query_ids"]
            or set(truth_row.get("query_labels", {})) != set(packet_row["query_ids"])
            or any(
                type(value) is not str or value not in packet["classes"]
                for value in truth_row.get("query_labels", {}).values()
            )
            or set(truth_row.get("query_labels", {}).values())
            != set(packet["classes"])
        ):
            raise SCXMapHeldError("prediction/truth row identity drift")
        query = packet_row["query_ids"]
        y = [truth_row["query_labels"][value] for value in query]
        pseudo_new = packet_row["pseudo_new"]
        old_mask = np.asarray([value != pseudo_new for value in y])
        new_mask = ~old_mask
        row_metrics: dict[str, Any] = {}
        if tuple(pred_row["before"]) != ARMS or tuple(pred_row["after"]) != ARMS:
            raise SCXMapHeldError("prediction arm order/schema drift")
        for arm in ARMS:
            for stage, expected_classes in (
                ("before", packet_row["old_classes"]),
                ("after", packet["classes"]),
            ):
                payload = pred_row[stage].get(arm)
                if (
                    not isinstance(payload, Mapping)
                    or set(payload)
                    != {
                        "classes",
                        "prediction",
                        "logits",
                        "top_score_tie_rows",
                    }
                    or payload.get("classes") != expected_classes
                    or type(payload.get("prediction")) is not list
                    or len(payload["prediction"]) != len(query)
                    or any(
                        type(value) is not str or value not in expected_classes
                        for value in payload["prediction"]
                    )
                    or not isinstance(payload.get("logits"), Mapping)
                    or type(payload.get("top_score_tie_rows")) is not int
                    or payload["top_score_tie_rows"] < 0
                ):
                    raise SCXMapHeldError("prediction arm payload drift")
                try:
                    logits = r2._decode_array(payload["logits"])
                except (ValueError, TypeError, KeyError) as exc:
                    raise SCXMapHeldError("prediction logits encoding drift") from exc
                if (
                    logits.dtype not in (np.dtype("float32"), np.dtype("float64"))
                    or logits.shape != (len(query), len(expected_classes))
                    or not np.isfinite(logits).all()
                ):
                    raise SCXMapHeldError("prediction logits shape/dtype/finite drift")
                recomputed, recomputed_tie_count = _stable_argmax_predictions(
                    logits,
                    expected_classes,
                    {
                        class_id: packet_row["class_tie_tokens"][class_id]
                        for class_id in expected_classes
                    },
                )
                if (
                    payload["prediction"] != recomputed
                    or payload["top_score_tie_rows"] != recomputed_tie_count
                ):
                    raise SCXMapHeldError("prediction argmax/logit binding drift")
            before = pred_row["before"][arm]["prediction"]
            after = pred_row["after"][arm]["prediction"]
            old_before = float(
                np.mean([before[index] == y[index] for index in np.flatnonzero(old_mask)])
            )
            old_after = float(
                np.mean([after[index] == y[index] for index in np.flatnonzero(old_mask)])
            )
            seen_new = float(
                np.mean([after[index] == y[index] for index in np.flatnonzero(new_mask)])
            )
            per_class = {
                class_id: float(
                    np.mean(
                        [
                            after[index] == y[index]
                            for index in range(len(y))
                            if y[index] == class_id
                        ]
                    )
                )
                for class_id in packet["classes"]
            }
            h_score = (
                0.0
                if old_after + seen_new == 0.0
                else 2.0 * old_after * seen_new / (old_after + seen_new)
            )
            row_metrics[arm] = {
                "old_before": old_before,
                "old_after": old_after,
                "seen_new": seen_new,
                "H_old_new": h_score,
                "floor": min(per_class.values()),
                "per_class": per_class,
            }
        m0 = pred_row["after"]["M0"]["prediction"]
        mda = pred_row["after"]["M_DA"]["prediction"]
        wrong_to_correct = sum(
            left != target and right == target
            for left, right, target in zip(m0, mda, y)
        )
        correct_to_wrong = sum(
            left == target and right != target
            for left, right, target in zip(m0, mda, y)
        )
        changes = sum(left != right for left, right in zip(m0, mda))
        record = {
            "row_id": packet_row["row_id"],
            "pseudo_new": pseudo_new,
            "scene": packet_row["scene"],
            "K": packet_row["K"],
            "arms": row_metrics,
            "argmax_changes": changes,
            "wrong_to_correct": wrong_to_correct,
            "correct_to_wrong": correct_to_wrong,
            "beta_fp32": packet_row["scxmap_state"]["beta_fp32"],
            "resource": packet_row["resource"],
        }
        metrics.append(record)
        summary = aggregates.setdefault(
            int(packet_row["K"]),
            {
                "rows": 0,
                "query_count": 0,
                "argmax_changes": 0,
                "wrong_to_correct": 0,
                "correct_to_wrong": 0,
                "M0_old_after_sum": 0.0,
                "M_DA_old_after_sum": 0.0,
                "M0_seen_new_sum": 0.0,
                "M_DA_seen_new_sum": 0.0,
                "M0_H_sum": 0.0,
                "M_DA_H_sum": 0.0,
                "active_beta_rows": 0,
            },
        )
        summary["rows"] += 1
        summary["query_count"] += len(y)
        summary["argmax_changes"] += changes
        summary["wrong_to_correct"] += wrong_to_correct
        summary["correct_to_wrong"] += correct_to_wrong
        summary["M0_old_after_sum"] += row_metrics["M0"]["old_after"]
        summary["M_DA_old_after_sum"] += row_metrics["M_DA"]["old_after"]
        summary["M0_seen_new_sum"] += row_metrics["M0"]["seen_new"]
        summary["M_DA_seen_new_sum"] += row_metrics["M_DA"]["seen_new"]
        summary["M0_H_sum"] += row_metrics["M0"]["H_old_new"]
        summary["M_DA_H_sum"] += row_metrics["M_DA"]["H_old_new"]
        summary["active_beta_rows"] += int(record["beta_fp32"] > 0.0)
    summaries: list[dict[str, Any]] = []
    for k_shot in K_VALUES:
        item = aggregates[k_shot]
        rows = item["rows"]
        summary = {
            "K": k_shot,
            **{key: value for key, value in item.items() if not key.endswith("_sum")},
            "M0_old_after": item["M0_old_after_sum"] / rows,
            "M_DA_old_after": item["M_DA_old_after_sum"] / rows,
            "M0_seen_new": item["M0_seen_new_sum"] / rows,
            "M_DA_seen_new": item["M_DA_seen_new_sum"] / rows,
            "M0_H_old_new": item["M0_H_sum"] / rows,
            "M_DA_H_old_new": item["M_DA_H_sum"] / rows,
        }
        summary["old_delta"] = summary["M_DA_old_after"] - summary["M0_old_after"]
        summary["new_delta"] = summary["M_DA_seen_new"] - summary["M0_seen_new"]
        summary["H_delta"] = summary["M_DA_H_old_new"] - summary["M0_H_old_new"]
        summary["gate_pass"] = bool(
            summary["argmax_changes"] > 0
            and summary["wrong_to_correct"] > summary["correct_to_wrong"]
            and summary["old_delta"] >= 0.0
            and summary["new_delta"] >= 0.0
            and summary["H_delta"] > 0.0
        )
        summaries.append(summary)

    def stratify(fields: tuple[str, ...]) -> list[dict[str, Any]]:
        grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        for record in metrics:
            grouped.setdefault(tuple(record[field] for field in fields), []).append(
                record
            )
        output: list[dict[str, Any]] = []
        for key in sorted(grouped):
            records = grouped[key]
            rows = len(records)
            item: dict[str, Any] = {
                field: value for field, value in zip(fields, key)
            }
            item.update(
                {
                    "rows": rows,
                    "argmax_changes": sum(
                        int(record["argmax_changes"]) for record in records
                    ),
                    "wrong_to_correct": sum(
                        int(record["wrong_to_correct"]) for record in records
                    ),
                    "correct_to_wrong": sum(
                        int(record["correct_to_wrong"]) for record in records
                    ),
                    "old_delta": float(
                        np.mean(
                            [
                                record["arms"]["M_DA"]["old_after"]
                                - record["arms"]["M0"]["old_after"]
                                for record in records
                            ]
                        )
                    ),
                    "new_delta": float(
                        np.mean(
                            [
                                record["arms"]["M_DA"]["seen_new"]
                                - record["arms"]["M0"]["seen_new"]
                                for record in records
                            ]
                        )
                    ),
                    "H_delta": float(
                        np.mean(
                            [
                                record["arms"]["M_DA"]["H_old_new"]
                                - record["arms"]["M0"]["H_old_new"]
                                for record in records
                            ]
                        )
                    ),
                }
            )
            item["gate_pass"] = bool(
                item["argmax_changes"] > 0
                and item["wrong_to_correct"] > item["correct_to_wrong"]
                and item["old_delta"] >= 0.0
                and item["new_delta"] >= 0.0
                and item["H_delta"] > 0.0
            )
            output.append(item)
        return output

    summary_by_K_scene = stratify(("K", "scene"))
    summary_by_K_pseudo_new = stratify(("K", "pseudo_new"))
    all_strata_gate_pass = all(
        item["gate_pass"]
        for item in summary_by_K_scene + summary_by_K_pseudo_new
    )
    return {
        "schema": SCHEMA + ".score.v1",
        "candidate": CANDIDATE,
        "evaluation_scope": SCOPE,
        "formal_phase2_eligible": False,
        "bundle_created": False,
        "target25_release_authorized": False,
        "target25_blocked_reason": (
            "PHASE1_HELD_PROXY_REQUIRES_INDEPENDENT_REVIEW_AND_SEPARATE_"
            "TARGET25_PREREGISTRATION"
        ),
        "packet_sha256": packet["packet_sha256"],
        "COMMIT": commit,
        "truth_sha256": truth_sha256,
        "metrics": metrics,
        "summary_by_K": summaries,
        "summary_by_K_scene": summary_by_K_scene,
        "summary_by_K_pseudo_new": summary_by_K_pseudo_new,
        "all_K_gate_pass": all(item["gate_pass"] for item in summaries),
        "all_strata_gate_pass": all_strata_gate_pass,
        "proxy_gate_pass": bool(
            all(item["gate_pass"] for item in summaries)
            and all_strata_gate_pass
        ),
    }


def _query_arrays(
    packet: Mapping[str, Any], archive: Mapping[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    wanted = sorted({value for row in packet["rows"] for value in row["query_ids"]})
    index = {str(value): i for i, value in enumerate(archive["physical_ids"].tolist())}
    if any(value not in index for value in wanted):
        raise SCXMapHeldError("packet query IDs missing from archive")
    return (
        np.asarray(wanted, dtype=np.str_),
        np.asarray([archive["z_id"][index[value]] for value in wanted], dtype=np.float32),
        np.asarray([archive["z_dom"][index[value]] for value in wanted], dtype=np.float32),
    )


def _write_query_new(
    path: str | Path, query_ids: np.ndarray, z_id: np.ndarray, z_dom: np.ndarray
) -> None:
    target = Path(path)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"refusing to overwrite {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as handle:
        np.savez_compressed(handle, query_ids=query_ids, z_id=z_id, z_dom=z_dom)
        handle.flush()
        os.fsync(handle.fileno())


def real_checkpoint_support_only_smoke(
    *,
    archive_path: str | Path,
    manifest_path: str | Path,
    coverage_path: str | Path,
    coverage_sha256: str,
    checkpoint_path: str | Path,
    checkpoint_sha256: str,
) -> dict[str, Any]:
    """Exercise SCXMAP on real support without reading query rows or truth."""

    paths = {
        "archive": Path(archive_path),
        "manifest": Path(manifest_path),
        "coverage": Path(coverage_path),
        "checkpoint": Path(checkpoint_path),
    }
    for name, value in paths.items():
        if value.is_symlink() or not value.is_file():
            raise SCXMapHeldError(f"support smoke {name} must be a regular file")
        paths[name] = value.resolve()
    expected_checkpoint = _require_file_sha(
        checkpoint_sha256, "support smoke checkpoint SHA256"
    )
    expected_coverage = _require_file_sha(
        coverage_sha256, "support smoke coverage SHA256"
    )
    if (
        _sha_file(paths["checkpoint"]) != expected_checkpoint
        or _sha_file(paths["coverage"]) != expected_coverage
    ):
        raise SCXMapHeldError("support smoke checkpoint/coverage path SHA drift")
    manifest = _read_json(paths["manifest"])
    access = manifest.get("access_audit")
    inputs = manifest.get("inputs")
    if (
        manifest.get("schema") != r2.DUAL_ARCHIVE_SCHEMA
        or manifest.get("status") != "DEVELOPMENT_ONLY_NOT_FORMAL"
        or manifest.get("artifact_stage") != "phase1_offline_before_target_access"
        or manifest.get("formal_phase2_eligible") is not False
        or manifest.get("bundle_created") is not False
        or manifest.get("exact_member_allowlist") != list(DUAL_ARCHIVE_MEMBERS)
        or not isinstance(access, Mapping)
        or access.get("query_access") is not False
        or access.get("target_access") is not False
        or access.get("clean_iq_access") is not False
        or not isinstance(inputs, Mapping)
        or inputs.get("checkpoint_sha256") != expected_checkpoint
        or manifest.get("artifact", {}).get("sha256")
        != _sha_file(paths["archive"])
    ):
        raise SCXMapHeldError("support smoke archive manifest lineage drift")
    with np.load(paths["archive"], allow_pickle=False) as loaded:
        if tuple(loaded.files) != DUAL_ARCHIVE_MEMBERS:
            raise SCXMapHeldError("support smoke archive member order drift")
        all_arrays = {name: np.asarray(loaded[name]) for name in loaded.files}
    declared_arrays = manifest.get("array_sha256")
    if (
        not isinstance(declared_arrays, Mapping)
        or set(declared_arrays) != set(DUAL_ARCHIVE_MEMBERS)
        or any(
            declared_arrays[name] != _exporter_array_sha256(value)
            for name, value in all_arrays.items()
        )
    ):
        raise SCXMapHeldError("support smoke archive array SHA drift")
    archive = r2._validate_archive(all_arrays)
    binding = r2._validate_coverage_receipt(
        _read_json(paths["coverage"]),
        archive_sha256=_sha_file(paths["archive"]),
        manifest_sha256=_sha_file(paths["manifest"]),
        coverage_sha256=expected_coverage,
    )

    # weights_only plus an explicit project enum allowlist avoids arbitrary
    # checkpoint unpickling while still proving the real checkpoint is readable.
    import torch
    from baseline_origin_sat_view import SatViewStage

    with torch.serialization.safe_globals([SatViewStage]):
        checkpoint = torch.load(
            paths["checkpoint"], map_location="cpu", weights_only=True
        )
    if (
        not isinstance(checkpoint, Mapping)
        or not isinstance(checkpoint.get("model"), Mapping)
        or not isinstance(checkpoint.get("ema_model"), Mapping)
        or not checkpoint["model"]
        or set(checkpoint["model"]) != set(checkpoint["ema_model"])
    ):
        raise SCXMapHeldError("support smoke checkpoint state schema drift")
    tensor_count = 0
    parameter_count = 0
    for state_name in ("model", "ema_model"):
        for value in checkpoint[state_name].values():
            if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
                raise SCXMapHeldError("support smoke checkpoint tensor drift")
            tensor_count += 1
            parameter_count += int(value.numel())

    receivers = tuple(sorted(set(archive["receiver_ids"].tolist())))
    held_receiver = r2._coverage_receiver(receivers, expected_coverage)
    classes = tuple(sorted(archive["class_ids"].tolist()))
    pseudo_new = classes[0]
    old = tuple(value for value in classes if value != pseudo_new)
    scene = SCENES[0]
    k_shot = 1
    support_idx, support_labels, _ = _support_indices(
        archive, held_receiver, scene, old, k_shot, expected_coverage
    )
    support_ids = [str(archive["physical_ids"][index]) for index in support_idx]
    lock, lock_audit = _phase1_lock(
        archive, held_receiver, pseudo_new, expected_coverage
    )
    support_receipt = _receipt(
        "scxmap-real-checkpoint-support-only-smoke",
        {
            "checkpoint_sha256": expected_checkpoint,
            "physical_ids": support_ids,
            "classes": list(old),
            "K": k_shot,
        },
    )
    state = fit_scxmap_state(
        lock,
        archive["z_id"][support_idx].astype(np.float32),
        archive["z_dom"][support_idx].astype(np.float32),
        support_labels,
        support_receipt_sha256=support_receipt,
    )
    transformed = transform_scxmap_rows(
        lock,
        state,
        archive["z_id"][support_idx].astype(np.float32),
        archive["z_dom"][support_idx].astype(np.float32),
    )
    if transformed.shape != (len(old), Z_DIM) or not np.isfinite(transformed).all():
        raise SCXMapHeldError("support smoke transform closure drift")
    result = {
        "schema": SCHEMA + ".real-support-smoke.v1",
        "candidate": CANDIDATE,
        "evaluation_scope": "LOCAL_REAL_CHECKPOINT_SUPPORT_ONLY_NO_QUERY",
        "status": "PASS",
        "formal_phase2_eligible": False,
        "bundle_created": False,
        "target25_release_authorized": False,
        "query_access": False,
        "query_rows_read": 0,
        "truth_access": False,
        "checkpoint_weights_only": True,
        "checkpoint_sha256": expected_checkpoint,
        "checkpoint_tensor_count": tensor_count,
        "checkpoint_parameter_count_model_plus_ema": parameter_count,
        "input_artifact_binding": binding,
        "archive_array_sha_verified": True,
        "live_export_runtime_contract_invoked": False,
        "live_export_runtime_contract_note": (
            "support-only smoke consumes sealed archive features and does not "
            "re-extract them"
        ),
        "held_receiver": held_receiver,
        "pseudo_new": pseudo_new,
        "scene": scene,
        "K": k_shot,
        "support_rows": len(support_ids),
        "support_physical_ids_sha256": _sha(support_ids),
        "support_receipt_sha256": support_receipt,
        "state_receipt_sha256": state.state_receipt_sha256,
        "transformed_support_sha256": hashlib.sha256(
            np.ascontiguousarray(transformed).tobytes()
        ).hexdigest(),
        "lock_audit": lock_audit,
        "resource": audit_scxmap_resources(lock, state),
    }
    result["receipt_sha256"] = _sha(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    build = sub.add_parser("build")
    for name in (
        "archive",
        "manifest",
        "coverage",
        "coverage-sha256",
        "packet",
        "truth",
        "query",
        "build-receipt",
    ):
        build.add_argument("--" + name, required=True)
    predict = sub.add_parser("predict")
    for name in (
        "packet",
        "query",
        "build-receipt",
        "build-receipt-sha256",
        "output",
    ):
        predict.add_argument("--" + name, required=True)
    score = sub.add_parser("score")
    for name in (
        "packet",
        "prediction",
        "truth",
        "query",
        "build-receipt",
        "build-receipt-sha256",
        "truth-sha256",
        "commit",
        "output",
    ):
        score.add_argument("--" + name, required=True)
    smoke = sub.add_parser("support-smoke")
    for name in (
        "archive",
        "manifest",
        "coverage",
        "coverage-sha256",
        "checkpoint",
        "checkpoint-sha256",
        "output",
    ):
        smoke.add_argument("--" + name, required=True)
    args = parser.parse_args()
    if args.cmd == "build":
        archive, binding = r2._load_archive(
            args.archive, args.manifest, args.coverage, args.coverage_sha256
        )
        packet, truth = build_packet(
            archive,
            coverage_sha256=args.coverage_sha256,
            artifact_binding=binding,
        )
        ids, zid, zdom = _query_arrays(packet, archive)
        _write_new(args.packet, _canon(packet) + b"\n")
        _write_new(args.truth, _canon(truth) + b"\n")
        _write_query_new(args.query, ids, zid, zdom)
        receipt = {
            "schema": BUILD_RECEIPT_SCHEMA,
            "candidate": CANDIDATE,
            "evaluation_scope": SCOPE,
            "formal_phase2_eligible": False,
            "bundle_created": False,
            "target25_release_authorized": False,
            "packet_file_sha256": _sha_file(args.packet),
            "truth_file_sha256": _sha_file(args.truth),
            "query_file_sha256": _sha_file(args.query),
            "packet_sha256": packet["packet_sha256"],
            "packet_core_sha256": packet["packet_core_sha256"],
            "truth_commitment_sha256": packet["truth_commitment_sha256"],
            "query_binding_sha256": packet["query_binding_sha256"],
        }
        receipt["receipt_sha256"] = _sha(receipt)
        _write_new(args.build_receipt, _canon(receipt) + b"\n")
    elif args.cmd == "predict":
        packet = _read_json(args.packet)
        receipt = _read_json(args.build_receipt)
        if _sha_file(args.build_receipt) != args.build_receipt_sha256:
            raise SCXMapHeldError("build receipt file SHA256 drift")
        _verify_build_receipt(
            receipt,
            receipt_file_sha256=args.build_receipt_sha256,
            packet=packet,
            packet_file_sha256=_sha_file(args.packet),
            query_file_sha256=_sha_file(args.query),
        )
        ids, zid, zdom = _read_query_file(args.query)
        prediction = predict_packet(packet, ids, zid, zdom)
        _write_new(args.output, _canon(prediction) + b"\n")
    elif args.cmd == "score":
        packet = _read_json(args.packet)
        truth = _read_json(args.truth)
        receipt = _read_json(args.build_receipt)
        if _sha_file(args.build_receipt) != args.build_receipt_sha256:
            raise SCXMapHeldError("build receipt file SHA256 drift")
        _verify_build_receipt(
            receipt,
            receipt_file_sha256=args.build_receipt_sha256,
            packet=packet,
            packet_file_sha256=_sha_file(args.packet),
            query_file_sha256=_sha_file(args.query),
            truth_file_sha256=_sha_file(args.truth),
        )
        ids, zid, zdom = _read_query_file(args.query)
        if _query_binding(ids, zid, zdom) != packet["query_binding_sha256"]:
            raise SCXMapHeldError("score query feature bytes drift")
        result = score_packet(
            packet,
            _read_json(args.prediction),
            truth,
            commit=args.commit,
            truth_sha256=args.truth_sha256,
        )
        _write_new(args.output, _canon(result) + b"\n")
    else:
        result = real_checkpoint_support_only_smoke(
            archive_path=args.archive,
            manifest_path=args.manifest,
            coverage_path=args.coverage,
            coverage_sha256=args.coverage_sha256,
            checkpoint_path=args.checkpoint,
            checkpoint_sha256=args.checkpoint_sha256,
        )
        _write_new(args.output, _canon(result) + b"\n")


if __name__ == "__main__":
    main()
