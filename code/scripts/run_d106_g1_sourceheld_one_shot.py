#!/usr/bin/env python3
"""Run the fixed D106 G1 source-held quartet, then score in a separate call.

``predict`` never accepts or reads truth. ``score`` consumes only an already
sealed 63-row prediction root and the independently published D104 truth file.
This is source-held evidence, never Target25/P2 formal-runner authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cvsrffi import stage2_d106_rcmr_g0 as g0  # noqa: E402
from cvsrffi import stage2_d106_rdce_asset as rdce_asset  # noqa: E402
from cvsrffi.stage2_d106_rdce_runtime import (  # noqa: E402
    MAX_ATTENUATION_FP16,
    MIN_ATTENUATION_FP16,
)
from cvsrffi.stage2_d106_rdce_asset import (  # noqa: E402
    D106RDCEAssetLineage,
    decode_d106_rdce_basis,
    decode_d106_rdce_tau,
    deserialize_d106_rdce_asset,
)
from cvsrffi.rxid_metabias4_held_execution import (  # noqa: E402
    build_receiver_package_indices,
    canonical_sha256,
    package_id,
)
from cvsrffi.stage2_d104_source_split import CANDIDATE_ID as D104_CANDIDATE_ID  # noqa: E402
from cvsrffi.stage2_zid_student_t_qknn import (  # noqa: E402
    Phase1ZIDStudentTLock,
    build_typed_zid_support_bank,
    identity_shared_psd_metric,
    score_zid_student_t_logits,
)


ARMS = ("M0", "M_DA", "M_HEAD", "M_JOINT")
K_VALUES = (1, 5, 10)
SPLIT_ID = "d104_source_seed104713_v2"
PREDICTION_SCHEMA = "cvs.d106.g1.sourceheld.predictions.v1"
SCORE_SCHEMA = "cvs.d106.g1.sourceheld.scores.v1"
PACKAGE_SCHEMA = "cvs.d104_r1.rxid_angq.held_packages.v2"
PACKAGE_KEYS = {
    "support_pre_relu", "support_zdom", "support_labels",
    "support_physical_ids", "query_pre_relu", "query_physical_ids",
    "registered_classes",
}
SOURCE_VAL_KEYS = {
    "z_id", "z_dom", "pre_relu", "labels", "receiver_ids", "day_ids",
    "physical_ids", "scenario_names", "observation_ids", "class_ids",
}
RCMR_LOCK_SHA256 = "be452cc52da8e5c43d3addc73568580d63a83f146310ec3559bb5daa99076b0c"
D105_LOCK_SHA256 = "7324ff469cf18d34cdc3795e36d053570e60ba341c112167b49d759a150dda08"


class D106G1Error(ValueError):
    pass


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise D106G1Error(f"expected JSON object: {path}")
    return value


def _write_new(path: Path, value: Any) -> None:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)


def fixed_row_specs(receivers: Sequence[str], classes: Sequence[str]) -> tuple[tuple[str, str | None, int], ...]:
    """The complete preregistered 63 rows; callers cannot select a subset."""

    return tuple(
        [(receiver, None, k) for receiver in receivers for k in K_VALUES]
        + [(receiver, class_id, 1) for receiver in receivers for class_id in classes]
    )


def _parse_asset_wire(path: Path, expected_sha256: str):
    wire = path.read_bytes()
    if hashlib.sha256(wire).hexdigest() != expected_sha256:
        raise D106G1Error("RDCE asset wire SHA256 mismatch")
    magic = rdce_asset.WIRE_MAGIC
    if not wire.startswith(magic) or len(wire) < len(magic) + 4:
        raise D106G1Error("RDCE asset wire framing drift")
    offset = len(magic)
    size = struct.unpack(">I", wire[offset : offset + 4])[0]
    header = json.loads(wire[offset + 4 : offset + 4 + size].decode("utf-8"))
    asset = header.get("asset") if isinstance(header, dict) else None
    if not isinstance(asset, dict):
        raise D106G1Error("RDCE asset header missing")
    lineage = D106RDCEAssetLineage(
        checkpoint_sha256=asset["checkpoint_sha256"], runtime_sha256=asset["runtime_sha256"],
        method_lock_sha256=asset["method_lock_sha256"], split_id=asset["split_id"],
        tap_sha256=asset["tap_sha256"], construction_code_sha256=asset["construction_code_sha256"],
        content_root_sha256=asset["content_root_sha256"], source_receipt_sha256=asset["source_receipt_sha256"],
        tap_receipt_sha256=asset["tap_receipt_sha256"], tap_authority_sha256=asset["tap_authority_sha256"],
    )
    return deserialize_d106_rdce_asset(wire, expected_wire_sha256=expected_sha256, expected_lineage=lineage)


def _normalized(rows: np.ndarray) -> np.ndarray:
    values = np.ascontiguousarray(rows, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 160 or not np.isfinite(values).all():
        raise D106G1Error("D106 feature layout drift")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 1.0e-12):
        raise D106G1Error("D106 feature contains a zero-norm row")
    return values / norms


def fit_rdce_sourceheld_state(asset: Any, support_zid: np.ndarray, labels: Sequence[str], k_shot: int) -> dict[str, Any]:
    """Frozen RDCE math on support only; explicitly source-held/non-Target."""

    support = _normalized(support_zid)
    typed_labels = tuple(str(value) for value in labels)
    basis = decode_d106_rdce_basis(asset)
    tau = decode_d106_rdce_tau(asset)
    if k_shot == 1:
        attenuation = np.full(3, np.float16(0.3), dtype=np.float16)
    else:
        scatters = []
        for class_id in sorted(set(typed_labels)):
            group = support[np.asarray(typed_labels) == class_id]
            if len(group) != k_shot:
                raise D106G1Error("support is not exactly K per class")
            projected = (group - np.mean(group, axis=0)) @ basis.T
            scatters.append(np.sum(np.square(projected), axis=0) / float(k_shot - 1))
        scatter = np.mean(np.stack(scatters), axis=0)
        a0 = min(0.95, 1.5 * k_shot / float(k_shot + 4))
        raw = a0 + 0.2 * np.tanh(np.log((scatter + 1.0e-8) / (tau + 1.0e-8)))
        attenuation = np.asarray(
            np.clip(
                raw,
                float(MIN_ATTENUATION_FP16),
                float(MAX_ATTENUATION_FP16),
            ),
            dtype=np.float16,
        )
    payload = {
        "scope": "SOURCE_HELD_NON_TARGET_NO_P2_AUTHORITY",
        "asset_receipt_sha256": asset.asset_receipt_sha256,
        "K": k_shot,
        "attenuation_fp16": [float(value) for value in attenuation],
        "support_root_sha256": hashlib.sha256(np.ascontiguousarray(support, dtype=np.float64).tobytes()).hexdigest(),
        "query_rows_used_for_fit": 0,
        "query_state_updates": 0,
    }
    return {"basis": basis, "attenuation": attenuation.astype(np.float64), "receipt": _sha(payload), "payload": payload}


def apply_rdce_state(state: Mapping[str, Any], rows: np.ndarray) -> np.ndarray:
    values = _normalized(rows)
    basis = np.asarray(state["basis"], dtype=np.float64)
    coefficient = 1.0 - np.sqrt(1.0 - np.asarray(state["attenuation"], dtype=np.float64))
    transformed = values - ((values @ basis.T) * coefficient) @ basis
    transformed /= np.linalg.norm(transformed, axis=1, keepdims=True)
    return np.ascontiguousarray(transformed, dtype=np.float32)


def _lock(k_shot: int, package_sha256: str) -> Phase1ZIDStudentTLock:
    return Phase1ZIDStudentTLock(
        active_k=k_shot, phase1_lodo_receipt_sha256=D105_LOCK_SHA256,
        quantization_margin_audit_sha256=package_sha256,
        **dict(g0.PREDECESSOR_NUMERIC_LOCK),
    )


def _argmax(scores: np.ndarray, registry: Sequence[str]) -> list[str]:
    result = []
    for row in np.asarray(scores, dtype=np.float64):
        maximum = max(float(value) for value in row)
        winners = [index for index, value in enumerate(row) if g0._rcmr_module._same_binary64(float(value), maximum)]
        if len(winners) != 1:
            raise D106G1Error("cross-class tie is fail-closed")
        result.append(str(registry[winners[0]]))
    return result


def _baseline(support: np.ndarray, labels: Sequence[str], query: np.ndarray, registry: Sequence[str], lock: Phase1ZIDStudentTLock) -> list[str]:
    bank = build_typed_zid_support_bank(support, labels, registry, config=lock)
    metric = identity_shared_psd_metric(config=lock)
    return _argmax(score_zid_student_t_logits(bank, query, metric=metric), registry)


def _rcmr(support_plus: np.ndarray, support_signed: np.ndarray, labels: Sequence[str], physical_ids: Sequence[str], query_plus: np.ndarray, query_signed: np.ndarray, registry: Sequence[str], k_shot: int, da_receipt: str) -> tuple[list[str], str]:
    support_ids = tuple(str(value) for value in physical_ids)
    paired = _sha({
        "paired_view_receipt_sha256": g0._paired_view_receipt(
            support_ids, support_plus, support_signed
        ),
        "da_factor_receipt_sha256": da_receipt,
    })
    state = g0._nonformal_state_from_support(
        support_plus, support_signed, tuple(str(value) for value in labels), support_ids,
        tuple(str(value) for value in registry), active_k=k_shot,
        support_root_sha256=g0._support_root(support_ids), paired_view_receipt_sha256=paired,
        rcmr_method_lock_sha256=RCMR_LOCK_SHA256,
    )
    context = g0._prepare_nonformal_context(state)
    predictions = [
        g0._score_nonformal_query(state, context, plus, signed)
        for plus, signed in zip(query_plus, query_signed, strict=True)
    ]
    return predictions, state.state_receipt_sha256


def prepare(args: argparse.Namespace) -> int:
    """Separate the frozen D104 scorer archive into 21 predictor packages and truth."""

    archive_path = args.source_val_archive.resolve(strict=True)
    manifest_path = args.source_val_manifest.resolve(strict=True)
    output = args.output_dir.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"immutable D106 G1 package root exists: {output}")
    manifest = _read_json(manifest_path)
    if (manifest.get("candidate_id") != D104_CANDIDATE_ID
            or manifest.get("split_id") != SPLIT_ID
            or manifest.get("role") != "source_val_scorer_only"
            or manifest.get("archive", {}).get("sha256") != _file_sha(archive_path)
            or manifest.get("asset_access") is not False
            or manifest.get("gradient_access") is not False
            or manifest.get("selection_access") is not False
            or manifest.get("target_access") is not False
            or manifest.get("formal_query_access") is not False):
        raise D106G1Error("D104 source-val scorer manifest drift")
    with np.load(archive_path, allow_pickle=False) as archive:
        if set(archive.files) != SOURCE_VAL_KEYS:
            raise D106G1Error("D104 source-val member closure drift")
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    receivers = tuple(sorted(set(arrays["receiver_ids"].astype(str).tolist())))
    classes = tuple(sorted(set(arrays["labels"].astype(str).tolist())))
    days = tuple(sorted(set(arrays["day_ids"].astype(str).tolist())))
    if len(receivers) != 7 or len(classes) != 6 or len(days) != 4:
        raise D106G1Error("D104 source-val registry drift")
    output.mkdir(parents=True, exist_ok=False)
    predictor_root = output / "predictor_packages"
    predictor_root.mkdir()
    package_rows = []
    truth_rows = []
    for receiver in receivers:
        for k_shot in K_VALUES:
            support, query = build_receiver_package_indices(
                arrays["receiver_ids"], arrays["labels"], arrays["physical_ids"],
                held_receiver=receiver, registered_classes=classes, k_shot=k_shot,
            )
            identity = package_id(receiver, k_shot)
            package_path = predictor_root / f"{identity}.npz"
            with package_path.open("xb") as stream:
                np.savez(
                    stream,
                    support_pre_relu=np.asarray(arrays["pre_relu"][support], dtype=np.float32),
                    support_zdom=np.asarray(arrays["z_dom"][support], dtype=np.float32),
                    support_labels=arrays["labels"][support].astype(str),
                    support_physical_ids=arrays["physical_ids"][support].astype(str),
                    query_pre_relu=np.asarray(arrays["pre_relu"][query], dtype=np.float32),
                    query_physical_ids=arrays["physical_ids"][query].astype(str),
                    registered_classes=np.asarray(classes, dtype=str),
                )
            support_ids = arrays["physical_ids"][support].astype(str).tolist()
            query_ids = arrays["physical_ids"][query].astype(str).tolist()
            package_rows.append({
                "package_id": identity, "held_receiver": receiver, "K": k_shot,
                "path": str(Path("predictor_packages") / package_path.name),
                "sha256": _file_sha(package_path),
                "support_physical_id_root_sha256": canonical_sha256(support_ids),
                "query_physical_id_root_sha256": canonical_sha256(query_ids),
                "query_truth_present": False,
                "support_query_physical_disjoint": True,
            })
            truth_rows.append({
                "package_id": identity, "query_physical_ids": query_ids,
                "query_truth_labels": arrays["labels"][query].astype(str).tolist(),
            })
    truth = {
        "schema": "cvs.d104_r1.rxid_angq.held_truth.v2", "split_id": SPLIT_ID,
        "package_count": 21, "packages": truth_rows, "predictor_access": False,
    }
    truth_seal = {
        "schema": "cvs.d104_r1.rxid_angq.truth_input_seal.v1", "split_id": SPLIT_ID,
        "package_count": 21, "package_ids": [row["package_id"] for row in truth_rows],
        "query_physical_id_roots": {
            row["package_id"]: canonical_sha256(row["query_physical_ids"])
            for row in truth_rows
        },
        "truth_package_root_sha256": canonical_sha256(truth_rows),
        "source_val_scorer_manifest_sha256": _file_sha(manifest_path),
        "source_val_scorer_archive_sha256": _file_sha(archive_path),
        "predictor_truth_access": False,
    }
    scorer_root = output / "scorer_only"
    scorer_root.mkdir()
    seal_path = scorer_root / "truth_input_seal.json"
    _write_new(seal_path, truth_seal)
    package_manifest = {
        "schema": PACKAGE_SCHEMA, "candidate_id": D104_CANDIDATE_ID,
        "split_id": SPLIT_ID, "receiver_ids": list(receivers),
        "class_ids": list(classes), "day_ids": list(days),
        "package_count": 21, "packages": package_rows,
        "registered_class_root_sha256": canonical_sha256(list(classes)),
        "source_val_scorer_manifest_sha256": _file_sha(manifest_path),
        "source_val_scorer_archive_sha256": _file_sha(archive_path),
        "truth_input_seal_sha256": _file_sha(seal_path),
        "query_truth_present": False, "target_access": False,
        "formal_query_state_updates": 0,
    }
    _write_new(output / "package_manifest.json", package_manifest)
    _write_new(scorer_root / "truth.json", truth)
    print(output / "package_manifest.json")
    return 0


def predict(args: argparse.Namespace) -> int:
    root = args.package_root.resolve(strict=True)
    output = args.output_dir.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"immutable D106 G1 output exists: {output}")
    manifest_path = root / "package_manifest.json"
    manifest = _read_json(manifest_path)
    receivers = tuple(str(value) for value in manifest.get("receiver_ids", ()))
    classes = tuple(str(value) for value in manifest.get("class_ids", ()))
    packages = manifest.get("packages")
    if (manifest.get("schema") != PACKAGE_SCHEMA or manifest.get("split_id") != SPLIT_ID
            or manifest.get("query_truth_present") is not False or len(receivers) != 7
            or len(classes) != 6 or not isinstance(packages, list) or len(packages) != 21):
        raise D106G1Error("D104 truth-free package manifest closure drift")
    package_by_key = {(str(row["held_receiver"]), int(row["K"])): row for row in packages}
    expected = {(receiver, k) for receiver in receivers for k in K_VALUES}
    if set(package_by_key) != expected:
        raise D106G1Error("D104 package matrix is not complete")
    asset = _parse_asset_wire(args.rdce_asset_wire.resolve(strict=True), args.rdce_wire_sha256)
    if asset.split_id != SPLIT_ID:
        raise D106G1Error("RDCE asset/source-held split mismatch")
    if _file_sha(args.rcmr_method_lock.resolve(strict=True)) != args.rcmr_method_lock_sha256 or args.rcmr_method_lock_sha256 != RCMR_LOCK_SHA256:
        raise D106G1Error("RCMR method lock mismatch")
    g0._rcmr_module.load_d106_rcmr_2v_method_lock(args.rcmr_method_lock, expected_sha256=RCMR_LOCK_SHA256)
    output.mkdir(parents=True, exist_ok=False)
    row_root = output / "rows"
    row_root.mkdir()
    rows = []
    package_cache: dict[tuple[str, int], tuple[dict[str, np.ndarray], str]] = {}
    for receiver, held_class, k_shot in fixed_row_specs(receivers, classes):
        key = (receiver, k_shot)
        package_row = package_by_key[key]
        if key not in package_cache:
            relative = Path(str(package_row["path"]))
            path = (root / relative).resolve(strict=True)
            if relative.is_absolute() or not path.is_relative_to(root) or _file_sha(path) != package_row["sha256"]:
                raise D106G1Error("D104 package path/SHA drift")
            with np.load(path, allow_pickle=False) as archive:
                if set(archive.files) != PACKAGE_KEYS:
                    raise D106G1Error("D104 package member closure drift")
                package_cache[key] = ({name: np.array(archive[name], copy=True) for name in archive.files}, package_row["sha256"])
        package, package_sha = package_cache[key]
        registry = tuple(package["registered_classes"].astype(str).tolist())
        support_ids = tuple(package["support_physical_ids"].astype(str).tolist())
        query_ids = tuple(package["query_physical_ids"].astype(str).tolist())
        labels = tuple(package["support_labels"].astype(str).tolist())
        if registry != classes or set(support_ids).intersection(query_ids) or len(labels) != len(classes) * k_shot:
            raise D106G1Error("D104 package registry/disjoint/K closure drift")
        support_signed = np.ascontiguousarray(package["support_pre_relu"], dtype=np.float32)
        query_signed = np.ascontiguousarray(package["query_pre_relu"], dtype=np.float32)
        support_plus = np.ascontiguousarray(np.maximum(support_signed, np.float32(0.0)))
        query_plus = np.ascontiguousarray(np.maximum(query_signed, np.float32(0.0)))
        da_state = fit_rdce_sourceheld_state(asset, support_plus, labels, k_shot)
        da_support_plus = apply_rdce_state(da_state, support_plus)
        da_query_plus = apply_rdce_state(da_state, query_plus)
        da_support_signed = apply_rdce_state(da_state, support_signed)
        da_query_signed = apply_rdce_state(da_state, query_signed)
        identity_view_receipt = _sha({
            "support_plus": hashlib.sha256(support_plus.tobytes()).hexdigest(),
            "support_signed": hashlib.sha256(support_signed.tobytes()).hexdigest(),
            "query_plus": hashlib.sha256(query_plus.tobytes()).hexdigest(),
            "query_signed": hashlib.sha256(query_signed.tobytes()).hexdigest(),
        })
        lock = _lock(k_shot, package_sha)
        m0 = _baseline(support_plus, labels, query_plus, registry, lock)
        mda = _baseline(da_support_plus, labels, da_query_plus, registry, lock)
        mhead, head_state = _rcmr(support_plus, support_signed, labels, support_ids, query_plus, query_signed, registry, k_shot, _sha("identity"))
        mjoint, joint_state = _rcmr(da_support_plus, da_support_signed, labels, support_ids, da_query_plus, da_query_signed, registry, k_shot, da_state["receipt"])
        row = {
            "schema": PREDICTION_SCHEMA + ".row", "held_receiver": receiver,
            "held_class": held_class, "K": k_shot,
            "package_id": str(package_row["package_id"]),
            "registered_classes": list(registry),
            "query_physical_ids": list(query_ids),
            "arm_predictions": {"M0": m0, "M_DA": mda, "M_HEAD": mhead, "M_JOINT": mjoint},
            "shared_component_receipts": {
                "M_DA_M_JOINT_rdce_state_sha256": da_state["receipt"],
                "M0_M_HEAD_identity_view_sha256": identity_view_receipt,
                "M0_M_DA_student_t_lock_sha256": lock.lock_digest,
                "M_HEAD_M_JOINT_rcmr_method_lock_sha256": RCMR_LOCK_SHA256,
                "M_HEAD_state_sha256": head_state, "M_JOINT_state_sha256": joint_state,
            },
            "query_truth_access": False, "target_access": False,
            "formal_p2_authority": False, "query_state_updates": 0,
        }
        row["prediction_receipt_sha256"] = _sha(row)
        row_id = _sha({"receiver": receiver, "held_class": held_class, "K": k_shot})
        path = row_root / f"{row_id}.json"
        _write_new(path, row)
        rows.append({"held_receiver": receiver, "held_class": held_class, "K": k_shot, "package_id": str(package_row["package_id"]), "path": str(Path("rows") / path.name), "sha256": _file_sha(path), "prediction_receipt_sha256": row["prediction_receipt_sha256"]})
    receipts = [row["prediction_receipt_sha256"] for row in rows]
    if len(rows) != 63 or len(set(receipts)) != 63:
        raise D106G1Error("D106 G1 prediction coverage did not close at 63 rows")
    result = {
        "schema": PREDICTION_SCHEMA, "split_id": SPLIT_ID, "row_count": 63,
        "arm_row_prediction_unit_count": 252, "rows": rows,
        "package_manifest_sha256": _file_sha(manifest_path),
        "rdce_asset_wire_sha256": args.rdce_wire_sha256,
        "rcmr_method_lock_sha256": RCMR_LOCK_SHA256,
        "query_truth_access": False, "target_access": False, "query_state_updates": 0,
        "sourceheld_non_target": True, "formal_p2_authority": False,
        "sealed_at_unix_ns": time.time_ns(),
    }
    result["prediction_set_receipt_sha256"] = _sha(result)
    _write_new(output / "prediction_manifest.json", result)
    print(output / "prediction_manifest.json")
    return 0


def _metric(truth: np.ndarray, predicted: Sequence[str], classes: Sequence[str], held_class: str | None) -> dict[str, Any]:
    prediction = np.asarray(predicted, dtype=str)
    per_class = {}
    for class_id in classes:
        mask = truth == class_id
        if not np.any(mask):
            raise D106G1Error("truth lacks a registered class")
        per_class[class_id] = float(np.mean(prediction[mask] == class_id))
    old_classes = tuple(class_id for class_id in classes if class_id != held_class)
    old_ba = float(np.mean([per_class[class_id] for class_id in old_classes]))
    new_acc = None if held_class is None else per_class[held_class]
    harmonic = None if new_acc is None or old_ba + new_acc == 0.0 else 2.0 * old_ba * new_acc / (old_ba + new_acc)
    return {
        "old_balanced_accuracy": old_ba, "seen_new_accuracy": new_acc,
        "H_old_new": harmonic, "old_floor": min(per_class[class_id] for class_id in old_classes),
        "all_class_floor": min(per_class.values()), "balanced_accuracy": float(np.mean(list(per_class.values()))),
        "correct_count": int(np.sum(prediction == truth)), "query_count": len(truth),
        "per_class_accuracy": per_class,
    }


def score(args: argparse.Namespace) -> int:
    root = args.prediction_root.resolve(strict=True)
    output = args.output_json.resolve()
    event_path = args.truth_open_event_json.resolve()
    truth_input_seal_path = args.truth_input_seal_json.resolve(strict=True)
    if output.exists() or event_path.exists():
        raise FileExistsError("immutable D106 G1 score/event output exists")
    manifest_path = root / "prediction_manifest.json"
    manifest = _read_json(manifest_path)
    rows = manifest.get("rows")
    if (manifest.get("schema") != PREDICTION_SCHEMA or manifest.get("row_count") != 63
            or manifest.get("arm_row_prediction_unit_count") != 252
            or manifest.get("query_truth_access") is not False or not isinstance(rows, list) or len(rows) != 63):
        raise D106G1Error("sealed D106 G1 prediction manifest closure drift")
    expected_receipt = manifest.get("prediction_set_receipt_sha256")
    if _sha({key: value for key, value in manifest.items() if key != "prediction_set_receipt_sha256"}) != expected_receipt:
        raise D106G1Error("prediction set receipt drift")
    artifacts = []
    package_ids = set()
    query_ids_by_package: dict[str, list[str]] = {}
    for entry in rows:
        relative = Path(str(entry["path"]))
        path = (root / relative).resolve(strict=True)
        if relative.is_absolute() or not path.is_relative_to(root) or _file_sha(path) != entry["sha256"]:
            raise D106G1Error("prediction row seal drift")
        artifact = _read_json(path)
        if _sha({key: value for key, value in artifact.items() if key != "prediction_receipt_sha256"}) != artifact.get("prediction_receipt_sha256"):
            raise D106G1Error("prediction row receipt drift")
        package_id = str(artifact.get("package_id"))
        query_ids = artifact.get("query_physical_ids")
        if not isinstance(query_ids, list) or not query_ids:
            raise D106G1Error("prediction query physical-ID closure drift")
        if (artifact.get("schema") != PREDICTION_SCHEMA + ".row"
                or artifact.get("query_truth_access") is not False
                or artifact.get("target_access") is not False
                or artifact.get("formal_p2_authority") is not False
                or artifact.get("query_state_updates") != 0
                or set(artifact.get("arm_predictions", {})) != set(ARMS)
                or any(len(artifact["arm_predictions"][arm]) != len(query_ids) for arm in ARMS)
                or any(entry.get(name) != artifact.get(name) for name in ("held_receiver", "held_class", "K", "package_id", "prediction_receipt_sha256"))):
            raise D106G1Error("prediction row lifecycle/arm identity drift")
        prior = query_ids_by_package.setdefault(package_id, query_ids)
        if prior != query_ids:
            raise D106G1Error("package query physical IDs drift across matched rows")
        package_ids.add(package_id)
        artifacts.append(artifact)
    if len(package_ids) != 21:
        raise D106G1Error("sealed predictions do not cover 21 packages")
    receivers = tuple(sorted({str(artifact["held_receiver"]) for artifact in artifacts}))
    classes = tuple(str(value) for value in artifacts[0]["registered_classes"])
    actual_rows = {(str(row["held_receiver"]), row["held_class"], int(row["K"])) for row in artifacts}
    if (len(receivers) != 7 or len(classes) != 6
            or any(tuple(str(value) for value in row["registered_classes"]) != classes for row in artifacts)
            or actual_rows != set(fixed_row_specs(receivers, classes))):
        raise D106G1Error("sealed prediction rows are not the fixed complete 63-row matrix")
    truth_input_seal = _read_json(truth_input_seal_path)
    if (truth_input_seal.get("schema") != "cvs.d104_r1.rxid_angq.truth_input_seal.v1"
            or truth_input_seal.get("split_id") != SPLIT_ID
            or truth_input_seal.get("package_count") != 21
            or truth_input_seal.get("predictor_truth_access") is not False
            or set(truth_input_seal.get("package_ids", ())) != package_ids):
        raise D106G1Error("D104 truth input seal closure drift")
    manifest_mtime = manifest_path.stat().st_mtime_ns
    event = {"schema": SCORE_SCHEMA + ".truth_open_event", "prediction_manifest_sha256": _file_sha(manifest_path), "truth_input_seal_sha256": _file_sha(truth_input_seal_path), "prediction_manifest_mtime_ns": manifest_mtime, "truth_opened_after_all_predictions_committed": True, "opened_at_unix_ns": time.time_ns()}
    if event["opened_at_unix_ns"] <= manifest_mtime:
        raise D106G1Error("truth-open timestamp is not after prediction seal")
    _write_new(event_path, event)
    truth = _read_json(args.truth_json.resolve(strict=True))
    truth_packages = truth.get("packages")
    if (truth.get("schema") != "cvs.d104_r1.rxid_angq.held_truth.v2"
            or truth.get("split_id") != SPLIT_ID or truth.get("package_count") != 21
            or not isinstance(truth_packages, list) or len(truth_packages) != 21
            or truth.get("predictor_access") is not False):
        raise D106G1Error("independent truth package closure drift")
    truth_by_package = {str(row["package_id"]): row for row in truth_packages}
    if (set(truth_by_package) != package_ids
            or canonical_sha256(truth_packages) != truth_input_seal.get("truth_package_root_sha256")):
        raise D106G1Error("opened truth root/package identity drift")
    scored_rows = []
    negative: dict[str, dict[str, int]] = {name: {metric: 0 for metric in ("old_balanced_accuracy", "seen_new_accuracy", "H_old_new", "old_floor")} for name in ("DA_AT_BASE", "DA_AT_HEAD", "HEAD_AT_ID", "HEAD_AT_DA", "JOINT_VS_M0")}
    pairs = {"DA_AT_BASE": ("M_DA", "M0"), "DA_AT_HEAD": ("M_JOINT", "M_HEAD"), "HEAD_AT_ID": ("M_HEAD", "M0"), "HEAD_AT_DA": ("M_JOINT", "M_DA"), "JOINT_VS_M0": ("M_JOINT", "M0")}
    for artifact in artifacts:
        matching = truth_by_package[str(artifact["package_id"])]
        if matching["query_physical_ids"] != artifact["query_physical_ids"]:
            raise D106G1Error("truth/prediction physical-ID alignment drift")
        labels = np.asarray(matching["query_truth_labels"], dtype=str)
        metrics = {arm: _metric(labels, artifact["arm_predictions"][arm], artifact["registered_classes"], artifact["held_class"]) for arm in ARMS}
        effects = {}
        for name, (left, right) in pairs.items():
            effects[name] = {}
            for metric in negative[name]:
                lv, rv = metrics[left][metric], metrics[right][metric]
                delta = None if lv is None or rv is None else float(lv) - float(rv)
                effects[name][metric] = delta
                if delta is not None and delta < 0.0:
                    negative[name][metric] += 1
        scored_rows.append({"held_receiver": artifact["held_receiver"], "held_class": artifact["held_class"], "K": artifact["K"], "arm_metrics": metrics, "same_row_effects": effects, "prediction_receipt_sha256": artifact["prediction_receipt_sha256"]})
    result = {"schema": SCORE_SCHEMA, "split_id": SPLIT_ID, "performance_rows": scored_rows, "negative_tail_row_counts": negative, "prediction_manifest_sha256": _file_sha(manifest_path), "truth_input_seal_sha256": _file_sha(truth_input_seal_path), "truth_sha256": _file_sha(args.truth_json), "truth_open_event_sha256": _file_sha(event_path), "prediction_artifact_committed_before_truth": True, "target_access": False}
    result["score_set_receipt_sha256"] = _sha(result)
    _write_new(output, result)
    print(output)
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    preparer = commands.add_parser("prepare")
    preparer.add_argument("--source-val-archive", type=Path, required=True)
    preparer.add_argument("--source-val-manifest", type=Path, required=True)
    preparer.add_argument("--output-dir", type=Path, required=True)
    predictor = commands.add_parser("predict")
    predictor.add_argument("--package-root", type=Path, required=True)
    predictor.add_argument("--rdce-asset-wire", type=Path, required=True)
    predictor.add_argument("--rdce-wire-sha256", required=True)
    predictor.add_argument("--rcmr-method-lock", type=Path, required=True)
    predictor.add_argument("--rcmr-method-lock-sha256", required=True)
    predictor.add_argument("--output-dir", type=Path, required=True)
    scorer = commands.add_parser("score")
    scorer.add_argument("--prediction-root", type=Path, required=True)
    scorer.add_argument("--truth-json", type=Path, required=True)
    scorer.add_argument("--truth-input-seal-json", type=Path, required=True)
    scorer.add_argument("--truth-open-event-json", type=Path, required=True)
    scorer.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "prepare":
        return prepare(args)
    return predict(args) if args.command == "predict" else score(args)


if __name__ == "__main__":
    raise SystemExit(main())
