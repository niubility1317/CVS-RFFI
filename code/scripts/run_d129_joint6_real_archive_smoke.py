#!/usr/bin/env python3
"""Run one immutable D129 no-truth smoke on a real checkpoint-derived archive.

This entry reads the SHA256-pinned D106 588-row feature archive plus its pinned
real-integration fixture, constructs one complete seen-class LOCO fold, and
executes both D129 candidates with one shared R0.  It writes no prediction or
performance field and never opens a truth-side scorer.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import stat
import sys
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np


SCRIPT_ROOT = Path(__file__).resolve().parent
CODE_ROOT = SCRIPT_ROOT.parent
for candidate in (str(SCRIPT_ROOT), str(CODE_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from run_d106_rcmr_g0_one_shot import (  # noqa: E402
    _predecessor_locks,
    _read_pinned_archive,
)
from cvsrffi import stage2_d129_joint6_da as da  # noqa: E402
from cvsrffi import stage2_d129_joint6_heads as heads  # noqa: E402
from cvsrffi import stage2_d129_joint6_matrix as matrix  # noqa: E402
from cvsrffi import stage2_d129_joint6_runtime as runtime  # noqa: E402


SCHEMA = "cvs.stage2.d129.joint6.real_archive_no_truth_smoke.v1"
STATUS = "REAL_CHECKPOINT_DERIVED_ARCHIVE_SMOKE_EXECUTED_NO_PERFORMANCE_RESULT"
LS_ARCHIVE_MEMBERS = (
    "z_dom",
    "pre_relu",
    "receiver_ids",
    "day_ids",
    "tx_labels",
    "physical_ids",
)


class D129RealArchiveSmokeError(ValueError):
    """Raised when the pinned smoke input or immutable output drifts."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: str, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or value.lower() != value
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise D129RealArchiveSmokeError(f"{name} must be a lowercase SHA256")
    return value


def _read_pinned_json(path: Path, expected_sha256: str) -> Mapping[str, Any]:
    if not path.is_absolute():
        raise D129RealArchiveSmokeError("fixture path must be absolute")
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise D129RealArchiveSmokeError("cannot stat fixture") from exc
    if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        raise D129RealArchiveSmokeError("fixture must be a regular non-symlink file")
    payload = path.read_bytes()
    if _sha256_bytes(payload) != _require_sha256(expected_sha256, "fixture SHA256"):
        raise D129RealArchiveSmokeError("fixture SHA256 mismatch")
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise D129RealArchiveSmokeError("fixture must be UTF-8 JSON") from exc
    if not isinstance(value, Mapping):
        raise D129RealArchiveSmokeError("fixture root must be a mapping")
    return value


def _load_d104_ls_rows(archive_bytes: bytes, *, archive_sha256: str) -> Any:
    """Read the six-member D104 ``L_s/features.npz`` used by the fixture.

    The older D106 one-shot helper consumes the later eight-member strict-tap
    archive.  The real-integration fixture instead pins the original D104 L_s
    archive, so the D129 smoke validates that exact schema directly.
    """

    try:
        with np.load(io.BytesIO(archive_bytes), allow_pickle=False) as loaded:
            if tuple(loaded.files) != LS_ARCHIVE_MEMBERS:
                raise D129RealArchiveSmokeError("D104 L_s archive member set/order drift")
            arrays = {name: np.asarray(loaded[name]) for name in LS_ARCHIVE_MEMBERS}
    except D129RealArchiveSmokeError:
        raise
    except (OSError, ValueError, KeyError) as exc:
        raise D129RealArchiveSmokeError("invalid no-pickle D104 L_s archive") from exc

    for name in ("z_dom", "pre_relu"):
        array = arrays[name]
        if array.dtype != np.dtype(np.float32) or array.shape != (588, 160):
            raise D129RealArchiveSmokeError(f"D104 L_s {name} dtype/shape drift")
        if not np.isfinite(array).all():
            raise D129RealArchiveSmokeError(f"D104 L_s {name} contains non-finite values")
    for name in ("receiver_ids", "day_ids", "tx_labels", "physical_ids"):
        array = arrays[name]
        if array.dtype.kind not in {"U", "S"} or array.shape != (588,):
            raise D129RealArchiveSmokeError(f"D104 L_s {name} dtype/shape drift")
        if any(not str(value).strip() for value in array.tolist()):
            raise D129RealArchiveSmokeError(f"D104 L_s {name} contains blanks")

    pre_relu = np.ascontiguousarray(arrays["pre_relu"].copy())
    z_id = np.ascontiguousarray(np.maximum(pre_relu, np.float32(0.0)))
    receipt = {
        "schema": "cvs.stage2.d129.d104_ls_archive_view.v1",
        "protocol_schema": "p2_min_v1",
        "archive_sha256": archive_sha256,
        "archive_member_names": list(LS_ARCHIVE_MEMBERS),
        "row_count": 588,
        "z_id_policy": "relu_pre_relu",
        "truth_loaded": False,
    }
    return SimpleNamespace(
        z_id=z_id,
        z_dom=np.ascontiguousarray(arrays["z_dom"].copy()),
        tx_labels=np.ascontiguousarray(arrays["tx_labels"].copy()),
        receiver_ids=np.ascontiguousarray(arrays["receiver_ids"].copy()),
        day_ids=np.ascontiguousarray(arrays["day_ids"].copy()),
        physical_ids=np.ascontiguousarray(arrays["physical_ids"].copy()),
        receipt=receipt,
    )


def _ordered_cell_indices(
    *,
    receiver: str,
    class_id: str,
    receiver_ids: Sequence[str],
    class_ids: Sequence[str],
    physical_ids: Sequence[str],
) -> tuple[int, ...]:
    indices = [
        index
        for index, (observed_receiver, observed_class) in enumerate(
            zip(receiver_ids, class_ids, strict=True)
        )
        if observed_receiver == receiver and observed_class == class_id
    ]
    if len(indices) != 14:
        raise D129RealArchiveSmokeError("every real archive cell must contain 14 rows")
    return tuple(
        sorted(
            indices,
            key=lambda index: hashlib.sha256(
                f"{da.LOCO_SALT}|{receiver}|{class_id}|{physical_ids[index]}".encode(
                    "utf-8"
                )
            ).hexdigest(),
        )
    )


def run_real_archive_smoke(
    *,
    archive_path: Path,
    archive_sha256: str,
    fixture_path: Path,
    fixture_sha256: str,
    checkpoint_sha256: str,
    held_receiver: str,
    held_class: str,
    run_id: str,
) -> Mapping[str, Any]:
    """Execute both candidates on one fully bound K5 fold without truth."""

    archive_sha256 = _require_sha256(archive_sha256, "archive SHA256")
    checkpoint_sha256 = _require_sha256(checkpoint_sha256, "checkpoint SHA256")
    fixture = _read_pinned_json(fixture_path, fixture_sha256)
    if (
        fixture.get("schema") != "cvs.d106.real_integration_fixture.v1"
        or fixture.get("protocol_schema") != "p2_min_v1"
        or fixture.get("ls_archive_sha256") != archive_sha256
        or fixture.get("checkpoint_sha256") != checkpoint_sha256
    ):
        raise D129RealArchiveSmokeError("fixture/archive/checkpoint provenance drift")
    rows = _load_d104_ls_rows(
        _read_pinned_archive(archive_path, expected_sha256=archive_sha256),
        archive_sha256=archive_sha256,
    )
    receiver_ids = tuple(str(value) for value in rows.receiver_ids.tolist())
    class_ids = tuple(str(value) for value in rows.tx_labels.tolist())
    physical_ids = tuple(str(value) for value in rows.physical_ids.tolist())
    receivers = tuple(sorted(set(receiver_ids)))
    classes = tuple(sorted(set(class_ids)))
    if held_receiver == "AUTO_FIRST":
        held_receiver = receivers[0]
    if held_class == "AUTO_FIRST":
        held_class = classes[0]
    if (
        len(receivers) != 7
        or len(classes) != 6
        or held_receiver not in receivers
        or held_class not in classes
        or not run_id
    ):
        raise D129RealArchiveSmokeError("held fold or real archive registry drift")
    cell_indices = {
        (receiver, class_id): _ordered_cell_indices(
            receiver=receiver,
            class_id=class_id,
            receiver_ids=receiver_ids,
            class_ids=class_ids,
            physical_ids=physical_ids,
        )
        for receiver in receivers
        for class_id in classes
    }
    loco = da.build_d129_loco_plan(
        da.D129LOCORecord(receiver, class_id, physical_ids[index])
        for (receiver, class_id), indices in cell_indices.items()
        for index in indices
    )
    fold = next(
        value
        for value in loco.folds
        if value.held_receiver == held_receiver and value.held_class == held_class
    )
    retained = tuple(value for value in classes if value != held_class)
    registry = retained + (held_class,)
    row_k1 = matrix.Joint6LocoRow(
        row_id=f"rx={held_receiver}|held={held_class}|K=1",
        held_receiver=held_receiver,
        held_class=held_class,
        active_k=1,
        retained_classes=retained,
        registered_classes=registry,
    )
    row_k5 = matrix.Joint6LocoRow(
        row_id=f"rx={held_receiver}|held={held_class}|K=5",
        held_receiver=held_receiver,
        held_class=held_class,
        active_k=5,
        retained_classes=retained,
        registered_classes=registry,
    )
    phase1_indices = tuple(
        index
        for receiver in receivers
        if receiver != held_receiver
        for class_id in classes
        if class_id != held_class
        for index in cell_indices[(receiver, class_id)]
    )
    support5_indices = {
        class_id: cell_indices[(held_receiver, class_id)][:5]
        for class_id in registry
    }
    support1_indices = {
        class_id: indices[:1] for class_id, indices in support5_indices.items()
    }
    query_indices = {
        class_id: cell_indices[(held_receiver, class_id)][5:]
        for class_id in registry
    }
    binding = matrix.bind_joint6_physical_ids(
        row_k1=row_k1,
        row_k5=row_k5,
        loco_fold_receipt=fold.as_dict(),
        phase1_fit_ids=tuple(physical_ids[index] for index in phase1_indices),
        k1_support_ids_by_class={
            class_id: tuple(physical_ids[index] for index in indices)
            for class_id, indices in support1_indices.items()
        },
        k5_support_ids_by_class={
            class_id: tuple(physical_ids[index] for index in indices)
            for class_id, indices in support5_indices.items()
        },
        query_ids_by_class={
            class_id: tuple(physical_ids[index] for index in indices)
            for class_id, indices in query_indices.items()
        },
    )
    phase1 = np.stack(
        [
            np.stack(
                [rows.z_id[list(cell_indices[(receiver, class_id)])] for class_id in retained]
            )
            for receiver in receivers
            if receiver != held_receiver
        ]
    ).astype(np.float32, copy=False)
    assets = da.build_d129_phase1_assets(
        phase1,
        checkpoint_sha256=checkpoint_sha256,
        phase1_seal_sha256=binding["phase1_seal_sha256"],
    )
    support_indices = tuple(
        index for class_id in registry for index in support5_indices[class_id]
    )
    held_query_indices = tuple(
        index for class_id in registry for index in query_indices[class_id]
    )
    support = np.ascontiguousarray(rows.z_id[list(support_indices)], dtype=np.float32)
    query = np.ascontiguousarray(rows.z_id[list(held_query_indices)], dtype=np.float32)
    labels = tuple(class_id for class_id in registry for _ in range(5))
    support_ids = tuple(physical_ids[index] for index in support_indices)
    query_ids = tuple(physical_ids[index] for index in held_query_indices)
    lock = next(value for value in _predecessor_locks(rows) if value.active_k == 5)
    common_r0 = heads.build_d129_common_r0(
        base_support_zid=support,
        base_query_zid=query,
        support_labels=labels,
        registered_classes=registry,
        old_class_count=5,
        partition_semantics="phase1_seen_class_loco_directional_proxy",
        opaque_query_ids=query_ids,
        qknn_lock=lock,
    )
    candidate_receipts: dict[str, Any] = {}
    for asset in assets:
        result = runtime.run_d129_candidate_joint6(
            asset=asset,
            base_support_zid160=support,
            base_query_zid160=query,
            support_labels=labels,
            support_physical_ids=support_ids,
            registered_classes=registry,
            retained_class_count=5,
            opaque_query_ids=query_ids,
            qknn_lock=lock,
            fold_binding=binding,
            common_r0=common_r0,
        )
        candidate_receipts[result.candidate_id] = {
            "candidate_function_status": (
                "PASS_REAL_NO_TRUTH_SMOKE"
                if result.smoke_receipt["smoke_pass"] is True
                else "REJECT_REVISION_NO_FUNCTION"
            ),
            "asset_sha256": da.d129_joint6_asset_sha256(asset),
            "asset_numeric_payload_bytes": asset.numeric_payload_bytes,
            "query_read_only_receipt": dict(result.query_read_only_receipt),
            "smoke_receipt": dict(result.smoke_receipt),
            "runtime_receipt": dict(result.runtime_receipt),
            "head_causal_resource_receipt": dict(
                result.six_arm.head_causal_resource_receipt
            ),
            "system_formal_replacement_resource_receipt": dict(
                result.six_arm.system_formal_replacement_resource_receipt
            ),
        }
    passed_candidates = sorted(
        candidate_id
        for candidate_id, receipt in candidate_receipts.items()
        if receipt["smoke_receipt"]["smoke_pass"] is True
    )
    if not passed_candidates:
        raise D129RealArchiveSmokeError(
            "all D129 candidates failed the real no-truth functional smoke"
        )
    return {
        "schema": SCHEMA,
        "status": STATUS,
        "run_id": run_id,
        "protocol_schema": "p2_min_v1",
        "truth_loaded": False,
        "prediction_artifact_emitted": False,
        "performance_result": False,
        "formal_new_registration_claim": False,
        "archive_sha256": archive_sha256,
        "fixture_sha256": fixture_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_derived_archive_provenance_verified": True,
        "held_receiver": held_receiver,
        "held_class": held_class,
        "evaluation_semantics": "phase1_seen_class_loco_directional_proxy",
        "phase1_fit_count": binding["phase1_fit_count"],
        "support_count": binding["k5_support_count"],
        "query_count": binding["query_count"],
        "binding_sha256": binding["binding_sha256"],
        "phase1_seal_sha256": binding["phase1_seal_sha256"],
        "query_physical_root_sha256": binding["query_physical_root_sha256"],
        "common_r0_sha256": common_r0.receipt["common_r0_sha256"],
        "common_r0_head_fit_count_total": 3,
        "common_r0_candidate_refit_count": 0,
        "passed_candidate_ids": passed_candidates,
        "rejected_no_function_candidate_ids": sorted(
            set(candidate_receipts) - set(passed_candidates)
        ),
        "candidate_receipts": candidate_receipts,
    }


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--fixture-sha256", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--held-receiver", required=True)
    parser.add_argument("--held-class", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.output.is_absolute() or args.output.exists() or not args.output.parent.is_dir():
        raise D129RealArchiveSmokeError(
            "output must be a new absolute file under an existing directory"
        )
    result = run_real_archive_smoke(
        archive_path=args.archive,
        archive_sha256=args.archive_sha256,
        fixture_path=args.fixture,
        fixture_sha256=args.fixture_sha256,
        checkpoint_sha256=args.checkpoint_sha256,
        held_receiver=args.held_receiver,
        held_class=args.held_class,
        run_id=args.run_id,
    )
    args.output.write_text(
        json.dumps(_plain(result), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
