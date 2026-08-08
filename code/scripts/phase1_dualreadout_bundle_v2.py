#!/usr/bin/env python
"""Export, build, and exercise the Phase1 dual-readout bundle v2."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.checkpoint_loading import build_exact_ssdg_model_from_checkpoint
from cvsrffi.phase1_dualreadout_bundle_v2 import (
    build_bundle,
    canonical_json,
    fit_source_calibration,
    load_bundle,
    sha256_file,
)
from cvsrffi.phase3_care_poe import SCHEMA as LOCAL_EVIDENCE_SCHEMA
from cvsrffi.phase3_care_poe import seal_local_evidence, write_jsonl


class FullDualReadoutRuntime(nn.Module):
    def __init__(self, model: nn.Module, *, runtime_batch_size: int = 256) -> None:
        super().__init__()
        self.model = model
        self.runtime_batch_size = int(runtime_batch_size)

    def forward(self, rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        count = rows.size(0)
        padded = rows.new_zeros((self.runtime_batch_size, rows.size(1), rows.size(2)))
        padded[:count].copy_(rows)
        result = self.model(padded)
        z_id = F.normalize(result["z_id"].float(), dim=1)
        z_dom = F.normalize(result["z_dom"].float(), dim=1)
        logits = result["tx_logits"].float()
        return z_id[:count], z_dom[:count], logits[:count]


def _load_checkpoint(path: Path) -> Mapping[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, Mapping):
        raise ValueError("checkpoint root must be a mapping")
    return value


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    if tuple(left.shape) != tuple(right.shape):
        raise ValueError("runtime parity shape mismatch")
    value = float(torch.max(torch.abs(left.float().cpu() - right.float().cpu())).item())
    if not np.isfinite(value):
        raise ValueError("runtime parity is non-finite")
    return value


@torch.no_grad()
def export_runtime(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint_path = Path(args.checkpoint)
    if sha256_file(checkpoint_path) != str(args.expected_checkpoint_sha256).lower():
        raise ValueError("checkpoint SHA256 mismatch")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoint = _load_checkpoint(checkpoint_path)
    model, audit = build_exact_ssdg_model_from_checkpoint(
        checkpoint, input_len=int(args.input_len), device=device
    )
    wrapper = FullDualReadoutRuntime(model.to(device).eval()).to(device).eval()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(args.parity_seed))
    probes = torch.randn(64, 2, int(args.input_len), generator=generator).to(device)
    output = Path(args.runtime_out)
    if output.exists():
        raise FileExistsError("refusing to overwrite runtime output")
    output.parent.mkdir(parents=True, exist_ok=True)
    traced = torch.jit.trace(wrapper, probes[:2], strict=False, check_trace=False)
    torch.jit.save(traced, str(output))
    runtime = torch.jit.load(str(output), map_location=device).eval()
    maximum = 0.0
    rows = []
    for batch_size in (1, 8, 64):
        eager = wrapper(probes[:batch_size])
        scripted = runtime(probes[:batch_size])
        deltas = [_max_abs(left, right) for left, right in zip(eager, scripted)]
        maximum = max(maximum, *deltas)
        rows.append({"batch_size": batch_size, "z_id_max_abs": deltas[0], "z_dom_max_abs": deltas[1], "logit_max_abs": deltas[2]})
    if maximum > float(args.max_abs_tolerance):
        raise ValueError(f"runtime parity exceeded tolerance: {maximum}")
    receipt = {
        "schema": "cvs.phase1.dualreadout_runtime_parity.v2",
        "checkpoint_sha256": str(args.expected_checkpoint_sha256).lower(),
        "runtime_sha256": sha256_file(output),
        "input_len": int(args.input_len),
        "validated_batch_sizes": [1, 8, 64],
        "max_abs": maximum,
        "tolerance": float(args.max_abs_tolerance),
        "checkpoint_load_audit": audit,
        "vectors": rows,
    }
    receipt_path = Path(args.receipt_out)
    if receipt_path.exists():
        raise FileExistsError("refusing to overwrite parity receipt")
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(canonical_json(receipt) + "\n", encoding="utf-8")
    return receipt


def _load_feature_npz(path: str | Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        required = {"features", "tx_logits", "tx_ids", "rx_ids", "day_ids", "sig_ids", "dataset_role"}
        if not required.issubset(archive.files):
            raise ValueError(f"feature NPZ missing fields: {sorted(required.difference(archive.files))}")
        return {key: np.array(archive[key], copy=True) for key in archive.files}


def _verify_same_rows(left: Mapping[str, np.ndarray], right: Mapping[str, np.ndarray]) -> None:
    for key in ("tx_ids", "rx_ids", "day_ids", "eq_ids", "sig_ids", "dataset_role", "channel_views", "sat_scenarios"):
        if key in left or key in right:
            if key not in left or key not in right or not np.array_equal(left[key], right[key]):
                raise ValueError(f"feature NPZ physical-row binding differs for {key}")


def _validate_parity_receipt(
    path: str | Path,
    *,
    runtime_path: str | Path,
    checkpoint_sha256: str,
) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    fields = {
        "schema", "checkpoint_sha256", "runtime_sha256", "input_len",
        "validated_batch_sizes", "max_abs", "tolerance", "checkpoint_load_audit", "vectors",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("runtime parity receipt field allowlist mismatch")
    if value.get("schema") != "cvs.phase1.dualreadout_runtime_parity.v2":
        raise ValueError("runtime parity receipt schema mismatch")
    if str(value.get("checkpoint_sha256", "")).lower() != str(checkpoint_sha256).lower():
        raise ValueError("runtime parity checkpoint binding mismatch")
    if value.get("runtime_sha256") != sha256_file(runtime_path):
        raise ValueError("runtime parity runtime binding mismatch")
    if int(value.get("input_len", -1)) != 256 or value.get("validated_batch_sizes") != [1, 8, 64]:
        raise ValueError("runtime parity shape/batch contract mismatch")
    maximum = float(value.get("max_abs", float("nan")))
    tolerance = float(value.get("tolerance", float("nan")))
    if not np.isfinite(maximum) or not np.isfinite(tolerance) or tolerance <= 0.0 or maximum > tolerance:
        raise ValueError("runtime parity numerical gate failed")
    return dict(value)


def build(args: argparse.Namespace) -> dict[str, Any]:
    angular = _load_feature_npz(args.angular_zid_npz)
    robust = _load_feature_npz(args.robust_zid_npz)
    robust_dom = _load_feature_npz(args.robust_zdom_npz)
    _verify_same_rows(angular, robust)
    _verify_same_rows(robust, robust_dom)
    if str(args.calibration_roles) != "source":
        raise ValueError("--calibration-roles is frozen to source")
    _validate_parity_receipt(
        args.angular_parity_receipt,
        runtime_path=args.angular_runtime,
        checkpoint_sha256=args.angular_checkpoint_sha256,
    )
    _validate_parity_receipt(
        args.robust_parity_receipt,
        runtime_path=args.robust_runtime,
        checkpoint_sha256=args.robust_checkpoint_sha256,
    )
    handles = [value for value in str(args.class_handles).split(",") if value]
    calibration, receipt = fit_source_calibration(
        angular_logits=angular["tx_logits"],
        robust_z_id=robust["features"],
        robust_z_dom=robust_dom["features"],
        robust_logits=robust["tx_logits"],
        tx_ids=robust["tx_ids"].astype(str).tolist(),
        roles=robust["dataset_role"].astype(str).tolist(),
        rx_ids=robust["rx_ids"].astype(str).tolist(),
        day_ids=robust["day_ids"].astype(str).tolist(),
        physical_ids=robust["sig_ids"].astype(str).tolist(),
        class_handles=handles,
        calibration_roles=["source"],
    )
    receipt.update(
        {
            "angular_feature_npz_sha256": sha256_file(args.angular_zid_npz),
            "robust_zid_npz_sha256": sha256_file(args.robust_zid_npz),
            "robust_zdom_npz_sha256": sha256_file(args.robust_zdom_npz),
            "angular_runtime_parity_receipt_sha256": sha256_file(args.angular_parity_receipt),
            "robust_runtime_parity_receipt_sha256": sha256_file(args.robust_parity_receipt),
        }
    )
    manifest = build_bundle(
        args.output_dir,
        angular_runtime=args.angular_runtime,
        robust_runtime=args.robust_runtime,
        calibration=calibration,
        calibration_receipt=receipt,
        angular_checkpoint_sha256=args.angular_checkpoint_sha256,
        robust_checkpoint_sha256=args.robust_checkpoint_sha256,
    )
    loaded = load_bundle(
        args.output_dir,
        expected_content_root_sha256=manifest["content_root_sha256"],
    )
    signals = loaded.evaluate_arrays(
        angular_logits=angular["tx_logits"],
        robust_z_id=robust["features"],
        robust_z_dom=robust_dom["features"],
        robust_logits=robust["tx_logits"],
    )
    source_mask = np.asarray(robust["dataset_role"]).astype(str) == "source"
    smoke = {
        "bundle_content_root_sha256": manifest["content_root_sha256"],
        "rows": int(len(source_mask)),
        "source_rows": int(source_mask.sum()),
        "source_decision_counts": {
            value: int(np.sum(signals["local_decision"][source_mask] == value))
            for value in ("registered", "unknown", "defer")
        },
        "no_proxy_or_target_fit": receipt["threshold_scope"] == "source_joint_correct_only_no_proxy_or_target_tuning",
    }
    smoke_path = Path(args.smoke_receipt)
    if smoke_path.exists():
        raise FileExistsError("refusing to overwrite bundle smoke receipt")
    smoke_path.parent.mkdir(parents=True, exist_ok=True)
    smoke_path.write_text(canonical_json(smoke) + "\n", encoding="utf-8")
    return {"manifest": manifest, "smoke": smoke}


def emit(args: argparse.Namespace) -> dict[str, Any]:
    angular = _load_feature_npz(args.angular_zid_npz)
    robust = _load_feature_npz(args.robust_zid_npz)
    robust_dom = _load_feature_npz(args.robust_zdom_npz)
    _verify_same_rows(angular, robust)
    _verify_same_rows(robust, robust_dom)
    loaded = load_bundle(
        args.bundle,
        expected_content_root_sha256=args.expected_content_root_sha256,
    )
    signals = loaded.evaluate_arrays(
        angular_logits=angular["tx_logits"],
        robust_z_id=robust["features"],
        robust_z_dom=robust_dom["features"],
        robust_logits=robust["tx_logits"],
    )
    count = len(signals["q"])
    limit = count if int(args.max_rows) <= 0 else min(count, int(args.max_rows))
    rows = []
    for index in range(limit):
        node_id = f"RX-{str(robust['rx_ids'][index])}"
        opaque = hashlib.sha256(f"{args.base_manifest_id}|{index}".encode("utf-8")).hexdigest()[:24]
        rows.append(
            seal_local_evidence(
                {
                    "schema_version": LOCAL_EVIDENCE_SCHEMA,
                    "linkage_mode": "proxy_unverified",
                    "proxy_group_id": f"proxy-{opaque}",
                    "satellite_reception_id": f"reception-{opaque}",
                    "node_id": node_id,
                    "base_manifest_id": str(args.base_manifest_id),
                    "bundle_id": loaded.manifest["content_root_sha256"],
                    "class_handles": list(loaded.manifest["class_handles"]),
                    "p_local": signals["p_local"][index].astype(float).tolist(),
                    "q": float(signals["q"][index]),
                    "correlation_group_id": node_id,
                    "delay_ms": 0.0,
                    "deadline_ms": float(args.deadline_ms),
                    "local_decision": str(signals["local_decision"][index]),
                    "local_label": signals["local_label"][index],
                    "reason_code": str(signals["reason_code"][index]),
                    "sealed_at_ms": 0.0,
                    "z_id": robust["features"][index].astype(float).tolist(),
                    "z_dom": robust_dom["features"][index].astype(float).tolist(),
                    "d_class": signals["d_class"][index].astype(float).tolist(),
                    "e_unknown": float(signals["e_unknown"][index]),
                    "js_disagreement": float(signals["js_disagreement"][index]),
                }
            )
        )
    write_jsonl(args.output_jsonl, rows)
    receipt = {
        "schema": "cvs.phase1.dualreadout_local_evidence_emission.v2",
        "evidence_level": "PROXY_UNVERIFIED_PER_RECEPTION_NO_SAME_EVENT_CLAIM",
        "bundle_content_root_sha256": loaded.manifest["content_root_sha256"],
        "rows": len(rows),
        "output_sha256": sha256_file(args.output_jsonl),
        "role_or_truth_written": False,
        "same_event_claim": False,
        "shot_count_semantics": "one_proxy_group_per_reception",
    }
    Path(args.receipt_out).write_text(canonical_json(receipt) + "\n", encoding="utf-8")
    return receipt


def score(args: argparse.Namespace) -> dict[str, Any]:
    feature = _load_feature_npz(args.truth_feature_npz)
    evidence = [json.loads(line) for line in Path(args.evidence_jsonl).read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(evidence) != len(feature["tx_ids"]):
        raise ValueError("evidence/truth-sidecar row count differs")
    known_roles = {value for value in str(args.known_roles).split(",") if value}
    unknown_roles = {value for value in str(args.unknown_roles).split(",") if value}
    if not known_roles or not unknown_roles or known_roles.intersection(unknown_roles):
        raise ValueError("known/unknown role sets must be non-empty and disjoint")
    roles = feature["dataset_role"].astype(str)
    tx_ids = feature["tx_ids"].astype(str)
    known_total = known_correct = 0
    unknown_total = unknown_accept = unknown_reject = unknown_defer = 0
    class_total: dict[str, int] = {}
    class_correct: dict[str, int] = {}
    for index, row in enumerate(evidence):
        if "role" in row or "true_label" in row:
            raise ValueError("sealed evidence contains forbidden truth fields")
        role = roles[index]
        truth = tx_ids[index]
        if role in known_roles:
            known_total += 1
            class_total[truth] = class_total.get(truth, 0) + 1
            correct = row.get("local_decision") == "registered" and row.get("local_label") == truth
            known_correct += int(correct)
            class_correct[truth] = class_correct.get(truth, 0) + int(correct)
        elif role in unknown_roles:
            unknown_total += 1
            unknown_accept += int(row.get("local_decision") == "registered")
            unknown_reject += int(row.get("local_decision") == "unknown")
            unknown_defer += int(row.get("local_decision") == "defer")
    if known_total == 0 or unknown_total == 0:
        raise ValueError("requested scoring roles produced an empty slice")
    per_class = {key: class_correct.get(key, 0) / value for key, value in sorted(class_total.items())}
    result = {
        "schema": "cvs.phase1.dualreadout_bundle_proxy_scorer.v2",
        "evidence_level": "SOURCE_HELD_PROXY_NONDEPLOYMENT_DIAGNOSTIC",
        "threshold_scope": "bundle_frozen_before_truth_sidecar_open",
        "known_roles": sorted(known_roles),
        "unknown_roles": sorted(unknown_roles),
        "known_total": known_total,
        "known_accuracy_reject_defer_as_error": known_correct / known_total,
        "min_class_known_accuracy": min(per_class.values()),
        "per_class_known_accuracy": per_class,
        "unknown_total": unknown_total,
        "unknown_false_accept_rate": unknown_accept / unknown_total,
        "unknown_safe_rejection_rate": unknown_reject / unknown_total,
        "unknown_defer_rate": unknown_defer / unknown_total,
        "formal_phase3_performance": False,
        "same_event_claim": False,
    }
    target = Path(args.output_json)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(canonical_json(result) + "\n", encoding="utf-8")
    return result


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export-runtime")
    export.add_argument("--checkpoint", required=True)
    export.add_argument("--expected-checkpoint-sha256", required=True)
    export.add_argument("--input-len", type=int, default=256)
    export.add_argument("--runtime-out", required=True)
    export.add_argument("--receipt-out", required=True)
    export.add_argument("--device", default="cuda:0")
    export.add_argument("--parity-seed", type=int, default=20260808)
    export.add_argument("--max-abs-tolerance", type=float, default=1e-5)
    build_p = sub.add_parser("build")
    for name in ("angular-zid-npz", "robust-zid-npz", "robust-zdom-npz", "angular-runtime", "robust-runtime", "angular-parity-receipt", "robust-parity-receipt", "angular-checkpoint-sha256", "robust-checkpoint-sha256", "output-dir", "smoke-receipt"):
        build_p.add_argument(f"--{name}", required=True)
    build_p.add_argument("--class-handles", required=True)
    build_p.add_argument("--calibration-roles", default="source")
    emit_p = sub.add_parser("emit-evidence")
    for name in ("bundle", "expected-content-root-sha256", "angular-zid-npz", "robust-zid-npz", "robust-zdom-npz", "base-manifest-id", "output-jsonl", "receipt-out"):
        emit_p.add_argument(f"--{name}", required=True)
    emit_p.add_argument("--deadline-ms", type=float, default=100.0)
    emit_p.add_argument("--max-rows", type=int, default=0)
    score_p = sub.add_parser("score-evidence")
    score_p.add_argument("--evidence-jsonl", required=True)
    score_p.add_argument("--truth-feature-npz", required=True)
    score_p.add_argument("--known-roles", default="source")
    score_p.add_argument("--unknown-roles", required=True)
    score_p.add_argument("--output-json", required=True)
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "export-runtime":
        result = export_runtime(args)
    elif args.command == "build":
        result = build(args)
    elif args.command == "emit-evidence":
        result = emit(args)
    else:
        result = score(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
