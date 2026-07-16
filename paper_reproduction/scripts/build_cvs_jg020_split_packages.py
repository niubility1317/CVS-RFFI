#!/usr/bin/env python
"""Split a verified Stage2-C bundle into JG020 enrollment/apply packages.

This is a Phase2-external offline builder.  It may inspect the original sealed
support+query bundle and scorer truth, but it creates two runtime roots whose
member role sets are mutually exclusive.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
for value in (str(REPO_ROOT), str(CODE_ROOT)):
    while value in sys.path:
        sys.path.remove(value)
for value in (str(REPO_ROOT), str(CODE_ROOT)):
    sys.path.insert(0, value)

from cvsrffi.stage2_metric_scorer import (  # noqa: E402
    SCORING_MANIFEST_SCHEMA,
    load_verified_scoring_sidecar,
)
from cvsrffi.stage2_predictor_bundle import (  # noqa: E402
    open_regular_member_same_fd as open_source_member,
    preflight_stage2_predictor_package,
    sha256_file as source_sha256_file,
)
from paper_reproduction.cvs_aligned.jg020_stage2c import (  # noqa: E402
    APPLY_PROFILE,
    ENROLLMENT_PROFILE,
    FORMAL_SCENARIOS,
    HEAD_SCHEMA,
    PHASE2_CONTRACT,
    RECEIPT_SCHEMA,
    descriptor_by_role,
    head_npz_members,
    make_member_descriptor,
    preflight_package,
    sha256_file,
    validate_locked_candidate,
    write_package_manifest_and_seal,
)


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _copy_path_new(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.is_file() or destination.exists():
        raise ValueError("split-package file copy must be regular, non-symlink and non-overwriting")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as reader, destination.open("xb") as writer:
        shutil.copyfileobj(reader, writer, length=1024 * 1024)
        writer.flush()
        os.fsync(writer.fileno())


def _copy_source_member_new(source_root: Path, descriptor: Mapping[str, Any], destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open_source_member(source_root, descriptor["relative_path"]) as reader, destination.open("xb") as writer:
        shutil.copyfileobj(reader, writer, length=1024 * 1024)
        writer.flush()
        os.fsync(writer.fileno())
    if source_sha256_file(destination) != descriptor["sha256"]:
        raise ValueError("split-package source member hash drift")


def _load_lock(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("JG020 candidate lock must be a regular non-symlink file")
    return validate_locked_candidate(json.loads(path.read_text(encoding="utf-8-sig")))


def _source_preflight(args: argparse.Namespace):
    source_root = Path(args.source_package_root).resolve(strict=True)
    manifest, seal, audit = preflight_stage2_predictor_package(
        source_root,
        detached_seal_path=Path(args.source_detached_seal),
        expected_seal_sha256=str(args.source_expected_seal_sha256),
    )
    if manifest["stage"] != "stage2c":
        raise ValueError("JG020 requires a sealed Stage2-C source bundle")
    return source_root, manifest, seal, audit


def build_enrollment(args: argparse.Namespace) -> dict[str, Any]:
    source_root, source_manifest, _source_seal, source_audit = _source_preflight(args)
    lock_path = Path(args.candidate_lock).resolve(strict=True)
    lock = _load_lock(lock_path)
    if sha256_file(lock_path) != source_manifest["candidate_lock_sha256"]:
        raise ValueError("source bundle is not bound to the JG020 candidate lock")
    checks = {
        "receiver": source_manifest["receiver"] == lock["receiver"],
        "seed": source_manifest["seed"] == lock["seed"],
        "new_class_count": source_manifest["new_class_count"] == lock["new_class_count"],
        "registered_class_count": source_manifest["registered_class_count"] == 6 + lock["new_class_count"],
        "checkpoint_sha256": sha256_file(args.checkpoint_full) == lock["checkpoint_sha256"],
        "ground_adapter_sha256": sha256_file(args.ground_adapter) == lock["ground_adapter_sha256"],
        "direct_class_mapping_sha256": sha256_file(args.direct_class_mapping)
        == lock["direct_class_mapping_sha256"],
    }
    failed = [key for key, value in checks.items() if not value]
    if failed:
        raise ValueError(f"JG020 source/lock input binding failed: {failed}")
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    copied = {
        "checkpoint_full": output_root / "checkpoint_full.pth",
        "ground_adapter": output_root / "ground_adapter.pt",
        "candidate_lock": output_root / "candidate_lock.json",
        "direct_class_mapping": output_root / "class_registry_map.json",
    }
    _copy_path_new(Path(args.checkpoint_full), copied["checkpoint_full"])
    _copy_path_new(Path(args.ground_adapter), copied["ground_adapter"])
    _copy_path_new(lock_path, copied["candidate_lock"])
    _copy_path_new(Path(args.direct_class_mapping), copied["direct_class_mapping"])
    source_roles = {item["artifact_role"]: item for item in source_manifest["members"]}
    members = [
        make_member_descriptor(copied["checkpoint_full"], role="checkpoint_full", schema="adv3b02.full_checkpoint.v1"),
        make_member_descriptor(copied["ground_adapter"], role="ground_adapter", schema="cvs.p4_projection_lora.v1"),
        make_member_descriptor(copied["candidate_lock"], role="candidate_lock", schema=lock["schema"]),
        make_member_descriptor(
            copied["direct_class_mapping"],
            role="direct_class_mapping",
            schema="cvs.adv3b02.class_mapping.v1",
        ),
    ]
    for scenario in FORMAL_SCENARIOS:
        source_descriptor = source_roles[f"support:{scenario}"]
        destination = output_root / f"support_{scenario}.npz"
        _copy_source_member_new(source_root, source_descriptor, destination)
        members.append(make_member_descriptor(
            destination,
            role=f"support:{scenario}",
            schema=source_descriptor["schema"],
            scenario=scenario,
            npz_members=source_descriptor["npz_members"],
        ))
    metadata = {
        "stage": "stage2c",
        "receiver": lock["receiver"],
        "seed": lock["seed"],
        "k_shot": lock["k_shot"],
        "new_class_count": lock["new_class_count"],
        "registered_class_count": source_manifest["registered_class_count"],
        "registered_classes": source_manifest["registered_classes"],
        "candidate_lock_sha256": sha256_file(lock_path),
        "target_channel_scenarios": list(FORMAL_SCENARIOS),
        "phase2_contract": PHASE2_CONTRACT,
        "lineage": {
            "source_package_root_sha256": source_manifest["package_root_sha256"],
            "source_package_seal_sha256": str(args.source_expected_seal_sha256),
            "enrollment_package_root_sha256": None,
        },
    }
    document, _seal = write_package_manifest_and_seal(
        output_root,
        profile=ENROLLMENT_PROFILE,
        metadata=metadata,
        members=members,
        detached_seal=Path(args.output_detached_seal),
    )
    seal_sha = sha256_file(args.output_detached_seal)
    _manifest, final_audit = preflight_package(
        output_root,
        detached_seal=args.output_detached_seal,
        expected_seal_sha256=seal_sha,
        expected_profile=ENROLLMENT_PROFILE,
    )
    return {
        "status": "PASS",
        "profile": ENROLLMENT_PROFILE,
        "package_root": str(output_root),
        "package_root_sha256": document["package_root_sha256"],
        "detached_seal": str(Path(args.output_detached_seal).resolve()),
        "detached_seal_sha256": seal_sha,
        "source_preflight": source_audit,
        "final_preflight": final_audit,
    }


def build_apply(args: argparse.Namespace) -> dict[str, Any]:
    source_root, source_manifest, _source_seal, source_audit = _source_preflight(args)
    enrollment_manifest, enrollment_audit = preflight_package(
        args.enrollment_package_root,
        detached_seal=args.enrollment_detached_seal,
        expected_seal_sha256=args.enrollment_expected_seal_sha256,
        expected_profile=ENROLLMENT_PROFILE,
    )
    lock_path = Path(args.candidate_lock).resolve(strict=True)
    lock = _load_lock(lock_path)
    if enrollment_manifest["candidate_lock_sha256"] != sha256_file(lock_path):
        raise ValueError("enrollment package/candidate lock binding drift")
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    artifact_specs = (
        ("candidate_lock", lock_path, "candidate_lock.json", lock["schema"], ()),
        ("candidate_runtime", Path(args.candidate_runtime), "candidate_runtime.ts", "adv3b02.p4_jg020.torchscript.v1", ()),
        ("identity_runtime", Path(args.identity_runtime), "identity_runtime.ts", "adv3b02.p4_identity.torchscript.v1", ()),
        ("direct_runtime", Path(args.direct_runtime), "direct_runtime.ts", "adv3b02.direct.torchscript.v1", ()),
        ("prototype_head", Path(args.prototype_head), "prototype_head.npz", HEAD_SCHEMA, tuple(head_npz_members())),
        ("enrollment_receipt", Path(args.enrollment_receipt), "enrollment_receipt.json", RECEIPT_SCHEMA, ()),
    )
    members = []
    for role, source, filename, schema, npz_members in artifact_specs:
        destination = output_root / filename
        _copy_path_new(source, destination)
        members.append(make_member_descriptor(
            destination, role=role, schema=schema, npz_members=npz_members
        ))
    source_roles = {item["artifact_role"]: item for item in source_manifest["members"]}
    for scenario in FORMAL_SCENARIOS:
        source_descriptor = source_roles[f"query:{scenario}"]
        destination = output_root / f"query_{scenario}.npz"
        _copy_source_member_new(source_root, source_descriptor, destination)
        members.append(make_member_descriptor(
            destination,
            role=f"query:{scenario}",
            schema=source_descriptor["schema"],
            scenario=scenario,
            npz_members=source_descriptor["npz_members"],
        ))
    metadata = {
        "stage": "stage2c",
        "receiver": lock["receiver"],
        "seed": lock["seed"],
        "k_shot": lock["k_shot"],
        "new_class_count": lock["new_class_count"],
        "registered_class_count": source_manifest["registered_class_count"],
        "registered_classes": source_manifest["registered_classes"],
        "candidate_lock_sha256": sha256_file(lock_path),
        "target_channel_scenarios": list(FORMAL_SCENARIOS),
        "phase2_contract": PHASE2_CONTRACT,
        "lineage": {
            "source_package_root_sha256": source_manifest["package_root_sha256"],
            "source_package_seal_sha256": str(args.source_expected_seal_sha256),
            "enrollment_package_root_sha256": enrollment_manifest["package_root_sha256"],
        },
    }
    document, _seal = write_package_manifest_and_seal(
        output_root,
        profile=APPLY_PROFILE,
        metadata=metadata,
        members=members,
        detached_seal=args.output_detached_seal,
    )
    seal_sha = sha256_file(args.output_detached_seal)
    _manifest, final_audit = preflight_package(
        output_root,
        detached_seal=args.output_detached_seal,
        expected_seal_sha256=seal_sha,
        expected_profile=APPLY_PROFILE,
    )
    original_truth, _original_scoring, truth_audit = load_verified_scoring_sidecar(
        Path(args.source_scoring_manifest)
    )
    if original_truth["receiver"] != lock["receiver"] or original_truth["seed"] != lock["seed"]:
        raise ValueError("source truth sidecar does not match JG020 cell")
    scorer_root = Path(args.scorer_out_root).resolve()
    scorer_root.mkdir(parents=True, exist_ok=False)
    truth_source = Path(truth_audit["truth_sidecar"])
    truth_destination = scorer_root / "truth_sidecar.json"
    _copy_path_new(truth_source, truth_destination)
    scoring_manifest = {
        "schema": SCORING_MANIFEST_SCHEMA,
        "predictor_package_root_sha256": document["package_root_sha256"],
        "predictor_package_seal_sha256": seal_sha,
        "truth_sidecar_json": truth_destination.name,
        "truth_sidecar_sha256": sha256_file(truth_destination),
        "scorer_output_must_not_feed_predictor": True,
    }
    scoring_path = scorer_root / "scoring_manifest.json"
    _write_json_new(scoring_path, scoring_manifest)
    load_verified_scoring_sidecar(scoring_path)
    return {
        "status": "PASS",
        "profile": APPLY_PROFILE,
        "package_root": str(output_root),
        "package_root_sha256": document["package_root_sha256"],
        "detached_seal": str(Path(args.output_detached_seal).resolve()),
        "detached_seal_sha256": seal_sha,
        "scoring_manifest": str(scoring_path),
        "source_preflight": source_audit,
        "enrollment_preflight": enrollment_audit,
        "final_preflight": final_audit,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("enrollment", "apply"):
        child = subparsers.add_parser(name)
        child.add_argument("--source-package-root", type=Path, required=True)
        child.add_argument("--source-detached-seal", type=Path, required=True)
        child.add_argument("--source-expected-seal-sha256", required=True)
        child.add_argument("--candidate-lock", type=Path, required=True)
        child.add_argument("--output-root", type=Path, required=True)
        child.add_argument("--output-detached-seal", type=Path, required=True)
    enrollment = subparsers.choices["enrollment"]
    enrollment.add_argument("--checkpoint-full", type=Path, required=True)
    enrollment.add_argument("--ground-adapter", type=Path, required=True)
    enrollment.add_argument("--direct-class-mapping", type=Path, required=True)
    apply = subparsers.choices["apply"]
    apply.add_argument("--enrollment-package-root", type=Path, required=True)
    apply.add_argument("--enrollment-detached-seal", type=Path, required=True)
    apply.add_argument("--enrollment-expected-seal-sha256", required=True)
    apply.add_argument("--candidate-runtime", type=Path, required=True)
    apply.add_argument("--identity-runtime", type=Path, required=True)
    apply.add_argument("--direct-runtime", type=Path, required=True)
    apply.add_argument("--prototype-head", type=Path, required=True)
    apply.add_argument("--enrollment-receipt", type=Path, required=True)
    apply.add_argument("--source-scoring-manifest", type=Path, required=True)
    apply.add_argument("--scorer-out-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build_enrollment(args) if args.command == "enrollment" else build_apply(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
