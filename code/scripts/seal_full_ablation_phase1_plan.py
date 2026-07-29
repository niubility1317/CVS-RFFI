#!/usr/bin/env python3
"""Seal a reviewed Phase1 plan; this does not launch any experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

from cvsrffi.phase1_ablation_factory import (
    apply_phase1_ablation,
    phase1_ablation_config_hash,
)
from scripts.run_full_ablation_phase1_t1 import (
    Phase1RunnerError,
    validate_phase1_release_plan,
)
from SSDG.train_ssdg import build_arg_parser


RELEASE_RELATIVE_PATHS = (
    "code/SSDG/train_ssdg.py",
    "code/model_dual_cvsincnet.py",
    "code/post_stage_common.py",
    "code/cvsrffi/full_ablation_spec.py",
    "code/cvsrffi/phase1_ablation_factory.py",
    "code/cvsrffi/phase2_prototypes.py",
    "code/scripts/build_full_ablation_plan.py",
    "code/scripts/reexport_phase1_prototypes.py",
    "code/scripts/run_full_ablation_phase1_t1.py",
    "code/scripts/seal_full_ablation_phase1_plan.py",
    "configs/full_ablation_20260728/phase1_t1_reuse_v5.json",
    "configs/full_ablation_20260728/seed_registry.json",
)


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git_blob_sha256(
    repo_root: Path,
    commit: str,
    relative_path: str,
) -> str:
    blob = subprocess.run(
        ["git", "show", f"{commit}:{relative_path}"],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
    ).stdout
    return hashlib.sha256(blob).hexdigest()


def _arm_resolved_hash(
    ablation_id: str,
    commit: str,
    wisig_pkl_sha256: str,
) -> str:
    args = build_arg_parser().parse_args(
        [
            "--output_dir",
            "SEALED_RUNTIME_OUTPUT",
            "--formal_ablation",
            "true",
            "--ablation_id",
            ablation_id,
            "--candidate_id",
            ablation_id,
            "--run_id",
            "SEALED_RUNTIME_RUN",
            "--git_commit",
            commit,
            "--wisig_pkl_sha256",
            wisig_pkl_sha256,
            "--dry_run",
        ]
    )
    return str(apply_phase1_ablation(args)["config_hash"])


def seal_plan(
    plan: Mapping[str, Any],
    review: Mapping[str, Any],
    seed_registry: Mapping[str, Any],
    *,
    run_id: str,
    commit: str,
    release_files: Mapping[str, str],
) -> dict[str, Any]:
    validate_phase1_release_plan(plan, require_launch_authority=False)
    registered_seeds = [
        int(value)
        for value in list(
            seed_registry.get("phase1_train_seeds") or []
        )
    ]
    if (
        seed_registry.get("schema")
        != "cvs.full_ablation.seed_registry.v1"
        or seed_registry.get("design_id") != plan.get("design_id")
        or registered_seeds
        != list(plan.get("registered_phase1_train_seeds") or [])
    ):
        raise Phase1RunnerError(
            "plan Phase1 seeds differ from the sealed seed registry"
        )
    if review.get("schema") != "cvs.independent_review.v1":
        raise Phase1RunnerError("unexpected independent review schema")
    if str(review.get("git_commit", "")).lower() != str(commit).lower():
        raise Phase1RunnerError("reviewed commit differs from release commit")
    if int(review.get("p0_count", -1)) != 0 or int(
        review.get("p1_count", -1)
    ) != 0:
        raise Phase1RunnerError("release review is not P0=0,P1=0")
    if not str(run_id).strip():
        raise Phase1RunnerError("run_id is required")
    if not release_files:
        raise Phase1RunnerError("release file hashes are required")
    wisig_pkl_sha256 = str(
        plan.get("wisig_pkl_sha256", "")
    ).lower()
    if len(wisig_pkl_sha256) != 64 or any(
        char not in "0123456789abcdef"
        for char in wisig_pkl_sha256
    ):
        raise Phase1RunnerError(
            "Phase1 plan lacks immutable WiSig SHA256"
        )
    seed_registry_relative = (
        "configs/full_ablation_20260728/seed_registry.json"
    )
    if (
        seed_registry_relative not in release_files
        or str(plan.get("seed_registry_sha256", "")).lower()
        != str(release_files[seed_registry_relative]).lower()
    ):
        raise Phase1RunnerError(
            "plan seed-registry hash differs from reviewed release file"
        )
    sealed = json.loads(json.dumps(dict(plan)))
    sealed["run_id"] = str(run_id)
    sealed["git_commit"] = str(commit).lower()
    sealed["formal_launch_authority"] = True
    sealed["release_gate"] = "P0_0_P1_0_LOCAL_VERIFIED"
    sealed["independent_review"] = dict(review)
    sealed["release_files"] = dict(sorted(release_files.items()))
    for row in sealed["rows"]:
        arm_id = str(row["ablation_id"])
        method_hash = phase1_ablation_config_hash(arm_id)
        if row.get("method_config_hash") != method_hash:
            raise Phase1RunnerError(f"method config hash drift: {arm_id}")
        row["git_commit"] = str(commit).lower()
        row["config_hash"] = _arm_resolved_hash(
            arm_id,
            str(commit).lower(),
            wisig_pkl_sha256,
        )
        row["executor_status"] = "LOCAL_VERIFIED"
    sealed["sealed_content_sha256"] = _canonical_hash(
        {
            key: value
            for key, value in sealed.items()
            if key != "sealed_content_sha256"
        }
    )
    validate_phase1_release_plan(sealed, require_launch_authority=True)
    return sealed


def _git_release_state(repo_root: Path) -> tuple[str, dict[str, str]]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip().lower()
    tracked_status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if tracked_status:
        raise Phase1RunnerError("tracked working tree is not clean")
    hashes: dict[str, str] = {}
    for relative_path in RELEASE_RELATIVE_PATHS:
        path = repo_root / relative_path
        if not path.is_file():
            raise Phase1RunnerError(f"release file missing: {relative_path}")
        hashes[relative_path] = _git_blob_sha256(
            repo_root,
            commit,
            relative_path,
        )
    return commit, hashes


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--review", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite sealed plan: {output}")
    repo_root = Path(args.repo_root).resolve()
    commit, release_files = _git_release_state(repo_root)
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8-sig"))
    review = json.loads(Path(args.review).read_text(encoding="utf-8-sig"))
    seed_registry = json.loads(
        (
            repo_root
            / "configs/full_ablation_20260728/seed_registry.json"
        ).read_text(encoding="utf-8-sig")
    )
    sealed = seal_plan(
        plan,
        review,
        seed_registry,
        run_id=args.run_id,
        commit=commit,
        release_files=release_files,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(sealed, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "run_id": sealed["run_id"],
                "git_commit": sealed["git_commit"],
                "row_count": len(sealed["rows"]),
                "formal_launch_authority": True,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
