from __future__ import annotations

import copy
import hashlib
import subprocess
from pathlib import Path

import pytest

from cvsrffi.full_ablation_spec import build_phase1_t1_rows
from scripts.run_full_ablation_phase1_t1 import Phase1RunnerError
from scripts.seal_full_ablation_phase1_plan import (
    _git_blob_sha256,
    seal_plan,
)


_RELEASE_FILES = {
    "code/example.py": "b" * 64,
    "configs/full_ablation_20260728/seed_registry.json": "c" * 64,
}
_SEED_REGISTRY = {
    "schema": "cvs.full_ablation.seed_registry.v1",
    "design_id": "cvs_full_ablation_phase1_phase2_20260728",
    "phase1_train_seeds": [
        7281101,
        7281102,
        7281103,
        7281104,
        7281105,
    ],
}


def _plan() -> dict:
    return {
        "schema": "cvs.full_ablation.plan.v1",
        "design_id": "cvs_full_ablation_phase1_phase2_20260728",
        "phase": "phase1",
        "stage": "t1",
        "git_commit": "a" * 40,
        "formal_launch_authority": False,
        "seed_registry_sha256": "c" * 64,
        "wisig_pkl_sha256": "d" * 64,
        "python_environment_id": "CVS-RFFI",
        "registered_phase1_train_seeds": list(
            _SEED_REGISTRY["phase1_train_seeds"]
        ),
        "rows": build_phase1_t1_rows(
            [7281101, 7281102, 7281103, 7281104, 7281105],
            git_commit="a" * 40,
        ),
    }


def _review() -> dict:
    return {
        "schema": "cvs.independent_review.v1",
        "git_commit": "a" * 40,
        "p0_count": 0,
        "p1_count": 0,
        "reviewer": "independent-test",
    }


def test_seal_binds_review_commit_config_and_release_hashes() -> None:
    sealed = seal_plan(
        _plan(),
        _review(),
        _SEED_REGISTRY,
        run_id="phase1-t1-v1",
        commit="a" * 40,
        release_files=_RELEASE_FILES,
    )
    assert sealed["formal_launch_authority"] is True
    assert len(sealed["rows"]) == 30
    assert len({row["config_hash"] for row in sealed["rows"]}) == 6
    assert all(row["executor_status"] == "LOCAL_VERIFIED" for row in sealed["rows"])
    assert len(sealed["sealed_content_sha256"]) == 64


def test_seal_rejects_review_findings_or_commit_drift() -> None:
    review = _review()
    review["p1_count"] = 1
    with pytest.raises(Phase1RunnerError, match="P0=0,P1=0"):
        seal_plan(
            _plan(),
            review,
            _SEED_REGISTRY,
            run_id="phase1-t1-v1",
            commit="a" * 40,
            release_files=_RELEASE_FILES,
        )
    review = _review()
    review["git_commit"] = "b" * 40
    with pytest.raises(Phase1RunnerError, match="differs"):
        seal_plan(
            _plan(),
            review,
            _SEED_REGISTRY,
            run_id="phase1-t1-v1",
            commit="a" * 40,
            release_files=_RELEASE_FILES,
        )


def test_seal_rejects_method_hash_drift() -> None:
    plan = copy.deepcopy(_plan())
    plan["rows"][0]["method_config_hash"] = "0" * 64
    with pytest.raises(
        Phase1RunnerError,
        match="canonical row drift|method config hash",
    ):
        seal_plan(
            plan,
            _review(),
            _SEED_REGISTRY,
            run_id="phase1-t1-v1",
            commit="a" * 40,
            release_files=_RELEASE_FILES,
        )


def test_seal_rejects_seed_registry_semantic_drift() -> None:
    registry = copy.deepcopy(_SEED_REGISTRY)
    registry["phase1_train_seeds"][-1] = 9999999
    with pytest.raises(
        Phase1RunnerError,
        match="differ from the sealed seed registry",
    ):
        seal_plan(
            _plan(),
            _review(),
            registry,
            run_id="phase1-t1-v1",
            commit="a" * 40,
            release_files=_RELEASE_FILES,
        )


def test_git_release_hashes_bind_commit_blobs_not_checkout_eol() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert len(commit) == 40
    relative_path = "code/scripts/seal_full_ablation_phase1_plan.py"
    blob = subprocess.run(
        ["git", "show", f"{commit}:{relative_path}"],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
    ).stdout
    assert _git_blob_sha256(repo_root, commit, relative_path) == (
        hashlib.sha256(blob).hexdigest()
    )
