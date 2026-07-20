from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from cvsrffi.phase2_runtime_contract import PHASE2_FULL_CONTRACT
from cvsrffi.stage2_diag_cosine_exploration import (
    CANDIDATE_D1_B0_CAP,
    CANDIDATE_D3_SCENARIO_OLDLOCK_NEWFIT,
)
from scripts import run_cvs_somph_diag_row_pipeline as pipeline


def _readonly(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    os.chmod(path, stat.S_IREAD)


def _inputs(tmp_path: Path) -> dict[str, Path]:
    result = {
        "cache": tmp_path / "cache_set.json",
        "authority": tmp_path / "authority",
        "checkpoint": tmp_path / "phase1.pth",
        "runtime": tmp_path / "runtime.pt",
        "method": tmp_path / "method.json",
    }
    result["authority"].mkdir()
    for key in ("cache", "checkpoint", "runtime", "method"):
        result[key].write_bytes(key.encode("ascii"))
    return result


def _fake_build(tmp_path: Path) -> dict:
    offline = tmp_path / "row_material"
    truth = offline / "scorer" / "query_truth.json"
    pair = offline / "scorer" / "registration_pair_staging.json"
    _readonly(truth, b'{"schema":"cvs.phase2.query_truth_sidecar.v2","rows":[]}\n')
    _readonly(pair, b'{"schema":"staging"}\n')
    states = {}
    for state, stage in (("before", "stage2b"), ("after", "stage2c")):
        root = offline / "predictor" / state
        enrollment = root / "enrollment"
        staging = root / "apply"
        seal = offline / "seals" / f"{state}_enrollment.seal.json"
        authority = offline / "seals" / f"{state}_authority.json"
        authority_seal = offline / "seals" / f"{state}_authority.seal.json"
        for directory in (enrollment, staging):
            directory.mkdir(parents=True)
        for path in (seal, authority, authority_seal):
            _readonly(path, state.encode("ascii"))
        states[state] = {
            "stage": stage,
            "enrollment_package_root": str(enrollment),
            "enrollment_package_seal": str(seal),
            "enrollment_package_seal_sha256": (
                "a" * 64 if state == "before" else "b" * 64
            ),
            "apply_staging_root": str(staging),
            "apply_staging_authority": str(authority),
            "apply_staging_authority_seal": str(authority_seal),
            "apply_staging_authority_seal_sha256": (
                "c" * 64 if state == "before" else "d" * 64
            ),
        }
    return {
        "row_handle": "row_" + "1" * 64,
        "row_manifest_sha256": "2" * 64,
        "registration_pair_manifest": str(pair),
        "truth_sidecar": str(truth),
        "states": states,
    }


@pytest.mark.parametrize(
    "candidate",
    [CANDIDATE_D1_B0_CAP, CANDIDATE_D3_SCENARIO_OLDLOCK_NEWFIT],
)
def test_pipeline_orders_head_finalization_predictions_then_truth_join(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, candidate: str
) -> None:
    inputs = _inputs(tmp_path)
    events: list[str] = []
    built = _fake_build(tmp_path)

    def fake_builder(**kwargs):
        events.append("build")
        assert kwargs["query_per_tx"] == 20
        assert kwargs["new_class_count"] == 10
        return built

    def fake_enrollment(**kwargs):
        request = json.loads(Path(kwargs["request_json"]).read_text(encoding="utf-8"))
        state = "before" if "before" in str(kwargs["package_root"]) else "after"
        events.append(f"enroll_{state}")
        assert request == {
            "schema": pipeline.SOMPH_ENROLLMENT_REQUEST_SCHEMA,
            "package_seal_sha256": (
                "a" * 64 if state == "before" else "b" * 64
            ),
            "head_output_leaf": "head_capsule.npz",
            "device": "cpu",
            "support_batch_size": 64,
            **PHASE2_FULL_CONTRACT,
        }
        assert "truth_sidecar" not in request
        assert "evaluation_role" not in request
        head = Path(kwargs["output_root"]) / "head_capsule.npz"
        _readonly(head, f"head-{state}".encode("ascii"))
        return {
            "head_output_leaf": head.name,
            "head_capsule_sha256": (
                "3" * 64 if state == "before" else "4" * 64
            ),
            "enrollment_binding_sha256": (
                "5" * 64 if state == "before" else "6" * 64
            ),
        }

    def fake_finalize(**kwargs):
        state = (
            "before"
            if "before" in str(kwargs["apply_staging_root"])
            else "after"
        )
        events.append(f"finalize_{state}")
        _readonly(Path(kwargs["detached_seal_path"]), f"apply-{state}".encode())
        return {
            "package_root_sha256": (
                "7" * 64 if state == "before" else "8" * 64
            ),
            "package_seal_sha256": (
                "9" * 64 if state == "before" else "0" * 64
            ),
        }

    def fake_pair(**kwargs):
        events.append("final_pair")
        output = Path(kwargs["output_path"])
        _readonly(
            output,
            json.dumps(
                {"schema": "cvs.phase2.somph_registration_pair.v1"}
            ).encode(),
        )
        return {"schema": "cvs.phase2.somph_registration_pair.v1"}

    def fake_diag(**kwargs):
        state = (
            "before"
            if "before" in str(kwargs["enrollment_package_root"])
            else "after"
        )
        events.append(f"diag_{state}")
        assert kwargs["candidate"] == candidate
        diag_root = Path(kwargs["output_root"])
        if (
            candidate == CANDIDATE_D3_SCENARIO_OLDLOCK_NEWFIT
            and state == "after"
        ):
            before_root = diag_root.parent / "before"
            assert Path(kwargs["parent_diag_root"]) == before_root
            assert kwargs["expected_parent_commit_sha256"] == (
                pipeline.sha256_file(before_root / "COMMIT.json")
            )
        else:
            assert "parent_diag_root" not in kwargs
            assert "expected_parent_commit_sha256" not in kwargs
        prediction = diag_root / "prediction_artifact.npz"
        _readonly(prediction, f"prediction-{state}".encode("ascii"))
        prediction_sha256 = pipeline.sha256_file(prediction)
        receipt = diag_root / "execution_receipt.json"
        _readonly(
            receipt,
            json.dumps(
                {
                    "schema":
                    "cvs.phase2.diag_cosine_exploration_receipt.v1",
                    "artifacts": {
                        "prediction_artifact.npz": prediction_sha256
                    },
                },
                sort_keys=True,
            ).encode("utf-8"),
        )
        receipt_sha256 = pipeline.sha256_file(receipt)
        commit = diag_root / "COMMIT.json"
        _readonly(
            commit,
            json.dumps(
                {
                    "schema":
                    "cvs.phase2.diag_cosine_exploration_commit.v1",
                    "execution_receipt_sha256": receipt_sha256,
                    "prediction_artifact_sha256": prediction_sha256,
                    "members": [
                        {
                            "relative_path": "execution_receipt.json",
                            "sha256": receipt_sha256,
                            "size_bytes": receipt.stat().st_size,
                        }
                    ],
                },
                sort_keys=True,
            ).encode("utf-8"),
        )
        return {
            "prediction_artifact_sha256": prediction_sha256
        }

    def fake_score(**kwargs):
        events.append("score")
        assert kwargs["candidate"] == candidate
        assert Path(kwargs["truth_sidecar_path"]) == Path(built["truth_sidecar"])
        for key in ("before_prediction_path", "after_prediction_path"):
            prediction = Path(kwargs[key])
            assert prediction.is_file()
            assert not prediction.stat().st_mode & stat.S_IWUSR
        _readonly(Path(kwargs["output_path"]), b'{"score":"complete"}\n')
        return {"score_artifact_sha256": "f" * 64}

    monkeypatch.setattr(pipeline, "build_somph_offline_row_pair", fake_builder)
    monkeypatch.setattr(pipeline, "run_somph_enrollment", fake_enrollment)
    monkeypatch.setattr(pipeline, "finalize_somph_apply_package", fake_finalize)
    monkeypatch.setattr(
        pipeline, "finalize_registration_pair_manifest", fake_pair
    )
    monkeypatch.setattr(
        pipeline, "run_diag_cosine_exploration", fake_diag
    )
    monkeypatch.setattr(pipeline, "score_diag_cosine_pair", fake_score)

    output = tmp_path / "pipeline"
    result = pipeline.run_pipeline(
        cache_set_manifest_path=inputs["cache"],
        authority_bundle_root=inputs["authority"],
        expected_authority_commit_sha256="e" * 64,
        phase1_checkpoint_path=inputs["checkpoint"],
        sealed_feature_runtime_path=inputs["runtime"],
        method_lock_path=inputs["method"],
        output_root=output,
        receiver="20-1",
        seed=713101,
        k_shot=10,
        new_class_count=10,
        device="cpu",
        candidate=candidate,
    )
    assert events == [
        "build",
        "enroll_before",
        "finalize_before",
        "enroll_after",
        "finalize_after",
        "final_pair",
        "diag_before",
        "diag_after",
        "score",
    ]
    assert result["formal_launch_authority"] is False
    receipt = json.loads(
        (output / "pipeline_receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["truth_sidecar_exposed_to_predictor"] is False
    assert receipt["truth_join_started_after_both_immutable_predictions"] is True
    assert receipt["states"]["before"]["stage"] == "stage2b"
    assert receipt["states"]["after"]["stage"] == "stage2c"
    for state in ("before", "after"):
        diag_root = output / "diag" / state
        assert receipt["states"][state]["diag_commit_sha256"] == (
            pipeline.sha256_file(diag_root / "COMMIT.json")
        )
        assert receipt["states"][state]["execution_receipt_sha256"] == (
            pipeline.sha256_file(diag_root / "execution_receipt.json")
        )
    assert receipt["candidate"] == candidate
    if candidate == CANDIDATE_D3_SCENARIO_OLDLOCK_NEWFIT:
        assert receipt["states"]["after"][
            "parent_before_diag_commit_sha256"
        ] == receipt["states"]["before"]["diag_commit_sha256"]


def test_pipeline_refuses_existing_output_before_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    called = False

    def fake_builder(**_kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(pipeline, "build_somph_offline_row_pair", fake_builder)
    with pytest.raises(FileExistsError):
        pipeline.run_pipeline(
            cache_set_manifest_path=tmp_path / "cache.json",
            authority_bundle_root=tmp_path / "authority",
            expected_authority_commit_sha256="e" * 64,
            phase1_checkpoint_path=tmp_path / "phase1.pth",
            sealed_feature_runtime_path=tmp_path / "runtime.pt",
            method_lock_path=tmp_path / "method.json",
            output_root=output,
            receiver="20-1",
            seed=713101,
            k_shot=5,
            new_class_count=20,
            device="cpu",
        )
    assert called is False


def test_parser_exposes_required_row_inputs() -> None:
    args = pipeline.parser().parse_args(
        [
            "--cache-manifest",
            "cache.json",
            "--authority-bundle",
            "authority",
            "--authority-commit-sha256",
            "a" * 64,
            "--phase1-checkpoint",
            "phase1.pth",
            "--sealed-runtime",
            "runtime.pt",
            "--method-lock",
            "method.json",
            "--output-root",
            "row",
            "--receiver",
            "20-1",
            "--seed",
            "713101",
            "--k-shot",
            "5",
            "--new-count",
            "20",
            "--device",
            "cpu",
        ]
    )
    assert args.k_shot == 5
    assert args.new_count == 20
    assert args.device == "cpu"
    assert args.candidate == pipeline.CANDIDATE_D1

    selected = pipeline.parser().parse_args(
        [
            "--cache-manifest",
            "cache.json",
            "--authority-bundle",
            "authority",
            "--authority-commit-sha256",
            "a" * 64,
            "--phase1-checkpoint",
            "phase1.pth",
            "--sealed-runtime",
            "runtime.pt",
            "--method-lock",
            "method.json",
            "--output-root",
            "row",
            "--receiver",
            "20-1",
            "--seed",
            "713101",
            "--k-shot",
            "10",
            "--new-count",
            "5",
            "--device",
            "cpu",
            "--candidate",
            CANDIDATE_D1_B0_CAP,
        ]
    )
    assert selected.candidate == CANDIDATE_D1_B0_CAP

    d93 = pipeline.parser().parse_args(
        [
            "--cache-manifest",
            "cache.json",
            "--authority-bundle",
            "authority",
            "--authority-commit-sha256",
            "a" * 64,
            "--phase1-checkpoint",
            "checkpoint.pth",
            "--sealed-runtime",
            "runtime.pt",
            "--method-lock",
            "method.json",
            "--output-root",
            "output",
            "--receiver",
            "20-1",
            "--seed",
            "713101",
            "--k-shot",
            "1",
            "--new-count",
            "20",
            "--device",
            "cpu",
            "--candidate",
            "d93_paired_ground_transport_interaction",
            "--ground-component-dir",
            "ground",
            "--ground-manifest-sha256",
            "a" * 64,
        ]
    )
    assert d93.candidate == "d93_paired_ground_transport_interaction"
