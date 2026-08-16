from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

import pytest

from scripts import run_d92_ccoc_g0
from cvsrffi import stage2_d92_e0d_query_evaluation as e0d


def _minimal_args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        outer_key=run_d92_ccoc_g0.G0_OUTER_KEY,
        reference_arm="E0_FULL_ONLY",
        candidate_arm="E0_FULL_CROSS_CLASS_OFFBLOCK_CONSENSUS",
        reference_output_root=str(tmp_path / "reference"),
        candidate_output_root=str(tmp_path / "candidate"),
        before_enrollment_package_root="before-enrollment",
        before_enrollment_seal_path="before-enrollment.seal",
        before_enrollment_seal_sha256="a" * 64,
        before_apply_package_root="before-apply",
        before_apply_seal_path="before-apply.seal",
        before_apply_seal_sha256="b" * 64,
        after_enrollment_package_root="after-enrollment",
        after_enrollment_seal_path="after-enrollment.seal",
        after_enrollment_seal_sha256="c" * 64,
        after_apply_package_root="after-apply",
        after_apply_seal_path="after-apply.seal",
        after_apply_seal_sha256="d" * 64,
        ground_component_dir="ground",
        ground_manifest_sha256="e" * 64,
        device="cuda:0",
    )


def test_parser_freezes_outer_and_two_arms() -> None:
    parser = run_d92_ccoc_g0.parser()
    args = parser.parse_args(
        [
            "--before-enrollment-package-root",
            "before-enrollment",
            "--before-enrollment-seal-path",
            "before-enrollment.seal",
            "--before-enrollment-seal-sha256",
            "a" * 64,
            "--before-apply-package-root",
            "before-apply",
            "--before-apply-seal-path",
            "before-apply.seal",
            "--before-apply-seal-sha256",
            "b" * 64,
            "--after-enrollment-package-root",
            "after-enrollment",
            "--after-enrollment-seal-path",
            "after-enrollment.seal",
            "--after-enrollment-seal-sha256",
            "c" * 64,
            "--after-apply-package-root",
            "after-apply",
            "--after-apply-seal-path",
            "after-apply.seal",
            "--after-apply-seal-sha256",
            "d" * 64,
            "--ground-component-dir",
            "ground",
            "--ground-manifest-sha256",
            "e" * 64,
            "--reference-output-root",
            "reference_e0",
            "--candidate-output-root",
            "candidate_ccoc",
            "--device",
            "cuda:0",
        ]
    )

    assert args.outer_key == run_d92_ccoc_g0.G0_OUTER_KEY
    assert args.reference_arm == "E0_FULL_ONLY"
    assert (
        args.candidate_arm
        == "E0_FULL_CROSS_CLASS_OFFBLOCK_CONSENSUS"
    )


def test_parser_does_not_expose_truth_or_score_switches() -> None:
    help_text = run_d92_ccoc_g0.parser().format_help().lower()

    assert "truth" not in help_text
    assert "score" not in help_text


def test_existing_output_root_is_rejected(tmp_path) -> None:
    parser = run_d92_ccoc_g0.parser()
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    reference.mkdir()
    args = argparse.Namespace(
        outer_key=run_d92_ccoc_g0.G0_OUTER_KEY,
        reference_arm="E0_FULL_ONLY",
        candidate_arm="E0_FULL_CROSS_CLASS_OFFBLOCK_CONSENSUS",
        reference_output_root=str(reference),
        candidate_output_root=str(candidate),
        before_enrollment_package_root="before-enrollment",
        before_enrollment_seal_path="before-enrollment.seal",
        before_enrollment_seal_sha256="a" * 64,
        before_apply_package_root="before-apply",
        before_apply_seal_path="before-apply.seal",
        before_apply_seal_sha256="b" * 64,
        after_enrollment_package_root="after-enrollment",
        after_enrollment_seal_path="after-enrollment.seal",
        after_enrollment_seal_sha256="c" * 64,
        after_apply_package_root="after-apply",
        after_apply_seal_path="after-apply.seal",
        after_apply_seal_sha256="d" * 64,
        ground_component_dir="ground",
        ground_manifest_sha256="e" * 64,
        device="cuda:0",
    )

    with pytest.raises(run_d92_ccoc_g0.D92CCOCG0EntryError, match="overwrite"):
        run_d92_ccoc_g0.run(args)


def test_run_joins_persisted_fit_audit_and_emits_marker(tmp_path, monkeypatch) -> None:
    reference_root = tmp_path / "reference"
    candidate_root = tmp_path / "candidate"
    validation_path = tmp_path / "g0_validation.json"
    args = argparse.Namespace(
        outer_key=run_d92_ccoc_g0.G0_OUTER_KEY,
        reference_arm="E0_FULL_ONLY",
        candidate_arm="E0_FULL_CROSS_CLASS_OFFBLOCK_CONSENSUS",
        reference_output_root=str(reference_root),
        candidate_output_root=str(candidate_root),
        g0_validation_path=str(validation_path),
        before_enrollment_package_root="before-enrollment",
        before_enrollment_seal_path="before-enrollment.seal",
        before_enrollment_seal_sha256="a" * 64,
        before_apply_package_root="before-apply",
        before_apply_seal_path="before-apply.seal",
        before_apply_seal_sha256="b" * 64,
        after_enrollment_package_root="after-enrollment",
        after_enrollment_seal_path="after-enrollment.seal",
        after_enrollment_seal_sha256="c" * 64,
        after_apply_package_root="after-apply",
        after_apply_seal_path="after-apply.seal",
        after_apply_seal_sha256="d" * 64,
        ground_component_dir="ground",
        ground_manifest_sha256="e" * 64,
        device="cuda:0",
    )

    def fake_run(*, arm_id, output_root, technical_support_receipt_sink, **_kwargs):
        output = Path(output_root)
        (output / "after").mkdir(parents=True)
        fit_rows = []
        for index, scene in enumerate(run_d92_ccoc_g0.G0_SCENES):
            technical_support_receipt_sink(
                {
                    "scene": scene,
                    "canonical_support_identity_sha256": "support-sha",
                    "canonical_class_handles": ["old-0", "new-0"],
                    "canonical_support_handles": ["row-0"],
                    "cross_group_margin_by_support_handle": [
                        {
                            "canonical_row_handle": "row-0",
                            "cross_group_margin": 1.0
                            if arm_id == "E0_FULL_ONLY"
                            else 3.0,
                        }
                    ],
                    "support_block_absmax": {"z160": 2.0, "fft96": 1.0, "rf32": 1.0},
                    "scale1_block_max_abs": [1.0, 1.0, 1.0],
                    "scale2_block_max_abs": [1.0, 1.0, 1.0],
                    "state_fingerprint_sha256": (
                        "e0-state" if arm_id == "E0_FULL_ONLY" else "ccoc-state"
                    ),
                    "query_access": False,
                    "truth_access": False,
                }
            )
            fit_rows.append(
                {
                    "scenario": scene,
                    "after_actual_component_inventory": {
                        "actual_component_fit_count": 1
                    },
                    "after_state_bytes": 456,
                    "query_macs": 123,
                    "after_registration_resource": {
                        "registration_wall_time_ns": 100_000_000
                        if arm_id == "E0_FULL_ONLY"
                        else 120_000_000,
                        "registration_incremental_peak_working_set_bytes": 100
                        if arm_id == "E0_FULL_ONLY"
                        else 524_288,
                    },
                    "d92_e0d_ccoc_active": arm_id != "E0_FULL_ONLY",
                    "d92_e0d_ccoc_fallback_active": False,
                    "d92_e0d_ccoc_old_rho": 0.5,
                    "d92_e0d_ccoc_new_rho": 1.0,
                }
            )
        with (output / "after" / "fit_audit.json").open(
            "w", encoding="utf-8", newline="\n"
        ) as handle:
            json.dump(fit_rows, handle)
        return {"schema": "fake"}

    monkeypatch.setattr(e0d, "run_d92_e0d_query_evaluation", fake_run)
    before = e0d._CCOC_ARM_IDS
    result = run_d92_ccoc_g0.run(args)

    assert result["marker"] == run_d92_ccoc_g0.G0_MARKER
    assert json.loads(validation_path.read_text(encoding="utf-8"))["status"] == (
        run_d92_ccoc_g0.G0_MARKER
    )
    assert e0d._CCOC_ARM_IDS == before


def test_reference_arm_uses_formal_sink_without_private_ccoc_hooks(
    tmp_path, monkeypatch
) -> None:
    args = _minimal_args(tmp_path)
    observed: list[tuple[str, object]] = []

    monkeypatch.setattr(
        e0d,
        "_CCOC_ARM_IDS",
        frozenset({run_d92_ccoc_g0.CANDIDATE_ARM}),
    )
    monkeypatch.delattr(e0d, "_ccoc_support_receipt", raising=False)

    def fake_run(*, arm_id, technical_support_receipt_sink, **_kwargs):
        assert arm_id not in e0d._CCOC_ARM_IDS
        assert not hasattr(e0d, "_ccoc_support_receipt")
        observed.append((arm_id, technical_support_receipt_sink))
        return {"schema": "fake"}

    monkeypatch.setattr(e0d, "run_d92_e0d_query_evaluation", fake_run)
    receipts: list[dict] = []

    run_d92_ccoc_g0._run_arm(
        args,
        arm_id=run_d92_ccoc_g0.REFERENCE_ARM,
        output_root=tmp_path / "reference",
        receipts=receipts,
    )

    assert observed[0][0] == run_d92_ccoc_g0.REFERENCE_ARM
    assert observed[0][1].__self__ is receipts
    assert observed[0][1].__name__ == "append"


def test_prereg_report_has_no_prefilled_runtime_pass_claim() -> None:
    report_path = (
        Path(__file__).parents[1]
        / "automation_reports/CV-SincNet/"
        "d92_e0_full_ccoc_g0_k10_20260813_v1/report.md"
    )
    text = report_path.read_text(encoding="utf-8")

    assert (
        "LOCAL_VERIFIED_READY_FOR_N607_HANDOFF / "
        "NO_G0_RUNTIME_RESULT / NO_PERFORMANCE_RESULT"
    ) in text
    assert "G0_MECHANISM_RESOURCE_PASS" not in text
    assert "expected_marker" in text
    assert "D92_CCOC_G0_ACTIVE_QUANTUM_RESOURCE_PASS" in text


def test_source_manifest_includes_tracked_cvsrffi_init() -> None:
    root = Path(__file__).parents[1]
    manifest_path = root / "code/CCOC_G0_SOURCE_MANIFEST.sha256"
    expected_sha256 = (
        "13cc5247133854c79ed160269ee8fa9816cb8dae3d162e724ad86d0ad8fad7a2"
    )

    tracked_init = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "show",
            "HEAD:code/cvsrffi/__init__.py",
        ],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    assert hashlib.sha256(tracked_init).hexdigest() == expected_sha256
    entries = {
        line.split(maxsplit=1)[1]: line.split(maxsplit=1)[0]
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    }
    assert entries["code/cvsrffi/__init__.py"] == expected_sha256


def test_v5_launch_freezes_all_four_seal_sha256_arguments() -> None:
    root = Path(__file__).parents[1]
    launch_path = (
        root
        / "automation_reports/CV-SincNet/"
        "d92_e0_full_ccoc_g0_k10_20260816_v5/launch.sh"
    )
    assert launch_path.is_file(), f"missing v5 launch artifact: {launch_path}"
    text = launch_path.read_text(encoding="utf-8")
    expected = {
        "--before-enrollment-seal-sha256": (
            "e3da38668a1e6ec4053e65e669e2a1845bb43198891644d830950e4550b5cea9"
        ),
        "--before-apply-seal-sha256": (
            "736852188c32255647b8105bc7a68d4cc92ca73615e4734d0ed5f4bdd0f04473"
        ),
        "--after-enrollment-seal-sha256": (
            "2600a21ee9a2f95a8d17fa1f4d2263b0e04d243424e3257474502953ed6d9286"
        ),
        "--after-apply-seal-sha256": (
            "afbdc2ebae59fcc311b0cd44aafd27898d7c4af65c9ef03c1085154c8d13020a"
        ),
    }
    for option, expected_sha256 in expected.items():
        matches = re.findall(rf"{re.escape(option)}\s+([0-9a-f]+)", text)
        assert len(matches) == 1, (option, matches)
        actual_sha256 = matches[0]
        assert re.fullmatch(r"[0-9a-f]{64}", actual_sha256)
        assert actual_sha256 == expected_sha256


def test_v6_launch_checks_the_persisted_g0_marker_shape() -> None:
    root = Path(__file__).parents[1]
    launch_path = (
        root
        / "automation_reports/CV-SincNet/"
        "d92_e0_full_ccoc_g0_k10_20260816_v6/launch.sh"
    )
    assert launch_path.is_file(), f"missing v6 launch artifact: {launch_path}"
    text = launch_path.read_text(encoding="utf-8")

    assert 'value.get("status") == "D92_CCOC_G0_ACTIVE_QUANTUM_RESOURCE_PASS"' in text
    assert 'validation.get("marker") == "D92_CCOC_G0_ACTIVE_QUANTUM_RESOURCE_PASS"' in text
    assert 'value.get("marker") == "D92_CCOC_G0_ACTIVE_QUANTUM_RESOURCE_PASS"' not in text
