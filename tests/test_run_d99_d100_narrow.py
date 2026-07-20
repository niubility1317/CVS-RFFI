import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from scripts import run_d99_d100_narrow as runner


def _prediction(path: Path, rows: list[tuple[str, str, str]]) -> None:
    np.savez(
        path,
        query_tokens=np.asarray([row[0] for row in rows]),
        scenarios=np.asarray([row[1] for row in rows]),
        predicted_class_handles=np.asarray([row[2] for row in rows]),
    )


def test_detailed_score_keeps_same_scenario_joint_metrics_and_confusion(tmp_path: Path) -> None:
    scenarios = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
    truth_rows = []
    before_rows = []
    after_rows = []
    for index, scenario in enumerate(scenarios):
        old_token = f"old-{index}"
        new_token = f"new-{index}"
        truth_rows.extend(
            [
                {
                    "query_token": old_token,
                    "true_class_handle": "old",
                    "transmitter_label": "tx-old",
                    "evaluation_role": "target_old",
                },
                {
                    "query_token": new_token,
                    "true_class_handle": "new",
                    "transmitter_label": "tx-new",
                    "evaluation_role": "target_new",
                },
            ]
        )
        before_rows.append((old_token, scenario, "old"))
        after_rows.extend([(old_token, scenario, "new"), (new_token, scenario, "new")])
    before = tmp_path / "before.npz"
    after = tmp_path / "after.npz"
    truth = tmp_path / "truth.json"
    _prediction(before, before_rows)
    _prediction(after, after_rows)
    truth.write_text(
        json.dumps({"schema": "cvs.phase2.query_truth_sidecar.v2", "rows": truth_rows}),
        encoding="utf-8",
    )
    result = runner._detailed_score(before, after, truth)
    assert len(result["rows"]) == 3
    for row in result["rows"]:
        assert row["old_acc_before_increment"] == 1.0
        assert row["old_acc_after_increment"] == 0.0
        assert row["seen_new_acc"] == 1.0
        assert row["average_forgetting"] == 1.0
        assert row["min_old_class_acc_after"] == 0.0
        assert row["min_new_class_acc_after"] == 1.0
        assert row["min_registered_class_acc_after"] == 0.0
        assert row["registered_balanced_accuracy_after"] == 0.5
        assert row["after_all_confusion_matrix_counts"]["old"]["new"] == 1


def test_narrow_matrix_and_cpu_thread_contract() -> None:
    configured = runner._configure_threads(3)
    assert configured["environment"]["OMP_NUM_THREADS"] == "3"
    assert configured["environment"]["CVSRFFI_CPU_INTEROP_THREADS"] == "1"
    assert configured["torch_num_threads"] == 3
    assert configured["torch_num_interop_threads"] == 1
    with pytest.raises(runner.D99D100NarrowRunnerError, match="positive"):
        runner._configure_threads(0)


def test_parser_exposes_existing_row_inputs_and_frozen_method_inputs() -> None:
    options = {action.dest for action in runner.parser()._actions}
    assert {
        "cache_manifest",
        "authority_bundle",
        "authority_commit_sha256",
        "phase1_checkpoint",
        "sealed_runtime",
        "method_lock",
        "d81_ground_component_dir",
        "d81_ground_manifest_sha256",
        "d99_ground_bundle_npz",
        "d99_ground_manifest",
        "base_d99_lock",
        "phase1_lodo_json",
        "class_binding_json",
        "class_binding_sha256",
        "output_root",
        "receiver",
        "seed",
        "k_shot",
        "new_count",
        "device",
        "cpu_threads",
    } <= options


def test_real_d19_v1_shape_is_deterministically_typed_and_raw_sha_bound() -> None:
    handles = (
        "cls_75aa6d506081240f50cf3b79a0bd91714fa0084a635a472ca63194e57ec1dca2",
        "cls_8b02d99905a8fe579368ac8e37eff51c505aaa89a646eba8892d5d800aa08416",
        "cls_1f33441efa14970113b27483344b7df852a9041984b38d34ce570fafbab6689c",
        "cls_f8dfc2edcccc5344f8e2535a959f13b53a1cddfd6fb22aed6e714de382b58d24",
        "cls_a53ca1280d8fca58e3f4d6d1e9ddabfdab6027a941ee8c3f8c01d9d8ec945725",
        "cls_33bbd16556c6e6305d1b7162f5ea71393afba910a922f9abca5999d5921a2d9d",
    )
    raw_payload = {
        "schema": "cvs.phase2.d19_adv3b02_class_binding.v1",
        "checkpoint_sha256": (
            "2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98"
        ),
        "entries": [
            {
                "class_index": index,
                "phase1_tx": tx,
                "registered_class_handle": handles[index],
            }
            for index, tx in enumerate(
                ("14-10", "14-7", "20-15", "20-19", "6-15", "8-20")
            )
        ],
        "evidence": {
            "phase1_prototype_metadata_checkpoint_path": (
                "/home/szu2070436088/2510044040/CV-SincNet/runs/"
                "phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/"
                "best_joint_safe_ssdg.pth"
            ),
            "phase1_prototype_artifact_sha256": (
                "e6ef79ce0c002539317efa79c1aac605ecb660003190c41b8a4481a9a4affcbd"
            ),
            "phase2_registration_mapping_source": (
                "offline_scorer_truth_sidecar_before_predictor_boundary"
            ),
            "query_truth_exposed_to_predictor": False,
        },
    }
    raw = json.dumps(raw_payload, indent=2).encode("utf-8")
    typed = runner._typed_class_binding_payload(raw_payload)
    assert typed["schema"] == "cvs.phase2.d20_adv3b02_class_binding.v2"
    assert [row["direct_logit_index"] for row in typed["entries"]] == list(range(6))
    from cvsrffi import stage2_d99_d100_query_evaluation as evaluation

    tx_to_handle, _ = evaluation.class_binding_maps(
        typed,
        payload_sha256=__import__("hashlib").sha256(raw).hexdigest(),
        payload_bytes=raw,
        checkpoint_sha256=(
            "2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98"
        ),
        old_handles=handles,
    )
    assert tuple(tx_to_handle) == (
        "14-10",
        "14-7",
        "20-15",
        "20-19",
        "6-15",
        "8-20",
    )


@pytest.mark.parametrize("cli_tokens", (("--cpu-threads", "3"), ("--cpu-threads=4",)))
def test_cpu_thread_bootstrap_accepts_both_argparse_forms(cli_tokens) -> None:
    code_root = Path(runner.__file__).resolve().parents[1]
    probe = (
        "import json,os; from scripts import run_d99_d100_narrow as r; "
        "print(json.dumps([r._BOOTSTRAP_CPU_THREADS,os.environ['OMP_NUM_THREADS']]))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe, *cli_tokens],
        cwd=code_root,
        check=True,
        capture_output=True,
        text=True,
    )
    expected = int(cli_tokens[-1].split("=", 1)[-1])
    assert json.loads(completed.stdout.strip()) == [expected, str(expected)]


def test_cli_main_forwards_class_binding_and_fixed_matrix(monkeypatch, capsys) -> None:
    captured = {}

    def fake_run(args):
        captured.update(vars(args))
        return {"status": "ok"}

    monkeypatch.setattr(runner, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_d99_d100_narrow.py",
            "--cache-manifest",
            "cache.json",
            "--authority-bundle",
            "authority",
            "--authority-commit-sha256",
            "a" * 64,
            "--phase1-checkpoint",
            "checkpoint.pt",
            "--sealed-runtime",
            "runtime.json",
            "--method-lock",
            "method.json",
            "--d81-ground-component-dir",
            "d81",
            "--d81-ground-manifest-sha256",
            "b" * 64,
            "--d99-ground-bundle-npz",
            "ground.npz",
            "--d99-ground-manifest",
            "ground.json",
            "--base-d99-lock",
            "d99.json",
            "--phase1-lodo-json",
            "lodo.json",
            "--class-binding-json",
            "binding.json",
            "--class-binding-sha256",
            "c" * 64,
            "--output-root",
            "out",
            "--receiver",
            "rx",
            "--seed",
            "1337",
            "--k-shot",
            "10",
        ],
    )
    assert runner.main() == 0
    assert captured["class_binding_json"] == "binding.json"
    assert captured["class_binding_sha256"] == "c" * 64
    assert captured["new_count"] == 20
    assert captured["k_shot"] == 10
    assert json.loads(capsys.readouterr().out) == {"status": "ok"}
