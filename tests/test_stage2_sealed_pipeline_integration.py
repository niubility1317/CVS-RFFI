from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = ROOT / "code" / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import build_cvs_stage2_predictor_bundle as bundle_builder  # noqa: E402
import build_cvs_stage2_predictor_request as request_builder  # noqa: E402
import run_cvs_stage2_predictor as predictor_cli  # noqa: E402
from cvsrffi.phase2_runtime_contract import PRE_RUN_RUNTIME_EVIDENCE_REQUIRED_FIELDS  # noqa: E402
from cvsrffi.phase2_runtime_closure import build_phase2_runtime_closure  # noqa: E402
from cvsrffi.phase2_pre_run_evidence import build_phase2_pre_run_evidence  # noqa: E402
from cvsrffi.phase2_isolated_runner import execute_phase2_isolated  # noqa: E402
from cvsrffi.stage2_metric_scorer import score_sealed_prediction  # noqa: E402
from cvsrffi.stage2_prediction_artifact import verify_prediction_artifact  # noqa: E402
from cvsrffi.stage2_predictor_bundle import sha256_file  # noqa: E402


class TinyRuntime(torch.nn.Module):
    def forward(self, rows: torch.Tensor):
        features = rows.mean(dim=2)
        logits = torch.stack((features[:, 0], features[:, 1]), dim=1)
        return {"features": features, "logits": logits}


def _cache_arrays():
    labels = ["old-a", "old-b", "new-a"]
    roles = ["target_old", "target_old", "target_new"]
    rows = []
    for class_index, (label, role) in enumerate(zip(labels, roles)):
        for sample_index in range(2):
            rows.append((label, role, class_index, sample_index))
    result = {}
    for scenario_index, scenario in enumerate(bundle_builder.FORMAL_LEO_WEAK_SCENARIOS):
        iq = np.zeros((len(rows), 2, 8), dtype=np.float32)
        for index, (_label, _role, class_index, _sample_index) in enumerate(rows):
            iq[index, class_index % 2] = float(class_index + 1 + scenario_index * 0.01)
        result[scenario] = {
            "leo_weak_iq": iq,
            "tx_ids": np.asarray([row[0] for row in rows]),
            "dataset_role": np.asarray([row[1] for row in rows]),
            "rx_ids": np.asarray(["20-1"] * len(rows)),
            "day_ids": np.asarray(["1"] * len(rows)),
            "sig_ids": np.asarray([str(row[3]) for row in rows]),
            "sample_ids": np.asarray([
                f"{row[1]}|{row[0]}|20-1|1|{row[3]}" for row in rows
            ]),
            "overlay_ids": np.asarray([
                f"overlay|{scenario}|{index}" for index in range(len(rows))
            ]),
            "satellite_seeds": np.asarray([100 + scenario_index] * len(rows), dtype=np.int64),
        }
    return result


def test_end_to_end_truth_free_predictor_produces_sealed_artifact(
    tmp_path: Path, monkeypatch
) -> None:
    arrays = _cache_arrays()
    monkeypatch.setattr(
        bundle_builder,
        "load_verified_leo_weak_cache_set",
        lambda *_args, **_kwargs: (
            arrays,
            {"schema": "test-cache", "cache_scope": "stage2_registered"},
            {"status": "PASS", "offline_source": "not_predictor_reachable"},
        ),
    )
    source = tmp_path / "source"
    source.mkdir()
    checkpoint = source / "runtime.pt"
    traced = torch.jit.trace(
        TinyRuntime(), torch.zeros((1, 2, 8), dtype=torch.float32), strict=False
    )
    torch.jit.save(traced, checkpoint)
    adapter = source / "adapter.json"
    adapter.write_text(json.dumps({
        "schema": "cvs.phase2.feature_adapter.v1", "mode": "identity",
        "trainable_parameters": 0, "adapt_epochs": 0,
        "persistent_state_bytes": 0, "fft_dim": 0, "fft_weight": 1.0,
    }), encoding="utf-8")
    head = source / "head.json"
    head.write_text(json.dumps({
        "schema": "cvs.phase2.prototype_head.v1", "metric": "cosine",
        "temperature": 10.0,
    }), encoding="utf-8")
    tta = source / "tta.json"
    tta.write_text(json.dumps({
        "schema": "cvs.phase2.adaptive_rxlight_tta.v1", "mode": "base_only",
        "base_views": 1, "max_views": 1,
    }), encoding="utf-8")
    candidate = source / "candidate.lock"
    candidate.write_text("locked", encoding="ascii")
    cache = source / "cache.json"
    cache.write_text("{}", encoding="ascii")
    predictor_root = tmp_path / "predictor"
    scorer_root = tmp_path / "scorer"
    built = bundle_builder.build(
        argparse.Namespace(
            target_cache_set=cache, expected_cache_scope="stage2_registered",
            predictor_out_root=predictor_root, scorer_out_root=scorer_root,
            detached_seal_path=None, stage="stage2c", receiver="20-1",
            seed=713101, old_class_labels="old-a,old-b", new_class_labels="new-a",
            stage2b_reference_new_class_labels="", new_class_count=1,
            support_pool_max_k=1, query_per_tx=1, candidate_lock=candidate,
            checkpoint=checkpoint, adapter=adapter, head_artifact=head,
            tta_policy_json=tta,
        ),
        token_secret=b"z" * 32,
    )
    seal_path = Path(built["predictor_package_seal"])
    runtime_closure_root = tmp_path / "runtime_closure"
    runtime_closure = build_phase2_runtime_closure(
        ROOT / "code", runtime_closure_root
    )
    system_root = tmp_path / "system"
    system_root.mkdir()
    executables = []
    for name in ("bwrap", "strace", "python"):
        executable = system_root / name
        executable.write_bytes(name.encode("ascii"))
        executable.chmod(0o555)
        executables.append(executable)
    monkeypatch.setattr("cvsrffi.phase2_pre_run_evidence.platform.system", lambda: "Linux")
    monkeypatch.setattr(
        "cvsrffi.phase2_pre_run_evidence._trusted_system_root_allowlist",
        lambda: [system_root.resolve()],
    )
    evidence_result = build_phase2_pre_run_evidence(
        runtime_closure_root=runtime_closure_root,
        package_root=predictor_root,
        detached_seal=seal_path,
        expected_package_seal_sha256=sha256_file(seal_path),
        output_root=tmp_path / "pre_run_evidence",
        bwrap_executable=executables[0],
        strace_executable=executables[1],
        python_executable=executables[2],
        system_read_roots=[system_root],
        forbidden_scorer_truth_roots=[scorer_root],
    )
    evidence_path = Path(evidence_result["runtime_isolation_evidence"])
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert set(evidence) == set(PRE_RUN_RUNTIME_EVIDENCE_REQUIRED_FIELDS)
    assert evidence_result["post_run_filesystem_access_audit_pending"] is True
    request_path = tmp_path / "request.json"
    request_builder.build_request(argparse.Namespace(
        predictor_package_root=predictor_root, detached_seal_path=seal_path,
        expected_seal_sha256=sha256_file(seal_path),
        runtime_evidence_json=evidence_path,
        k_shot=1,
        request_id="cell-clear", row_id="cell", output_relative_path="prediction.cvspred",
        output_json=request_path,
    ))
    request = json.loads(request_path.read_text(encoding="utf-8"))
    assert request["runtime_code_sha256"] == runtime_closure["root_sha256"]
    assert (
        request["phase2_runtime_isolation_evidence"]["runtime_code_sha256"]
        == runtime_closure["root_sha256"]
    )
    output_root = tmp_path / "output"
    output_root.mkdir()
    result = predictor_cli.run(argparse.Namespace(
        request_json=request_path, predictor_package_root=predictor_root,
        detached_seal_path=seal_path, expected_seal_sha256=sha256_file(seal_path),
        output_root=output_root, device="cpu", batch_size=8,
    ))
    verified = verify_prediction_artifact(
        output_root / "prediction.cvspred",
        expected_artifact_sha256=result["artifact_sha256"],
        expected_seal_sha256=result["seal_sha256"],
    )
    assert verified["manifest"]["stage"] == "Stage2-C"
    assert verified["manifest"]["row_count"] == 9
    assert all(
        value.startswith("cls_")
        for value in verified["arrays"]["candidate_after"].tolist()
    )
    assert not (predictor_root / "truth_sidecar.json").exists()

    formal_rows, formal_predictions, scoring_receipt = score_sealed_prediction(
        output_root / "prediction.cvspred",
        built["scoring_manifest"],
        expected_prediction_artifact_sha256=result["artifact_sha256"],
        expected_prediction_seal_sha256=result["seal_sha256"],
        expected_scoring_manifest_sha256=built["scoring_manifest_sha256"],
    )
    assert len(formal_rows["rows"]) == 3
    assert {row["scenario"] for row in formal_rows["rows"]} == set(
        bundle_builder.FORMAL_LEO_WEAK_SCENARIOS
    )
    assert len(formal_predictions["predictions"]) == 9
    assert scoring_receipt["formal_row_count"] == 3
    assert scoring_receipt["formal_prediction_count"] == 9
    assert scoring_receipt["truth_join_after_prediction_only"] is True
    assert scoring_receipt["scorer_output_must_not_feed_predictor"] is True

    isolated_output = tmp_path / "isolated_output"
    isolated_output.mkdir()

    def fake_bwrap_run(command, **kwargs):
        assert "pass_fds" not in kwargs
        trace_target = Path(command[command.index("-o") + 1])
        assert trace_target.parent == isolated_output.parent
        assert isolated_output not in trace_target.parents
        trace_target.write_text(
            "\n".join(
                [
                    f'301 execve({json.dumps(str(executables[2].resolve()))}, ["python"], 0x0) = 0',
                    '301 openat(AT_FDCWD, "/sealed/request.json", O_RDONLY) = 3</sealed/request.json>',
                    '301 openat(AT_FDCWD, "/sealed/package.seal.json", O_RDONLY) = 4</sealed/package.seal.json>',
                    '301 openat(AT_FDCWD, "/sealed/package/manifest.json", O_RDONLY) = 5</sealed/package/manifest.json>',
                    '301 openat(AT_FDCWD, "/runtime/code/scripts/run_cvs_stage2_predictor.py", O_RDONLY) = 6</runtime/code/scripts/run_cvs_stage2_predictor.py>',
                    '301 openat(AT_FDCWD, "/output/prediction.cvspred", O_WRONLY|O_CREAT|O_EXCL) = 7</output/prediction.cvspred>',
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        inner = predictor_cli.run(argparse.Namespace(
            request_json=request_path, predictor_package_root=predictor_root,
            detached_seal_path=seal_path, expected_seal_sha256=sha256_file(seal_path),
            output_root=isolated_output, device="cpu", batch_size=8,
        ))
        return SimpleNamespace(returncode=0, stdout=json.dumps(inner), stderr="")

    isolated_result = execute_phase2_isolated(
        bwrap=executables[0],
        strace_executable=executables[1],
        runtime_closure_root=runtime_closure_root,
        pre_run_evidence_root=evidence_path.parent,
        package_root=predictor_root,
        detached_seal=seal_path,
        expected_package_seal_sha256=sha256_file(seal_path),
        request_json=request_path,
        output_root=isolated_output,
        python_executable=executables[2],
        system_read_roots=[system_root],
        forbidden_roots=[scorer_root],
        forbidden_project_roots=[str(tmp_path / "unmounted_project")],
        device="cpu",
        batch_size=8,
        command_runner=fake_bwrap_run,
    )
    assert isolated_result["status"] == "LOCAL_DIAGNOSTIC_PASS"
    assert isolated_result["formal_launch_authority"] is False
    isolated_verified = verify_prediction_artifact(
        isolated_output / "prediction.cvspred",
        expected_artifact_sha256=isolated_result["prediction_artifact_sha256"],
        expected_seal_sha256=isolated_result["prediction_seal_sha256"],
    )
    assert isolated_verified["manifest"]["row_count"] == 9
    diagnostic_post_run_evidence = json.loads(
        Path(isolated_result["diagnostic_post_run_runtime_evidence"]).read_text(encoding="utf-8")
    )
    assert diagnostic_post_run_evidence["formal_launch_authority"] is False
    assert diagnostic_post_run_evidence["protocol_valid_claim_allowed"] is False
    assert (
        diagnostic_post_run_evidence["formal_post_run_contract_evidence"]
        ["filesystem_access_audit_status"]
        == "PASS"
    )
