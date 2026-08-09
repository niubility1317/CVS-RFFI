from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
from torch import nn

from cvsrffi import phase1_single_control_bundle_v1 as scb
from cvsrffi import phase3_cirf_track_v3 as cirf
from scripts import phase3_scb_real_n1_bridge as bridge


def _sha(char: str) -> str:
    return char * 64


class _RealFixtureRuntime(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("runtime_capacity_token", torch.tensor([1], dtype=torch.int64))

    def forward(self, rows: torch.Tensor):
        mean = rows.mean(dim=2)
        z_id = torch.stack((mean[:, 0] + 1.0, mean[:, 1] + 0.5), dim=1)
        logits = torch.stack((z_id[:, 0], z_id[:, 1], -z_id[:, 0], -z_id[:, 1]), dim=1)
        return z_id, logits


def _runtime_bytes() -> bytes:
    traced = torch.jit.trace(_RealFixtureRuntime().eval(), torch.zeros((1, 2, 256), dtype=torch.float32))
    stream = io.BytesIO()
    torch.jit.save(traced, stream)
    return stream.getvalue()


def _labeled_keys() -> list[tuple[str, str, str, int]]:
    return [(label, "0", "0", sig_i) for label in scb.LOCAL4_HANDLES for sig_i in range(32)]


def _unlabeled_keys() -> list[tuple[str, str, str, int]]:
    return [(label, "0", "0", 32 + sig_i) for label in scb.LOCAL4_HANDLES for sig_i in range(16)]


def _calibration_keys() -> list[tuple[str, str, str, int]]:
    return [(scb.LOCAL4_HANDLES[index % 4], "0", "1", index) for index in range(199)]


def _source_split() -> dict[str, object]:
    return {
        "schema": "cvs.phase1.source_split_receipt.v1",
        "seed": 7281105,
        "split_mode": "tx_rx_day_1_6_3",
        "source_days": ["0", "1"],
        "target_days": ["2", "3"],
        "source_receivers": ["0", "1", "2", "3", "4", "5", "6"],
        "target_receivers": ["10", "11", "7", "8", "9"],
        "source_target_receiver_overlap_count": 0,
        "labeled_indices_sha256": _sha("1"),
        "unlabeled_indices_sha256": _sha("2"),
        "source_validation_indices_sha256": _sha("3"),
        "split_manifest_sha256": _sha("4"),
        "labeled_size": 3920,
        "unlabeled_size": 35280,
        "source_validation_size": 16800,
        "source_pool_size": 56000,
        "requested_labeled_ratio": 0.07,
        "requested_unlabeled_ratio": 0.63,
        "requested_source_val_ratio": 0.30,
        "requested_rho_label": 0.10,
        "realized_rho_label": 0.1,
        "realized_rho_tolerance": 0.002,
        "realized_rho_within_tolerance": True,
        "realized_source_val_fraction": 0.3,
        "realized_source_val_tolerance": 0.002,
        "realized_source_val_within_tolerance": True,
    }


def _partition() -> dict[str, object]:
    return scb.build_source_partition_receipt(
        dataset_sha256=_sha("d"),
        source_split_projection=_source_split(),
        tx_partition_receipt={"enabled": True, "source_known_train_tx": list(scb.LOCAL4_HANDLES)},
        labeled_keys=_labeled_keys(),
        unlabeled_keys=_unlabeled_keys(),
        calibration_keys=_calibration_keys(),
        excluded_role_keys={
            "proxy": [("14-10", "0", "0", 0)],
            "held": [("14-7", "0", "0", 0)],
            "target": [("99-99", "0", "0", 0)],
        },
    )


def _state(*, calibration_set_sha256: str) -> scb.BundleState:
    return scb.BundleState(
        geometry=scb.ClassGeometry(
            class_handles=scb.LOCAL4_HANDLES,
            centers=np.asarray(((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)), dtype=np.float64),
            radii=np.ones(4, dtype=np.float64),
            class_counts=np.full(4, 32, dtype=np.int64),
        ),
        descriptor=scb.DescriptorStats(
            median=np.zeros(5, dtype=np.float64),
            scale=np.ones(5, dtype=np.float64),
            descriptor_count=4 * (128 + 64),
        ),
        tail=scb.TailSummary(
            levels=scb.TAIL_LEVELS.copy(),
            distance_values=np.linspace(0.0, 5.0, 129, dtype=np.float64),
            energy_values=np.linspace(-5.0, 5.0, 129, dtype=np.float64),
            domain_values=np.linspace(0.0, 8.0, 129, dtype=np.float64),
            n_calibration=199,
            calibration_set_sha256=calibration_set_sha256,
        ),
    )


def _resource() -> dict[str, object]:
    return {
        "schema": "cvs.phase1.single_control_resource_receipt.v1",
        "input_shape": [1, 2, 256],
        "input_dtype": "torch.float32",
        "input_sha256": scb.tensor_sha256(scb.resource_probe_model_input()),
        "input_seed": scb.RESOURCE_INPUT_SEED,
        "torch_num_threads": 1,
        "cpu_rss_baseline_bytes": 100,
        "cpu_rss_peak_bytes": 200,
        "cpu_rss_delta_bytes": 100,
        "cpu_warmups": 20,
        "cpu_trials": 100,
        "cpu_latency_p99_ms": 1.0,
        "cpu_latency_quantile_method": "higher_q99_100",
        "cuda_available": False,
        "cuda_peak_bytes": 0,
        "cuda_latency_recorded": False,
        "cuda_latency_p99_ms": 0.0,
        "measurement_scope": "fresh_process_bundle_load_warmup_full_local_evidence",
        "evidence_bytes": 100,
        "measurement_process": "fresh_python_subprocess_v1",
        "baseline_before_payload_load": True,
        "state_payload_reloaded": True,
        "runtime_state_before_sha256": _sha("9"),
        "runtime_state_after_sha256": _sha("9"),
    }


def _build_real_fixture(root: Path) -> str:
    runtime = _runtime_bytes()
    loaded = torch.jit.load(io.BytesIO(runtime), map_location="cpu").eval()
    partition = _partition()
    state = _state(calibration_set_sha256=str(partition["calibration_physical_set_sha256"]))
    checkpoint_sha = _sha("a")
    config_sha = _sha("b")
    result = scb.build_bundle(
        output_dir=root,
        bundle_status=scb.BUNDLE_STATUS,
        runtime_source=runtime,
        state=state,
        checkpoint_binding={
            "checkpoint_sha256": checkpoint_sha,
            "resolved_config_sha256": config_sha,
            "checkpoint_role": "training_final_only",
            "strict_state_tensor_schema_sha256": _sha("e"),
            "strict_load_audit": {"strict": True, "missing_keys": [], "unexpected_keys": []},
        },
        class_binding={
            "class_handles": list(scb.LOCAL4_HANDLES),
            "local_to_head_class_ids": [0, 1, 2, 3],
            "class_order_binding_sha256": _sha("f"),
            "checkpoint_head_class_count": 4,
            "live_head_class_count": 4,
            "checkpoint_train_tx_class_order": list(scb.LOCAL4_HANDLES),
        },
        source_partition_receipt=partition,
        runtime_parity_receipt=scb.make_runtime_parity_receipt(
            eager_runtime=loaded,
            runtime=loaded,
            state=state,
            checkpoint_sha256=checkpoint_sha,
            resolved_config_digest=config_sha,
            runtime_bytes=runtime,
        ),
        resource_receipt=_resource(),
        checkpoint_sha256=checkpoint_sha,
        resolved_config_digest=config_sha,
        dataset_sha256=_sha("d"),
        preprocessing_code_sha256=_sha("8"),
        scenario_registry_sha256=_sha("7"),
    )
    return str(result["content_root"])


def _context() -> dict[str, object]:
    return {
        "linkage_mode": "proxy_unverified",
        "proxy_group_id": "event-1",
        "satellite_reception_id": "rx-1",
        "node_id": "node-1",
        "base_manifest_id": "base-1",
        "correlation_group_id": "group-1",
        "delay_ms": 1.0,
        "deadline_ms": 10.0,
        "sealed_at_ms": 1.0,
    }


def _write_inputs(root: Path, *, iq: np.ndarray | None = None, context: dict[str, object] | None = None) -> tuple[Path, Path, str]:
    root.mkdir(parents=True, exist_ok=True)
    iq_path = root / "iq.npy"
    np.save(iq_path, np.asarray(iq if iq is not None else np.stack((np.sin(np.arange(256) / 5.0), np.cos(np.arange(256) / 7.0)), axis=1), dtype=np.float32))
    context_path = root / "context.json"
    context_path.write_text(json.dumps(context or _context(), ensure_ascii=False), encoding="utf-8")
    return iq_path, context_path, hashlib.sha256(iq_path.read_bytes()).hexdigest()


@pytest.fixture()
def fixture_paths(tmp_path: Path) -> dict[str, object]:
    bundle = tmp_path / "bundle"
    content_root = _build_real_fixture(bundle)
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    iq_path, context_path, iq_sha = _write_inputs(inputs)
    return {
        "root": tmp_path,
        "bundle": bundle,
        "content_root": content_root,
        "iq": iq_path,
        "context": context_path,
        "iq_sha": iq_sha,
    }


def _run(paths: dict[str, object], output: Path, **overrides: object) -> dict[str, object]:
    arguments = {
        "bundle_root": paths["bundle"],
        "expected_content_root": paths["content_root"],
        "iq_path": paths["iq"],
        "expected_iq_sha256": paths["iq_sha"],
        "context_json": paths["context"],
        "output_dir": output,
    }
    arguments.update(overrides)
    return bridge.run_bridge(**arguments)


def test_real_n1_bridge_success_care_cirf_and_manifest(fixture_paths: dict[str, object]) -> None:
    output = fixture_paths["root"] / "out"
    manifest = _run(fixture_paths, output)
    assert set(path.name for path in output.iterdir()) == set(bridge.OUTPUT_NAMES)
    local = json.loads((output / "local_evidence.json").read_text(encoding="utf-8"))
    care = json.loads((output / "care_n1_identity_receipt.json").read_text(encoding="utf-8"))
    cirf_receipt = json.loads((output / "cirf_n1_passthrough_receipt.json").read_text(encoding="utf-8"))
    emitted_manifest = json.loads((output / "bridge_manifest.json").read_text(encoding="utf-8"))
    assert local["schema_version"] == "cvs.phase3.local_evidence.v3"
    assert care["identity_receipt"]["decision"] == local["local_decision"]
    assert care["identity_receipt"]["label"] == local["local_label"]
    assert care["identity_receipt"]["reason_code"] == local["reason_code"]
    assert care["identity_receipt"]["p_fused"] == local["p_local"]
    local_bytes = (output / "local_evidence.json").read_bytes()
    assert cirf.n1_passthrough_bytes(local_bytes) == local_bytes
    assert cirf_receipt["byte_identical"] is True
    assert cirf_receipt["local_evidence_sha256"] == hashlib.sha256(local_bytes).hexdigest()
    assert manifest == emitted_manifest
    for field, value in {
        "technical_only": True,
        "performance_result": False,
        "truth_sidecar_opened": False,
        "same_event_claim": False,
        "collaborative_gain_claim": False,
        "n_sat": 1,
    }.items():
        assert emitted_manifest[field] is value


@pytest.mark.parametrize("kind", ["external_root", "iq_sha", "nonfinite", "shape"])
def test_real_n1_bridge_rejects_input_drift_without_output(fixture_paths: dict[str, object], kind: str) -> None:
    output = fixture_paths["root"] / f"reject-{kind}"
    kwargs: dict[str, object] = {}
    if kind == "external_root":
        kwargs["expected_content_root"] = "0" * 64
    elif kind == "iq_sha":
        kwargs["expected_iq_sha256"] = "0" * 64
    elif kind == "nonfinite":
        iq_path, _, _ = _write_inputs(fixture_paths["root"] / "nonfinite-input", iq=np.full((256, 2), np.nan, dtype=np.float32))
        kwargs["iq_path"] = iq_path
        kwargs["expected_iq_sha256"] = hashlib.sha256(iq_path.read_bytes()).hexdigest()
    elif kind == "shape":
        iq_path, _, _ = _write_inputs(fixture_paths["root"] / "shape-input", iq=np.zeros((3, 4), dtype=np.float32))
        kwargs["iq_path"] = iq_path
        kwargs["expected_iq_sha256"] = hashlib.sha256(iq_path.read_bytes()).hexdigest()
    with pytest.raises((bridge.BridgeContractError, FileExistsError)):
        _run(fixture_paths, output, **kwargs)
    assert not output.exists()


@pytest.mark.parametrize("forbidden", ["extra", "truth", "role"])
def test_real_n1_bridge_context_allowlist_rejects_extra_truth_role(fixture_paths: dict[str, object], forbidden: str) -> None:
    context = _context()
    context[forbidden] = "forbidden"
    context_path = fixture_paths["root"] / f"context-{forbidden}.json"
    context_path.write_text(json.dumps(context), encoding="utf-8")
    output = fixture_paths["root"] / f"reject-context-{forbidden}"
    with pytest.raises(bridge.BridgeContractError):
        _run(fixture_paths, output, context_json=context_path)
    assert not output.exists()


def test_real_n1_bridge_output_exists_is_refused_and_timeout_defers(fixture_paths: dict[str, object]) -> None:
    existing = fixture_paths["root"] / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        _run(fixture_paths, existing)

    late_context = _context()
    late_context["delay_ms"] = 11.0
    late_context_path = fixture_paths["root"] / "late-context.json"
    late_context_path.write_text(json.dumps(late_context), encoding="utf-8")
    late_output = fixture_paths["root"] / "late-output"
    # SCB marks delay>deadline as ``SCB_CONTEXT_DEFER``.  CARE's existing
    # fusion contract then has no on-time row for its N=1 identity branch, so
    # this bridge takes the permitted fail-closed path and emits no output.
    with pytest.raises(bridge.BridgeContractError, match="fail-closed"):
        _run(fixture_paths, late_output, context_json=late_context_path)
    assert not late_output.exists()


def test_real_n1_bridge_cli_smoke_and_output_allowlist(fixture_paths: dict[str, object]) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "phase3_scb_real_n1_bridge.py"
    output = fixture_paths["root"] / "cli-output"
    completed = subprocess.run(
        [
            sys.executable,
            "-X",
            "utf8",
            str(script),
            "--bundle-root",
            str(fixture_paths["bundle"]),
            "--expected-content-root",
            str(fixture_paths["content_root"]),
            "--iq",
            str(fixture_paths["iq"]),
            "--expected-iq-sha256",
            str(fixture_paths["iq_sha"]),
            "--context-json",
            str(fixture_paths["context"]),
            "--output-dir",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert json.loads(completed.stdout)["n_sat"] == 1
    assert set(path.name for path in output.iterdir()) == set(bridge.OUTPUT_NAMES)
