from __future__ import annotations

import io
import inspect
import json
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
import pytest
import torch
from torch import nn

from cvsrffi import phase1_single_control_bundle_v1 as scb


def _sha(char: str) -> str:
    return char * 64


class _FixtureRuntime(nn.Module):
    def forward(self, rows: torch.Tensor):
        mean = rows.mean(dim=2)
        z_id = torch.stack((mean[:, 0] + 1.0, mean[:, 1] + 0.5), dim=1)
        logits = torch.stack((z_id[:, 0], z_id[:, 1], -z_id[:, 0], -z_id[:, 1]), dim=1)
        return z_id, logits


class _MutatingBufferRuntime(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("counter", torch.zeros((), dtype=torch.float32))

    def forward(self, rows: torch.Tensor):
        self.counter.add_(1.0)
        mean = rows.mean(dim=2)
        z_id = torch.stack((mean[:, 0] + 1.0, mean[:, 1] + 0.5), dim=1)
        logits = torch.stack((z_id[:, 0], z_id[:, 1], -z_id[:, 0], -z_id[:, 1]), dim=1)
        return z_id, logits


class _DeviceBoundRuntime(nn.Module):
    """A real-parameter module that rejects a CPU input after CUDA placement."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    def forward(self, rows: torch.Tensor):
        if rows.device != self.scale.device:
            raise RuntimeError("runtime input did not follow parameter device")
        mean = rows.mean(dim=2) * self.scale
        z_id = torch.stack((mean[:, 0] + 1.0, mean[:, 1] + 0.5), dim=1)
        logits = torch.stack((z_id[:, 0], z_id[:, 1], -z_id[:, 0], -z_id[:, 1]), dim=1)
        return z_id, logits


def _runtime_bytes() -> bytes:
    traced = torch.jit.trace(_FixtureRuntime().eval(), torch.zeros((1, 2, 256), dtype=torch.float32))
    stream = io.BytesIO()
    torch.jit.save(traced, stream)
    return stream.getvalue()


def _labeled_keys() -> list[tuple[str, str, str, int]]:
    return [(label, "0", "0", sig_i) for label in scb.LOCAL4_HANDLES for sig_i in range(32)]


def _unlabeled_keys() -> list[tuple[str, str, str, int]]:
    return [(label, "0", "0", 32 + sig_i) for label in scb.LOCAL4_HANDLES for sig_i in range(16)]


def _calibration_keys() -> list[tuple[str, str, str, int]]:
    return [(scb.LOCAL4_HANDLES[index % 4], "0", "1", index) for index in range(199)]


def _state(*, calibration_set_sha256: str) -> scb.BundleState:
    return scb.BundleState(
        geometry=scb.ClassGeometry(
            class_handles=scb.LOCAL4_HANDLES,
            centers=np.asarray(((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)), dtype=np.float64),
            radii=np.ones(4, dtype=np.float64), class_counts=np.full(4, 32, dtype=np.int64),
        ),
        descriptor=scb.DescriptorStats(
            median=np.zeros(5, dtype=np.float64), scale=np.ones(5, dtype=np.float64), descriptor_count=4 * (128 + 64)
        ),
        tail=scb.TailSummary(
            levels=scb.TAIL_LEVELS.copy(),
            distance_values=np.linspace(0.0, 5.0, 129, dtype=np.float64),
            energy_values=np.linspace(-5.0, 5.0, 129, dtype=np.float64),
            domain_values=np.linspace(0.0, 8.0, 129, dtype=np.float64),
            n_calibration=199, calibration_set_sha256=calibration_set_sha256,
        ),
    )


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
        labeled_keys=_labeled_keys(), unlabeled_keys=_unlabeled_keys(), calibration_keys=_calibration_keys(),
        excluded_role_keys={
            "proxy": [("14-10", "0", "0", 0)],
            "held": [("14-7", "0", "0", 0)],
            "target": [("99-99", "0", "0", 0)],
        },
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


def _materials() -> dict[str, object]:
    runtime = _runtime_bytes()
    loaded = torch.jit.load(io.BytesIO(runtime), map_location="cpu").eval()
    checkpoint_sha = _sha("a")
    config_sha = _sha("b")
    partition = _partition()
    state = _state(calibration_set_sha256=str(partition["calibration_physical_set_sha256"]))
    return {
        "runtime_source": runtime,
        "state": state,
        "checkpoint_binding": {
            "checkpoint_sha256": checkpoint_sha,
            "resolved_config_sha256": config_sha,
            "checkpoint_role": "training_final_only",
            "strict_state_tensor_schema_sha256": _sha("e"),
            "strict_load_audit": {"strict": True, "missing_keys": [], "unexpected_keys": []},
        },
        "class_binding": {
            "class_handles": list(scb.LOCAL4_HANDLES),
            "local_to_head_class_ids": [0, 1, 2, 3],
            "class_order_binding_sha256": _sha("f"),
            "checkpoint_head_class_count": 4,
            "live_head_class_count": 4,
            "checkpoint_train_tx_class_order": list(scb.LOCAL4_HANDLES),
        },
        "source_partition_receipt": partition,
        "runtime_parity_receipt": scb.make_runtime_parity_receipt(
            eager_runtime=loaded,
            runtime=loaded,
            state=state,
            checkpoint_sha256=checkpoint_sha,
            resolved_config_digest=config_sha,
            runtime_bytes=runtime,
        ),
        "resource_receipt": _resource(),
        "checkpoint_sha256": checkpoint_sha,
        "resolved_config_digest": config_sha,
        "dataset_sha256": _sha("d"),
        "preprocessing_code_sha256": _sha("8"),
        "scenario_registry_sha256": _sha("7"),
        "eager_runtime": loaded,
    }


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


def _minimal_real_parser_checkpoint(*, extra_args: dict[str, object] | None = None) -> dict[str, object]:
    """Minimal checkpoint shape that exercises the actual SSDG parser/merge path."""

    args: dict[str, object] = {"num_classes": 4}
    if extra_args is not None:
        args.update(extra_args)
    return {"args": args, "model": {"dom_head.net.3.bias": torch.zeros(2)}}


def test_fixture_bundle_has_exact_ten_members_and_full_path_care_n1_parity(tmp_path: Path) -> None:
    material = _materials()
    result = scb.build_bundle(output_dir=tmp_path / "bundle", bundle_status=scb.FIXTURE_STATUS, **{key: value for key, value in material.items() if key != "eager_runtime"})
    assert result["member_count"] == 10
    bundle = scb.load_bundle(
        tmp_path / "bundle", expected_content_root=result["content_root"], expected_bundle_status=scb.FIXTURE_STATUS
    )
    raw = torch.stack((torch.sin(torch.arange(256, dtype=torch.float32) / 5.0), torch.cos(torch.arange(256, dtype=torch.float32) / 7.0)), dim=1)
    parity = scb.assert_full_path_parity(
        eager_runtime=material["eager_runtime"], loaded_bundle=bundle, raw_iq=raw, context=_context()
    )
    assert parity["state_unchanged"] is True
    evidence = scb.local_evidence_from_bundle(bundle, raw_iq=raw, context=_context())
    n1 = scb.care_n1_parity(evidence)
    assert n1["decision"] == evidence["local_decision"]
    assert set(path.relative_to(tmp_path / "bundle").as_posix() for path in (tmp_path / "bundle").rglob("*") if path.is_file()) == set(scb.ALL_BUNDLE_MEMBERS)


def test_bundle_rejects_overwrite_runtime_swap_and_semantic_tamper(tmp_path: Path) -> None:
    material = _materials()
    target = tmp_path / "bundle"
    result = scb.build_bundle(output_dir=target, bundle_status=scb.FIXTURE_STATUS, **{key: value for key, value in material.items() if key != "eager_runtime"})
    with pytest.raises(FileExistsError):
        scb.build_bundle(output_dir=target, bundle_status=scb.FIXTURE_STATUS, **{key: value for key, value in material.items() if key != "eager_runtime"})
    swapped = dict(material)
    swapped["runtime_source"] = _runtime_bytes() + b"swap"
    with pytest.raises(scb.SingleControlBundleError, match="runtime parity"):
        scb.build_bundle(output_dir=tmp_path / "swap", bundle_status=scb.FIXTURE_STATUS, **{key: value for key, value in swapped.items() if key != "eager_runtime"})
    manifest_path = target / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["q_semantics"] = "forged"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    with pytest.raises(scb.SingleControlBundleError):
        scb.load_bundle(target, expected_content_root=result["content_root"], expected_bundle_status=scb.FIXTURE_STATUS)


def test_canonicalizer_and_partition_reject_unsafe_or_overlapping_inputs() -> None:
    for value in (("tuple",), np.int64(2), np.float64(2.0)):
        with pytest.raises(scb.SingleControlBundleError):
            scb.canonical_json_bytes(value)
    with pytest.raises(scb.SingleControlBundleError, match="overlap"):
        scb.build_source_partition_receipt(
            dataset_sha256=_sha("d"),
            source_split_projection=_source_split(),
            tx_partition_receipt={"enabled": True},
            labeled_keys=[("20-15", "0", "0", 0), ("20-19", "0", "0", 0), ("6-15", "0", "0", 0), ("8-20", "0", "0", 0)],
            unlabeled_keys=[("20-15", "0", "0", 0)],
            calibration_keys=[("20-15", "0", "1", 0), ("20-19", "0", "1", 0), ("6-15", "0", "1", 0), ("8-20", "0", "1", 0)],
            excluded_role_keys={"proxy": [], "held": [], "target": []},
        )
    with pytest.raises(scb.SingleControlBundleError, match="duplicate"):
        scb.build_source_partition_receipt(
            dataset_sha256=_sha("d"),
            source_split_projection=_source_split(),
            tx_partition_receipt={"enabled": True},
            labeled_keys=[
                ("20-15", "0", "0", 0), ("20-15", "0", "0", 0), ("20-19", "0", "0", 0),
                ("6-15", "0", "0", 0), ("8-20", "0", "0", 0),
            ],
            unlabeled_keys=[("20-15", "0", "0", 1)],
            calibration_keys=[("20-15", "0", "1", 0), ("20-19", "0", "1", 0), ("6-15", "0", "1", 0), ("8-20", "0", "1", 0)],
            excluded_role_keys={"proxy": [], "held": [], "target": []},
        )


def test_resource_receipt_and_context_fail_closed() -> None:
    invalid = _resource()
    invalid["cpu_trials"] = 99
    with pytest.raises(scb.SingleControlBundleError):
        scb.validate_resource_receipt(invalid)
    with pytest.raises(scb.SingleControlBundleError):
        scb._safe_context({"role": "oracle"})


def test_real_build_rejects_existing_output_before_expensive_inputs(tmp_path: Path) -> None:
    target = tmp_path / "already-exists"
    target.mkdir()
    with pytest.raises(FileExistsError, match="overwrite"):
        scb.build_real_bundle_from_paths(
            project_root=tmp_path,
            checkpoint_path=tmp_path / "missing-checkpoint.pth",
            wisig_pkl_path=tmp_path / "missing-manysig.pkl",
            completion_receipt_path=tmp_path / "missing-completion.json",
            terminal_receipt_path=tmp_path / "missing-terminal.json",
            cp_terminal_receipt_path=tmp_path / "missing-cp.json",
            output_dir=target,
        )


def test_opaque_descriptor_schema_and_state_partition_closure_fail_closed(tmp_path: Path) -> None:
    material = _materials()
    opaque = scb.opaque_source_sample_hash(7)
    assert len(opaque) == 64 and "20-15" not in opaque
    with pytest.raises(scb.SingleControlBundleError, match="opaque_hash"):
        scb.fit_descriptor_stats([{"opaque_token": "20-15", "iq_views": []}])
    bad_geometry = scb.ClassGeometry(
        class_handles=scb.LOCAL4_HANDLES,
        centers=material["state"].geometry.centers,
        radii=material["state"].geometry.radii,
        class_counts=np.full(4, 33, dtype=np.int64),
    )
    bad_state = scb.BundleState(geometry=bad_geometry, descriptor=material["state"].descriptor, tail=material["state"].tail)
    bad = dict(material)
    bad["state"] = bad_state
    with pytest.raises(scb.SingleControlBundleError, match="labeled physical partition"):
        scb.build_bundle(output_dir=tmp_path / "bad-state", bundle_status=scb.FIXTURE_STATUS, **{k: v for k, v in bad.items() if k != "eager_runtime"})
    bad_tail = scb.TailSummary(
        material["state"].tail.levels, material["state"].tail.distance_values, material["state"].tail.energy_values,
        material["state"].tail.domain_values, material["state"].tail.n_calibration, _sha("d"),
    )
    bad["state"] = scb.BundleState(material["state"].geometry, material["state"].descriptor, bad_tail)
    with pytest.raises(scb.SingleControlBundleError, match="tail calibration set SHA"):
        scb.build_bundle(output_dir=tmp_path / "bad-tail", bundle_status=scb.FIXTURE_STATUS, **{k: v for k, v in bad.items() if k != "eager_runtime"})


def test_lock_allowlists_runtime_value_mutation_and_default_fixture_rejection(tmp_path: Path) -> None:
    material = _materials()
    invalid = dict(material)
    checkpoint = dict(material["checkpoint_binding"])
    checkpoint["truth_sidecar"] = "forbidden"
    invalid["checkpoint_binding"] = checkpoint
    with pytest.raises(scb.SingleControlBundleError, match="checkpoint binding key allowlist"):
        scb.build_bundle(output_dir=tmp_path / "bad-checkpoint", bundle_status=scb.FIXTURE_STATUS, **{k: v for k, v in invalid.items() if k != "eager_runtime"})
    invalid = dict(material)
    source = dict(material["source_partition_receipt"])
    source["sample_cache"] = "forbidden"
    invalid["source_partition_receipt"] = source
    with pytest.raises(scb.SingleControlBundleError, match="source partition receipt key allowlist"):
        scb.build_bundle(output_dir=tmp_path / "bad-source", bundle_status=scb.FIXTURE_STATUS, **{k: v for k, v in invalid.items() if k != "eager_runtime"})
    mutating = _MutatingBufferRuntime().eval()
    with pytest.raises(scb.SingleControlBundleError, match="state mutation"):
        scb.make_runtime_parity_receipt(
            eager_runtime=mutating, runtime=mutating, state=material["state"], checkpoint_sha256=_sha("a"),
            resolved_config_digest=_sha("b"), runtime_bytes=material["runtime_source"],
        )
    result = scb.build_bundle(output_dir=tmp_path / "bundle", bundle_status=scb.FIXTURE_STATUS, **{k: v for k, v in material.items() if k != "eager_runtime"})
    with pytest.raises(scb.SingleControlBundleError, match="status"):
        scb.load_bundle(tmp_path / "bundle", expected_content_root=result["content_root"])


def test_evidence_byte_gate_and_fresh_resource_state_reload(tmp_path: Path) -> None:
    material = _materials()
    result = scb.build_bundle(output_dir=tmp_path / "bundle", bundle_status=scb.FIXTURE_STATUS, **{k: v for k, v in material.items() if k != "eager_runtime"})
    bundle = scb.load_bundle(tmp_path / "bundle", expected_content_root=result["content_root"], expected_bundle_status=scb.FIXTURE_STATUS)
    raw = torch.stack((torch.sin(torch.arange(256, dtype=torch.float32)), torch.cos(torch.arange(256, dtype=torch.float32))), dim=1)
    oversized = dict(_context())
    oversized["proxy_group_id"] = "x" * scb.MAX_EVIDENCE_BYTES
    with pytest.raises(scb.SingleControlBundleError, match="byte gate"):
        scb.local_evidence_from_bundle(bundle, raw_iq=raw, context=oversized)
    incomplete = dict(_context())
    incomplete.pop("node_id")
    with pytest.raises(scb.SingleControlBundleError):
        scb.local_evidence_from_bundle(bundle, raw_iq=raw, context=incomplete)
    late = dict(_context())
    late["delay_ms"] = 11.0
    timed = scb.local_evidence_from_bundle(bundle, raw_iq=raw, context=late)
    assert timed["local_decision"] == "defer" and timed["local_label"] is None
    assert timed["reason_code"] == "SCB_CONTEXT_DEFER"
    receipt = scb.make_resource_receipt(runtime_bytes=material["runtime_source"], state=material["state"], device=torch.device("cuda"))
    assert receipt["measurement_process"] == "fresh_python_subprocess_v1"
    assert receipt["baseline_before_payload_load"] is True and receipt["state_payload_reloaded"] is True
    assert receipt["runtime_state_before_sha256"] == receipt["runtime_state_after_sha256"]
    assert receipt["cpu_trials"] == 100 and receipt["cpu_warmups"] == 20
    assert receipt["cuda_available"] is bool(torch.cuda.is_available())


def test_rss_bytes_has_no_psutil_runtime_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "psutil", None)
    assert scb._rss_bytes() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a local CUDA runtime")
def test_local_runtime_moves_cpu_input_to_actual_parameter_device() -> None:
    material = _materials()
    runtime = _DeviceBoundRuntime().cuda().eval()
    probe = scb.resource_probe_model_input()
    with pytest.raises(RuntimeError, match="parameter device"):
        runtime(probe)
    fields = scb._local_fields_from_runtime(runtime, material["state"], probe)
    assert fields.z_id.shape == (2,) and fields.p_local.shape == (5,)
    assert np.isfinite(fields.z_id).all() and np.isfinite(fields.p_local).all()


def test_label_blind_source_view_and_descriptor_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    import cvsrffi.eval as cv_eval
    import training_controls

    signature = inspect.signature(scb._scb_views_for_source_sample)
    assert set(signature.parameters) == {"raw_iq", "opaque_sample_index", "args"}
    seed_source = inspect.getsource(scb.derive_source_view_seed)
    assert "tx_label" not in seed_source and "class_id" not in seed_source and "physical_token" not in seed_source

    monkeypatch.setattr(
        training_controls,
        "sat_channel_config_for_scenario",
        lambda scenario: {"channel_model": "leo_residual"},
    )

    def fake_satellite(clean, scenario, args, *, gen, return_meta):
        assert gen.device.type == "cpu" and return_meta is False
        return clean + 0.001 * torch.randn(clean.shape, generator=gen, dtype=clean.dtype), None

    monkeypatch.setattr(cv_eval, "apply_sat_channel_for_scenario", fake_satellite)
    raw = torch.stack(
        (torch.sin(torch.arange(256, dtype=torch.float32) / 5.0), torch.cos(torch.arange(256, dtype=torch.float32) / 7.0)),
        dim=1,
    )
    args = types.SimpleNamespace(sat_fs_hz=25000000.0, sat_fc_hz=2462000000.0)
    # No TX/class argument exists: a label permutation or erasure cannot alter
    # either the four frozen view bytes or the descriptor input identity.
    first = scb._scb_views_for_source_sample(raw_iq=raw, opaque_sample_index=17, args=args)
    second = scb._scb_views_for_source_sample(raw_iq=raw, opaque_sample_index=17, args=args)
    different = scb._scb_views_for_source_sample(raw_iq=raw, opaque_sample_index=18, args=args)
    assert all(torch.equal(left, right) for left, right in zip(first, second))
    assert any(not torch.equal(left, right) for left, right in zip(first[1:], different[1:]))
    opaque = scb.opaque_source_sample_hash(17)
    assert all(handle not in opaque for handle in scb.LOCAL4_HANDLES)
    with pytest.raises(scb.SingleControlBundleError, match="descriptor rows"):
        scb.fit_descriptor_stats([{"opaque_hash": opaque, "iq_views": [first[0][0]], "tx_label": "20-15"}])

    def descriptor_rows(fake_labels: dict[int, str]):
        # ``fake_labels`` simulates a TX relabeling but never crosses the
        # descriptor interface; only label-blind opaque indices reach it.
        del fake_labels
        rows = []
        generator = torch.Generator(device="cpu").manual_seed(19)
        for index in range(32):
            view = torch.randn((2, 256), generator=generator, dtype=torch.float32).contiguous()
            view = view / torch.sqrt(torch.mean(view.square()) + 1.0e-12)
            rows.append({"opaque_hash": scb.opaque_source_sample_hash(index), "iq_views": [view]})
        return rows

    stats_a = scb.fit_descriptor_stats(descriptor_rows({index: scb.LOCAL4_HANDLES[index % 4] for index in range(32)}))
    stats_b = scb.fit_descriptor_stats(descriptor_rows({index: scb.LOCAL4_HANDLES[(index + 1) % 4] for index in range(32)}))
    assert np.array_equal(stats_a.median, stats_b.median)
    assert np.array_equal(stats_a.scale, stats_b.scale)
    assert stats_a.descriptor_count == stats_b.descriptor_count == 32


def test_streaming_descriptor_and_opaque_index_partition_boundaries() -> None:
    with pytest.raises(scb.SingleControlBundleError, match="duplicate"):
        scb._validate_source_view_indices([0, 0], [1], [2])
    with pytest.raises(scb.SingleControlBundleError, match="pairwise"):
        scb._validate_source_view_indices([0], [1], [1])

    def rows_and_reference():
        generator = torch.Generator(device="cpu").manual_seed(91)
        streamed = []
        reference = []
        for index in range(300):
            view = torch.randn((2, 256), generator=generator, dtype=torch.float32).contiguous()
            view = view / torch.sqrt(torch.mean(view.square()) + 1.0e-12)
            reference.append(scb.domain_descriptor(view))
            streamed.append({"opaque_hash": scb.opaque_source_sample_hash(index), "iq_views": [view]})
        return iter(streamed), np.asarray(reference, dtype=np.float64)

    stream, reference = rows_and_reference()
    stats = scb.fit_descriptor_stats(stream)
    expected_median = np.median(reference, axis=0)
    expected_scale = 1.4826 * np.median(np.abs(reference - expected_median[None, :]), axis=0)
    assert stats.descriptor_count == 300
    assert np.array_equal(stats.median, expected_median)
    assert np.array_equal(stats.scale, expected_scale)
    implementation = inspect.getsource(scb.fit_descriptor_stats)
    assert "np.vstack" not in implementation and "array(\"d\")" in implementation


def test_preprocess_seed_tail_and_config_contracts() -> None:
    raw = torch.stack((torch.arange(300, dtype=torch.float32), -torch.arange(300, dtype=torch.float32)), dim=1)
    actual = scb.preprocess_iq(raw)
    expected = raw[22:278].transpose(0, 1)
    expected = expected / torch.sqrt(torch.mean(expected[0].square() + expected[1].square()) + 1.0e-12)
    assert torch.equal(actual, expected.contiguous())
    padded = scb.preprocess_iq(torch.ones((255, 2), dtype=torch.float32))
    assert padded.shape == (2, 256) and float(padded[:, -1].abs().sum()) == 0.0
    seed = scb.derive_source_view_seed(split_seed=7281105, opaque_sample_index=7, scenario="leo_clear_weak")
    assert seed == scb.derive_source_view_seed(split_seed=7281105, opaque_sample_index=7, scenario="leo_clear_weak")
    assert seed != scb.derive_source_view_seed(split_seed=7281105, opaque_sample_index=7, scenario="leo_rain_weak")
    assert seed != scb.derive_source_view_seed(split_seed=7281105, opaque_sample_index=8, scenario="leo_clear_weak")
    summary = scb.TailSummary(
        levels=scb.TAIL_LEVELS.copy(), distance_values=np.zeros(129), energy_values=np.zeros(129),
        domain_values=np.zeros(129), n_calibration=199, calibration_set_sha256=_sha("a"),
    )
    assert scb.tail_rank(summary, "distance", 1.0) == pytest.approx(max(1.0e-4, 1.0 / 200.0))
    args = {key: default for key, (_, default) in scb.MODEL_CONFIG_SPEC.items()}
    args.update({"num_classes": 4, "sample_rate_hz": -1.0})
    checkpoint = {"args": args, "model": {"dom_head.net.3.bias": torch.zeros(2)}}
    namespace = {**args, "sample_rate_hz": 0.0, "num_classes": 4, "num_domains": 2, "input_len": 256}
    config = scb.resolve_model_config_projection(checkpoint=checkpoint, resolved_namespace=namespace)
    assert config["sample_rate_hz"] == 25000000.0 and config["arch_family"] == "cvsincnet"
    namespace["sample_rate_hz"] = "bad"
    with pytest.raises(scb.SingleControlBundleError):
        scb.resolve_model_config_projection(checkpoint=checkpoint, resolved_namespace=namespace)


def test_receipt_day_and_rx_strings_resolve_as_training_axis_indices() -> None:
    from dataset_wisig import _resolve_days, _resolve_rxs

    physical_days = ["2021-03-01", "2021-03-08", "2021-03-15", "2021-03-22"]
    physical_receivers = [f"receiver-{index:02d}" for index in range(12)]
    source_days = scb._frozen_receipt_axis_indices(
        ["0", "1"], physical_days, axis="day", expected_indices=scb.F1C_SOURCE_DAY_INDICES
    )
    source_receivers = scb._frozen_receipt_axis_indices(
        [str(index) for index in scb.F1C_SOURCE_RX_INDICES],
        physical_receivers,
        axis="receiver",
        expected_indices=scb.F1C_SOURCE_RX_INDICES,
    )
    target_days = scb._frozen_receipt_axis_indices(
        ["2", "3"], physical_days, axis="day", expected_indices=scb.F1C_TARGET_DAY_INDICES
    )
    target_receivers = scb._frozen_receipt_axis_indices(
        [str(index) for index in scb.F1C_TARGET_RX_INDICES],
        physical_receivers,
        axis="receiver",
        expected_indices=scb.F1C_TARGET_RX_INDICES,
    )
    assert source_days == _resolve_days(physical_days, [0, 1], [])
    assert source_receivers == _resolve_rxs(physical_receivers, list(scb.F1C_SOURCE_RX_INDICES), [])
    assert target_days == _resolve_days(physical_days, [2, 3], [])
    assert target_receivers == _resolve_rxs(physical_receivers, list(scb.F1C_TARGET_RX_INDICES), [])
    with pytest.raises(scb.SingleControlBundleError, match="duplicates"):
        scb._frozen_receipt_axis_indices(
            ["0", "0"], physical_days, axis="day", expected_indices=scb.F1C_SOURCE_DAY_INDICES
        )
    with pytest.raises(scb.SingleControlBundleError, match="out of range"):
        scb._frozen_receipt_axis_indices(
            ["0", "1"], physical_days[:1], axis="day", expected_indices=scb.F1C_SOURCE_DAY_INDICES
        )


def test_real_parser_namespace_materializes_exact_nine_model_fallbacks() -> None:
    checkpoint = _minimal_real_parser_checkpoint()
    namespace = scb._resolved_ssdg_namespace(checkpoint, device=torch.device("cpu"))
    observed_absent = {key for key in scb.MODEL_CONFIG_SPEC if not hasattr(namespace, key)}
    assert observed_absent == set(scb.MODEL_NAMESPACE_ABSENT_DEFAULTS)
    config = scb.resolve_model_config_projection(checkpoint=checkpoint, resolved_namespace=namespace)
    assert {key: config[key] for key in scb.MODEL_NAMESPACE_ABSENT_DEFAULTS} == scb.MODEL_NAMESPACE_ABSENT_DEFAULTS


@pytest.mark.parametrize("key", sorted(scb.MODEL_NAMESPACE_ABSENT_DEFAULTS))
def test_model_fallback_keys_reject_explicit_none_wrong_type_and_checkpoint_namespace_loss(key: str) -> None:
    checkpoint = _minimal_real_parser_checkpoint()
    namespace = scb._resolved_ssdg_namespace(checkpoint, device=torch.device("cpu"))
    base = dict(vars(namespace))
    explicit_none = dict(base)
    explicit_none[key] = None
    with pytest.raises(scb.SingleControlBundleError, match=key):
        scb.resolve_model_config_projection(checkpoint=checkpoint, resolved_namespace=explicit_none)
    wrong_type = dict(base)
    wrong_type[key] = 7 if isinstance(scb.MODEL_NAMESPACE_ABSENT_DEFAULTS[key], str) else "wrong"
    with pytest.raises(scb.SingleControlBundleError, match=key):
        scb.resolve_model_config_projection(checkpoint=checkpoint, resolved_namespace=wrong_type)

    checkpoint_with_key = _minimal_real_parser_checkpoint(
        extra_args={key: scb.MODEL_NAMESPACE_ABSENT_DEFAULTS[key]}
    )
    namespace_with_key = dict(
        vars(scb._resolved_ssdg_namespace(checkpoint_with_key, device=torch.device("cpu")))
    )
    namespace_with_key.pop(key)
    with pytest.raises(scb.SingleControlBundleError, match="despite checkpoint args"):
        scb.resolve_model_config_projection(checkpoint=checkpoint_with_key, resolved_namespace=namespace_with_key)


def test_model_config_rejects_checkpoint_none_and_nonallowlist_namespace_absence() -> None:
    checkpoint = _minimal_real_parser_checkpoint(extra_args={"dom_feature_key": None})
    namespace = scb._resolved_ssdg_namespace(checkpoint, device=torch.device("cpu"))
    with pytest.raises(scb.SingleControlBundleError, match="dom_feature_key"):
        scb.resolve_model_config_projection(checkpoint=checkpoint, resolved_namespace=namespace)

    clean_checkpoint = _minimal_real_parser_checkpoint()
    clean_namespace = dict(vars(scb._resolved_ssdg_namespace(clean_checkpoint, device=torch.device("cpu"))))
    clean_namespace.pop("model_variant")
    with pytest.raises(scb.SingleControlBundleError, match="model_variant is absent"):
        scb.resolve_model_config_projection(checkpoint=clean_checkpoint, resolved_namespace=clean_namespace)


def test_real_receipt_projection_and_false_false_satellite_semantics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    artifact = Path(r"E:\type10-7\automation_reports\CV-SincNet\phase1_cp_sfce12_20260809_v2\artifacts\run_outputs\F1C_CP_SFCE12")
    if not artifact.is_dir():
        pytest.skip("real F1C receipt artifact is unavailable in this checkout")
    receipt = scb.validate_f1c_receipts(
        completion_path=artifact / "phase1_training_completion_receipt.json",
        terminal_path=artifact / "phase1_terminal_status.json",
        cp_terminal_path=artifact / "phase1_cp_sfce_terminal_receipt.json",
        checkpoint_sha256=scb.F1C_CHECKPOINT_SHA256,
        dataset_sha256=scb.EXPECTED_DATASET_SHA256,
    )
    assert receipt["satellite_protocol"]["disjoint"] is False
    assert receipt["satellite_protocol"]["require_disjoint"] is False
    assert receipt["cp_terminal"]["source_known_validation_tx"] == "14-7"
    assert scb._recompute_satellite_protocol_projection() == receipt["satellite_protocol"]
    altered = json.loads((artifact / "phase1_terminal_status.json").read_text(encoding="utf-8"))
    altered["satellite_protocol"]["disjoint"] = True
    completion_copy = tmp_path / "completion.json"
    terminal_copy = tmp_path / "terminal.json"
    cp_copy = tmp_path / "cp.json"
    for source, destination in (
        (artifact / "phase1_training_completion_receipt.json", completion_copy),
        (artifact / "phase1_terminal_status.json", terminal_copy),
        (artifact / "phase1_cp_sfce_terminal_receipt.json", cp_copy),
    ):
        destination.write_bytes(source.read_bytes())
    terminal_copy.write_text(json.dumps(altered), encoding="utf-8")
    with pytest.raises(scb.SingleControlBundleError):
        scb.validate_f1c_receipts(
            completion_path=completion_copy,
            terminal_path=terminal_copy,
            cp_terminal_path=cp_copy,
            checkpoint_sha256=scb.F1C_CHECKPOINT_SHA256,
            dataset_sha256=scb.EXPECTED_DATASET_SHA256,
            fixture_mode=True,
        )
    live_drift = dict(receipt["satellite_protocol"])
    live_drift["registry_sha256"] = _sha("0")
    monkeypatch.setattr(scb, "_recompute_satellite_protocol_projection", lambda: live_drift)
    with pytest.raises(scb.SingleControlBundleError, match="live satellite protocol"):
        scb.validate_f1c_receipts(
            completion_path=artifact / "phase1_training_completion_receipt.json",
            terminal_path=artifact / "phase1_terminal_status.json",
            cp_terminal_path=artifact / "phase1_cp_sfce_terminal_receipt.json",
            checkpoint_sha256=scb.F1C_CHECKPOINT_SHA256,
            dataset_sha256=scb.EXPECTED_DATASET_SHA256,
        )


def test_code_allowlist_parity_receipt_and_member_negative_paths(tmp_path: Path) -> None:
    material = _materials()
    code_map = {name: _sha("a") for name in scb.EXPECTED_CODE_SHA_PATHS}
    digest = scb.resolved_config_sha256(
        receipt_projection={"receipt": "fixture"}, checkpoint_sha256=_sha("a"), dataset_sha256=_sha("b"),
        model_config={"cfg": "fixture"}, code_sha256=code_map, preprocessing_code_sha256=_sha("c"), scenario_code_sha256=_sha("d"),
    )
    assert len(digest) == 64
    with pytest.raises(scb.SingleControlBundleError):
        scb.resolved_config_sha256(
            receipt_projection={"receipt": "fixture"}, checkpoint_sha256=_sha("a"), dataset_sha256=_sha("b"),
            model_config={"cfg": "fixture"}, code_sha256={**code_map, "extra.py": _sha("e")},
            preprocessing_code_sha256=_sha("c"), scenario_code_sha256=_sha("d"),
        )
    result = scb.build_bundle(output_dir=tmp_path / "bundle", bundle_status=scb.FIXTURE_STATUS, **{k: v for k, v in material.items() if k != "eager_runtime"})
    (tmp_path / "bundle" / "state" / "rank_tail_summary.npz").unlink()
    with pytest.raises(scb.SingleControlBundleError):
        scb.load_bundle(tmp_path / "bundle", expected_content_root=result["content_root"], expected_bundle_status=scb.FIXTURE_STATUS)
    wrong_parity = dict(material["runtime_parity_receipt"])
    wrong_parity["max_abs"] = dict(wrong_parity["max_abs"])
    wrong_parity["max_abs"]["q"] = 1.0
    with pytest.raises(scb.SingleControlBundleError):
        scb._validate_runtime_parity_binding(wrong_parity, runtime_bytes=material["runtime_source"], state=material["state"])


def test_cli_fixture_build_and_external_root_verify(tmp_path: Path) -> None:
    script = Path(__file__).parents[1] / "scripts" / "build_phase1_single_control_bundle_v1.py"
    target = tmp_path / "cli-bundle"
    built = subprocess.run(
        [sys.executable, str(script), "--fixture-build", "--output-dir", str(target)],
        check=True, capture_output=True, text=True,
    )
    root = json.loads(built.stdout)["content_root"]
    default_status = subprocess.run(
        [sys.executable, str(script), "--verify-bundle", "--bundle-dir", str(target), "--expected-content-root", root],
        capture_output=True, text=True,
    )
    assert default_status.returncode != 0
    verified = subprocess.run(
        [sys.executable, str(script), "--verify-bundle", "--bundle-dir", str(target), "--expected-content-root", root,
         "--expected-status", scb.FIXTURE_STATUS],
        check=True, capture_output=True, text=True,
    )
    assert json.loads(verified.stdout)["content_root"] == root
