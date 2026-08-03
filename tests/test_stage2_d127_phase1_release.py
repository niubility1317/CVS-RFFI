from __future__ import annotations

from dataclasses import replace
import hashlib
import io
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cvsrffi import stage2_d106_phase1_tap as d106
from cvsrffi import stage2_d127_da_candidates as da
from cvsrffi import stage2_d127_phase1_assets as assets
from cvsrffi import stage2_d127_phase1_release as release


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_method_lock(
    tmp_path: Path,
    *,
    source_iq: str = "a" * 64,
    source_receipt: str = "c" * 64,
    source_join: str = "d" * 64,
) -> tuple[Path, str]:
    payload = {
        "schema": "cvs.stage2.d127.joint_s0.method_lock.v1",
        "candidate_id": "D127-LIGHT-DA-X-D92-LITE-S0",
        "protocol_schema": "p2_min_v1",
        "checkpoint": {"sha256": "b" * 64},
        "phase1_asset_build": {
            "source_received_iq_sha256": source_iq,
            "source_received_iq_receipt_sha256": source_receipt,
            "source_label_join_archive_sha256": source_join,
            "partition_schema": "d127-phase1-v1",
            "receiver_held_fold_count": 7,
            "physical_samples_per_receiver_class": 14,
            "support_pool_count": 5,
            "outer_query_pool_count": 9,
            "active_k": [1, 5],
            "final_episode_count": 14,
            "k1_is_first_k5_support": True,
            "support_query_globally_disjoint": True,
            "class_loco_training_count": 0,
            "persist_source_rows_or_features": False,
            "persist_fp32_sidecar": False,
            "optimizer": {
                "name": "full_batch_lbfgs",
                "max_iter": 128,
                "line_search_fn": "strong_wolfe",
                "initialization_count": 1,
                "early_stop": False,
                "parameter_scan": False,
            },
        },
        "domain_adaptation": {
            "query_fit_count": 0,
            "query_update_count": 0,
            "query_selection_count": 0,
            "candidates": [
                {"candidate_id": da.CANDIDATE_A, "tap": da.TAP_A},
                {"candidate_id": da.CANDIDATE_B, "tap": da.TAP_B},
                {"candidate_id": da.CANDIDATE_C, "tap": da.TAP_C},
            ],
        },
        "student_t_qknn": {
            "active_k": [1, 5],
            "student_nu": 3,
            "kernel_effective_dim": 12,
            "kernel_volume_gamma": 1,
            "shared_h0": 0.35,
            "scale_prior_strength": 2,
            "scale_min_ratio": 0.5,
            "scale_max_ratio": 2,
            "temperature": 0.85,
            "support_storage": "int8_fp16_scale",
            "phase1_lodo_receipt_sha256": "e" * 64,
            "quantization_margin_audit_sha256": "f" * 64,
        },
    }
    path = tmp_path / "method_lock.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path, _sha(path)


def _joined_rows() -> d106.D106JoinedLSRows:
    received: list[np.ndarray] = []
    labels: list[str] = []
    receivers: list[str] = []
    days: list[str] = []
    physical: list[str] = []
    scenes: list[str] = []
    observations: list[str] = []
    for receiver_index in range(7):
        for class_index in range(6):
            for sample_index in range(14):
                received.append(np.full((2, 256), receiver_index + class_index / 10.0 + sample_index / 100.0, dtype=np.float32))
                labels.append(f"source-tx-{class_index:02d}")
                receivers.append(f"source-rx-{receiver_index:02d}")
                days.append(f"day-{sample_index % 4}")
                physical.append(f"physical-{receiver_index:02d}-{class_index:02d}-{sample_index:02d}")
                scenes.append("leo_clear_weak")
                observations.append(f"observation-{receiver_index:02d}-{class_index:02d}-{sample_index:02d}")
    return d106.D106JoinedLSRows(
        received_iq=np.ascontiguousarray(np.stack(received), dtype=np.float32),
        tx_labels=np.asarray(labels, dtype=np.str_),
        receiver_ids=np.asarray(receivers, dtype=np.str_),
        day_ids=np.asarray(days, dtype=np.str_),
        physical_ids=np.asarray(physical, dtype=np.str_),
        scenario_names=np.asarray(scenes, dtype=np.str_),
        observation_ids=np.asarray(observations, dtype=np.str_),
    )


def _historical_selected_iq_arrays() -> dict[str, np.ndarray]:
    rows = _joined_rows()
    return {
        "received_iq": np.ascontiguousarray(rows.received_iq, dtype=np.float32),
        "receiver_ids": np.asarray(rows.receiver_ids, dtype=np.str_),
        "day_ids": np.asarray(rows.day_ids, dtype=np.str_),
        "physical_ids": np.asarray(rows.physical_ids, dtype=np.str_),
        "scenario_names": np.asarray(rows.scenario_names, dtype=np.str_),
        "observation_ids": np.asarray(rows.observation_ids, dtype=np.str_),
    }


def _historical_execution_closure() -> dict[str, object]:
    """Create a valid extraction receipt whose callable hash is historical."""

    value = json.loads(json.dumps(d106._execution_closure("extract")))
    row = value["callables"]["_load_completion_marker"]
    current = str(row["code_sha256"])
    row["code_sha256"] = ("0" if current[0] != "0" else "1") + current[1:]
    payload = {
        "schema": value["schema"],
        "stage": value["stage"],
        "callables": value["callables"],
        "construction_closure": value["construction_closure"],
    }
    value["execution_content_root_sha256"] = release._canonical_sha256(payload)
    return value


def _write_historical_completion_marker(root: Path) -> None:
    names = (
        d106.LS_IQ_ARCHIVE_NAME,
        d106.LS_IQ_RECEIPT_NAME,
        d106.LS_IQ_VALIDATOR_NAME,
    )
    marker = {
        "schema": d106.COMPLETION_MARKER_SCHEMA,
        "artifact_kind": "d106_ls_received_iq",
        "member_order": list(names),
        "member_sha256": {name: _sha(root / name) for name in names},
        "publication_policy": "atomic_output_reservation_exact_members_marker_last",
        "directory_atomic_visibility_claimed": False,
        "partial_output_acceptable": False,
    }
    (root / d106.COMPLETION_MARKER_NAME).write_bytes(d106._canonical_bytes(marker))


def _write_historical_selected_iq_fixture(
    tmp_path: Path,
    *,
    input_ls_archive_sha256: str = "d" * 64,
    extra_archive_member: bool = False,
    receipt_mutator: object | None = None,
) -> dict[str, object]:
    root = tmp_path / "historical-selected-iq"
    root.mkdir(parents=True)
    arrays = _historical_selected_iq_arrays()
    archive = root / d106.LS_IQ_ARCHIVE_NAME
    if extra_archive_member:
        payload = io.BytesIO()
        np.savez(payload, **(arrays | {"unexpected_member": np.asarray([1])}))
        archive.write_bytes(payload.getvalue())
    else:
        archive.write_bytes(d106._deterministic_npz_bytes(arrays))
    archive_sha = _sha(archive)
    array_hashes = {name: d106._array_sha256(value) for name, value in arrays.items()}
    physical_root = d106._ordered_id_root(arrays["physical_ids"].astype(str).tolist())
    selected_content = {
        "array_sha256": array_hashes,
        "row_count": d106.EXPECTED_COUNTS["L_s"],
        "physical_id_root_sha256": physical_root,
        "selection_salt_sha256": "e" * 64,
        "input_ls_archive_sha256": input_ls_archive_sha256,
    }
    execution = _historical_execution_closure()
    execution_root = str(execution["execution_content_root_sha256"])
    receipt: dict[str, object] = {
        "schema": d106.LS_IQ_RECEIPT_SCHEMA,
        "candidate_id": d106.CANDIDATE_ID,
        "split_id": d106.SPLIT_ID,
        "protocol_schema": d106.PROTOCOL_SCHEMA,
        "tap_input_schema": d106.LS_IQ_SCHEMA,
        "archive_name": d106.LS_IQ_ARCHIVE_NAME,
        "archive_sha256": archive_sha,
        "archive_members": list(d106.LS_IQ_MEMBERS),
        "array_sha256": array_hashes,
        "selected_content_root_sha256": release._canonical_sha256(selected_content),
        "row_count": d106.EXPECTED_COUNTS["L_s"],
        "rho_label": d106.RHO_LABEL,
        "physical_id_root_sha256": physical_root,
        "selection_salt_sha256": "e" * 64,
        "input_ls_archive_sha256": input_ls_archive_sha256,
        "scenario_order": list(d106.FORMAL_LEO_WEAK_SCENARIOS),
        "execution_closure": execution,
        "execution_pre_root_sha256": execution_root,
        "execution_post_root_sha256": execution_root,
        "contains_only_selected_ls_rows": True,
        "source_pool_labels_persisted": False,
        "clean_iq_access": False,
        "target_access": False,
        "formal_query_access": False,
    }
    if receipt_mutator is not None:
        receipt_mutator(receipt)
    receipt_path = root / d106.LS_IQ_RECEIPT_NAME
    receipt_path.write_bytes(d106._canonical_bytes(receipt))
    (root / d106.LS_IQ_VALIDATOR_NAME).write_bytes(b"historical-validator-receipt")
    _write_historical_completion_marker(root)
    return {
        "root": root,
        "archive": archive,
        "receipt": receipt_path,
        "archive_sha256": _sha(archive),
        "receipt_sha256": _sha(receipt_path),
    }


def _load_historical_fixture(fixture: dict[str, object]) -> d106.D106SelectedLSIQ:
    return release._load_d127_historical_d106_selected_ls_iq(
        fixture["archive"],
        fixture["receipt"],
        expected_archive_sha256=str(fixture["archive_sha256"]),
        expected_receipt_sha256=str(fixture["receipt_sha256"]),
    )


def test_historical_d106_receipt_accepts_fixed_hash_closure_but_live_d106_rejects(
    tmp_path: Path,
) -> None:
    fixture = _write_historical_selected_iq_fixture(tmp_path)
    with pytest.raises(d106.D106Phase1TapError, match="receipt closure drift"):
        d106.load_d106_ls_received_iq(
            fixture["archive"],
            fixture["receipt"],
            expected_archive_sha256=str(fixture["archive_sha256"]),
            expected_receipt_sha256=str(fixture["receipt_sha256"]),
        )
    loaded = _load_historical_fixture(fixture)
    assert isinstance(loaded, d106.D106SelectedLSIQ)
    assert loaded.received_iq.shape == (588, 2, 256)
    assert all(not value.flags.writeable for value in (
        loaded.received_iq,
        loaded.receiver_ids,
        loaded.day_ids,
        loaded.physical_ids,
        loaded.scenario_names,
        loaded.observation_ids,
    ))
    with pytest.raises(TypeError):
        loaded.receipt["clean_iq_access"] = True  # type: ignore[index]


def test_historical_d106_receipt_callable_set_is_frozen_from_current_d106(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _write_historical_selected_iq_fixture(tmp_path)
    current = tuple(d106.EXTRACT_EXECUTION_CALLABLES)
    monkeypatch.setattr(
        d106,
        "EXTRACT_EXECUTION_CALLABLES",
        (*current[1:], "d127_current_only_callable"),
    )
    loaded = _load_historical_fixture(fixture)
    assert loaded.receipt["execution_closure"]["stage"] == "extract"


@pytest.mark.parametrize(
    "kind",
    (
        "array_hash",
        "content_root",
        "protocol",
        "access",
        "execution_root",
        "callable_hash",
    ),
)
def test_historical_d106_receipt_rejects_semantic_drift(
    tmp_path: Path, kind: str
) -> None:
    def mutate(receipt: dict[str, object]) -> None:
        if kind == "array_hash":
            receipt["array_sha256"] = dict(receipt["array_sha256"])
            receipt["array_sha256"]["received_iq"] = "0" * 64
        elif kind == "content_root":
            receipt["selected_content_root_sha256"] = "0" * 64
        elif kind == "protocol":
            receipt["protocol_schema"] = "not-p2-min-v1"
        elif kind == "access":
            receipt["clean_iq_access"] = True
        elif kind == "execution_root":
            receipt["execution_closure"]["execution_content_root_sha256"] = "0" * 64
        elif kind == "callable_hash":
            receipt["execution_closure"]["callables"]["_load_completion_marker"]["code_sha256"] = "invalid"
        else:  # pragma: no cover - parameter guard.
            raise AssertionError(kind)

    fixture = _write_historical_selected_iq_fixture(
        tmp_path, receipt_mutator=mutate
    )
    with pytest.raises(release.D127Phase1ReleaseError):
        _load_historical_fixture(fixture)


def test_historical_d106_receipt_rejects_hash_member_marker_and_canonical_drift(
    tmp_path: Path,
) -> None:
    fixture = _write_historical_selected_iq_fixture(tmp_path / "hash")
    with pytest.raises(release.D127Phase1ReleaseError, match="archive SHA256"):
        release._load_d127_historical_d106_selected_ls_iq(
            fixture["archive"],
            fixture["receipt"],
            expected_archive_sha256="0" * 64,
            expected_receipt_sha256=str(fixture["receipt_sha256"]),
        )
    with pytest.raises(release.D127Phase1ReleaseError, match="receipt SHA256"):
        release._load_d127_historical_d106_selected_ls_iq(
            fixture["archive"],
            fixture["receipt"],
            expected_archive_sha256=str(fixture["archive_sha256"]),
            expected_receipt_sha256="0" * 64,
        )

    member_fixture = _write_historical_selected_iq_fixture(
        tmp_path / "member", extra_archive_member=True
    )
    with pytest.raises(release.D127Phase1ReleaseError, match="member drift"):
        _load_historical_fixture(member_fixture)

    marker_fixture = _write_historical_selected_iq_fixture(tmp_path / "marker")
    marker_path = Path(marker_fixture["root"]) / d106.COMPLETION_MARKER_NAME
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["member_sha256"][d106.LS_IQ_ARCHIVE_NAME] = "0" * 64
    marker_path.write_bytes(d106._canonical_bytes(marker))
    with pytest.raises(release.D127Phase1ReleaseError, match="completion closure"):
        _load_historical_fixture(marker_fixture)

    canonical_fixture = _write_historical_selected_iq_fixture(tmp_path / "canonical")
    receipt_path = Path(canonical_fixture["receipt"])
    receipt_path.write_bytes(receipt_path.read_bytes() + b"\n")
    _write_historical_completion_marker(Path(canonical_fixture["root"]))
    canonical_fixture["receipt_sha256"] = _sha(receipt_path)
    with pytest.raises(release.D127Phase1ReleaseError, match="not canonical"):
        _load_historical_fixture(canonical_fixture)


def test_real_joined_loader_uses_historical_path_and_binds_input_ls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    label_archive = tmp_path / "d104_ls_labels.npz"
    label_archive.write_bytes(b"sealed-L_s-label-join")
    label_sha = _sha(label_archive)
    fixture = _write_historical_selected_iq_fixture(
        tmp_path, input_ls_archive_sha256=label_sha
    )
    lock_path, lock_sha = _write_method_lock(
        tmp_path,
        source_iq=str(fixture["archive_sha256"]),
        source_receipt=str(fixture["receipt_sha256"]),
        source_join=label_sha,
    )
    method_lock = release.load_d127_phase1_method_lock(
        lock_path, expected_sha256=lock_sha
    )

    def unexpected_live_loader(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("D127 must not use the live D106 closure loader")

    monkeypatch.setattr(d106, "load_d106_ls_received_iq", unexpected_live_loader)
    monkeypatch.setattr(
        d106,
        "join_d106_ls_observations",
        lambda *_args, **_kwargs: _joined_rows(),
    )
    joined = release._load_real_d127_phase1_joined_rows(
        selected_iq_archive=fixture["archive"],
        selected_iq_archive_sha256=str(fixture["archive_sha256"]),
        selected_iq_receipt=fixture["receipt"],
        selected_iq_receipt_sha256=str(fixture["receipt_sha256"]),
        ls_label_join_archive=label_archive,
        ls_label_join_archive_sha256=label_sha,
        method_lock=method_lock,
    )
    assert joined.received_iq.shape == (588, 2, 256)


def _quantized_assets() -> dict[str, release.QuantizedD127Asset]:
    fsrg_a = da.FSRGAsset(
        candidate_id=da.CANDIDATE_A,
        tap_name=da.TAP_A,
        U=torch.tensor([[0.1, 0.2], [0.2, -0.1], [0.05, 0.03], [-0.2, 0.1]], dtype=torch.float32),
        V=torch.tensor([[0.1, 0.2, 0.3, 0.4], [-0.2, 0.1, -0.3, 0.2]], dtype=torch.float32),
        d_f_diag=torch.tensor([1.0, 2.0], dtype=torch.float32),
        rho=0.5,
    )
    fsrg_b = da.FSRGAsset(
        candidate_id=da.CANDIDATE_B,
        tap_name=da.TAP_B,
        U=torch.tensor([[0.3, -0.1], [-0.1, 0.2], [0.12, 0.04], [0.2, 0.3]], dtype=torch.float32),
        V=torch.tensor([[0.2, -0.3, 0.4, 0.1], [0.1, 0.2, -0.1, -0.2]], dtype=torch.float32),
        d_f_diag=torch.tensor([1.5, 1.2], dtype=torch.float32),
        rho=0.4,
    )
    rdha = da.RDHAAsset(
        U=torch.full((da.JOINT_PROJ_INPUT_DIM, 2), 0.01, dtype=torch.float32),
        V=torch.full((2, da.JOINT_PROJ_INPUT_DIM), -0.01, dtype=torch.float32),
        Q=torch.tensor([[0.2, 0.1, 0.0, 0.1, 0.2], [0.1, -0.1, 0.2, 0.0, 0.1]], dtype=torch.float32),
        b=torch.tensor([0.05, -0.05], dtype=torch.float32),
        mean_p1=torch.zeros(5, dtype=torch.float32),
        std_p1=torch.ones(5, dtype=torch.float32),
        a_max=0.2,
    )
    return {
        da.CANDIDATE_A: assets.quantize_fsrg_asset(fsrg_a),
        da.CANDIDATE_B: assets.quantize_fsrg_asset(fsrg_b),
        da.CANDIDATE_C: assets.quantize_rdah_asset(rdha),
    }


def test_real_joined_rows_build_deterministic_k1_k5_episode_manifest_without_source_keys(tmp_path: Path) -> None:
    lock_path, lock_sha = _write_method_lock(tmp_path)
    method_lock = release.load_d127_phase1_method_lock(lock_path, expected_sha256=lock_sha)
    qknn_locks = release.build_d127_phase1_qknn_locks(method_lock)
    assert tuple(qknn_locks) == (1, 5)
    assert qknn_locks[1].phase1_lodo_receipt_sha256 == "e" * 64
    assert qknn_locks[5].quantization_margin_audit_sha256 == "f" * 64

    rows = _joined_rows()
    first = release.build_d127_phase1_episode_plan(rows, method_lock=method_lock)
    second = release.build_d127_phase1_episode_plan(rows, method_lock=method_lock)
    assert first.manifest == second.manifest
    assert first.contract_sha256 == second.contract_sha256
    assert len(first.episodes) == 14
    assert tuple((item.fold_ordinal, item.k_shot) for item in first.episodes) == tuple((fold, k) for fold in range(7) for k in (1, 5))
    for fold in range(7):
        k1 = next(item for item in first.episodes if item.fold_ordinal == fold and item.k_shot == 1)
        k5 = next(item for item in first.episodes if item.fold_ordinal == fold and item.k_shot == 5)
        assert set(k1.support_indices).issubset(k5.support_indices)
        assert k1.query_indices == k5.query_indices
        assert not set(k5.support_physical_ids).intersection(k5.query_physical_ids)

    persisted = json.dumps(dict(first.manifest), sort_keys=True)
    assert "source-rx-" not in persisted
    assert "source-tx-" not in persisted
    assert "physical-00-00-00" not in persisted
    assert first.manifest["source_rows_or_features_persisted"] is False
    assert first.manifest["receiver_or_class_keys_persisted"] is False


def test_single_candidate_wire_merge_and_readonly_full_bundle(tmp_path: Path) -> None:
    lock_path, lock_sha = _write_method_lock(tmp_path)
    method_lock = release.load_d127_phase1_method_lock(lock_path, expected_sha256=lock_sha)
    plan = release.build_d127_phase1_episode_plan(_joined_rows(), method_lock=method_lock)
    quantized = _quantized_assets()
    parts: list[Path] = []
    for index, candidate in enumerate(release.CANDIDATE_IDS):
        output = tmp_path / f"single-{index}"
        result = release.write_d127_phase1_single_candidate_bundle(
            output_dir=output,
            candidate_id=candidate,
            asset=quantized[candidate],
            method_lock=method_lock,
            episode_plan=plan,
        )
        assert result["candidate_id"] == candidate
        assert (output / release.MANIFEST_FILE_NAME).is_file()
        parts.append(output)

    merged = release.merge_d127_phase1_asset_bundles((parts[1], parts[2], parts[0]), output_dir=tmp_path / "merged")
    values = release.load_d127_phase1_asset_bundle(tmp_path / "merged", merged["manifest_sha256"])
    assert tuple(values) == release.CANDIDATE_IDS
    assert isinstance(values[da.CANDIDATE_A], assets.QuantizedFSRGAsset)
    assert isinstance(values[da.CANDIDATE_B], assets.QuantizedFSRGAsset)
    assert isinstance(values[da.CANDIDATE_C], assets.QuantizedRDHAAsset)
    assert all(item.persistent_fp32_sidecar is False for item in values.values())
    assert all(item.decode().candidate_id == candidate for candidate, item in values.items())
    with pytest.raises(release.D127Phase1ReleaseError, match="overwrite"):
        release.write_d127_phase1_single_candidate_bundle(
            output_dir=parts[0],
            candidate_id=da.CANDIDATE_A,
            asset=quantized[da.CANDIDATE_A],
            method_lock=method_lock,
            episode_plan=plan,
        )
    (tmp_path / "merged" / "unexpected_fp32_sidecar.npy").write_bytes(b"not-permitted")
    with pytest.raises(release.D127Phase1ReleaseError, match="unapproved sidecar"):
        release.load_d127_phase1_asset_bundle(tmp_path / "merged", merged["manifest_sha256"])


def test_merge_rejects_a_different_source_or_method_binding(tmp_path: Path) -> None:
    first_dir = tmp_path / "first"
    first_dir.mkdir()
    path_a, sha_a = _write_method_lock(first_dir)
    lock_a = release.load_d127_phase1_method_lock(path_a, expected_sha256=sha_a)
    plan_a = release.build_d127_phase1_episode_plan(_joined_rows(), method_lock=lock_a)
    foreign_lock = replace(lock_a, lock_sha256="1" * 64, source_received_iq_sha256="2" * 64)
    plan_b = release.build_d127_phase1_episode_plan(_joined_rows(), method_lock=foreign_lock)
    quantized = _quantized_assets()
    output_a = tmp_path / "a"
    output_b = tmp_path / "b"
    output_c = tmp_path / "c"
    release.write_d127_phase1_single_candidate_bundle(output_dir=output_a, candidate_id=da.CANDIDATE_A, asset=quantized[da.CANDIDATE_A], method_lock=lock_a, episode_plan=plan_a)
    release.write_d127_phase1_single_candidate_bundle(output_dir=output_b, candidate_id=da.CANDIDATE_B, asset=quantized[da.CANDIDATE_B], method_lock=lock_a, episode_plan=plan_a)
    release.write_d127_phase1_single_candidate_bundle(output_dir=output_c, candidate_id=da.CANDIDATE_C, asset=quantized[da.CANDIDATE_C], method_lock=foreign_lock, episode_plan=plan_b)
    with pytest.raises(release.D127Phase1ReleaseError, match="binding drift"):
        release.merge_d127_phase1_asset_bundles((output_a, output_b, output_c), output_dir=tmp_path / "must-not-merge")


class _LightweightCheckpointBridge:
    """A test-only bridge with the real production bridge method shapes."""

    def __init__(self, _model: object, *, candidate_id: str, episode_iq_by_id: object) -> None:
        assert candidate_id == da.CANDIDATE_A
        self._episodes = dict(episode_iq_by_id)

    @staticmethod
    def _forward_from_iq(iq: torch.Tensor) -> SimpleNamespace:
        scalar = iq[:, 0, 0]
        receiver = torch.floor(scalar)
        class_code = torch.remainder(torch.round((scalar - receiver) * 10.0), 10.0)
        tap = torch.stack((scalar, receiver.square() + class_code * 0.7, receiver.pow(3) + class_code.square() * 0.3, receiver.pow(4) + class_code.pow(3) * 0.08), dim=1).to(dtype=torch.float32)
        return _LightweightCheckpointBridge._forward_from_tap(tap)

    @staticmethod
    def _forward_from_tap(tap: torch.Tensor) -> SimpleNamespace:
        basis = torch.arange(1, 161, dtype=torch.float32, device=tap.device).reshape(1, -1)
        projection = torch.cat((torch.sin(basis * 0.07), torch.cos(basis * 0.11), torch.sin(basis * 0.17), torch.cos(basis * 0.23)), dim=0)
        pre_relu = tap @ projection * 0.05 + basis * 0.001 + 2.0
        z_id = torch.nn.functional.normalize(pre_relu, dim=1)
        hidden = tap.mean(dim=1, keepdim=True) + torch.arange(320, dtype=torch.float32, device=tap.device).reshape(1, -1) * 0.0001
        return SimpleNamespace(tap=tap, hidden=hidden, pre_relu=pre_relu, z_id=z_id)

    def capture_raw(self, episode_id: str, *, split: str) -> SimpleNamespace:
        raw = self._episodes[episode_id]
        return self._forward_from_iq(raw.support_iq if split == "support" else raw.query_iq)

    def forward_with_replacement(self, _episode: object, *, split: str, replacement: torch.Tensor) -> SimpleNamespace:
        assert split in ("support", "query")
        return self._forward_from_tap(replacement)

    def fsrg_loss_callbacks(self, *, qknn_locks: object) -> assets.FSRGLossCallbacks:
        assert tuple(qknn_locks) == (1, 5)

        def support_per_sample(episode: assets.FSRGEpisode, adapted: torch.Tensor) -> torch.Tensor:
            rows = _LightweightCheckpointBridge._forward_from_tap(adapted).z_id
            classes, inverse = torch.unique(episode.support_labels, sorted=True, return_inverse=True)
            prototypes = torch.stack(
                [rows[inverse == class_index].mean(dim=0) for class_index in range(len(classes))]
            )
            prototypes = torch.nn.functional.normalize(prototypes, dim=1).detach()
            return torch.nn.functional.cross_entropy(rows @ prototypes.transpose(0, 1), inverse, reduction="none").to(dtype=torch.float32)

        def outer_query_per_sample(episode: assets.FSRGEpisode, support: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
            support_rows = _LightweightCheckpointBridge._forward_from_tap(support).z_id
            query_rows = _LightweightCheckpointBridge._forward_from_tap(query).z_id
            classes, inverse = torch.unique(episode.support_labels, sorted=True, return_inverse=True)
            prototypes = torch.stack(
                [support_rows[inverse == class_index].mean(dim=0) for class_index in range(len(classes))]
            )
            prototypes = torch.nn.functional.normalize(prototypes, dim=1)
            target = torch.argmax(
                (episode.query_labels.reshape(-1, 1) == classes.reshape(1, -1)).to(dtype=torch.int64),
                dim=1,
            )
            return torch.nn.functional.cross_entropy(query_rows @ prototypes.transpose(0, 1), target, reduction="none").to(dtype=torch.float32)

        return assets.FSRGLossCallbacks(
            support_per_sample=support_per_sample,
            outer_query_per_sample=outer_query_per_sample,
        )

    def deployment_qknn_logits(self, episode: assets.FSRGEpisode, *, support_zid: torch.Tensor, query_zid: torch.Tensor, qknn_locks: object) -> torch.Tensor:
        assert episode.k_shot in qknn_locks
        prototype = torch.stack(
            [support_zid[episode.support_labels == class_id].mean(dim=0) for class_id in range(6)]
        )
        return query_zid @ prototype.transpose(0, 1)


def test_internal_real_shape_bridge_path_runs_outer7_final14_without_public_injection(tmp_path: Path) -> None:
    lock_path, lock_sha = _write_method_lock(tmp_path)
    method_lock = release.load_d127_phase1_method_lock(lock_path, expected_sha256=lock_sha)
    result = release._build_d127_phase1_single_candidate_from_joined_rows(
        candidate_id=da.CANDIDATE_A,
        output_dir=tmp_path / "trained-a",
        method_lock=method_lock,
        joined_rows=_joined_rows(),
        model=object(),
        device=torch.device("cpu"),
        bridge_factory=_LightweightCheckpointBridge,
    )
    assert result["outer_fold_count"] == 7
    assert result["final_episode_count"] == 14
    manifest = json.loads((tmp_path / "trained-a" / release.MANIFEST_FILE_NAME).read_text(encoding="utf-8"))
    receipt = manifest["phase1_training_receipt"]
    assert receipt["execution"] == "REAL_CHECKPOINT_SOURCE_ONLY_OUTER7_FINAL14"
    assert len(receipt["outer_folds"]) == 7
    assert all(item["performance_threshold_or_ranking_used"] is False for item in receipt["outer_folds"])
    assert all(item["source_held_isolation"] and item["fixed_cyclic_label_equivariant"] and item["nonzero_state_or_gradient"] and item["query_function_changed"] for item in receipt["outer_folds"])
    public_parameters = set(inspect.signature(release.build_d127_phase1_single_candidate_from_source).parameters)
    assert "bridge_factory" not in public_parameters


def test_real_episode_iq_and_labels_survive_disabled_numpy_torch_abi_bridge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _blocked(*_args: object, **_kwargs: object) -> object:
        raise TypeError("simulated NumPy2/Torch2.1 ABI mismatch")

    monkeypatch.setattr(torch, "from_numpy", _blocked)
    monkeypatch.setattr(torch, "as_tensor", _blocked)
    lock_path, lock_sha = _write_method_lock(tmp_path)
    method_lock = release.load_d127_phase1_method_lock(lock_path, expected_sha256=lock_sha)
    plan = release.build_d127_phase1_episode_plan(_joined_rows(), method_lock=method_lock)
    raw = release._episode_raw_iq_by_id(plan, _joined_rows(), device=torch.device("cpu"))
    assert len(raw) == 14
    assert all(item.support_iq.dtype == torch.float32 for item in raw.values())
    runtime = release._candidate_episode_runtime(
        candidate_id=da.CANDIDATE_A,
        plan=plan,
        joined_rows=_joined_rows(),
        model=object(),
        device=torch.device("cpu"),
        bridge_factory=_LightweightCheckpointBridge,
        method_lock=method_lock,
    )
    assert len(runtime.episodes) == 14
    assert all(item.support_labels.dtype == torch.long for item in runtime.episodes)
    assert all(item.query_labels.dtype == torch.long for item in runtime.episodes)
