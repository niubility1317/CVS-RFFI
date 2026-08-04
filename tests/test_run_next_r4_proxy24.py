from __future__ import annotations

import dataclasses
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from cvsrffi import stage2_next_r4_fa_rdce3 as fa
from cvsrffi import stage2_next_r4_matrix as matrix
from cvsrffi import stage2_next_r4_runtime as runtime
from cvsrffi import stage2_zid_student_t_qknn as qknn


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "code" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import run_next_r4_proxy24 as runner  # noqa: E402


# Reuse the repository's deterministic artifact fixture registry so the
# lifecycle test can feed its already-validated 24-row sealed result set.
CLASSES = ("tx-z", "tx-a", "tx-f", "tx-c", "tx-e", "tx-b")
CHECKPOINT_BYTES = b"synthetic-checkpoint-for-runner-test"


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _unit(primary: int, variant: int = 0) -> np.ndarray:
    value = np.zeros(runtime.Z_DIM, dtype=np.float32)
    value[20 + primary] = np.float32(0.94)
    # The synthetic FA basis observes the first three coordinates.  Give each
    # class a distinct nonnegative code there so CER has a unique float32 top
    # after the low-rank transform (the production bridge is unaffected).
    value[0] = np.float32(0.08 + 0.07 * primary + 0.005 * variant)
    value[1] = np.float32(0.04 + 0.031 * ((primary * 2) % 5))
    value[2] = np.float32(0.06 + 0.019 * primary)
    if variant:
        value[80 + (primary + variant) % 40] = np.float32(0.01 * variant)
    value /= np.float32(np.linalg.norm(value.astype(np.float64)))
    return value


def _lock(active_k: int) -> qknn.Phase1ZIDStudentTLock:
    return qknn.Phase1ZIDStudentTLock(
        active_k=active_k,
        student_nu=3.0,
        kernel_effective_dim=12,
        kernel_volume_gamma=1.0,
        shared_h0=0.35,
        scale_prior_strength=2.0,
        scale_min_ratio=0.5,
        scale_max_ratio=2.0,
        temperature=0.85,
        phase1_lodo_receipt_sha256="1" * 64,
        quantization_margin_audit_sha256="2" * 64,
    )


def _asset(old_classes: tuple[str, ...], checkpoint_sha256: str) -> fa.FARDCE3Phase1Asset:
    basis = np.zeros((fa.RANK, fa.Z_DIM), dtype=np.float32)
    basis[0, 0] = 1.0
    basis[1, 1] = 1.0
    basis[2, 2] = 1.0
    return fa.build_fa_rdce3_phase1_asset(
        old_classes=old_classes,
        aggregate_samples_per_class=tuple(3 for _ in old_classes),
        class_centers_3d=np.zeros((len(old_classes), fa.RANK), dtype=np.float32),
        fisher_precision_3d=np.asarray([0.5, 0.6, 0.7], dtype=np.float32),
        residual_variance_3d=np.asarray([0.8, 0.9, 1.0], dtype=np.float32),
        fisher_radius=np.asarray([0.75], dtype=np.float32),
        rdce_kappa_3d=np.asarray([0.25, 0.10, 0.05], dtype=np.float32),
        basis_3x160=basis,
        checkpoint_sha256=checkpoint_sha256,
        phase1_bundle_sha256=_sha(b"bundle"),
        phase1_aggregate_receipt_sha256=_sha(b"aggregate"),
        method_lock_sha256=_sha(b"method-lock"),
    )


@dataclasses.dataclass
class _Fixture:
    received: Path
    metadata: Path
    asset_manifest: Path
    checkpoint: Path
    received_sha256: str
    metadata_sha256: str
    asset_manifest_sha256: str
    checkpoint_sha256: str
    truth: dict[str, str]
    feature_by_physical: dict[str, np.ndarray]


def _write_npz(path: Path, **arrays: np.ndarray) -> str:
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    path.write_bytes(buffer.getvalue())
    return _sha(path.read_bytes())


def _make_fixture(tmp_path: Path) -> _Fixture:
    checkpoint_sha = _sha(CHECKPOINT_BYTES)
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(CHECKPOINT_BYTES)
    plan = matrix.build_next_r4_proxy24_plan(CLASSES)

    physical_ids: list[str] = []
    observation_ids: list[str] = []
    receiver_ids: list[str] = []
    feature_by_physical: dict[str, np.ndarray] = {}
    truth: dict[str, str] = {}
    phase1_by_outer: dict[tuple[str, str], tuple[str, ...]] = {}
    support1_by_outer: dict[tuple[str, str], dict[str, tuple[str, ...]]] = {}
    support5_by_outer: dict[tuple[str, str], dict[str, tuple[str, ...]]] = {}
    query_by_outer: dict[tuple[str, str], dict[str, tuple[str, ...]]] = {}
    obs_by_outer: dict[tuple[str, str], dict[str, tuple[str, ...]]] = {}

    for planned in plan["rows"]:
        outer = (str(planned["held_receiver"]), str(planned["held_class"]))
        if outer in query_by_outer:
            continue
        support5: dict[str, tuple[str, ...]] = {}
        support1: dict[str, tuple[str, ...]] = {}
        query: dict[str, tuple[str, ...]] = {}
        observation: dict[str, tuple[str, ...]] = {}
        for class_index, class_id in enumerate(CLASSES):
            supports = tuple(f"p|{outer[0]}|{outer[1]}|{class_id}|s{index}" for index in range(5))
            queries = tuple(f"p|{outer[0]}|{outer[1]}|{class_id}|q{index}" for index in range(2))
            observations = tuple(f"o|{outer[0]}|{outer[1]}|{class_id}|q{index}" for index in range(2))
            support5[class_id] = supports
            support1[class_id] = supports[:1]
            query[class_id] = queries
            observation[class_id] = observations
            for variant, physical_id in enumerate(supports, start=1):
                physical_ids.append(physical_id)
                observation_ids.append(f"obs-support|{physical_id}")
                receiver_ids.append(outer[0])
                feature_by_physical[physical_id] = _unit(class_index, variant)
            for variant, physical_id in enumerate(queries):
                physical_ids.append(physical_id)
                observation_ids.append(observations[variant])
                receiver_ids.append(outer[0])
                feature_by_physical[physical_id] = _unit(class_index, 0 if variant == 0 else 6)
                truth[physical_id] = class_id
        phase1_by_outer[outer] = tuple(f"phase1|{outer[0]}|{outer[1]}|{index}" for index in range(2))
        # Formal D106 inputs carry the Phase1 fit members in the same
        # received-IQ physical archive.  They remain prepare-only and are
        # never requested by the predictor bridge.
        for phase1_id in phase1_by_outer[outer]:
            physical_ids.append(phase1_id)
            observation_ids.append(f"obs-phase1|{phase1_id}")
            receiver_ids.append(outer[0])
            feature_by_physical[phase1_id] = _unit(0, 9)
        support1_by_outer[outer] = support1
        support5_by_outer[outer] = support5
        query_by_outer[outer] = query
        obs_by_outer[outer] = observation

    received = tmp_path / "received_iq.npz"
    received_sha = _write_npz(
        received,
        received_iq=np.zeros((len(physical_ids), 2, 16), dtype="<f4"),
        receiver_ids=np.asarray(receiver_ids, dtype="<U16"),
        physical_ids=np.asarray(physical_ids, dtype="<U96"),
        observation_ids=np.asarray(observation_ids, dtype="<U112"),
        scenario_names=np.asarray(["leo_fixture"] * len(physical_ids), dtype="<U16"),
    )

    asset_dir = tmp_path / "assets"
    asset_dir.mkdir()
    entries: dict[str, dict[str, str]] = {}
    for (receiver, held_class), phase1_ids in phase1_by_outer.items():
        asset = _asset(tuple(class_id for class_id in CLASSES if class_id != held_class), checkpoint_sha)
        asset_path = asset_dir / f"{receiver.replace('-', '_')}__{held_class}.fa"
        asset_bytes = fa.serialize_fa_rdce3_phase1_asset(asset)
        asset_path.write_bytes(asset_bytes)
        key = f"{receiver}|{held_class}"
        entries[key] = {
            "asset_path": str(asset_path.resolve()),
            "asset_sha256": _sha(asset_bytes),
            "checkpoint_sha256": checkpoint_sha,
            "phase1_fit_physical_root_sha256": runner._ordered_root(phase1_ids),
        }
    asset_manifest = tmp_path / "fa_asset_manifest.json"
    asset_manifest.write_text(
        json.dumps({"schema": runner.ASSET_MANIFEST_SCHEMA, "entries": entries}, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    asset_manifest_sha = _sha(asset_manifest.read_bytes())

    rows: list[dict[str, object]] = []
    for planned in plan["rows"]:
        outer = (str(planned["held_receiver"]), str(planned["held_class"]))
        rows.append(
            {
                "row_id": planned["row_id"],
                "held_receiver": planned["held_receiver"],
                "held_class": planned["held_class"],
                "active_k": planned["active_k"],
                "k1_support_ids_by_class": {key: list(value) for key, value in support1_by_outer[outer].items()},
                "k5_support_ids_by_class": {key: list(value) for key, value in support5_by_outer[outer].items()},
                "query_ids_by_class": {key: list(value) for key, value in query_by_outer[outer].items()},
                "query_observation_ids_by_class": {key: list(value) for key, value in obs_by_outer[outer].items()},
                "phase1_fit_ids": list(phase1_by_outer[outer]),
            }
        )
    metadata_value = {
        "schema": "cvs.stage2.next_r4.proxy24.capsule_metadata.v1",
        "protocol_schema": matrix.PROTOCOL_SCHEMA,
        "phase2_data_status": "VALIDATED_ONCE",
        "capsule_id": _sha(b"capsule"),
        "split_id": _sha(b"split"),
        "validator_receipt_sha256": _sha(b"validator"),
        "class_registry": list(CLASSES),
        "held_receivers": list(matrix.HELD_RECEIVERS),
        "rows": rows,
        "seed": 7,
        "qknn_lock_by_k": {str(active_k): dataclasses.asdict(_lock(active_k)) for active_k in matrix.K_VALUES},
    }
    metadata = tmp_path / "capsule_metadata.json"
    metadata.write_text(json.dumps(metadata_value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return _Fixture(
        received=received,
        metadata=metadata,
        asset_manifest=asset_manifest,
        checkpoint=checkpoint,
        received_sha256=received_sha,
        metadata_sha256=_sha(metadata.read_bytes()),
        asset_manifest_sha256=asset_manifest_sha,
        checkpoint_sha256=checkpoint_sha,
        truth=truth,
        feature_by_physical=feature_by_physical,
    )


class _Bridge:
    def __init__(self, capsule: runner.ReceivedCapsule, feature_by_physical: dict[str, np.ndarray], checkpoint_sha256: str) -> None:
        self.capsule = capsule
        self.feature_by_physical = feature_by_physical
        self.checkpoint_sha256 = checkpoint_sha256

    def forward_indices(self, indices: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
        values = np.stack([self.feature_by_physical[self.capsule.physical_ids[index]] for index in indices]).astype(np.float32)
        return np.zeros((len(indices), 6), dtype=np.float32), values


def _args(fixture: _Fixture, output_dir: Path) -> object:
    return runner._parser().parse_args(
        [
            "prepare",
            "--output-dir",
            str(output_dir),
            "--received-iq",
            str(fixture.received),
            "--received-iq-sha256",
            fixture.received_sha256,
            "--capsule-metadata",
            str(fixture.metadata),
            "--capsule-metadata-sha256",
            fixture.metadata_sha256,
            "--fa-asset-manifest",
            str(fixture.asset_manifest),
            "--fa-asset-manifest-sha256",
            fixture.asset_manifest_sha256,
            "--checkpoint-sha256",
            fixture.checkpoint_sha256,
        ]
    )


def _contains_key(value: object, target: str) -> bool:
    if isinstance(value, dict):
        return target in value or any(_contains_key(item, target) for item in value.values())
    if isinstance(value, list):
        return any(_contains_key(item, target) for item in value)
    return False


def test_help_and_predict_cli_do_not_offer_truth_or_split_label_inputs() -> None:
    parser = runner._parser()
    subparsers = next(action for action in parser._actions if getattr(action, "choices", None))
    predict_help = subparsers.choices["predict"].format_help()
    assert "--truth" not in predict_help
    assert "--truth-sha256" not in predict_help
    assert "split-label" not in predict_help.lower()
    assert "--fa-asset-manifest" in predict_help


def test_missing_real_input_fails_closed_and_dynamic_capsule_has_no_588_gate(tmp_path: Path) -> None:
    path = tmp_path / "small.npz"
    payload = io.BytesIO()
    np.savez(
        payload,
        received_iq=np.zeros((2, 2, 8), dtype="<f4"),
        receiver_ids=np.asarray(["1-1", "1-1"]),
        physical_ids=np.asarray(["p0", "p1"]),
        observation_ids=np.asarray(["o0", "o1"]),
    )
    path.write_bytes(payload.getvalue())
    loaded = runner._load_received_capsule(path, _sha(path.read_bytes()))
    assert len(loaded.physical_ids) == 2
    # D106 day provenance is accepted at the capsule boundary but is not
    # propagated into the predictor-facing object.
    with_day = tmp_path / "small_with_day_ids.npz"
    day_payload = io.BytesIO()
    np.savez(
        day_payload,
        received_iq=np.zeros((2, 2, 8), dtype="<f4"),
        receiver_ids=np.asarray(["1-1", "1-1"]),
        physical_ids=np.asarray(["p0", "p1"]),
        observation_ids=np.asarray(["o0", "o1"]),
        day_ids=np.asarray(["day-a", "day-b"], dtype="<U8"),
    )
    with_day.write_bytes(day_payload.getvalue())
    loaded_with_day = runner._load_received_capsule(with_day, _sha(with_day.read_bytes()))
    assert not hasattr(loaded_with_day, "day_ids")

    malformed_day = tmp_path / "malformed_day_ids.npz"
    malformed_payload = io.BytesIO()
    np.savez(
        malformed_payload,
        received_iq=np.zeros((2, 2, 8), dtype="<f4"),
        receiver_ids=np.asarray(["1-1", "1-1"]),
        physical_ids=np.asarray(["p0", "p1"]),
        observation_ids=np.asarray(["o0", "o1"]),
        day_ids=np.asarray([["day-a", "day-b"]], dtype="<U8"),
    )
    malformed_day.write_bytes(malformed_payload.getvalue())
    with pytest.raises(runner.MissingRealInputArtifacts, match="day_ids"):
        runner._load_received_capsule(malformed_day, _sha(malformed_day.read_bytes()))

    unknown_member = tmp_path / "unknown_member.npz"
    unknown_payload = io.BytesIO()
    np.savez(
        unknown_payload,
        received_iq=np.zeros((2, 2, 8), dtype="<f4"),
        receiver_ids=np.asarray(["1-1", "1-1"]),
        physical_ids=np.asarray(["p0", "p1"]),
        observation_ids=np.asarray(["o0", "o1"]),
        unexpected=np.asarray(["x", "y"]),
    )
    unknown_member.write_bytes(unknown_payload.getvalue())
    with pytest.raises(runner.MissingRealInputArtifacts, match="member drift"):
        runner._load_received_capsule(unknown_member, _sha(unknown_member.read_bytes()))

    with pytest.raises(runner.MissingRealInputArtifacts, match=runner.MISSING_PREFIX):
        runner._load_received_capsule(tmp_path / "missing.npz", "0" * 64)


def test_prepare_predict_score_lifecycle_keeps_truth_out_of_predictor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _make_fixture(tmp_path)
    prepared = runner.run_prepare(_args(fixture, tmp_path / "prepared"))
    capsule_for_authority = runner._load_received_capsule(fixture.received, fixture.received_sha256)
    assert any(item.startswith("phase1|") for item in capsule_for_authority.physical_ids)
    package_value = json.loads(Path(prepared["package"]).read_text(encoding="utf-8"))
    assert package_value["truth_free"] is True
    assert package_value["truth_loaded"] is False
    assert not _contains_key(package_value, "truth")
    assert not _contains_key(package_value, "truth_by_query_id")
    assert not _contains_key(package_value, "phase1_fit_ids")
    for forbidden in ("query_ids_by_class", "query_observation_ids_by_class", "query_count_by_class"):
        assert not _contains_key(package_value, forbidden)
    assert all(
        set(row) == {"row_id", "held_receiver", "held_class", "active_k", "physical_binding_receipt"}
        for row in package_value["rows"]
    )

    capsule = runner._load_received_capsule(fixture.received, fixture.received_sha256)
    bridge = _Bridge(capsule, fixture.feature_by_physical, fixture.checkpoint_sha256)
    monkeypatch.setattr(runner, "_load_checkpoint_bridge", lambda args, received, checkpoint_sha256: bridge)
    # The full 24-row scorer contract is exercised with a deterministic
    # mechanical row provider.  The direct runtime has a separate focused
    # functional smoke below, keeping this CLI test bounded on CPU.
    from test_stage2_next_r4_artifact import _row_results

    _, expected_rows, _ = _row_results()
    rows_by_id = {row["row_id"]: row for row in expected_rows}

    def sealed_row(*, row: object, **kwargs: object) -> dict:
        planned = next(item for item in runner.matrix.build_next_r4_proxy24_plan(CLASSES)["rows"] if item["row_id"] == row.row_id)
        package_row = next(item for item in package_value["rows"] if item["row_id"] == row.row_id)
        value = deepcopy(rows_by_id[row.row_id])
        value["row_id"] = planned["row_id"]
        value["held_receiver"] = planned["held_receiver"]
        value["held_class"] = planned["held_class"]
        value["active_k"] = planned["active_k"]
        value["binding_receipt"] = deepcopy(package_row["physical_binding_receipt"])
        binding_value = value["binding_receipt"]
        for registration in value["registrations"].values():
            for state in registration["states"].values():
                state["query_physical_ids"] = list(binding_value["query_physical_ids"])
                state["query_observation_ids"] = list(binding_value["query_observation_ids"])
        return value

    monkeypatch.setattr(
        runner.runtime,
        "execute_next_r4_logical_row",
        sealed_row,
    )
    predict_args = runner._parser().parse_args(
        [
            "predict",
            "--run-id",
            "test-run",
            "--run-root",
            str(tmp_path / "run"),
            "--received-iq",
            str(fixture.received),
            "--received-iq-sha256",
            fixture.received_sha256,
            "--package",
            prepared["package"],
            "--package-sha256",
            _sha(Path(prepared["package"]).read_bytes()),
            "--fa-asset-manifest",
            str(fixture.asset_manifest),
            "--fa-asset-manifest-sha256",
            fixture.asset_manifest_sha256,
            "--checkpoint",
            str(fixture.checkpoint),
            "--checkpoint-sha256",
            fixture.checkpoint_sha256,
            "--prepare-receipt",
            prepared["receipt"],
            "--prepare-receipt-sha256",
            _sha(Path(prepared["receipt"]).read_bytes()),
            "--device",
            "cpu",
        ]
    )
    completion = runner.run_predict(predict_args)
    assert completion["status"] == "ARTIFACTS_COMPLETE_NOT_SCORED"
    prediction = json.loads((tmp_path / "run" / "prediction.json").read_text(encoding="utf-8"))
    assert prediction["rows_complete"] is True
    assert prediction["truth_loaded"] is False
    assert not _contains_key(prediction, "truth")

    truth_path = Path(prepared["truth"])
    score_args = runner._parser().parse_args(
        [
            "score",
            "--run-root",
            str(tmp_path / "run"),
            "--truth",
            str(truth_path),
            "--truth-sha256",
            _sha(truth_path.read_bytes()),
            "--prepare-receipt",
            prepared["receipt"],
            "--prepare-receipt-sha256",
            _sha(Path(prepared["receipt"]).read_bytes()),
            "--output",
            str(tmp_path / "score.json"),
        ]
    )
    scored = runner.run_score(score_args)
    assert scored["truth_opened_after_complete_prediction"] is True
    assert json.loads((tmp_path / "score.json").read_text(encoding="utf-8"))["schema"].endswith("proxy_score.v1")


def test_prepare_rejects_unknown_or_support_query_overlapping_phase1_ids(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    metadata_value = json.loads(fixture.metadata.read_text(encoding="utf-8"))

    unknown_value = deepcopy(metadata_value)
    unknown_value["rows"][0]["phase1_fit_ids"][0] = "phase1-not-in-received"
    unknown_path = tmp_path / "metadata_unknown_phase1.json"
    unknown_path.write_text(json.dumps(unknown_value, sort_keys=True) + "\n", encoding="utf-8")
    unknown_args = _args(fixture, tmp_path / "prepare_unknown_phase1")
    unknown_args.capsule_metadata = unknown_path
    unknown_args.capsule_metadata_sha256 = _sha(unknown_path.read_bytes())
    with pytest.raises(runner.MissingRealInputArtifacts, match="must belong to received physical IDs"):
        runner.run_prepare(unknown_args)

    overlap_value = deepcopy(metadata_value)
    overlap_value["rows"][0]["phase1_fit_ids"][0] = overlap_value["rows"][0]["k5_support_ids_by_class"][CLASSES[0]][0]
    overlap_path = tmp_path / "metadata_overlap_phase1.json"
    overlap_path.write_text(json.dumps(overlap_value, sort_keys=True) + "\n", encoding="utf-8")
    overlap_args = _args(fixture, tmp_path / "prepare_overlap_phase1")
    overlap_args.capsule_metadata = overlap_path
    overlap_args.capsule_metadata_sha256 = _sha(overlap_path.read_bytes())
    with pytest.raises(runner.MissingRealInputArtifacts, match="overlap K5 support/query"):
        runner.run_prepare(overlap_args)


def test_real_runtime_single_row_functional_smoke() -> None:
    # Keep one direct call to the frozen runtime in the CLI test module.  The
    # complete 24-row lifecycle above intentionally uses a fast sealed-row
    # provider because the direct qKNN implementation is CPU-expensive.  This
    # is a functional runtime smoke, not a real-checkpoint claim.
    from test_stage2_next_r4_runtime import _run

    result, _ = _run(1)
    assert result["query_isolation_receipt"]["query_truth_access"] is False
    assert result["registrations"]["REG0"]["states"]["DA0_REG0"]["state_id"] == "DA0_REG0"


def test_foreign_runtime_binding_is_rejected_before_artifact_projection() -> None:
    from test_stage2_next_r4_artifact import _row_results

    plan, rows, _ = _row_results()
    result = deepcopy(rows[0])
    foreign = deepcopy(result["binding_receipt"])
    foreign["query_physical_ids"][0] = "foreign-query-id"
    unsigned = dict(foreign)
    unsigned.pop("binding_sha256", None)
    foreign["binding_sha256"] = matrix.canonical_sha256(unsigned)
    result["binding_receipt"] = foreign
    with pytest.raises(runner.NextR4Proxy24Error, match="binding"):
        runner._validate_runtime_result(
            result=result,
            planned=plan["rows"][0],
            binding=rows[0]["binding_receipt"],
        )


def test_predictor_package_rejects_class_grouped_query_metadata(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    prepared = runner.run_prepare(_args(fixture, tmp_path / "prepared"))
    package = json.loads(Path(prepared["package"]).read_text(encoding="utf-8"))
    package["rows"][0]["physical_binding_receipt"]["query_ids_by_class"] = {"leak": ["q"]}
    bad_package = tmp_path / "bad_package.json"
    bad_package.write_text(json.dumps(package, sort_keys=True) + "\n", encoding="utf-8")
    received = runner._load_received_capsule(fixture.received, fixture.received_sha256)
    with pytest.raises(runner.NextR4Proxy24Error, match="class-grouped"):
        runner._load_package(
            bad_package,
            _sha(bad_package.read_bytes()),
            received=received,
            checkpoint_sha256=fixture.checkpoint_sha256,
            asset_manifest_sha256=fixture.asset_manifest_sha256,
        )


def test_output_root_overwrite_refused_and_incomplete_score_does_not_open_truth(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(runner.NextR4Proxy24Error, match="overwrite|new absolute"):
        runner._new_root(existing)

    root = tmp_path / "incomplete"
    root.mkdir()
    (root / "completion.json").write_text(json.dumps({"status": "RUNNING"}), encoding="utf-8")
    truth = tmp_path / "must-not-be-read.json"
    truth.write_text("not opened", encoding="utf-8")
    with pytest.raises(runner.NextR4Proxy24Error, match="complete NEXT-R4 prediction artifacts"):
        runner.run_score(
            runner._parser().parse_args(
                [
                    "score",
                    "--run-root",
                    str(root),
                    "--truth",
                    str(truth),
                    "--truth-sha256",
                    _sha(truth.read_bytes()),
                    "--prepare-receipt",
                    str(tmp_path / "missing-prepare-receipt.json"),
                    "--prepare-receipt-sha256",
                    "0" * 64,
                    "--output",
                    str(tmp_path / "score.json"),
                ]
            )
        )
    assert not (tmp_path / "score.json").exists()
