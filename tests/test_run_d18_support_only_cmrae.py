from __future__ import annotations

import hashlib
import importlib.util
import json
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

import cvsrffi.somph_predictor_bundle as somph_bundle
import cvsrffi.somph_predictor_runtime as somph_runtime
from cvsrffi.somph_predictor_bundle import (
    ENROLLMENT_ONLY,
    SOMPH_SUPPORT_IQ_SCHEMA,
    sha256_file,
    write_somph_predictor_bundle,
)
from cvsrffi.somph_predictor_runtime import expected_somph_method_lock
from cvsrffi.stage2_cmrae import (
    _build_runtime_authorized_received_iq_artifact_internal,
    _seal_runtime_authorized_backbone_internal,
    fit_before_after_locked,
    preregistered_candidates,
)


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "code"
    / "scripts"
    / "run_d18_support_only_cmrae.py"
)
SPEC = importlib.util.spec_from_file_location("run_d18_support_only_cmrae", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def _payload(prefix: str = "a"):
    labels = np.repeat(np.arange(2, dtype=np.int64), 10)
    ranks = np.tile(np.arange(10, dtype=np.int64), 2)
    iq = np.stack(
        [
            np.stack(
                [
                    np.linspace(0.1 + index, 1.0 + index, 16),
                    np.linspace(0.2 + index, 0.8 + index, 16),
                ]
            ).astype(np.float32)
            for index in range(20)
        ]
    )
    hashes = np.asarray(
        [
            hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest()
            for row in iq
        ]
    )
    tokens = np.asarray([f"sid_{prefix}_{index:02d}" for index in range(20)])
    overlays = np.asarray(
        [f"oid_{hashlib.sha256(f'{prefix}_{index}'.encode()).hexdigest()}" for index in range(20)]
    )
    payload = {
        "support_leo_weak_iq": iq,
        "support_class_indices": labels,
        "support_rank_within_class": ranks,
        "support_tokens": tokens,
        "support_post_channel_iq_sha256": hashes,
        "support_overlay_tokens": overlays,
        "support_satellite_seeds": np.full(20, 71310200, dtype=np.int64),
    }
    manifest = {
        "registered_classes": [
            {"class_handle": "old0"},
            {"class_handle": "old1"},
        ]
    }
    provenance = {
        (str(tokens[i]), str(hashes[i]), "leo_clear_weak"): {
            "sample_token": str(tokens[i]),
            "post_channel_iq_sha256": str(hashes[i]),
            "scenario": "leo_clear_weak",
            "overlay_token": str(overlays[i]),
            "satellite_seed": 71310200,
            "source_leo_cache_sha256": hashlib.sha256(
                b"sealed_cache_clear"
            ).hexdigest(),
            "source_leo_provenance_sha256": hashlib.sha256(
                b"sealed_leo_provenance"
            ).hexdigest(),
        }
        for i in range(20)
    }
    return payload, manifest, provenance


def test_rows_bind_real_npz_overlay_token_and_seed() -> None:
    payload, manifest, provenance = _payload()
    rows = runner._rows_with_overlay(
        payload, manifest, provenance, scenario="leo_clear_weak"
    )
    assert len(rows["iq"]) == 20
    assert set(rows["satellite_seeds"].tolist()) == {71310200}
    assert all(len(value) == 64 for value in rows["source_leo_provenance_sha256"])
    assert all(len(value) == 64 for value in rows["source_leo_cache_sha256"])
    assert all(len(value) == 64 for value in rows["overlay_provenance_record_sha256"])
    assert rows["overlay_tokens"][0].startswith("oid_")
    assert rows["source_leo_provenance_sha256"][0] != rows["overlay_tokens"][0][4:]
    assert rows["overlay_provenance_record_sha256"][0] != rows["overlay_tokens"][0][4:]
    broken = dict(payload)
    broken["support_overlay_tokens"] = payload["support_overlay_tokens"].copy()
    broken["support_overlay_tokens"][0] = "oid_" + "f" * 64
    with pytest.raises(runner.D18RunnerError, match="binding drift"):
        runner._rows_with_overlay(
            broken, manifest, provenance, scenario="leo_clear_weak"
        )


def test_overlay_member_is_same_fd_hash_checked_before_parse(tmp_path: Path) -> None:
    payload = {
        "schema": "cvs.phase2.somph_overlay_provenance.v1",
        "receiver": "rx",
        "seed": 7,
        "samples": [],
    }
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    member_path = tmp_path / "overlay_provenance.json"
    member_path.write_bytes(raw)
    manifest = {
        "members": [
            {
                "kind": "overlay_provenance",
                "relative_path": member_path.name,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "size_bytes": len(raw),
            }
        ]
    }
    loaded, audit = runner._safe_verified_json_member(
        tmp_path, manifest, kind="overlay_provenance"
    )
    assert loaded == payload
    assert audit["same_file_descriptor_hash_and_parse"] is True
    member_path.write_bytes(raw + b" ")
    with pytest.raises(runner.D18RunnerError, match="hash/size drift"):
        runner._safe_verified_json_member(
            tmp_path, manifest, kind="overlay_provenance"
        )


def test_authority_gate_precedes_any_iq_materialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = {"preflight": 0, "materialize": 0}

    def fake_preflight(*args, **kwargs):
        calls["preflight"] += 1
        return (
            {"package_root_sha256": "a" * 64},
            {},
            {
                "iq_payload_materialized": False,
                "formal_launch_authority": False,
                "formal_metric_claim_allowed": False,
                "control_state": "LOCAL_PROTOCOL_REPAIR_REQUIRED",
            },
        )

    def forbidden_materialize(*args, **kwargs):
        calls["materialize"] += 1
        raise AssertionError("IQ materialization was reached before authority")

    monkeypatch.setattr(
        runner, "preflight_somph_predictor_bundle_with_authority", fake_preflight
    )
    monkeypatch.setattr(
        runner,
        "materialize_somph_enrollment_with_authority",
        forbidden_materialize,
    )
    with pytest.raises(runner.D18RunnerError, match="before IQ materialization"):
        runner.run(
            before_root=tmp_path / "before" / "enrollment_only",
            before_seal=tmp_path / "before.seal.json",
            expected_before_seal_sha256="1" * 64,
            before_authority_bundle_root=tmp_path / "before_authority",
            expected_before_authority_commit_sha256="3" * 64,
            before_formal_policy=tmp_path / "before_policy.json",
            before_formal_policy_authorization=tmp_path / "before_auth.json",
            before_signed_policy_authorization_envelope=(
                tmp_path / "before_envelope.json"
            ),
            expected_before_signed_policy_authorization_envelope_sha256="4" * 64,
            after_root=tmp_path / "after" / "enrollment_only",
            after_seal=tmp_path / "after.seal.json",
            expected_after_seal_sha256="2" * 64,
            after_authority_bundle_root=tmp_path / "after_authority",
            expected_after_authority_commit_sha256="6" * 64,
            after_formal_policy=tmp_path / "after_policy.json",
            after_formal_policy_authorization=tmp_path / "after_auth.json",
            after_signed_policy_authorization_envelope=(
                tmp_path / "after_envelope.json"
            ),
            expected_after_signed_policy_authorization_envelope_sha256="7" * 64,
            output=tmp_path / "output",
        )
    assert calls == {"preflight": 2, "materialize": 0}

    ready = {
        "iq_payload_materialized": False,
        "iq_archive_opened": False,
        "np_load_invoked": False,
        "iq_open_authorized": True,
        "formal_launch_authority": False,
        "formal_metric_claim_allowed": False,
        "external_authority_lock_verified": True,
        "status": "AUTHORITY_PREFLIGHT_PASS_IQ_OPEN_AUTHORIZED",
        "control_state": "AUTHORITY_PREFLIGHT_PASS_IQ_OPEN_AUTHORIZED",
        "phase2_protocol_evidence_status": (
            "AUTHORITY_PREFLIGHT_PASS_IQ_OPEN_AUTHORIZED"
        ),
    }
    runner._require_pre_iq_open_authority(ready, ready)


def test_cross_scene_disjointness_covers_physical_parent_and_overlay() -> None:
    rows = {}
    for scene_index, scene in enumerate(runner.FORMAL_LEO_WEAK_SCENARIOS):
        values = {
            "tokens": np.asarray([f"p{scene_index}_{i}" for i in range(3)]),
            "hashes": np.asarray(
                [hashlib.sha256(f"h{scene_index}_{i}".encode()).hexdigest() for i in range(3)]
            ),
            "overlay_tokens": np.asarray(
                [f"oid_{hashlib.sha256(f'o{scene_index}_{i}'.encode()).hexdigest()}" for i in range(3)]
            ),
        }
        rows[scene] = values
    assert runner._cross_scene_disjointness(rows)["all_pairwise_disjoint"] is True
    bad = {key: dict(value) for key, value in rows.items()}
    bad[runner.FORMAL_LEO_WEAK_SCENARIOS[1]]["overlay_tokens"] = bad[
        runner.FORMAL_LEO_WEAK_SCENARIOS[1]
    ]["overlay_tokens"].copy()
    bad[runner.FORMAL_LEO_WEAK_SCENARIOS[1]]["overlay_tokens"][0] = rows[
        runner.FORMAL_LEO_WEAK_SCENARIOS[0]
    ]["overlay_tokens"][0]
    with pytest.raises(runner.D18RunnerError, match="cross-scene"):
        runner._cross_scene_disjointness(bad)


@pytest.mark.parametrize(
    ("force_zero", "expected_used"), ((True, False), (False, True))
)
def test_inventory_records_and_validates_every_fixed_received_iq_view(
    force_zero: bool, expected_used: bool
) -> None:
    rows = {
        "labels": np.asarray(["old0", "new0"]),
        "ranks": np.asarray([0, 0], dtype=np.int64),
        "tokens": np.asarray(["pid_old", "pid_new"]),
        "hashes": np.asarray(["a" * 64, "b" * 64]),
        "overlay_tokens": np.asarray(["oid_" + "c" * 64, "oid_" + "d" * 64]),
        "satellite_seeds": np.asarray([71310200, 71310200], dtype=np.int64),
        "source_leo_cache_sha256": np.asarray(["e" * 64, "e" * 64]),
        "source_leo_provenance_sha256": np.asarray(["f" * 64, "f" * 64]),
        "overlay_provenance_record_sha256": np.asarray(["1" * 64, "2" * 64]),
    }
    state = SimpleNamespace(
        operator_id="cmrae_dct8_fixed_received_iq",
        hyperparameters=SimpleNamespace(force_zero=force_zero),
    )
    inventory = runner._support_inventory(
        rows,
        old_classes={"old0"},
        scenario="leo_clear_weak",
        state=state,
    )
    assert len(inventory) == len({row["physical_sample_id"] for row in inventory})
    for row in inventory:
        assert row["parent_received_iq_sha256"] in {"a" * 64, "b" * 64}
        assert row["operator_id"] == "cmrae_dct8_fixed_received_iq"
        assert row["source_leo_cache_sha256"] == "e" * 64
        assert row["source_leo_provenance_sha256"] == "f" * 64
        assert row["overlay_provenance_record_sha256"] in {"1" * 64, "2" * 64}
        assert row["view_seed"] == 0
        assert row["view_seed_policy"] == "deterministic_fixed_zero_no_random_view"
        assert row["post_reception_view_used"] is expected_used
        assert row["post_reception_view_count"] == 1
        assert row["post_reception_view_counts_as_additional_physical_sample"] is False
        assert row["additional_leo_channel_state_generation"] is False
        assert row["counts_as_one_physical_support"] is True
    broken = [dict(row) for row in inventory]
    broken[0]["post_reception_view_count"] = 2
    with pytest.raises(runner.D18RunnerError, match="inventory"):
        runner._validate_support_inventory(
            broken, state=state, scenario="leo_clear_weak"
        )


def _artifact(classes: tuple[str, ...], *, old=None):
    iq_rows = []
    labels = []
    ranks = []
    tokens = []
    hashes = []
    overlays = []
    for class_index, class_name in enumerate(classes):
        for rank in range(10):
            if old is not None and class_name in old:
                row, token, parent, overlay = old[(class_name, rank)]
            else:
                axis = np.linspace(0, 2 * np.pi, 32, endpoint=False)
                complex_row = (1 + 0.1 * class_index) * np.exp(
                    1j * (axis * (class_index + 1) + rank * 0.01)
                )
                row = np.stack([complex_row.real, complex_row.imag]).astype(np.float32)
                token = f"pid_{class_name}_{rank}"
                parent = hashlib.sha256(row.tobytes()).hexdigest()
                overlay = hashlib.sha256(f"overlay_{class_name}_{rank}".encode()).hexdigest()
            iq_rows.append(row)
            labels.append(class_name)
            ranks.append(rank)
            tokens.append(token)
            hashes.append(parent)
            overlays.append(overlay)
    artifact = _build_runtime_authorized_received_iq_artifact_internal(
        np.stack(iq_rows),
        physical_sample_ids=tokens,
        parent_received_iq_sha256=hashes,
        overlay_tokens=[f"overlay-token::{value}" for value in tokens],
        source_leo_provenance_sha256=["d" * 64] * len(iq_rows),
        source_leo_cache_sha256=["e" * 64] * len(iq_rows),
        target_channel_views=["leo_clear_weak"] * len(iq_rows),
        satellite_seeds=[1] * len(iq_rows),
        overlay_provenance_sha256=overlays,
        sealed_runtime_sha256="a" * 64,
        sealed_phase1_checkpoint_sha256="b" * 64,
        feature_code_sha256="c" * 64,
        purpose="support",
    )
    old_map = {
        (labels[i], ranks[i]): (
            np.array(iq_rows[i], copy=True),
            tokens[i],
            hashes[i],
            overlays[i],
        )
        for i in range(len(labels))
    }
    return artifact, np.asarray(labels), np.asarray(ranks), old_map


def test_state_payload_roundtrip_uses_actual_bytes_and_external_commit(tmp_path: Path) -> None:
    before, before_labels, before_ranks, old = _artifact(("old0", "old1"))
    after, after_labels, after_ranks, _ = _artifact(
        ("old0", "old1", "new0"), old=old
    )

    def extract(iq: np.ndarray) -> np.ndarray:
        complex_iq = iq[:, 0] + 1j * iq[:, 1]
        return np.stack(
            [
                np.mean(complex_iq.real, axis=1),
                np.mean(complex_iq.imag, axis=1),
                np.std(complex_iq.real, axis=1),
                np.std(complex_iq.imag, axis=1),
            ],
            axis=1,
        ).astype(np.float32)

    backbone = _seal_runtime_authorized_backbone_internal(
        extract,
        feature_code_sha256="c" * 64,
        sealed_phase1_checkpoint_sha256="b" * 64,
    )
    fitted = fit_before_after_locked(
        before,
        before_labels,
        before_ranks,
        after,
        after_labels,
        after_ranks,
        k_shot=10,
        hyperparameters=preregistered_candidates()[0],
        backbone=backbone,
    )
    audit = runner._write_state_roundtrip(tmp_path / "state", fitted.after_state)
    assert audit["semantic_roundtrip_verified"] is True
    assert audit["actual_full_state_under_256kib"] is True
    assert audit["adapter_state_under_16kib"] is True
    assert audit["actual_full_serialized_state_bytes"] == (
        tmp_path / "state" / "state.npz"
    ).stat().st_size
    assert {value.name for value in (tmp_path / "state").iterdir()} == {
        "state.npz",
        "COMMIT",
    }
    with pytest.raises(runner.D18RunnerError, match="already exists"):
        runner._write_state_roundtrip(tmp_path / "state", fitted.after_state)


def test_cli_and_receipt_are_support_only_and_keep_d8b_authority() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "from run_d14_support_only_pairwise_fisher_guard import" in source
    assert "materialize_somph_enrollment_with_authority" in source
    assert "finalize_somph_enrollment_authority_after_materialization" in source
    assert "_load_enrollment" not in source
    assert "_materialization_receipt" not in source
    assert "--before-formal-policy" in source
    assert "--before-signed-policy-authorization-envelope" in source
    assert "--after-formal-policy" in source
    assert "--after-signed-policy-authorization-envelope" in source
    assert "_payload_rows" in source
    assert "support_overlay_tokens" in source
    assert "support_satellite_seeds" in source
    assert "select_k10_candidate_three_scene" in source
    assert "k10_lock_certificate" in source
    assert "--query" not in source
    assert "--truth" not in source
    assert "--scorer" not in source
    assert '"formal_launch_authority": False' in source
    assert '"authority": "development_diagnostic_only"' in source
    assert "actual_full_serialized_state_bytes" in source
    assert "peak_cuda_allocated_bytes" in source
    assert "identity_single_qknn" in source


def test_candidate_surface_is_exact_and_three_scene_atomic() -> None:
    candidates = preregistered_candidates()
    assert [(value.candidate_id, value.lambda_equalizer) for value in candidates] == [
        ("D18_Z0", 0.0),
        ("D18_CMRAE_L0125", 0.125),
        ("D18_CMRAE_L0250", 0.25),
    ]
    assert tuple(runner.FORMAL_LEO_WEAK_SCENARIOS) == (
        "leo_clear_weak",
        "leo_low_elev_weak",
        "leo_rain_weak",
    )


def test_external_k10_authority_anchor_binds_packages_code_and_full_selection() -> None:
    rows_by_scenario = {}
    for scene_index, scenario in enumerate(runner.FORMAL_LEO_WEAK_SCENARIOS):
        state_rows = {}
        for state_index, state in enumerate(("before", "after")):
            count = 2 + state_index
            state_rows[state] = {
                "labels": np.asarray([f"cls_{i}" for i in range(count)]),
                "ranks": np.zeros(count, dtype=np.int64),
                "tokens": np.asarray(
                    [f"pid_{scene_index}_{state_index}_{i}" for i in range(count)]
                ),
                "hashes": np.asarray(
                    [hashlib.sha256(f"iq_{scene_index}_{state_index}_{i}".encode()).hexdigest() for i in range(count)]
                ),
                "overlay_tokens": np.asarray(
                    [f"oid_{hashlib.sha256(f'overlay_{scene_index}_{state_index}_{i}'.encode()).hexdigest()}" for i in range(count)]
                ),
                "satellite_seeds": np.full(count, 71310200 + scene_index, dtype=np.int64),
                "source_leo_cache_sha256": np.asarray(
                    [hashlib.sha256(f"cache_{scene_index}".encode()).hexdigest()] * count
                ),
                "source_leo_provenance_sha256": np.asarray(["a" * 64] * count),
                "overlay_provenance_record_sha256": np.asarray(
                    [hashlib.sha256(f"record_{scene_index}_{state_index}_{i}".encode()).hexdigest() for i in range(count)]
                ),
            }
        rows_by_scenario[scenario] = state_rows
    before_manifest = {
        "package_root_sha256": "b" * 64,
        "feature_runtime_sha256": "c" * 64,
        "phase1_checkpoint_sha256": "d" * 64,
    }
    after_manifest = {
        "package_root_sha256": "e" * 64,
        "feature_runtime_sha256": "c" * 64,
        "phase1_checkpoint_sha256": "d" * 64,
    }
    before_authority = {
        "manifest_sha256": "f" * 64,
        "formal_launch_authority": True,
        "formal_metric_claim_allowed": True,
        "status": "CURRENT_PROTOCOL_REAL_INPUT_AUDIT_PASS",
    }
    after_authority = {
        "manifest_sha256": "1" * 64,
        "formal_launch_authority": True,
        "formal_metric_claim_allowed": True,
        "status": "CURRENT_PROTOCOL_REAL_INPUT_AUDIT_PASS",
    }
    kwargs = dict(
        before_manifest=before_manifest,
        after_manifest=after_manifest,
        before_authority_audit=before_authority,
        after_authority_audit=after_authority,
        before_seal_sha256="2" * 64,
        after_seal_sha256="3" * 64,
        code_hashes={"runner": "4" * 64, "core": "5" * 64},
        rows_by_scenario=rows_by_scenario,
    )
    first = runner._build_k10_selection_authority_anchor(**kwargs)
    assert len(first["k10_selection_authority_anchor_sha256"]) == 64
    record = first["payload"]["selection_records"]["leo_clear_weak"]["before"][0]
    assert record["overlay_token"].startswith("oid_")
    assert record["source_leo_provenance_sha256"] == "a" * 64
    assert record["overlay_provenance_record_sha256"] != record["overlay_token"][4:]
    rows_by_scenario["leo_clear_weak"]["before"]["satellite_seeds"] = np.asarray(
        [999, 71310200], dtype=np.int64
    )
    second = runner._build_k10_selection_authority_anchor(**kwargs)
    assert (
        first["k10_selection_authority_anchor_sha256"]
        != second["k10_selection_authority_anchor_sha256"]
    )


def _e2e_enrollment(
    root: Path,
    *,
    state: str,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, str, dict[str, object]]:
    class_count = 6 if state == "before" else 11
    stage = "stage2b" if state == "before" else "stage2c"
    checkpoint_sha256 = "e" * 64
    monkeypatch.setattr(
        somph_bundle, "ADV3B02_PHASE1_CHECKPOINT_SHA256", checkpoint_sha256
    )
    monkeypatch.setattr(
        somph_runtime, "ADV3B02_PHASE1_CHECKPOINT_SHA256", checkpoint_sha256
    )
    root.mkdir(parents=True)
    (root / somph_bundle.FEATURE_RUNTIME_RELATIVE_PATH).write_bytes(
        b"synthetic-authority-tested-d18-torchscript-runtime"
    )
    method_lock = expected_somph_method_lock()
    method_lock["checkpoint_sha256"] = checkpoint_sha256
    (root / "method_lock.json").write_bytes(
        somph_bundle.canonical_json_bytes(method_lock)
    )
    provenance_rows: list[dict[str, object]] = []
    for scene_index, scenario in enumerate(runner.FORMAL_LEO_WEAK_SCENARIOS):
        iq_rows = []
        labels = []
        ranks = []
        tokens = []
        hashes = []
        overlays = []
        seeds = []
        for class_index in range(class_count):
            for rank in range(10):
                axis = np.arange(32, dtype=np.float64)
                phase = 0.013 * rank + 0.071 * scene_index
                frequency = class_index + 1.0 + 0.05 * scene_index
                amplitude = 1.0 + 0.16 * class_index + 0.002 * rank
                complex_iq = amplitude * np.exp(
                    1j * (2.0 * np.pi * frequency * axis / 32.0 + phase)
                )
                complex_iq += (rank + 1) * 1e-5 * (axis + 1)
                iq = np.stack([complex_iq.real, complex_iq.imag]).astype(np.float32)
                token = "sid_" + f"{scene_index * 10000 + class_index * 10 + rank + 1:064x}"
                parent = hashlib.sha256(np.ascontiguousarray(iq).tobytes()).hexdigest()
                overlay = "oid_" + hashlib.sha256(
                    f"overlay_scene{scene_index}_class{class_index}_rank{rank}".encode()
                ).hexdigest()
                satellite_seed = 71310200 + scene_index
                row = {
                    "sample_token": token,
                    "post_channel_iq_sha256": parent,
                    "scenario": scenario,
                    "overlay_token": overlay,
                    "satellite_seed": satellite_seed,
                    "source_leo_cache_sha256": "a" * 64,
                    # A common authority provenance is legal across scenes; the
                    # per-row canonical overlay record remains scene-specific.
                    "source_leo_provenance_sha256": "b" * 64,
                }
                iq_rows.append(iq)
                labels.append(class_index)
                ranks.append(rank)
                tokens.append(token)
                hashes.append(parent)
                overlays.append(overlay)
                seeds.append(satellite_seed)
                provenance_rows.append(row)
        embedded = {
            "schema": SOMPH_SUPPORT_IQ_SCHEMA,
            "scenario": scenario,
            "registration_state": state,
            "registered_class_count": class_count,
            "support_pool_max_k": 10,
            "token_scheme": "hmac_sha256_opaque_v1",
        }
        with (root / f"support_{scenario}.npz").open("xb") as handle:
            np.savez(
                handle,
                support_leo_weak_iq=np.stack(iq_rows),
                support_class_indices=np.asarray(labels, dtype=np.int64),
                support_rank_within_class=np.asarray(ranks, dtype=np.int64),
                support_tokens=np.asarray(tokens),
                support_overlay_tokens=np.asarray(overlays),
                support_satellite_seeds=np.asarray(seeds, dtype=np.int64),
                support_post_channel_iq_sha256=np.asarray(hashes),
                manifest_json=np.asarray(json.dumps(embedded, sort_keys=True)),
            )
    overlay_value = {
        "schema": "cvs.phase2.somph_overlay_provenance.v1",
        "profile": ENROLLMENT_ONLY,
        "receiver": "20-1",
        "seed": 713101,
        "samples": provenance_rows,
    }
    overlay_raw = json.dumps(
        overlay_value, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    overlay_path = root / "overlay_provenance.json"
    overlay_path.write_bytes(overlay_raw)
    seal_path = root.parent / f"{state}_enrollment.seal.json"
    _manifest_path, _seal_path, manifest, _seal = write_somph_predictor_bundle(
        root,
        profile=ENROLLMENT_ONLY,
        stage=stage,
        registration_state=state,
        receiver="20-1",
        seed=713101,
        k_shot=10,
        registered_classes=[
            {
                "class_index": index,
                "class_handle": "cls_" + f"{index + 1:064x}",
            }
            for index in range(class_count)
        ],
        expected_method_lock_sha256=sha256_file(root / "method_lock.json"),
        expected_overlay_provenance_sha256=hashlib.sha256(overlay_raw).hexdigest(),
        detached_seal_path=seal_path,
        support_pool_max_k=10,
    )
    return root, seal_path, sha256_file(seal_path), manifest


def _formal_authority_payloads() -> tuple[dict, dict, dict]:
    old_tx_ids = list(somph_bundle.OLD_TX_IDS)
    new_tx_ids = list(somph_bundle.NEW_TX_IDS[:5])
    lock = {
        "receiver": "20-1",
        "seed": 713101,
        "cache_scope": "stage2_registered",
        "old_tx_ids": old_tx_ids,
        "new_tx_ids": new_tx_ids,
        "cache_sha256_by_scenario": {
            scenario: "a" * 64 for scenario in runner.FORMAL_LEO_WEAK_SCENARIOS
        },
        "physical_sample_scenario_assignment_policy": (
            somph_bundle.lineage_authority.PHYSICAL_SAMPLE_SCENARIO_ASSIGNMENT_POLICY
        ),
        "physical_sample_ids_sha256_by_scenario": {
            scenario: "1" * 64 for scenario in runner.FORMAL_LEO_WEAK_SCENARIOS
        },
        "physical_sample_scenario_assignment_sha256": "4" * 64,
        "cross_scenario_physical_disjointness_audit": "PASS",
        "single_observation_contract_audit": "PASS",
        "post_channel_iq_sha256_root_by_scenario": {
            scenario: "2" * 64 for scenario in runner.FORMAL_LEO_WEAK_SCENARIOS
        },
        "overlay_ids_sha256_by_scenario": {
            scenario: "3" * 64 for scenario in runner.FORMAL_LEO_WEAK_SCENARIOS
        },
        "cache_role_inputs_root_sha256": "5" * 64,
        "datasets": [
            {
                "role": "target_old",
                "path": "E:/sealed/offline/ManySig.pkl",
                "sha256": "6" * 64,
                "size_bytes": 1,
                "tx_ids": old_tx_ids,
            },
            {
                "role": "target_new",
                "path": "E:/sealed/offline/ManyTx.pkl",
                "sha256": "7" * 64,
                "size_bytes": 1,
                "tx_ids": new_tx_ids,
            },
        ],
        **somph_bundle.lineage_authority.PHASE2_SINGLE_OBSERVATION_CONTRACT,
    }
    attestation = {
        "structural_receipt_sha256": "b" * 64,
        "dataset_authority_root_sha256": "6" * 64,
    }
    commit = {
        "authority_lock_sha256": "7" * 64,
        "members": [
            {
                "name": somph_bundle.lineage_authority.AUTHORITY_ATTESTATION_NAME,
                "sha256": "9" * 64,
            }
        ],
    }
    return lock, attestation, commit


def _formal_policy_authorization(
    *,
    manifest: dict,
    seal_path: Path,
    seal_sha256: str,
    commit_sha256: str,
    lock: dict,
    attestation: dict,
    commit: dict,
    policy_sha256: str,
    code_closure_sha256: str,
    package_control_roots: dict,
) -> dict:
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    return {
        "schema": somph_bundle.SOMPH_FORMAL_POLICY_AUTHORIZATION_SCHEMA,
        "status": somph_bundle.SOMPH_FORMAL_POLICY_AUTHORIZATION_STATUS,
        "formal_launch_authority": True,
        "formal_metric_claim_allowed": True,
        "package_root_sha256": manifest["package_root_sha256"],
        "package_detached_seal_sha256": seal_sha256,
        "artifact_member_allowlist_sha256": seal["artifact_member_allowlist_sha256"],
        "manifest_sha256": seal["manifest_sha256"],
        "overlay_provenance_sha256": manifest["overlay_provenance_sha256"],
        "authority_commit_sha256": commit_sha256,
        "authority_lock_sha256": commit["authority_lock_sha256"],
        "authority_attestation_sha256": "9" * 64,
        "receiver": manifest["receiver"],
        "seed": manifest["seed"],
        "stage": manifest["stage"],
        "registration_state": manifest["registration_state"],
        "k_shot": manifest["k_shot"],
        "cache_scope": "stage2_registered",
        "old_tx_ids": lock["old_tx_ids"],
        "new_tx_ids": lock["new_tx_ids"],
        "dataset_authority_root_sha256": attestation["dataset_authority_root_sha256"],
        "cache_role_inputs_root_sha256": lock["cache_role_inputs_root_sha256"],
        "physical_sample_ids_sha256_by_scenario": lock[
            "physical_sample_ids_sha256_by_scenario"
        ],
        "physical_sample_scenario_assignment_sha256": lock[
            "physical_sample_scenario_assignment_sha256"
        ],
        "post_channel_iq_sha256_root_by_scenario": lock[
            "post_channel_iq_sha256_root_by_scenario"
        ],
        "overlay_ids_sha256_by_scenario": lock["overlay_ids_sha256_by_scenario"],
        "preflight_code_sha256": sha256_file(Path(somph_bundle.__file__)),
        "formal_policy_sha256": policy_sha256,
        "code_closure_sha256": code_closure_sha256,
        **package_control_roots,
    }


def _formal_policy() -> dict:
    return {
        "schema": somph_bundle.SOMPH_FORMAL_POLICY_SCHEMA,
        "status": somph_bundle.SOMPH_FORMAL_POLICY_STATUS,
        "formal_receivers": list(somph_bundle.FORMAL_RECEIVERS),
        "old_tx_ids": list(somph_bundle.OLD_TX_IDS),
        "nested_new_tx_ids": [
            list(somph_bundle.NEW_TX_IDS[:count])
            for count in somph_bundle.FORMAL_NEW_CLASS_COUNTS
        ],
        "cache_scope": "stage2_registered",
        "old_dataset_basename": "ManySig.pkl",
        "new_dataset_basename": "ManyTx.pkl",
        "physical_sample_scenario_assignment_policy": (
            somph_bundle.lineage_authority.PHYSICAL_SAMPLE_SCENARIO_ASSIGNMENT_POLICY
        ),
        "single_observation_contract": (
            somph_bundle.lineage_authority.PHASE2_SINGLE_OBSERVATION_CONTRACT
        ),
        "required_code_closure_members": list(
            somph_bundle.CODE_CLOSURE_LOGICAL_MEMBERS
        ),
    }


def _signed_policy_envelope(payload: dict) -> tuple[dict, bytes]:
    seed = b"\x17" * 32
    hashed = hashlib.sha512(seed).digest()
    scalar = int.from_bytes(
        bytes([hashed[0] & 248])
        + hashed[1:31]
        + bytes([(hashed[31] & 63) | 64]),
        "little",
    )
    public_key = somph_bundle.lineage_authority._ed_encode(
        somph_bundle.lineage_authority._ed_scalar_mult(
            somph_bundle.lineage_authority._ED_B, scalar
        )
    )
    envelope = {
        "schema": somph_bundle.SOMPH_SIGNED_POLICY_ENVELOPE_SCHEMA,
        "domain": somph_bundle.SOMPH_SIGNED_POLICY_ENVELOPE_DOMAIN,
        "issuer": somph_bundle.lineage_authority.PINNED_AUTHORITY_ISSUER,
        "key_id": somph_bundle.lineage_authority.PINNED_AUTHORITY_KEY_ID,
        **payload,
        "signature_ed25519_hex": "",
    }
    message = somph_bundle._policy_signature_message(envelope)
    nonce = int.from_bytes(
        hashlib.sha512(hashed[32:] + message).digest(), "little"
    ) % somph_bundle.lineage_authority._ED_L
    encoded_r = somph_bundle.lineage_authority._ed_encode(
        somph_bundle.lineage_authority._ed_scalar_mult(
            somph_bundle.lineage_authority._ED_B, nonce
        )
    )
    challenge = int.from_bytes(
        hashlib.sha512(encoded_r + public_key + message).digest(), "little"
    ) % somph_bundle.lineage_authority._ED_L
    signature_scalar = (
        nonce + challenge * scalar
    ) % somph_bundle.lineage_authority._ED_L
    envelope["signature_ed25519_hex"] = (
        encoded_r + signature_scalar.to_bytes(32, "little")
    ).hex()
    return envelope, public_key


def _policy_inputs(
    base: Path,
    *,
    manifest: dict,
    root: Path,
    seal_path: Path,
    seal_sha256: str,
    commit_sha256: str,
    lock: dict,
    attestation: dict,
    commit: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, object]:
    base.mkdir(parents=True)
    policy_path = base / "formal_policy.json"
    policy_path.write_text(json.dumps(_formal_policy(), sort_keys=True), "utf-8")
    policy_sha256 = sha256_file(policy_path)
    _members, code_closure_sha256 = somph_bundle._code_closure()
    provenance_payload = json.loads(
        (root / "overlay_provenance.json").read_text(encoding="utf-8")
    )
    provenance = somph_bundle._validate_provenance(
        provenance_payload,
        profile=manifest["profile"],
        receiver=manifest["receiver"],
        seed=manifest["seed"],
    )
    control_roots = somph_bundle._package_control_roots(
        manifest, provenance, new_tx_ids=lock["new_tx_ids"]
    )
    authorization = _formal_policy_authorization(
        manifest=manifest,
        seal_path=seal_path,
        seal_sha256=seal_sha256,
        commit_sha256=commit_sha256,
        lock=lock,
        attestation=attestation,
        commit=commit,
        policy_sha256=policy_sha256,
        code_closure_sha256=code_closure_sha256,
        package_control_roots=control_roots,
    )
    authorization_path = base / "formal_policy_authorization.json"
    authorization_path.write_text(
        json.dumps(authorization, sort_keys=True), "utf-8"
    )
    envelope, public_key = _signed_policy_envelope(
        {
            "authorization_canonical_sha256": somph_bundle.sha256_bytes(
                somph_bundle.canonical_json_bytes(authorization)
            ),
            "formal_policy_sha256": policy_sha256,
            "package_root_sha256": manifest["package_root_sha256"],
            "package_detached_seal_sha256": seal_sha256,
            "authority_commit_sha256": commit_sha256,
            "receiver": manifest["receiver"],
            "seed": manifest["seed"],
            "stage": manifest["stage"],
            "registration_state": manifest["registration_state"],
            "k_shot": manifest["k_shot"],
            "code_closure_sha256": code_closure_sha256,
        }
    )
    envelope_path = base / "signed_policy_envelope.json"
    envelope_path.write_text(json.dumps(envelope, sort_keys=True), "utf-8")
    monkeypatch.setattr(
        runner,
        "preflight_somph_predictor_bundle_with_authority",
        somph_bundle._make_test_authority_preflight(public_key),
    )
    return {
        "formal_policy": policy_path,
        "formal_policy_authorization": authorization_path,
        "signed_policy_authorization_envelope": envelope_path,
        "expected_signed_policy_authorization_envelope_sha256": sha256_file(
            envelope_path
        ),
    }


def test_synthetic_end_to_end_run_preserves_single_observation_k10_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    before_root = tmp_path / "before" / "enrollment_only"
    after_root = tmp_path / "after" / "enrollment_only"
    before_root, before_seal, before_seal_sha, before_manifest = _e2e_enrollment(
        before_root, state="before", monkeypatch=monkeypatch
    )
    after_root, after_seal, after_seal_sha, after_manifest = _e2e_enrollment(
        after_root, state="after", monkeypatch=monkeypatch
    )
    lock, attestation, commit = _formal_authority_payloads()
    commit_sha256 = "8" * 64
    before_policy = _policy_inputs(
        tmp_path / "before_policy",
        manifest=before_manifest,
        root=before_root,
        seal_path=before_seal,
        seal_sha256=before_seal_sha,
        commit_sha256=commit_sha256,
        lock=lock,
        attestation=attestation,
        commit=commit,
        monkeypatch=monkeypatch,
    )
    after_policy = _policy_inputs(
        tmp_path / "after_policy",
        manifest=after_manifest,
        root=after_root,
        seal_path=after_seal,
        seal_sha256=after_seal_sha,
        commit_sha256=commit_sha256,
        lock=lock,
        attestation=attestation,
        commit=commit,
        monkeypatch=monkeypatch,
    )
    monkeypatch.setattr(
        somph_bundle.lineage_authority,
        "verify_somph_lineage_authority_bundle",
        lambda *_args, **_kwargs: (lock, attestation, commit),
    )
    call_order: list[str] = []
    real_authority_preflight = (
        runner.preflight_somph_predictor_bundle_with_authority
    )
    real_materialize = runner.materialize_somph_enrollment_with_authority

    def tracked_authority_preflight(root: Path, **kwargs):
        call_order.append(f"authority:{root.parent.name}")
        return real_authority_preflight(root, **kwargs)

    def tracked_materialize(root: Path, *args, **kwargs):
        call_order.append(f"materialize:{root.parent.name}")
        return real_materialize(root, *args, **kwargs)

    def fake_base_feature(_model, _device, iq: np.ndarray) -> np.ndarray:
        assert iq.shape[0] == 1
        complex_iq = iq[:, 0] + 1j * iq[:, 1]
        spectrum = np.abs(np.fft.fft(complex_iq, axis=1))
        return np.stack(
            [
                np.mean(complex_iq.real, axis=1),
                np.mean(complex_iq.imag, axis=1),
                np.std(complex_iq.real, axis=1),
                np.std(complex_iq.imag, axis=1),
                *[spectrum[:, index] for index in range(1, 5)],
            ],
            axis=1,
        ).astype(np.float32)

    monkeypatch.setattr(
        runner,
        "preflight_somph_predictor_bundle_with_authority",
        tracked_authority_preflight,
    )
    monkeypatch.setattr(
        runner,
        "materialize_somph_enrollment_with_authority",
        tracked_materialize,
    )
    monkeypatch.setattr(
        runner, "load_torchscript_backbone_same_fd", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(runner, "_base_feature", fake_base_feature)
    output = tmp_path / "d18_output"
    receipt = runner.run(
        before_root=before_root,
        before_seal=before_seal,
        expected_before_seal_sha256=before_seal_sha,
        before_authority_bundle_root=tmp_path / "before_authority_bundle",
        expected_before_authority_commit_sha256=commit_sha256,
        before_formal_policy=before_policy["formal_policy"],
        before_formal_policy_authorization=before_policy[
            "formal_policy_authorization"
        ],
        before_signed_policy_authorization_envelope=before_policy[
            "signed_policy_authorization_envelope"
        ],
        expected_before_signed_policy_authorization_envelope_sha256=before_policy[
            "expected_signed_policy_authorization_envelope_sha256"
        ],
        after_root=after_root,
        after_seal=after_seal,
        expected_after_seal_sha256=after_seal_sha,
        after_authority_bundle_root=tmp_path / "after_authority_bundle",
        expected_after_authority_commit_sha256=commit_sha256,
        after_formal_policy=after_policy["formal_policy"],
        after_formal_policy_authorization=after_policy[
            "formal_policy_authorization"
        ],
        after_signed_policy_authorization_envelope=after_policy[
            "signed_policy_authorization_envelope"
        ],
        expected_after_signed_policy_authorization_envelope_sha256=after_policy[
            "expected_signed_policy_authorization_envelope_sha256"
        ],
        output=output,
        device_name="cpu",
    )
    assert call_order[:4] == [
        "authority:before",
        "authority:after",
        "materialize:before",
        "materialize:after",
    ]
    assert receipt["query_opened"] is False
    inventory = json.loads((output / "support_inventory.json").read_text("utf-8"))
    assert len(inventory) == 3 * 11 * 10
    assert len({row["physical_sample_id"] for row in inventory}) == len(inventory)
    assert all(row["post_reception_view_count"] == 1 for row in inventory)
    assert all(
        row["post_reception_view_counts_as_additional_physical_sample"] is False
        and row["additional_leo_channel_state_generation"] is False
        for row in inventory
    )
    audit = json.loads((output / "support_audit.json").read_text("utf-8"))
    assert audit["cross_scenario_support_disjointness"]["all_pairwise_disjoint"] is True
    assert audit["runtime_access_audit"]["query_package_opened"] is False
    assert audit["runtime_access_audit"]["additional_leo_channel_state_generation"] is False
    anchor = audit["k10_selection_authority_anchor"]
    assert len(anchor["k10_selection_authority_anchor_sha256"]) == 64
    assert (
        anchor["payload"]["before_authority_audit"]["status"]
        == "CURRENT_PROTOCOL_REAL_INPUT_AUDIT_PASS"
    )
    assert anchor["payload"]["before_authority_audit"]["iq_archive_opened"] is True
    assert anchor["payload"]["after_authority_audit"]["iq_payload_materialized"] is True
    assert all(
        row["one_forward_per_unique_physical_support_verified"] is True
        for row in audit["runtime_access_audit"]["selected_full_deployment_fit"].values()
    )
    assert (output / "training_log.jsonl").is_file()
    assert (output / "report.md").is_file()
    assert (output / "RECEIPT.json").is_file()


def test_authority_failure_never_opens_or_crc_checks_iq_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, seal, seal_sha, _manifest = _e2e_enrollment(
        tmp_path / "blocked" / "enrollment_only",
        state="before",
        monkeypatch=monkeypatch,
    )
    iq_open_calls = {"count": 0}
    materialize_calls = {"count": 0}

    def forbidden_iq_open(*_args, **_kwargs):
        iq_open_calls["count"] += 1
        raise AssertionError("authority gate opened or CRC-checked IQ archive")

    def forbidden_materialize(*_args, **_kwargs):
        materialize_calls["count"] += 1
        raise AssertionError("authority gate reached D14 IQ materializer")

    def rejected_authority(*_args, **_kwargs):
        raise ValueError("synthetic authority rejection")

    monkeypatch.setattr(somph_bundle, "_inspect_iq_member", forbidden_iq_open)
    monkeypatch.setattr(
        somph_bundle.lineage_authority,
        "verify_somph_lineage_authority_bundle",
        rejected_authority,
    )
    monkeypatch.setattr(
        runner,
        "materialize_somph_enrollment_with_authority",
        forbidden_materialize,
    )
    output = tmp_path / "blocked_output"
    with pytest.raises(
        somph_bundle.PredictorPackageError,
        match="external lineage authority verification failed",
    ):
        runner.run(
            before_root=root,
            before_seal=seal,
            expected_before_seal_sha256=seal_sha,
            before_authority_bundle_root=tmp_path / "missing_authority_bundle",
            expected_before_authority_commit_sha256="8" * 64,
            before_formal_policy=tmp_path / "missing_formal_policy.json",
            before_formal_policy_authorization=tmp_path / "missing_policy.json",
            before_signed_policy_authorization_envelope=(
                tmp_path / "missing_policy_envelope.json"
            ),
            expected_before_signed_policy_authorization_envelope_sha256="9" * 64,
            after_root=root,
            after_seal=seal,
            expected_after_seal_sha256=seal_sha,
            after_authority_bundle_root=tmp_path / "missing_after_authority_bundle",
            expected_after_authority_commit_sha256="8" * 64,
            after_formal_policy=tmp_path / "missing_after_formal_policy.json",
            after_formal_policy_authorization=tmp_path / "missing_after_policy.json",
            after_signed_policy_authorization_envelope=(
                tmp_path / "missing_after_policy_envelope.json"
            ),
            expected_after_signed_policy_authorization_envelope_sha256="9" * 64,
            output=output,
            device_name="cpu",
        )
    assert iq_open_calls["count"] == 0
    assert materialize_calls["count"] == 0
    assert output.exists() is False
