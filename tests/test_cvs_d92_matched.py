from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("cvs_d92_matched", ROOT / "tools/cvs_d92_matched.py")
matched = importlib.util.module_from_spec(spec)
spec.loader.exec_module(matched)


def fixture(new_count=2, k=1):
    rng = np.random.default_rng(9413)
    classes = tuple(f"class{c}" for c in range(6 + new_count))
    labels = np.repeat(np.arange(len(classes)), k)
    x = rng.normal(size=(len(labels), 288)).astype(np.float32)
    basis, _ = np.linalg.qr(rng.normal(size=(160, 3)))
    ground = dict(ground_basis=basis, ground_spectral_weights=np.array([0.5, 0.3, 0.2]),
        ground_audit=dict(d81_basis_sha256="a" * 64, d81_spectral_weight_sha256="b" * 64,
            d81_participation_ratio_effective_rank=2.6, d81_retained_rank=3,
            d81_rank_policy="ceil_participation_ratio_effective_rank", ground_component_input_count=18,
            ground_statistic_semantics="class_centered_cross_domain_centroid_drift_eigenspectrum"))
    return dict(support_features=x, support_labels=labels, classes=classes, ground=ground, seed=12)


@pytest.mark.parametrize("new_count", [0, 2])
def test_real_fit_true256_and_query_partition_invariance(new_count):
    from cvsrffi import stage2_ablation_executors as executor
    torch.set_num_threads(1)
    original = executor.get_stage2_arm
    inputs = fixture(new_count)
    support_before = inputs["support_features"].copy()
    state = matched.fit_d92(**inputs)
    assert executor.get_stage2_arm is original
    assert state.active_feature_dim == 256
    assert state.compiled_affine_state is not None
    np.testing.assert_array_equal(inputs["support_features"], support_before)
    query = np.random.default_rng(4).normal(size=(4, 288)).astype(np.float32)
    batch = matched.predict_scores(state, query)
    individual = np.concatenate([matched.predict_scores(state, row[None]) for row in query])
    np.testing.assert_allclose(batch, individual, rtol=2e-5, atol=2e-4)
    np.testing.assert_array_equal(batch.argmax(1), individual.argmax(1))
    np.testing.assert_allclose(matched.predict_scores(state, query[::-1]), batch[::-1], rtol=2e-5, atol=2e-4)
    assert batch.shape == (4, 6 + new_count)


def test_registered_fit_matches_unmodified_executor():
    from cvsrffi.stage2_ablation_executors import fit_stage2_ablation
    torch.set_num_threads(1)
    args = fixture(5)
    actual = matched.fit_d92(**args)
    x, y, classes = args["support_features"], args["support_labels"], args["classes"]
    expected = fit_stage2_ablation(ablation_id=matched.ARM,
        old_support_features=x[:6], old_support_labels=classes[:6], old_classes=classes[:6],
        new_support_features=x[6:], new_support_labels=classes[6:], new_classes=classes[6:],
        seed=12, device="cpu", **args["ground"])
    np.testing.assert_array_equal(actual.score(x), expected.score(x))


def test_two_new_classes_active_covariance_k5(monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr(torch.Tensor, "numpy", unavailable_numpy_bridge)
    monkeypatch.setattr(torch, "from_numpy", unavailable_numpy_bridge)
    state = matched.fit_d92(**fixture(2, k=5))
    assert state.audit["d92_registration_balanced_active"] is True
    assert state.resource["query_rows_used_for_fit"] == 0


def test_covariance_helper_has_only_registry_change():
    original = (ROOT / "code/cvsrffi/stage2_d92_registration_balanced_covariance.py").read_text(encoding="utf-8")
    copied = (ROOT / "tools/cvs_d92_covariance_matched.py").read_text(encoding="utf-8")
    copied = copied.split("\n", 1)[1].replace(
        "            OLD_CLASS_COUNT + 2,  # Matched matrix adds two registered new classes.\n", "")
    assert copied == original


def test_input_failure_and_old_only_descriptor_restoration():
    from cvsrffi import stage2_ablation_executors as executor
    args = fixture(0)
    args["support_features"] = args["support_features"][:, :256]
    with pytest.raises(ValueError, match="contract mismatch"):
        matched.fit_d92(**args)
    original = executor.get_stage2_arm
    with pytest.raises(RuntimeError):
        with matched._old_only_registry(executor, True):
            assert executor.get_stage2_arm(matched.ARM).stage == "stage2b"
            raise RuntimeError("synthetic failure")
    assert executor.get_stage2_arm is original


def test_ground_only_exact_L_and_current_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.Tensor, "numpy", unavailable_numpy_bridge)
    monkeypatch.setattr(torch, "from_numpy", unavailable_numpy_bridge)
    rng = np.random.default_rng(112)
    labels = np.repeat(np.arange(6), 8)
    domains = np.tile(np.repeat(np.arange(4), 2), 6)
    ids = np.asarray([f"L{i}" for i in range(48)])
    features = tmp_path / "features.npz"
    np.savez(features, identity160=rng.normal(size=(48, 160)).astype(np.float32),
             labels=labels, domains=domains, ids=ids)
    contract = tmp_path / "source_contract.json"
    contract.write_text(json.dumps(dict(role_ids=dict(L_s=ids.tolist(), U_s=["U1"], V=["V1"]),
        native_role_comparison="EXACT_MATCH", classes=[f"class{i}" for i in range(6)])), encoding="utf-8")
    checkpoint = tmp_path / "final_ssdg.pth"
    checkpoint.write_bytes(b"synthetic-checkpoint-bound-by-extractor-provenance")
    provenance = tmp_path / "initialization.json"
    metadata = dict(scratch_only=True, checkpoint_sources=[], target_contact=False,
                    target_training_contact=False, source_roles="EXACT_MATCH")
    provenance.write_text(json.dumps(metadata), encoding="utf-8")
    (tmp_path / "completion.json").write_text(json.dumps(dict(status="TRAINING_COMPLETE",
        epochs=200, target_evaluated=False, checkpoint="final_ssdg.pth")), encoding="utf-8")
    kwargs = dict(features_path=features, source_run=tmp_path, output_dir=tmp_path / "component")
    result = matched.build_ground(**kwargs)
    assert Path(result["npz_path"]).is_file()
    loaded = matched.load_ground(kwargs["output_dir"])
    assert loaded["ground_basis"].shape[0] == 160
    assert loaded["ground_audit"]["ground_bundle_contains_aggregated_p90_radius"] is True
    assert loaded["ground_audit"]["ground_statistic_semantics"].startswith("v2_cell_radius")
    assert read_manifest(result["manifest_path"])["residual_rank"] == 3
    capsule = tmp_path / "capsule"
    (capsule / "splits").mkdir(parents=True)
    query_ids = np.asarray([f"target{i}" for i in range(12)])
    np.savez(capsule / "received.npz", iq=rng.normal(size=(12, 2, 256)).astype(np.float32), ids=query_ids)
    (capsule / "manifest.json").write_text(json.dumps(dict(protocol_schema="p2_min_v1",
        phase2_data_status="VALIDATED_ONCE", capsule_id="synthetic")), encoding="utf-8")
    split = dict(split_id="test", capsule_id="synthetic", receiver="rx", scenario="residual_noeq",
        k=1, support_seed=1, registered_classes=[f"class{i}" for i in range(6)],
        support_indices=list(range(6)), support_labels=list(range(6)), query_indices=list(range(6, 12)))
    (capsule / "splits/test.json").write_text(json.dumps(split), encoding="utf-8")
    received = tmp_path / "received_features.npz"
    np.savez(received, identity160=rng.normal(size=(12, 160)).astype(np.float32),
        logits=rng.normal(size=(12, 6)).astype(np.float32), ids=query_ids)
    prediction_dir = tmp_path / "row"
    (prediction_dir / "source_features").mkdir(parents=True)
    result_prediction = matched.predict_capsule(features_path=received, capsule=capsule,
        ground_dir=kwargs["output_dir"], output_dir=prediction_dir, seed=12)
    assert result_prediction["predictions"] == 2
    assert result_prediction["truth_read"] is False
    rows = [json.loads(line) for line in (prediction_dir / "predictions.jsonl").read_text().splitlines()]
    assert {row["mode"] for row in rows} == {"d92_registration", "frozen_dg"}
    assert all(np.asarray(row["scores"]).shape == (6, 6) for row in rows)
    with np.load(result["npz_path"]) as payload:
        from cvsrffi.phase1_center_lowrank_prototype_bundle import ALLOWED_NPZ_MEMBERS
        assert set(payload.files) == ALLOWED_NPZ_MEMBERS
        assert "domain_class_q" not in payload.files
    with pytest.raises(FileExistsError):
        matched.build_ground(**kwargs)
    metadata["target_contact"] = True
    provenance.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="provenance"):
        matched.build_ground(**kwargs)
    metadata["target_contact"] = False
    # Relabeling a V record as L cannot bypass exact physical membership.
    np.savez(features, identity160=rng.normal(size=(48, 160)).astype(np.float32),
             labels=labels, domains=domains, ids=np.asarray([*ids[:-1], "V1"]))
    provenance.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="physical records"):
        matched.build_ground(**kwargs)


def read_manifest(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def unavailable_numpy_bridge(*args, **kwargs):
    raise RuntimeError("Synthetic Torch/NumPy ABI bridge unavailable")
