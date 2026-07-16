from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import cvsrffi.stage2_diag_cosine_exploration as route


def _separable(seed: int = 7):
    rng = np.random.default_rng(seed)
    centers = np.eye(3, 24, dtype=np.float32)
    support = np.vstack(
        [centers[index] + 0.01 * rng.normal(size=(9, 24)) for index in range(3)]
    ).astype(np.float32)
    query = np.vstack(
        [centers[index] + 0.01 * rng.normal(size=(5, 24)) for index in range(3)]
    ).astype(np.float32)
    labels = np.repeat(["class-a", "class-b", "class-c"], 9)
    truth = np.repeat(["class-a", "class-b", "class-c"], 5)
    return support, labels, query, truth


def test_fit_has_no_query_argument_and_prediction_is_batch_extension_invariant():
    support, labels, query, truth = _separable()
    state = route.fit_diag_cosine_state(
        support, labels, seed=19, device=torch.device("cpu")
    )
    first = route.predict_diag_cosine(state, query)
    extended = route.predict_diag_cosine(
        state, np.vstack([query, np.full((7, query.shape[1]), 99.0, dtype=np.float32)])
    )
    assert np.mean(first.astype(str) == truth.astype(str)) == 1.0
    assert np.array_equal(first, extended[: len(first)])
    assert state.resource["query_rows_used_for_fit"] == 0
    assert state.resource["query_features_used_for_fit"] is False
    assert state.resource["query_role_oracle_access"] is False
    assert state.resource["query_class_quota_access"] is False
    assert state.resource["trainable_parameters"] <= 50_000
    assert state.resource["persistent_state_bytes"] <= 256 * 1024
    assert len(state.trace) == 20

    stable = route.fit_diag_cosine_state(
        support,
        labels,
        seed=19,
        device=torch.device("cpu"),
        candidate=route.CANDIDATE_D2,
    )
    stable_first = route.predict_diag_cosine(stable, query)
    stable_extended = route.predict_diag_cosine(
        stable,
        np.vstack([query, np.full((3, query.shape[1]), -77.0, dtype=np.float32)]),
    )
    assert np.array_equal(stable_first, stable_extended[: len(stable_first)])
    assert stable.resource["trainable_parameters"] == support.shape[1]
    assert (
        stable.resource["classifier_state_policy"]
        == "current_registry_fixed_support_prototypes_zero_class_bias_shared_diag_only"
    )


def test_fft_rf_features_are_same_row_gain_normalized_and_128d():
    rng = np.random.default_rng(3)
    iq = rng.normal(size=(4, 2, 64)).astype(np.float32)
    fft = route.spectral_logmag_sketch(iq)
    rf = route.rf_statistics(iq)
    scaled_fft = route.spectral_logmag_sketch(7.5 * iq)
    scaled_rf = route.rf_statistics(7.5 * iq)
    assert fft.shape == (4, 96)
    assert rf.shape == (4, 32)
    assert np.allclose(fft, scaled_fft, atol=1.0e-6)
    assert np.allclose(rf, scaled_rf, atol=1.0e-6)
    zid = np.tile(np.eye(1, 160, dtype=np.float32), (4, 1))
    assert route.registered_feature(iq, zid).shape == (4, 288)


class _FakeRuntime(torch.nn.Module):
    def forward(self, rows):
        flat = rows.reshape(rows.shape[0], -1)
        features = torch.zeros((len(rows), 160), dtype=torch.float32, device=rows.device)
        features[:, : flat.shape[1]] = flat
        logits = torch.zeros((len(rows), 6), dtype=torch.float32, device=rows.device)
        return features, logits


def _manifest(profile: str) -> dict:
    classes = [
        {"class_handle": "cls_" + "1" * 64},
        {"class_handle": "cls_" + "2" * 64},
    ]
    return {
        "profile": profile,
        "stage": "stage2c",
        "registration_state": "after",
        "receiver": "20-1",
        "seed": 713101,
        "k_shot": 2,
        "row_handle": (
            None if profile == route.ENROLLMENT_ONLY else "row_" + "3" * 64
        ),
        "row_manifest_sha256": (
            None if profile == route.ENROLLMENT_ONLY else "4" * 64
        ),
        "phase1_checkpoint_sha256": "5" * 64,
        "feature_runtime_sha256": "6" * 64,
        "method_lock_sha256": "7" * 64,
        "package_root_sha256": ("8" if profile == route.ENROLLMENT_ONLY else "9") * 64,
        "registered_classes": classes,
        "members": [{"kind": "feature_runtime", "relative_path": "sealed_feature_runtime.pt"}],
    }


def _payloads(profile: str) -> dict:
    result = {}
    for scenario_index, scenario in enumerate(route.FORMAL_LEO_WEAK_SCENARIOS):
        if profile == route.ENROLLMENT_ONLY:
            rows = []
            labels = []
            ranks = []
            for class_index, sign in enumerate((1.0, -1.0)):
                for rank in range(3):
                    iq = np.zeros((2, 16), dtype=np.float32)
                    iq[0, class_index] = sign * (1.0 + 0.01 * rank)
                    iq[1, 2 + scenario_index] = 0.1
                    rows.append(iq)
                    labels.append(class_index)
                    ranks.append(rank)
            result[scenario] = {
                "support_leo_weak_iq": np.stack(rows),
                "support_class_indices": np.asarray(labels, dtype=np.int64),
                "support_rank_within_class": np.asarray(ranks, dtype=np.int64),
            }
        else:
            rows = []
            for class_index, sign in enumerate((1.0, -1.0)):
                iq = np.zeros((2, 16), dtype=np.float32)
                iq[0, class_index] = sign
                iq[1, 2 + scenario_index] = 0.1
                rows.append(iq)
            result[scenario] = {
                "query_leo_weak_iq": np.stack(rows),
                "query_tokens": np.asarray(
                    [
                        f"qid_{scenario_index}{class_index}" + "a" * 62
                        for class_index in range(2)
                    ]
                ),
            }
    return result


def test_run_writes_unlabeled_prediction_before_any_scorer(
    tmp_path: Path, monkeypatch
):
    enrollment_root = tmp_path / "enrollment"
    apply_root = tmp_path / "apply"
    output_root = tmp_path / "output"
    enrollment_root.mkdir()
    apply_root.mkdir()
    output_root.mkdir()
    enrollment_manifest = _manifest(route.ENROLLMENT_ONLY)
    apply_manifest = _manifest(route.APPLY_ONLY)

    calls = iter(
        [
            (_payloads(route.ENROLLMENT_ONLY), enrollment_manifest, {"status": "PASS"}),
            (_payloads(route.APPLY_ONLY), apply_manifest, {"status": "PASS"}),
        ]
    )
    monkeypatch.setattr(
        route, "load_verified_somph_predictor_bundle", lambda *args, **kwargs: next(calls)
    )
    monkeypatch.setattr(
        route,
        "load_torchscript_backbone_same_fd",
        lambda *args, **kwargs: _FakeRuntime(),
    )
    result = route.run_diag_cosine_exploration(
        enrollment_package_root=enrollment_root,
        enrollment_seal_path=tmp_path / "enrollment.seal.json",
        enrollment_seal_sha256="a" * 64,
        apply_package_root=apply_root,
        apply_seal_path=tmp_path / "apply.seal.json",
        apply_seal_sha256="b" * 64,
        output_root=output_root,
        device="cpu",
        candidate=route.CANDIDATE_D2,
    )
    assert result["formal_launch_authority"] is False
    with np.load(output_root / "prediction_artifact.npz", allow_pickle=False) as archive:
        assert tuple(archive.files) == (
            "query_tokens",
            "scenarios",
            "predicted_class_handles",
        )
        assert len(archive["query_tokens"]) == 6
    receipt = json.loads(
        (output_root / "execution_receipt.json").read_text(encoding="utf-8")
    )
    raw = json.dumps(receipt, sort_keys=True)
    assert "true_label" not in raw
    assert "role_label" not in raw
    assert receipt["query_truth_present_in_predictor"] is False
    assert receipt["resource"]["query_rows_used_for_fit"] == 0
    assert receipt["resource"]["support_enrollment_rows"] == 12
    assert receipt["resource"]["query_backbone_forwards_per_sample"] == 1
    assert receipt["candidate"]["name"] == route.CANDIDATE_D2
    assert (output_root / "COMMIT.json").is_file()
