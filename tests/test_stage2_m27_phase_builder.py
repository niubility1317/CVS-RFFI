from __future__ import annotations

import numpy as np

from cvsrffi.somph_predictor_bundle import FORMAL_LEO_WEAK_SCENARIOS
from cvsrffi import stage2_m27_phase_builder as builder


def test_phase_builder_uses_only_sealed_stage2c_iq_and_preserves_base_tokens(
    tmp_path, monkeypatch
) -> None:
    old = tuple(f"cls_{index:032x}" for index in range(6))
    new = tuple(f"cls_{index + 6:032x}" for index in range(5))
    classes = old + new
    rng = np.random.default_rng(82730)
    base_payloads = {}
    support_payloads = {}
    query_payloads = {}
    for scene_index, scene in enumerate(FORMAL_LEO_WEAK_SCENARIOS):
        tokens = np.asarray(
            [f"qid_{scene_index:02x}{index:02x}" + "a" * 60 for index in range(7)]
        )
        base_payloads[scene] = {"query_tokens": tokens}
        support_payloads[scene] = {
            "support_pool_rank_within_class": np.tile(np.arange(2), len(classes)),
            "support_pool_class_indices": np.repeat(np.arange(len(classes)), 2),
            "support_pool_leo_weak_iq": rng.normal(
                size=(len(classes) * 2, 2, 128)
            ).astype(np.float32),
        }
        query_payloads[scene] = {
            "query_leo_weak_iq": rng.normal(size=(len(tokens), 2, 128)).astype(
                np.float32
            ),
            "query_tokens": tokens,
        }
    base = {
        "manifest": {
            "receiver": "3-19",
            "method_seed": 7282101,
            "k_shot": 2,
            "capsule_id": "capsule-fixed",
            "split_id": "split-fixed",
            "phase2_data_status": "VALIDATED_ONCE",
            "package_root_sha256": "1" * 64,
            "package_seal_sha256": "2" * 64,
        },
        "old_classes": old,
        "new_classes": new,
        "scenario_payloads": base_payloads,
    }
    package_manifest = {
        "stage": "stage2c",
        "receiver": "3-19",
        "new_class_count": len(new),
        "package_root_sha256": "1" * 64,
        "registered_classes": [
            {"class_index": index, "class_handle": name}
            for index, name in enumerate(classes)
        ],
    }
    monkeypatch.setattr(builder, "load_feature_cache", lambda *_args, **_kwargs: base)
    monkeypatch.setattr(
        builder,
        "_load_package",
        lambda *_args, **_kwargs: (
            support_payloads,
            query_payloads,
            package_manifest,
            {},
        ),
    )
    captured = {}

    def publish(*_args, **kwargs):
        captured.update(kwargs)
        return {"payload_sha256": "3" * 64, "manifest_sha256": "4" * 64}

    monkeypatch.setattr(builder, "publish_phase_side_cache", publish)
    result = builder.build_phase_side_cache_from_sealed_stage2c(
        base_feature_cache_payload=tmp_path / "features.npz",
        base_feature_cache_manifest=tmp_path / "features.manifest.json",
        base_feature_cache_payload_sha256="5" * 64,
        base_feature_cache_manifest_sha256="6" * 64,
        after_package_root=tmp_path / "package",
        after_seal_path=tmp_path / "seal.json",
        after_seal_sha256="2" * 64,
        output_root=tmp_path / "phase",
    )
    assert result["query_truth_opened"] is False
    assert result["raw_dataset_opened"] is False
    assert captured["base_manifest_sha256"] == "6" * 64
    assert set(captured["scenario_payloads"]) == set(FORMAL_LEO_WEAK_SCENARIOS)
    for scene in FORMAL_LEO_WEAK_SCENARIOS:
        payload = captured["scenario_payloads"][scene]
        assert payload["old_support_phase32"].shape == (12, 32)
        assert payload["new_support_phase32"].shape == (10, 32)
        assert payload["query_phase32"].shape == (7, 32)
        np.testing.assert_array_equal(payload["query_tokens"], base_payloads[scene]["query_tokens"])
