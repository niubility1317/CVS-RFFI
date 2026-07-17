from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from cvsrffi.phase1_int8_prototype_bundle import (
    ALLOWED_NPZ_MEMBERS,
    build_int8_component,
    quantize_domain_class_centroids,
    save_int8_component,
    validate_int8_component,
)


CHECKPOINT_SHA = "a" * 64
SOURCE_SHA = "b" * 64


def _package() -> dict:
    generator = torch.Generator().manual_seed(17)
    proto = torch.nn.functional.normalize(torch.randn(3, 4, 8, generator=generator), dim=-1)
    counts = torch.tensor([[5, 0, 4, 3], [2, 3, 0, 2], [1, 4, 5, 0]])
    proto[counts == 0] = 0
    return {"feature_key": "z_id", "tx_domain_prototypes": proto, "tx_domain_counts": counts}


def test_quantizer_is_int8_and_high_fidelity() -> None:
    source = _package()
    payload, audit = quantize_domain_class_centroids(
        source["tx_domain_prototypes"], source["tx_domain_counts"] > 0
    )
    assert payload["domain_class_q"].dtype == np.int8
    assert payload["domain_class_q"].shape == (4, 3, 8)
    assert payload["domain_class_scale"].dtype == np.float16
    assert audit["active_domain_class_cells"] == 9
    assert audit["min_cosine"] > 0.999


def test_saved_component_has_only_compressed_allowlisted_payload(tmp_path) -> None:
    payload, manifest = build_int8_component(
        _package(),
        class_registry=["tx-a", "tx-b", "tx-c"],
        checkpoint_sha256=CHECKPOINT_SHA,
        source_prototype_artifact_sha256=SOURCE_SHA,
        provenance_status="UNVERIFIED_UNDER_CURRENT_PROTOCOL",
        formal_phase2_eligible=False,
    )
    result = save_int8_component(tmp_path, payload, manifest)
    validated = validate_int8_component(tmp_path)
    assert validated["formal_phase2_eligible"] is False
    assert validated["phase2_phase1_prototype_update_access"] is False
    assert len(result["deployment_bundle_root_sha256"]) == 64
    with np.load(tmp_path / "int8_domain_class_prototypes.npz", allow_pickle=False) as arrays:
        assert set(arrays.files) == ALLOWED_NPZ_MEMBERS
        assert not any(token in key for key in arrays.files for token in ("sample", "features", "count", "path"))
        assert arrays["domain_class_q"].dtype == np.int8
    text = (tmp_path / "manifest.json").read_text(encoding="utf-8").lower()
    for forbidden in ("checkpoint_path", "dataset_path", "sample_id", "source_count"):
        assert forbidden not in text


def test_validator_rejects_hash_mismatch(tmp_path) -> None:
    payload, manifest = build_int8_component(
        _package(),
        class_registry=["tx-a", "tx-b", "tx-c"],
        checkpoint_sha256=CHECKPOINT_SHA,
        source_prototype_artifact_sha256=SOURCE_SHA,
        provenance_status="STRICT_PHASE1_EXPORT",
        formal_phase2_eligible=True,
    )
    save_int8_component(tmp_path, payload, manifest)
    path = tmp_path / "manifest.json"
    parsed = json.loads(path.read_text(encoding="utf-8"))
    parsed["component_npz_sha256"] = "0" * 64
    path.write_text(json.dumps(parsed), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        validate_int8_component(tmp_path)


def test_build_rejects_wrong_registry_size() -> None:
    with pytest.raises(ValueError, match="class_registry length"):
        build_int8_component(
            _package(),
            class_registry=["tx-a"],
            checkpoint_sha256=CHECKPOINT_SHA,
            source_prototype_artifact_sha256=SOURCE_SHA,
            provenance_status="STRICT_PHASE1_EXPORT",
            formal_phase2_eligible=True,
        )
