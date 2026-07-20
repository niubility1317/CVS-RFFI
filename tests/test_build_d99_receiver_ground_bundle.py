from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from cvsrffi import stage2_d99_ra_cgtmk_d81 as d99
from cvsrffi import stage2_d99_d100_phase1_lodo as lodo
from cvsrffi.phase1_int8_prototype_bundle import (
    build_int8_component,
    save_int8_component,
)
from cvsrffi.phase2_candidate_capsule import BASE_CHECKPOINT_SHA256
from scripts import run_d99_d100_narrow as narrow_runner


SCRIPT = Path(__file__).resolve().parents[1] / "code" / "scripts" / "build_d99_receiver_ground_bundle.py"
SPEC = importlib.util.spec_from_file_location("build_d99_receiver_ground_bundle", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source(tmp_path: Path) -> tuple[Path, str]:
    rng = np.random.default_rng(17)
    vectors = rng.normal(size=(26, 6, d99.Z_DIM)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=2, keepdims=True)
    active = np.zeros((26, 6), dtype=bool)
    for _receiver, pair in module.SOURCE_DOMAIN_PAIRS:
        active[np.asarray(pair), :] = True
    counts = np.where(active, 2, 0).T
    prototypes = np.transpose(vectors, (1, 0, 2))
    prototypes[counts == 0] = 0.0
    payload, manifest = build_int8_component(
        {
            "feature_key": "z_id",
            "tx_domain_prototypes": torch.from_numpy(prototypes),
            "tx_domain_counts": torch.from_numpy(counts),
        },
        class_registry=module.OLD_TX_REGISTRY,
        checkpoint_sha256=BASE_CHECKPOINT_SHA256,
        source_prototype_artifact_sha256="b" * 64,
        provenance_status="UNVERIFIED_UNDER_CURRENT_PROTOCOL",
        formal_phase2_eligible=False,
    )
    root = tmp_path / "source"
    result = save_int8_component(root, payload, manifest)
    return root, result["manifest_sha256"]


def test_builds_seven_receiver_typed_bundle_and_base_lock(tmp_path: Path) -> None:
    source, manifest_sha = _source(tmp_path)
    output = tmp_path / "output"
    result = module.build_bundle(source, manifest_sha, output)
    assert result["domain_count"] == 7
    assert result["class_count"] == 6
    assert result["formal_phase1_eligible"] is False
    assert result["min_requantization_cosine"] > 0.999
    with np.load(output / module.OUTPUT_NPZ, allow_pickle=False) as payload:
        assert tuple(payload.files) == module.OUTPUT_MEMBERS
        arrays = {name: np.asarray(payload[name]) for name in payload.files}
    release_raw = (output / module.OUTPUT_MANIFEST).read_bytes()
    receipt_data = json.loads(release_raw.decode("utf-8"))["aggregation_receipt"]
    receipt = d99.ExternalGroundAggregationReceipt(**receipt_data)
    bundle = d99.produce_typed_ground_aggregate_bundle(
        **arrays, aggregation_receipt=receipt
    )
    authority = lodo.load_ground_release_authority(
        release_raw, hashlib.sha256(release_raw).hexdigest(), bundle
    )
    assert tuple(authority.receiver_domain_map) == tuple(
        receiver for receiver, _pair in module.SOURCE_DOMAIN_PAIRS
    )
    assert authority.formal_phase1_eligible is False
    wrapper = json.loads((output / module.OUTPUT_LOCK).read_text())
    assert wrapper["schema"] == module.DEVELOPMENT_D99_PRIOR_SCHEMA
    assert wrapper["status"] == module.DEVELOPMENT_D99_PRIOR_STATUS
    assert tuple(wrapper["placeholder_evidence_fields"]) == (
        module.DEVELOPMENT_D99_PLACEHOLDER_EVIDENCE_FIELDS
    )
    values = dict(wrapper["values"])
    values["ground_old_registry"] = tuple(values["ground_old_registry"])
    lock = d99.Phase1D99Lock(**values)
    assert lock.ground_bundle_receipt_sha256 == bundle.bundle_sha256
    assert lock.ground_old_registry == tuple(arrays["ground_old_registry"].astype(str))
    loaded_by_narrow = narrow_runner._load_d99_lock(output / module.OUTPUT_LOCK)
    assert loaded_by_narrow.lock_digest == lock.lock_digest
    spec = json.loads((output / module.OUTPUT_SPEC).read_text())
    assert spec["target_rows_used"] == 0 and spec["query_rows_used"] == 0
    assert spec["pair_weights"] == [0.5, 0.5]


def test_builder_is_deterministic_nonoverwriting_and_rejects_layout_drift(tmp_path: Path) -> None:
    source, manifest_sha = _source(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"
    module.build_bundle(source, manifest_sha, first)
    module.build_bundle(source, manifest_sha, second)
    for name in (
        module.OUTPUT_NPZ,
        module.OUTPUT_MANIFEST,
        module.OUTPUT_SPEC,
        module.OUTPUT_LOCK,
        module.OUTPUT_RESULT,
    ):
        assert (first / name).read_bytes() == (second / name).read_bytes()
    with pytest.raises(module.D99GroundBundleBuildError, match="already exists"):
        module.build_bundle(source, manifest_sha, first)

    with np.load(source / "int8_domain_class_prototypes.npz", allow_pickle=False) as payload:
        arrays = {name: np.asarray(payload[name]) for name in payload.files}
    arrays["domain_class_mask"][2, :] = 1
    np.savez_compressed(source / "int8_domain_class_prototypes.npz", **arrays)
    manifest = json.loads((source / "manifest.json").read_text())
    manifest["component_npz_sha256"] = _sha(source / "int8_domain_class_prototypes.npz")
    (source / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(module.D99GroundBundleBuildError, match="array layout"):
        module.build_bundle(source, _sha(source / "manifest.json"), tmp_path / "bad")
