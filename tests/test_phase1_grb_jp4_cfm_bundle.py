"""F01/F02 traceability for the frozen GRB-JP4-CFM r2 design.

F01: a distinct r2 schema and method lock bind the checkpoint, old-class
registry, p2_min_v1 protocol, complete one-to-three multi-physical aggregate
ground prototypes per old class, immutable/non-scoring lifecycle, hashes, and
resource receipts.  Loading is read-only and every tamper fails closed.

F02: the builder deterministically constructs weighted/canonical D -> L_g,
W0 -> R, direction energy a, equal-weight decoded ground barycenters, and
Phase1-only delta_q/tau_q, then seals symmetric INT8 plus FP16-RNE state.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from cvsrffi.phase1_grb_jp4_bundle import SCHEMA as R1_SCHEMA
from cvsrffi.phase1_grb_jp4_cfm_bundle import (
    AGGREGATION_RECEIPT_SCHEMA,
    CLASS_COUNT,
    COMPONENT_PROFILE,
    FEATURE_DIM,
    MARGIN_RECEIPT_SCHEMA,
    MAX_PROTOTYPES_PER_CLASS,
    METHOD_ID,
    METHOD_LOCK_SCHEMA,
    NPZ_NAME,
    PROTOCOL_SCHEMA,
    RANK,
    RECEIVER_DAY_MEAN_SCHEMA,
    SCHEMA,
    _canonical_json_bytes,
    _canonical_svd_rows,
    _pre_sign_content_root,
    build_grb_jp4_cfm_component,
    canonical_array_sha256,
    class_handle_binding_sha256,
    load_grb_jp4_cfm_component,
    save_grb_jp4_cfm_component,
    sha256_file,
    validate_grb_jp4_cfm_component,
)


CHECKPOINT_SHA = "a" * 64
CODE_SHA = "b" * 64
CONFIG_SHA = "c" * 64
QKNN_LOCK_SHA_BY_K = {
    "1": "d" * 64,
    "5": "e" * 64,
    "10": "f" * 64,
}
MARGIN_EVIDENCE_SHA = "e" * 64
CLASSES = tuple(f"tx-{index}" for index in range(CLASS_COUNT))


def _method_lock() -> dict:
    return {
        "schema": METHOD_LOCK_SCHEMA,
        "method_id": METHOD_ID,
        "candidate_id": METHOD_ID,
        "protocol_schema": PROTOCOL_SCHEMA,
        "feature_schema": "ADV3B02:z_id:unit_l2:160:v1",
        "checkpoint_sha256": CHECKPOINT_SHA,
        "class_handle_binding_sha256": class_handle_binding_sha256(CLASSES),
        "qknn_lock_sha256_by_k": QKNN_LOCK_SHA_BY_K,
        "rank": 4,
        "old_class_count": 6,
        "allowed_k": [1, 5, 10],
        "ground_old_multiprototype_enabled": True,
        "ground_old_multiprototype_max_per_class": 3,
        "ground_old_multiprototype_min_physical_samples": 2,
        "ground_old_multiprototype_old_classes_only": True,
        "ground_prototypes_enter_qknn_bank": False,
        "ground_prototypes_generate_logits": False,
        "ground_prototypes_add_k": False,
        "ground_component_phase2_mutable": False,
        "delta_tau_source": (
            "phase1_receiver_lodo_correct_held_pseudoquery_only"
        ),
        "active_set_steps": 2,
        "ridge_fraction": 0.01,
        "theta_box_abs": 1.0,
        "trust_divisor_squared": 160,
        "g_denominator": 4,
        "target25_release_authorized": False,
        "query_fit_access": False,
        "query_rows_used_for_fit": 0,
    }


def _normalized(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    return value / np.linalg.norm(value)


def _source() -> dict:
    rng = np.random.default_rng(60720260724)
    class_records = []
    per_class_count = (1, 2, 3, 1, 2, 3)
    for class_index, (handle, count) in enumerate(
        zip(CLASSES, per_class_count)
    ):
        items = []
        for prototype_index in range(count):
            vector = _normalized(rng.normal(size=FEATURE_DIM))
            items.append(
                {
                    "vector": vector,
                    "aggregation_receipt": {
                        "schema": AGGREGATION_RECEIPT_SCHEMA,
                        "class_handle": handle,
                        "prototype_index": prototype_index,
                        "distinct_physical_sample_count": 2 + prototype_index,
                        "aggregation_radius": (
                            0.025 * (1 + class_index + prototype_index)
                        ),
                        "physical_sample_commitment_sha256": (
                            f"{class_index + 1:x}" * 64
                        )[:64],
                        "prototype_sha256": canonical_array_sha256(vector),
                        "phase1_before_target_access": True,
                        "multi_physical_aggregation": True,
                        "member_ids_included": False,
                        "sample_features_included": False,
                        "source_path_included": False,
                    },
                }
            )
        class_records.append({"class_handle": handle, "prototypes": items})

    domain_count = 7
    means = rng.normal(size=(CLASS_COUNT, domain_count, FEATURE_DIM))
    # Distinct deterministic domain amplitudes avoid an accidental rank or
    # singular-gap ambiguity in the ordinary fixture.
    means *= np.linspace(0.7, 1.9, domain_count)[None, :, None]
    mask = np.ones((CLASS_COUNT, domain_count), dtype=np.bool_)
    mask[0, -1] = False
    counts = np.where(mask, 2, 0).astype(np.int16)
    margins = np.asarray(
        [-0.42, -0.17, -0.03, 0.08, 0.19, 0.31, 0.55, 0.92, 1.20, 1.70],
        dtype=np.float64,
    )
    return {
        "feature_key": "z_id",
        "protocol_schema": PROTOCOL_SCHEMA,
        "ground_multiprototypes": class_records,
        "receiver_day_mean_schema": RECEIVER_DAY_MEAN_SCHEMA,
        "receiver_day_means": means,
        "receiver_day_mask": mask,
        "receiver_day_physical_counts": counts,
        "phase1_qknn_margin_receipt": {
            "schema": MARGIN_RECEIPT_SCHEMA,
            "target_accessed": False,
            "receiver_lodo": True,
            "pseudo_support_query_physical_id_disjoint": True,
            "correct_predictions_only": True,
            "target_query_truth_used": False,
            "margin_definition": (
                "top1_minus_logsumexp_other_raw_qknn_score"
            ),
            "margin_evidence_sha256": MARGIN_EVIDENCE_SHA,
            "margins": margins,
        },
    }


def _weight() -> np.ndarray:
    rng = np.random.default_rng(2408)
    return rng.normal(size=(FEATURE_DIM, 320)).astype(np.float64)


def _build(source: dict | None = None, lock: dict | None = None):
    return build_grb_jp4_cfm_component(
        _source() if source is None else source,
        class_registry=CLASSES,
        checkpoint_joint_proj_weight=_weight(),
        checkpoint_sha256=CHECKPOINT_SHA,
        class_handle_binding_sha256=class_handle_binding_sha256(CLASSES),
        generation_code_sha256=CODE_SHA,
        generation_config_sha256=CONFIG_SHA,
        method_lock=_method_lock() if lock is None else lock,
        provenance_status="UNIT_TEST_PHASE1_AGGREGATE_ONLY",
    )


def _manifest_sha(path: Path) -> None:
    (path.parent / "manifest.sha256").write_text(
        f"{sha256_file(path)}  manifest.json\n", encoding="ascii"
    )


def _rewrite_payload_and_reseal(
    output: Path, mutation
) -> None:
    manifest_path = output / "manifest.json"
    final = json.loads(manifest_path.read_text(encoding="utf-8"))
    npz_path = output / NPZ_NAME
    with np.load(npz_path, allow_pickle=False) as archive:
        altered = {
            key: np.array(archive[key], copy=True) for key in archive.files
        }
    mutation(altered)
    np.savez_compressed(npz_path, **altered)
    final["component_npz_sha256"] = sha256_file(npz_path)
    final["serialized_component_bytes"] = npz_path.stat().st_size
    final["array_sha256"] = {
        key: canonical_array_sha256(value)
        for key, value in sorted(altered.items())
    }
    final["pre_sign_content_root_sha256"] = _pre_sign_content_root(
        final, final["component_npz_sha256"]
    )
    manifest_path.write_text(
        json.dumps(final, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _manifest_sha(manifest_path)


def test_f01_f02_build_save_load_is_distinct_deterministic_and_read_only(
    tmp_path: Path,
) -> None:
    payload, manifest = _build()
    assert SCHEMA != R1_SCHEMA
    assert manifest["schema"] == SCHEMA
    assert manifest["component_profile"] == COMPONENT_PROFILE
    assert manifest["protocol_schema"] == PROTOCOL_SCHEMA
    assert manifest["receiver_day_mean_schema"] == RECEIVER_DAY_MEAN_SCHEMA
    assert manifest["ground_old_multiprototype_enabled"] is True
    assert manifest["target25_release_authorized"] is False
    assert manifest["formal_phase2_eligible"] is False
    assert manifest["method_lock"]["delta_q"] == float(payload["delta_q"])
    assert manifest["method_lock"]["tau_q"] == float(payload["tau_q"])
    assert (
        manifest["method_lock"]["qknn_lock_sha256_by_k"]
        == QKNN_LOCK_SHA_BY_K
    )
    assert payload["p_g_q"].shape == (6, 3, 160)
    assert payload["p_g_mask"].sum(axis=1).tolist() == [1, 2, 3, 1, 2, 3]
    assert np.all(payload["p_g_physical_counts"][payload["p_g_mask"]] >= 2)
    assert payload["p_g_weight"].dtype == np.float16
    assert payload["p_g_radius"].dtype == np.float16
    assert np.all(payload["p_g_radius"][payload["p_g_mask"]] >= 0.0)
    assert payload["p_g_quantization_max_abs_error"].dtype == np.float16
    for class_index, count in enumerate((1, 2, 3, 1, 2, 3)):
        assert np.all(
            payload["p_g_weight"][class_index, :count]
            == np.float16(1.0 / count)
        )
        assert np.all(payload["p_g_weight"][class_index, count:] == 0.0)
        assert np.all(payload["p_g_radius"][class_index, count:] == 0.0)
        for digest in payload[
            "p_g_quantization_certificate_sha256"
        ][class_index, :count]:
            assert len(bytes(digest).decode("ascii")) == 64
    assert payload["l_g_q"].shape == (4, 160)
    assert payload["r_q"].shape == (4, 320)
    assert payload["direction_energy_a"].dtype == np.float16
    assert np.isclose(
        np.linalg.norm(payload["direction_energy_a"].astype(np.float64)),
        1.0,
        atol=1.0e-3,
    )
    source_for_energy = _source()
    weighted_rows = []
    for class_index in range(CLASS_COUNT):
        observed = source_for_energy["receiver_day_means"][
            class_index,
            source_for_energy["receiver_day_mask"][class_index],
        ]
        center = observed.mean(axis=0)
        weighted_rows.extend(
            (observed - center)
            / np.sqrt(CLASS_COUNT * observed.shape[0])
        )
    singular = np.linalg.svd(
        np.stack(weighted_rows, axis=0), compute_uv=False
    )[:RANK]
    expected_energy = singular / np.sqrt(np.sum(singular**2))
    assert np.array_equal(
        payload["direction_energy_a"],
        np.asarray(expected_energy, dtype=np.float16),
    )
    for codes, scales in (
        (payload["l_g_q"], payload["l_g_scale"]),
        (payload["r_q"], payload["r_scale"]),
    ):
        decoded = codes.astype(np.float64) * scales.astype(np.float64)[:, None]
        assert np.linalg.matrix_rank(decoded, tol=1.0e-6) == RANK
        for row in decoded:
            pivot = int(np.argmax(np.abs(row)))
            assert row[pivot] > 0.0

    margins = np.asarray(
        _source()["phase1_qknn_margin_receipt"]["margins"], dtype=np.float64
    )
    tau = max(
        2.0**-10,
        1.4826 * float(np.median(np.abs(margins - np.median(margins)))),
    )
    delta = max(0.0, float(np.quantile(margins / tau, 0.10, method="linear")))
    assert payload["tau_q"] == np.asarray(tau, dtype=np.float16)
    assert payload["delta_q"] == np.asarray(delta, dtype=np.float16)

    resource = manifest["resource_audit"]
    assert resource["jp4_update_factor_wire_bytes"] <= 4096
    expected_ground_wire = sum(
        payload[field].nbytes
        for field in (
            "p_g_q",
            "p_g_scale",
            "p_g_weight",
            "p_g_radius",
            "p_g_mask",
            "p_g_physical_counts",
            "p_g_receipt_sha256",
            "p_g_source_prototype_sha256",
            "p_g_quantization_max_abs_error",
            "p_g_quantization_certificate_sha256",
        )
    )
    assert resource["ground_wire_bytes"] == expected_ground_wire
    assert resource["total_component_bytes"] > resource["ground_wire_bytes"]
    assert resource["total_component_bytes"] <= 262_144

    output = tmp_path / "component"
    saved = save_grb_jp4_cfm_component(output, payload, manifest)
    validated = validate_grb_jp4_cfm_component(
        output,
        expected_checkpoint_sha256=CHECKPOINT_SHA,
        expected_class_handle_binding_sha256=class_handle_binding_sha256(
            CLASSES
        ),
        expected_method_lock_sha256=manifest["method_lock_sha256"],
        expected_pre_sign_content_root_sha256=saved[
            "pre_sign_content_root_sha256"
        ],
    )
    with pytest.raises(ValueError, match="outer joint seal"):
        load_grb_jp4_cfm_component(
            output,
            expected_checkpoint_sha256=CHECKPOINT_SHA,
            expected_class_handle_binding_sha256=class_handle_binding_sha256(
                CLASSES
            ),
            expected_method_lock_sha256=manifest["method_lock_sha256"],
            expected_pre_sign_content_root_sha256=saved[
                "pre_sign_content_root_sha256"
            ],
        )
    loaded = load_grb_jp4_cfm_component(
        output,
        expected_checkpoint_sha256=CHECKPOINT_SHA,
        expected_class_handle_binding_sha256=class_handle_binding_sha256(
            CLASSES
        ),
        expected_method_lock_sha256=manifest["method_lock_sha256"],
        expected_pre_sign_content_root_sha256=saved[
            "pre_sign_content_root_sha256"
        ],
        allow_pending_outer_joint_seal_development=True,
    )
    assert validated["method_lock_sha256"] == manifest["method_lock_sha256"]
    assert loaded.class_registry == CLASSES
    assert loaded.ground_multiprototypes().shape == (6, 3, 160)
    assert loaded.ground_barycenters().shape == (6, 160)
    assert np.allclose(
        np.linalg.norm(loaded.ground_barycenters(), axis=1), 1.0, atol=1e-6
    )
    decoded = loaded.ground_multiprototypes().astype(np.float64)
    expected_centers = []
    for class_index in range(CLASS_COUNT):
        center = np.sum(
            decoded[class_index]
            * loaded.p_g_weight[class_index].astype(np.float64)[:, None],
            axis=0,
        )
        expected_centers.append(center / np.linalg.norm(center))
    assert np.allclose(
        loaded.ground_barycenters(),
        np.stack(expected_centers),
        atol=1.0e-7,
    )
    assert not loaded.p_g_q.flags.writeable
    assert not loaded.p_g_weight.flags.writeable
    assert not loaded.p_g_radius.flags.writeable
    assert not loaded.ground_barycenters().flags.writeable
    with pytest.raises(TypeError):
        loaded.method_lock["rank"] = 3
    with pytest.raises(FileExistsError):
        save_grb_jp4_cfm_component(output, payload, manifest)


@pytest.mark.parametrize(
    "mutation,error",
    [
        (
            lambda source: source["ground_multiprototypes"].pop(),
            "cover every old class",
        ),
        (
            lambda source: source["ground_multiprototypes"][0][
                "prototypes"
            ].extend(
                copy.deepcopy(
                    source["ground_multiprototypes"][0]["prototypes"]
                )
                * 3
            ),
            "one to three",
        ),
        (
            lambda source: source["ground_multiprototypes"][0]["prototypes"][
                0
            ]["aggregation_receipt"].__setitem__(
                "distinct_physical_sample_count", 1
            ),
            "at least two",
        ),
        (
            lambda source: source["ground_multiprototypes"][0]["prototypes"][
                0
            ]["aggregation_receipt"].__setitem__("aggregation_radius", -0.1),
            "radius must be finite non-negative",
        ),
        (
            lambda source: source["ground_multiprototypes"][0]["prototypes"][
                0
            ]["aggregation_receipt"].__setitem__(
                "aggregation_radius", float("inf")
            ),
            "radius must be finite non-negative",
        ),
        (
            lambda source: source["ground_multiprototypes"][0]["prototypes"][
                0
            ]["aggregation_receipt"].__setitem__(
                "prototype_sha256", "f" * 64
            ),
            "does not bind",
        ),
        (
            lambda source: source["ground_multiprototypes"][0]["prototypes"][
                0
            ]["aggregation_receipt"].__setitem__("member_ids_included", True),
            "semantic drift",
        ),
        (
            lambda source: source.__setitem__("protocol_schema", "legacy"),
            "protocol schema",
        ),
        (
            lambda source: source.__setitem__(
                "receiver_day_mean_schema", "ADV3B02:z_id:unit_l2:160:v1"
            ),
            "mean representation schema",
        ),
        (
            lambda source: source.__setitem__("source_path", "forbidden"),
            "field allowlist",
        ),
    ],
)
def test_f01_ground_coverage_receipts_and_protocol_fail_closed(
    mutation, error: str
) -> None:
    source = copy.deepcopy(_source())
    mutation(source)
    with pytest.raises(ValueError, match=error):
        _build(source=source)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", "phase1_grb_jp4_compact_component_v1"),
        ("ground_old_multiprototype_enabled", False),
        ("target25_release_authorized", True),
        ("query_fit_access", True),
        ("rank", 3),
        (
            "qknn_lock_sha256_by_k",
            {"1": "d" * 64, "5": "e" * 64, "10": "bad"},
        ),
        (
            "qknn_lock_sha256_by_k",
            {"1": "d" * 64, "5": "d" * 64, "10": "f" * 64},
        ),
    ],
)
def test_f01_method_lock_is_exact_and_cannot_impersonate_r1(
    field: str, value
) -> None:
    lock = _method_lock()
    lock[field] = value
    with pytest.raises(ValueError, match="method lock|SHA256"):
        _build(lock=lock)


def test_f02_degenerate_svd_uses_standard_basis_projection() -> None:
    base = np.eye(RANK, FEATURE_DIM, dtype=np.float64)
    random = np.random.default_rng(19).normal(size=(RANK, RANK))
    rotation, _unused = np.linalg.qr(random)
    first, singular_first = _canonical_svd_rows(
        base, rank=RANK, columns=FEATURE_DIM, field="base"
    )
    second, singular_second = _canonical_svd_rows(
        rotation @ base, rank=RANK, columns=FEATURE_DIM, field="rotated"
    )
    assert np.array_equal(first, second)
    assert np.array_equal(first, base)
    assert np.allclose(singular_first, singular_second, atol=0.0)


def test_f02_domain_row_permutation_preserves_quantized_factors() -> None:
    source = _source()
    permuted = copy.deepcopy(source)
    order = np.asarray([3, 0, 6, 2, 5, 1, 4])
    permuted["receiver_day_means"] = source["receiver_day_means"][:, order]
    permuted["receiver_day_mask"] = source["receiver_day_mask"][:, order]
    permuted["receiver_day_physical_counts"] = source[
        "receiver_day_physical_counts"
    ][:, order]
    payload_a, manifest_a = _build(source=source)
    payload_b, manifest_b = _build(source=permuted)
    for field in (
        "l_g_q",
        "l_g_scale",
        "r_q",
        "r_scale",
        "direction_energy_a",
        "delta_q",
        "tau_q",
    ):
        assert np.array_equal(payload_a[field], payload_b[field]), field
    # Generation evidence still binds the observed domain ordering/mask.
    assert (
        manifest_a["source_aggregate_generation_digest_sha256"]
        != manifest_b["source_aggregate_generation_digest_sha256"]
    )


def test_f01_saved_component_tampering_fails_closed(tmp_path: Path) -> None:
    payload, manifest = _build()
    output = tmp_path / "component"
    save_grb_jp4_cfm_component(output, payload, manifest)

    (output / "unexpected.txt").write_text("x", encoding="ascii")
    with pytest.raises(ValueError, match="directory member allowlist"):
        validate_grb_jp4_cfm_component(output)
    (output / "unexpected.txt").unlink()

    manifest_path = output / "manifest.json"
    tampered = json.loads(manifest_path.read_text(encoding="utf-8"))
    tampered["protocol_schema"] = "legacy"
    manifest_path.write_text(
        json.dumps(tampered, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _manifest_sha(manifest_path)
    with pytest.raises(ValueError, match="profile/method/protocol/feature"):
        validate_grb_jp4_cfm_component(output)


@pytest.mark.parametrize(
    "mutation,error",
    [
        (
            lambda payload: payload["p_g_weight"].__setitem__((0, 0), 0.5),
            "persisted weight",
        ),
        (
            lambda payload: payload["p_g_radius"].__setitem__((0, 0), -0.5),
            "present metadata",
        ),
        (
            lambda payload: payload[
                "p_g_quantization_certificate_sha256"
            ].__setitem__((0, 0), b"f" * 64),
            "quantization certificate mismatch",
        ),
    ],
)
def test_f01_ground_weight_radius_and_quantization_certificate_tamper_fail(
    tmp_path: Path, mutation, error: str
) -> None:
    payload, manifest = _build()
    output = tmp_path / "component"
    save_grb_jp4_cfm_component(output, payload, manifest)
    _rewrite_payload_and_reseal(output, mutation)
    with pytest.raises(ValueError, match=error):
        validate_grb_jp4_cfm_component(output)


def test_f01_registry_tamper_fails_even_after_rehashing_container(
    tmp_path: Path,
) -> None:
    payload, manifest = _build()
    output = tmp_path / "component"
    save_grb_jp4_cfm_component(output, payload, manifest)
    _rewrite_payload_and_reseal(
        output,
        lambda altered: altered["class_registry"].__setitem__(
            0, "tx-tampered"
        ),
    )
    with pytest.raises(
        ValueError,
        match="quantization certificate mismatch|class registry binding",
    ):
        validate_grb_jp4_cfm_component(output)
