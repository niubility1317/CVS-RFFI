from __future__ import annotations

import copy

import pytest

from cvsrffi.phase2_runtime_contract import PHASE2_FULL_CONTRACT
from cvsrffi.somph_runtime_request import (
    SOMPH_APPLY_BATCH_SIZE,
    SOMPH_APPLY_REQUEST_SCHEMA,
    SOMPH_ENROLLMENT_REQUEST_SCHEMA,
    SomphRuntimeRequestError,
    validate_somph_apply_request,
    validate_somph_enrollment_request,
    validate_somph_runtime_request,
)


def _enrollment_request() -> dict[str, object]:
    return {
        "schema": SOMPH_ENROLLMENT_REQUEST_SCHEMA,
        "package_seal_sha256": "1" * 64,
        "head_output_leaf": "head_capsule.npz",
        "device": "cpu",
        "support_batch_size": 32,
        **PHASE2_FULL_CONTRACT,
    }


def _apply_request() -> dict[str, object]:
    return {
        "schema": SOMPH_APPLY_REQUEST_SCHEMA,
        "package_seal_sha256": "1" * 64,
        "head_capsule_sha256": "2" * 64,
        "head_enrollment_binding_sha256": "3" * 64,
        "row_handle": "row_" + "4" * 64,
        "row_manifest_sha256": "5" * 64,
        "prediction_output_leaf": "predictions",
        "device": "cuda:0",
        **PHASE2_FULL_CONTRACT,
    }


def test_enrollment_request_is_exact_and_returns_detached_copy() -> None:
    request = _enrollment_request()
    safe = validate_somph_enrollment_request(request)
    assert safe == request
    assert safe is not request
    request["support_batch_size"] = 1
    assert safe["support_batch_size"] == 32


def test_apply_request_is_exact_and_batch_is_not_request_controlled() -> None:
    request = _apply_request()
    safe = validate_somph_apply_request(request)
    assert safe == request
    assert safe is not request
    assert SOMPH_APPLY_BATCH_SIZE == 1
    runtime_controls = set(safe) - set(PHASE2_FULL_CONTRACT)
    assert not any("batch" in key for key in runtime_controls)


def test_dispatch_accepts_only_the_two_exact_schemas() -> None:
    assert validate_somph_runtime_request(_enrollment_request())["schema"] == (
        SOMPH_ENROLLMENT_REQUEST_SCHEMA
    )
    assert validate_somph_runtime_request(_apply_request())["schema"] == (
        SOMPH_APPLY_REQUEST_SCHEMA
    )
    request = _apply_request()
    request["schema"] = "cvs.phase2.somph.runtime_request.v1"
    with pytest.raises(SomphRuntimeRequestError, match="schema_version"):
        validate_somph_runtime_request(request)


@pytest.mark.parametrize(
    "value",
    ["row-abc", "row_" + "A" * 64, "SOMPH_rx8-8_n5"],
)
def test_apply_requires_an_opaque_row_handle(value: str) -> None:
    request = _apply_request()
    request["row_handle"] = value
    with pytest.raises(SomphRuntimeRequestError, match="row_handle"):
        validate_somph_apply_request(request)


@pytest.mark.parametrize(
    "field",
    [
        "truth",
        "query_role",
        "old_boundary",
        "new_boundary",
        "new_class_count",
        "query_count",
        "query_class_quota",
        "query_order",
        "dataset_path",
        "cache_path",
        "raw_path",
        "clean_path",
        "entrypoint",
    ],
)
@pytest.mark.parametrize("factory", [_enrollment_request, _apply_request])
def test_all_forbidden_or_extra_fields_fail_closed(factory, field: str) -> None:
    request = factory()
    request[field] = "x"
    with pytest.raises(SomphRuntimeRequestError, match="request_schema"):
        validate_somph_runtime_request(request)


@pytest.mark.parametrize(
    "field,value",
    [
        ("head_output_leaf", "truth.npz"),
        ("head_output_leaf", "new_boundary.npz"),
        ("head_output_leaf", "clean_cache.npz"),
        ("head_output_leaf", "runner_entrypoint.npz"),
    ],
)
def test_enrollment_rejects_forbidden_semantics_hidden_in_leaf(
    field: str, value: str
) -> None:
    request = _enrollment_request()
    request[field] = value
    with pytest.raises(SomphRuntimeRequestError, match="forbidden_value"):
        validate_somph_enrollment_request(request)


@pytest.mark.parametrize(
    "value",
    [
        "query_order.json",
        "old_predictions",
        "dataset_cache",
        "raw_clean_output",
        "score_entrypoint",
        "querycount",
        "main.py",
        "run_predictor.ps1",
    ],
)
def test_apply_rejects_forbidden_semantics_hidden_in_leaf(value: str) -> None:
    request = _apply_request()
    request["prediction_output_leaf"] = value
    with pytest.raises(SomphRuntimeRequestError, match="forbidden_value"):
        validate_somph_apply_request(request)


@pytest.mark.parametrize(
    "field",
    [
        "batch_size",
        "apply_batch_size",
        "query_batch_size",
        "query_count",
    ],
)
def test_apply_has_no_batch_or_query_composition_knob(field: str) -> None:
    request = _apply_request()
    request[field] = 1
    with pytest.raises(SomphRuntimeRequestError, match="request_schema"):
        validate_somph_apply_request(request)


@pytest.mark.parametrize("field", list(PHASE2_FULL_CONTRACT))
def test_contract_fields_are_present_and_immutable(field: str) -> None:
    request = _apply_request()
    request[field] = (
        True if PHASE2_FULL_CONTRACT[field] is False else "not-the-locked-policy"
    )
    with pytest.raises(SomphRuntimeRequestError, match="phase2_contract"):
        validate_somph_apply_request(request)


@pytest.mark.parametrize(
    "field",
    [
        "package_seal_sha256",
        "head_capsule_sha256",
        "head_enrollment_binding_sha256",
    ],
)
@pytest.mark.parametrize(
    "value",
    [
        "a" * 63,
        "A" * 64,
        "g" * 64,
        7,
    ],
)
def test_apply_requires_strict_lowercase_sha256(field: str, value: object) -> None:
    request = _apply_request()
    request[field] = value
    with pytest.raises(SomphRuntimeRequestError, match="sha256"):
        validate_somph_apply_request(request)


@pytest.mark.parametrize(
    "value",
    [
        "../head.npz",
        "dir/head.npz",
        r"dir\head.npz",
        ".hidden",
        "..",
        "",
        "a" * 129,
    ],
)
def test_output_must_be_one_safe_leaf(value: str) -> None:
    request = _enrollment_request()
    request["head_output_leaf"] = value
    with pytest.raises(SomphRuntimeRequestError, match="output_leaf"):
        validate_somph_enrollment_request(request)


@pytest.mark.parametrize("value", [0, -1, 257, True, 1.0, "1"])
def test_support_batch_size_is_a_bounded_integer(value: object) -> None:
    request = _enrollment_request()
    request["support_batch_size"] = value
    with pytest.raises(SomphRuntimeRequestError, match="support_batch_size"):
        validate_somph_enrollment_request(request)


@pytest.mark.parametrize("value", ["cuda", "gpu:0", "cuda:-1", "cuda:32", 0])
def test_device_is_not_a_path_or_free_form_control(value: object) -> None:
    request = _apply_request()
    request["device"] = value
    with pytest.raises(SomphRuntimeRequestError, match="device"):
        validate_somph_apply_request(request)


def test_safe_copy_drops_mapping_subclass_behavior() -> None:
    class HostileDict(dict):
        pass

    request = HostileDict(copy.deepcopy(_apply_request()))
    safe = validate_somph_apply_request(request)
    assert type(safe) is dict
