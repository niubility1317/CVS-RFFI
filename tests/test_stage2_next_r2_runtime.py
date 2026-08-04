from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from cvsrffi import stage2_next_r2_cvfr as cvfr
from cvsrffi import stage2_next_r2_matrix as matrix
from cvsrffi import stage2_next_r2_runtime as runtime


RECEIVERS = tuple(f"rx-{index}" for index in range(7))
CLASSES = tuple(f"class-{index}" for index in range(6))


def _outer_key(k: int = 1) -> matrix.NextR2OuterKey:
    plan = matrix.build_next_r2_proxy24_plan(
        RECEIVERS, CLASSES, source_identity_sha256="5" * 64
    )
    return next(
        matrix.outer_key_from_mapping(item)
        for item in plan["keys"]
        if item["active_k"] == k
    )


def _base_arrays(k: int):
    support_rows: list[np.ndarray] = []
    support_ids: list[str] = []
    support_labels: list[str] = []
    query_rows: list[np.ndarray] = []
    query_ids: list[str] = []
    for class_index, class_id in enumerate(CLASSES):
        for shot in range(k):
            row = np.zeros(cvfr.Z_DIM, dtype=np.float32)
            row[class_index] = np.float32(1.0 + 0.03 * shot)
            row[20 + class_index] = np.float32(0.05 * (shot + 1))
            support_rows.append(row)
            support_ids.append(f"support-{class_id}-{shot}")
            support_labels.append(class_id)
        for query_index in range(matrix.QUERY_PER_CLASS):
            row = np.zeros(cvfr.Z_DIM, dtype=np.float32)
            row[class_index] = np.float32(1.0)
            row[40 + class_index] = np.float32(0.001 * (query_index + 1))
            query_rows.append(row)
            query_ids.append(f"query-{class_id}-{query_index}")
    return (
        np.ascontiguousarray(support_rows, dtype=np.float32),
        tuple(support_labels),
        tuple(support_ids),
        np.ascontiguousarray(query_rows, dtype=np.float32),
        tuple(query_ids),
    )


def _blocks(array: np.ndarray, selected: tuple[str, ...], rows_per_class: int) -> np.ndarray:
    by_class = {
        class_id: array[index * rows_per_class : (index + 1) * rows_per_class]
        for index, class_id in enumerate(CLASSES)
    }
    return np.ascontiguousarray(
        np.concatenate([by_class[item] for item in selected], axis=0), dtype=np.float32
    )


def _id_blocks(values: tuple[str, ...], selected: tuple[str, ...], rows_per_class: int):
    by_class = {
        class_id: values[index * rows_per_class : (index + 1) * rows_per_class]
        for index, class_id in enumerate(CLASSES)
    }
    return tuple(item for class_id in selected for item in by_class[class_id])


def _four_inputs(k: int = 1):
    key = _outer_key(k)
    support, labels, support_ids, query, query_ids = _base_arrays(k)
    states = {}
    for state_id in matrix.STATE_IDS:
        selected = matrix.registered_classes_for_state(key, state_id)
        canonical = _blocks(support, selected, k)
        states[state_id] = runtime.NextR2StateInputs(
            outer_key_id=key.outer_key_id,
            state_id=state_id,
            capsule_id="capsule-fixed",
            split_id="split-fixed",
            active_k=k,
            registered_classes=selected,
            support_canonical=canonical,
            support_phase_plus=canonical.copy(),
            support_phase_minus=canonical.copy(),
            support_labels=_id_blocks(labels, selected, k),
            support_physical_ids=_id_blocks(support_ids, selected, k),
            query_canonical=_blocks(query, selected, matrix.QUERY_PER_CLASS),
            query_physical_ids=_id_blocks(
                query_ids, selected, matrix.QUERY_PER_CLASS
            ),
        )
    return key, states


@pytest.mark.parametrize("k", [1, 5])
def test_four_state_execution_fits_fresh_heads_and_keeps_truth_out(k: int) -> None:
    key, inputs = _four_inputs(k)
    results = runtime.execute_next_r2_outer_key(key, inputs)
    assert tuple(item.state_id for item in results) == matrix.STATE_IDS
    assert len({id(item.bssdg_state) for item in results}) == 4
    for result in results:
        expected_rows = 54 if result.state_id in matrix.REG1_STATES else 45
        assert result.scores.shape == (expected_rows, len(result.registered_classes))
        assert len(result.predictions) == expected_rows
        assert result.receipt["query_truth_input_count"] == 0
        assert result.receipt["query_rows_used_for_fit"] == 0
        assert result.receipt["query_state_updates"] == 0
        assert result.receipt["query_selection_count"] == 0
        assert result.receipt["bssdg_state_input_digest"] == (
            result.bssdg_state.binding.canonical_sha256
        )
        assert "capsule_id" in result.receipt["bssdg_state_input_digest_fields"]
        assert "support_physical_id_root" in result.receipt[
            "bssdg_state_input_digest_fields"
        ]
        assert "UNSPECIFIED" not in str(result.receipt)
        latency = result.receipt["execution_latency_ns"]
        assert latency["bssdg_fit"] >= 0
        assert latency["bssdg_score"] >= 0
        assert latency["excluded_from_cvfr_bssdg_deploy_state_digest"] is True
        if result.state_id in matrix.REG1_STATES:
            assert result.receipt["new_class_metric"] != "NA"
        else:
            assert result.receipt["new_class_metric"] == "NA"
            assert result.receipt["harmonic_metric"] == "NA"
        if result.state_id in matrix.DA1_STATES:
            assert result.cvfr_state is not None
            assert result.cvfr_state.status == cvfr.STATUS_IDENTITY_UNIDENTIFIABLE
            cost = result.receipt["cvfr_transform_cost"]
            assert cost["per_transform_fixed_helmert_h_a_multiplications"] == 160 * 159
            assert cost["per_transform_fixed_helmert_h_a_additions"] == 160 * 158
            assert cost["per_row_scale_multiplications"] == 160
            assert cost["per_row_shift_additions"] == 160
            assert cost["is_end_to_end_transform_total"] is False
            assert latency["cvfr_fit"] >= 0
            assert latency["cvfr_query_transform"] >= 0
        else:
            assert result.cvfr_state is None


def test_base_input_hash_binds_support_labels_but_not_query_truth() -> None:
    _key, inputs = _four_inputs(1)
    value = inputs["DA0_REG1"]
    changed_labels = tuple(reversed(value.support_labels))
    changed = replace(value, support_labels=changed_labels)
    assert changed.base_input_sha256 != value.base_input_sha256
    assert "query_truth" not in value.__dataclass_fields__


def test_da_pairs_must_share_inputs_and_reg0_must_be_exact_reg1_subset() -> None:
    key, inputs = _four_inputs(1)
    drifted = dict(inputs)
    altered = np.array(drifted["DA0_REG0"].query_canonical, copy=True)
    altered[0, 159] = np.float32(0.25)
    drifted["DA0_REG0"] = replace(
        drifted["DA0_REG0"], query_canonical=np.ascontiguousarray(altered)
    )
    drifted["DA1_REG0"] = replace(
        drifted["DA1_REG0"], query_canonical=np.ascontiguousarray(altered)
    )
    with pytest.raises(runtime.NextR2RuntimeError, match="exact retained-only"):
        runtime.validate_four_state_inputs(key, drifted)


@pytest.mark.parametrize(
    ("field", "foreign"),
    (("capsule_id", "foreign-capsule"), ("split_id", "foreign-split")),
)
def test_all_four_states_must_share_capsule_and_split(field: str, foreign: str) -> None:
    key, inputs = _four_inputs(1)
    drifted = dict(inputs)
    drifted["DA0_REG0"] = replace(drifted["DA0_REG0"], **{field: foreign})
    drifted["DA1_REG0"] = replace(drifted["DA1_REG0"], **{field: foreign})
    with pytest.raises(runtime.NextR2RuntimeError, match="one capsule_id and split_id"):
        runtime.validate_four_state_inputs(key, drifted)


def test_sealed_manifest_requires_exact_order_and_all_96_states() -> None:
    plan = matrix.build_next_r2_proxy24_plan(
        RECEIVERS, CLASSES, source_identity_sha256="5" * 64
    )
    artifacts = []
    counter = 1
    for key in plan["keys"]:
        for state_id in matrix.STATE_IDS:
            artifacts.append(
                {
                    "outer_key_id": key["outer_key_id"],
                    "state_id": state_id,
                    "json_path": f"states/{counter}.json",
                    "json_sha256": f"{counter:064x}",
                    "npz_path": f"states/{counter}.npz",
                    "npz_sha256": f"{counter + 1000:064x}",
                    "state_seal_sha256": f"{counter + 2000:064x}",
                }
            )
            counter += 1
    manifest = runtime.build_next_r2_sealed_manifest(plan, artifacts)
    assert manifest["state_prediction_count"] == 96
    assert manifest["truth_opened"] is False
    assert manifest["sealed_before_scoring"] is True
    with pytest.raises(runtime.NextR2RuntimeError):
        runtime.build_next_r2_sealed_manifest(plan, artifacts[:-1])
    swapped = list(artifacts)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    with pytest.raises(runtime.NextR2RuntimeError):
        runtime.build_next_r2_sealed_manifest(plan, swapped)
