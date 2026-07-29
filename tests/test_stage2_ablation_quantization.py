from __future__ import annotations

import inspect

import numpy as np
import pytest

from cvsrffi.stage2_ablation_quantization import (
    F0,
    F1,
    F2,
    F3,
    QUANTIZATION_RECEIPT_SCHEMA,
    RESOURCE_SCHEMA,
    Stage2AblationQuantizationError,
    compile_affine_state,
    decode_affine_state,
    decode_cost,
    quantization_receipt,
    resource_report,
    score_affine_state,
)


BLOCK_SIZES = (3, 2, 3)


def test_public_interfaces_expose_no_data_fit_label_or_truth_input() -> None:
    for function in (
        compile_affine_state,
        decode_affine_state,
        score_affine_state,
        quantization_receipt,
        decode_cost,
        resource_report,
    ):
        names = inspect.signature(function).parameters
        assert not any(
            forbidden in name.lower()
            for name in names
            for forbidden in ("data", "fit", "label", "truth")
        )


@pytest.fixture
def affine_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260729)
    coefficient = rng.normal(0.0, 0.35, size=(5, 8)).astype(np.float32)
    coefficient[0, :3] = 0.0
    bias = rng.normal(0.0, 0.1, size=5).astype(np.float32)
    features = rng.normal(0.0, 0.6, size=(17, 8)).astype(np.float32)
    return coefficient, bias, features


@pytest.mark.parametrize(
    ("arm_id", "coefficient_dtypes", "scale_count", "bias_dtype"),
    (
        (F0, (np.dtype(np.float32),), 0, np.dtype(np.float32)),
        (F1, (np.dtype(np.float16),), 0, np.dtype(np.float16)),
        (F2, (np.dtype(np.int8),), 1, np.dtype(np.float16)),
        (
            F3,
            (np.dtype(np.int8), np.dtype(np.int8)),
            2,
            np.dtype(np.float16),
        ),
    ),
)
def test_f0_f3_compile_exact_locked_storage_layout(
    affine_inputs: tuple[np.ndarray, np.ndarray, np.ndarray],
    arm_id: str,
    coefficient_dtypes: tuple[np.dtype, ...],
    scale_count: int,
    bias_dtype: np.dtype,
) -> None:
    coefficient, bias, _features = affine_inputs
    state = compile_affine_state(
        coefficient,
        bias,
        arm_id=arm_id,
        block_sizes=BLOCK_SIZES,
    )

    assert tuple(value.dtype for value in state.coefficient_layers) == (
        coefficient_dtypes
    )
    assert len(state.scale_layers) == scale_count
    assert all(value.dtype == np.float16 for value in state.scale_layers)
    assert all(value.shape == (5, 3) for value in state.scale_layers)
    assert state.bias.dtype == bias_dtype
    assert state.has_fp32_coefficient_sidecar is False
    assert all(value.flags.writeable is False for value in state.coefficient_layers)
    assert all(value.flags.writeable is False for value in state.scale_layers)
    assert state.bias.flags.writeable is False

    if arm_id in {F2, F3}:
        assert not any(
            value.dtype == np.float32 for value in state.coefficient_layers
        )
        assert not any(
            value.dtype == np.float32 for value in state.scale_layers
        )


def test_state_bytes_are_exact_and_f0_to_f3_are_storage_ordered(
    affine_inputs: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    coefficient, bias, _features = affine_inputs
    states = {
        arm_id: compile_affine_state(
            coefficient,
            bias,
            arm_id=arm_id,
            block_sizes=BLOCK_SIZES,
        )
        for arm_id in (F0, F1, F2, F3)
    }

    for state in states.values():
        expected = sum(
            value.nbytes
            for value in (
                *state.coefficient_layers,
                *state.scale_layers,
                state.bias,
            )
        )
        assert state.state_bytes == expected
        assert state.state_bytes == (
            state.coefficient_state_bytes
            + state.scale_state_bytes
            + state.bias_state_bytes
        )

    assert states[F1].state_bytes < states[F0].state_bytes
    assert states[F2].state_bytes < states[F1].state_bytes
    assert states[F2].state_bytes < states[F3].state_bytes
    assert states[F3].state_bytes < states[F0].state_bytes


def test_decode_and_score_use_only_compiled_state(
    affine_inputs: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    coefficient, bias, features = affine_inputs
    state = compile_affine_state(
        coefficient, bias, arm_id=F3, block_sizes=BLOCK_SIZES
    )
    decoded_coefficient, decoded_bias = decode_affine_state(state)

    assert decoded_coefficient.dtype == np.float32
    assert decoded_bias.dtype == np.float32
    assert score_affine_state(state, features).shape == (17, 5)
    np.testing.assert_allclose(
        score_affine_state(state, features),
        features @ decoded_coefficient.T + decoded_bias[None, :],
        rtol=0.0,
        atol=0.0,
    )
    assert score_affine_state(state, features[0]).shape == (5,)


def test_f0_is_exact_reference_and_f3_reduces_f2_coefficient_error(
    affine_inputs: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    coefficient, bias, features = affine_inputs
    states = {
        arm_id: compile_affine_state(
            coefficient,
            bias,
            arm_id=arm_id,
            block_sizes=BLOCK_SIZES,
        )
        for arm_id in (F0, F2, F3)
    }
    reference_scores = features @ coefficient.T + bias[None, :]
    np.testing.assert_array_equal(
        score_affine_state(states[F0], features), reference_scores
    )

    decoded_f2, _ = decode_affine_state(states[F2])
    decoded_f3, _ = decode_affine_state(states[F3])
    f2_error = np.max(np.abs(decoded_f2 - coefficient))
    f3_error = np.max(np.abs(decoded_f3 - coefficient))
    assert f3_error <= f2_error


@pytest.mark.parametrize("arm_id", (F0, F1, F2, F3))
def test_quantization_receipt_matches_truth_side_exact_schema(
    affine_inputs: tuple[np.ndarray, np.ndarray, np.ndarray], arm_id: str
) -> None:
    coefficient, bias, features = affine_inputs
    state = compile_affine_state(
        coefficient,
        bias,
        arm_id=arm_id,
        block_sizes=BLOCK_SIZES,
    )
    receipt = quantization_receipt(
        state,
        reference_coefficient=coefficient,
        reference_bias=bias,
        query_features=features,
    )

    assert set(receipt) == {
        "schema",
        "max_logit_abs_error",
        "mean_logit_abs_error",
        "argmax_flip_rate",
        "prediction_agreement_rate",
    }
    assert receipt["schema"] == QUANTIZATION_RECEIPT_SCHEMA
    assert 0.0 <= receipt["mean_logit_abs_error"] <= receipt[
        "max_logit_abs_error"
    ]
    assert 0.0 <= receipt["argmax_flip_rate"] <= 1.0
    assert receipt["prediction_agreement_rate"] == pytest.approx(
        1.0 - receipt["argmax_flip_rate"], abs=0.0
    )
    if arm_id == F0:
        assert receipt["max_logit_abs_error"] == 0.0
        assert receipt["mean_logit_abs_error"] == 0.0
        assert receipt["argmax_flip_rate"] == 0.0


def test_decode_cost_distinguishes_float_single_and_residual_states(
    affine_inputs: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    coefficient, bias, _features = affine_inputs
    costs = {
        arm_id: decode_cost(
            compile_affine_state(
                coefficient,
                bias,
                arm_id=arm_id,
                block_sizes=BLOCK_SIZES,
            )
        )
        for arm_id in (F0, F1, F2, F3)
    }

    assert costs[F0]["scale_multiply_count"] == 0
    assert costs[F0]["storage_to_fp32_cast_count"] == 0
    assert costs[F1]["int8_to_fp32_cast_count"] == 0
    assert costs[F1]["fp16_to_fp32_cast_count"] > 0
    assert costs[F2]["int8_to_fp32_cast_count"] == coefficient.size
    assert costs[F2]["scale_multiply_count"] == coefficient.size
    assert costs[F2]["residual_add_count"] == 0
    assert costs[F3]["int8_to_fp32_cast_count"] == 2 * coefficient.size
    assert costs[F3]["scale_multiply_count"] == 2 * coefficient.size
    assert costs[F3]["residual_add_count"] == coefficient.size


@pytest.mark.parametrize("arm_id", (F0, F1, F2, F3))
def test_resource_report_has_state_decode_and_batch1_latency_fields(
    affine_inputs: tuple[np.ndarray, np.ndarray, np.ndarray], arm_id: str
) -> None:
    coefficient, bias, features = affine_inputs
    state = compile_affine_state(
        coefficient,
        bias,
        arm_id=arm_id,
        block_sizes=BLOCK_SIZES,
    )
    report = resource_report(
        state,
        query_feature=features[0],
        latency_repeats=3,
        latency_warmup=1,
    )

    assert report["schema"] == RESOURCE_SCHEMA
    assert report["arm_id"] == arm_id
    assert report["state_bytes"] == state.state_bytes
    assert report["has_fp32_coefficient_sidecar"] is False
    assert report["decode_cost"] == dict(decode_cost(state))
    assert report["batch1_latency_ms"] >= 0.0
    assert report["batch1_latency_mean_ms"] >= 0.0
    assert report["batch1_latency_repeats"] == 3
    assert report["integer_kernel_used"] is False
    assert report["formal_int8_acceleration_claim_allowed"] is False
    assert report["deployment_claim"] == "storage_compression_only"


@pytest.mark.parametrize(
    ("coefficient", "bias", "arm_id", "block_sizes", "match"),
    (
        (np.zeros((0, 3)), np.zeros(0), F0, (3,), "nonempty"),
        (np.zeros((2, 3)), np.zeros(3), F0, (3,), "bias"),
        (np.zeros((2, 3)), np.zeros(2), "P2-F9", (3,), "unsupported"),
        (np.zeros((2, 3)), np.zeros(2), F2, (2,), "sum"),
        (
            np.array([[np.nan, 0.0], [0.0, 0.0]]),
            np.zeros(2),
            F3,
            (2,),
            "finite",
        ),
    ),
)
def test_compile_rejects_invalid_affine_inputs(
    coefficient: np.ndarray,
    bias: np.ndarray,
    arm_id: str,
    block_sizes: tuple[int, ...],
    match: str,
) -> None:
    with pytest.raises(Stage2AblationQuantizationError, match=match):
        compile_affine_state(
            coefficient,
            bias,
            arm_id=arm_id,
            block_sizes=block_sizes,
        )


def test_query_interface_rejects_wrong_shape_nonfinite_and_non_batch1_profile(
    affine_inputs: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    coefficient, bias, features = affine_inputs
    state = compile_affine_state(
        coefficient, bias, arm_id=F3, block_sizes=BLOCK_SIZES
    )

    with pytest.raises(Stage2AblationQuantizationError, match="shape"):
        score_affine_state(state, np.zeros((2, 7), dtype=np.float32))
    invalid = features.copy()
    invalid[0, 0] = np.inf
    with pytest.raises(Stage2AblationQuantizationError, match="finite"):
        score_affine_state(state, invalid)
    with pytest.raises(Stage2AblationQuantizationError, match="batch-1"):
        resource_report(
            state,
            query_feature=features[:2],
            latency_repeats=1,
            latency_warmup=0,
        )


@pytest.mark.parametrize("arm_id", (F1, F2, F3))
def test_fp16_storage_overflow_fails_closed(arm_id: str) -> None:
    coefficient = np.full((2, 4), 1.0e20, dtype=np.float32)
    bias = np.zeros(2, dtype=np.float32)
    with pytest.raises(Stage2AblationQuantizationError, match="FP16|finite"):
        compile_affine_state(
            coefficient,
            bias,
            arm_id=arm_id,
            block_sizes=(2, 2),
        )
