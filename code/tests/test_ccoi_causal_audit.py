import sys
from pathlib import Path

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.ccoi_causal_audit import (  # noqa: E402
    build_factor_indices,
    complementarity_table,
    factorized_holdout_metrics,
    group_paired_bootstrap,
    normalized_energy_fit_score,
    pair_relation_counts,
    pair_relation_sweep,
    token_code_audit,
)


def test_energy_fit_score_uses_energy_normalization_without_r_squared_claim():
    assert normalized_energy_fit_score(0.12593) == pytest.approx(0.87407)


def test_token_code_audit_separates_token_hard_from_packet_dominant():
    probabilities = torch.tensor(
        [
            [[0.9, 0.1], [0.4, 0.6]],
            [[0.8, 0.2], [0.3, 0.7]],
        ]
    )

    result = token_code_audit(probabilities)

    assert result["token_hard_observed"] == 2
    assert result["packet_dominant_observed"] == 1
    assert result["codes_per_packet_mean"] == pytest.approx(2.0)
    assert result["token_hard_histogram"] == [2, 2]


def test_pair_relation_counts_are_global_and_include_cross_tx_negatives():
    q = torch.tensor([[1.0, 0.0], [0.99, 0.10], [1.0, 0.0]])

    result = pair_relation_counts(
        q,
        torch.tensor([0, 0, 1]),
        torch.tensor([0, 1, 0]),
        min_cosine=0.90,
    )

    assert result["same_tx_cross_rx_matched"] == 1
    assert result["same_tx_cross_rx_total"] == 1
    assert result["cross_tx_same_rx_matched"] == 1
    assert result["cross_tx_same_rx_total"] == 1


def test_pair_relation_sweep_matches_single_threshold_counts():
    q = torch.tensor([[1.0, 0.0], [0.99, 0.10], [1.0, 0.0]])
    tx = torch.tensor([0, 0, 1])
    receiver = torch.tensor([0, 1, 0])

    sweep = pair_relation_sweep(q, tx, receiver, thresholds=(0.90, 0.999))

    assert sweep["0.900"]["same_tx_cross_rx_matched"] == 1
    assert sweep["0.999"]["same_tx_cross_rx_matched"] == 0


def test_complementarity_reports_rescue_harm_and_oracle():
    result = complementarity_table(
        base_prediction=torch.tensor([0, 1, 1, 0]),
        side_prediction=torch.tensor([0, 0, 1, 1]),
        truth=torch.tensor([0, 0, 1, 0]),
    )

    assert result["base_wrong_side_correct"] == 1
    assert result["base_correct_side_wrong"] == 1
    assert result["both_correct"] == 2
    assert result["oracle_accuracy"] == pytest.approx(1.0)


def test_factorized_holdout_metrics_use_one_common_target_energy():
    target = torch.tensor([[1.0, 1.0]])
    predictions = {
        "H0": torch.tensor([[0.0, 1.0]]),
        "H2": torch.tensor([[1.0, 1.0]]),
    }

    result = factorized_holdout_metrics(predictions, target)

    assert result["target_energy"] == pytest.approx(2.0)
    assert result["rows"]["H0"]["nmse"] == pytest.approx(0.5)
    assert result["rows"]["H0"]["normalized_energy_fit_score"] == pytest.approx(0.5)
    assert result["rows"]["H2"]["nmse"] == pytest.approx(0.0)


def test_group_paired_bootstrap_resamples_groups_not_rows():
    result = group_paired_bootstrap(
        reference_error=torch.ones(4),
        candidate_error=torch.zeros(4),
        groups=torch.tensor([[0, 0], [0, 0], [1, 1], [1, 1]]),
        resamples=100,
        seed=7,
    )

    assert result["relative_gain"] == pytest.approx(1.0)
    assert result["ci95_low"] == pytest.approx(1.0)
    assert result["ci95_high"] == pytest.approx(1.0)


def test_factor_indices_enforce_requested_tx_receiver_and_day_relations():
    tx = torch.tensor([0, 0, 0, 1, 1, 1])
    receiver = torch.tensor([0, 1, 0, 0, 1, 0])
    day = torch.tensor([0, 0, 1, 0, 0, 1])

    indices = build_factor_indices(tx, receiver, day, seed=5)

    valid_h4 = indices["H4"] >= 0
    valid_h5 = indices["H5"] >= 0
    valid_h6 = indices["H6"] >= 0
    assert torch.all(tx[indices["H4"][valid_h4]] != tx[valid_h4])
    assert torch.all(tx[indices["H5"][valid_h5]] == tx[valid_h5])
    assert torch.all(receiver[indices["H5"][valid_h5]] != receiver[valid_h5])
    assert torch.all(tx[indices["H6"][valid_h6]] == tx[valid_h6])
    assert torch.all(day[indices["H6"][valid_h6]] != day[valid_h6])


def test_factor_indices_choose_nearest_q_within_each_relation():
    tx = torch.tensor([0, 0, 0, 1])
    receiver = torch.tensor([0, 1, 2, 0])
    day = torch.tensor([0, 0, 0, 0])
    q = torch.tensor([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0], [1.0, 0.0]])

    indices = build_factor_indices(tx, receiver, day, seed=5, q=q)

    assert int(indices["H5"][0]) == 1
    assert int(indices["H4"][0]) == 3


def test_metric_functions_reject_nonfinite_or_misaligned_inputs():
    with pytest.raises(ValueError, match="finite"):
        normalized_energy_fit_score(float("nan"))
    with pytest.raises(ValueError, match="same number"):
        complementarity_table(torch.tensor([0]), torch.tensor([0, 1]), torch.tensor([0]))
