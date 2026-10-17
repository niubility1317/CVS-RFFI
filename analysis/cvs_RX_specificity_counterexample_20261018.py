"""Check the scope of a historical RX-specificity gate, without real data or NN.

This does not change the historical decision or reopen a closed candidate.
The constructed nuisance is known from its generating equation, independently
of the historical estimator. Sharing a nuisance value does not erase an RX effect.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.cvs_iq_circularity16.core import analyze_cells


def additive_cells(receiver_means):
    tx = np.array([-.003, -.002, -.001, .001, .002, .003])
    day = np.array([-.002, 0., .002])
    return np.asarray(receiver_means)[:, None, None] + day[None, :, None] + tx[None, None, :]


def run():
    # All cases use every RX, TX and day. There is no model training or held RX.
    stable = {
        "distinct_RX": additive_cells([.02, .07, .12, .17]),
        "shared_RX_groups": additive_cells([.03, .03, .12, .12]),
        "near_shared_RX_groups": additive_cells([.0299, .0301, .1199, .1201]),
    }
    tx_only = additive_cells([.1, .1, .1, .1])
    group = np.array([-.045, -.045, .045, .045])[:, None, None]
    unstable_day = .1 + group * np.array([-1., 0., 1.])[None, :, None] + np.zeros((4, 3, 6))
    unstable_tx = .1 + group * np.array([-1., -1., -1., 1., 1., 1.])[None, None, :] + np.zeros((4, 3, 6))
    cases = dict(stable, TX_only=tx_only, RX_day_reversal=unstable_day, RX_TX_reversal=unstable_tx)
    results = {}
    for name, cells in cases.items():
        if cells.shape != (4, 3, 6) or not np.isfinite(cells).all() or not ((cells > 0) & (cells < 1)).all():
            raise AssertionError("Synthetic scalar geometry changed")
        # Odd/even observations exactly repeat the known scalar. No IQ is made.
        value = analyze_cells(cells, cells.copy(), cells.copy())
        results[name] = {key: value[key] for key in (
            "decision", "conditions", "mean_risk_gain", "baseline_cell_risk",
            "RX_cell_risk", "gain_over_best_wrong_RX", "positive_folds",
            "min_TX_split_center_cosine", "RX_center_repeatability_ratio",
        )}
        if name in stable:
            # Independently known RX contribution from the generating equation.
            receiver_effect = cells.mean((1, 2)) - cells.mean()
            if np.var(receiver_effect) <= 0 or value["mean_risk_gain"] < .98:
                raise AssertionError("Constructed strong transferable RX effect lost")
        results[name]["known_stable_additive_RX_effect"] = name in stable

    if results["distinct_RX"]["decision"] != "PASS":
        raise AssertionError("Distinct RX positive control must pass the original gate")
    for name in ("shared_RX_groups", "near_shared_RX_groups"):
        failed = [key for key, passed in results[name]["conditions"].items() if not passed]
        if failed != ["above_best_wrong_RX"]:
            raise AssertionError("Counterexample must fail only the RX-specificity condition")
    for name in ("TX_only", "RX_day_reversal", "RX_TX_reversal"):
        if results[name]["decision"] != "FAIL":
            raise AssertionError("Historical gate must still reject negative controls")

    # Exact population identity: swapping within equal-valued RX groups changes
    # neither the nuisance nor its prediction. A positive uniqueness margin is
    # therefore impossible, even with arbitrarily many noiseless observations.
    effect = np.array([.03, .03, .12, .12])
    derangement = np.array([1, 0, 3, 2])
    if any(i == j for i, j in enumerate(derangement)) or not np.array_equal(effect, effect[derangement]):
        raise AssertionError("Population derangement is not nuisance-preserving")

    return dict(
        status="VERIFIED_ANALYTIC_COUNTEREXAMPLE",
        hypothesis="Beating every wrong RX assignment is necessary for a stable transferable RX-associated nuisance",
        hypothesis_supported=False,
        conclusion="The uniqueness margin can reject a known stable shared RX nuisance; it is not a necessary condition for RX-associated variation.",
        limitation="Does not identify actual RX hardware, validate an IQ operator, establish label preservation, or prove recognition benefit.",
        historical_verdict_changed=False,
        closed_candidate_promoted=False,
        new_candidate_launched=False,
        source_or_target_data_read=False,
        synthetic_scalar_cells=len(cases) * 72,
        synthetic_IQ=0, real_IQ=0, CP=0, NN=0, teacher_NN=0, gradients=0, updates=0,
        results=results,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    args = parser.parse_args()
    result = run()
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        path = Path(args.output)
        if path.exists():
            raise FileExistsError(path)
        path.write_text(text, encoding="utf8")
    print(text)
