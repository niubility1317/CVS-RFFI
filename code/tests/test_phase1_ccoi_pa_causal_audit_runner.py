import sys
from pathlib import Path

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
LAUNCHER = CODE_ROOT / "scripts" / "launch_phase1_ccoi_pa_v2_causal_audit_20260825.sh"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from audit_phase1_ccoi_pa_v2 import (  # noqa: E402
    build_arg_parser,
    evaluate_stop_rules,
    fit_knn_probe,
    fit_torch_probe,
    run_holdout_factorization,
    validate_audit_output_root,
    validate_sidecar_payload,
    run,
)


def _args(tmp_path, output_name="new"):
    return build_arg_parser().parse_args(
        [
            "--output_dir",
            str(tmp_path / output_name),
            "--checkpoint",
            "base.pt",
            "--sidecar",
            "sidecar.pth",
            "--wisig_pkl",
            "wisig.pkl",
        ]
    )


def test_runner_freezes_fit_and_evaluation_roles(tmp_path):
    args = _args(tmp_path)

    assert args.fit_role == "L_s"
    assert args.eval_role == "V_select"
    assert args.target_or_query_access is False


def test_runner_requires_c4_v2_sidecar():
    validate_sidecar_payload(
        {
            "schema": "cvs.phase1.ccoi_pa_sidecar.v2",
            "row": "C4",
            "sample_level_source_state_included": False,
        }
    )
    with pytest.raises(ValueError, match="C4"):
        validate_sidecar_payload({"schema": "cvs.phase1.ccoi_pa_sidecar.v2", "row": "C3"})
    with pytest.raises(ValueError, match="schema"):
        validate_sidecar_payload({"schema": "old", "row": "C4"})
    with pytest.raises(ValueError, match="sample-level"):
        validate_sidecar_payload(
            {
                "schema": "cvs.phase1.ccoi_pa_sidecar.v2",
                "row": "C4",
                "sample_level_source_state_included": True,
            }
        )


def test_runner_refuses_existing_output(tmp_path):
    args = _args(tmp_path, output_name="existing")
    Path(args.output_dir).mkdir()

    with pytest.raises(FileExistsError, match="overwrite"):
        validate_audit_output_root(args)


def test_stop_rules_use_preregistered_effect_sizes():
    verdict = evaluate_stop_rules(
        {
            "q_tx_normalized_gain": 0.11,
            "q_rx_normalized_gain": 0.02,
            "negative_anchor_coverage": 0.90,
            "h2_vs_h0_relative_gain": 0.08,
            "h2_vs_shuffle_relative_gain": 0.07,
            "h2_vs_other_tx_relative_gain": 0.06,
            "h2_vs_h0_ci_low": 0.01,
            "h2_vs_shuffle_ci_low": 0.01,
            "h2_vs_other_tx_ci_low": 0.01,
            "cross_rx_stability_pass": True,
            "cross_day_stability_pass": True,
            "source_leo_oracle_gain_pp": 0.40,
            "rescue_minus_harm": 3,
        }
    )

    assert verdict["promotable"] is False
    assert "Q_TX_LEAKAGE" in verdict["stop_reasons"]


def test_synthetic_smoke_writes_all_expected_artifacts(tmp_path):
    args = _args(tmp_path, output_name="smoke")
    args.synthetic_smoke = True

    assert run(args) == 0

    expected = {
        "protocol_and_smoke.json",
        "feature_audit.json",
        "probe_audit.json",
        "pair_geometry.json",
        "holdout_factorization.json",
        "complementarity.json",
        "audit_manifest.json",
    }
    assert {path.name for path in Path(args.output_dir).iterdir()} == expected


def test_mlp_and_knn_probes_recover_a_separable_source_label():
    train_x = torch.tensor([[-2.0], [-1.0], [-0.5], [0.5], [1.0], [2.0]])
    train_y = torch.tensor([0, 0, 0, 1, 1, 1])
    eval_x = torch.tensor([[-1.5], [-0.75], [0.75], [1.5]])
    eval_y = torch.tensor([0, 0, 1, 1])

    eval_groups = torch.tensor([[0], [0], [1], [1]])
    mlp = fit_torch_probe(
        train_x,
        train_y,
        eval_x,
        eval_y,
        steps=120,
        seed=9,
        hidden_dim=8,
        eval_groups=eval_groups,
        bootstrap_resamples=50,
    )
    knn = fit_knn_probe(
        train_x,
        train_y,
        eval_x,
        eval_y,
        neighbors=3,
        eval_groups=eval_groups,
        bootstrap_resamples=50,
    )

    assert mlp["balanced_accuracy"] >= 0.99
    assert knn["balanced_accuracy"] >= 0.99
    assert mlp["normalized_gain"] >= 0.98
    assert mlp["normalized_gain_ci95_low"] >= 0.98


def test_holdout_factorization_detects_theta_information():
    generator = torch.Generator().manual_seed(3)

    def make_features(count):
        theta = torch.randn(count, 2, generator=generator)
        return {
            "q_holdout": torch.zeros(count, 1),
            "support_theta": theta,
            "heldout_target": (2.0 * theta[:, :1] - theta[:, 1:2]),
            "tx": torch.arange(count).remainder(2),
            "receiver": torch.arange(count).remainder(3),
            "day": torch.arange(count).remainder(2),
        }

    result = run_holdout_factorization(
        make_features(96),
        make_features(48),
        device=torch.device("cpu"),
        steps=160,
        batch_size=32,
        seed=11,
        bootstrap_resamples=50,
    )

    assert result["rows"]["H2"]["nmse"] < result["rows"]["H0"]["nmse"]
    assert result["comparisons"]["h2_vs_h0"]["relative_gain"] > 0.5
    assert result["hr_common_cross_fitted"] is True


def test_launcher_runs_only_the_new_immutable_causal_audit():
    text = LAUNCHER.read_text(encoding="utf-8")

    assert "audit_phase1_ccoi_pa_v2.py" in text
    assert "train_phase1_ccoi_pa.py" not in text
    assert "--rows C0,C1,C2,C3,C4" not in text
    assert "[[ ! -e \"${OUT_ROOT}\" ]]" in text
    assert "[[ ! -e \"${LOG_ROOT}/${RUN_ID}.out\" ]]" in text
    assert "[[ ! -e \"${LOG_ROOT}/${RUN_ID}_smoke.out\" ]]" in text
    assert text.index("--smoke_only") < text.index("--probe_steps 400")
