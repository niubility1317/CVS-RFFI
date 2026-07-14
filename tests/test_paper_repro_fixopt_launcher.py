from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_fixopt_launcher_keeps_full_table3_and_eight_drift_variants():
    text = (ROOT / "code/scripts/launch_paper_repro_fixopt_matrix_20260714.sh").read_text(encoding="utf-8")

    assert "jobs=20 drift_variants=8 riei_table3=12" in text
    assert text.count('"D0') == 8
    assert text.count('_to_rx1_19|') == 6
    assert text.count('_to_rx14_7|') == 6
    assert "RIEI_LAMBDA_FEATURE_NORM=${riei_feature_norm}" in text
    assert "DRIFT_MSE_CAP=${mse_cap}" in text
    assert "DRIFT_LAMBDA_MSE=${lambda_mse}" in text


def test_paper_scope_patch_defaults_are_backward_compatible_and_exposes_guards():
    text = (ROOT / "code/patches/run_wisig_paper_scope_queue_fixopt_env_20260714.patch").read_text(
        encoding="utf-8"
    )

    assert 'RIEI_LAMBDA_FEATURE_NORM="${RIEI_LAMBDA_FEATURE_NORM:-0}"' in text
    assert 'DRIFT_MSE_CAP="${DRIFT_MSE_CAP:-0}"' in text
    assert 'DRIFT_LAMBDA_MSE="${DRIFT_LAMBDA_MSE:-0.02}"' in text
    assert '--lambda_feature_norm "${RIEI_LAMBDA_FEATURE_NORM}"' in text
    assert '--mse_cap "${DRIFT_MSE_CAP}"' in text
