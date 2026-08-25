from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = PROJECT_ROOT / "code" / "scripts" / "launch_phase1_ccoi_pa_m21_20260825.sh"


def test_launcher_owns_only_new_m21_run_and_smoke_precedes_full_audit():
    text = LAUNCHER.read_text(encoding="utf-8")

    assert "audit_phase1_ccoi_pa_m21.py" in text
    assert "audit_phase1_ccoi_pa_v2.py" not in text
    assert "train_phase1_ccoi_pa.py" not in text
    assert "PHASE1_CCOI_PA_M21_THETA_TRANSFER_AUDIT_S20260824_20260825C" in text
    assert "--legacy_migration_mode" in text
    assert text.index("--smoke_only") < text.index("--factor_steps 800")


def test_launcher_refuses_all_output_and_log_collisions_and_checks_fourteen_artifacts():
    text = LAUNCHER.read_text(encoding="utf-8")

    assert '[[ ! -e "${OUT_ROOT}" ]]' in text
    assert '[[ ! -e "${SMOKE_ROOT}" ]]' in text
    assert '[[ ! -e "${LOG_ROOT}/${RUN_ID}.out" ]]' in text
    assert '[[ ! -e "${LOG_ROOT}/${RUN_ID}_smoke.out" ]]' in text
    assert "split_manifest.json" in text
    assert "gate_calibration_summary.json" in text
    assert "gate_audit_summary.json" in text
    assert "decision_manifest.json" in text
    assert "final_report.md" in text
