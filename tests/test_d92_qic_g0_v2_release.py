from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RELEASE_ROOT = (
    REPO_ROOT
    / "automation_reports"
    / "CV-SincNet"
    / "d92_e0_full_d42_qic_g0_k10_20260817_v2"
)


def test_v2_launcher_does_not_reject_its_own_redirected_driver_files() -> None:
    launch = (RELEASE_ROOT / "launch.sh").read_text(encoding="utf-8")

    assert "RUN_ID=d92_e0_full_d42_qic_g0_k10_20260817_v2" in launch
    assert 'test ! -e "$SOURCE_ROOT"' in launch
    assert 'test ! -e "$RUN_ROOT"' in launch
    assert 'test ! -e "$LOG_ROOT"' in launch
    assert 'test ! -e "$RUNS/$LAUNCH_BASENAME.out"' not in launch
    assert 'test ! -e "$RUNS/$LAUNCH_BASENAME.err"' not in launch
