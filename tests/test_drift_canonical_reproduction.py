from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def test_canonical_drift_launcher_fixes_mean_v1_protocol():
    proc = subprocess.run(
        [
            "bash",
            "code/scripts/launch_drift_paper_reproduction.sh",
            "--dry-run",
            "--run-id",
            "tmp_drift_canonical",
            "--gpu-ids",
            "0,1,2,3,4",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, out
    assert out.count("[JOB]") == 5
    assert "version=mean_v1" in out
    assert "DRIFT_PAPER_EVAL_LAST_N=1" in out
    assert "DRIFT_BATCH_SIZE=256" in out
    assert "DRIFT_PAPER_SAMPLE_STRATEGY=random" in out
    assert "DRIFT_WISIG_RMS_NORMALIZE=0" in out
    assert "DRIFT_MSE_REDUCTION=mean" in out
    assert "DRIFT_MSE_CAP=0" in out
