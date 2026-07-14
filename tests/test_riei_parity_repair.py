from pathlib import Path
import subprocess

import torch


ROOT = Path(__file__).resolve().parents[1]


def test_riei_optimizer_supports_paper_gradient_descent_and_adam_control():
    from baselines.riei_fd.train_cvs import build_riei_optimizer

    layer = torch.nn.Linear(2, 2)
    sgd = build_riei_optimizer("sgd", layer.parameters(), lr=1e-4, momentum=0.0)
    adam = build_riei_optimizer("adam", layer.parameters(), lr=1e-4)
    assert isinstance(sgd, torch.optim.SGD)
    assert sgd.defaults["momentum"] == 0.0
    assert isinstance(adam, torch.optim.Adam)


def test_riei_parity_launcher_exposes_eight_controlled_candidates():
    proc = subprocess.run(
        ["bash", "code/scripts/launch_riei_parity_repair_matrix_20260714.sh", "--dry-run"],
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
    assert out.count("[JOB]") == 8
    assert "RIEI_PAPER_EVAL_LAST_N=5" in out
    assert "RIEI_WISIG_RMS_NORMALIZE=0" in out
    assert "RIEI_OPTIMIZER=sgd" in out
    assert "P08_adam_sum_rms_fixopt_control" in out
