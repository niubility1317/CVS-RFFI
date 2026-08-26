from pathlib import Path


LAUNCHER = Path(__file__).parents[1] / "scripts" / "launch_phase1_jmrs02_j1_20260826.sh"


def test_launcher_is_non_overwriting_and_runs_smoke_before_formal_scoring():
    text = LAUNCHER.read_text(encoding="utf-8")
    assert '[[ -e "${OUTPUT_ROOT}" || -e "${SMOKE_ROOT}" ]]' in text
    assert text.count("audit_phase1_jmrs02_j1.py") == 2
    assert text.index("--smoke_only") < text.index("score_phase1_jmrs02_j1.py")
    assert 'ROWS="B0,RZ0,RZ1,RX1,D1P,P0"' in text
    assert "RDP" not in text


def test_launcher_uses_one_fixed_gpu_and_no_process_kill():
    text = LAUNCHER.read_text(encoding="utf-8")
    assert text.count("--device cuda:0") == 2
    assert "pkill" not in text
    assert "killall" not in text
