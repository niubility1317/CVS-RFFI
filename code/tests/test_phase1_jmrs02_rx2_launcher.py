from pathlib import Path


LAUNCHER = Path(__file__).parents[1] / "scripts" / "launch_phase1_jmrs02_rx2_20260826.sh"


def test_rx2_launcher_is_focused_smoke_first_and_non_overwriting():
    text = LAUNCHER.read_text(encoding="utf-8")
    assert 'ROWS="B0,RX0,RX2"' in text
    assert "--focused_rx2" in text
    assert text.count("audit_phase1_jmrs02_j1.py") == 2
    assert text.index("--smoke_only") < text.index("score_phase1_jmrs02_rx2.py")
    assert '[[ -e "${OUTPUT_ROOT}" || -e "${SMOKE_ROOT}" ]]' in text
    assert 'DEVICE="${DEVICE:-cuda:1}"' in text
    assert "--learning_rate 3e-4" in text  # runner applies the registered 0.10 RX factor
