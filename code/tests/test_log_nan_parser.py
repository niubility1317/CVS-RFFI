import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from scripts.collect_accept_domain_v2 import classify_log_nan_lines, is_valid_effective_candidate  # noqa: E402


def test_log_nan_parser_separates_skipped_placeholder_aux_and_loss_nan():
    text = """
    [TEST] overall_tx=nan% (0/0)
    [GRAD] aux=nan domain_aux=nan
    loss=nan
    Traceback (most recent call last)
    """

    summary = classify_log_nan_lines(text)

    assert summary["skipped_test_placeholder"] == 1
    assert summary["aux_grad_telemetry"] == 1
    assert summary["real_loss_nan"] == 1
    assert summary["fatal_nan"] >= 1


def test_valid_candidate_filter_excludes_aborted_t_tail_artifacts():
    assert is_valid_effective_candidate("T15_LATE60_TAILHI_E260") is True
    assert is_valid_effective_candidate("R31_Q2_EXAMPLE_E280") is True
    assert is_valid_effective_candidate("T16_ABORTED_ARTIFACT") is False
    assert is_valid_effective_candidate("T31_ABORTED_ARTIFACT") is False

