import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.gate_metrics import binary_reject_metrics, summarize_gate_decisions  # noqa: E402


def test_gate_summary_counts_core_tail_and_reject_separately():
    rows = [
        {"decision": "ACCEPT_KNOWN_CORE"},
        {"decision": "REVIEW_KNOWN_TAIL"},
        {"decision": "REJECT_LOW_DENSITY"},
        {"decision": "REJECT_LOW_DENSITY"},
    ]

    summary = summarize_gate_decisions(rows)

    assert summary["known_auto_accept"] == 0.25
    assert summary["known_tail_review"] == 0.25
    assert summary["reject_rate"] == 0.5
    assert summary["reject_reason_counts"]["REJECT_LOW_DENSITY"] == 2


def test_binary_reject_metrics_return_null_unknown_metrics_when_no_unknown():
    y_unknown = torch.tensor([False, False])
    scores = torch.tensor([0.1, 0.2])
    accepted = torch.tensor([True, False])

    metrics = binary_reject_metrics(y_unknown, scores, accepted)

    assert metrics["unknown_FAR"] is None
    assert metrics["FPR95"] is None
    assert metrics["AUROC_unknown"] is None

