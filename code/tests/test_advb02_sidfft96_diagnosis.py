import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from eval_feature_diagnosis import (  # noqa: E402
    collect_feature_dict,
    summarize_sid_transitions,
)


def test_sid_transition_counts_are_same_row_and_exhaustive():
    raw = torch.tensor([[3.0, 0.0], [0.0, 3.0], [0.0, 3.0], [3.0, 0.0]])
    sid = torch.tensor([[3.0, 0.0], [3.0, 0.0], [3.0, 0.0], [3.0, 0.0]])
    labels = torch.tensor([0, 0, 1, 1])

    output = summarize_sid_transitions(raw, sid, labels)

    assert output == {
        "kept_correct": 1,
        "rescued": 1,
        "harmed": 1,
        "kept_wrong": 1,
        "count": 4,
    }


def test_collect_feature_dict_adds_sid_fields_only_when_present():
    base = {
        "z_id": torch.randn(2, 4),
        "z_dom": torch.randn(2, 4),
        "aux_id": {},
        "aux_dom": {},
    }
    assert set(collect_feature_dict(base)) == {"z_id", "z_dom"}

    sid = dict(base)
    sid.update(
        {
            "z_id_raw": torch.randn(2, 4),
            "z_id_sid": torch.randn(2, 4),
            "sid_fft96": torch.randn(2, 96),
            "sid_group_norms": torch.randn(2, 5),
            "sid_valid_bin_ratio": torch.ones(2),
        }
    )
    features = collect_feature_dict(sid)
    assert {
        "z_id_raw",
        "z_id_sid",
        "sid_fft96",
        "sid_group_norms",
        "sid_valid_bin_ratio",
    }.issubset(features)
