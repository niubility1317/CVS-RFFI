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
from cvsrffi.eval import collect_hsid_predictions, evaluate_loader  # noqa: E402


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
            "sid_quality": torch.randn(2, 7),
            "sid_spectral_embedding": torch.randn(2, 48),
            "sid_spec_logits": torch.randn(2, 16),
            "logits_raw": torch.randn(2, 16),
            "logits_fused": torch.randn(2, 16),
            "sid_fusion_gate": torch.rand(2),
            "sid_raw_margin": torch.rand(2),
            "sid_spec_margin": torch.rand(2),
            "sid_js_divergence": torch.rand(2),
            "sid_agreement": torch.ones(2),
        }
    )
    features = collect_feature_dict(sid)
    assert {
        "z_id_raw",
        "z_id_sid",
        "sid_fft96",
        "sid_group_norms",
        "sid_valid_bin_ratio",
        "sid_quality",
        "sid_spectral_embedding",
        "sid_spec_logits",
        "logits_raw",
        "logits_fused",
        "sid_fusion_gate",
        "sid_raw_margin",
        "sid_spec_margin",
        "sid_js_divergence",
        "sid_agreement",
    }.issubset(features)


def test_source_validation_reports_receiver_and_receiver_day_floors():
    class FixedLogitModel(torch.nn.Module):
        def forward(self, x, **_kwargs):
            return {
                "tx_logits": x,
                "dom_logits": torch.zeros((x.size(0), 1), dtype=x.dtype),
            }

    batch = (
        torch.tensor([[3.0, 0.0], [0.0, 3.0], [3.0, 0.0], [0.0, 3.0]]),
        torch.tensor([0, 1, 1, 0]),
        torch.zeros(4, dtype=torch.long),
        {
            "rx_i": torch.tensor([0, 1, 0, 1]),
            "day_i": torch.tensor([0, 0, 1, 1]),
        },
    )
    stats = evaluate_loader(FixedLogitModel(), [batch], torch.device("cpu"), {0: 0})

    assert stats["receiver_floor"] == 50.0
    assert stats["receiver_day_floor"] == 0.0
    assert stats["receiver_group_count"] == 2
    assert stats["receiver_day_group_count"] == 4


def test_hsid_prediction_export_keeps_same_row_labels_metadata_and_diagnostics():
    class FixedHSIDModel(torch.nn.Module):
        def forward(self, x, **_kwargs):
            quality = torch.arange(x.size(0) * 7, dtype=x.dtype).reshape(x.size(0), 7)
            return {
                "logits_raw": x,
                "sid_spec_logits": x.flip(dims=(1,)),
                "logits_fused": x + 0.1,
                "sid_quality": quality,
                "sid_fusion_gate": torch.full((x.size(0),), 0.2),
            }

    batch = (
        torch.tensor([[3.0, 0.0], [0.0, 3.0]]),
        torch.tensor([0, 1]),
        torch.zeros(2, dtype=torch.long),
        {"rx_i": torch.tensor([4, 5]), "day_i": torch.tensor([1, 2])},
    )
    rows = collect_hsid_predictions(
        FixedHSIDModel(),
        [batch],
        torch.device("cpu"),
        {0: 0},
        split_name="test_rx",
    )

    assert rows["y"].tolist() == [0, 1]
    assert rows["raw_pred"].tolist() == [0, 1]
    assert rows["spec_pred"].tolist() == [1, 0]
    assert rows["rx"].tolist() == [4, 5]
    assert rows["day"].tolist() == [1, 2]
    assert rows["split"].tolist() == ["test_rx", "test_rx"]
    assert rows["scenario"].tolist() == ["clean", "clean"]
    assert rows["quality"].shape == (2, 7)
