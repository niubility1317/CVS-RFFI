from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def test_training_test_policy_never_is_strictly_truth_blind() -> None:
    from training_test_eval import should_run_training_test

    for epoch in (1, 199, 200):
        assert not should_run_training_test(
            "never", epoch=epoch, epochs=200, val_improved=True
        )


def test_truth_sidecar_scores_across_independent_process_boundary(tmp_path: Path) -> None:
    from cvsrffi.truth_last import build_truth_sidecar, score_predictions

    identities = [
        {"physical_id": "tx0:rx0:day0:eq1:sig0", "label": 0},
        {"physical_id": "tx1:rx0:day0:eq1:sig0", "label": 1},
    ]
    truth_path = tmp_path / "truth.json"
    sidecar = build_truth_sidecar(
        identities,
        output_path=truth_path,
        split_binding="ManySig|tx_rx_day_1_7_2|392005",
    )
    prediction_path = tmp_path / "predictions.json"
    records = []
    for scenario in ("clean", "leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak"):
        for row in sidecar["records"]:
            records.append(
                {
                    "sample_id": row["sample_id"],
                    "scenario": scenario,
                    "predicted_class": row["label"],
                    "run_id": "run",
                    "row_id": "R7",
                }
            )
    prediction_path.write_text(json.dumps({"records": records}), encoding="utf-8")
    score = score_predictions(prediction_path, truth_path)
    assert score["record_count"] == 8
    assert score["metrics"]["clean"]["accuracy"] == 1.0
    assert score["metrics"]["leo_rain_weak"]["accuracy"] == 1.0


def test_truth_last_scorer_rejects_missing_prediction(tmp_path: Path) -> None:
    from cvsrffi.truth_last import build_truth_sidecar, score_predictions

    truth_path = tmp_path / "truth.json"
    build_truth_sidecar(
        [{"physical_id": "tx0:rx0:day0:eq1:sig0", "label": 0}],
        output_path=truth_path,
        split_binding="ManySig|tx_rx_day_1_7_2|392005",
    )
    prediction_path = tmp_path / "predictions.json"
    prediction_path.write_text(json.dumps({"records": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="coverage"):
        score_predictions(prediction_path, truth_path)


def test_release_launcher_freezes_exact_protocol() -> None:
    root = Path(__file__).resolve().parents[2]
    launcher = root / "code" / "scripts" / "launch_phase1_adv3b02_fcr_r1r8_s392005_20260903.sh"
    text = launcher.read_text(encoding="utf-8")
    assert "SEED=\"${SEED:-392005}\"" in text
    assert "SOURCE_DAYS='1,2,3'" in text
    assert "SOURCE_RXS='1,3,4,6,8'" in text
    assert "TARGET_DAYS='0,1,2,3'" in text
    assert "TARGET_RXS='0,2,5,7,9,10,11'" in text
    assert "--test_eval_policy never" in text
    assert "final_checkpoint=final.pth" in text
    assert "best_test_save_path" not in text
    assert text.index("ADV3B02_CORE90_SOFT_E200") < text.index("ROWS=(R1 R2 R3 R4 R5 R6 R7 R8)")
    assert "V_cal" not in text and "V_select" not in text
    assert "STAGE2_MAX_ACTIVE_PER_GPU=999" in text


def test_predictor_process_reads_only_label_free_package() -> None:
    root = Path(__file__).resolve().parents[2]
    script = (root / "code" / "scripts" / "predict_phase1_truth_last.py").read_text(
        encoding="utf-8"
    )
    predictor_class = script.split("class _OpaqueTargetDataset", 1)[1].split(
        "def _ids_and_truth", 1
    )[0]
    assert "WiSigCompactDataset" not in predictor_class
    assert "self.base" not in predictor_class
    assert 'manifest["sample_ids"]' in predictor_class
    assert '"contains_labels": False' in script
    assert "build_exact_ssdg_model_from_checkpoint" in script
    assert '"num_domains": 15' not in script
    launcher = (
        root
        / "code"
        / "scripts"
        / "launch_phase1_adv3b02_fcr_r1r8_s392005_20260903.sh"
    ).read_text(encoding="utf-8")
    predict_blocks = launcher.split("--mode predict")[1:]
    assert predict_blocks
    for block in predict_blocks:
        assert "--wisig-pkl" not in block.split("--mode", 1)[0]


def test_legacy_checkpoint_model_defaults_use_valid_physical_sources(monkeypatch) -> None:
    import post_stage_common

    captured = {}

    def fake_build(*args, **kwargs):
        captured.update(kwargs)
        return torch.nn.Identity()

    monkeypatch.setattr(post_stage_common, "build_dual_model", fake_build)
    post_stage_common.build_baseline_model(
        SimpleNamespace(num_classes=6, num_domains=15, sample_rate_hz=0.0),
        torch.device("cpu"),
    )
    assert captured["freq_feature_source"] == "raw_fft"
    assert captured["pa_feature_source"] == "raw_iq"
    assert captured["sample_rate_hz"] == 25e6
