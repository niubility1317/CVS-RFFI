from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import torch


def _strict_meta() -> dict[str, list[object]]:
    return {
        "physical_sample_id": ["tx0:rx0:day0:eq0:sig0", "tx0:rx0:day0:eq0:sig0", "tx1:rx0:day0:eq0:sig2", "tx2:rx0:day0:eq0:sig3"],
        "crop_offset": [0, 4, 0, 0],
        "rx_i": [0, 0, 0, 0],
        "day_i": [0, 0, 0, 0],
        "label_visible": [True, True, True, True],
        "content_record_id": ["record-a", "record-a", "record-b", "record-c"],
        "common_preamble_id": [None, None, "preamble-1", "preamble-1"],
        "view_type": ["clean", "clean", "clean", "clean"],
        "link_condition": ["eq0", "eq0", "eq0", "eq0"],
        "excitation_bin": [0, 0, 7, 7],
    }


def _leo_view(clean: torch.Tensor, meta: dict[str, list[object]]) -> SimpleNamespace:
    return SimpleNamespace(
        x=clean + 1.0,
        applied=True,
        physical_sample_id=tuple(meta["physical_sample_id"]),
        pair_id=tuple(meta["physical_sample_id"]),
        crop_offset=torch.tensor(meta["crop_offset"]),
        nuisance=torch.zeros((clean.size(0), 2)),
        nuisance_valid=torch.ones(clean.size(0), dtype=torch.bool),
    )


def test_builder_uses_only_strict_three_axis_pairs() -> None:
    """A pair exists only when its named physical condition is explicit."""

    from cvsrffi.phase1_fcr_interventions import InterventionCubeBatchBuilder

    clean = torch.zeros(4, 2, 8)
    labels = torch.tensor([0, 0, 1, 2])
    domains = torch.tensor([0, 0, 0, 0])
    meta = _strict_meta()

    builder = InterventionCubeBatchBuilder()
    batch = builder.build(clean, _leo_view(clean, meta), labels, domains, meta)

    assert batch.nuisance_pair_index.tolist() == [0, 1, 2, 3]
    assert batch.pair_valid_mask["nuisance"].tolist() == [True, True, True, True]
    assert batch.content_pair_index[:2].tolist() == [1, 0]
    assert batch.pair_valid_mask["content"][:2].tolist() == [True, True]
    assert batch.fingerprint_pair_index[2:].tolist() == [3, 2]
    assert batch.pair_valid_mask["fingerprint"][2:].tolist() == [True, True]


def test_builder_returns_explicit_invalid_indices_without_strict_evidence() -> None:
    """Missing common-preamble or second-window facts cannot trigger a fallback pair."""

    from cvsrffi.phase1_fcr_interventions import InterventionCubeBatchBuilder, invalid_indices

    clean = torch.zeros(2, 2, 8)
    labels = torch.tensor([-1, -1])
    domains = torch.tensor([0, 0])
    meta = {
        "physical_sample_id": ["sample:a", "sample:b"],
        "crop_offset": [0, 0],
        "rx_i": [0, 0],
        "day_i": [0, 0],
        "label_visible": [False, False],
    }
    builder = InterventionCubeBatchBuilder()
    batch = builder.build(clean, _leo_view(clean, meta), labels, domains, meta)

    assert torch.equal(invalid_indices(2, clean.device), torch.tensor([-1, -1]))
    assert batch.content_pair_index.tolist() == [-1, -1]
    assert batch.fingerprint_pair_index.tolist() == [-1, -1]
    assert batch.pair_valid_mask["content"].tolist() == [False, False]
    assert batch.pair_valid_mask["fingerprint"].tolist() == [False, False]
    assert builder.capability.reason["content"] == "missing_content_window_metadata"
    assert builder.capability.reason["fingerprint"] == "missing_common_preamble_metadata"


def test_sanitization_keeps_only_non_reversible_unlabeled_identity() -> None:
    """A hidden-role caller cannot reconstruct its TX label from FCR metadata."""

    from cvsrffi.phase1_fcr_interventions import build_physical_sample_id, sanitize_fcr_meta

    raw = {"tx_i": 9, "rx_i": 2, "day_i": 3, "eq_i": 1, "sig_i": 8, "true_tx_i": 9}
    assert build_physical_sample_id(raw) == "tx9:rx2:day3:eq1:sig8"

    hidden = sanitize_fcr_meta(raw, label_visible=False)

    assert set(hidden).isdisjoint({"tx_i", "tx", "true_tx_i"})
    assert hidden["physical_sample_id"].startswith("sample:")
    assert "tx9" not in hidden["physical_sample_id"]


def test_read_only_audit_reports_missing_strict_capabilities(tmp_path: Path) -> None:
    """Absent common-preamble facts are a zero-exit scientific result, not a fallback."""

    records = [
        {"tx_i": 0, "rx_i": 0, "day_i": 0, "eq_i": 0, "sig_i": 0, "crop_offset": 0},
        {"tx_i": 1, "rx_i": 0, "day_i": 0, "eq_i": 0, "sig_i": 1, "crop_offset": 0},
    ]
    index_path = tmp_path / "wisig-index.json"
    output_path = tmp_path / "audit.json"
    index_path.write_text(json.dumps(records), encoding="utf-8")
    script = Path(__file__).resolve().parents[1] / "scripts" / "audit_phase1_fcr_interventions.py"

    completed = subprocess.run(
        [sys.executable, str(script), "--index", str(index_path), "--output", str(output_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["common_preamble_configured"] is False
    assert result["content_window_pairs"] == 0
    assert result["fingerprint_pair_candidates"] == 0
    assert result["reasons"]["fingerprint"] == "missing_common_preamble_metadata"
