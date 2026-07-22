from __future__ import annotations

import importlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "code"
    / "scripts"
    / "run_d10_support_only_enrollment.py"
)
SPEC = importlib.util.spec_from_file_location("d10_support_runner", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_parser_has_no_query_truth_prediction_score_or_scorer_argument():
    destinations = {
        action.dest for action in runner.build_parser()._actions
    }
    assert destinations == {
        "help",
        "before_root",
        "before_seal",
        "after_root",
        "after_seal",
        "output",
        "device",
    }
    assert not destinations.intersection(
        {"query", "truth", "prediction", "score", "scorer"}
    )


def test_runner_operator_bank_has_no_cfo_derotation():
    assert runner.OPERATORS == (
        "base",
        "wl_iq_circularize",
        "fft_envelope_eq",
    )
    assert runner.FFT_ENVELOPE_SHRINK == pytest.approx(0.12)
    assert runner.FFT_GAIN_MIN > 0.0
    assert runner.FFT_GAIN_MAX < 1.5


def test_state_writer_is_create_only_and_uses_real_operator_names(
    tmp_path: Path,
):
    d10 = importlib.import_module(
        "cvsrffi.stage2_blind_receiver_operator_bank"
    )
    labels = np.repeat(np.asarray(["a", "b"]), 10)
    rng = np.random.default_rng(29)
    features = {
        operator: rng.normal(size=(20, 8)).astype(np.float32)
        for operator in d10.OPERATORS
    }
    hashes = tuple(f"{index:064x}" for index in range(20))
    state = d10.fit_blind_receiver_operator_bank(
        features,
        d10.build_operator_feature_provenance(hashes, view_seed=0),
        labels,
        physical_sample_ids=tuple(
            f"sid_{index:064x}" for index in range(20)
        ),
        parent_received_iq_sha256=hashes,
        base_resource_audit={
            "persistent_state_bytes": 0,
            "estimated_head_macs_per_query": 0,
        },
        received_iq_length=64,
    )
    hashes_written = runner._write_state_new(
        tmp_path, stem="state", state=state
    )
    assert set(hashes_written) == {
        "npz_sha256",
        "metadata_sha256",
    }
    metadata = json.loads(
        (tmp_path / "state.json").read_text(encoding="utf-8")
    )
    assert metadata["selection_lock_k"] == 10
    assert set(metadata["used_operators"]).issubset(
        set(d10.OPERATORS)
    )
    with pytest.raises(FileExistsError):
        runner._write_state_new(
            tmp_path, stem="state", state=state
        )
