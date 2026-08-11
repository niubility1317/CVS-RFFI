from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from scripts import run_d92_be_prediction as entry


def _args(tmp_path: Path):
    values = [
        "--before-enrollment-package-root", "be",
        "--before-enrollment-seal-path", "bes",
        "--before-enrollment-seal-sha256", "a" * 64,
        "--before-apply-package-root", "ba",
        "--before-apply-seal-path", "bas",
        "--before-apply-seal-sha256", "b" * 64,
        "--after-enrollment-package-root", "ae",
        "--after-enrollment-seal-path", "aes",
        "--after-enrollment-seal-sha256", "c" * 64,
        "--after-apply-package-root", "aa",
        "--after-apply-seal-path", "aas",
        "--after-apply-seal-sha256", "d" * 64,
        "--ground-component-dir", "ground",
        "--ground-manifest-sha256", "e" * 64,
        "--arm", "B0E0",
        "--output-root", str(tmp_path / "out"),
        "--device", "cpu",
    ]
    return entry.parser().parse_args(values)


def _readonly(path: Path, data: bytes = b"x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    os.chmod(path, stat.S_IREAD)


def test_prediction_entry_never_accepts_or_forwards_truth(monkeypatch, tmp_path: Path):
    args = _args(tmp_path)
    assert all("truth" not in action.dest for action in entry.parser()._actions)

    def fake_evaluator(**kwargs):
        assert all("truth" not in key for key in kwargs)
        root = Path(kwargs["output_root"])
        for state in ("before", "after"):
            _readonly(root / state / "prediction_artifact.npz")
            _readonly(root / state / "COMMIT.json")
        return {
            "candidate": "d92_be_b0e0",
            "schema": "cvs.phase2.d92_be.b0e0.full_query_evaluation.v1",
            "arm_id": "B0E0",
            "states": {"before": {}, "after": {}},
        }

    monkeypatch.setattr(entry, "run_d92_be_query_evaluation", fake_evaluator)
    result = entry.run(args)
    assert result["status"] == "D92_BE_TRUTH_FREE_PREDICTIONS_COMPLETE"
    assert result["query_truth_access"] is False
    assert result["query_fit_access"] is False
    assert result["query_update_access"] is False
    assert result["query_selection_access"] is False


def test_prediction_entry_refuses_existing_output_before_evaluation(
    monkeypatch, tmp_path: Path
):
    args = _args(tmp_path)
    Path(args.output_root).mkdir(parents=True)
    (Path(args.output_root) / "owned.txt").write_text("owned", encoding="utf-8")
    called = []
    monkeypatch.setattr(
        entry,
        "run_d92_be_query_evaluation",
        lambda **_kwargs: called.append(True),
    )
    with pytest.raises(entry.D92BEPredictionEntryError, match="output"):
        entry.run(args)
    assert called == []
