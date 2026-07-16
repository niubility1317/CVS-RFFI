from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import numpy as np

from cvsrffi.stage2_diag_cosine_scorer import score_diag_cosine_pair
from cvsrffi.somph_predictor_bundle import FORMAL_LEO_WEAK_SCENARIOS


def _prediction(path: Path, *, after: bool) -> None:
    tokens = ["old-a", "old-b"] + (["new-c"] if after else [])
    predicted = ["h-old-a", "h-old-b"] + (["h-new-c"] if after else [])
    with path.open("xb") as handle:
        np.savez(
            handle,
            query_tokens=np.asarray(tokens * 3),
            scenarios=np.concatenate(
                [np.asarray([scenario] * len(tokens)) for scenario in FORMAL_LEO_WEAK_SCENARIOS]
            ),
            predicted_class_handles=np.asarray(predicted * 3),
        )
    os.chmod(path, stat.S_IREAD)


def test_scores_only_after_frozen_predictions(tmp_path: Path) -> None:
    before = tmp_path / "before.npz"
    after = tmp_path / "after.npz"
    truth = tmp_path / "truth.json"
    output = tmp_path / "score.json"
    _prediction(before, after=False)
    _prediction(after, after=True)
    truth.write_text(
        json.dumps(
            {
                "schema": "cvs.phase2.query_truth_sidecar.v2",
                "rows": [
                    {
                        "query_token": "old-a",
                        "true_class_handle": "h-old-a",
                        "transmitter_label": "old-a",
                        "evaluation_role": "target_old",
                    },
                    {
                        "query_token": "old-b",
                        "true_class_handle": "h-old-b",
                        "transmitter_label": "old-b",
                        "evaluation_role": "target_old",
                    },
                    {
                        "query_token": "new-c",
                        "true_class_handle": "h-new-c",
                        "transmitter_label": "new-c",
                        "evaluation_role": "target_new",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    result = score_diag_cosine_pair(
        before_prediction_path=before,
        after_prediction_path=after,
        truth_sidecar_path=truth,
        output_path=output,
        candidate="unit",
    )
    assert result["before"]["old_acc"] == 1.0
    assert result["after"]["old_acc"] == 1.0
    assert result["after"]["seen_new_acc"] == 1.0
    assert result["old_forgetting_pp"] == 0.0
    assert output.is_file()
    assert not output.stat().st_mode & stat.S_IWUSR
