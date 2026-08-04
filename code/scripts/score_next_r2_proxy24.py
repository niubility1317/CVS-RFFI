#!/usr/bin/env python3
"""Run the independent truth-side scorer for a complete NEXT-R2 run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping


SCRIPT_ROOT = Path(__file__).resolve().parent
CODE_ROOT = SCRIPT_ROOT.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi import stage2_next_r2_score as scorer  # noqa: E402


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _write_new(path: Path, value: Mapping[str, Any]) -> str:
    if path.exists() or path.is_symlink() or not path.parent.is_dir():
        raise scorer.NextR2ScoreError(f"output path must be a new child: {path}")
    raw = json.dumps(_plain(value), ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    with path.open("xb") as handle:
        handle.write(raw)
    import hashlib

    return hashlib.sha256(raw).hexdigest()


def _run_root_child(root: Path, candidate: Path, *, name: str) -> Path:
    if candidate.is_symlink():
        raise scorer.NextR2ScoreError(f"{name} must not be a symlink")
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(root.resolve(strict=True))
    except ValueError as error:
        raise scorer.NextR2ScoreError(f"{name} must be inside run_root") from error
    if resolved == root.resolve(strict=True):
        raise scorer.NextR2ScoreError(f"{name} must be a run-root child file")
    return resolved


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--prediction-capsule", required=True, type=Path)
    parser.add_argument("--prediction-capsule-sha256", required=True)
    parser.add_argument("--ls-label-join", required=True, type=Path)
    parser.add_argument("--ls-label-join-sha256", required=True)
    parser.add_argument("--score-output", type=Path)
    parser.add_argument("--scoring-completion-output", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    root = args.run_root.resolve(strict=True)
    score_path = _run_root_child(root, args.score_output or (root / "score.json"), name="score output")
    completion_path = _run_root_child(root, args.scoring_completion_output or (root / "scoring_completion.json"), name="scoring completion output")
    if score_path == completion_path:
        raise scorer.NextR2ScoreError("score and scoring-completion paths must differ")
    # Reserve both names before doing any truth-side work: neither file may be
    # overwritten and a pre-existing score cannot be relabelled as fresh.
    if score_path.exists() or completion_path.exists():
        raise scorer.NextR2ScoreError("score.json/scoring_completion.json already exists")
    result = scorer.score_next_r2_proxy24(
        run_root=root,
        prediction_capsule=args.prediction_capsule,
        prediction_capsule_sha256=args.prediction_capsule_sha256,
        ls_label_join_archive=args.ls_label_join,
        ls_label_join_archive_sha256=args.ls_label_join_sha256,
    )
    score_sha = _write_new(score_path, result)
    completion = {
        "schema": scorer.SCORING_COMPLETION_SCHEMA,
        "run_id": result.get("run_id"),
        "candidate_id": result["candidate_id"],
        "status": "SCORED_COMPLETE",
        "capsule_id": result["capsule_id"],
        "split_id": result["split_id"],
        "matrix_sha256": result["matrix_sha256"],
        "outer_key_count": result["outer_key_count"],
        "states_completed": result["state_prediction_count"],
        "truth_opened_after_complete_predictions": True,
        "formal_target_claim": False,
        "score_path": str(score_path),
        "score_sha256": score_sha,
        "label_join_archive_sha256": result["ls_label_join_archive_sha256"],
        "decision": result["decision"],
    }
    _write_new(completion_path, completion)
    print(json.dumps(_plain({"score": result, "scoring_completion": completion}), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
