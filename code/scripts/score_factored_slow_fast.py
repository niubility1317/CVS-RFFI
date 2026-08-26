"""Join source truth only after CVS-FSFA-V2 predictions close."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.factored_slow_fast_eval import score_nested_predictions  # noqa: E402
from cvsrffi.slow_fast_cache import GroundFeatureCache  # noqa: E402


_CACHE_KEYS = frozenset(
    {"features", "labels", "receivers", "days", "scenes", "physical_sample_ids", "views", "roles"}
)


def _load_cache(path: str | Path) -> GroundFeatureCache:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"ground feature cache is not a regular file: {source}")
    payload = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != set(_CACHE_KEYS):
        raise ValueError("ground feature cache field allowlist mismatch")
    return GroundFeatureCache(**payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--ground-cache", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    prediction_path = Path(args.predictions)
    if prediction_path.is_symlink() or not prediction_path.is_file():
        raise ValueError("prediction artifact must be a regular file")
    with prediction_path.open("r", encoding="utf-8") as handle:
        predictions = json.load(handle)
    score = score_nested_predictions(predictions, _load_cache(args.ground_cache))
    output = Path(args.output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"score output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(score, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({key: score[key] for key in ("schema", "status", "selected_strategy")}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
