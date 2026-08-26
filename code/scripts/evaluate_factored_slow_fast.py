"""Generate truth-blind outer-receiver predictions for CVS-FSFA-V2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.factored_slow_fast_eval import generate_nested_predictions  # noqa: E402
from cvsrffi.slow_fast_bundle import load_slow_fast_bundle_strict  # noqa: E402
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


def _write_json(path: str | Path, payload: dict[str, object]) -> None:
    output = Path(path)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-cache", required=True)
    parser.add_argument("--film-bundle", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--k-shot", type=int, default=10)
    parser.add_argument("--draws", type=int, default=10)
    parser.add_argument("--query-per-class", type=int, default=10)
    parser.add_argument("--seed", type=int, default=392002)
    parser.add_argument("--rank-receiver", type=int, default=4)
    parser.add_argument("--rank-leo", type=int, default=4)
    parser.add_argument("--meta-steps", type=int, default=50)
    parser.add_argument("--inner-ridge-grid", type=float, nargs="+", default=(0.03, 0.1, 0.3))
    args = parser.parse_args()

    cache = _load_cache(args.ground_cache)
    _state, audit = load_slow_fast_bundle_strict(args.film_bundle)
    predictions = generate_nested_predictions(
        cache,
        audit["prototypes"],
        audit["class_ids"],
        k_shot=args.k_shot,
        draws=args.draws,
        query_per_class=args.query_per_class,
        seed=args.seed,
        rank_rx=args.rank_receiver,
        rank_leo=args.rank_leo,
        meta_steps=args.meta_steps,
        inner_ridge_grid=tuple(args.inner_ridge_grid),
    )
    _write_json(args.output, predictions)
    print(
        json.dumps(
            {
                "schema": predictions["schema"],
                "outer_receivers": predictions["outer_receivers"],
                "episode_count": len(predictions["rows"]),
                "query_truth_opened": predictions["query_truth_opened"],
                "target_support_used": predictions["target_support_used"],
                "target_query_used": predictions["target_query_used"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
