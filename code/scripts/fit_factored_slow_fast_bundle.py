"""Fit one all-source int8 deployment bundle after nested source selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.factored_slow_fast import fit_factored_state  # noqa: E402
from cvsrffi.factored_slow_fast_bundle import save_factored_bundle  # noqa: E402
from cvsrffi.factored_slow_fast_eval import meta_refine_factored_state  # noqa: E402
from cvsrffi.slow_fast_bundle import load_slow_fast_bundle_strict  # noqa: E402
from cvsrffi.slow_fast_cache import GroundFeatureCache  # noqa: E402


_CACHE_KEYS = frozenset(
    {"features", "labels", "receivers", "days", "scenes", "physical_sample_ids", "views", "roles"}
)


def _load_cache(path: str | Path) -> GroundFeatureCache:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != set(_CACHE_KEYS):
        raise ValueError("ground feature cache field allowlist mismatch")
    return GroundFeatureCache(**payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-cache", required=True)
    parser.add_argument("--film-bundle", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--candidate", choices=("B3", "B5"), required=True)
    parser.add_argument("--rank-receiver", type=int, default=4)
    parser.add_argument("--rank-leo", type=int, default=4)
    parser.add_argument("--ridge", type=float, default=0.1)
    parser.add_argument("--meta-steps", type=int, default=50)
    parser.add_argument("--k-shot", type=int, default=10)
    parser.add_argument("--query-per-class", type=int, default=10)
    parser.add_argument("--seed", type=int, default=392002)
    args = parser.parse_args()

    cache = _load_cache(args.ground_cache)
    _legacy_state, legacy_audit = load_slow_fast_bundle_strict(args.film_bundle)
    state, fit_audit = fit_factored_state(
        cache,
        legacy_audit["prototypes"],
        legacy_audit["class_ids"],
        rank_rx=args.rank_receiver,
        rank_leo=args.rank_leo,
        ridge_receiver=args.ridge,
        ridge_leo=args.ridge,
    )
    meta_audit: dict[str, object] = {"steps": 0}
    if args.candidate == "B5":
        state, meta_audit = meta_refine_factored_state(
            cache,
            state,
            excluded_receivers=(),
            steps=args.meta_steps,
            k_shot=args.k_shot,
            query_per_class=args.query_per_class,
            seed=args.seed,
        )
    save_factored_bundle(
        args.output,
        state,
        candidate=args.candidate,
        base_checkpoint_id=str(legacy_audit["base_checkpoint_id"]),
    )
    print(
        json.dumps(
            {
                "status": "FACTORED_BUNDLE_WRITTEN",
                "candidate": args.candidate,
                "fast_parameter_count": state.fast_parameter_count,
                "aggregate_storage_dtype": "int8",
                "fit_receivers": fit_audit["fit_receivers"],
                "meta_steps": meta_audit["steps"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
