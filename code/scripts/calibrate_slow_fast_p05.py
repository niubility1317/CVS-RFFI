"""Source-only receiver-held-out calibration CLI for Slow-Fast P0.5."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.slow_fast_bundle import load_slow_fast_bundle_strict  # noqa: E402
from cvsrffi.slow_fast_cache import GroundFeatureCache  # noqa: E402
from cvsrffi.slow_fast_calibration import (  # noqa: E402
    build_receiver_heldout_episodes,
    calibrate_p05_gate,
    default_p05_candidates,
    save_calibration_json,
)


_CACHE_KEYS = frozenset(
    {
        "features",
        "labels",
        "receivers",
        "days",
        "scenes",
        "physical_sample_ids",
        "views",
        "roles",
    }
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
    parser.add_argument("--ground-cache", required=True)
    parser.add_argument("--film-bundle", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--k-shot", type=int, default=10)
    parser.add_argument("--seed", type=int, default=392002)
    args = parser.parse_args()

    cache = _load_cache(args.ground_cache)
    state, audit = load_slow_fast_bundle_strict(args.film_bundle)
    episodes = build_receiver_heldout_episodes(
        cache,
        k_shot=args.k_shot,
        seed=args.seed,
    )
    calibration = calibrate_p05_gate(
        episodes,
        default_p05_candidates(float(audit["trust_radius"])),
        prototypes=audit["prototypes"],
        initial_state=state,
        logit_scale=float(audit["support_logit_scale"]),
        seed=args.seed,
    )
    save_calibration_json(args.output, calibration)
    print(json.dumps(calibration, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
