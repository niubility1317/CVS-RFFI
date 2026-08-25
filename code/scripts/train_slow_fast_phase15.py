"""Build the source-only ground cache and fit three aggregate Phase1.5 bundles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import sys


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from dataset_wisig import load_wisig_compact_pkl  # noqa: E402
from cvsrffi.meta_phase1_entry import (  # noqa: E402
    _source_role_manifest,
    load_meta_phase1_config,
)
from cvsrffi.slow_fast_phase15_entry import (  # noqa: E402
    build_ground_feature_cache,
    fit_and_save_slow_fast_bundles,
)
from cvsrffi.stage2_meta_adapter_runner import (  # noqa: E402
    _load_npz,
    _prototype_tensors,
)
from cvsrffi.stage2_structured_late_block_runner import _load_frozen_checkpoint  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wisig-pkl", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--prototype-path", required=True)
    parser.add_argument("--source-config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--base-checkpoint-id", default="ADV3B02_CORE90_SOFT_E200")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--meta-steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=392002)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"immutable Phase1.5 output already exists: {output_root}")
    for label, value in (
        ("wisig", args.wisig_pkl),
        ("base checkpoint", args.base_checkpoint),
        ("prototype", args.prototype_path),
        ("source config", args.source_config),
    ):
        if not Path(value).is_file():
            raise FileNotFoundError(f"{label} file is missing: {value}")

    source_config = load_meta_phase1_config(args.source_config)
    if int(source_config["seed"]) != int(args.seed):
        raise ValueError("Phase1.5 seed must match the frozen source split config")
    ds_w = load_wisig_compact_pkl(args.wisig_pkl)
    source_args = SimpleNamespace(
        wisig_train_days=source_config["source_days"],
        wisig_test_days=source_config["clean_test_days"],
    )
    source_manifest = _source_role_manifest(ds_w, source_config, source_args)
    if not source_manifest.get("available") or not source_manifest.get("source_only"):
        raise ValueError("source-only L_s manifest is unavailable")
    l_s_dataset = source_manifest["role_datasets"]["L_s"]

    prototype_payload = _load_npz(
        args.prototype_path,
        allowed=frozenset({"prototypes", "class_ids"}),
        label="prototype",
    )
    prototypes, class_ids = _prototype_tensors(prototype_payload)
    class_id_to_row = {
        int(class_id): row for row, class_id in enumerate(class_ids.tolist())
    }
    model = _load_frozen_checkpoint(args.base_checkpoint, device=args.device)
    cache = build_ground_feature_cache(
        model,
        l_s_dataset,
        class_id_to_row=class_id_to_row,
        seed=args.seed,
        device=args.device,
        batch_size=args.batch_size,
    )
    summary = fit_and_save_slow_fast_bundles(
        cache,
        prototypes,
        class_ids,
        output_root,
        base_checkpoint_id=args.base_checkpoint_id,
        steps=args.steps,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
        meta_steps=args.meta_steps,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
