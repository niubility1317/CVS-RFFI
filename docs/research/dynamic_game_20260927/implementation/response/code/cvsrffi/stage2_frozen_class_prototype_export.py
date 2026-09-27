"""Ground-only export of immutable 2D ADV3B02 class prototypes.

This module is not a Phase2 runtime loader.  It extracts the already fused
class-level ``z_id`` anchors before target-domain adaptation starts and writes
an artifact whose exhaustive member set is ``prototypes`` and ``class_ids``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


def export_minimal_frozen_class_prototypes(
    source_package: Mapping[str, Any],
    *,
    class_ids: Sequence[int],
    output_path: str | Path,
) -> dict[str, Any]:
    """Extract precomputed fused class anchors without runtime reconstruction."""

    if str(source_package.get("feature_key", "")) != "z_id":
        raise ValueError("ground prototype package must use feature_key=z_id")
    prototypes = source_package.get("prototypes")
    if not torch.is_tensor(prototypes) or prototypes.ndim != 2:
        raise ValueError(
            "ground package must contain precomputed prototypes[class,feature]"
        )
    if prototypes.shape[0] < 1 or prototypes.shape[1] < 1:
        raise ValueError("fused class prototypes must be nonempty")
    if not torch.isfinite(prototypes).all():
        raise ValueError("fused class prototypes contain non-finite values")

    ids = np.asarray([int(value) for value in class_ids], dtype=np.int64)
    if ids.ndim != 1 or ids.shape[0] != int(prototypes.shape[0]):
        raise ValueError("class_ids must align with fused class prototypes")
    if len(set(ids.tolist())) != int(ids.size):
        raise ValueError("class_ids must be unique")

    normalized = F.normalize(prototypes.detach().float().cpu(), dim=1)
    if not torch.isfinite(normalized).all() or torch.any(normalized.norm(dim=1) == 0):
        raise ValueError("fused class prototypes must have nonzero finite norms")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        prototypes=normalized.numpy().astype(np.float32, copy=False),
        class_ids=ids,
    )
    return {
        "schema": "adv3b02_frozen_class_prototypes_v1",
        "feature_key": "z_id",
        "class_count": int(ids.size),
        "feature_dim": int(normalized.shape[1]),
        "source_member": "prototypes",
        "output_members": ["class_ids", "prototypes"],
        "phase2_immutable": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-prototype-package", type=Path, required=True)
    parser.add_argument("--class-mapping-json", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    args = parser.parse_args()

    source = torch.load(
        args.ground_prototype_package,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(source, Mapping):
        raise ValueError("ground prototype package must be a mapping")
    mapping = json.loads(args.class_mapping_json.read_text(encoding="utf-8-sig"))
    registered = mapping.get("class_id_to_tx")
    if not isinstance(registered, list) or not registered:
        raise ValueError("class mapping must contain nonempty class_id_to_tx")
    audit = export_minimal_frozen_class_prototypes(
        source,
        class_ids=range(len(registered)),
        output_path=args.output_path,
    )
    print(json.dumps(audit, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["export_minimal_frozen_class_prototypes", "main"]
