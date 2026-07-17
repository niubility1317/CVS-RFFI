#!/usr/bin/env python3
"""Build the strict compressed ADV3B02 domain-class prototype component."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from cvsrffi.phase1_int8_prototype_bundle import (
    build_int8_component,
    save_int8_component,
    sha256_file,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_prototype_pt", required=True)
    parser.add_argument("--class_mapping_json", required=True)
    parser.add_argument("--checkpoint_sha256", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--provenance_status", required=True)
    parser.add_argument("--formal_phase2_eligible", action="store_true")
    args = parser.parse_args()

    source_path = Path(args.source_prototype_pt)
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    mapping = json.loads(Path(args.class_mapping_json).read_text(encoding="utf-8"))
    payload, manifest = build_int8_component(
        source,
        class_registry=mapping["class_id_to_tx"],
        checkpoint_sha256=args.checkpoint_sha256,
        source_prototype_artifact_sha256=sha256_file(source_path),
        provenance_status=args.provenance_status,
        formal_phase2_eligible=args.formal_phase2_eligible,
    )
    result = save_int8_component(args.output_dir, payload, manifest)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
