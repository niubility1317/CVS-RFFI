#!/usr/bin/env python
import argparse
import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import torch  # noqa: E402

from cvsrffi.phase2_prototypes import PrototypeFusionConfig, fuse_tx_domain_prototypes, save_phase2_prototype_export  # noqa: E402


def load_phase2_package(path: str | Path) -> dict:
    package = torch.load(Path(path), map_location="cpu")
    if not isinstance(package, dict):
        raise ValueError(f"Phase2 package must be a dict: {path}")
    required = {"prototypes", "tx_domain_prototypes", "tx_domain_counts"}
    missing = sorted(k for k in required if k not in package)
    if missing:
        raise ValueError(f"Input is not a Phase2 prototype package; missing {missing}")
    return package


def main() -> int:
    parser = argparse.ArgumentParser(description="Re-export Phase2 local component prototype package without retraining.")
    parser.add_argument("--prototype_package", "--checkpoint", dest="prototype_package", required=True)
    parser.add_argument("--output", required=True, help="Output .pt path for the fused package.")
    parser.add_argument("--phase2_fuse_prototypes", type=str, default="true")
    parser.add_argument("--phase2_global_ball_accept", type=str, default="false")
    parser.add_argument("--phase2_tail_auto_accept", type=str, default="false")
    parser.add_argument("--phase2_fuse_max_components", "--phase2_max_components_per_class", type=int, default=6)
    parser.add_argument("--phase2_fuse_merge_angle_deg", type=float, default=6.0)
    parser.add_argument("--phase2_fuse_radius_cap_deg", type=float, default=25.0)
    parser.add_argument("--phase2_fuse_tail_abs_deg", type=float, default=30.0)
    parser.add_argument("--phase2_fuse_accept_policy", type=str, default="local_component")
    parser.add_argument("--phase2_fuse_accept_radius_key", type=str, default="p95")
    args = parser.parse_args()

    if str(args.phase2_fuse_prototypes).lower() not in ("1", "true", "yes", "y"):
        raise ValueError("This exporter is for fused local component packages; keep --phase2_fuse_prototypes true.")
    package = load_phase2_package(args.prototype_package)
    cfg = PrototypeFusionConfig(
        max_components_per_tx=int(args.phase2_fuse_max_components),
        merge_angle_deg=float(args.phase2_fuse_merge_angle_deg),
        radius_cap_deg=float(args.phase2_fuse_radius_cap_deg),
        tail_abs_deg=float(args.phase2_fuse_tail_abs_deg),
        accept_policy=str(args.phase2_fuse_accept_policy),
        accept_radius_key=str(args.phase2_fuse_accept_radius_key),
        global_ball_accept=str(args.phase2_global_ball_accept).lower() in ("1", "true", "yes", "y"),
        tail_auto_accept=str(args.phase2_tail_auto_accept).lower() in ("1", "true", "yes", "y"),
    )
    fused = fuse_tx_domain_prototypes(package, cfg)
    paths = save_phase2_prototype_export(fused, args.output)
    print(f"[PHASE2-LOCAL-COMPONENTS] pt={paths['pt_path']} json={paths['json_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

