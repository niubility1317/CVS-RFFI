#!/usr/bin/env python
"""Build or validate the frozen F1C single-control deployment bundle.

The fixture route is intentionally technical-only.  The real route accepts
only the F1C control checkpoint, three locked receipts, and the externally
anchored ManySig PKL; it has no proxy/held/query input option.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi import phase1_single_control_bundle_v1 as scb


class _FixtureRuntime(nn.Module):
    def forward(self, rows: torch.Tensor):
        mean = rows.mean(dim=2)
        z_id = torch.stack((mean[:, 0] + 1.0, mean[:, 1] + 0.5), dim=1)
        logits = torch.stack((z_id[:, 0], z_id[:, 1], -z_id[:, 0], -z_id[:, 1]), dim=1)
        return z_id, logits


def _sha(char: str) -> str:
    return char * 64


def _fixture_runtime_bytes() -> bytes:
    traced = torch.jit.trace(_FixtureRuntime().eval(), torch.zeros((1, 2, 256), dtype=torch.float32))
    stream = io.BytesIO()
    torch.jit.save(traced, stream)
    return stream.getvalue()


def _fixture_source_split() -> dict[str, object]:
    return {
        "schema": "cvs.phase1.source_split_receipt.v1", "seed": 7281105, "split_mode": "tx_rx_day_1_6_3",
        "source_days": ["0", "1"], "target_days": ["2", "3"],
        "source_receivers": ["0", "1", "2", "3", "4", "5", "6"],
        "target_receivers": ["10", "11", "7", "8", "9"], "source_target_receiver_overlap_count": 0,
        "labeled_indices_sha256": _sha("1"), "unlabeled_indices_sha256": _sha("2"),
        "source_validation_indices_sha256": _sha("3"), "split_manifest_sha256": _sha("4"),
        "labeled_size": 3920, "unlabeled_size": 35280, "source_validation_size": 16800, "source_pool_size": 56000,
        "requested_labeled_ratio": 0.07, "requested_unlabeled_ratio": 0.63, "requested_source_val_ratio": 0.30,
        "requested_rho_label": 0.10, "realized_rho_label": 0.1, "realized_rho_tolerance": 0.002,
        "realized_rho_within_tolerance": True, "realized_source_val_fraction": 0.3,
        "realized_source_val_tolerance": 0.002, "realized_source_val_within_tolerance": True,
    }


def _fixture_labeled_keys() -> list[tuple[str, str, str, int]]:
    return [(label, "0", "0", sig_i) for label in scb.LOCAL4_HANDLES for sig_i in range(32)]


def _fixture_unlabeled_keys() -> list[tuple[str, str, str, int]]:
    return [(label, "0", "0", 32 + sig_i) for label in scb.LOCAL4_HANDLES for sig_i in range(16)]


def _fixture_calibration_keys() -> list[tuple[str, str, str, int]]:
    return [(scb.LOCAL4_HANDLES[index % 4], "0", "1", index) for index in range(199)]


def _fixture_materials() -> dict[str, object]:
    runtime_bytes = _fixture_runtime_bytes()
    runtime = torch.jit.load(io.BytesIO(runtime_bytes), map_location="cpu").eval()
    split = _fixture_source_split()
    partition = scb.build_source_partition_receipt(
        dataset_sha256=_sha("d"), source_split_projection=split, tx_partition_receipt={"enabled": True},
        labeled_keys=_fixture_labeled_keys(), unlabeled_keys=_fixture_unlabeled_keys(),
        calibration_keys=_fixture_calibration_keys(),
        excluded_role_keys={"proxy": [("14-10", "0", "0", 0)], "held": [("14-7", "0", "0", 0)], "target": [("99-99", "0", "0", 0)]},
    )
    state = scb.BundleState(
        geometry=scb.ClassGeometry(
            class_handles=scb.LOCAL4_HANDLES,
            centers=np.asarray(((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)), dtype=np.float64),
            radii=np.ones(4, dtype=np.float64), class_counts=np.full(4, 32, dtype=np.int64),
        ),
        descriptor=scb.DescriptorStats(np.zeros(5, dtype=np.float64), np.ones(5, dtype=np.float64), 4 * (128 + 64)),
        tail=scb.TailSummary(
            scb.TAIL_LEVELS.copy(), np.linspace(0.0, 5.0, 129), np.linspace(-5.0, 5.0, 129),
            np.linspace(0.0, 8.0, 129), 199, str(partition["calibration_physical_set_sha256"]),
        ),
    )
    checkpoint_sha, config_sha = _sha("a"), _sha("b")
    parity = scb.make_runtime_parity_receipt(
        eager_runtime=runtime, runtime=runtime, state=state, checkpoint_sha256=checkpoint_sha,
        resolved_config_digest=config_sha, runtime_bytes=runtime_bytes,
    )
    resource = {
        "schema": "cvs.phase1.single_control_resource_receipt.v1", "input_shape": [1, 2, 256],
        "input_dtype": "torch.float32", "input_sha256": scb.tensor_sha256(scb.resource_probe_model_input()),
        "input_seed": scb.RESOURCE_INPUT_SEED, "torch_num_threads": 1,
        "cpu_rss_baseline_bytes": 0, "cpu_rss_peak_bytes": 0, "cpu_rss_delta_bytes": 0,
        "cpu_warmups": 20, "cpu_trials": 100, "cpu_latency_p99_ms": 0.0,
        "cpu_latency_quantile_method": "higher_q99_100", "cuda_available": False, "cuda_peak_bytes": 0,
        "cuda_latency_recorded": False, "cuda_latency_p99_ms": 0.0,
        "measurement_scope": "fresh_process_bundle_load_warmup_full_local_evidence", "evidence_bytes": 1,
        "measurement_process": "fresh_python_subprocess_v1", "baseline_before_payload_load": True,
        "state_payload_reloaded": True, "runtime_state_before_sha256": _sha("9"),
        "runtime_state_after_sha256": _sha("9"),
    }
    return {
        "runtime_source": runtime_bytes, "state": state,
        "checkpoint_binding": {
            "checkpoint_sha256": checkpoint_sha, "resolved_config_sha256": config_sha,
            "checkpoint_role": "training_final_only", "strict_state_tensor_schema_sha256": _sha("e"),
            "strict_load_audit": {"strict": True, "missing_keys": [], "unexpected_keys": []},
        },
        "class_binding": {"class_handles": list(scb.LOCAL4_HANDLES), "local_to_head_class_ids": [0, 1, 2, 3],
                          "class_order_binding_sha256": _sha("f"), "checkpoint_head_class_count": 4,
                          "live_head_class_count": 4, "checkpoint_train_tx_class_order": list(scb.LOCAL4_HANDLES)},
        "source_partition_receipt": partition, "runtime_parity_receipt": parity, "resource_receipt": resource,
        "checkpoint_sha256": checkpoint_sha, "resolved_config_digest": config_sha, "dataset_sha256": _sha("d"),
        "preprocessing_code_sha256": _sha("8"), "scenario_registry_sha256": _sha("7"),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fixture-build", action="store_true", help="build a local technical fixture bundle")
    mode.add_argument("--real-build", action="store_true", help="build only from locked F1C control inputs")
    mode.add_argument("--verify-bundle", action="store_true", help="strictly verify an existing externally anchored bundle")
    parser.add_argument("--output-dir", type=Path, help="new fixture/real bundle root")
    parser.add_argument("--bundle-dir", type=Path, help="existing bundle root for --verify-bundle")
    parser.add_argument("--expected-content-root", help="external root anchor for --verify-bundle")
    parser.add_argument("--expected-status", choices=(scb.BUNDLE_STATUS, scb.FIXTURE_STATUS), help="required manifest status")
    parser.add_argument("--project-root", type=Path, default=CODE_ROOT.parent, help="repository snapshot root")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--wisig-pkl", type=Path)
    parser.add_argument("--completion-receipt", type=Path)
    parser.add_argument("--terminal-receipt", type=Path)
    parser.add_argument("--cp-terminal-receipt", type=Path)
    parser.add_argument("--device", default="cpu")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.verify_bundle:
        if args.bundle_dir is None or not args.expected_content_root:
            raise SystemExit("--verify-bundle requires --bundle-dir and --expected-content-root")
        bundle = scb.load_bundle(
            args.bundle_dir,
            expected_content_root=args.expected_content_root,
            expected_bundle_status=args.expected_status or scb.BUNDLE_STATUS,
        )
        print(json.dumps({"bundle_dir": str(bundle.root), "content_root": bundle.content_root, "status": bundle.manifest["bundle_status"]}, sort_keys=True))
        return 0
    if args.output_dir is None:
        raise SystemExit("--fixture-build/--real-build requires --output-dir")
    if args.fixture_build:
        result = scb.build_bundle(output_dir=args.output_dir, bundle_status=scb.FIXTURE_STATUS, **_fixture_materials())
        print(json.dumps(result, sort_keys=True))
        return 0
    required = (args.checkpoint, args.wisig_pkl, args.completion_receipt, args.terminal_receipt, args.cp_terminal_receipt)
    if any(value is None for value in required):
        raise SystemExit("--real-build requires checkpoint, ManySig PKL, and all three frozen F1C receipts")
    result = scb.build_real_bundle_from_paths(
        project_root=args.project_root, checkpoint_path=args.checkpoint, wisig_pkl_path=args.wisig_pkl,
        completion_receipt_path=args.completion_receipt, terminal_receipt_path=args.terminal_receipt,
        cp_terminal_receipt_path=args.cp_terminal_receipt, output_dir=args.output_dir, device=args.device,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
