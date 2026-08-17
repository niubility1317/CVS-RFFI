#!/usr/bin/env python
"""Export an immutable source-only D92 DA0 old-class state from Phase1 inputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

CODE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_ROOT.parent
for candidate in (str(CODE_ROOT), str(REPO_ROOT)):
    while candidate in sys.path:
        sys.path.remove(candidate)
for candidate in (str(REPO_ROOT), str(CODE_ROOT)):
    sys.path.insert(0, candidate)

from cvsrffi.leo_weak_cache import (  # noqa: E402
    FORMAL_LEO_WEAK_SCENARIOS,
    load_verified_leo_weak_cache_set,
    sha256_file,
)
from cvsrffi.phase1_adv3b02_deployment_bundle import (  # noqa: E402
    load_formal_adv3b02_deployment_bundle,
)
from cvsrffi.stage2_d92_da0_phase1_old_state import (  # noqa: E402
    build_source_only_joint288,
    fit_source_only_old_state,
    seal_source_only_old_state,
)


def _sha256(value: str, *, field: str) -> str:
    result = str(value).lower()
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{field} must be a lowercase SHA256")
    return result


def _expect_file_sha256(path: Path, expected: str, *, field: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"{field} is missing: {path}")
    actual = sha256_file(path)
    if actual != _sha256(expected, field=field):
        raise ValueError(f"{field} SHA256 mismatch")
    return actual


def _old_registry(class_binding: dict[str, Any]) -> tuple[str, ...]:
    rows = class_binding.get("class_id_to_handle")
    if not isinstance(rows, list):
        raise ValueError("sealed ADV3B02 class binding is invalid")
    handles = tuple(str(row.get("class_handle", "")) for row in rows)
    if not handles or len(set(handles)) != len(handles) or any(not value for value in handles):
        raise ValueError("sealed ADV3B02 old registry is invalid")
    return handles


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-bundle-root", type=Path, required=True)
    parser.add_argument("--detached-seal", type=Path, required=True)
    parser.add_argument("--expected-detached-seal-sha256", required=True)
    parser.add_argument("--signature-envelope", type=Path, required=True)
    parser.add_argument("--expected-signature-envelope-sha256", required=True)
    parser.add_argument("--expected-checkpoint-lineage-sha256", required=True)
    parser.add_argument("--expected-runtime-sha256", required=True)
    parser.add_argument("--expected-component-pre-sign-content-root-sha256", required=True)
    parser.add_argument("--expected-class-handle-binding-sha256", required=True)
    parser.add_argument("--expected-parity-receipt-sha256", required=True)
    parser.add_argument("--expected-generation-lock-sha256", required=True)
    parser.add_argument("--expected-method-lock-sha256", required=True)
    parser.add_argument("--expected-generation-config-sha256", required=True)
    parser.add_argument("--expected-generation-code-sha256", required=True)
    parser.add_argument("--expected-outer-content-root-sha256", required=True)
    parser.add_argument("--source-cache-set-manifest", type=Path, required=True)
    parser.add_argument("--expected-source-cache-set-manifest-sha256", required=True)
    parser.add_argument("--expected-source-dataset-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260817)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if int(args.batch_size) < 1:
        raise ValueError("batch-size must be positive")
    source_manifest_path = args.source_cache_set_manifest.resolve()
    source_manifest_sha = _expect_file_sha256(
        source_manifest_path,
        args.expected_source_cache_set_manifest_sha256,
        field="source cache-set manifest",
    )
    runtime_device = torch.device(str(args.device))
    if runtime_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA source-only export requested but CUDA is unavailable")
    bundle = load_formal_adv3b02_deployment_bundle(
        args.phase1_bundle_root,
        detached_seal_path=args.detached_seal,
        expected_detached_seal_sha256=args.expected_detached_seal_sha256,
        signature_envelope_path=args.signature_envelope,
        expected_signature_envelope_sha256=args.expected_signature_envelope_sha256,
        expected_checkpoint_lineage_sha256=args.expected_checkpoint_lineage_sha256,
        expected_runtime_sha256=args.expected_runtime_sha256,
        expected_component_pre_sign_content_root_sha256=args.expected_component_pre_sign_content_root_sha256,
        expected_class_handle_binding_sha256=args.expected_class_handle_binding_sha256,
        expected_parity_receipt_sha256=args.expected_parity_receipt_sha256,
        expected_generation_lock_sha256=args.expected_generation_lock_sha256,
        expected_method_lock_sha256=args.expected_method_lock_sha256,
        expected_generation_config_sha256=args.expected_generation_config_sha256,
        expected_generation_code_sha256=args.expected_generation_code_sha256,
        expected_outer_content_root_sha256=args.expected_outer_content_root_sha256,
    )
    arrays_by_scenario, _source_manifest, source_audit = load_verified_leo_weak_cache_set(
        source_manifest_path,
        expected_scope="source_train",
        allowed_roles={"source"},
    )
    old_registry = _old_registry(dict(bundle.class_binding))
    dataset_hashes = {
        str(value)
        for scenario in FORMAL_LEO_WEAK_SCENARIOS
        for value in np.asarray(
            arrays_by_scenario[scenario]["source_dataset_sha256"]
        ).astype(str)
    }
    expected_dataset_sha = _sha256(
        args.expected_source_dataset_sha256, field="source dataset"
    )
    if dataset_hashes != {expected_dataset_sha}:
        raise ValueError("source cache dataset SHA256 closure drift")
    states = {}
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        arrays = arrays_by_scenario[scenario]
        labels = np.asarray(arrays["tx_ids"]).astype(str)
        if set(labels.tolist()) != set(old_registry):
            raise ValueError("source cache old registry does not match sealed bundle")
        joint288 = build_source_only_joint288(
            bundle.runtime,
            np.asarray(arrays["leo_weak_iq"], dtype=np.float32),
            device=runtime_device,
            batch_size=int(args.batch_size),
        )
        states[scenario] = fit_source_only_old_state(
            joint288,
            labels,
            old_registry,
            seed=int(args.seed),
            device=runtime_device,
        )
    cache_audits = dict(source_audit["cache_audits"])
    receipt = seal_source_only_old_state(
        args.output,
        states_by_scenario=states,
        provenance={
            "checkpoint_sha256": bundle.formal_phase2_context[
                "checkpoint_lineage_sha256"
            ],
            "runtime_sha256": bundle.formal_phase2_context["runtime_sha256"],
            "source_cache_set_manifest_sha256": source_manifest_sha,
            "source_dataset_sha256": expected_dataset_sha,
            "class_handle_binding_sha256": bundle.formal_phase2_context[
                "class_handle_binding_sha256"
            ],
            "source_cache_member_sha256_by_scenario": {
                scenario: cache_audits[scenario]["sha256"]
                for scenario in FORMAL_LEO_WEAK_SCENARIOS
            },
            "source_cache_physical_id_root_by_scenario": {
                scenario: source_audit["physical_sample_ids_sha256_by_scenario"][
                    scenario
                ]
                for scenario in FORMAL_LEO_WEAK_SCENARIOS
            },
        },
    )
    print(
        json.dumps(
            {
                "status": "SOURCE_ONLY_D92_DA0_OLD_STATE_SEALED",
                "output": str(args.output),
                "manifest_sha256": receipt["manifest_sha256"],
                "content_root_sha256": receipt["content_root_sha256"],
                "target_receiver_old_support_opened": False,
                "target_receiver_new_support_opened": False,
                "target_query_opened": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
