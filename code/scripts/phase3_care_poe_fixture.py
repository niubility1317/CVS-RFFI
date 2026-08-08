#!/usr/bin/env python
"""Create a deterministic truth-separated CARE-PoE G0 fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.phase3_care_poe import SCHEMA, canonical_json, seal_local_evidence, sha256_json, write_jsonl


def _probabilities(kind: str, node_index: int, *, new: bool) -> list[float]:
    jitter = 0.01 * (node_index % 3)
    if kind == "known_a":
        return [0.82 + (0.04 if new else 0.0) - jitter, 0.10, 0.08 - (0.04 if new else 0.0) + jitter]
    if kind == "known_b":
        return [0.09, 0.83 + (0.03 if new else 0.0) - jitter, 0.08 - (0.03 if new else 0.0) + jitter]
    return [0.19 - (0.05 if new else 0.0) + jitter, 0.17, 0.64 + (0.05 if new else 0.0) - jitter]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    nodes = [f"SAT-{index:02d}" for index in range(1, 6)]
    events = [
        ("EVT-KA-001", "known_a", "registered", "tx-a"),
        ("EVT-KB-001", "known_b", "registered", "tx-b"),
        ("EVT-U-001", "unknown", "unknown", "tx-u"),
    ]
    base_records = []
    new_records = []
    for event_id, kind, local_kind, _ in events:
        for node_index, node in enumerate(nodes, start=1):
            correlation_group = "CHAIN-A" if node_index <= 2 else ("CHAIN-B" if node_index == 3 else "CHAIN-C")
            for bundle_kind, target, is_new in (
                ("base", base_records, False),
                ("new", new_records, True),
            ):
                probabilities = _probabilities(kind, node_index, new=is_new)
                decision = local_kind
                label = None
                if decision == "registered":
                    label = "tx-a" if kind == "known_a" else "tx-b"
                target.append(
                    seal_local_evidence(
                        {
                            "schema_version": SCHEMA,
                            "linkage_mode": "verified_physical",
                            "emission_event_id": event_id,
                            "satellite_reception_id": f"{event_id}-{node}",
                            "node_id": node,
                            "base_manifest_id": "SYNTH-G0-SAME-INPUT-V1",
                            "bundle_id": f"synthetic-{bundle_kind}-bundle-v1",
                            "class_handles": ["tx-a", "tx-b"],
                            "p_local": probabilities,
                            "q": 0.94 - 0.02 * (node_index - 1),
                            "correlation_group_id": correlation_group,
                            "delay_ms": 5.0 * node_index,
                            "deadline_ms": 100.0,
                            "local_decision": decision,
                            "local_label": label,
                            "reason_code": f"SYNTHETIC_{decision.upper()}",
                            "sealed_at_ms": 1_000.0 + node_index,
                        }
                    )
                )
    truth = [
        {"event_key": event_id, "role": "unknown" if kind == "unknown" else "registered", "true_label": label}
        for event_id, kind, _, label in events
    ]
    write_jsonl(output / "base_evidence.jsonl", base_records)
    write_jsonl(output / "new_evidence.jsonl", new_records)
    write_jsonl(output / "truth_sidecar.jsonl", truth)
    credential = {
        "anonymous_entity_id": "TO_BE_BOUND_BY_LIFECYCLE_ENTRY",
        "candidate_identity": "authorized-new-tx-001",
        "evidence_sources": ["trusted-registry-a", "trusted-operator-b"],
        "independent_sources": True,
        "conflicts": [],
        "confidence": 0.98,
        "valid_until_ms": 9_999_999.0,
        "registration_authorized": True,
        "signature": "synthetic-external-signature",
    }
    support = [
        {
            "linkage_mode": "verified_physical",
            "emission_event_id": f"EVT-FRESH-{index:02d}",
            "physical_sample_id": f"PHY-FRESH-{index:02d}",
            "candidate_identity": "authorized-new-tx-001",
        }
        for index in range(1, 6)
    ]
    (output / "credential_template.json").write_text(canonical_json(credential) + "\n", encoding="utf-8")
    write_jsonl(output / "fresh_support.jsonl", support)
    manifest = {
        "evidence_level": "TECHNICAL_SYNTHETIC_NO_PERFORMANCE_RESULT",
        "base_manifest_id": "SYNTH-G0-SAME-INPUT-V1",
        "node_roster": nodes,
        "event_count": len(events),
        "reception_count_per_bundle": len(base_records),
        "fixture_hash": sha256_json({"base": base_records, "new": new_records}),
    }
    (output / "fixture_manifest.json").write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
