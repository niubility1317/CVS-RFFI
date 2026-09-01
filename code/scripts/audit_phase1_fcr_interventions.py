from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _records_from_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list) and all(isinstance(item, Mapping) for item in payload):
        return [dict(item) for item in payload]
    if isinstance(payload, Mapping) and isinstance(payload.get("records"), list):
        records = payload["records"]
        if all(isinstance(item, Mapping) for item in records):
            return [dict(item) for item in records]
    raise ValueError("index JSON must be a list of metadata records or {'records': [...]}")


def audit_records(records: list[Mapping[str, Any]], config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Report strict pair evidence from existing metadata without mutating data."""

    configured = bool((config or {}).get("common_preamble_configured", False)) or any(
        record.get("common_preamble_id") not in (None, "") for record in records
    )
    content_groups: dict[str, set[int]] = {}
    for record in records:
        record_id = record.get("content_record_id")
        offset = record.get("crop_offset")
        if record_id in (None, "") or offset is None:
            continue
        try:
            content_groups.setdefault(str(record_id), set()).add(int(offset))
        except (TypeError, ValueError):
            continue
    content_pairs = sum(len(offsets) * (len(offsets) - 1) // 2 for offsets in content_groups.values())

    fingerprint_groups: dict[tuple[Any, ...], set[int]] = {}
    for record in records:
        key = (
            record.get("common_preamble_id"),
            record.get("rx_i"),
            record.get("day_i"),
            record.get("view_type"),
            record.get("link_condition"),
            record.get("excitation_bin"),
        )
        if any(value is None or value == "" for value in key):
            continue
        try:
            tx_i = int(record["tx_i"])
        except (KeyError, TypeError, ValueError):
            continue
        fingerprint_groups.setdefault(key, set()).add(tx_i)
    fingerprint_candidates = sum(len(tx_ids) * (len(tx_ids) - 1) // 2 for tx_ids in fingerprint_groups.values())

    reasons = {
        "content": "available" if content_pairs else "missing_content_window_metadata",
        "fingerprint": (
            "available"
            if fingerprint_candidates
            else "missing_common_preamble_metadata"
            if not configured
            else "no_matched_different_tx_common_preamble_pair"
        ),
    }
    return {
        "common_preamble_configured": configured,
        "content_window_pairs": int(content_pairs),
        "fingerprint_pair_candidates": int(fingerprint_candidates),
        "reasons": reasons,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only strict FCR intervention capability audit")
    parser.add_argument("--index", type=Path, help="Existing JSON WiSig metadata index; never guessed")
    parser.add_argument("--config", type=Path, help="Optional existing JSON common-preamble configuration")
    parser.add_argument("--output", type=Path, required=True, help="JSON report destination")
    args = parser.parse_args()

    if args.index is None:
        report = {
            "common_preamble_configured": False,
            "content_window_pairs": 0,
            "fingerprint_pair_candidates": 0,
            "reasons": {"audit": "live_audit_unmeasured_no_explicit_index"},
        }
    else:
        records = _records_from_payload(_load_json(args.index))
        config = _load_json(args.config) if args.config is not None else None
        if config is not None and not isinstance(config, Mapping):
            raise ValueError("config JSON must be an object")
        report = audit_records(records, config)
        report["index_path"] = str(args.index)
        report["record_count"] = len(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
