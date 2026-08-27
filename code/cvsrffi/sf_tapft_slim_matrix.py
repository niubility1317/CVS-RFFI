"""Strict row resolver for the support-only SF-TAPFT slimming matrix."""

from __future__ import annotations

from dataclasses import fields
from typing import Any, Mapping

from .target_only_progressive_adapt import SFTAPFTConfig


_TOP_LEVEL_KEYS = frozenset(
    {"schema", "run_id", "shared_config", "base_sf_tapft", "rows"}
)
_ROW_KEYS = frozenset({"row_id", "candidate_id", "gpu", "overrides"})


def validate_slim_matrix(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or frozenset(value) != _TOP_LEVEL_KEYS:
        raise ValueError("slim matrix top-level allowlist mismatch")
    matrix = dict(value)
    if matrix["schema"] != "cvs.sf_tapft.slim_matrix.v1":
        raise ValueError("slim matrix schema mismatch")
    if not isinstance(matrix["run_id"], str) or not matrix["run_id"].strip():
        raise ValueError("slim matrix run_id must be non-empty")
    if not isinstance(matrix["shared_config"], Mapping):
        raise ValueError("slim matrix shared_config must be a mapping")
    if not isinstance(matrix["base_sf_tapft"], Mapping):
        raise ValueError("slim matrix base_sf_tapft must be a mapping")
    allowed = {field.name for field in fields(SFTAPFTConfig)}
    if set(matrix["base_sf_tapft"]).difference(allowed):
        raise ValueError("slim matrix base_sf_tapft contains unknown fields")
    rows = matrix["rows"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("slim matrix rows must be a non-empty list")
    row_ids: set[str] = set()
    candidate_ids: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping) or frozenset(row) != _ROW_KEYS:
            raise ValueError("slim matrix row allowlist mismatch")
        row_id = row["row_id"]
        candidate_id = row["candidate_id"]
        if not isinstance(row_id, str) or not row_id.strip() or row_id in row_ids:
            raise ValueError("slim matrix row_id must be non-empty and unique")
        if (
            not isinstance(candidate_id, str)
            or not candidate_id.strip()
            or candidate_id in candidate_ids
        ):
            raise ValueError("slim matrix candidate_id must be non-empty and unique")
        gpu = row["gpu"]
        if isinstance(gpu, bool) or not isinstance(gpu, int) or gpu < 0 or gpu > 7:
            raise ValueError("slim matrix gpu must be an integer from 0 to 7")
        overrides = row["overrides"]
        if not isinstance(overrides, Mapping) or set(overrides).difference(allowed):
            raise ValueError("slim matrix override contains an unknown field")
        row_ids.add(row_id)
        candidate_ids.add(candidate_id)
    return matrix


def build_row_config(
    matrix: Mapping[str, Any], row_id: str
) -> tuple[dict[str, Any], int]:
    checked = validate_slim_matrix(matrix)
    matches = [row for row in checked["rows"] if row["row_id"] == row_id]
    if len(matches) != 1:
        raise ValueError(f"unknown slim matrix row_id: {row_id}")
    row = matches[0]
    sf_tapft = dict(checked["base_sf_tapft"])
    sf_tapft.update(dict(row["overrides"]))
    config = dict(checked["shared_config"])
    config["candidate_id"] = row["candidate_id"]
    config["sf_tapft"] = sf_tapft
    return config, int(row["gpu"])


__all__ = ["build_row_config", "validate_slim_matrix"]
