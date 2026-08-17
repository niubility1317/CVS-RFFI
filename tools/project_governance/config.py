"""Validated, versioned configuration for project asset inventory."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from .models import Location


CONFIG_SCHEMA_VERSION = 1
DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "project_governance_inventory_v1.json"
)

_EXPECTED_ROOT_IDS = {
    Location.LOCAL: "TYPE10_7",
    Location.N607: "N607_CVS_SINCNET",
}
_EXPECTED_ROOTS = {
    Location.LOCAL: "E:/type10-7",
    Location.N607: "/home/szu2070436088/2510044040/CV-SincNet",
}
_LIMIT_SECTIONS = {
    "discovery": (
        "control_evidence_max_depth",
        "hash_max_bytes",
        "text_read_max_bytes",
    ),
    "output": ("git_file_max_bytes", "git_scan_max_bytes"),
}
_WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]|^\\\\")


@dataclass(frozen=True)
class CarrierSurface:
    relative_path: str
    status: str


@dataclass(frozen=True)
class LocationConfig:
    location: Location
    root_id: str
    root: str
    carrier_surfaces: tuple[CarrierSurface, ...]


@dataclass(frozen=True)
class DiscoveryConfig:
    control_evidence_max_depth: int
    hash_max_bytes: int
    text_read_max_bytes: int


@dataclass(frozen=True)
class OutputConfig:
    git_file_max_bytes: int
    git_scan_max_bytes: int


@dataclass(frozen=True)
class GovernanceConfig:
    schema_version: int
    local: LocationConfig
    n607: LocationConfig
    discovery: DiscoveryConfig
    output: OutputConfig

    def for_location(self, location: Location | str) -> LocationConfig:
        selected = _coerce_location(location)
        return self.local if selected is Location.LOCAL else self.n607


def _coerce_location(value: Location | str) -> Location:
    if isinstance(value, Location):
        return value
    try:
        return Location(str(value).upper())
    except ValueError as exc:
        raise ValueError(f"unknown location: {value!r}") from exc


def _is_absolute_carrier(value: str) -> bool:
    return value.startswith("/") or _WINDOWS_ABSOLUTE.match(value) is not None


def _validate_carrier_path(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("carrier surface must be a non-empty relative string")
    normalized = value.replace("\\", "/")
    if _is_absolute_carrier(value) or _is_absolute_carrier(normalized):
        raise ValueError(f"carrier surface must be relative: {value!r}")
    parts = PurePosixPath(normalized).parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"carrier surface contains an invalid component: {value!r}")
    return "/".join(parts)


def _validate_root(location: Location, root_id: Any, root: Any) -> tuple[str, str]:
    expected_id = _EXPECTED_ROOT_IDS[location]
    if root_id != expected_id:
        raise ValueError(
            f"{location.value} root_id does not match requested location: {root_id!r}"
        )
    if not isinstance(root, str) or not root:
        raise ValueError(f"{location.value} root must be a non-empty absolute path")

    if location is Location.LOCAL:
        if not _WINDOWS_ABSOLUTE.match(root):
            raise ValueError(f"LOCAL root does not match requested location: {root!r}")
        # Constructing PureWindowsPath catches malformed drive roots while
        # avoiding any filesystem access during validation.
        if not PureWindowsPath(root).is_absolute():
            raise ValueError(f"LOCAL root does not match requested location: {root!r}")
    else:
        if not root.startswith("/") or not PurePosixPath(root).is_absolute():
            raise ValueError(f"N607 root does not match requested location: {root!r}")
    normalized_root = root.replace("\\", "/") if location is Location.LOCAL else root
    expected_root = _EXPECTED_ROOTS[location]
    if (
        normalized_root.casefold() if location is Location.LOCAL else normalized_root
    ) != (expected_root.casefold() if location is Location.LOCAL else expected_root):
        raise ValueError(f"{location.value} root does not match requested location: {root!r}")
    return normalized_root, root_id


def _surface_status(location: Location, root: str, relative_path: str) -> str:
    if location is Location.N607:
        # The configuration loader is intentionally local and read-only.  A
        # remote surface is left as NOT_PRESENT until the N607 collector
        # verifies it over its bounded read-only connection.
        return "NOT_PRESENT"
    try:
        path = Path(root)
        for component in relative_path.split("/"):
            path /= component
        return "PRESENT" if path.exists() else "NOT_PRESENT"
    except (OSError, ValueError):
        return "NOT_PRESENT"


def _parse_location_config(
    location: Location, payload: Any, *, probe_local_paths: bool
) -> LocationConfig:
    if not isinstance(payload, dict):
        raise ValueError(f"{location.value} configuration must be an object")
    root, root_id = _validate_root(location, payload.get("root_id"), payload.get("root"))
    carrier_values = payload.get("carrier_surfaces")
    if not isinstance(carrier_values, list):
        raise ValueError(f"{location.value} carrier_surfaces must be a list")

    surfaces: list[CarrierSurface] = []
    for raw_value in carrier_values:
        relative_path = _validate_carrier_path(raw_value)
        status = (
            _surface_status(location, root, relative_path)
            if probe_local_paths or location is Location.N607
            else "NOT_PRESENT"
        )
        surfaces.append(CarrierSurface(relative_path=relative_path, status=status))
    return LocationConfig(
        location=location,
        root_id=root_id,
        root=root,
        carrier_surfaces=tuple(surfaces),
    )


def _parse_limits(section_name: str, payload: Any) -> dict[str, int]:
    if not isinstance(payload, dict):
        raise ValueError(f"{section_name} configuration must be an object")
    limits: dict[str, int] = {}
    for key in _LIMIT_SECTIONS[section_name]:
        value = payload.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{section_name}.{key} must be positive")
        limits[key] = value
    return limits


def load_config(
    path: str | Path | None = None,
    location: Location | str | None = None,
    *,
    probe_local_paths: bool = True,
) -> GovernanceConfig:
    """Load and validate the inventory configuration.

    ``location`` is an optional selector for callers that need one root, but
    both fixed root sections are validated so a malformed unused section
    cannot silently enter a later scan.  Surface probing is a single local
    existence check only; no directory enumeration or remote connection is
    performed here.
    """

    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to read governance config: {config_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("governance config must be a JSON object")

    schema_version = payload.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != CONFIG_SCHEMA_VERSION
    ):
        raise ValueError(f"unknown schema_version: {schema_version!r}")

    selected_location = _coerce_location(location) if location is not None else None
    local = _parse_location_config(
        Location.LOCAL,
        payload.get("local"),
        probe_local_paths=probe_local_paths and selected_location in {None, Location.LOCAL},
    )
    n607 = _parse_location_config(
        Location.N607,
        payload.get("n607"),
        probe_local_paths=False,
    )
    discovery_limits = _parse_limits("discovery", payload.get("discovery"))
    output_limits = _parse_limits("output", payload.get("output"))

    return GovernanceConfig(
        schema_version=schema_version,
        local=local,
        n607=n607,
        discovery=DiscoveryConfig(**discovery_limits),
        output=OutputConfig(**output_limits),
    )


__all__ = [
    "CONFIG_SCHEMA_VERSION",
    "CarrierSurface",
    "DEFAULT_CONFIG_PATH",
    "DiscoveryConfig",
    "GovernanceConfig",
    "LocationConfig",
    "OutputConfig",
    "load_config",
]
