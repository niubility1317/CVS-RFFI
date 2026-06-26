from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json_config(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("config must be a JSON object")
    return data


def contains_unspecified(value: object) -> bool:
    if isinstance(value, str):
        return "paper-unspecified" in value
    if isinstance(value, list):
        return any(contains_unspecified(item) for item in value)
    if isinstance(value, dict):
        return any(contains_unspecified(item) for item in value.values())
    return False


def contains_unresolved_placeholder(value: object) -> bool:
    placeholders = (
        "two source receivers",
        "held-out target receiver",
        "source WiSig days/domains",
        "held-out WiSig days/domains",
        "must be filled",
    )
    if isinstance(value, str):
        return any(token in value for token in placeholders)
    if isinstance(value, list):
        return any(contains_unresolved_placeholder(item) for item in value)
    if isinstance(value, dict):
        return any(contains_unresolved_placeholder(item) for item in value.values())
    return False
