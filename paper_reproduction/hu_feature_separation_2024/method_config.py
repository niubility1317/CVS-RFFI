from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class MethodConfigError(ValueError):
    pass


@dataclass(frozen=True)
class MethodConfig:
    raw: dict[str, Any]

    def method_metadata(self) -> dict[str, Any]:
        return {
            "method": self.raw["method"],
            "parity_status": self.raw["parity_status"],
            "unpublished_defaults": self.raw["unpublished_defaults"],
        }


def load_method_config(path: Path | None = None) -> MethodConfig:
    config_path = path or Path(__file__).with_name("strict_method.json")
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MethodConfigError(f"cannot load method config: {config_path}") from exc
    if raw.get("method") != "hu_feature_separation_2024":
        raise MethodConfigError("method must be hu_feature_separation_2024")
    if raw.get("parity_status") != "PAPER_METHOD_PARITY_WITH_UNPUBLISHED_DEFAULTS":
        raise MethodConfigError("unsupported parity_status")
    defaults = raw.get("unpublished_defaults")
    if not isinstance(defaults, dict) or not defaults:
        raise MethodConfigError("unpublished_defaults must be a nonempty object")
    for name, item in defaults.items():
        if not isinstance(item, dict) or item.get("status") != "UNPUBLISHED_DEFAULT":
            raise MethodConfigError(f"unpublished default {name} must have UNPUBLISHED_DEFAULT status")
        if "value" not in item:
            raise MethodConfigError(f"unpublished default {name} must have a value")
        if not isinstance(item.get("rationale"), str) or not item["rationale"].strip():
            raise MethodConfigError(f"unpublished default {name} must have a rationale")
    return MethodConfig(raw=raw)
