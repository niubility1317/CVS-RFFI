from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any, Dict

import numpy as np
import torch


def load_config(path: str | None) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    try:
        import yaml

        data = yaml.safe_load(text)
    except ImportError as exc:
        data = _simple_yaml_load(text)
    return data or {}


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value in ("null", "None", "~", ""):
        return None
    if value in ("true", "True"):
        return True
    if value in ("false", "False"):
        return False
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part.strip()) for part in inner.split(",")]
    try:
        if any(ch in value for ch in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value.strip("\"'")


def _simple_yaml_load(text: str) -> Dict[str, Any]:
    """Tiny fallback parser for the simple mapping YAML configs in this repo."""

    root: Dict[str, Any] = {}
    stack: list[tuple[int, Dict[str, Any]]] = [(-1, root)]
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value == "":
            child: Dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(value)
    return root


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def save_json(obj: Dict[str, Any], path: str) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def default_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=None, help="YAML config path.")
    parser.add_argument("--output_dir", default=None, help="Override output directory.")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs.")
    parser.add_argument("--device", default=None, help="cuda/cpu override.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override.")
    return parser


def get_nested(cfg: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur
