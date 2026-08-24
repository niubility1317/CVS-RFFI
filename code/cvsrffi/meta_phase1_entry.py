"""Executable source-only Phase1 entrypoint for the frozen meta-adapter run.

This module owns orchestration only.  The typed Task2 carriers, Task6 inner
loop, Task7 trainer and Task4 checkpoint migration remain the implementation
authorities; this entrypoint wires them together and persists only artifacts
that were produced by the real path.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import random
import traceback
from types import SimpleNamespace
from typing import Any, Mapping, MutableMapping, Sequence

import numpy as np
import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "phase1_adv3b02_meta_adapter_tri_r4_v1_s392002_20260824.json"
META_PHASE1_SCHEMA = "cvs.phase1.meta_adapter.tri_r4.v1"
REGISTERED_META_PHASE1_SCHEMAS = (
    META_PHASE1_SCHEMA,
    "cvs.phase1.meta_adapter.r4.v1",
)
CANONICAL_SOURCE_ROLES = {
    "L_s": 0.07,
    "U_s": 0.63,
    "V_cal": 0.15,
    "V_select": 0.15,
}
CANONICAL_SOURCE_RECEIVER_IDS = (0, 1, 2, 3, 4, 5, 6)
CANONICAL_SOURCE_SPLIT = "tx_rx_day_1_7_2"
CANONICAL_SOURCE_DAYS = (0, 1)
CANONICAL_CLEAN_TEST_DAYS = (2, 3)
CANONICAL_WISIG_EQUALIZED = 1
CANONICAL_WISIG_OUT_LEN = 256
CANONICAL_WISIG_DOMAIN = "rx_day"
CANONICAL_WISIG_MAX_DAY123_PER_COMBO = 0
CANONICAL_SAT_FS_HZ = 25.0e6
CANONICAL_SAT_FC_HZ = 2.462e9
CANONICAL_ADAPTER = {
    "rank": 4,
    "sites": ("time", "freq", "fusion"),
    "inner_steps": 3,
    "deployment_max_steps": 5,
    "source_diagnostic_max_steps": 10,
}
REGISTERED_ADAPTER_SITE_PROFILES = (
    CANONICAL_ADAPTER["sites"],
    ("fusion",),
)
CANONICAL_EPISODE_WEIGHTS = {
    "Q_SAME_DOMAIN": 0.40,
    "Q_RX_HOLDOUT": 0.20,
    "Q_DAY_CHANNEL_HOLDOUT": 0.15,
    "Q_CLEAN_TO_LEO": 0.15,
    "Q_LEO_CROSS": 0.10,
}
CANONICAL_K_CHOICES = (1, 2, 5, 10)
CANONICAL_EVALUATE_STEPS = (0, 1, 3, 5, 10)
CANONICAL_DEFAULT_META_TRAIN_STEPS = 200
CANONICAL_DEFAULT_META_EVAL_EPISODES = 4
CANONICAL_DEFAULT_META_QUERY_PER_CLASS = 2
CANONICAL_CANDIDATE_PLAN = (
    ("P0", "frozen_base", False),
    ("P1", "random_adapter", False),
    ("P2", "supervised_adapter", False),
    ("P3", "fomaml_fixed_lr", False),
    ("P4", "fomaml_meta_sgd", True),
)


def _numpy_array_abi_safe(value: torch.Tensor, *, dtype: Any) -> np.ndarray:
    """Bridge bounded Phase1 state without Torch's NumPy C-API boundary.

    N607 pairs NumPy 2.x with a Torch build whose ``Tensor.numpy()`` can return
    an ndarray owned by a different NumPy type identity.  NumPy's
    ``__array_function__`` dispatcher then rejects that array in ``np.savez``.
    Frozen class prototypes are tiny source-only state, so a detached list
    bridge is bounded and preserves the requested dtype without touching any
    training or query path.
    """

    return np.asarray(value.detach().cpu().tolist(), dtype=dtype)


def _as_mapping(value: Any, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    return value


def _finite_number(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be a finite number")
    return result


def _require_token(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty path/token")
    return value.strip()


def _require_int(value: Any, *, field_name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field_name} must be an integer >= {minimum}")
    return int(value)


def _check_exact_float_map(
    value: Any,
    expected: Mapping[str, float],
    *,
    field_name: str,
    tolerance: float = 1.0e-12,
) -> dict[str, float]:
    mapping = _as_mapping(value, field_name=field_name)
    if set(mapping) != set(expected):
        raise ValueError(
            f"{field_name} keys must be exactly {tuple(expected)!r}; "
            f"got {tuple(mapping)!r}"
        )
    result: dict[str, float] = {}
    for key, target in expected.items():
        actual = _finite_number(mapping[key], field_name=f"{field_name}.{key}")
        if abs(actual - float(target)) > tolerance:
            raise ValueError(
                f"{field_name}.{key} must be frozen at {target:g}; got {actual:g}"
            )
        result[key] = actual
    return result


def _check_exact_int_sequence(
    value: Any,
    expected: Sequence[int],
    *,
    field_name: str,
) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} must be an integer sequence")
    actual = tuple(_require_int(item, field_name=f"{field_name} entry", minimum=0) for item in value)
    frozen = tuple(int(item) for item in expected)
    if actual != frozen:
        raise ValueError(f"{field_name} must be frozen at {list(frozen)!r}; got {list(actual)!r}")
    return actual


def _check_source_receiver_ids(value: Any) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("source_receiver_ids must be an explicit non-empty integer sequence")
    result = tuple(_require_int(item, field_name="source_receiver_ids entry", minimum=0) for item in value)
    if len(set(result)) != len(result):
        raise ValueError("source_receiver_ids must not contain duplicates")
    if result != CANONICAL_SOURCE_RECEIVER_IDS:
        raise ValueError(
            "source_receiver_ids must match the frozen WiSig source split "
            f"{list(CANONICAL_SOURCE_RECEIVER_IDS)!r}; got {list(result)!r}"
        )
    return result


def _validate_model_config(value: Any) -> dict[str, Any]:
    """Validate the small frozen model-builder surface used by Task8."""

    raw = dict(_as_mapping(value, field_name="model"))
    builder = str(raw.get("builder", "dual")).strip().lower()
    if builder not in {"single", "dual"}:
        raise ValueError("model.builder must be 'single' or 'dual'")
    raw["builder"] = builder
    for name in ("num_classes", "input_len"):
        if name in raw:
            _require_int(raw[name], field_name=f"model.{name}", minimum=1)
    if builder == "dual":
        _require_int(raw.get("num_domains"), field_name="model.num_domains", minimum=1)
    if "sample_rate_hz" in raw:
        _finite_number(raw["sample_rate_hz"], field_name="model.sample_rate_hz")
    for name in ("model_size", "dataset", "model_variant", "branch_ablation", "domain_branch_ablation"):
        if name in raw and not isinstance(raw[name], str):
            raise ValueError(f"model.{name} must be a string")
    return raw


def _validate_wisig_view_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the immutable WiSig view construction contract."""

    equalized = config["wisig_equalized"]
    if isinstance(equalized, bool):
        raise ValueError("wisig_equalized must be frozen at 1")
    if isinstance(equalized, str):
        if equalized.strip() != "1":
            raise ValueError("wisig_equalized must be frozen at 1")
    elif not isinstance(equalized, int) or int(equalized) != CANONICAL_WISIG_EQUALIZED:
        raise ValueError("wisig_equalized must be frozen at 1")
    out_len = _require_int(
        config["wisig_out_len"], field_name="wisig_out_len", minimum=1
    )
    if out_len != CANONICAL_WISIG_OUT_LEN:
        raise ValueError(
            f"wisig_out_len must be frozen at {CANONICAL_WISIG_OUT_LEN}; got {out_len}"
        )
    domain = _require_token(config["wisig_domain"], field_name="wisig_domain").lower()
    if domain != CANONICAL_WISIG_DOMAIN:
        raise ValueError(
            f"wisig_domain must be frozen at {CANONICAL_WISIG_DOMAIN!r}; got {domain!r}"
        )
    max_per_combo = _require_int(
        config["wisig_max_day123_per_combo"],
        field_name="wisig_max_day123_per_combo",
        minimum=0,
    )
    if max_per_combo != CANONICAL_WISIG_MAX_DAY123_PER_COMBO:
        raise ValueError(
            "wisig_max_day123_per_combo must be frozen at "
            f"{CANONICAL_WISIG_MAX_DAY123_PER_COMBO}; got {max_per_combo}"
        )
    return {
        "wisig_equalized": CANONICAL_WISIG_EQUALIZED,
        "wisig_out_len": CANONICAL_WISIG_OUT_LEN,
        "wisig_domain": CANONICAL_WISIG_DOMAIN,
        "wisig_max_day123_per_combo": CANONICAL_WISIG_MAX_DAY123_PER_COMBO,
    }


def validate_meta_phase1_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize the Task8 frozen source-only configuration."""

    if not isinstance(config, Mapping):
        raise TypeError("meta Phase1 config must be a mapping")

    # Reject target-bearing fields before accepting any optional metadata.  A
    # target receiver is never needed to construct the Phase1 source split.
    for key in config:
        marker = str(key).lower().replace("-", "_")
        if "target_receiver" in marker or marker in {"target_receivers", "target_receiver_ids"}:
            raise ValueError("target receiver fields are forbidden in the Phase1 config")

    required = {
        "schema",
        "run_id",
        "seed",
        "base_checkpoint",
        "wisig_pkl",
        "source_receiver_ids",
        "source_split",
        "source_days",
        "clean_test_days",
        "wisig_equalized",
        "wisig_out_len",
        "wisig_domain",
        "wisig_max_day123_per_combo",
        "source_roles",
        "adapter",
        "episode_weights",
        "k_choices",
        "meta_batch_size",
        "phase1c_backbone_lr_ratio",
        "evaluate_steps",
        "meta_train_steps",
        "meta_eval_episodes",
        "meta_query_per_class",
        "candidate_plan",
        "model",
    }
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(f"meta Phase1 config is missing required fields: {missing}")

    schema = _require_token(config["schema"], field_name="schema")
    if schema not in REGISTERED_META_PHASE1_SCHEMAS:
        raise ValueError(
            f"schema must match one of {REGISTERED_META_PHASE1_SCHEMAS!r}; got {schema!r}"
        )
    run_id = _require_token(config["run_id"], field_name="run_id")
    seed = _require_int(config["seed"], field_name="seed", minimum=0)
    base_checkpoint = _require_token(config["base_checkpoint"], field_name="base_checkpoint")
    wisig_pkl = _require_token(config["wisig_pkl"], field_name="wisig_pkl")
    source_receiver_ids = _check_source_receiver_ids(config["source_receiver_ids"])
    source_split = _require_token(config["source_split"], field_name="source_split")
    if source_split != CANONICAL_SOURCE_SPLIT:
        raise ValueError(
            f"source_split must be frozen at {CANONICAL_SOURCE_SPLIT!r}; got {source_split!r}"
        )
    source_days = _check_exact_int_sequence(
        config["source_days"], CANONICAL_SOURCE_DAYS, field_name="source_days"
    )
    clean_test_days = _check_exact_int_sequence(
        config["clean_test_days"],
        CANONICAL_CLEAN_TEST_DAYS,
        field_name="clean_test_days",
    )
    if set(source_days).intersection(clean_test_days):
        raise ValueError("clean_test_days must be disjoint from source_days")
    wisig_view = _validate_wisig_view_config(config)

    source_roles = _check_exact_float_map(
        config["source_roles"],
        CANONICAL_SOURCE_ROLES,
        field_name="source_roles",
    )
    adapter_raw = _as_mapping(config["adapter"], field_name="adapter")
    if set(adapter_raw) != set(CANONICAL_ADAPTER):
        raise ValueError(
            f"adapter keys must be exactly {tuple(CANONICAL_ADAPTER)!r}; "
            f"got {tuple(adapter_raw)!r}"
        )
    adapter = {
        "rank": _require_int(adapter_raw["rank"], field_name="adapter.rank", minimum=1),
        "sites": tuple(str(item) for item in adapter_raw["sites"])
        if isinstance(adapter_raw["sites"], (list, tuple))
        else (),
        "inner_steps": _require_int(adapter_raw["inner_steps"], field_name="adapter.inner_steps", minimum=1),
        "deployment_max_steps": _require_int(
            adapter_raw["deployment_max_steps"], field_name="adapter.deployment_max_steps", minimum=1
        ),
        "source_diagnostic_max_steps": _require_int(
            adapter_raw["source_diagnostic_max_steps"],
            field_name="adapter.source_diagnostic_max_steps",
            minimum=1,
        ),
    }
    if adapter["rank"] != CANONICAL_ADAPTER["rank"]:
        raise ValueError("adapter.rank must be frozen at 4")
    if adapter["sites"] not in REGISTERED_ADAPTER_SITE_PROFILES:
        raise ValueError(
            "adapter.sites must match one of the registered profiles "
            f"{[list(profile) for profile in REGISTERED_ADAPTER_SITE_PROFILES]!r}; "
            f"got {list(adapter['sites'])!r}"
        )
    if schema == META_PHASE1_SCHEMA and adapter["sites"] != CANONICAL_ADAPTER["sites"]:
        raise ValueError("the legacy tri_r4 schema requires time,freq,fusion adapter sites")
    for key in ("inner_steps", "deployment_max_steps", "source_diagnostic_max_steps"):
        if adapter[key] != CANONICAL_ADAPTER[key]:
            raise ValueError(f"adapter.{key} must be frozen at {CANONICAL_ADAPTER[key]}")

    episode_weights = _check_exact_float_map(
        config["episode_weights"],
        CANONICAL_EPISODE_WEIGHTS,
        field_name="episode_weights",
    )
    if abs(sum(episode_weights.values()) - 1.0) > 1.0e-12:
        raise ValueError("episode_weights must sum to 1.0")
    k_choices = _check_exact_int_sequence(config["k_choices"], CANONICAL_K_CHOICES, field_name="k_choices")
    meta_batch_size = _require_int(config["meta_batch_size"], field_name="meta_batch_size", minimum=1)
    if meta_batch_size != 4:
        raise ValueError("meta_batch_size must be frozen at 4")
    phase1c_ratio = _finite_number(
        config["phase1c_backbone_lr_ratio"], field_name="phase1c_backbone_lr_ratio"
    )
    if abs(phase1c_ratio - 0.05) > 1.0e-12:
        raise ValueError("phase1c_backbone_lr_ratio must be frozen at 0.05")
    evaluate_steps = _check_exact_int_sequence(
        config["evaluate_steps"], CANONICAL_EVALUATE_STEPS, field_name="evaluate_steps"
    )
    meta_train_steps = _require_int(
        config["meta_train_steps"], field_name="meta_train_steps", minimum=1
    )
    meta_eval_episodes = _require_int(
        config["meta_eval_episodes"], field_name="meta_eval_episodes", minimum=2
    )
    meta_query_per_class = _require_int(
        config["meta_query_per_class"], field_name="meta_query_per_class", minimum=1
    )
    raw_candidate_plan = config["candidate_plan"]
    if not isinstance(raw_candidate_plan, (list, tuple)):
        raise ValueError("candidate_plan must be the frozen P0-P4 sequence")
    candidate_plan: list[dict[str, Any]] = []
    for index, expected in enumerate(CANONICAL_CANDIDATE_PLAN):
        if index >= len(raw_candidate_plan) or not isinstance(raw_candidate_plan[index], Mapping):
            raise ValueError("candidate_plan must contain exactly P0,P1,P2,P3,P4")
        row = dict(raw_candidate_plan[index])
        if set(row) != {"candidate_id", "training_mode", "learn_step_sizes"}:
            raise ValueError("candidate_plan rows must contain candidate_id,training_mode,learn_step_sizes")
        actual = (
            str(row["candidate_id"]),
            str(row["training_mode"]),
            row["learn_step_sizes"],
        )
        if actual != expected:
            raise ValueError(
                f"candidate_plan row {index} must be frozen at {expected!r}; got {actual!r}"
            )
        candidate_plan.append(dict(row))
    if len(raw_candidate_plan) != len(CANONICAL_CANDIDATE_PLAN):
        raise ValueError("candidate_plan must contain exactly P0,P1,P2,P3,P4")
    active_candidate_id = str(config.get("active_candidate_id", "P4"))
    if active_candidate_id not in {row[0] for row in CANONICAL_CANDIDATE_PLAN[1:]}:
        raise ValueError("active_candidate_id must be one of P1,P2,P3,P4")
    model = _validate_model_config(config["model"])

    normalized: dict[str, Any] = dict(config)
    normalized.update(
        {
            "schema": schema,
            "run_id": run_id,
            "seed": seed,
            "base_checkpoint": base_checkpoint,
            "wisig_pkl": wisig_pkl,
            "source_receiver_ids": list(source_receiver_ids),
            "source_split": source_split,
            "source_days": list(source_days),
            "clean_test_days": list(clean_test_days),
            **wisig_view,
            "source_roles": source_roles,
            "adapter": {
                "rank": adapter["rank"],
                "sites": list(adapter["sites"]),
                "inner_steps": adapter["inner_steps"],
                "deployment_max_steps": adapter["deployment_max_steps"],
                "source_diagnostic_max_steps": adapter["source_diagnostic_max_steps"],
            },
            "episode_weights": episode_weights,
            "k_choices": list(k_choices),
            "meta_batch_size": meta_batch_size,
            "phase1c_backbone_lr_ratio": phase1c_ratio,
            "evaluate_steps": list(evaluate_steps),
            "meta_train_steps": meta_train_steps,
            "meta_eval_episodes": meta_eval_episodes,
            "meta_query_per_class": meta_query_per_class,
            "candidate_plan": candidate_plan,
            "active_candidate_id": active_candidate_id,
            "model": model,
        }
    )
    return normalized


def load_meta_phase1_config(path: os.PathLike[str] | str) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"meta Phase1 config not found: {config_path}")
    try:
        with config_path.open("r", encoding="utf-8-sig") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in meta Phase1 config: {config_path}") from exc
    return validate_meta_phase1_config(payload)


def _meta_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--use_cvs_meta_adapter", dest="use_cvs_meta_adapter", action="store_true")
    parser.add_argument("--no_use_cvs_meta_adapter", dest="use_cvs_meta_adapter", action="store_false")
    parser.set_defaults(
        use_cvs_meta_adapter=False,
        meta_adapter_rank=4,
        meta_adapter_sites="time,freq,fusion",
        meta_inner_steps=3,
        meta_inner_max_steps=5,
        meta_config=str(DEFAULT_CONFIG_PATH),
        meta_output_root="",
    )
    parser.add_argument("--meta_adapter_rank", type=int)
    parser.add_argument("--meta_adapter_sites", type=str)
    parser.add_argument("--meta_inner_steps", type=int)
    parser.add_argument("--meta_inner_max_steps", type=int)
    parser.add_argument("--meta_config", type=str)
    parser.add_argument("--meta_output_root", type=str)
    return parser


def parse_args_for_test(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the small frozen CLI surface used by Task8 tests and tooling."""

    return _meta_cli_parser().parse_args(list(argv) if argv is not None else None)


def _parse_index_list(value: Any, *, field_name: str, default: Sequence[int]) -> list[int]:
    if value is None or str(value).strip() == "":
        return [int(item) for item in default]
    if isinstance(value, (list, tuple)):
        raw = list(value)
    else:
        raw = [part.strip() for part in str(value).split(",") if part.strip()]
    result: list[int] = []
    for item in raw:
        if isinstance(item, bool):
            raise ValueError(f"{field_name} must contain integer indices")
        try:
            number = int(item)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} must contain integer indices") from exc
        if number < 0:
            raise ValueError(f"{field_name} must contain non-negative indices")
        result.append(number)
    return result


def _seed_everything(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _validate_frozen_wisig_args(args: Any, config: Mapping[str, Any]) -> None:
    expected_equalized = int(config["wisig_equalized"])
    requested_equalized = getattr(args, "wisig_equalized", expected_equalized)
    try:
        if isinstance(requested_equalized, str) and requested_equalized.strip().lower() == "both":
            raise ValueError
        requested_equalized = int(requested_equalized)
    except (TypeError, ValueError) as exc:
        raise ValueError("wisig_equalized must match frozen config value 1") from exc
    if requested_equalized != expected_equalized:
        raise ValueError(
            "wisig_equalized does not match frozen config: "
            f"{requested_equalized!r} != {expected_equalized!r}"
        )

    requested_out_len = getattr(args, "wisig_out_len", CANONICAL_WISIG_OUT_LEN)
    try:
        requested_out_len = int(requested_out_len)
    except (TypeError, ValueError) as exc:
        raise ValueError("wisig_out_len must match frozen config value 256") from exc
    if requested_out_len != int(config["wisig_out_len"]):
        raise ValueError(
            "wisig_out_len does not match frozen config: "
            f"{requested_out_len!r} != {config['wisig_out_len']!r}"
        )

    requested_domain = str(getattr(args, "wisig_domain", CANONICAL_WISIG_DOMAIN)).strip().lower()
    if requested_domain != str(config["wisig_domain"]):
        raise ValueError(
            "wisig_domain does not match frozen config: "
            f"{requested_domain!r} != {config['wisig_domain']!r}"
        )

    requested_max = getattr(args, "wisig_max_day123_per_combo", CANONICAL_WISIG_MAX_DAY123_PER_COMBO)
    try:
        requested_max = int(requested_max)
    except (TypeError, ValueError) as exc:
        raise ValueError("wisig_max_day123_per_combo must match frozen config value 0") from exc
    if requested_max != int(config["wisig_max_day123_per_combo"]):
        raise ValueError(
            "wisig_max_day123_per_combo does not match frozen config: "
            f"{requested_max!r} != {config['wisig_max_day123_per_combo']!r}"
        )
    requested_test_days = _parse_index_list(
        getattr(args, "wisig_test_days", config["clean_test_days"]),
        field_name="wisig_test_days",
        default=config["clean_test_days"],
    )
    if tuple(requested_test_days) != tuple(config["clean_test_days"]):
        raise ValueError(
            "wisig_test_days does not match frozen clean_test_days: "
            f"{requested_test_days!r} != {config['clean_test_days']!r}"
        )


def _iter_dataset_physical_ids(dataset: Any):
    """Yield physical IDs without decoding IQ when a WiSig index is available."""

    sample_index = getattr(dataset, "index", None)
    if sample_index is not None:
        if len(sample_index) != len(dataset):
            raise ValueError("WiSig dataset index length does not match dataset length")
        from dataset_wisig import wisig_physical_sample_id

        for item in sample_index:
            yield str(wisig_physical_sample_id(item))
        return

    for index in range(len(dataset)):
        _x, _y, metadata = _dataset_item(dataset, index)
        yield str(metadata.get("physical_sample_id", ""))


def _source_role_manifest(ds_w: Mapping[str, Any], config: Mapping[str, Any], args: Any) -> dict[str, Any]:
    """Build only the frozen source role split; no target split is opened."""

    if config.get("source_split") != CANONICAL_SOURCE_SPLIT:
        raise ValueError("Phase1 meta entry requires source_split=tx_rx_day_1_7_2")
    if tuple(config.get("source_days", ())) != CANONICAL_SOURCE_DAYS:
        raise ValueError("Phase1 meta entry requires source_days=[0,1]")

    rx_list = list(ds_w.get("rx_list", []))
    for receiver_id in config["source_receiver_ids"]:
        if int(receiver_id) >= len(rx_list):
            raise ValueError(
                f"source_receiver_ids contains {receiver_id}, but WiSig asset exposes "
                f"only {len(rx_list)} receiver entries"
            )
    if "data" not in ds_w:
        # Unit callers may validate routing with a metadata-only asset.  Do not
        # fabricate a split; return the validated source contract instead.
        return {
            "available": False,
            "source_only": True,
            "reason": "WiSig data payload is not loaded; split construction deferred",
            "source_receiver_ids": tuple(config["source_receiver_ids"]),
            "source_roles": dict(config["source_roles"]),
        }

    from dataset_wisig import WiSigCompactDataset, WiSigSubsetDataset
    from SSDG.train_ssdg import split_tx_rx_day_1_7_2_roles

    equalized = int(config["wisig_equalized"])
    configured_days = config["source_days"]
    train_days = _parse_index_list(
        configured_days if configured_days is not None else getattr(args, "wisig_train_days", None),
        field_name="source_days" if configured_days is not None else "wisig_train_days",
        default=(0, 1),
    )
    if not train_days:
        raise ValueError("Phase1 meta entry requires at least one source training day")
    max_per_combo = int(config["wisig_max_day123_per_combo"])
    source_base = WiSigCompactDataset(
        ds_w,
        out_len=int(config["wisig_out_len"]),
        crop_mode="center",
        normalize=True,
        equalized=equalized,
        day_keep=train_days,
        rx_keep=list(config["source_receiver_ids"]),
        domain=str(config["wisig_domain"]),
        max_samples_per_combo=None if max_per_combo <= 0 else max_per_combo,
        seed=int(config["seed"]),
        build_index=True,
    )
    labeled_idx, unlabeled_idx, v_cal_idx, v_select_idx = split_tx_rx_day_1_7_2_roles(
        source_base,
        labeled_ratio=float(config["source_roles"]["L_s"]),
        unlabeled_ratio=float(config["source_roles"]["U_s"]),
        source_cal_ratio=float(config["source_roles"]["V_cal"]),
        source_select_ratio=float(config["source_roles"]["V_select"]),
    )
    role_datasets = {
        "L_s": WiSigSubsetDataset(source_base, labeled_idx, split_source="meta_phase1_L_s"),
        "U_s": WiSigSubsetDataset(source_base, unlabeled_idx, split_source="meta_phase1_U_s"),
        "V_cal": WiSigSubsetDataset(source_base, v_cal_idx, split_source="meta_phase1_V_cal"),
        "V_select": WiSigSubsetDataset(source_base, v_select_idx, split_source="meta_phase1_V_select"),
    }
    clean_test_days = _parse_index_list(
        getattr(args, "wisig_test_days", config["clean_test_days"]),
        field_name="wisig_test_days",
        default=config["clean_test_days"],
    )
    clean_test_dataset = WiSigCompactDataset(
        ds_w,
        out_len=int(config["wisig_out_len"]),
        crop_mode="center",
        normalize=True,
        equalized=equalized,
        day_keep=clean_test_days,
        rx_keep=list(config["source_receiver_ids"]),
        domain=str(config["wisig_domain"]),
        max_samples_per_combo=None if max_per_combo <= 0 else max_per_combo,
        seed=int(config["seed"]),
        build_index=True,
    )
    if len(clean_test_dataset) <= 0:
        raise ValueError("declared clean test dataset is empty")
    role_physical_ids: set[str] = set()
    for dataset in role_datasets.values():
        for physical_id in _iter_dataset_physical_ids(dataset):
            if not physical_id:
                raise ValueError("Phase1 role dataset is missing physical_sample_id")
            role_physical_ids.add(physical_id)
    clean_test_physical_ids: set[str] = set()
    for physical_id in _iter_dataset_physical_ids(clean_test_dataset):
        if not physical_id:
            raise ValueError("declared clean test dataset is missing physical_sample_id")
        if physical_id in clean_test_physical_ids:
            raise ValueError("declared clean test dataset contains duplicate physical_sample_id")
        clean_test_physical_ids.add(physical_id)
    overlap = role_physical_ids.intersection(clean_test_physical_ids)
    if overlap:
        raise ValueError(
            "declared clean test physical IDs overlap Phase1 train/selection roles: "
            f"count={len(overlap)}"
        )
    return {
        "available": True,
        "source_only": True,
        "source_receiver_ids": tuple(config["source_receiver_ids"]),
        "source_days": tuple(train_days),
        "role_sizes": {name: len(dataset) for name, dataset in role_datasets.items()},
        "role_datasets": role_datasets,
        "clean_test_dataset": clean_test_dataset,
        "clean_test_days": tuple(clean_test_days),
        "clean_test_size": len(clean_test_dataset),
        "clean_test_physical_disjoint": True,
        "supervised_training_role": "L_s",
        "unlabeled_role": "U_s",
        "validation_roles": ("V_cal", "V_select"),
        "u_s_labels_for_supervised_loss": False,
    }


def _resolve_config_path(value: str | os.PathLike[str], *, config_source: Any) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    config_path = Path(config_source) if isinstance(config_source, (str, os.PathLike)) else None
    if config_path is not None and config_path.is_file():
        beside_config = (config_path.parent / path).resolve()
        if beside_config.exists():
            return beside_config
    return (PROJECT_ROOT / path).resolve()


def _require_readable_file(path: Path, *, field_name: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"{field_name} is not a readable regular file: {path}")
    try:
        with path.open("rb") as handle:
            handle.read(1)
    except OSError as exc:
        raise OSError(f"{field_name} is not readable: {path}") from exc


def _load_checkpoint_payload(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        from post_stage_common import load_checkpoint

        loaded = load_checkpoint(str(path), device)
        if isinstance(loaded, Mapping):
            return dict(loaded)
    except (ImportError, ModuleNotFoundError):
        pass
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location=device)
    if isinstance(payload, Mapping) and isinstance(payload.get("model"), Mapping):
        return dict(payload)
    if isinstance(payload, Mapping):
        return {"model": dict(payload), "args": {}}
    raise ValueError(f"base checkpoint payload must be a mapping: {path}")


def _model_args_for_run(
    config: Mapping[str, Any],
    ds_w: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    model_config = dict(config.get("model", {}))
    checkpoint_args = payload.get("args", {})
    merged: dict[str, Any] = dict(checkpoint_args) if isinstance(checkpoint_args, Mapping) else {}
    merged.update(model_config)
    merged.pop("builder", None)
    builder = str(model_config.get("builder", "dual")).strip().lower()
    class_count = int(model_config.get("num_classes", len(ds_w.get("tx_list", [])) or 1))
    merged.setdefault("num_classes", class_count)
    merged.setdefault("dataset", "wisig")
    merged.setdefault("input_len", int(getattr(ds_w, "input_len", 256) or 256))
    merged.setdefault("model_size", "M" if builder == "dual" else "M")
    merged.setdefault("model_variant", "lite_d" if builder == "dual" else "base")
    if builder == "dual":
        merged.setdefault(
            "num_domains",
            max(1, len(config["source_receiver_ids"]) * len(config["source_days"])),
        )
        merged.setdefault("id_feature_key", "feat_joint")
        merged.setdefault("dom_feature_key", "feat_imp")
    merged["meta_adapter_rank"] = int(config["adapter"]["rank"])
    merged["meta_adapter_sites"] = ",".join(config["adapter"]["sites"])
    merged["builder"] = builder
    return merged


def _build_meta_model(
    config: Mapping[str, Any],
    ds_w: Mapping[str, Any],
    payload: Mapping[str, Any],
    device: torch.device,
    *,
    adapter_rank: int | None = None,
    adapter_sites: str | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    model_args = _model_args_for_run(config, ds_w, payload)
    if adapter_rank is not None:
        model_args["meta_adapter_rank"] = int(adapter_rank)
        model_args["meta_adapter_sites"] = str(adapter_sites or "")
    builder = str(model_args.pop("builder", "dual")).lower()
    if builder == "dual":
        from model_dual_cvsincnet import build_dual_model

        builder_fn = build_dual_model
    else:
        from model import build_model

        builder_fn = build_model
    import inspect

    allowed = set(inspect.signature(builder_fn).parameters)
    kwargs = {key: value for key, value in model_args.items() if key in allowed}
    # Historical SSDG checkpoints store ``use_mixstyle`` while the repository
    # builder uses the explicit ``mixstyle_on`` spelling.
    if "mixstyle_on" in allowed and "mixstyle_on" not in kwargs and "use_mixstyle" in model_args:
        kwargs["mixstyle_on"] = bool(model_args["use_mixstyle"])
    model = builder_fn(**kwargs).to(device)
    model_args = dict(kwargs)
    model_args["builder"] = builder
    return model, model_args


def _validate_rank4_adapter_model(model: nn.Module, config: Mapping[str, Any]) -> None:
    from cvsrffi.meta_adapter import ResidualMetaAdapter

    rank = int(config["adapter"]["rank"])
    sites = tuple(str(site) for site in config["adapter"]["sites"])
    for site in sites:
        suffix = f"meta_adapter_{site}"
        modules = [
            module
            for name, module in model.named_modules()
            if str(name).endswith(suffix) and isinstance(module, ResidualMetaAdapter)
        ]
        if not modules:
            raise ValueError(f"meta model is missing rank-{rank} {site} adapter")
        if any(int(module.down.out_features) != rank for module in modules):
            observed = sorted({int(module.down.out_features) for module in modules})
            raise ValueError(
                f"meta_adapter_{site} rank drift: expected {rank}, got {observed}"
            )


def _load_legacy_checkpoint_into_meta_model(
    model: nn.Module,
    config: Mapping[str, Any],
    ds_w: Mapping[str, Any],
    payload: Mapping[str, Any],
    device: torch.device,
    *,
    allow_nested_dual_bridge: bool,
) -> tuple[Any, str]:
    """Load Task4 legacy weights, bridging nested dual adapters when needed.

    Task4's loader intentionally allows only top-level ``meta_adapter_*``
    missing keys.  The dual backbone registers the same adapters below
    ``id_backbone`` and ``dom_backbone``, so a rank-4 dual target otherwise
    looks like a legacy checkpoint with missing non-adapter keys.  Build a
    rank-0 shell through the same repository builder, let Task4 validate and
    load the legacy checkpoint there, then copy only shape-compatible base
    tensors into the already validated rank-4 target.  Injected test models
    stay on the direct Task4 path and cannot silently opt into this bridge.
    """

    from cvsrffi.meta_checkpoint import load_legacy_base_for_meta

    direct_error: ValueError | None = None
    try:
        return load_legacy_base_for_meta(model, payload), "direct"
    except ValueError as exc:
        direct_error = exc
        if not allow_nested_dual_bridge or "missing non-adapter keys" not in str(direct_error):
            raise

    from cvsrffi.meta_adapter import ResidualMetaAdapter

    nested_adapter_modules = tuple(
        name
        for name, module in model.named_modules()
        if isinstance(module, ResidualMetaAdapter) and "." in str(name)
    )
    if not nested_adapter_modules:
        raise ValueError(
            "Task4 legacy migration failed and no nested dual adapter bridge is available"
        ) from direct_error

    legacy_model, _legacy_model_args = _build_meta_model(
        config,
        ds_w,
        payload,
        device,
        adapter_rank=0,
        adapter_sites="",
    )
    legacy_audit = load_legacy_base_for_meta(legacy_model, payload)
    legacy_state = legacy_model.state_dict()
    target_state = model.state_dict()
    incompatible_shapes = tuple(
        key
        for key, value in legacy_state.items()
        if key not in target_state
        or not torch.is_tensor(value)
        or not torch.is_tensor(target_state[key])
        or tuple(value.shape) != tuple(target_state[key].shape)
    )
    if incompatible_shapes:
        raise ValueError(
            "rank-0 legacy shell has base tensors incompatible with rank-4 target: "
            f"keys={list(incompatible_shapes)}"
        ) from direct_error

    compatible_state = {
        key: value
        for key, value in legacy_state.items()
        if key in target_state
        and torch.is_tensor(value)
        and torch.is_tensor(target_state[key])
        and tuple(value.shape) == tuple(target_state[key].shape)
    }
    incompatible = model.load_state_dict(compatible_state, strict=False)
    unexpected = tuple(sorted(str(key) for key in incompatible.unexpected_keys))
    missing_non_adapter = tuple(
        sorted(
            str(key)
            for key in incompatible.missing_keys
            if "meta_adapter_" not in str(key)
        )
    )
    if unexpected or missing_non_adapter:
        raise ValueError(
            "rank-0 legacy shell migration did not cover the rank-4 base: "
            f"missing={list(missing_non_adapter)} unexpected={list(unexpected)}"
        ) from direct_error
    return legacy_audit, "rank0_legacy_shell"


def _model_forward(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> Mapping[str, Any]:
    import inspect

    parameters = inspect.signature(model.forward).parameters
    kwargs: dict[str, Any] = {"return_aux": True}
    if "y" in parameters:
        kwargs["y"] = y
    elif "y_tx" in parameters:
        kwargs["y_tx"] = y
    return model(x, **kwargs)


def _embedding_from_output(output: Mapping[str, Any]) -> torch.Tensor:
    for key in ("z_id", "feat_cls", "id_feat_cls", "id_feat_joint", "feat_joint", "base"):
        value = output.get(key)
        if torch.is_tensor(value) and value.ndim == 2:
            return value
    raise ValueError("meta Phase1 model output has no identity embedding")


def _dataset_item(dataset: Any, index: int) -> tuple[torch.Tensor, int, Mapping[str, Any]]:
    item = dataset[int(index)]
    if not isinstance(item, (tuple, list)) or len(item) < 3:
        raise ValueError("source role dataset item must contain x,y,domain,metadata")
    x = item[0]
    y = item[1]
    metadata = item[-1]
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)
    if not isinstance(metadata, Mapping):
        raise ValueError("source role dataset item metadata must be a mapping")
    return x.detach().float(), int(y), metadata


_SOURCE_META_VIEWS = ("clean", "leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")


def _build_refs(dataset: Any, role: str) -> tuple[tuple[Any, ...], Any]:
    from cvsrffi.meta_episodes import MetaSampleRef

    refs: list[MetaSampleRef] = []
    sample_index = getattr(dataset, "index", None)
    if sample_index is not None:
        if len(sample_index) != len(dataset):
            raise ValueError("WiSig dataset index length does not match dataset length")
        from dataset_wisig import wisig_capture_block_id, wisig_physical_sample_id

        base = getattr(dataset, "base", None)
        capture_block_size = int(
            getattr(dataset, "capture_block_size", getattr(base, "capture_block_size", 8))
        )

        def iter_rows():
            for index, item in enumerate(sample_index):
                required = ("tx_i", "rx_i", "day_i", "eq_i", "sig_i")
                if any(not hasattr(item, key) for key in required):
                    raise ValueError("WiSig index item is missing required source metadata")
                yield (
                    int(index),
                    int(item.tx_i),
                    {
                        "rx_i": int(item.rx_i),
                        "day_i": int(item.day_i),
                        "eq_i": int(item.eq_i),
                        "capture_block_i": int(
                            wisig_capture_block_id(item, capture_block_size)
                        ),
                        "physical_sample_id": str(wisig_physical_sample_id(item)),
                    },
                )
    else:
        def iter_rows():
            for index in range(len(dataset)):
                _x, y, meta = _dataset_item(dataset, index)
                yield int(index), int(y), meta

    for index, y, meta in iter_rows():
        required = ("rx_i", "day_i", "eq_i", "capture_block_i", "physical_sample_id")
        if any(key not in meta for key in required):
            raise ValueError(f"{role} dataset metadata is missing one of {required!r}")
        for view in _SOURCE_META_VIEWS:
            refs.append(
                MetaSampleRef(
                    dataset_index=int(index),
                    tx_i=int(y),
                    rx_i=int(meta["rx_i"]),
                    day_i=int(meta["day_i"]),
                    eq_i=int(meta["eq_i"]),
                    capture_block_i=int(meta["capture_block_i"]),
                    physical_sample_id=str(meta["physical_sample_id"]),
                    role=str(role),
                    view=view,
                )
            )
    if not refs:
        raise ValueError(f"{role} source role has no rows for meta episodes")
    return tuple(refs), dataset


def _stable_view_seed(base_seed: int, physical_sample_id: str, view: str) -> int:
    payload = f"{int(base_seed)}|{str(physical_sample_id)}|{str(view)}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False) & 0x7FFF_FFFF_FFFF_FFFF


def _materialize_ref_view(
    x: torch.Tensor,
    ref: Any,
    *,
    view_seed: int,
) -> torch.Tensor:
    view = str(ref.view)
    if view == "clean":
        return x.detach().float().clone()
    if view not in _SOURCE_META_VIEWS[1:]:
        raise ValueError(f"unsupported source meta view: {view!r}")
    from cvsrffi.eval import apply_sat_channel_for_scenario

    channel_args = SimpleNamespace(
        sat_fs_hz=CANONICAL_SAT_FS_HZ,
        sat_fc_hz=CANONICAL_SAT_FC_HZ,
    )
    generator = torch.Generator(device=x.device)
    generator.manual_seed(_stable_view_seed(view_seed, str(ref.physical_sample_id), view))
    transformed, _metadata = apply_sat_channel_for_scenario(
        x.unsqueeze(0),
        view,
        channel_args,
        gen=generator,
        return_meta=True,
    )
    if not torch.is_tensor(transformed):
        raise TypeError(f"LEO view {view!r} transform must return a tensor")
    if transformed.shape != x.unsqueeze(0).shape:
        raise ValueError(
            f"LEO view {view!r} changed IQ shape: {tuple(transformed.shape)} "
            f"!= {tuple(x.unsqueeze(0).shape)}"
        )
    transformed = transformed[0].detach().to(device=x.device, dtype=x.dtype)
    if not transformed.is_floating_point() or not bool(torch.isfinite(transformed).all()):
        raise ValueError(f"LEO view {view!r} must be finite floating IQ")
    return transformed


def _sample_episode(
    sampler: Any,
    *,
    seed: int,
    require_clean_query: bool = False,
) -> Any:
    last_error: Exception | None = None
    for offset in range(256):
        try:
            episode = sampler.sample(int(seed) + offset)
            if not episode.guard_class_ids or not episode.query_guard:
                raise ValueError("frozen episode requires non-empty Y_guard")
            if require_clean_query and not any(
                str(row.view) == "clean"
                for row in episode.query_adapt + episode.query_guard
            ):
                raise ValueError("frozen V_select episode requires clean query evidence")
            return episode
        except (ValueError, RuntimeError) as exc:
            last_error = exc
    raise ValueError(f"cannot construct source meta episode after 256 seeds: {last_error}")


def _episode_batch(
    episode: Any,
    dataset: Any,
    *,
    model: nn.Module,
    num_classes: int,
    device: torch.device,
    view_seed: int = 0,
) -> Any:
    from cvsrffi.meta_trainer import MetaEpisodeBatch

    refs = episode.support + episode.query_adapt + episode.query_guard
    tensors: dict[tuple[int, str], torch.Tensor] = {}
    for ref in refs:
        key = (int(ref.dataset_index), str(ref.view))
        if key not in tensors:
            x, y, _meta = _dataset_item(dataset, int(ref.dataset_index))
            del y
            tensors[key] = _materialize_ref_view(x, ref, view_seed=int(view_seed))

    def stack(rows: Sequence[Any]) -> tuple[torch.Tensor, torch.Tensor]:
        if not rows:
            raise ValueError("meta episode partition cannot be empty")
        values = [tensors[(int(row.dataset_index), str(row.view))] for row in rows]
        ys = [int(row.tx_i) for row in rows]
        return torch.stack(values).to(device), torch.tensor(ys, dtype=torch.long, device=device)

    support_x, support_y = stack(episode.support)
    query_rows = episode.query_adapt + episode.query_guard
    query_x, query_y = stack(query_rows)
    adapt_mask = torch.tensor(
        [True] * len(episode.query_adapt) + [False] * len(episode.query_guard),
        dtype=torch.bool,
        device=device,
    )
    guard_mask = ~adapt_mask
    with torch.no_grad():
        embedding = _embedding_from_output(_model_forward(model, support_x[:1], support_y[:1]))
    frozen_prototypes = torch.zeros(
        int(num_classes), int(embedding.shape[1]), dtype=embedding.dtype, device=device
    )
    return MetaEpisodeBatch(
        episode=episode,
        support_x=support_x,
        support_y=support_y,
        query_x=query_x,
        query_y=query_y,
        adapt_mask=adapt_mask,
        guard_mask=guard_mask,
        frozen_prototypes=frozen_prototypes,
    )


def _compute_frozen_class_prototypes(
    model: nn.Module,
    batches: Sequence[Any],
    *,
    class_count: int,
) -> torch.Tensor:
    """Compute one immutable source-only class mean per registered class."""

    if isinstance(class_count, bool) or int(class_count) <= 0:
        raise ValueError("class_count must be a positive integer")
    if not batches:
        raise ValueError("source prototype construction requires L_s batches")
    sums: dict[int, torch.Tensor] = {}
    counts = {class_id: 0 for class_id in range(int(class_count))}
    training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for batch in batches:
                rows = batch.episode.support + batch.episode.query_adapt + batch.episode.query_guard
                if {str(row.role) for row in rows} != {"L_s"}:
                    raise ValueError("frozen prototypes may only read L_s source carriers")
                for values, labels in (
                    (batch.support_x, batch.support_y),
                    (batch.query_x, batch.query_y),
                ):
                    embeddings = _embedding_from_output(
                        _model_forward(model, values, labels)
                    ).detach()
                    for class_id in torch.unique(labels, sorted=True).tolist():
                        class_id = int(class_id)
                        if class_id not in counts:
                            raise ValueError(
                                f"source class {class_id} is outside registered class_mapping"
                            )
                        selected = embeddings[labels == class_id]
                        value = selected.sum(dim=0).detach().cpu()
                        sums[class_id] = value if class_id not in sums else sums[class_id] + value
                        counts[class_id] += int(selected.size(0))
    finally:
        model.train(training)
    missing = [class_id for class_id, count in counts.items() if count <= 0]
    if missing:
        raise ValueError(
            f"source prototypes require samples for every registered class; missing={missing}"
        )
    prototypes = torch.stack(
        [sums[class_id] / float(counts[class_id]) for class_id in range(int(class_count))]
    )
    if not prototypes.is_floating_point() or not bool(torch.isfinite(prototypes).all()):
        raise ValueError("source prototypes must be finite floating class means")
    zero_rows = torch.linalg.vector_norm(prototypes, dim=1) <= 0
    if bool(zero_rows.any()):
        raise ValueError(
            "source prototypes must be non-zero for every registered class; "
            f"zero_classes={torch.nonzero(zero_rows).flatten().tolist()}"
        )
    return prototypes.detach().clone()


def _randomize_adapter_parameters(model: nn.Module, *, seed: int) -> None:
    from cvsrffi.meta_adapter import iter_inner_adapter_parameters

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    with torch.no_grad():
        for _name, parameter in iter_inner_adapter_parameters(model):
            values = torch.randn(
                parameter.shape, generator=generator, dtype=torch.float32
            ).to(device=parameter.device, dtype=parameter.dtype)
            parameter.copy_(values * 0.01)


def _scenario_accuracy(
    model: nn.Module,
    values: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, Any]:
    model_device = next(model.parameters()).device
    predictions_parts = []
    labels_cpu = labels.detach().cpu().to(dtype=torch.long)
    with torch.no_grad():
        for start in range(0, int(labels.numel()), 128):
            stop = min(start + 128, int(labels.numel()))
            batch_labels = labels[start:stop].to(model_device, dtype=torch.long)
            output = _model_forward(
                model, values[start:stop].to(model_device), batch_labels
            )
            logits = output.get("logits", output.get("tx_logits"))
            if not torch.is_tensor(logits) or logits.ndim != 2:
                raise ValueError("final checkpoint evaluation requires fixed-head logits")
            predictions_parts.append(logits.argmax(dim=1).detach().cpu())
    predictions = torch.cat(predictions_parts)
    labels = labels_cpu
    pairs = []
    for class_id in torch.unique(labels, sorted=True).tolist():
        mask = labels == int(class_id)
        pairs.append(
            {
                "class_id": int(class_id),
                "accuracy": float((predictions[mask] == labels[mask]).float().mean().item()),
                "count": int(mask.sum().item()),
            }
        )
    return {
        "accuracy": float((predictions == labels).float().mean().item()),
        "count": int(labels.numel()),
        "per_class": pairs,
    }


def _stream_dataset_scenario_accuracy(
    model: nn.Module,
    dataset: Any,
    *,
    view: str,
    seed: int,
    batch_size: int = 128,
) -> dict[str, Any]:
    """Score one declared source scenario without materializing the full split."""

    from cvsrffi.meta_episodes import MetaSampleRef

    if view not in _SOURCE_META_VIEWS:
        raise ValueError(f"unsupported declared clean test view: {view!r}")
    if isinstance(batch_size, bool) or int(batch_size) <= 0:
        raise ValueError("scenario evaluation batch_size must be positive")
    model_device = next(model.parameters()).device
    class_counts: dict[int, int] = {}
    class_correct: dict[int, int] = {}
    total_count = 0
    total_correct = 0
    xs: list[torch.Tensor] = []
    ys: list[int] = []

    def consume_batch() -> None:
        nonlocal total_count, total_correct
        if not xs:
            return
        values = torch.stack(xs).to(model_device)
        labels = torch.tensor(ys, dtype=torch.long, device=model_device)
        with torch.no_grad():
            output = _model_forward(model, values, labels)
            logits = output.get("logits", output.get("tx_logits"))
            if not torch.is_tensor(logits) or logits.ndim != 2:
                raise ValueError("final checkpoint evaluation requires fixed-head logits")
            predictions = logits.argmax(dim=1).detach().cpu()
        labels_cpu = labels.detach().cpu()
        total_count += int(labels_cpu.numel())
        total_correct += int((predictions == labels_cpu).sum().item())
        for class_id in torch.unique(labels_cpu, sorted=True).tolist():
            class_id = int(class_id)
            mask = labels_cpu == class_id
            class_counts[class_id] = class_counts.get(class_id, 0) + int(mask.sum().item())
            class_correct[class_id] = class_correct.get(class_id, 0) + int(
                (predictions[mask] == labels_cpu[mask]).sum().item()
            )
        xs.clear()
        ys.clear()

    for index in range(len(dataset)):
        x, y, metadata = _dataset_item(dataset, index)
        required = ("rx_i", "day_i", "eq_i", "capture_block_i", "physical_sample_id")
        if any(key not in metadata for key in required):
            raise ValueError(
                "declared clean test metadata is missing a physical/domain field"
            )
        ref = MetaSampleRef(
            dataset_index=int(index),
            tx_i=int(y),
            rx_i=int(metadata["rx_i"]),
            day_i=int(metadata["day_i"]),
            eq_i=int(metadata["eq_i"]),
            capture_block_i=int(metadata["capture_block_i"]),
            physical_sample_id=str(metadata["physical_sample_id"]),
            role="declared_clean_test",
            view=str(view),
        )
        xs.append(_materialize_ref_view(x, ref, view_seed=int(seed)))
        ys.append(int(y))
        if len(xs) >= int(batch_size):
            consume_batch()
    consume_batch()
    if total_count <= 0:
        raise ValueError(f"declared clean test has no rows for {view}")
    return {
        "accuracy": float(total_correct / total_count),
        "count": int(total_count),
        "per_class": [
            {
                "class_id": int(class_id),
                "accuracy": float(class_correct[class_id] / class_counts[class_id]),
                "count": int(class_counts[class_id]),
            }
            for class_id in sorted(class_counts)
        ],
    }


def _evaluate_final_checkpoint_scenarios(
    model: nn.Module,
    *,
    source_manifest: Mapping[str, Any],
    eval_batches: Sequence[Any],
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    """Evaluate clean and each LEO weak family on the declared held-out test."""

    scenarios: dict[str, Any] = {}
    training = model.training
    model.eval()
    try:
        if source_manifest.get("available"):
            if source_manifest.get("clean_test_physical_disjoint") is not True:
                raise ValueError("declared clean test lacks physical-disjoint evidence")
            dataset = source_manifest["clean_test_dataset"]
            for view in _SOURCE_META_VIEWS:
                scenarios[view] = _stream_dataset_scenario_accuracy(
                    model,
                    dataset,
                    view=view,
                    seed=int(seed),
                )
            evidence_origin = "declared_clean_test_source_iq"
            split = "declared_clean_test"
            test_days = list(source_manifest["clean_test_days"])
            physical_disjoint = True
            formal_test_evidence = True
        else:
            selected = [
                batch
                for batch in eval_batches
                if {str(row.role) for row in batch.episode.support + batch.episode.query_adapt + batch.episode.query_guard}
                == {"V_select"}
            ]
            if not selected:
                raise ValueError("injected final evaluation lacks V_select source carriers")
            values = torch.cat([batch.query_x for batch in selected], dim=0)
            labels = torch.cat([batch.query_y for batch in selected], dim=0)
            metric = _scenario_accuracy(model, values, labels)
            scenarios = {view: dict(metric) for view in _SOURCE_META_VIEWS}
            evidence_origin = "injected_source_fixture"
            split = "fixture_not_declared_clean_test"
            test_days = []
            physical_disjoint = False
            formal_test_evidence = False
    finally:
        model.train(training)
    return {
        "source_only": True,
        "split": split,
        "test_days": test_days,
        "physical_disjoint_from_phase1_roles": physical_disjoint,
        "formal_test_evidence": formal_test_evidence,
        "evidence_origin": evidence_origin,
        "scenarios": scenarios,
    }


def _build_source_batches(
    role_datasets: Mapping[str, Any],
    config: Mapping[str, Any],
    model: nn.Module,
    device: torch.device,
) -> tuple[list[Any], list[Any]]:
    from cvsrffi.meta_episodes import HierarchicalMetaEpisodeSampler, MetaEpisodeSamplerConfig

    weights = config["episode_weights"]
    train_refs, train_dataset = _build_refs(role_datasets["L_s"], "L_s")
    sampler_config = MetaEpisodeSamplerConfig(
        k_choices=tuple(config["k_choices"]),
        query_per_class=int(config["meta_query_per_class"]),
        allowed_roles=("L_s",),
        training=True,
        episode_weights=weights,
    )
    train_sampler = HierarchicalMetaEpisodeSampler(train_refs, sampler_config)
    num_classes = int(config["model"].get("num_classes", len(getattr(train_dataset, "tx_list", []) or [])) or 1)
    train_batches = [
        _episode_batch(
            _sample_episode(train_sampler, seed=int(config["seed"]) + index),
            train_dataset,
            model=model,
            num_classes=num_classes,
            device=device,
            view_seed=int(config["seed"]),
        )
        for index in range(int(config["meta_batch_size"]))
    ]

    eval_config = MetaEpisodeSamplerConfig(
        k_choices=tuple(config["k_choices"]),
        query_per_class=int(config["meta_query_per_class"]),
        allowed_roles=("V_cal", "V_select"),
        training=False,
        episode_weights=weights,
    )
    eval_batches: list[Any] = []
    eval_count = int(config["meta_eval_episodes"])
    for role_index, role in enumerate(("V_cal", "V_select")):
        refs, dataset = _build_refs(role_datasets[role], role)
        sampler = HierarchicalMetaEpisodeSampler(refs, eval_config)
        count = max(1, eval_count // 2 + (1 if role_index < eval_count % 2 else 0))
        for index in range(count):
            eval_batches.append(
                _episode_batch(
                    _sample_episode(
                        sampler,
                        seed=int(config["seed"]) + 10000 + role_index * 1000 + index,
                        require_clean_query=(role == "V_select" and index == 0),
                    ),
                    dataset,
                    model=model,
                    num_classes=num_classes,
                    device=device,
                    view_seed=int(config["seed"]),
                )
            )
    return train_batches, eval_batches


def _validate_injected_batches(value: Any, config: Mapping[str, Any]) -> tuple[list[Any], list[Any]]:
    from cvsrffi.meta_trainer import MetaEpisodeBatch

    if not isinstance(value, Mapping):
        raise TypeError("meta_episode_batch_factory must return a mapping")
    train_batches = list(value.get("train", ()))
    eval_batches = list(value.get("eval", ()))
    if len(train_batches) != int(config["meta_batch_size"]):
        raise ValueError("injected train episode count does not match meta_batch_size")
    if len(eval_batches) < 2:
        raise ValueError("injected evaluation episodes must contain V_cal and V_select")
    if any(not isinstance(batch, MetaEpisodeBatch) for batch in train_batches + eval_batches):
        raise TypeError("injected episode carriers must be MetaEpisodeBatch values")
    train_roles = {str(row.role) for batch in train_batches for row in batch.episode.support + batch.episode.query_adapt + batch.episode.query_guard}
    eval_roles = {str(row.role) for batch in eval_batches for row in batch.episode.support + batch.episode.query_adapt + batch.episode.query_guard}
    if train_roles != {"L_s"}:
        raise ValueError(f"injected training carriers must be L_s-only, got {sorted(train_roles)!r}")
    if not eval_roles.issubset({"V_cal", "V_select"}) or not {"V_cal", "V_select"}.issubset(eval_roles):
        raise ValueError("injected evaluation carriers must cover V_cal and V_select only")
    return train_batches, eval_batches


def _resolve_output_root(args: Any, config: Mapping[str, Any]) -> Path:
    value = str(getattr(args, "meta_output_root", "") or "").strip()
    root = Path(value) if value else PROJECT_ROOT / "runs" / str(config["run_id"])
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    return root.resolve()


def _write_json_exclusive(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        handle.write("\n")


def _curve_payload(curve: Any) -> dict[str, Any]:
    return {
        "steps": list(curve.steps),
        "source_only": bool(curve.source_only),
        "rows": [asdict(row) for row in curve.rows],
    }


def _curve_for_role(curve: Any, role: str) -> Any:
    from cvsrffi.meta_trainer import AdaptationCurve

    expected = str(role)
    rows = tuple(row for row in curve.rows if str(row.role) == expected)
    if not rows:
        raise ValueError(f"source curve has no rows for frozen role {expected!r}")
    return AdaptationCurve(steps=tuple(curve.steps), rows=rows, source_only=bool(curve.source_only))


def _candidate_from_curves(
    baseline: Any,
    final: Any,
    *,
    candidate_id: str,
    model: nn.Module,
) -> Any:
    from cvsrffi.meta_trainer import SourceCheckpointCandidate, SourceHoldoutDelta

    baseline_rows = {
        (int(row.episode_index), int(row.step)): row
        for row in baseline.rows
        if str(row.role) == "V_select"
    }
    final_rows = {
        (int(row.episode_index), int(row.step)): row
        for row in final.rows
        if str(row.role) == "V_select"
    }
    holdouts: list[SourceHoldoutDelta] = []
    clean_deltas: list[float] = []
    guard_deltas: list[float] = []
    for key, final_row in sorted(final_rows.items()):
        base_row = baseline_rows.get(key)
        if base_row is None:
            continue
        if int(final_row.step) == 0:
            if final_row.clean_step0_accuracy is not None and base_row.clean_step0_accuracy is not None:
                clean_deltas.append(100.0 * (float(final_row.clean_step0_accuracy) - float(base_row.clean_step0_accuracy)))
            if (
                final_row.guard_floor_accuracy is not None
                and base_row.guard_floor_accuracy is not None
            ):
                guard_deltas.append(
                    100.0
                    * (
                        float(final_row.guard_floor_accuracy)
                        - float(base_row.guard_floor_accuracy)
                    )
                )
        if int(final_row.step) == 3 and final_row.mean_accuracy is not None and base_row.mean_accuracy is not None:
            holdouts.append(
                SourceHoldoutDelta(
                    holdout_id=f"{final_row.role}:{final_row.episode_index}",
                    a0=float(base_row.mean_accuracy),
                    a3=float(final_row.mean_accuracy),
                )
            )
    if not holdouts:
        raise ValueError("source V_select curve produced no finite step-3 holdouts")
    if not clean_deltas or not guard_deltas:
        raise ValueError(
            "source V_select candidate requires explicit clean and Y_guard floor evidence"
        )
    return SourceCheckpointCandidate(
        candidate_id=candidate_id,
        clean_delta_pp=float(sum(clean_deltas) / len(clean_deltas)),
        guard_floor_delta_pp=min(guard_deltas),
        worst_a3_delta_pp=min(item.delta_pp for item in holdouts),
        parameter_count=int(sum(parameter.numel() for parameter in model.parameters())),
        latency_ms=float(sum(float(row.latency_ms) for row in final_rows.values()) / max(1, len(final_rows))),
        source_holdouts=tuple(holdouts),
    )


def _bundle_model_args(model_args: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(model_args)
    result["meta_adapter_rank"] = int(config["adapter"]["rank"])
    result["meta_adapter_sites"] = ",".join(config["adapter"]["sites"])
    return result


def _write_failed_summary(root: Path, exc: BaseException) -> None:
    path = root / "run_summary.json"
    if path.exists():
        return
    _write_json_exclusive(
        path,
        {
            "status": "FAILED",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        },
    )


def run_meta_phase1(args: Any, ds_w: Mapping[str, Any]) -> dict[str, Any]:
    """Run the real source-only Phase1 Task2→Task7→Task4 path."""

    if not isinstance(ds_w, Mapping):
        raise TypeError("run_meta_phase1 expects the loaded WiSig mapping")
    config_source = getattr(args, "meta_config", None) or getattr(args, "meta_phase1_config", None)
    if isinstance(config_source, Mapping):
        config = validate_meta_phase1_config(config_source)
    else:
        config = load_meta_phase1_config(config_source or DEFAULT_CONFIG_PATH)
    _validate_frozen_wisig_args(args, config)
    _seed_everything(int(config["seed"]))

    requested_rank = getattr(args, "meta_adapter_rank", None)
    if requested_rank not in (None, 0, int(config["adapter"]["rank"])):
        raise ValueError(
            f"meta_adapter_rank must match frozen adapter.rank={config['adapter']['rank']}"
        )
    requested_sites = getattr(args, "meta_adapter_sites", None)
    if requested_sites not in (None, "", ",".join(config["adapter"]["sites"])):
        raise ValueError("meta_adapter_sites must match config adapter.sites")
    requested_steps = getattr(args, "meta_inner_steps", None)
    if requested_steps not in (None, 0, int(config["adapter"]["inner_steps"])):
        raise ValueError("meta_inner_steps must match frozen inner_steps=3")
    requested_max_steps = getattr(args, "meta_inner_max_steps", None)
    if requested_max_steps not in (None, 0, int(config["adapter"]["deployment_max_steps"])):
        raise ValueError("meta_inner_max_steps must match frozen deployment_max_steps=5")

    requested_source_rxs = getattr(args, "wisig_train_rxs", None)
    if requested_source_rxs not in (None, ""):
        parsed_source_rxs = tuple(
            _parse_index_list(requested_source_rxs, field_name="wisig_train_rxs", default=())
        )
        if parsed_source_rxs != tuple(config["source_receiver_ids"]):
            raise ValueError(
                "wisig_train_rxs does not match frozen source_receiver_ids: "
                f"{parsed_source_rxs!r} != {tuple(config['source_receiver_ids'])!r}"
            )
    configured_days = config["source_days"]
    if configured_days is not None and getattr(args, "wisig_train_days", None) not in (None, ""):
        parsed_source_days = tuple(
            _parse_index_list(getattr(args, "wisig_train_days"), field_name="wisig_train_days", default=())
        )
        if parsed_source_days != tuple(configured_days):
            raise ValueError(
                "wisig_train_days does not match frozen source_days: "
                f"{parsed_source_days!r} != {tuple(configured_days)!r}"
            )

    config_source = getattr(args, "meta_config", None) or getattr(args, "meta_phase1_config", None)
    requested_base_path = (
        getattr(args, "init_checkpoint", None) or getattr(args, "base_checkpoint", None)
    )
    requested_wisig_path = getattr(args, "wisig_pkl", None)
    base_path = (
        Path(str(requested_base_path)).expanduser().resolve()
        if requested_base_path is not None and str(requested_base_path).strip()
        else _resolve_config_path(config["base_checkpoint"], config_source=config_source)
    )
    wisig_path = (
        Path(str(requested_wisig_path)).expanduser().resolve()
        if requested_wisig_path is not None and str(requested_wisig_path).strip()
        else _resolve_config_path(config["wisig_pkl"], config_source=config_source)
    )
    _require_readable_file(base_path, field_name="base_checkpoint")
    _require_readable_file(wisig_path, field_name="wisig_pkl")

    output_root = _resolve_output_root(args, config)
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"immutable meta Phase1 output root already exists: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)
    try:
        _write_json_exclusive(output_root / "config_snapshot.json", config)
        device = torch.device(str(getattr(args, "device", "cpu") or "cpu"))
        payload = _load_checkpoint_payload(base_path, device)
        model_factory = getattr(args, "meta_model_factory", None)
        if callable(model_factory):
            model = model_factory(config, ds_w, device)
            model_args = _bundle_model_args(dict(config["model"]), config)
        else:
            model, model_args = _build_meta_model(config, ds_w, payload, device)
        if not isinstance(model, nn.Module):
            raise TypeError("meta_model_factory must return torch.nn.Module")
        model = model.to(device)
        _validate_rank4_adapter_model(model, config)
        from cvsrffi.meta_checkpoint import load_legacy_base_for_meta, save_meta_bundle
        from cvsrffi.meta_trainer import (
            MetaTrainerConfig,
            build_phase1b_optimizer,
            evaluate_adaptation_curve,
            run_meta_train_step,
            run_supervised_adapter_step,
            select_source_checkpoint,
        )

        checkpoint_audit, adapter_migration = _load_legacy_checkpoint_into_meta_model(
            model,
            config,
            ds_w,
            payload,
            device,
            allow_nested_dual_bridge=not callable(model_factory),
        )
        source_manifest = _source_role_manifest(ds_w, config, args)
        batch_factory = getattr(args, "meta_episode_batch_factory", None)
        if callable(batch_factory):
            train_batches, eval_batches = _validate_injected_batches(
                batch_factory(config, ds_w, model), config
            )
        else:
            if not source_manifest.get("available"):
                raise ValueError("WiSig source data is required for the non-injected meta Phase1 path")
            train_batches, eval_batches = _build_source_batches(
                source_manifest["role_datasets"], config, model, device
            )
        class_count = int(model_args.get("num_classes", len(ds_w.get("tx_list", [])) or 1))
        frozen_prototypes = _compute_frozen_class_prototypes(
            model, train_batches, class_count=class_count
        ).to(device)
        train_batches = [
            replace(batch, frozen_prototypes=frozen_prototypes.clone())
            for batch in train_batches
        ]
        eval_batches = [
            replace(batch, frozen_prototypes=frozen_prototypes.clone())
            for batch in eval_batches
        ]
        candidate_row = next(
            row
            for row in config["candidate_plan"]
            if row["candidate_id"] == config["active_candidate_id"]
        )
        candidate_id = str(candidate_row["candidate_id"])
        training_mode = str(candidate_row["training_mode"])
        trainer_config = MetaTrainerConfig(
            source_receiver_ids=tuple(config["source_receiver_ids"]),
            meta_batch_size=int(config["meta_batch_size"]),
            inner_steps=int(config["adapter"]["inner_steps"]),
            phase1c_backbone_lr_ratio=float(config["phase1c_backbone_lr_ratio"]),
            learn_step_sizes=bool(candidate_row["learn_step_sizes"]),
        )
        baseline_curve = evaluate_adaptation_curve(model, eval_batches, trainer_config)
        baseline_v_cal_curve = _curve_for_role(baseline_curve, "V_cal")
        baseline_v_select_curve = _curve_for_role(baseline_curve, "V_select")
        p0_evaluation = _evaluate_final_checkpoint_scenarios(
            model,
            source_manifest=source_manifest,
            eval_batches=eval_batches,
            device=device,
            seed=int(config["seed"]),
        )
        _write_json_exclusive(output_root / "p0_control_evaluation.json", p0_evaluation)
        if training_mode in {"random_adapter", "supervised_adapter"}:
            _randomize_adapter_parameters(model, seed=int(config["seed"]) + 101)
        optimizer = None
        if training_mode in {"supervised_adapter", "fomaml_fixed_lr", "fomaml_meta_sgd"}:
            optimizer = build_phase1b_optimizer(model, trainer_config)
        outer_steps = 0 if training_mode == "random_adapter" else int(config["meta_train_steps"])
        logs_path = output_root / "logs.jsonl"
        metrics_path = output_root / "metrics.csv"
        with logs_path.open("x", encoding="utf-8", newline="\n") as logs_handle, metrics_path.open(
            "x", encoding="utf-8", newline=""
        ) as metrics_handle:
            writer = csv.DictWriter(metrics_handle, fieldnames=("train_step", "outer_loss", "updated_parameter_count"))
            writer.writeheader()
            for train_step in range(outer_steps):
                if optimizer is None:
                    raise RuntimeError("training candidate is missing its optimizer")
                step_fn = (
                    run_supervised_adapter_step
                    if training_mode == "supervised_adapter"
                    else run_meta_train_step
                )
                result = step_fn(model, train_batches, optimizer, trainer_config)
                writer.writerow(
                    {
                        "train_step": int(train_step),
                        "outer_loss": float(result.loss.detach().cpu().item()),
                        "updated_parameter_count": len(result.updated_parameter_names),
                    }
                )
                for episode_log in result.episode_logs:
                    record = {"train_step": int(train_step), **dict(episode_log)}
                    logs_handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        final_curve = evaluate_adaptation_curve(model, eval_batches, trainer_config)
        final_v_cal_curve = _curve_for_role(final_curve, "V_cal")
        final_v_select_curve = _curve_for_role(final_curve, "V_select")
        candidate = None
        try:
            candidate = _candidate_from_curves(
                baseline_v_select_curve,
                final_v_select_curve,
                candidate_id=f"{config['run_id']}:{candidate_id}",
                model=model,
            )
            selected = select_source_checkpoint([candidate])
            scientific_verdict = "SOURCE_SELECTION_ELIGIBLE"
            selected_payload: dict[str, Any] | None = asdict(selected)
        except ValueError as selection_error:
            selected_payload = None
            scientific_verdict = "SCIENTIFIC_FAILURE_NO_PROMOTION"
            selection_failure = str(selection_error)
        curve_payload = {
            "baseline": _curve_payload(baseline_curve),
            "final": _curve_payload(final_curve),
            "v_calibration": {
                "baseline": _curve_payload(baseline_v_cal_curve),
                "final": _curve_payload(final_v_cal_curve),
                "selection_role": "diagnostic_only",
            },
            "v_select": {
                "baseline": _curve_payload(baseline_v_select_curve),
                "final": _curve_payload(final_v_select_curve),
                "selection_role": "checkpoint_ranking",
            },
            "selection_source_split": "V_select",
            "candidate_id": candidate_id,
            "training_mode": training_mode,
            "selected_candidate": selected_payload,
            "scientific_verdict": scientific_verdict,
            "source_only": True,
        }
        _write_json_exclusive(output_root / "source_adaptation_curve.json", curve_payload)
        final_evaluation = _evaluate_final_checkpoint_scenarios(
            model,
            source_manifest=source_manifest,
            eval_batches=eval_batches,
            device=device,
            seed=int(config["seed"]),
        )
        _write_json_exclusive(
            output_root / "final_checkpoint_evaluation.json", final_evaluation
        )
        tx_list = list(ds_w.get("tx_list", []))
        class_mapping = {
            str(index): str(tx_list[index] if index < len(tx_list) else index)
            for index in range(class_count)
        }
        prototype = frozen_prototypes.detach().cpu().clone()
        prototype_array = _numpy_array_abi_safe(prototype, dtype=np.float32)
        with (output_root / "frozen_prototypes.npz").open("xb") as handle:
            np.savez(
                handle,
                prototypes=prototype_array,
                class_ids=np.arange(class_count, dtype=np.int64),
            )
        bundle_config = {
            "model_args": _bundle_model_args(model_args, config),
            "meta_adapter_config": {
                "rank": int(config["adapter"]["rank"]),
                "sites": list(config["adapter"]["sites"]),
                "phase2_steps": int(config["adapter"]["deployment_max_steps"]),
            },
            "base_checkpoint": {
                "id": f"{config['run_id']}:legacy_adv3b02",
                "role": "legacy_adv3b02",
            },
            "class_mapping": class_mapping,
            "prototypes": prototype,
        }
        save_meta_bundle(
            output_root / "selected_meta_bundle.pt",
            model,
            bundle_config,
            {"source_split": "V_select", "criterion": "max_min_source_holdout_delta", "seed": int(config["seed"])},
        )
        summary = {
            "status": "ARTIFACTS_COMPLETE",
            "schema": config["schema"],
            "run_id": config["run_id"],
            "source_only": True,
            "source_receiver_ids": list(config["source_receiver_ids"]),
            "source_split": config["source_split"],
            "source_days": list(config["source_days"]),
            "clean_test_days": list(config["clean_test_days"]),
            "candidate_id": candidate_id,
            "training_mode": training_mode,
            "task7_outer_steps": outer_steps,
            "checkpoint_load": asdict(checkpoint_audit),
            "adapter_migration": adapter_migration,
            "candidate_result": None if candidate is None else asdict(candidate),
            "selected_candidate": selected_payload,
            "scientific_verdict": scientific_verdict,
            "artifacts": {
                name: str(output_root / name)
                for name in (
                    "logs.jsonl",
                    "metrics.csv",
                    "selected_meta_bundle.pt",
                    "source_adaptation_curve.json",
                    "run_summary.json",
                    "config_snapshot.json",
                    "p0_control_evaluation.json",
                    "final_checkpoint_evaluation.json",
                    "frozen_prototypes.npz",
                )
            },
        }
        if scientific_verdict == "SCIENTIFIC_FAILURE_NO_PROMOTION":
            summary["selection_failure"] = selection_failure
        _write_json_exclusive(output_root / "run_summary.json", summary)
        print(
            "[META-PHASE1] "
            f"run_id={config['run_id']} status=ARTIFACTS_COMPLETE "
            f"candidate={candidate_id} mode={training_mode} outer_steps={outer_steps} "
            f"source_receiver_ids={config['source_receiver_ids']}",
            flush=True,
        )
        return summary
    except Exception as exc:
        _write_failed_summary(output_root, exc)
        raise


__all__ = [
    "CANONICAL_ADAPTER",
    "CANONICAL_EVALUATE_STEPS",
    "CANONICAL_EPISODE_WEIGHTS",
    "CANONICAL_K_CHOICES",
    "CANONICAL_SOURCE_RECEIVER_IDS",
    "CANONICAL_SOURCE_SPLIT",
    "CANONICAL_SOURCE_DAYS",
    "CANONICAL_SOURCE_ROLES",
    "DEFAULT_CONFIG_PATH",
    "META_PHASE1_SCHEMA",
    "load_meta_phase1_config",
    "parse_args_for_test",
    "run_meta_phase1",
    "validate_meta_phase1_config",
]
