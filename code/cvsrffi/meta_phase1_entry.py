"""Thin, source-only entrypoint for the frozen Phase1 meta-adapter run.

Task8 owns configuration validation, source split construction and the CLI
handoff.  The actual FOMAML implementation remains in ``meta_trainer``; this
module deliberately does not duplicate a training loop or expose target/Phase2
inputs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "phase1_adv3b02_meta_adapter_tri_r4_v1_s392002_20260824.json"
META_PHASE1_SCHEMA = "cvs.phase1.meta_adapter.tri_r4.v1"
CANONICAL_SOURCE_ROLES = {
    "L_s": 0.07,
    "U_s": 0.63,
    "V_cal": 0.15,
    "V_select": 0.15,
}
CANONICAL_SOURCE_RECEIVER_IDS = (0, 1, 2, 3, 4, 5, 6)
CANONICAL_ADAPTER = {
    "rank": 4,
    "sites": ("time", "freq", "fusion"),
    "inner_steps": 3,
    "deployment_max_steps": 5,
    "source_diagnostic_max_steps": 10,
}
CANONICAL_EPISODE_WEIGHTS = {
    "Q_SAME_DOMAIN": 0.40,
    "Q_RX_HOLDOUT": 0.20,
    "Q_DAY_CHANNEL_HOLDOUT": 0.15,
    "Q_CLEAN_TO_LEO": 0.15,
    "Q_LEO_CROSS": 0.10,
}
CANONICAL_K_CHOICES = (1, 2, 5, 10)
CANONICAL_EVALUATE_STEPS = (0, 1, 3, 5, 10)


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
        "source_roles",
        "adapter",
        "episode_weights",
        "k_choices",
        "meta_batch_size",
        "phase1c_backbone_lr_ratio",
        "evaluate_steps",
    }
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(f"meta Phase1 config is missing required fields: {missing}")

    schema = _require_token(config["schema"], field_name="schema")
    if schema != META_PHASE1_SCHEMA:
        raise ValueError(f"schema must be {META_PHASE1_SCHEMA!r}; got {schema!r}")
    run_id = _require_token(config["run_id"], field_name="run_id")
    seed = _require_int(config["seed"], field_name="seed", minimum=0)
    base_checkpoint = _require_token(config["base_checkpoint"], field_name="base_checkpoint")
    wisig_pkl = _require_token(config["wisig_pkl"], field_name="wisig_pkl")
    source_receiver_ids = _check_source_receiver_ids(config["source_receiver_ids"])

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
    if adapter["sites"] != CANONICAL_ADAPTER["sites"]:
        raise ValueError(
            "adapter.sites must be frozen at ['time', 'freq', 'fusion']; "
            f"got {list(adapter['sites'])!r}"
        )
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
    source_days: tuple[int, ...] | None = None
    if "source_days" in config:
        source_days = _check_exact_int_sequence(config["source_days"], (0, 1), field_name="source_days")

    normalized: dict[str, Any] = dict(config)
    normalized.update(
        {
            "schema": schema,
            "run_id": run_id,
            "seed": seed,
            "base_checkpoint": base_checkpoint,
            "wisig_pkl": wisig_pkl,
            "source_receiver_ids": list(source_receiver_ids),
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
        }
    )
    if source_days is not None:
        normalized["source_days"] = list(source_days)
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


def _source_role_manifest(ds_w: Mapping[str, Any], config: Mapping[str, Any], args: Any) -> dict[str, Any]:
    """Build only source role metadata; no target split or query is opened."""

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

    equalized = getattr(args, "wisig_equalized", 1)
    if isinstance(equalized, str) and equalized.lower() != "both":
        equalized = int(equalized)
    configured_days = config.get("source_days")
    train_days = _parse_index_list(
        configured_days if configured_days is not None else getattr(args, "wisig_train_days", None),
        field_name="source_days" if configured_days is not None else "wisig_train_days",
        default=(0, 1),
    )
    if not train_days:
        raise ValueError("Phase1 meta entry requires at least one source training day")
    max_per_combo = int(getattr(args, "wisig_max_day123_per_combo", 0) or 0)
    source_base = WiSigCompactDataset(
        ds_w,
        out_len=int(getattr(args, "wisig_out_len", 256)),
        crop_mode="center",
        normalize=True,
        equalized=equalized,
        day_keep=train_days,
        rx_keep=list(config["source_receiver_ids"]),
        domain=str(getattr(args, "wisig_domain", "rx_day")),
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
    return {
        "available": True,
        "source_only": True,
        "source_receiver_ids": tuple(config["source_receiver_ids"]),
        "source_days": tuple(train_days),
        "role_sizes": {name: len(dataset) for name, dataset in role_datasets.items()},
        "role_datasets": role_datasets,
        "supervised_training_role": "L_s",
        "unlabeled_role": "U_s",
        "validation_roles": ("V_cal", "V_select"),
        "u_s_labels_for_supervised_loss": False,
    }


def run_meta_phase1(args: Any, ds_w: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the frozen entry and prepare its source-only role manifest.

    The function is intentionally a thin handoff.  Task7 owns the optimizer and
    episode step; this entry does not duplicate that loop or consume ``U_s``
    labels, target receivers, Phase2 data or query truth.
    """

    if not isinstance(ds_w, Mapping):
        raise TypeError("run_meta_phase1 expects the loaded WiSig mapping")
    config_source = getattr(args, "meta_config", None) or getattr(args, "meta_phase1_config", None)
    if isinstance(config_source, Mapping):
        config = validate_meta_phase1_config(config_source)
    else:
        config = load_meta_phase1_config(config_source or DEFAULT_CONFIG_PATH)

    requested_rank = getattr(args, "meta_adapter_rank", None)
    if requested_rank not in (None, 0, int(config["adapter"]["rank"])):
        raise ValueError(
            f"meta_adapter_rank must match frozen adapter.rank={config['adapter']['rank']}"
        )
    requested_sites = getattr(args, "meta_adapter_sites", None)
    if requested_sites not in (None, "", ",".join(config["adapter"]["sites"])):
        raise ValueError("meta_adapter_sites must match frozen time,freq,fusion sites")
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
    configured_days = config.get("source_days")
    if configured_days is not None and getattr(args, "wisig_train_days", None) not in (None, ""):
        parsed_source_days = tuple(
            _parse_index_list(getattr(args, "wisig_train_days"), field_name="wisig_train_days", default=())
        )
        if parsed_source_days != tuple(configured_days):
            raise ValueError(
                "wisig_train_days does not match frozen source_days: "
                f"{parsed_source_days!r} != {tuple(configured_days)!r}"
            )

    source_manifest = _source_role_manifest(ds_w, config, args)
    result = {
        "status": "READY",
        "schema": config["schema"],
        "run_id": config["run_id"],
        "seed": config["seed"],
        "source_only": True,
        "source_receiver_ids": tuple(config["source_receiver_ids"]),
        "source_roles": dict(config["source_roles"]),
        "supervised_training_role": "L_s",
        "validation_roles": ("V_cal", "V_select"),
        "u_s_labels_for_supervised_loss": False,
        "config": config,
        "source_manifest": source_manifest,
    }
    print(
        "[META-PHASE1] "
        f"run_id={config['run_id']} source_receiver_ids={config['source_receiver_ids']} "
        f"roles={config['source_roles']} supervised=L_s validation=V_cal,V_select",
        flush=True,
    )
    return result


__all__ = [
    "CANONICAL_ADAPTER",
    "CANONICAL_EVALUATE_STEPS",
    "CANONICAL_EPISODE_WEIGHTS",
    "CANONICAL_K_CHOICES",
    "CANONICAL_SOURCE_RECEIVER_IDS",
    "CANONICAL_SOURCE_ROLES",
    "DEFAULT_CONFIG_PATH",
    "META_PHASE1_SCHEMA",
    "load_meta_phase1_config",
    "parse_args_for_test",
    "run_meta_phase1",
    "validate_meta_phase1_config",
]
