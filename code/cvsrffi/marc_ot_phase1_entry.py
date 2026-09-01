"""Source-only production entry for the MARC-OT Phase1 weight bundle."""

from __future__ import annotations

import copy
import json
import math
import os
import random
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from .marc_ot_phase1 import (
    canonical_episode_task_domain_selection,
    run_marc_ot_phase1_bank_training,
)
from .marc_ot_source_experts import (
    MARCOTSourceExpertConfig,
    build_source_expert_bank,
)
from .marc_ot_support_features import MARC_OT_SUPPORT_ROW_DIM
from .meta_bank_trainer import MetaBankTrainerConfig
from .meta_episodes import (
    MARC_OT_CANONICAL_K,
    HierarchicalMetaEpisodeSampler,
    MetaEpisodeSamplerConfig,
    sample_marc_ot_coverage_schedule,
)
from .meta_support_set_encoder import SupportSetEncoder
from .meta_weight_bank import (
    DeltaBankEntry,
    WeightDeltaBank,
    fit_weight_delta_bank,
    parameter_block_key,
)


MARC_OT_PHASE1_BUNDLE_SCHEMA = "cvs.phase1.marc_ot.bundle.v1"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ROLE_RATIOS = {"L_s": 0.07, "U_s": 0.63, "V": 0.30}
_REQUIRED_TRAINABLE_PREFIXES = (
    "id_backbone.t1.",
    "id_backbone.t2.",
    "id_backbone.t3.",
    "id_backbone.f1.",
    "id_backbone.f2.",
    "id_backbone.f3.",
    "id_backbone.time_projection.",
    "id_backbone.time_proj.",
    "id_backbone.frequency_projection.",
    "id_backbone.f_proj.",
    "id_backbone.freq_stats_proj.",
    "id_backbone.fusion.",
    "id_backbone.time_fuse.",
    "id_backbone.freq_gate.",
    "id_backbone.identity_mapping.",
    "identity_mapping.",
)
_TOP_LEVEL_KEYS = {
    "schema",
    "run_id",
    "seed",
    "base_checkpoint",
    "base_checkpoint_id",
    "wisig_pkl",
    "source_receiver_ids",
    "source_days",
    "source_roles",
    "wisig",
    "model",
    "k_choices",
    "training_k",
    "query_per_class",
    "schedule_seed",
    "expert",
    "encoder",
    "trainer",
    "meta_outer_lr",
    "outer_cycles",
}
_WISIG_KEYS = {
    "equalized",
    "out_len",
    "crop_mode",
    "normalize",
    "domain",
    "max_samples_per_combo",
}
_MODEL_KEYS = {
    "builder",
    "num_classes",
    "num_domains",
    "model_size",
    "dataset",
    "input_len",
    "sample_rate_hz",
    "id_feature_key",
    "dom_feature_key",
    "model_variant",
    "branch_ablation",
    "domain_branch_ablation",
    "domain_enhancer",
    "domain_enhancer_strength",
}
_EXPERT_KEYS = {"steps", "lr", "max_rank", "trainable_prefixes"}
_ENCODER_KEYS = {"hidden_dim", "lr_min", "lr_max"}
_TRAINER_KEYS = {
    "inner_steps",
    "receiver_cvar_fraction",
    "receiver_cvar_weight",
    "worst_class_guard_weight",
}


def _exact_mapping(value: Any, expected: set[str], *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    observed = set(value)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(f"{name} exact-key mismatch: missing={missing!r}, extra={extra!r}")
    return value


def _integer(value: Any, *, name: str, minimum: int = 0, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        suffix = f" in [{minimum}, {maximum}]" if maximum is not None else f" >= {minimum}"
        raise ValueError(f"{name} must be{suffix}")
    return int(value)


def _finite(value: Any, *, name: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric) or (positive and numeric <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise ValueError(f"{name} must be {qualifier}")
    return numeric


def _nonempty_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{name} must be a non-empty string")
    return value


def validate_marc_ot_phase1_config(payload: Any) -> dict[str, Any]:
    """Validate the exact source-only MARC-OT Phase1 production contract."""

    root = _exact_mapping(payload, _TOP_LEVEL_KEYS, name="config")
    if root["schema"] != MARC_OT_PHASE1_BUNDLE_SCHEMA:
        raise ValueError(f"schema must be {MARC_OT_PHASE1_BUNDLE_SCHEMA!r}")
    run_id = _nonempty_string(root["run_id"], name="run_id")
    if Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("run_id must be one immutable path segment")
    _integer(root["seed"], name="seed")
    _integer(root["schedule_seed"], name="schedule_seed")
    for field in ("base_checkpoint", "base_checkpoint_id", "wisig_pkl"):
        _nonempty_string(root[field], name=field)

    receivers = root["source_receiver_ids"]
    if not isinstance(receivers, list) or not receivers:
        raise TypeError("source_receiver_ids must be a non-empty list")
    receiver_ids = tuple(
        _integer(value, name="source_receiver_ids item") for value in receivers
    )
    if len(set(receiver_ids)) != len(receiver_ids):
        raise ValueError("source_receiver_ids must be unique")
    if root["source_days"] != [0, 1]:
        raise ValueError("source_days must be exactly [0, 1]")

    roles = _exact_mapping(root["source_roles"], set(_ROLE_RATIOS), name="source_roles")
    for role, expected in _ROLE_RATIOS.items():
        observed = _finite(roles[role], name=f"source_roles.{role}")
        if observed != expected:
            raise ValueError(
                f"source_roles must be exactly L_s=0.07, U_s=0.63, V=0.30; {role}={observed}"
            )

    wisig = _exact_mapping(root["wisig"], _WISIG_KEYS, name="wisig")
    if wisig["equalized"] != 1:
        raise ValueError("wisig.equalized must be 1")
    _integer(wisig["out_len"], name="wisig.out_len", minimum=16)
    if wisig["crop_mode"] != "center" or wisig["normalize"] is not True:
        raise ValueError("WiSig preprocessing must use center crop and normalization")
    if wisig["domain"] != "rx_day":
        raise ValueError("wisig.domain must be rx_day")
    _integer(
        wisig["max_samples_per_combo"],
        name="wisig.max_samples_per_combo",
        minimum=0,
    )

    model = _exact_mapping(root["model"], _MODEL_KEYS, name="model")
    if model["builder"] != "dual" or model["dataset"] != "wisig":
        raise ValueError("model must use the ADV3B02 dual WiSig builder")
    if _integer(model["num_classes"], name="model.num_classes", minimum=1) != 6:
        raise ValueError("model.num_classes must be 6")
    expected_domains = len(receiver_ids) * len(root["source_days"])
    if _integer(model["num_domains"], name="model.num_domains", minimum=1) != expected_domains:
        raise ValueError(f"model.num_domains must equal source rx/day cells ({expected_domains})")
    if _integer(model["input_len"], name="model.input_len", minimum=16) != int(wisig["out_len"]):
        raise ValueError("model.input_len must equal wisig.out_len")
    _finite(model["sample_rate_hz"], name="model.sample_rate_hz", positive=True)
    _finite(
        model["domain_enhancer_strength"],
        name="model.domain_enhancer_strength",
        positive=True,
    )
    fixed_model_strings = {
        "model_size": "M",
        "id_feature_key": "feat_joint",
        "dom_feature_key": "feat_imp",
        "model_variant": "lite_d",
        "branch_ablation": "no_dac",
        "domain_branch_ablation": "no_stats",
        "domain_enhancer": "rcn_stats",
    }
    for field, expected in fixed_model_strings.items():
        if model[field] != expected:
            raise ValueError(f"model.{field} must be {expected!r}")

    if root["k_choices"] != list(MARC_OT_CANONICAL_K):
        raise ValueError("k_choices must be exactly [1, 2, 5, 10, 20]")
    if root["training_k"] != [10]:
        raise ValueError("training_k must be exactly [10] for the first production bundle")
    _integer(root["query_per_class"], name="query_per_class", minimum=1)

    expert = _exact_mapping(root["expert"], _EXPERT_KEYS, name="expert")
    if _integer(expert["steps"], name="expert.steps", minimum=1) != 25:
        raise ValueError("expert.steps must be 25")
    if _finite(expert["lr"], name="expert.lr", positive=True) != 3e-5:
        raise ValueError("expert.lr must be 3e-5")
    if _integer(expert["max_rank"], name="expert.max_rank", minimum=1) != 16:
        raise ValueError("expert.max_rank must be 16")
    if expert["trainable_prefixes"] != list(_REQUIRED_TRAINABLE_PREFIXES):
        raise ValueError(
            "expert.trainable_prefixes must exactly cover the canonical identity aliases"
        )
    if any(
        forbidden in prefix.lower()
        for prefix in expert["trainable_prefixes"]
        for forbidden in ("sinc", "head", "domain", "dom_backbone")
    ):
        raise ValueError("expert.trainable_prefixes may not include Sinc/head/domain branches")

    encoder = _exact_mapping(root["encoder"], _ENCODER_KEYS, name="encoder")
    _integer(encoder["hidden_dim"], name="encoder.hidden_dim", minimum=1)
    lr_min = _finite(encoder["lr_min"], name="encoder.lr_min", positive=True)
    lr_max = _finite(encoder["lr_max"], name="encoder.lr_max", positive=True)
    if lr_min >= lr_max:
        raise ValueError("encoder.lr_min must be smaller than encoder.lr_max")

    trainer = _exact_mapping(root["trainer"], _TRAINER_KEYS, name="trainer")
    _integer(trainer["inner_steps"], name="trainer.inner_steps", minimum=1, maximum=10)
    fraction = _finite(
        trainer["receiver_cvar_fraction"],
        name="trainer.receiver_cvar_fraction",
        positive=True,
    )
    if fraction > 1.0:
        raise ValueError("trainer.receiver_cvar_fraction must be in (0, 1]")
    for field in ("receiver_cvar_weight", "worst_class_guard_weight"):
        _finite(trainer[field], name=f"trainer.{field}", positive=True)
    if _finite(root["meta_outer_lr"], name="meta_outer_lr", positive=True) != 1e-4:
        raise ValueError("meta_outer_lr must be 1e-4")
    if _integer(root["outer_cycles"], name="outer_cycles", minimum=1) != 10:
        raise ValueError("outer_cycles must be 10")
    return copy.deepcopy(dict(root))


def build_marc_ot_functional_forward(
    model: nn.Module,
    base_state: Mapping[str, Tensor],
) -> Callable[[Mapping[str, Tensor], Tensor], Tensor]:
    """Build a frozen-head functional forward with canonical fast overrides only."""

    if not isinstance(model, nn.Module):
        raise TypeError("model must be torch.nn.Module")
    full_base = {name: value.detach().clone() for name, value in base_state.items()}
    model_state = model.state_dict()
    if set(full_base) != set(model_state):
        raise ValueError("base_state must contain the complete model parameters and buffers")
    parameter_names = {name for name, _parameter in model.named_parameters()}
    for name, value in full_base.items():
        expected = model_state[name]
        if not isinstance(value, Tensor) or value.shape != expected.shape or value.dtype != expected.dtype:
            raise ValueError(f"base_state geometry drift for {name!r}")

    def functional_forward(fast_state: Mapping[str, Tensor], values: Tensor) -> Tensor:
        if not isinstance(fast_state, Mapping) or not fast_state:
            raise ValueError("fast state must be a non-empty mapping")
        state = dict(full_base)
        for name, value in fast_state.items():
            if (
                name not in parameter_names
                or name not in full_base
                or parameter_block_key(name) is None
            ):
                raise ValueError(f"fast parameter is outside canonical identity blocks: {name!r}")
            if not isinstance(value, Tensor) or value.shape != full_base[name].shape:
                raise ValueError(f"fast parameter geometry drift: {name!r}")
            if value.dtype != full_base[name].dtype or value.device != full_base[name].device:
                raise ValueError(f"fast parameter dtype/device drift: {name!r}")
            state[name] = value
        try:
            output = torch.func.functional_call(
                model,
                state,
                (values,),
                {"return_aux": True},
                strict=True,
            )
        except TypeError:
            output = torch.func.functional_call(model, state, (values,), strict=True)
        if isinstance(output, Tensor):
            logits = output
        elif isinstance(output, Mapping):
            logits = output.get("tx_logits", output.get("logits"))
        else:
            logits = None
        if not isinstance(logits, Tensor) or logits.ndim != 2 or not logits.is_floating_point():
            raise ValueError("functional model must return frozen-head floating logits")
        if not bool(torch.isfinite(logits).all()):
            raise FloatingPointError("functional logits are non-finite")
        return logits

    return functional_forward


def build_task_coordinate_bank(
    base_checkpoint_id: str,
    task_deltas: Mapping[Any, Mapping[str, Tensor]],
    *,
    max_rank: int,
    validated_bank: WeightDeltaBank | None = None,
) -> WeightDeltaBank:
    """Build an exact expert-task coordinate bank for strict receiver masking.

    Unlike an SVD basis, every column has one stable task/receiver identity.
    This permits a leave-one-receiver-out fold to remove all and only the
    excluded receiver's historical expert directions.
    """

    validated = validated_bank
    if validated is None:
        validated = fit_weight_delta_bank(
            base_checkpoint_id,
            task_deltas,
            max_rank=max(0, int(max_rank)),
        )
    elif (
        validated.base_checkpoint_id != base_checkpoint_id
        or set(validated.task_keys) != set(task_deltas)
        or not validated.entries
    ):
        raise ValueError("validated expert bank identity/task coverage drift")
    task_count = len(validated.task_keys)
    if task_count > int(max_rank):
        raise ValueError(
            "exact LORO task-coordinate bank exceeds expert.max_rank; "
            "receiver columns may not be compressed or mixed"
        )
    entries: list[DeltaBankEntry] = []
    for validated_entry in validated.entries:
        spec = validated_entry.spec
        columns: list[Tensor] = []
        for task_key in validated.task_keys:
            delta = task_deltas[task_key]
            columns.append(
                torch.cat(
                    [
                        delta[name].detach().reshape(-1).to(device="cpu", dtype=torch.float32)
                        for name in spec.parameter_names
                    ]
                )
            )
        basis = torch.stack(columns, dim=1).contiguous().requires_grad_(True)
        entries.append(
            DeltaBankEntry(
                spec=spec,
                basis=basis,
                task_coefficients=torch.eye(task_count, dtype=torch.float32),
                effective_rank=task_count,
                relative_error=0.0,
            )
        )
    return WeightDeltaBank(
        schema=validated.schema,
        base_checkpoint_id=validated.base_checkpoint_id,
        task_keys=validated.task_keys,
        entries=tuple(entries),
    )


def build_loro_coefficient_mask(
    bank: WeightDeltaBank,
    *,
    excluded_receiver: int | str,
) -> Tensor:
    """Return the fixed-coordinate mask implementing historical bank M_-d."""

    receiver = str(int(excluded_receiver))
    task_mask = torch.tensor(
        [0.0 if key.receiver == receiver else 1.0 for key in bank.task_keys],
        dtype=torch.float32,
    )
    if not bool((task_mask == 0.0).any()):
        raise ValueError("pseudo-target receiver has no expert coordinate to hold out")
    if not bool((task_mask == 1.0).any()):
        raise ValueError("LORO fold must retain at least one other receiver expert")
    masks: list[Tensor] = []
    for entry in bank.entries:
        if entry.effective_rank != len(bank.task_keys):
            raise ValueError("LORO requires an exact task-coordinate bank")
        masks.append(task_mask)
    return torch.cat(masks, dim=0)


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(dict(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _load_config_argument(args: Any) -> Any:
    value = getattr(args, "config", None)
    if isinstance(value, Mapping):
        return dict(value)
    if value is None:
        raise ValueError("args.config is required")
    path = Path(value)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _physical_ids(dataset: Any) -> set[str]:
    sample_index = getattr(dataset, "index", None)
    values: list[str] = []
    if sample_index is not None:
        from dataset_wisig import wisig_physical_sample_id

        values = [str(wisig_physical_sample_id(item)) for item in sample_index]
    else:
        for index in range(len(dataset)):
            item = dataset[index]
            if not isinstance(item, (tuple, list)) or not isinstance(item[-1], Mapping):
                raise ValueError("source role dataset must expose physical_sample_id metadata")
            values.append(str(item[-1].get("physical_sample_id", "")))
    if not values or any(not value for value in values):
        raise ValueError("source role dataset has missing physical IDs")
    if len(set(values)) != len(values):
        raise ValueError("source role dataset repeats a physical ID")
    return set(values)


def _validate_role_datasets(role_datasets: Any) -> tuple[dict[str, Any], dict[str, int]]:
    roles = _exact_mapping(role_datasets, set(_ROLE_RATIOS), name="source role datasets")
    copied = dict(roles)
    sizes = {role: int(len(dataset)) for role, dataset in copied.items()}
    if any(size <= 0 for size in sizes.values()):
        raise ValueError("every source role dataset must be non-empty")
    ids = {role: _physical_ids(dataset) for role, dataset in copied.items()}
    for left, right in (("L_s", "U_s"), ("L_s", "V"), ("U_s", "V")):
        overlap = ids[left].intersection(ids[right])
        if overlap:
            raise ValueError(f"source role physical IDs overlap: {left}/{right} count={len(overlap)}")
    return copied, sizes


def _build_default_role_datasets(ds_w: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(ds_w, Mapping) or "data" not in ds_w:
        raise ValueError("default MARC-OT Phase1 path requires a loaded ManySig mapping")
    from dataset_wisig import WiSigCompactDataset, WiSigSubsetDataset
    from SSDG.train_ssdg import split_tx_rx_day_1_7_2

    wisig = config["wisig"]
    maximum = int(wisig["max_samples_per_combo"])
    base = WiSigCompactDataset(
        ds_w,
        out_len=int(wisig["out_len"]),
        crop_mode=str(wisig["crop_mode"]),
        normalize=bool(wisig["normalize"]),
        equalized=int(wisig["equalized"]),
        day_keep=list(config["source_days"]),
        rx_keep=list(config["source_receiver_ids"]),
        domain=str(wisig["domain"]),
        max_samples_per_combo=None if maximum <= 0 else maximum,
        seed=int(config["seed"]),
        build_index=True,
    )
    labeled, unlabeled, validation = split_tx_rx_day_1_7_2(
        base,
        labeled_ratio=float(config["source_roles"]["L_s"]),
        unlabeled_ratio=float(config["source_roles"]["U_s"]),
        source_val_ratio=float(config["source_roles"]["V"]),
    )
    return {
        "L_s": WiSigSubsetDataset(base, labeled, split_source="marc_ot_phase1_L_s"),
        "U_s": WiSigSubsetDataset(base, unlabeled, split_source="marc_ot_phase1_U_s"),
        "V": WiSigSubsetDataset(base, validation, split_source="marc_ot_phase1_V"),
    }


def _resolve_input_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _build_default_model(
    config: Mapping[str, Any], ds_w: Mapping[str, Any], device: torch.device
) -> nn.Module:
    from .meta_phase1_entry import (
        _build_meta_model,
        _load_checkpoint_payload,
        _load_legacy_checkpoint_into_meta_model,
    )

    checkpoint = _resolve_input_path(str(config["base_checkpoint"]))
    if not checkpoint.is_file() or checkpoint.is_symlink():
        raise FileNotFoundError(f"base checkpoint is not a regular file: {checkpoint}")
    payload = _load_checkpoint_payload(checkpoint, device)
    compatibility_config = dict(config)
    compatibility_config["adapter"] = {"rank": 0, "sites": []}
    model, _model_args = _build_meta_model(
        compatibility_config,
        ds_w,
        payload,
        device,
        adapter_rank=0,
        adapter_sites="",
    )
    _load_legacy_checkpoint_into_meta_model(
        model,
        compatibility_config,
        ds_w,
        payload,
        device,
        allow_nested_dual_bridge=True,
    )
    return model


def _materialize_expert_task_batches(
    selected_episodes: Sequence[Any],
    l_s_refs: Sequence[Any],
    l_s_dataset: Any,
    *,
    device: torch.device,
    view_seed: int,
) -> dict[Any, Mapping[str, Any]]:
    from .meta_phase1_entry import _materialize_ref_view

    episodes_by_key: dict[Any, list[Any]] = defaultdict(list)
    for episode in selected_episodes:
        selection = canonical_episode_task_domain_selection(episode)
        episodes_by_key[selection.task_key].append(episode)
    batches: dict[Any, Mapping[str, Any]] = {}
    for task_key, episodes in episodes_by_key.items():
        excluded = {
            str(ref.physical_sample_id)
            for episode in episodes
            for ref in episode.query_adapt + episode.query_guard
        }
        by_physical: dict[str, Any] = {}
        for ref in l_s_refs:
            physical_id = str(ref.physical_sample_id)
            if (
                int(ref.rx_i) == int(task_key.receiver)
                and int(ref.day_i) == int(task_key.day)
                and int(ref.capture_block_i) == int(task_key.capture_block)
                and str(ref.view) == str(task_key.scene)
                and physical_id not in excluded
            ):
                by_physical.setdefault(physical_id, ref)
        refs = tuple(sorted(by_physical.values(), key=lambda row: (int(row.tx_i), str(row.physical_sample_id))))
        if not refs:
            raise ValueError(f"source expert task has no non-query L_s rows: {task_key!r}")
        rows: list[Tensor] = []
        labels: list[int] = []
        for ref in refs:
            item = l_s_dataset[int(ref.dataset_index)]
            if not isinstance(item, (tuple, list)) or len(item) < 2:
                raise ValueError("L_s expert dataset item must contain IQ and TX label")
            x = item[0] if isinstance(item[0], Tensor) else torch.as_tensor(item[0])
            rows.append(
                _materialize_ref_view(
                    x.detach().float(),
                    ref,
                    view_seed=int(view_seed),
                )
            )
            labels.append(int(ref.tx_i))
        physical_ids = tuple(str(ref.physical_sample_id) for ref in refs)
        if set(physical_ids).intersection(excluded):
            raise RuntimeError("source expert batch contains an outer query physical ID")
        batches[task_key] = {
            "iq": torch.stack(rows).to(device),
            "labels": torch.tensor(labels, dtype=torch.long, device=device),
            "physical_ids": physical_ids,
            "excluded_outer_query_ids": frozenset(excluded),
            "refs": refs,
            "source_role": "L_s",
        }
    if len(batches) < 2:
        raise ValueError("MARC-OT Phase1 requires at least two unique source expert tasks")
    return batches


def _training_selector(training_k: Sequence[int]) -> Callable[[tuple[Any, ...]], tuple[Any, ...]]:
    allowed = frozenset(int(value) for value in training_k)

    def select(episodes: tuple[Any, ...]) -> tuple[Any, ...]:
        selected = tuple(episode for episode in episodes if int(episode.k_shot) in allowed)
        if len(selected) != 11 or {int(episode.k_shot) for episode in selected} != {10}:
            raise ValueError("training selector must materialize exactly the 11 canonical K=10 cells")
        return selected

    return select


def _summary_from_closure(
    config: Mapping[str, Any],
    closure: Any,
    bank: Any,
    role_sizes: Mapping[str, int],
) -> dict[str, Any]:
    if not Path(closure.bundle_path).is_file() or Path(closure.bundle_path).is_symlink():
        raise RuntimeError("MARC-OT bundle strict readback did not produce a regular file")
    if getattr(closure, "loaded_bundle", None) is None:
        raise RuntimeError("MARC-OT bundle strict readback result is missing")
    if getattr(closure, "pilot_executed", None) is not False:
        raise RuntimeError("Phase1 bundle entry must never claim pilot execution")
    software = dict(closure.software_coverage)
    training = dict(closure.training_coverage)
    entries = tuple(bank.entries)
    return {
        "schema": MARC_OT_PHASE1_BUNDLE_SCHEMA,
        "status": "BUNDLE_PRODUCED_NO_PERFORMANCE_CLAIM",
        "run_id": str(config["run_id"]),
        "base_checkpoint_id": str(config["base_checkpoint_id"]),
        "software_supported_k": [int(value) for value in software["software_supported_k"]],
        "software_schedule_cell_count": int(software["episode_count"]),
        "training_coverage_k": [int(value) for value in training["k_shot"]],
        "trained_episode_count": int(training["trained_episode_count"]),
        "optimizer_step_count": int(training["optimizer_step_count"]),
        "outer_cycles": int(training["outer_cycles"]),
        "task_count": int(len(bank.task_keys)),
        "bank": {
            "block_count": int(len(entries)),
            "effective_ranks": [int(entry.effective_rank) for entry in entries],
            "coefficient_dim": int(sum(int(entry.effective_rank) for entry in entries)),
            "coordinate_system": "EXACT_TASK_RECEIVER_COLUMNS",
        },
        "loro_memory_exclusion": "M_MINUS_D_FOR_EVERY_TRAINING_EPISODE",
        "source_role_sizes": {name: int(role_sizes[name]) for name in ("L_s", "U_s", "V")},
        "source_role_physical_ids_pairwise_disjoint": True,
        "gradient_training_roles": ["L_s"],
        "evidence_only_roles": ["U_s", "V"],
        "validation_state_updates": False,
        "phase1_source_only": True,
        "phase2_inputs_opened": False,
        "terrestrial_proxy": "WiSig/ManySig",
        "leo_view_semantics": "PHYSICS_INSPIRED_SIMULATION_NOT_IN_ORBIT_VALIDATION",
        "pilot_executed": False,
        "performance_claim": "NOT_EVALUATED",
        "bundle_path": str(Path(closure.bundle_path)),
    }


def run_marc_ot_phase1_bundle(
    args: Any,
    ds_w: Mapping[str, Any],
    *,
    injected_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one immutable source-only MARC-OT Phase1 deployment bundle."""

    output_root = Path(getattr(args, "output_root", ""))
    if not str(output_root):
        raise ValueError("args.output_root is required")
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"immutable output root already exists: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)
    config: dict[str, Any] | None = None
    try:
        config = validate_marc_ot_phase1_config(_load_config_argument(args))
        _write_json_exclusive(output_root / "config_snapshot.json", config)
        random.seed(int(config["seed"]))
        np.random.seed(int(config["seed"]) & 0xFFFF_FFFF)
        torch.manual_seed(int(config["seed"]))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(config["seed"]))
        device = torch.device(str(getattr(args, "device", "cpu") or "cpu"))
        context = dict(injected_context or {})
        model = context.get("model")
        if model is None:
            model = _build_default_model(config, ds_w, device)
        if not isinstance(model, nn.Module):
            raise TypeError("injected model must be torch.nn.Module")
        model = model.to(device)

        role_datasets = context.get("role_datasets")
        if role_datasets is None:
            role_datasets = _build_default_role_datasets(ds_w, config)
        role_datasets, role_sizes = _validate_role_datasets(role_datasets)

        from .meta_phase1_entry import _build_refs, _episode_batch

        l_s_refs, l_s_dataset = _build_refs(role_datasets["L_s"], "L_s")
        sampler = HierarchicalMetaEpisodeSampler(
            l_s_refs,
            MetaEpisodeSamplerConfig(
                k_choices=tuple(config["k_choices"]),
                query_per_class=int(config["query_per_class"]),
                allowed_roles=("L_s",),
                training=True,
                partial_coverage_probability=1.0,
                partial_class_fraction=(0.50, 0.80),
            ),
        )
        schedule = sample_marc_ot_coverage_schedule(
            sampler, seed=int(config["schedule_seed"])
        )
        selector = _training_selector(config["training_k"])
        selected = selector(schedule)
        task_batches = _materialize_expert_task_batches(
            selected,
            l_s_refs,
            l_s_dataset,
            device=device,
            view_seed=int(config["schedule_seed"]),
        )
        expert_config = MARCOTSourceExpertConfig(
            trainable_prefixes=tuple(config["expert"]["trainable_prefixes"]),
            base_checkpoint_id=str(config["base_checkpoint_id"]),
            steps=int(config["expert"]["steps"]),
            lr=float(config["expert"]["lr"]),
            max_rank=int(config["expert"]["max_rank"]),
        )
        expert_builder = context.get("build_source_expert_bank", build_source_expert_bank)
        if not callable(expert_builder):
            raise TypeError("build_source_expert_bank injection must be callable")
        expert_result = expert_builder(model, task_batches, expert_config)
        if not isinstance(getattr(expert_result, "task_deltas", None), Mapping):
            raise ValueError("source expert result must expose exact task_deltas for LORO")
        bank = build_task_coordinate_bank(
            str(config["base_checkpoint_id"]),
            expert_result.task_deltas,
            max_rank=int(config["expert"]["max_rank"]),
            validated_bank=expert_result.bank,
        )
        if bank.base_checkpoint_id != config["base_checkpoint_id"] or not bank.entries:
            raise ValueError("source expert bank identity or block coverage drift")
        coefficient_dim = sum(int(entry.effective_rank) for entry in bank.entries)
        encoder = SupportSetEncoder(
            feature_dim=MARC_OT_SUPPORT_ROW_DIM,
            coefficient_dim=coefficient_dim,
            block_count=len(bank.entries),
            hidden_dim=int(config["encoder"]["hidden_dim"]),
            lr_min=float(config["encoder"]["lr_min"]),
            lr_max=float(config["encoder"]["lr_max"]),
        ).to(device)
        basis_tensors = [entry.basis for entry in bank.entries]
        if any(not basis.requires_grad for basis in basis_tensors):
            raise ValueError("source expert bank bases must be trainable")
        if any(entry.task_coefficients.requires_grad for entry in bank.entries):
            raise ValueError("source expert coefficients must remain frozen")
        optimizer_parameters = [*encoder.parameters(), *basis_tensors]
        optimizer = torch.optim.SGD(
            optimizer_parameters,
            lr=float(config["meta_outer_lr"]),
        )
        expected_optimizer_ids = {id(value) for value in optimizer_parameters}
        actual_optimizer_ids = {
            id(value)
            for group in optimizer.param_groups
            for value in group["params"]
        }
        if actual_optimizer_ids != expected_optimizer_ids:
            raise RuntimeError("MARC-OT optimizer scope drift")

        base_state = {
            name: value.detach().clone() for name, value in model.state_dict().items()
        }
        functional_forward = build_marc_ot_functional_forward(model, base_state)
        batch_cache: dict[Any, Any] = {}

        def batch_builder(episode: Any) -> Any:
            if episode not in batch_cache:
                batch_cache[episode] = _episode_batch(
                    episode,
                    l_s_dataset,
                    model=model,
                    num_classes=int(config["model"]["num_classes"]),
                    device=device,
                    view_seed=int(config["schedule_seed"]),
                )
            return batch_cache[episode]

        trainer = config["trainer"]
        trainer_config = MetaBankTrainerConfig(
            source_receiver_ids=tuple(config["source_receiver_ids"]),
            inner_steps=int(trainer["inner_steps"]),
            receiver_cvar_fraction=float(trainer["receiver_cvar_fraction"]),
            receiver_cvar_weight=float(trainer["receiver_cvar_weight"]),
            worst_class_guard_weight=float(trainer["worst_class_guard_weight"]),
        )
        support_feature_model = context.get("support_feature_model", model)
        if not isinstance(support_feature_model, nn.Module):
            raise TypeError("support_feature_model injection must be torch.nn.Module")
        support_feature_model = support_feature_model.to(device)
        training_entry = context.get("run_bank_training", run_marc_ot_phase1_bank_training)
        if not callable(training_entry):
            raise TypeError("run_bank_training injection must be callable")
        closure = training_entry(
            sampler=sampler,
            batch_builder=batch_builder,
            functional_forward=functional_forward,
            base_state=base_state,
            base_checkpoint_id=str(config["base_checkpoint_id"]),
            bank=bank,
            support_encoder=encoder,
            support_feature_model=support_feature_model,
            trainer_config=trainer_config,
            optimizer=optimizer,
            expected_block_specs=tuple(entry.spec for entry in bank.entries),
            bundle_path=output_root / "marc_ot_weight_bundle.pt",
            training_episode_selector=selector,
            schedule_seed=int(config["schedule_seed"]),
            outer_cycles=int(config["outer_cycles"]),
            episode_coefficient_mask=lambda episode: build_loro_coefficient_mask(
                bank,
                excluded_receiver=(episode.query_adapt + episode.query_guard)[0].rx_i,
            ),
        )
        summary = _summary_from_closure(config, closure, bank, role_sizes)
        _write_json_exclusive(output_root / "summary.json", summary)
        return summary
    except Exception as error:
        failed = {
            "schema": MARC_OT_PHASE1_BUNDLE_SCHEMA,
            "status": "FAILED",
            "run_id": None if config is None else config.get("run_id"),
            "error_type": type(error).__name__,
            "error": str(error),
            "pilot_executed": False,
            "performance_claim": "NOT_EVALUATED",
        }
        summary_path = output_root / "summary.json"
        if not summary_path.exists() and not summary_path.is_symlink():
            _write_json_exclusive(summary_path, failed)
        raise


__all__ = [
    "MARC_OT_PHASE1_BUNDLE_SCHEMA",
    "build_loro_coefficient_mask",
    "build_marc_ot_functional_forward",
    "build_task_coordinate_bank",
    "run_marc_ot_phase1_bundle",
    "validate_marc_ot_phase1_config",
]
