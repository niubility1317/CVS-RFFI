#!/usr/bin/env python
"""Train source-only lightweight adapters around a frozen Phase1 backbone.

The formal effective-feature route consumes only sealed post-channel LEO_weak
source caches.  Legacy paired diagnostics remain separate and cannot enter the
formal source-only promotion path.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

CODE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_ROOT.parent
for path in (str(REPO_ROOT), str(CODE_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from cvsrffi.eval import apply_sat_channel_for_scenario  # noqa: E402
from cvsrffi.checkpoint_loading import build_exact_ssdg_model_from_checkpoint  # noqa: E402
from cvsrffi.identity_only_forward import (  # noqa: E402
    can_use_identity_only_forward,
    identity_only_feature_forward,
)
from cvsrffi.leo_weak_cache import (  # noqa: E402
    FORMAL_LEO_WEAK_SCENARIOS,
    PHASE2_SAMPLE_VIEW_POLICY,
    load_verified_leo_weak_cache_set,
)
from cvsrffi.tensors import make_torch_generator  # noqa: E402
from cvsrffi.wisig_fewshot_payload import canonical_tx_id, parse_tx_id_list  # noqa: E402
from dataset_wisig import WiSigSubsetDataset  # noqa: E402
from eval_feature_diagnosis import collect_feature_dict  # noqa: E402
from export_spaceborne_features import (  # noqa: E402
    SATELLITE_TTA_POLICIES,
    _build_wisig_dataset,
    _leo_repair_canonical_iq,
    _meta_to_list,
    _rms_normalize_iq,
    _resolve_tx_indices,
    _satellite_tta_view_count,
    _satellite_tta_views,
    _spectral_logmag_sketch_batch,
    _validate_star_ground_impl,
)
from training_controls import parse_sat_scenarios, sat_channel_config_for_scenario  # noqa: E402


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


FORMAL_RX_LIGHT5_ORDER = (
    "rx_base",
    "rx_shift_m2",
    "rx_shift_p2",
    "rx_cfo_m1e4",
    "rx_cfo_p1e4",
)


def _lazy_formal_training_view_pair(
    rows: torch.Tensor, extra_index: int
) -> tuple[tuple[str, torch.Tensor], tuple[str, torch.Tensor]]:
    """Build base plus one rotating view without pre-materializing rx_light5."""

    if int(extra_index) not in (1, 2, 3, 4):
        raise ValueError("formal extra receive-view index must be in [1,4]")
    name = FORMAL_RX_LIGHT5_ORDER[int(extra_index)]
    if name == "rx_shift_m2":
        extra = torch.roll(rows, shifts=-2, dims=-1)
    elif name == "rx_shift_p2":
        extra = torch.roll(rows, shifts=2, dims=-1)
    else:
        cfo_views = dict(_satellite_tta_views(rows, "rx_cfo3"))
        if tuple(cfo_views) != ("rx_base", "rx_cfo_m1e4", "rx_cfo_p1e4"):
            raise ValueError("rx_cfo3 definition drift")
        extra = cfo_views[name]
    return (("rx_base", rows), (name, extra))


class IQResidualPreAdapter(nn.Module):
    def __init__(self, hidden: int = 32, alpha: float = 0.25) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.net = nn.Sequential(
            nn.Conv1d(2, int(hidden), kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(int(hidden), int(hidden), kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(int(hidden), 2, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.float() + self.alpha * torch.tanh(self.net(x.float()))
        rms = torch.sqrt(torch.mean(y.square(), dim=(1, 2), keepdim=True).clamp_min(1e-8))
        return (y / rms).to(dtype=x.dtype)


class IdentityPreAdapter(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def _float32_numpy_to_tensor(value: np.ndarray) -> torch.Tensor:
    """Bridge NumPy IQ rows into Torch without changing the sample values.

    Torch 2.1's ``from_numpy`` rejects NumPy 2.x arrays even when the object is
    an exact ``np.ndarray``.  ``frombuffer`` uses the same contiguous float32
    storage and keeps the compatibility repair local to the host data bridge.
    """

    array = np.ascontiguousarray(value, dtype=np.float32)
    try:
        return torch.from_numpy(array)
    except TypeError as exc:
        if type(array) is not np.ndarray or "expected np.ndarray" not in str(exc):
            raise
        return torch.frombuffer(
            memoryview(array), dtype=torch.float32
        ).reshape(array.shape)


class SealedLeoWeakSourceDataset(Dataset):
    """Source rows from a verified Phase1 post-channel cache set."""

    def __init__(
        self,
        arrays_by_scenario: dict[str, dict[str, np.ndarray]],
        *,
        tx_labels: Sequence[str],
    ) -> None:
        labels = tuple(str(value) for value in tx_labels)
        if not labels or len(labels) != len(set(labels)):
            raise ValueError("source tx_labels must be nonempty and unique")
        class_by_tx = {label: index for index, label in enumerate(labels)}
        iq_parts: list[np.ndarray] = []
        class_parts: list[np.ndarray] = []
        scenario_parts: list[np.ndarray] = []
        sample_id_parts: list[np.ndarray] = []
        for scenario in FORMAL_LEO_WEAK_SCENARIOS:
            arrays = arrays_by_scenario[str(scenario)]
            observed = np.asarray(arrays["tx_ids"]).astype(str)
            if set(observed.tolist()) != set(labels):
                raise ValueError(
                    f"source TX set drift for {scenario}: "
                    f"{sorted(set(observed.tolist()))}!={sorted(labels)}"
                )
            iq_parts.append(np.asarray(arrays["leo_weak_iq"], dtype=np.float32))
            class_parts.append(
                np.asarray([class_by_tx[value] for value in observed], dtype=np.int64)
            )
            scenario_parts.append(np.asarray(arrays["sat_scenarios"]).astype(str))
            sample_id_parts.append(np.asarray(arrays["sample_ids"]).astype(str))
        self.iq = np.concatenate(iq_parts, axis=0)
        self.class_ids = np.concatenate(class_parts, axis=0)
        self.scenarios = np.concatenate(scenario_parts, axis=0)
        self.sample_ids = np.concatenate(sample_id_parts, axis=0)
        self.tx_labels = labels

    def __len__(self) -> int:
        return int(len(self.iq))

    def __getitem__(self, index: int):
        return (
            _float32_numpy_to_tensor(self.iq[int(index)]),
            torch.tensor(int(self.class_ids[int(index)]), dtype=torch.long),
            torch.tensor(0, dtype=torch.long),
            {
                "sat_scenario": str(self.scenarios[int(index)]),
                "sample_id": str(self.sample_ids[int(index)]),
                "phase2_sample_view_policy": PHASE2_SAMPLE_VIEW_POLICY,
            },
        )


def _apply_input_repair(x: torch.Tensor, mode: str) -> torch.Tensor:
    mode_norm = str(mode or "raw").strip().lower()
    if mode_norm in {"", "raw", "none", "identity"}:
        return x
    if mode_norm == "rms":
        return _rms_normalize_iq(x, clip_sigma=2.8)
    if mode_norm == "canonical":
        return _leo_repair_canonical_iq(x)
    if mode_norm == "canonical_m1e4":
        return _leo_repair_canonical_iq(x, residual_delta=-1.0e-4)
    if mode_norm == "canonical_p1e4":
        return _leo_repair_canonical_iq(x, residual_delta=1.0e-4)
    raise ValueError(f"unknown input_repair={mode!r}")


def _feature_forward(model: nn.Module, x: torch.Tensor, feature_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    identity_only = identity_only_feature_forward(model, x, feature_name)
    if identity_only is not None:
        return identity_only
    out = model(x, y_tx=None, grl_lambda=1.0, return_aux=True)
    feats = collect_feature_dict(out)
    if feature_name not in feats:
        raise KeyError(f"feature {feature_name!r} not found; available={sorted(feats.keys())}")
    logits = out.get("tx_logits", out.get("logits")) if isinstance(out, dict) else None
    if logits is None:
        raise KeyError("model output does not include tx_logits/logits")
    return feats[feature_name].float(), logits.float()


def _build_model(args: argparse.Namespace, source_ds, device: torch.device, *, freeze: bool = True) -> nn.Module:
    try:
        # Project checkpoints include trusted local training metadata classes;
        # PyTorch 2.6 changed the default to weights_only=True.
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(args.ckpt, map_location="cpu")
    model, checkpoint_load_audit = build_exact_ssdg_model_from_checkpoint(
        ckpt,
        input_len=int(args.wisig_out_len),
        device=device,
    )
    model._checkpoint_load_audit = checkpoint_load_audit
    model.eval()
    for p in model.parameters():
        p.requires_grad_(not bool(freeze))
    return model


def _configure_model_adapter(model: nn.Module, mode: str) -> dict[str, Any]:
    mode_norm = str(mode or "none").strip().lower()
    for p in model.parameters():
        p.requires_grad_(False)
    if mode_norm in {"", "none", "off", "0"}:
        return {"mode": "none", "trainable_parameters": 0, "trainable_tensors": []}

    def is_frozen_classifier_param(param_name: str) -> bool:
        return (
            ".cls_head.head." in param_name
            or ".cls_head.dac_head." in param_name
            or ".cls_head.pa_head." in param_name
            or ".classifier." in param_name
        )

    trainable: list[str] = []
    for name, p in model.named_parameters():
        lname = name.lower()
        allow = False
        if mode_norm == "id_feature_head":
            allow = (
                lname.startswith("id_backbone.cls_head.")
                and not is_frozen_classifier_param(lname)
            )
        elif mode_norm == "id_late_feature":
            allow = (
                lname.startswith("id_backbone.cls_head.")
                and not is_frozen_classifier_param(lname)
            ) or (
                lname.startswith("id_backbone.")
                and any(token in lname for token in (".fuse.", ".con_proj.", ".t_proj.", ".f_proj.", ".dac_proj.", ".pa_proj."))
            )
        elif mode_norm == "id_norm_late_feature":
            allow = (
                lname.startswith("id_backbone.cls_head.")
                and not is_frozen_classifier_param(lname)
            ) or (
                lname.startswith("id_backbone.")
                and any(token in lname for token in (".fuse.", ".con_proj.", ".t_proj.", ".f_proj.", ".dac_proj.", ".pa_proj."))
            ) or (
                lname.startswith("id_backbone.")
                and ("norm" in lname or "gate" in lname)
            )
        elif mode_norm == "id_full_feature":
            allow = (
                lname.startswith("id_backbone.")
                and not is_frozen_classifier_param(lname)
            )
        else:
            raise ValueError(f"unknown model_adapter_mode={mode!r}")
        if allow:
            p.requires_grad_(True)
            trainable.append(name)
    n_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    if n_params <= 0:
        raise ValueError(f"model_adapter_mode={mode!r} selected no trainable parameters")
    return {"mode": mode_norm, "trainable_parameters": n_params, "trainable_tensors": trainable}


def _make_source_loader(args: argparse.Namespace):
    if str(args.model_adapter_mode).strip().lower() == "lora_effective_feature":
        cache_set_path = Path(str(args.source_leo_weak_cache_set_manifest))
        arrays_by_scenario, cache_set_manifest, cache_set_audit = (
            load_verified_leo_weak_cache_set(
                cache_set_path,
                expected_scope="source_train",
                allowed_roles={"source"},
            )
        )
        source_tx_labels = parse_tx_id_list(str(args.source_tx_ids))
        if len(source_tx_labels) != int(args.num_old_classes):
            raise ValueError(
                "source TX label count must equal num_old_classes for sealed training"
            )
        source_ds = SealedLeoWeakSourceDataset(
            arrays_by_scenario,
            tx_labels=source_tx_labels,
        )
        expected_rxs = set(parse_tx_id_list(str(args.source_rxs)))
        observed_rxs = set(
            np.asarray(
                arrays_by_scenario[FORMAL_LEO_WEAK_SCENARIOS[0]]["rx_ids"]
            )
            .astype(str)
            .tolist()
        )
        if observed_rxs != expected_rxs:
            raise ValueError(
                "sealed source receiver scope does not match --source_rxs: "
                f"{sorted(observed_rxs)}!={sorted(expected_rxs)}"
            )
        if int(source_ds.iq.shape[-1]) != int(args.wisig_out_len):
            raise ValueError(
                "sealed source LEO_weak input length does not match wisig_out_len"
            )
        source_info = {
            "role": "source",
            "input_stage": "phase1_offline_prechannel_export",
            "phase2_sample_view_policy": PHASE2_SAMPLE_VIEW_POLICY,
            "clean_sample_access": False,
            "clean_derived_signal_access": False,
            "cache_set_manifest": str(cache_set_path),
            "cache_set_manifest_sha256": str(cache_set_audit["sha256"]),
            "cache_set_manifest_payload": cache_set_manifest,
            "cache_set_audit": cache_set_audit,
            "tx_labels": source_tx_labels,
            "size": len(source_ds),
        }
        args._source_leo_weak_cache_info = source_info
        return (
            DataLoader(
                source_ds,
                batch_size=int(args.batch_size),
                shuffle=True,
                num_workers=0,
                drop_last=False,
            ),
            source_ds,
            source_info,
        )
    source_ds, source_info = _build_wisig_dataset(
        pkl_path=str(args.wisig_pkl),
        tx_spec=str(args.source_tx_ids),
        role="source",
        equalized=str(args.wisig_equalized),
        out_len=int(args.wisig_out_len),
        domain=str(args.wisig_domain),
        days=args.source_days,
        rxs=args.source_rxs,
        max_samples_per_combo=int(args.max_samples_per_combo),
        max_samples_per_tx=int(args.max_source_samples_per_tx),
        seed=int(args.seed),
    )
    return DataLoader(source_ds, batch_size=int(args.batch_size), shuffle=True, num_workers=0, drop_last=False), source_ds, source_info


def _make_proxy_unknown_train_loader(args: argparse.Namespace):
    proxy_weights = (
        float(args.proxy_unknown_separation_weight)
        + float(args.proxy_unknown_supcon_weight)
        + float(args.proxy_unknown_proto_ce_weight)
        + float(args.proxy_unknown_pair_margin_weight)
        + float(args.proxy_unknown_old_margin_weight)
        + float(args.proxy_unknown_hard_pair_margin_weight)
        + float(args.proxy_unknown_hard_old_margin_weight)
    )
    if proxy_weights <= 0:
        return None, {}
    proxy_ds, proxy_info = _build_wisig_dataset(
        pkl_path=str(args.new_wisig_pkl),
        tx_spec=str(args.proxy_unknown_tx_ids),
        role="proxy_unknown_train",
        equalized=str(args.wisig_equalized),
        out_len=int(args.wisig_out_len),
        domain=str(args.wisig_domain),
        days=None,
        rxs=str(args.proxy_unknown_rxs),
        max_samples_per_combo=int(args.max_proxy_unknown_samples_per_combo),
        max_samples_per_tx=int(args.max_proxy_unknown_train_samples_per_tx),
        seed=int(args.seed) + 1801,
    )
    loader = DataLoader(proxy_ds, batch_size=int(args.batch_size), shuffle=True, num_workers=0, drop_last=False)
    return loader, proxy_info


def _proto_from_loader(model: nn.Module, loader: DataLoader, args: argparse.Namespace, device: torch.device) -> torch.Tensor:
    feat_parts: list[torch.Tensor] = []
    label_parts: list[torch.Tensor] = []
    with torch.no_grad():
        for x, y, _d, _meta in loader:
            x = x.to(device, non_blocking=True)
            z, _ = _feature_forward(model, x, str(args.feature_name))
            feat_parts.append(z.detach())
            label_parts.append(y.to(device).long())
    feats = torch.cat(feat_parts, dim=0)
    labels = torch.cat(label_parts, dim=0)
    protos = []
    for c in range(int(args.num_old_classes)):
        idx = torch.where(labels == c)[0]
        if idx.numel() == 0:
            raise ValueError(f"missing source class {c} for clean prototype")
        protos.append(feats.index_select(0, idx).mean(dim=0))
    return torch.stack(protos, dim=0)


def _leo_proto_from_loader(
    model: nn.Module,
    loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> torch.Tensor:
    """Build source prototypes directly from sealed post-channel LEO_weak rows."""

    feat_parts: list[torch.Tensor] = []
    label_parts: list[torch.Tensor] = []
    with torch.no_grad():
        for x, y, _d, meta in loader:
            x = x.to(device, non_blocking=True)
            policy_values = meta.get("phase2_sample_view_policy", [])
            if any(str(value) != PHASE2_SAMPLE_VIEW_POLICY for value in policy_values):
                raise ValueError("source loader exposed a non-LEO_weak-only row")
            z, _ = _feature_forward(model, x, str(args.feature_name))
            feat_parts.append(z.detach())
            label_parts.append(y.to(device).long())
    feats = torch.cat(feat_parts, dim=0)
    labels = torch.cat(label_parts, dim=0)
    protos = []
    for class_id in range(int(args.num_old_classes)):
        positions = torch.where(labels == class_id)[0]
        if positions.numel() == 0:
            raise ValueError(f"missing source class {class_id} for leo_weak prototype")
        protos.append(feats.index_select(0, positions).mean(dim=0))
    return torch.stack(protos, dim=0)


def _scenario_for_step(scenarios: Sequence[str], step: int) -> str:
    if not scenarios:
        raise ValueError("at least one satellite scenario is required")
    return str(scenarios[int(step) % len(scenarios)])


def _proto_margin_loss(z_ref: torch.Tensor, z_new: torch.Tensor, y: torch.Tensor, protos: torch.Tensor, tolerance: float) -> torch.Tensor:
    ref_sims = F.normalize(z_ref.detach(), dim=1) @ F.normalize(protos.detach(), dim=1).t()
    new_sims = F.normalize(z_new, dim=1) @ F.normalize(protos.detach(), dim=1).t()
    idx = y.long().view(-1, 1)
    true_ref = ref_sims.gather(1, idx).squeeze(1)
    true_new = new_sims.gather(1, idx).squeeze(1)
    mask = torch.ones_like(ref_sims, dtype=torch.bool)
    mask.scatter_(1, idx, False)
    other_ref = ref_sims.masked_fill(~mask, -1.0e9).max(dim=1).values
    other_new = new_sims.masked_fill(~mask, -1.0e9).max(dim=1).values
    ref_margin = true_ref - other_ref
    new_margin = true_new - other_new
    return F.relu(ref_margin - new_margin - float(tolerance)).mean()


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor | None) -> torch.Tensor:
    if weights is None:
        return values.mean()
    w = weights.to(device=values.device, dtype=values.dtype).view(-1)
    return (values.view(-1) * w).sum() / w.sum().clamp_min(1e-8)


def relation_gram_preservation_loss(
    adapted: torch.Tensor, reference: torch.Tensor
) -> torch.Tensor:
    """Preserve pairwise identity geometry of the frozen ADV3B02 teacher."""

    if adapted.ndim != 2 or reference.shape != adapted.shape:
        raise ValueError("relation loss requires matching [N,D] features")
    if adapted.shape[0] < 2:
        return adapted.sum() * 0.0
    student = F.normalize(adapted.float(), dim=1)
    teacher = F.normalize(reference.detach().float(), dim=1)
    mask = ~torch.eye(student.shape[0], dtype=torch.bool, device=student.device)
    return F.smooth_l1_loss(
        (student @ student.t())[mask], (teacher @ teacher.t())[mask]
    )


def prototype_gram_deconfusion_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    max_cosine: float = 0.65,
) -> torch.Tensor:
    """Penalize correlated class prototypes without old/new role information."""

    normalized = F.normalize(features.float(), dim=1)
    prototypes = []
    for class_id in torch.unique(labels, sorted=True):
        mask = labels == class_id
        prototypes.append(F.normalize(normalized[mask].mean(dim=0), dim=0))
    if len(prototypes) < 2:
        return normalized.sum() * 0.0
    banks = torch.stack(prototypes)
    gram = banks @ banks.t()
    mask = ~torch.eye(len(prototypes), dtype=torch.bool, device=gram.device)
    return F.relu(gram[mask] - float(max_cosine)).square().mean()


def nested_k_worst_prototype_risk(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    k_values: Sequence[int] = (1, 2, 5, 10, 20),
    temperature: float = 0.07,
    risk_tau: float = 0.2,
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    """Source-only nested-K episode risk with log-sum-exp worst-K pooling."""

    if features.ndim != 2 or labels.ndim != 1 or len(features) != len(labels):
        raise ValueError("nested-K risk requires features [N,D] and labels [N]")
    normalized = F.normalize(features.float(), dim=1)
    class_ids = torch.unique(labels, sorted=True)
    losses: dict[int, torch.Tensor] = {}
    for raw_k in k_values:
        k = int(raw_k)
        if k < 1:
            raise ValueError("nested-K values must be positive")
        if any(
            int(torch.sum(labels == class_id).item()) <= k
            for class_id in class_ids
        ):
            continue
        support_prototypes = []
        query_rows = []
        query_targets = []
        for local_class, class_id in enumerate(class_ids):
            positions = torch.nonzero(labels == class_id, as_tuple=False).reshape(-1)
            support_count = k
            support_prototypes.append(
                F.normalize(
                    normalized[positions[:support_count]].mean(dim=0), dim=0
                )
            )
            remaining = positions[support_count:]
            if len(remaining):
                query_rows.append(normalized[remaining])
                query_targets.append(
                    torch.full(
                        (len(remaining),),
                        int(local_class),
                        dtype=torch.long,
                        device=labels.device,
                    )
                )
        if len(support_prototypes) != len(class_ids) or not query_rows:
            continue
        prototypes = torch.stack(support_prototypes)
        queries = torch.cat(query_rows)
        targets = torch.cat(query_targets)
        losses[k] = F.cross_entropy(
            queries @ prototypes.t() / max(float(temperature), 1.0e-6),
            targets,
        )
    if not losses:
        return normalized.sum() * 0.0, losses
    tau = max(float(risk_tau), 1.0e-6)
    stacked = torch.stack(list(losses.values()))
    pooled = tau * (
        torch.logsumexp(stacked / tau, dim=0)
        - np.log(float(len(losses)))
    )
    return pooled, losses


def _parse_class_loss_weights(args: argparse.Namespace, device: torch.device) -> torch.Tensor | None:
    raw = str(args.class_loss_weights or "").strip()
    if not raw:
        return None
    vals = [float(v) for v in raw.split(",") if v.strip()]
    if len(vals) != int(args.num_old_classes):
        raise ValueError(f"class_loss_weights must have {args.num_old_classes} values, got {len(vals)}")
    return torch.tensor(vals, dtype=torch.float32, device=device)


def _parse_hard_pair_ids(
    raw: str,
    device: torch.device,
    *,
    tx_labels: Sequence[Any] | None = None,
    tx_idx: Sequence[int] | None = None,
) -> torch.Tensor:
    label_to_idx: dict[str, int] = {}
    if tx_labels is not None and tx_idx is not None:
        for label, idx in zip(tx_labels, tx_idx):
            label_to_idx[canonical_tx_id(label)] = int(idx)

    def resolve_token(token: str) -> int:
        value = str(token).strip()
        try:
            return int(value)
        except ValueError:
            pass
        key = canonical_tx_id(value)
        if key in label_to_idx:
            return int(label_to_idx[key])
        raise ValueError(f"cannot resolve hard pair TX token {token!r}")

    pairs: list[tuple[int, int]] = []
    for item in str(raw or "").split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"hard pair item must be left:right, got {item!r}")
        left, right = item.split(":", 1)
        pairs.append((resolve_token(left), resolve_token(right)))
    if not pairs:
        return torch.empty((0, 2), dtype=torch.long, device=device)
    return torch.tensor(pairs, dtype=torch.long, device=device)


def _proto_margin_loss_weighted(
    z_ref: torch.Tensor,
    z_new: torch.Tensor,
    y: torch.Tensor,
    protos: torch.Tensor,
    tolerance: float,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    ref_sims = F.normalize(z_ref.detach(), dim=1) @ F.normalize(protos.detach(), dim=1).t()
    new_sims = F.normalize(z_new, dim=1) @ F.normalize(protos.detach(), dim=1).t()
    idx = y.long().view(-1, 1)
    true_ref = ref_sims.gather(1, idx).squeeze(1)
    true_new = new_sims.gather(1, idx).squeeze(1)
    mask = torch.ones_like(ref_sims, dtype=torch.bool)
    mask.scatter_(1, idx, False)
    other_ref = ref_sims.masked_fill(~mask, -1.0e9).max(dim=1).values
    other_new = new_sims.masked_fill(~mask, -1.0e9).max(dim=1).values
    ref_margin = true_ref - other_ref
    new_margin = true_new - other_new
    return _weighted_mean(F.relu(ref_margin - new_margin - float(tolerance)), sample_weights)


def _proxy_unknown_separation_loss(z_unknown: torch.Tensor, protos: torch.Tensor, max_cos: float) -> torch.Tensor:
    sims = F.normalize(z_unknown, dim=1) @ F.normalize(protos.detach(), dim=1).t()
    nearest_old = sims.max(dim=1).values
    return F.relu(nearest_old - float(max_cos)).square().mean()


def _proxy_unknown_supcon_loss(z_unknown: torch.Tensor, y_unknown: torch.Tensor, temperature: float) -> torch.Tensor:
    labels = y_unknown.long().view(-1)
    if labels.numel() <= 1 or torch.unique(labels).numel() <= 1:
        return z_unknown.sum() * 0.0
    z = F.normalize(z_unknown, dim=1)
    logits = (z @ z.t()) / max(float(temperature), 1e-6)
    eye = torch.eye(labels.numel(), dtype=torch.bool, device=labels.device)
    positive = labels[:, None].eq(labels[None, :]) & ~eye
    valid = positive.any(dim=1)
    if not bool(valid.any()):
        return z_unknown.sum() * 0.0
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    exp_logits = torch.exp(logits).masked_fill(eye, 0.0)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    loss = -(log_prob.masked_fill(~positive, 0.0).sum(dim=1) / positive.sum(dim=1).clamp_min(1))
    return loss[valid].mean()


def _proxy_unknown_episode_losses(
    z_unknown: torch.Tensor,
    y_unknown: torch.Tensor,
    old_protos: torch.Tensor,
    *,
    temperature: float,
    pair_margin: float,
    old_margin: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    labels = y_unknown.long().view(-1)
    unique_labels = torch.unique(labels)
    zero = z_unknown.sum() * 0.0
    if labels.numel() <= 1 or unique_labels.numel() <= 1:
        return zero, zero, zero
    z = F.normalize(z_unknown, dim=1)
    protos = []
    targets = torch.empty_like(labels)
    for proto_i, label in enumerate(unique_labels.tolist()):
        mask = labels == int(label)
        protos.append(F.normalize(z[mask].mean(dim=0, keepdim=True), dim=1)[0])
        targets[mask] = int(proto_i)
    proto_t = torch.stack(protos, dim=0)
    logits = z @ proto_t.t() / max(float(temperature), 1e-6)
    proto_ce = F.cross_entropy(logits, targets)
    sims = z @ proto_t.t()
    own = sims.gather(1, targets.view(-1, 1)).squeeze(1)
    neg_mask = torch.ones_like(sims, dtype=torch.bool)
    neg_mask.scatter_(1, targets.view(-1, 1), False)
    hardest_proxy = sims.masked_fill(~neg_mask, -1.0e9).max(dim=1).values
    pair_margin_loss = F.relu(hardest_proxy - own + float(pair_margin)).mean()
    old_sims = z @ F.normalize(old_protos.detach(), dim=1).t()
    hardest_old = old_sims.max(dim=1).values
    old_margin_loss = F.relu(hardest_old - own + float(old_margin)).mean()
    return proto_ce, pair_margin_loss, old_margin_loss


def _proxy_unknown_hard_pair_loss(
    z_unknown: torch.Tensor,
    y_unknown: torch.Tensor,
    old_protos: torch.Tensor,
    hard_pairs: torch.Tensor,
    *,
    pair_margin: float,
    old_margin: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    labels = y_unknown.long().view(-1)
    zero = z_unknown.sum() * 0.0
    if hard_pairs.numel() == 0 or labels.numel() <= 1:
        return zero, zero
    z = F.normalize(z_unknown, dim=1)
    old_proto = F.normalize(old_protos.detach(), dim=1)
    pair_losses: list[torch.Tensor] = []
    old_losses: list[torch.Tensor] = []
    for left, right in hard_pairs.detach().cpu().tolist():
        left_mask = labels == int(left)
        if not bool(left_mask.any()):
            continue
        left_z = z[left_mask]
        right_mask = labels == int(right)
        if bool(right_mask.any()):
            right_proto = F.normalize(z[right_mask].mean(dim=0, keepdim=True), dim=1)[0]
            pair_losses.append(F.relu(left_z @ right_proto - 1.0 + float(pair_margin)).mean())
        if 0 <= int(right) < old_proto.shape[0]:
            old_losses.append(F.relu(left_z @ old_proto[int(right)] - 1.0 + float(old_margin)).mean())
    pair_loss = torch.stack(pair_losses).mean() if pair_losses else zero
    old_loss = torch.stack(old_losses).mean() if old_losses else zero
    return pair_loss, old_loss


def train_adapter(
    args: argparse.Namespace,
    model: nn.Module,
    teacher_model: nn.Module,
    source_loader: DataLoader,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    scenarios = parse_sat_scenarios(str(args.sat_scenarios))
    _validate_star_ground_impl(str(args.star_ground_channel_impl), scenarios, field="sat_scenarios")
    model_adapter_mode = str(args.model_adapter_mode).strip().lower()
    formal_effective = model_adapter_mode == "lora_effective_feature"
    reference_identity_weight = float(
        args.leo_reference_identity_weight
        if formal_effective
        else args.clean_identity_weight
    )
    reference_cos_weight = float(
        args.leo_reference_cos_weight if formal_effective else args.clean_cos_weight
    )
    reference_margin_weight = float(
        args.leo_reference_margin_weight
        if formal_effective
        else args.clean_feature_margin_weight
    )
    if (
        model_adapter_mode == "lora_effective_feature"
        and tuple(str(value) for value in scenarios) != FORMAL_LEO_WEAK_SCENARIOS
    ):
        raise ValueError(
            "formal effective-feature training requires the exact ordered LEO_weak scenarios"
        )
    proto_loader = DataLoader(source_loader.dataset, batch_size=int(args.batch_size), shuffle=False, num_workers=0, drop_last=False)
    clean_protos = (
        _leo_proto_from_loader(
            teacher_model, proto_loader, args, device
        )
        if model_adapter_mode == "lora_effective_feature"
        else _proto_from_loader(teacher_model, proto_loader, args, device)
    )
    class_loss_weights = _parse_class_loss_weights(args, device)
    proxy_weight_values = (
        float(args.proxy_unknown_separation_weight),
        float(args.proxy_unknown_supcon_weight),
        float(args.proxy_unknown_proto_ce_weight),
        float(args.proxy_unknown_pair_margin_weight),
        float(args.proxy_unknown_old_margin_weight),
        float(args.proxy_unknown_hard_pair_margin_weight),
        float(args.proxy_unknown_hard_old_margin_weight),
    )
    if formal_effective:
        # Fail before the legacy ManyTx loader is even called.  The formal
        # effective-feature route is source LEO_weak only, so raw proxy data
        # must be unreachable rather than loaded and rejected afterwards.
        if any(value != 0.0 for value in proxy_weight_values):
            raise ValueError(
                "formal lora_effective_feature forbids proxy-receiver training; "
                "all proxy_unknown_* weights must be zero"
            )
        proxy_loader, proxy_info = None, {}
    else:
        proxy_loader, proxy_info = _make_proxy_unknown_train_loader(args)
    hard_pairs = _parse_hard_pair_ids(
        str(args.proxy_unknown_hard_pair_ids),
        device,
        tx_labels=proxy_info.get("tx_labels") if proxy_info else None,
        tx_idx=proxy_info.get("tx_idx") if proxy_info else None,
    )
    proxy_iter = itertools.cycle(proxy_loader) if proxy_loader is not None else None
    adapter: nn.Module
    if bool(args.input_adapter_enabled):
        adapter = IQResidualPreAdapter(hidden=int(args.hidden_dim), alpha=float(args.alpha)).to(device)
    else:
        adapter = IdentityPreAdapter().to(device)
    lora_scope_by_mode = {
        "lora_full_feature": "full_feature",
        "lora_effective_feature": "effective_feature",
    }
    if model_adapter_mode in lora_scope_by_mode:
        from paper_reproduction.scripts.train_export_cvs_support_lora_adapter import (
            inject_feat_joint_lora,
        )

        epoch_cap = 20 if model_adapter_mode == "lora_effective_feature" else 40
        if not 1 <= int(args.epochs) <= epoch_cap:
            raise ValueError(
                f"ground source {model_adapter_mode} epochs must be in [1,{epoch_cap}]"
            )
        model_adapter = inject_feat_joint_lora(
            model,
            rank=int(args.lora_rank),
            alpha=float(args.lora_alpha),
            scope=lora_scope_by_mode[model_adapter_mode],
        )
        model_adapter["mode"] = model_adapter_mode
        model_adapter["ground_training_only"] = True
        if model_adapter_mode == "lora_effective_feature" and (
            bool(args.input_adapter_enabled)
            or str(args.input_repair).strip().lower() != "raw"
        ):
            raise ValueError(
                "formal lora_effective_feature requires no IQ input adapter and raw leo_weak input"
            )
        if model_adapter_mode == "lora_effective_feature" and proxy_loader is not None:
            raise RuntimeError("formal proxy-loader reachability guard failed")
        if model_adapter_mode == "lora_effective_feature" and any(
            float(value) <= 0.0
            for value in (
                args.relation_preservation_weight,
                args.prototype_gram_weight,
                args.worst_k_risk_weight,
            )
        ):
            raise ValueError(
                "lora_effective_feature requires positive relation, prototype-Gram, "
                "and worst-K risk weights"
            )
    else:
        model_adapter = _configure_model_adapter(model, model_adapter_mode)
    opt_params = list(adapter.parameters()) + [p for p in model.parameters() if p.requires_grad]
    if not opt_params:
        raise ValueError("no trainable adapter/model parameters selected")
    opt = torch.optim.AdamW(opt_params, lr=float(args.lr), weight_decay=float(args.weight_decay))
    gen = make_torch_generator(device, int(args.seed) + 1701)
    history: list[dict[str, float]] = []
    nested_k_values = tuple(
        int(value)
        for value in str(args.worst_k_values).split(",")
        if str(value).strip()
    )
    if not nested_k_values or any(value < 1 for value in nested_k_values):
        raise ValueError("worst_k_values must be a nonempty positive integer list")
    step = 0
    for epoch in range(int(args.epochs)):
        sums = {
            "loss": 0.0,
            "mse": 0.0,
            "cos": 0.0,
            "ce": 0.0,
            "clean": 0.0,
            "feat_margin": 0.0,
            "clean_margin": 0.0,
            "proxy_unknown_sep": 0.0,
            "proxy_unknown_supcon": 0.0,
            "proxy_unknown_proto_ce": 0.0,
            "proxy_unknown_pair_margin": 0.0,
            "proxy_unknown_old_margin": 0.0,
            "proxy_unknown_hard_pair": 0.0,
            "proxy_unknown_hard_old": 0.0,
            "view_consistency": 0.0,
            "relation": 0.0,
            "prototype_gram": 0.0,
            "worst_k": 0.0,
            "resid": 0.0,
        }
        for k_value in nested_k_values:
            sums[f"nested_k_{k_value}"] = 0.0
        count = 0
        for x, y, _d, meta in source_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device).long()
            sample_weights = class_loss_weights.index_select(0, y) if class_loss_weights is not None else None
            scenario = _scenario_for_step(scenarios, step)
            step += 1
            if model_adapter_mode == "lora_effective_feature":
                policy_values = meta.get("phase2_sample_view_policy", [])
                if any(
                    str(value) != PHASE2_SAMPLE_VIEW_POLICY
                    for value in policy_values
                ):
                    raise ValueError(
                        "formal effective-feature training received a nonsealed row"
                    )
                observed_scenarios = {
                    str(value) for value in meta.get("sat_scenario", [])
                }
                if not observed_scenarios.issubset(set(FORMAL_LEO_WEAK_SCENARIOS)):
                    raise ValueError("formal source batch contains an invalid scenario")
                x_sat = x
            else:
                with torch.no_grad():
                    x_sat, _ = apply_sat_channel_for_scenario(
                        x, scenario, args, gen=gen, return_meta=False
                    )
            x_sat_in = _apply_input_repair(x_sat, str(args.input_repair))
            with torch.no_grad():
                teacher_input = (
                    x_sat_in
                    if model_adapter_mode == "lora_effective_feature"
                    else x
                )
                z_clean, logits_clean_teacher = _feature_forward(
                    teacher_model, teacher_input, str(args.feature_name)
                )
            if model_adapter_mode in lora_scope_by_mode:
                extra_index = 1 + int((step - 1) % 4)
                selected_views = _lazy_formal_training_view_pair(
                    x_sat_in, extra_index
                )
                z_selected: list[torch.Tensor] = []
                logits_selected: list[torch.Tensor] = []
                for _view_name, x_view in selected_views:
                    z_view, logits_view = _feature_forward(
                        model, adapter(x_view), str(args.feature_name)
                    )
                    z_selected.append(z_view)
                    logits_selected.append(logits_view)
                z_rep = torch.stack(z_selected, dim=0).mean(dim=0)
                logits_rep = torch.stack(logits_selected, dim=0).mean(dim=0)
                view_consistency = (
                    1.0
                    - F.cosine_similarity(z_selected[0], z_selected[1], dim=1)
                ).mean()
                x_rep = x_sat_in
            else:
                x_rep = adapter(x_sat_in)
                z_rep, logits_rep = _feature_forward(
                    model, x_rep, str(args.feature_name)
                )
                view_consistency = z_rep.sum() * 0.0
            mse_per = F.smooth_l1_loss(z_rep, z_clean, reduction="none").mean(dim=1)
            mse = _weighted_mean(mse_per, sample_weights)
            cos_per = 1.0 - F.cosine_similarity(z_rep, z_clean, dim=1)
            cos = _weighted_mean(cos_per, sample_weights)
            if float(args.proto_ce_weight) > 0 or float(args.logit_ce_weight) > 0:
                proto_logits = F.normalize(z_rep, dim=1) @ F.normalize(clean_protos, dim=1).t() / max(float(args.proto_temperature), 1e-6)
                proto_ce = F.cross_entropy(proto_logits, y, reduction="none")
                logit_ce = F.cross_entropy(logits_rep, y, reduction="none")
                ce = _weighted_mean(proto_ce, sample_weights) + float(args.logit_ce_weight) * _weighted_mean(logit_ce, sample_weights)
            else:
                ce = z_rep.sum() * 0.0 + logits_rep.sum() * 0.0
            if model_adapter_mode == "lora_effective_feature":
                # The preservation anchor is the same leo_weak base view; no
                # clean source waveform enters the formal effective route.
                z_clean_rep = z_selected[0]
            else:
                clean_mode = str(args.input_repair if str(args.clean_input_repair_mode).lower() == "same" else "raw")
                x_clean_rep = adapter(_apply_input_repair(x, clean_mode))
                z_clean_rep, _ = _feature_forward(model, x_clean_rep, str(args.feature_name))
            clean_mse = _weighted_mean(F.smooth_l1_loss(z_clean_rep, z_clean, reduction="none").mean(dim=1), sample_weights)
            clean_cos = _weighted_mean(1.0 - F.cosine_similarity(z_clean_rep, z_clean, dim=1), sample_weights)
            clean_kl = F.kl_div(
                F.log_softmax(logits_rep / max(float(args.distill_temperature), 1e-6), dim=1),
                F.softmax(logits_clean_teacher / max(float(args.distill_temperature), 1e-6), dim=1),
                reduction="batchmean",
            ) * (float(args.distill_temperature) ** 2)
            clean_loss = clean_mse + reference_cos_weight * clean_cos
            feat_margin = _proto_margin_loss_weighted(z_clean, z_rep, y, clean_protos, float(args.feature_margin_tolerance), sample_weights)
            clean_margin = _proto_margin_loss_weighted(z_clean, z_clean_rep, y, clean_protos, float(args.feature_margin_tolerance), sample_weights)
            relation = relation_gram_preservation_loss(z_rep, z_clean)
            prototype_gram = prototype_gram_deconfusion_loss(
                z_rep, y, max_cosine=float(args.prototype_gram_max_cosine)
            )
            worst_k, nested_k_losses = nested_k_worst_prototype_risk(
                z_rep,
                y,
                k_values=nested_k_values,
                temperature=float(args.worst_k_proto_temperature),
                risk_tau=float(args.worst_k_tau),
            )
            if proxy_iter is not None:
                x_u, y_u, _d_u, _meta_u = next(proxy_iter)
                x_u = x_u.to(device, non_blocking=True)
                y_u = y_u.to(device).long()
                scenario_u = _scenario_for_step(scenarios, step)
                with torch.no_grad():
                    x_u_sat, _ = apply_sat_channel_for_scenario(x_u, scenario_u, args, gen=gen, return_meta=False)
                x_u_rep = adapter(_apply_input_repair(x_u_sat, str(args.input_repair)))
                z_u_rep, _ = _feature_forward(model, x_u_rep, str(args.feature_name))
                proxy_unknown_sep = _proxy_unknown_separation_loss(z_u_rep, clean_protos, float(args.proxy_unknown_max_cos))
                proxy_unknown_supcon = _proxy_unknown_supcon_loss(z_u_rep, y_u, float(args.proxy_unknown_supcon_temperature))
                proxy_unknown_proto_ce, proxy_unknown_pair_margin, proxy_unknown_old_margin = _proxy_unknown_episode_losses(
                    z_u_rep,
                    y_u,
                    clean_protos,
                    temperature=float(args.proxy_unknown_proto_temperature),
                    pair_margin=float(args.proxy_unknown_pair_margin),
                    old_margin=float(args.proxy_unknown_old_margin),
                )
                proxy_unknown_hard_pair, proxy_unknown_hard_old = _proxy_unknown_hard_pair_loss(
                    z_u_rep,
                    y_u,
                    clean_protos,
                    hard_pairs,
                    pair_margin=float(args.proxy_unknown_hard_pair_margin),
                    old_margin=float(args.proxy_unknown_hard_old_margin),
                )
            else:
                proxy_unknown_sep = z_rep.sum() * 0.0
                proxy_unknown_supcon = z_rep.sum() * 0.0
                proxy_unknown_proto_ce = z_rep.sum() * 0.0
                proxy_unknown_pair_margin = z_rep.sum() * 0.0
                proxy_unknown_old_margin = z_rep.sum() * 0.0
                proxy_unknown_hard_pair = z_rep.sum() * 0.0
                proxy_unknown_hard_old = z_rep.sum() * 0.0
            resid = (x_rep.float() - x_sat_in.float()).square().mean()
            loss = (
                float(args.mse_weight) * mse
                + float(args.cos_weight) * cos
                + float(args.proto_ce_weight) * ce
                + reference_identity_weight * clean_loss
                + float(args.feature_margin_weight) * feat_margin
                + reference_margin_weight * clean_margin
                + float(args.proxy_unknown_separation_weight) * proxy_unknown_sep
                + float(args.proxy_unknown_supcon_weight) * proxy_unknown_supcon
                + float(args.proxy_unknown_proto_ce_weight) * proxy_unknown_proto_ce
                + float(args.proxy_unknown_pair_margin_weight) * proxy_unknown_pair_margin
                + float(args.proxy_unknown_old_margin_weight) * proxy_unknown_old_margin
                + float(args.proxy_unknown_hard_pair_margin_weight) * proxy_unknown_hard_pair
                + float(args.proxy_unknown_hard_old_margin_weight) * proxy_unknown_hard_old
                + float(args.teacher_logit_distill_weight) * clean_kl
                + float(args.multiview_consistency_weight) * view_consistency
                + float(args.relation_preservation_weight) * relation
                + float(args.prototype_gram_weight) * prototype_gram
                + float(args.worst_k_risk_weight) * worst_k
                + float(args.residual_weight) * resid
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if float(args.grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(opt_params, float(args.grad_clip))
            opt.step()
            bs = int(x.shape[0])
            count += bs
            sums["loss"] += float(loss.detach().item()) * bs
            sums["mse"] += float(mse.detach().item()) * bs
            sums["cos"] += float(cos.detach().item()) * bs
            sums["ce"] += float(ce.detach().item()) * bs
            sums["clean"] += float(clean_loss.detach().item()) * bs
            sums["feat_margin"] += float(feat_margin.detach().item()) * bs
            sums["clean_margin"] += float(clean_margin.detach().item()) * bs
            sums["proxy_unknown_sep"] += float(proxy_unknown_sep.detach().item()) * bs
            sums["proxy_unknown_supcon"] += float(proxy_unknown_supcon.detach().item()) * bs
            sums["proxy_unknown_proto_ce"] += float(proxy_unknown_proto_ce.detach().item()) * bs
            sums["proxy_unknown_pair_margin"] += float(proxy_unknown_pair_margin.detach().item()) * bs
            sums["proxy_unknown_old_margin"] += float(proxy_unknown_old_margin.detach().item()) * bs
            sums["proxy_unknown_hard_pair"] += float(proxy_unknown_hard_pair.detach().item()) * bs
            sums["proxy_unknown_hard_old"] += float(proxy_unknown_hard_old.detach().item()) * bs
            sums["view_consistency"] += float(view_consistency.detach().item()) * bs
            sums["relation"] += float(relation.detach().item()) * bs
            sums["prototype_gram"] += float(prototype_gram.detach().item()) * bs
            sums["worst_k"] += float(worst_k.detach().item()) * bs
            for k_value in nested_k_values:
                nested_loss = nested_k_losses.get(int(k_value))
                sums[f"nested_k_{k_value}"] += (
                    0.0 if nested_loss is None else float(nested_loss.detach().item()) * bs
                )
            sums["resid"] += float(resid.detach().item()) * bs
        row = {k: v / max(1, count) for k, v in sums.items()}
        row["epoch"] = float(epoch + 1)
        history.append(row)
        if (epoch + 1) % max(1, int(args.log_every)) == 0:
            print(json.dumps({"epoch": epoch + 1, **row}, ensure_ascii=False))
    return adapter, {
        "epochs": int(args.epochs),
        "history_first": history[0] if history else {},
        "history_last": history[-1] if history else {},
        "scenarios": scenarios,
        "scenario_configs": {name: sat_channel_config_for_scenario(name) for name in scenarios},
        "input_repair": str(args.input_repair),
        "input_adapter_enabled": bool(args.input_adapter_enabled),
        "clean_input_repair_mode": str(args.clean_input_repair_mode),
        "model_adapter": model_adapter,
        "proxy_unknown_train": proxy_info,
        "loss_weights": {
            "mse": float(args.mse_weight),
            "cos": float(args.cos_weight),
            "proto_ce": float(args.proto_ce_weight),
            "logit_ce": float(args.logit_ce_weight),
            "leo_reference_identity": (
                reference_identity_weight if formal_effective else None
            ),
            "legacy_clean_identity": (
                None if formal_effective else reference_identity_weight
            ),
            "feature_margin": float(args.feature_margin_weight),
            "leo_reference_feature_margin": (
                reference_margin_weight if formal_effective else None
            ),
            "legacy_clean_feature_margin": (
                None if formal_effective else reference_margin_weight
            ),
            "proxy_unknown_separation": float(args.proxy_unknown_separation_weight),
            "proxy_unknown_max_cos": float(args.proxy_unknown_max_cos),
            "proxy_unknown_supcon": float(args.proxy_unknown_supcon_weight),
            "proxy_unknown_supcon_temperature": float(args.proxy_unknown_supcon_temperature),
            "proxy_unknown_proto_ce": float(args.proxy_unknown_proto_ce_weight),
            "proxy_unknown_proto_temperature": float(args.proxy_unknown_proto_temperature),
            "proxy_unknown_pair_margin": float(args.proxy_unknown_pair_margin),
            "proxy_unknown_pair_margin_weight": float(args.proxy_unknown_pair_margin_weight),
            "proxy_unknown_old_margin": float(args.proxy_unknown_old_margin),
            "proxy_unknown_old_margin_weight": float(args.proxy_unknown_old_margin_weight),
            "proxy_unknown_hard_pair_ids": str(args.proxy_unknown_hard_pair_ids or ""),
            "proxy_unknown_hard_pair_margin": float(args.proxy_unknown_hard_pair_margin),
            "proxy_unknown_hard_pair_margin_weight": float(args.proxy_unknown_hard_pair_margin_weight),
            "proxy_unknown_hard_old_margin": float(args.proxy_unknown_hard_old_margin),
            "proxy_unknown_hard_old_margin_weight": float(args.proxy_unknown_hard_old_margin_weight),
            "teacher_logit_distill": float(args.teacher_logit_distill_weight),
            "multiview_consistency": float(args.multiview_consistency_weight),
            "relation_preservation": float(args.relation_preservation_weight),
            "prototype_gram": float(args.prototype_gram_weight),
            "prototype_gram_max_cosine": float(args.prototype_gram_max_cosine),
            "worst_k_risk": float(args.worst_k_risk_weight),
            "worst_k_values": list(nested_k_values),
            "worst_k_tau": float(args.worst_k_tau),
            "worst_k_proto_temperature": float(args.worst_k_proto_temperature),
            "residual": float(args.residual_weight),
            "class_loss_weights": str(args.class_loss_weights or ""),
        },
    }


def _write_ground_lora_state_and_manifest(
    args: argparse.Namespace,
    model: nn.Module,
    train_info: dict[str, Any],
) -> dict[str, Any]:
    """Persist a source-only LoRA state with fail-closed provenance metadata."""

    model_adapter_mode = str(args.model_adapter_mode).strip().lower()
    scope_by_mode = {
        "lora_full_feature": "full_feature",
        "lora_effective_feature": "effective_feature",
    }
    if model_adapter_mode not in scope_by_mode:
        return {}
    state_path_raw = str(args.adapter_state_out or "").strip()
    manifest_path_raw = str(args.adapter_manifest_out or "").strip()
    if not state_path_raw or not manifest_path_raw:
        raise ValueError(
            f"{model_adapter_mode} requires --adapter_state_out and --adapter_manifest_out"
        )
    state_path = Path(state_path_raw)
    manifest_path = Path(manifest_path_raw)
    for path in (state_path, manifest_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite ground LoRA artifact: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

    trainable = {
        name: parameter.detach().cpu().to(dtype=torch.float16).contiguous()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if not trainable or any(".lora_" not in name for name in trainable):
        raise RuntimeError(
            "ground LoRA state must contain only injected lora_a/lora_b tensors"
        )
    if any(not bool(torch.isfinite(value).all()) for value in trainable.values()):
        raise FloatingPointError("ground LoRA state contains non-finite tensors")
    torch.save(trainable, state_path)

    resources = dict(train_info["model_adapter"])
    parameter_count = int(sum(value.numel() for value in trainable.values()))
    tensor_bytes = int(parameter_count * 2)
    if parameter_count != int(resources.get("trainable_parameters", -1)):
        raise RuntimeError(
            "ground LoRA parameter count drift: "
            f"{parameter_count}!={resources.get('trainable_parameters')}"
        )
    max_class_count = 26
    # The effective route deploys one shared symmetric head for all query
    # views.  Historical full_feature runs retain their five-bank accounting.
    prototype_bytes = int(
        15_688
        if model_adapter_mode == "lora_effective_feature"
        else 5 * max_class_count * 256 * 2
    )
    threshold_bytes = 12
    combined_bytes = int(tensor_bytes + prototype_bytes + threshold_bytes)
    state_cap = int(256 * 1024)
    resources.update(
        {
            "adapter_state_bytes_fp16": tensor_bytes,
            "max_registered_class_count": max_class_count,
            "deployment_head_state_upper_bound_bytes_fp16": prototype_bytes,
            "adaptive_threshold_state_bytes_fp32": threshold_bytes,
            "combined_persistent_state_bytes": combined_bytes,
            "combined_persistent_state_cap_bytes": state_cap,
            "combined_persistent_state_within_cap": combined_bytes <= state_cap,
            "mergeable_into_base_linear_weights": True,
            "deployment_added_macs_per_query_after_merge": 0,
        }
    )
    if not bool(resources["combined_persistent_state_within_cap"]):
        raise ValueError(
            "ground LoRA plus deployment head exceeds the 256KiB state cap"
        )
    state_sha256 = _sha256_file(state_path)
    source_cache_info = dict(getattr(args, "_source_leo_weak_cache_info", {}) or {})
    source_dataset_path: Path | None = None
    if model_adapter_mode == "lora_effective_feature":
        if (
            source_cache_info.get("phase2_sample_view_policy")
            != PHASE2_SAMPLE_VIEW_POLICY
            or source_cache_info.get("clean_sample_access") is not False
            or source_cache_info.get("clean_derived_signal_access") is not False
            or not source_cache_info.get("cache_set_manifest_sha256")
        ):
            raise ValueError("formal ground LoRA lacks verified source LEO_weak cache proof")
    else:
        source_dataset_path = Path(str(args.wisig_pkl)).resolve()
        if not source_dataset_path.is_file():
            raise FileNotFoundError(f"source dataset is missing: {source_dataset_path}")
    proxy_info = dict(train_info.get("proxy_unknown_train") or {})
    proxy_weight_sum = float(
        args.proxy_unknown_separation_weight
        + args.proxy_unknown_supcon_weight
        + args.proxy_unknown_proto_ce_weight
        + args.proxy_unknown_pair_margin_weight
        + args.proxy_unknown_old_margin_weight
        + args.proxy_unknown_hard_pair_margin_weight
        + args.proxy_unknown_hard_old_margin_weight
    )
    manifest = {
        "method": (
            "ground_source_effective_feature_lora_v1"
            if model_adapter_mode == "lora_effective_feature"
            else "ground_source_full_feature_lora_v1"
        ),
        "stage": (
            "Phase1_offline_ground_adapter_training"
            if model_adapter_mode == "lora_effective_feature"
            else "Stage2-C_prequery_source_training_diagnostic"
        ),
        "training_input_stage": (
            "phase1_offline_prechannel_export"
            if model_adapter_mode == "lora_effective_feature"
            else "legacy_runtime_dataset"
        ),
        "phase2_sample_view_policy": (
            PHASE2_SAMPLE_VIEW_POLICY
            if model_adapter_mode == "lora_effective_feature"
            else None
        ),
        "clean_sample_access": (
            False if model_adapter_mode == "lora_effective_feature" else None
        ),
        "clean_derived_signal_access": (
            False if model_adapter_mode == "lora_effective_feature" else None
        ),
        "source_only": True,
        "support_only": False,
        "query_update_forbidden": True,
        "query_features_used_for_training": False,
        "query_labels_used_for_training": False,
        "old_new_role_used_by_optimizer": False,
        "class_quota_used_at_inference": False,
        "target_receiver_data_used_for_training": False,
        "clean_samples_used_for_training": (
            model_adapter_mode != "lora_effective_feature"
        ),
        "formal_training_view": (
            "leo_weak_only"
            if model_adapter_mode == "lora_effective_feature"
            else "historical_clean_paired_diagnostic"
        ),
        "teacher_reference_view": (
            "same_leo_weak_base_view"
            if model_adapter_mode == "lora_effective_feature"
            else "clean_source_view"
        ),
        "proxy_data_used_for_training": bool(proxy_info),
        "proxy_training_rows": 0 if not proxy_info else int(proxy_info.get("sample_count", -1)),
        "proxy_loss_weight_sum": proxy_weight_sum,
        "source_receiver_scope": str(args.source_rxs),
        "source_tx_scope": str(args.source_tx_ids),
        "source_dataset": (
            None if source_dataset_path is None else str(source_dataset_path)
        ),
        "source_dataset_sha256": (
            None if source_dataset_path is None else _sha256_file(source_dataset_path)
        ),
        "source_dataset_size_bytes": (
            None if source_dataset_path is None else int(source_dataset_path.stat().st_size)
        ),
        "source_leo_weak_cache_set_manifest": source_cache_info.get(
            "cache_set_manifest"
        ),
        "source_leo_weak_cache_set_manifest_sha256": source_cache_info.get(
            "cache_set_manifest_sha256"
        ),
        "source_leo_weak_cache_set_audit": source_cache_info.get(
            "cache_set_audit"
        ),
        "proxy_receiver_scope": str(args.proxy_unknown_rxs),
        "proxy_tx_scope": str(args.proxy_unknown_tx_ids),
        "epochs": int(args.epochs),
        "resource_tier": (
            "preferred" if parameter_count <= 50_000 else "performance_relaxed"
        ),
        "adapter_state_format": "fp16_trainable_state",
        "adapter_state": str(state_path),
        "adapter_state_sha256": state_sha256,
        "checkpoint": str(args.ckpt),
        "checkpoint_sha256": _sha256_file(args.ckpt),
        "hyperparameters": {
            "scope": scope_by_mode[model_adapter_mode],
            "rank": int(args.lora_rank),
            "alpha": float(args.lora_alpha),
            "learning_rate": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "batch_size": int(args.batch_size),
            "sat_scenarios": parse_sat_scenarios(str(args.sat_scenarios)),
            "ground_receive_view_policy": "rx_light5_pair_cycle",
            "ground_receive_views_per_source_sample_per_step": 2,
            "ground_unique_receive_view_count": 5,
            "target_support_receive_view_count": 3,
            "target_support_receive_view_policy": "three_leo_weak_scenario_base_views",
            "multiview_consistency_weight": float(
                args.multiview_consistency_weight
            ),
        },
        "resources": resources,
        "training_trace_summary": {
            "history_first": train_info.get("history_first", {}),
            "history_last": train_info.get("history_last", {}),
            "loss_weights": train_info.get("loss_weights", {}),
        },
        "source_validation_required_before_target_query": True,
        "source_validation_pass": False,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "adapter_state": str(state_path),
        "adapter_state_sha256": state_sha256,
        "adapter_manifest": str(manifest_path),
        "resources": resources,
    }


def build_frozen_backbone_export_adapter(
    args: argparse.Namespace, device: torch.device
) -> tuple[nn.Module, dict[str, Any]]:
    """Return an identity IQ adapter while keeping ADV3B02 fully frozen.

    The exported z_id/FFT features are adapted later by the qKNN enrollment
    transform.  This path deliberately performs no gradient update on the
    checkpoint and therefore cannot be mistaken for another ADV3B02 training
    run.
    """
    if bool(args.input_adapter_enabled):
        raise ValueError("skip_adapter_training requires --no-input_adapter_enabled")
    if str(args.model_adapter_mode).strip().lower() != "none":
        raise ValueError("skip_adapter_training requires --model_adapter_mode none")
    scenarios = parse_sat_scenarios(str(args.sat_scenarios))
    _validate_star_ground_impl(
        str(args.star_ground_channel_impl), scenarios, field="sat_scenarios"
    )
    return IdentityPreAdapter().to(device), {
        "epochs": 0,
        "history_first": {},
        "history_last": {},
        "scenarios": scenarios,
        "scenario_configs": {
            name: sat_channel_config_for_scenario(name) for name in scenarios
        },
        "input_repair": str(args.input_repair),
        "input_adapter_enabled": False,
        "clean_input_repair_mode": str(args.clean_input_repair_mode),
        "model_adapter": {
            "mode": "none",
            "trainable_parameters": 0,
            "trainable_tensors": [],
        },
        "skip_adapter_training": True,
        "adv3b02_gradient_updates": 0,
        "downstream_feature_adapter": "qknnv42_support_diag_whiten_fisher",
        "uses_target_labels_for_training": False,
        "uses_unknown_query_for_threshold": False,
    }


def _dataset_for_role(args: argparse.Namespace, *, role: str, pkl: str, tx_ids: str, rxs: str | None, seed_offset: int):
    max_samples_per_tx = 0 if str(getattr(args, "export_reference_npz", "")).strip() else int(args.max_export_samples_per_tx)
    ds, info = _build_wisig_dataset(
        pkl_path=str(pkl),
        tx_spec=str(tx_ids),
        role=role,
        equalized=str(args.wisig_equalized),
        out_len=int(args.wisig_out_len),
        domain=str(args.wisig_domain),
        days=None,
        rxs=rxs,
        max_samples_per_combo=int(args.max_samples_per_combo),
        max_samples_per_tx=int(max_samples_per_tx),
        seed=int(args.seed) + int(seed_offset),
        dataset_cache=getattr(args, "_dataset_payload_cache", None),
    )
    return ds, info


def _reference_keys_by_role(reference_npz: str | Path) -> dict[str, list[tuple[str, str, str, str, str]]]:
    if not str(reference_npz).strip():
        return {}
    data = np.load(Path(reference_npz), allow_pickle=True)
    required = ("tx_ids", "rx_ids", "day_ids", "eq_ids", "sig_ids", "dataset_role")
    missing = [key for key in required if key not in data.files]
    if missing:
        raise ValueError(f"export_reference_npz missing keys: {missing}")
    arrays = {key: np.asarray(data[key], dtype=object).astype(str) for key in required}
    out: dict[str, list[tuple[str, str, str, str, str]]] = {}
    for index in range(int(arrays["tx_ids"].shape[0])):
        role = str(arrays["dataset_role"][index])
        key = (
            str(arrays["tx_ids"][index]),
            str(arrays["rx_ids"][index]),
            str(arrays["day_ids"][index]),
            str(arrays["eq_ids"][index]),
            str(arrays["sig_ids"][index]),
        )
        out.setdefault(role, []).append(key)
    return out


def _filter_dataset_to_reference(ds, *, role: str, reference_keys: dict[str, list[tuple[str, str, str, str, str]]]):
    desired = reference_keys.get(str(role))
    if not desired:
        return ds
    tx_labels = [canonical_tx_id(value) for value in ds.tx_list]
    rx_labels = [str(value) for value in ds.rx_list]
    day_labels = [str(value) for value in ds.day_list]
    eq_labels = [str(value) for value in ds.eq_list]
    index_by_key: dict[tuple[str, str, str, str, str], int] = {}
    for dataset_index, item in enumerate(ds.index):
        key = (
            tx_labels[int(item.tx_i)],
            rx_labels[int(item.rx_i)],
            day_labels[int(item.day_i)],
            eq_labels[int(item.eq_i)],
            str(int(item.sig_i)),
        )
        index_by_key[key] = int(dataset_index)
    missing = [key for key in desired if key not in index_by_key]
    if missing:
        preview = ",".join(":".join(key) for key in missing[:5])
        raise ValueError(f"reference samples missing for role={role}: count={len(missing)} preview={preview}")
    selected = [index_by_key[key] for key in desired]
    return WiSigSubsetDataset(ds, selected, split_source=f"{role}_reference_npz")


def _filter_optional_dataset_to_reference(
    ds, *, role: str, reference_keys: dict[str, list[tuple[str, str, str, str, str]]]
):
    if ds is None:
        return None
    return _filter_dataset_to_reference(ds, role=role, reference_keys=reference_keys)


@torch.no_grad()
def _export_role(
    model: nn.Module,
    adapter: nn.Module,
    loader: DataLoader,
    *,
    args: argparse.Namespace,
    device: torch.device,
    role: str,
    scenarios: Sequence[str],
    seed: int,
    channel_mode: str,
    use_adapter: bool,
) -> dict[str, np.ndarray]:
    gen = make_torch_generator(device, int(seed))
    feature_buf: list[np.ndarray] = []
    aux_fft_buf: list[np.ndarray] = []
    logit_buf: list[np.ndarray] = []
    labels: list[int] = []
    domains: list[int] = []
    txs: list[str] = []
    rxs: list[str] = []
    days: list[str] = []
    eqs: list[str] = []
    sigs: list[str] = []
    roles: list[str] = []
    views: list[str] = []
    scenario_buf: list[str] = []
    mode = str(channel_mode or "satellite").strip().lower()
    for bi, batch in enumerate(loader):
        x, y, d, meta = batch
        x = x.to(device, non_blocking=True)
        if mode not in {"satellite", "clean"}:
            raise ValueError(f"unknown channel_mode={channel_mode!r}")
        if mode == "satellite" and bool(getattr(args, "export_all_scenarios_per_sample", False)):
            batch_scenarios = [str(value) for value in scenarios]
        elif mode == "satellite":
            batch_scenarios = [_scenario_for_step(scenarios, bi)]
        else:
            batch_scenarios = [""]
        n = int(x.shape[0])
        meta_tx = _meta_to_list(meta, "tx", n)
        meta_rx = _meta_to_list(meta, "rx", n)
        meta_day = _meta_to_list(meta, "day", n)
        meta_eq = _meta_to_list(meta, "equalized", n)
        meta_sig = _meta_to_list(meta, "sig_i", n)
        for scenario in batch_scenarios:
            x_eval = x
            if mode == "satellite":
                x_eval, _ = apply_sat_channel_for_scenario(x, scenario, args, gen=gen, return_meta=False)
            tta_policy = str(getattr(args, "satellite_tta_policy", "none")) if mode == "satellite" else "none"
            tta_views = _satellite_tta_views(x_eval, tta_policy)
            z_views: list[torch.Tensor] = []
            logit_views: list[torch.Tensor] = []
            fft_views: list[np.ndarray] = []
            for _tta_name, x_view in tta_views:
                if int(getattr(args, "aux_fft_logmag_dim", 0)) > 0:
                    fft_views.append(
                        _spectral_logmag_sketch_batch(
                            x_view.detach().cpu().float().numpy(),
                            dim=int(args.aux_fft_logmag_dim),
                        )
                    )
                x_forward = x_view
                if use_adapter:
                    repair_mode = str(args.input_repair)
                    if mode == "clean" and str(args.clean_input_repair_mode).lower() != "same":
                        repair_mode = "raw"
                    x_forward = adapter(_apply_input_repair(x_forward, repair_mode))
                z_view, logits_view = _feature_forward(model, x_forward, str(args.feature_name))
                z_views.append(z_view.float())
                logit_views.append(logits_view.float())
            z = z_views[0] if len(z_views) == 1 else torch.stack(z_views, dim=0).mean(dim=0)
            logits = logit_views[0] if len(logit_views) == 1 else torch.stack(logit_views, dim=0).mean(dim=0)
            feature_buf.append(z.detach().cpu().float().numpy())
            logit_buf.append(logits.detach().cpu().float().numpy())
            if fft_views:
                fft = fft_views[0] if len(fft_views) == 1 else np.mean(np.stack(fft_views, axis=0), axis=0)
                fft -= fft.mean(axis=1, keepdims=True)
                fft /= np.maximum(np.linalg.norm(fft, axis=1, keepdims=True), 1.0e-8)
                aux_fft_buf.append(fft.astype(np.float32))
            labels.extend([int(v) for v in y.detach().cpu().reshape(-1).tolist()])
            domains.extend([int(v) for v in d.detach().cpu().reshape(-1).tolist()])
            txs.extend(meta_tx)
            rxs.extend(meta_rx)
            days.extend(meta_day)
            eqs.extend(meta_eq)
            sigs.extend(meta_sig)
            roles.extend([role] * n)
            adapted_view = "model_feature_adapter" if str(args.model_adapter_mode).lower() != "none" else "iq_frontend"
            view_name = adapted_view if use_adapter else ("identity_satellite" if mode == "satellite" else "clean")
            if mode == "satellite" and str(tta_policy).strip().lower() not in {"", "none", "off", "0"}:
                view_name = f"{view_name}|tta_mean={tta_policy}"
            views.extend([view_name] * n)
            scenario_buf.extend([scenario] * n)
    payload = {
        "features": np.concatenate(feature_buf, axis=0).astype(np.float32),
        "tx_logits": np.concatenate(logit_buf, axis=0).astype(np.float32),
        "raw_labels": np.asarray(labels, dtype=np.int64),
        "domain_labels": np.asarray(domains, dtype=np.int64),
        "tx_ids": np.asarray(txs),
        "rx_ids": np.asarray(rxs),
        "day_ids": np.asarray(days),
        "eq_ids": np.asarray(eqs),
        "sig_ids": np.asarray(sigs),
        "dataset_role": np.asarray(roles),
        "channel_views": np.asarray(views),
        "sat_scenarios": np.asarray(scenario_buf),
    }
    if int(getattr(args, "aux_fft_logmag_dim", 0)) > 0:
        payload["fft_logmag_features"] = np.concatenate(aux_fft_buf, axis=0).astype(np.float32)
    return payload


def _concat(parts: Sequence[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {k: np.concatenate([p[k] for p in parts], axis=0) for k in parts[0].keys()}


def _parse_export_cell(cell: str) -> dict[str, str]:
    parts = [part.strip() for part in str(cell).split(":")]
    if len(parts) == 3:
        name, target_rx, target_unknown_tx = parts
        target_new_tx = ""
    elif len(parts) == 4:
        name, target_rx, target_new_tx, target_unknown_tx = parts
    else:
        raise ValueError(
            "cell must be name:target_rx:target_unknown_tx_ids or "
            "name:target_rx:target_new_tx_ids:target_unknown_tx_ids"
        )
    if not name or not target_rx or not target_unknown_tx:
        raise ValueError(f"cell has empty required field: {cell!r}")
    new_ids = set(parse_tx_id_list(target_new_tx))
    unknown_ids = set(parse_tx_id_list(target_unknown_tx))
    overlap = sorted(new_ids & unknown_ids)
    if overlap:
        raise ValueError(f"target_new and target_unknown TX IDs must be disjoint: {overlap}")
    return {
        "name": name,
        "target_rx": target_rx,
        "target_new_tx": target_new_tx,
        "target_unknown_tx": target_unknown_tx,
    }


def _resolve_export_tta_policies(args: argparse.Namespace) -> list[str]:
    raw = str(getattr(args, "export_tta_policies", "") or "").strip()
    policies = [str(args.satellite_tta_policy)] if not raw else [item.strip().lower() for item in raw.split(",")]
    if any(not item for item in policies):
        raise ValueError("export_tta_policies contains an empty policy")
    unknown = [item for item in policies if item not in SATELLITE_TTA_POLICIES]
    if unknown:
        raise ValueError(f"unknown export TTA policies: {unknown}")
    if len(set(policies)) != len(policies):
        raise ValueError("export_tta_policies must not contain duplicates")
    return policies


def _tta_export_subdir(base: str, policy: str, template: str) -> str:
    rendered = str(template).format(
        base=str(base), policy=str(policy), view_count=_satellite_tta_view_count(str(policy))
    ).strip()
    if not rendered or Path(rendered).is_absolute() or ".." in Path(rendered).parts:
        raise ValueError(f"unsafe export TTA subdirectory: {rendered!r}")
    return rendered


def _resolve_tta_export_subdirs(args: argparse.Namespace, policies: Sequence[str]) -> list[str]:
    base = str(args.out_subdir)
    if not str(getattr(args, "export_tta_policies", "") or "").strip():
        return [base]
    subdirs = [
        _tta_export_subdir(base, policy, str(args.export_tta_subdir_template))
        for policy in policies
    ]
    if len(set(subdirs)) != len(subdirs):
        raise ValueError("export_tta_subdir_template must render a unique directory per policy")
    return subdirs


def export_cell(
    args: argparse.Namespace,
    model: nn.Module,
    adapter: nn.Module,
    identity_model: nn.Module,
    device: torch.device,
    cell: str,
    train_info: dict[str, Any],
) -> Path:
    cell_spec = _parse_export_cell(cell)
    name = cell_spec["name"]
    out_dir = Path(args.runs_root) / name / str(args.out_subdir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty export directory: {out_dir}")
    identity_out_dir = Path(args.runs_root) / name / str(args.identity_subdir)
    if bool(args.export_identity):
        if identity_out_dir == out_dir:
            raise ValueError("identity_subdir must differ from out_subdir")
        if identity_out_dir.exists() and any(identity_out_dir.iterdir()):
            raise FileExistsError(
                f"refusing to overwrite non-empty identity export directory: {identity_out_dir}"
            )
    target_rx = cell_spec["target_rx"]
    target_new_tx = cell_spec["target_new_tx"]
    unknown_tx = cell_spec["target_unknown_tx"]
    scenarios = parse_sat_scenarios(str(args.sat_scenarios))
    reference_keys = _reference_keys_by_role(args.export_reference_npz)
    role_scope = str(args.export_role_scope).strip().lower()
    registered_only = role_scope == "qknn_registered_only"
    source_ds = None
    source_info = None
    proxy_ds = None
    proxy_info = None
    if not registered_only:
        source_ds, source_info = _dataset_for_role(args, role="source", pkl=str(args.wisig_pkl), tx_ids=str(args.source_tx_ids), rxs=str(args.source_rxs), seed_offset=101)
        proxy_ds, proxy_info = _dataset_for_role(args, role="proxy_unknown", pkl=str(args.new_wisig_pkl), tx_ids=str(args.proxy_unknown_tx_ids), rxs=str(args.proxy_unknown_rxs), seed_offset=211)
    target_old_ds, target_old_info = _dataset_for_role(args, role="target_old", pkl=str(args.wisig_pkl), tx_ids=str(args.target_old_tx_ids), rxs=target_rx, seed_offset=307)
    target_new_ds = None
    target_new_info = None
    if parse_tx_id_list(target_new_tx):
        target_new_ds, target_new_info = _dataset_for_role(args, role="target_new", pkl=str(args.new_wisig_pkl), tx_ids=target_new_tx, rxs=target_rx, seed_offset=353)
    if registered_only and target_new_ds is None:
        raise ValueError("qknn_registered_only requires explicit target_new TX IDs")
    unknown_ds = None
    unknown_info = None
    if not registered_only:
        unknown_ds, unknown_info = _dataset_for_role(args, role="target_unknown", pkl=str(args.new_wisig_pkl), tx_ids=unknown_tx, rxs=target_rx, seed_offset=409)
    if source_ds is not None:
        source_ds = _filter_dataset_to_reference(source_ds, role="source", reference_keys=reference_keys)
    if proxy_ds is not None:
        proxy_ds = _filter_dataset_to_reference(proxy_ds, role="proxy_unknown", reference_keys=reference_keys)
    target_old_ds = _filter_dataset_to_reference(target_old_ds, role="target_old", reference_keys=reference_keys)
    if target_new_ds is not None:
        target_new_ds = _filter_dataset_to_reference(target_new_ds, role="target_new", reference_keys=reference_keys)
    unknown_ds = _filter_optional_dataset_to_reference(
        unknown_ds, role="target_unknown", reference_keys=reference_keys
    )
    for info, ds in (
        (source_info, source_ds),
        (proxy_info, proxy_ds),
        (target_old_info, target_old_ds),
        (target_new_info, target_new_ds),
        (unknown_info, unknown_ds),
    ):
        if info is None or ds is None:
            continue
        if reference_keys:
            info["reference_filtered"] = True
            info["size"] = int(len(ds))
    role_items = [("target_old", target_old_ds, 2021)]
    if not registered_only:
        role_items[0:0] = [
            ("source", source_ds, 2001),
            ("proxy_unknown", proxy_ds, 2011),
        ]
    if target_new_ds is not None:
        role_items.append(("target_new", target_new_ds, 2027))
    if not registered_only:
        role_items.append(("target_unknown", unknown_ds, 2031))
    parts = []
    clean_parts = []
    identity_parts = []
    identity_clean_parts = []
    for role, ds, offset in role_items:
        loader = DataLoader(ds, batch_size=int(args.batch_size), shuffle=False, num_workers=0, drop_last=False)
        parts.append(_export_role(model, adapter, loader, args=args, device=device, role=role, scenarios=scenarios, seed=int(args.seed) + offset, channel_mode="satellite", use_adapter=True))
        if bool(args.export_clean_control):
            clean_parts.append(_export_role(model, adapter, loader, args=args, device=device, role=role, scenarios=scenarios, seed=int(args.seed) + offset + 7000, channel_mode="clean", use_adapter=True))
        if bool(args.export_identity):
            identity_parts.append(_export_role(identity_model, adapter, loader, args=args, device=device, role=role, scenarios=scenarios, seed=int(args.seed) + offset, channel_mode="satellite", use_adapter=False))
            if bool(args.export_clean_control):
                identity_clean_parts.append(_export_role(identity_model, adapter, loader, args=args, device=device, role=role, scenarios=scenarios, seed=int(args.seed) + offset + 7000, channel_mode="clean", use_adapter=False))
    payload = _concat(parts)
    frozen_qknn_export = bool(args.skip_adapter_training)
    manifest = {
        "payload_source": (
            "qknnv42_frozen_adv3b02_identity_only_features_v1"
            if frozen_qknn_export
            else (
                "phase1_model_feature_adapter_satonly_features_v29"
                if str(args.model_adapter_mode).lower() != "none"
                else "phase1_iq_frontend_satonly_features_v28"
            )
        ),
        "feature_name": str(args.feature_name),
        "identity_only_forward": can_use_identity_only_forward(model, str(args.feature_name)),
        "domain_branch_executed_for_qknn": not can_use_identity_only_forward(
            model, str(args.feature_name)
        ),
        "checkpoint": str(args.ckpt),
        "source_checkpoint_sha256": _sha256_file(args.ckpt),
        "checkpoint_load_strict": True,
        "checkpoint_load_audit": dict(
            getattr(model, "_checkpoint_load_audit", {})
        ),
        "target_channel_view": "satellite/LEO",
        "export_role_scope": role_scope,
        "omitted_unused_qknn_roles": (
            ["source", "proxy_unknown", "target_unknown"] if registered_only else []
        ),
        "channel_views": [
            "frozen_adv3b02_identity_only_z_id"
            if frozen_qknn_export
            else (
                "model_feature_adapter"
                if str(args.model_adapter_mode).lower() != "none"
                else "iq_frontend"
            )
        ],
        "satellite_tta_policy": str(args.satellite_tta_policy),
        "satellite_tta_view_count": _satellite_tta_view_count(str(args.satellite_tta_policy)),
        "satellite_tta_aggregation": "feature_logit_mean_per_physical_sample",
        "export_all_scenarios_per_sample": bool(args.export_all_scenarios_per_sample),
        "aux_fft_logmag_dim": int(args.aux_fft_logmag_dim),
        "aux_fft_feature_key": "fft_logmag_features" if int(args.aux_fft_logmag_dim) > 0 else "",
        "aux_fft_view_alignment": "same_post_channel_view_as_backbone",
        "star_ground_channel_impl": str(args.star_ground_channel_impl),
        "sat_scenarios": scenarios,
        "source": source_info,
        "proxy_unknown": proxy_info,
        "target_old": target_old_info,
        "target_new": target_new_info,
        "target_unknown": unknown_info,
        "target_new_tx_ids": parse_tx_id_list(target_new_tx),
        "target_unknown_tx_ids": parse_tx_id_list(unknown_tx),
        "export_reference_npz": str(args.export_reference_npz or ""),
        "uses_target_clean": False,
        "uses_target_labels_for_training": False,
        "uses_unknown_query_for_threshold": False,
        "adapter": train_info,
    }
    payload["manifest_json"] = np.asarray(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / str(args.out_name)
    np.savez(out_path, **payload)
    if clean_parts:
        clean_payload = _concat(clean_parts)
        clean_manifest = dict(manifest)
        clean_manifest["target_channel_view"] = "clean"
        clean_manifest["channel_views"] = ["iq_frontend"]
        clean_payload["manifest_json"] = np.asarray(json.dumps(clean_manifest, ensure_ascii=False, sort_keys=True))
        np.savez(out_dir / str(args.clean_out_name), **clean_payload)
    if identity_parts:
        identity_payload = _concat(identity_parts)
        identity_manifest = dict(manifest)
        identity_manifest["adapter"] = {"identity_baseline": True, "scenarios": scenarios}
        identity_manifest["channel_views"] = ["identity_satellite"]
        identity_out_dir.mkdir(parents=True, exist_ok=True)
        identity_payload["manifest_json"] = np.asarray(json.dumps(identity_manifest, ensure_ascii=False, sort_keys=True))
        np.savez(identity_out_dir / str(args.out_name), **identity_payload)
        if identity_clean_parts:
            identity_clean_payload = _concat(identity_clean_parts)
            identity_clean_manifest = dict(identity_manifest)
            identity_clean_manifest["target_channel_view"] = "clean"
            identity_clean_manifest["channel_views"] = ["clean"]
            identity_clean_payload["manifest_json"] = np.asarray(json.dumps(identity_clean_manifest, ensure_ascii=False, sort_keys=True))
            np.savez(identity_out_dir / str(args.clean_out_name), **identity_clean_payload)
    return out_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--wisig_pkl", default="./Dataset_WigSig/ManySig.pkl")
    p.add_argument("--new_wisig_pkl", default="./Dataset_WigSig/ManyTx.pkl")
    p.add_argument("--runs_root", type=Path, required=True)
    p.add_argument("--out_subdir", default="ADV3B02_CORE90_SOFT_E200_PHASE1_IQPRE_V11")
    p.add_argument("--out_name", default="features_iqpre_v11.npz")
    p.add_argument("--clean_out_name", default="features_clean_repaired.npz")
    p.add_argument("--identity_subdir", default="LEOIQ28_IDENTITY")
    p.add_argument(
        "--export_role_scope",
        choices=("all", "qknn_registered_only"),
        default="all",
        help="qknn_registered_only exports only target_old/target_new rows consumed by Stage2-C qKNN",
    )
    p.add_argument("--export_clean_control", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--export_identity", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument(
        "--cells",
        default="",
        help=(
            "semicolon-separated name:target_rx:target_unknown_tx_ids legacy cells, "
            "or name:target_rx:target_new_tx_ids:target_unknown_tx_ids Stage2-C cells"
        ),
    )
    p.add_argument("--feature_name", default="z_id")
    p.add_argument("--aux_fft_logmag_dim", type=int, default=0)
    p.add_argument(
        "--export_all_scenarios_per_sample",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Export every physical sample under every requested LEO scenario.",
    )
    p.add_argument("--export_reference_npz", default="", help="optional feature NPZ whose tx/rx/day/eq/sig/role order defines export samples")
    p.add_argument("--dataset", default="wisig")
    p.add_argument("--num_classes", type=int, default=None)
    p.add_argument("--model_size", default=None)
    p.add_argument("--model_variant", default=None)
    p.add_argument("--branch_ablation", default=None)
    p.add_argument("--sample_rate_hz", type=float, default=None)
    p.add_argument("--source_tx_ids", default="0,1,2,3,4,5")
    p.add_argument("--target_old_tx_ids", default="0,1,2,3,4,5")
    p.add_argument("--source_rxs", default="0,1,2,3,4,5,6")
    p.add_argument("--source_days", default=None)
    p.add_argument(
        "--source_leo_weak_cache_set_manifest",
        type=Path,
        default=None,
        help=(
            "required sealed Phase1 source_train LEO_weak cache-set manifest for "
            "formal lora_effective_feature training"
        ),
    )
    p.add_argument("--proxy_unknown_tx_ids", default="9-1,8-3,8-18,8-13,8-1,7-11,7-10,6-6,6-1,5-5,4-11,4-1,3-8,3-18,3-13,20-8")
    p.add_argument("--proxy_unknown_rxs", default="1-1,1-19,14-7,18-2,19-2,2-1")
    p.add_argument("--wisig_equalized", default="1")
    p.add_argument("--wisig_domain", default="rx_day")
    p.add_argument("--wisig_out_len", type=int, default=256)
    p.add_argument("--max_samples_per_combo", type=int, default=0)
    p.add_argument("--max_source_samples_per_tx", type=int, default=1000)
    p.add_argument("--max_proxy_unknown_train_samples_per_tx", type=int, default=600)
    p.add_argument("--max_proxy_unknown_samples_per_combo", type=int, default=0)
    p.add_argument("--max_export_samples_per_tx", type=int, default=200)
    p.add_argument("--num_old_classes", type=int, default=6)
    p.add_argument("--sat_scenarios", default="leo_clear_weak,leo_low_elev_weak,leo_rain_weak")
    p.add_argument(
        "--satellite_tta_policy",
        default="none",
        choices=SATELLITE_TTA_POLICIES,
        help="receive-side LEO repair/TTA views averaged into one exported feature row per physical sample",
    )
    p.add_argument(
        "--export_tta_policies",
        default="",
        help="optional comma-separated policies exported after one adapter training; overrides satellite_tta_policy",
    )
    p.add_argument(
        "--export_tta_subdir_template",
        default="{base}_{policy}",
        help="subdirectory template used only when export_tta_policies is set",
    )
    p.add_argument("--star_ground_channel_impl", default="simplified_leo_residual", choices=["legacy_satellite", "simplified_leo_residual"])
    p.add_argument("--sat_fs_hz", type=float, default=25e6)
    p.add_argument("--sat_fc_hz", type=float, default=2.462e9)
    p.add_argument("--batch_size", type=int, default=384)
    p.add_argument("--epochs", type=int, default=45)
    p.add_argument(
        "--skip_adapter_training",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "export from the strictly loaded frozen ADV3B02 checkpoint without IQ/model "
            "adapter training; qKNN performs support-conditioned feature adaptation later"
        ),
    )
    p.add_argument("--hidden_dim", type=int, default=32)
    p.add_argument("--alpha", type=float, default=0.25)
    p.add_argument("--input_adapter_enabled", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--model_adapter_mode",
        default="none",
        choices=[
            "none",
            "id_feature_head",
            "id_late_feature",
            "id_norm_late_feature",
            "id_full_feature",
            "lora_full_feature",
            "lora_effective_feature",
        ],
    )
    p.add_argument("--lora_rank", type=int, default=12)
    p.add_argument("--lora_alpha", type=float, default=12.0)
    p.add_argument(
        "--adapter_state_out",
        default="",
        help="required FP16 state path for a ground-source LoRA mode",
    )
    p.add_argument(
        "--adapter_manifest_out",
        default="",
        help="required training manifest path for a ground-source LoRA mode",
    )
    p.add_argument(
        "--source_only_ground_lora",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "train and persist the formal effective8 ground LoRA, then stop "
            "before constructing any target support/query export dataset"
        ),
    )
    p.add_argument("--input_repair", default="raw", choices=["raw", "rms", "canonical", "canonical_m1e4", "canonical_p1e4"])
    p.add_argument("--clean_input_repair_mode", default="same", choices=["raw", "same"])
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--mse_weight", type=float, default=1.0)
    p.add_argument("--cos_weight", type=float, default=2.0)
    p.add_argument("--proto_ce_weight", type=float, default=0.0)
    p.add_argument("--logit_ce_weight", type=float, default=0.0)
    p.add_argument("--class_loss_weights", default="")
    p.add_argument("--clean_identity_weight", type=float, default=8.0)
    p.add_argument("--clean_cos_weight", type=float, default=1.0)
    p.add_argument(
        "--leo_reference_identity_weight",
        type=float,
        default=None,
        help="same sealed LEO_weak teacher-feature anchor weight for the formal route",
    )
    p.add_argument(
        "--leo_reference_cos_weight",
        type=float,
        default=None,
        help="cosine term inside the same-LEO reference anchor",
    )
    p.add_argument("--feature_margin_weight", type=float, default=2.0)
    p.add_argument("--clean_feature_margin_weight", type=float, default=2.0)
    p.add_argument(
        "--leo_reference_margin_weight",
        type=float,
        default=None,
        help="same sealed LEO_weak prototype-margin preservation weight",
    )
    p.add_argument("--feature_margin_tolerance", type=float, default=0.01)
    p.add_argument("--proxy_unknown_separation_weight", type=float, default=0.0)
    p.add_argument("--proxy_unknown_max_cos", type=float, default=0.18)
    p.add_argument("--proxy_unknown_supcon_weight", type=float, default=0.0)
    p.add_argument("--proxy_unknown_supcon_temperature", type=float, default=0.07)
    p.add_argument("--proxy_unknown_proto_ce_weight", type=float, default=0.0)
    p.add_argument("--proxy_unknown_proto_temperature", type=float, default=0.07)
    p.add_argument("--proxy_unknown_pair_margin_weight", type=float, default=0.0)
    p.add_argument("--proxy_unknown_pair_margin", type=float, default=0.04)
    p.add_argument("--proxy_unknown_old_margin_weight", type=float, default=0.0)
    p.add_argument("--proxy_unknown_old_margin", type=float, default=0.02)
    p.add_argument("--proxy_unknown_hard_pair_ids", default="")
    p.add_argument("--proxy_unknown_hard_pair_margin_weight", type=float, default=0.0)
    p.add_argument("--proxy_unknown_hard_pair_margin", type=float, default=0.08)
    p.add_argument("--proxy_unknown_hard_old_margin_weight", type=float, default=0.0)
    p.add_argument("--proxy_unknown_hard_old_margin", type=float, default=0.05)
    p.add_argument("--teacher_logit_distill_weight", type=float, default=0.0)
    p.add_argument(
        "--multiview_consistency_weight",
        type=float,
        default=0.0,
        help=(
            "used by ground LoRA modes to align the base view with one rotating "
            "rx_light5 view per source step"
        ),
    )
    p.add_argument("--relation_preservation_weight", type=float, default=0.0)
    p.add_argument("--prototype_gram_weight", type=float, default=0.0)
    p.add_argument("--prototype_gram_max_cosine", type=float, default=0.65)
    p.add_argument("--worst_k_risk_weight", type=float, default=0.0)
    p.add_argument("--worst_k_values", default="1,2,5,10,20")
    p.add_argument("--worst_k_tau", type=float, default=0.2)
    p.add_argument("--worst_k_proto_temperature", type=float, default=0.07)
    p.add_argument("--distill_temperature", type=float, default=2.0)
    p.add_argument("--residual_weight", type=float, default=0.03)
    p.add_argument("--proto_temperature", type=float, default=0.07)
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--log_every", type=int, default=5)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=4070391)
    return p.parse_args(argv)


def _validate_source_only_ground_lora_mode(args: argparse.Namespace) -> None:
    effective_ground_lora = (
        str(args.model_adapter_mode).strip().lower() == "lora_effective_feature"
    )
    if effective_ground_lora and not bool(args.source_only_ground_lora):
        raise ValueError(
            "formal lora_effective_feature requires --source_only_ground_lora; "
            "target support/query export is forbidden before source validation "
            "and candidate lock"
        )
    if effective_ground_lora:
        cache_path = getattr(args, "source_leo_weak_cache_set_manifest", None)
        if cache_path is None or not Path(cache_path).is_file():
            raise ValueError(
                "formal lora_effective_feature requires an existing "
                "--source_leo_weak_cache_set_manifest"
            )
        missing_reference_weights = [
            name
            for name in (
                "leo_reference_identity_weight",
                "leo_reference_cos_weight",
                "leo_reference_margin_weight",
            )
            if getattr(args, name, None) is None
        ]
        if missing_reference_weights:
            raise ValueError(
                "formal lora_effective_feature requires explicit same-LEO "
                f"reference weights: {missing_reference_weights}"
            )
    if bool(args.source_only_ground_lora) and not effective_ground_lora:
        raise ValueError(
            "--source_only_ground_lora is reserved for lora_effective_feature"
        )
    if not bool(args.source_only_ground_lora) and not str(args.cells).strip():
        raise ValueError("target export requires a nonempty --cells definition")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _validate_source_only_ground_lora_mode(args)
    policies = _resolve_export_tta_policies(args)
    export_subdirs = _resolve_tta_export_subdirs(args, policies)
    base_subdir = str(args.out_subdir)
    original_policy = str(args.satellite_tta_policy)
    device = torch.device(str(args.device) if torch.cuda.is_available() else "cpu")
    if bool(args.skip_adapter_training):
        args._dataset_payload_cache = {}
        source_loader = None
        model = _build_model(args, None, device, freeze=True)
        teacher_model = model
        adapter, train_info = build_frozen_backbone_export_adapter(args, device)
    else:
        source_loader, source_ds, _source_info = _make_source_loader(args)
        model = _build_model(args, source_ds, device, freeze=True)
        teacher_model = _build_model(args, source_ds, device, freeze=True)
        adapter, train_info = train_adapter(
            args, model, teacher_model, source_loader, device
        )
    ground_lora_artifacts = _write_ground_lora_state_and_manifest(
        args, model, train_info
    )
    if bool(args.source_only_ground_lora):
        summary = {
            "phase": "stage2c_effective8_ground_lora_source_only_v1",
            "exported": [],
            "target_dataset_constructed": False,
            "target_support_features_used": False,
            "target_query_features_used": False,
            "train_info": train_info,
            "ground_lora_artifacts": ground_lora_artifacts,
            "uses_clean": False,
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    exported = []
    for policy, export_subdir in zip(policies, export_subdirs):
        args.satellite_tta_policy = policy
        args.out_subdir = export_subdir
        for raw_cell in str(args.cells).split(";"):
            cell = raw_cell.strip()
            if not cell:
                continue
            exported.append(str(export_cell(args, model, adapter, teacher_model, device, cell, train_info)))
    args.out_subdir = base_subdir
    args.satellite_tta_policy = original_policy
    summary = {
        "phase": (
            "qknnv42_frozen_adv3b02_feature_export_v1"
            if bool(args.skip_adapter_training)
            else "phase1_iq_preadapter_v11"
        ),
        "exported": exported,
        "export_tta_policies": policies,
        "train_info": train_info,
        "ground_lora_artifacts": ground_lora_artifacts,
        "uses_target_clean": False,
        "uses_target_labels_for_training": False,
        "uses_unknown_query_for_threshold": False,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
