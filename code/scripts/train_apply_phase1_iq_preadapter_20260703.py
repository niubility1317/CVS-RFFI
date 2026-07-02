#!/usr/bin/env python
"""Train a source-only IQ pre-adapter before a frozen Phase1 backbone.

The adapter is trained with source old clean/LEO pairs only. Target receiver
old/unknown rows are exported for evaluation, but are not used in training or
threshold calibration.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

CODE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_ROOT.parent
for path in (str(REPO_ROOT), str(CODE_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from cvsrffi.eval import apply_sat_channel_for_scenario  # noqa: E402
from cvsrffi.tensors import make_torch_generator  # noqa: E402
from cvsrffi.wisig_fewshot_payload import canonical_tx_id, parse_tx_id_list  # noqa: E402
from eval_feature_diagnosis import (  # noqa: E402
    build_model_from_ckpt,
    collect_feature_dict,
    infer_num_domains,
    load_state_dict_safely,
    strip_module_prefix,
)
from export_spaceborne_features import (  # noqa: E402
    _build_wisig_dataset,
    _meta_to_list,
    _resolve_tx_indices,
    _validate_star_ground_impl,
)
from training_controls import parse_sat_scenarios, sat_channel_config_for_scenario  # noqa: E402


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


def _feature_forward(model: nn.Module, x: torch.Tensor, feature_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    out = model(x, y_tx=None, grl_lambda=1.0, return_aux=True)
    feats = collect_feature_dict(out)
    if feature_name not in feats:
        raise KeyError(f"feature {feature_name!r} not found; available={sorted(feats.keys())}")
    logits = out.get("tx_logits", out.get("logits")) if isinstance(out, dict) else None
    if logits is None:
        raise KeyError("model output does not include tx_logits/logits")
    return feats[feature_name].float(), logits.float()


def _build_model(args: argparse.Namespace, source_ds, device: torch.device) -> nn.Module:
    ckpt = torch.load(args.ckpt, map_location="cpu")
    if "args" not in ckpt or "model" not in ckpt:
        raise KeyError("checkpoint must contain 'args' and 'model'")
    state = strip_module_prefix(ckpt["model"])
    num_domains = infer_num_domains(source_ds, state=state, split_info={}, ckpt_args=ckpt["args"], cli_num_domains=None)
    model = build_model_from_ckpt(ckpt["args"], args, num_domains=num_domains, input_len=int(args.wisig_out_len), device=device)
    missing, unexpected, skipped = load_state_dict_safely(model, state)
    if missing or unexpected or skipped:
        print(json.dumps({"load_state": {"missing": missing, "unexpected": unexpected, "skipped": skipped}}, ensure_ascii=False))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _make_source_loader(args: argparse.Namespace):
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


def _scenario_for_step(scenarios: Sequence[str], step: int) -> str:
    if not scenarios:
        raise ValueError("at least one satellite scenario is required")
    return str(scenarios[int(step) % len(scenarios)])


def train_adapter(args: argparse.Namespace, model: nn.Module, source_loader: DataLoader, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    scenarios = parse_sat_scenarios(str(args.sat_scenarios))
    _validate_star_ground_impl(str(args.star_ground_channel_impl), scenarios, field="sat_scenarios")
    proto_loader = DataLoader(source_loader.dataset, batch_size=int(args.batch_size), shuffle=False, num_workers=0, drop_last=False)
    clean_protos = _proto_from_loader(model, proto_loader, args, device)
    adapter = IQResidualPreAdapter(hidden=int(args.hidden_dim), alpha=float(args.alpha)).to(device)
    opt = torch.optim.AdamW(adapter.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    gen = make_torch_generator(device, int(args.seed) + 1701)
    history: list[dict[str, float]] = []
    step = 0
    for epoch in range(int(args.epochs)):
        sums = {"loss": 0.0, "mse": 0.0, "cos": 0.0, "ce": 0.0, "resid": 0.0}
        count = 0
        for x, y, _d, _meta in source_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device).long()
            scenario = _scenario_for_step(scenarios, step)
            step += 1
            with torch.no_grad():
                x_sat, _ = apply_sat_channel_for_scenario(x, scenario, args, gen=gen, return_meta=False)
                z_clean, _ = _feature_forward(model, x, str(args.feature_name))
            x_rep = adapter(x_sat)
            z_rep, logits_rep = _feature_forward(model, x_rep, str(args.feature_name))
            proto_logits = F.normalize(z_rep, dim=1) @ F.normalize(clean_protos, dim=1).t() / max(float(args.proto_temperature), 1e-6)
            mse = F.smooth_l1_loss(z_rep, z_clean)
            cos = 1.0 - F.cosine_similarity(z_rep, z_clean, dim=1).mean()
            ce = F.cross_entropy(proto_logits, y) + float(args.logit_ce_weight) * F.cross_entropy(logits_rep, y)
            resid = (x_rep.float() - x_sat.float()).square().mean()
            loss = (
                float(args.mse_weight) * mse
                + float(args.cos_weight) * cos
                + float(args.proto_ce_weight) * ce
                + float(args.residual_weight) * resid
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if float(args.grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), float(args.grad_clip))
            opt.step()
            bs = int(x.shape[0])
            count += bs
            sums["loss"] += float(loss.detach().item()) * bs
            sums["mse"] += float(mse.detach().item()) * bs
            sums["cos"] += float(cos.detach().item()) * bs
            sums["ce"] += float(ce.detach().item()) * bs
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
    }


def _dataset_for_role(args: argparse.Namespace, *, role: str, pkl: str, tx_ids: str, rxs: str | None, seed_offset: int):
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
        max_samples_per_tx=int(args.max_export_samples_per_tx),
        seed=int(args.seed) + int(seed_offset),
    )
    return ds, info


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
) -> dict[str, np.ndarray]:
    gen = make_torch_generator(device, int(seed))
    feature_buf: list[np.ndarray] = []
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
    for bi, batch in enumerate(loader):
        x, y, d, meta = batch
        x = x.to(device, non_blocking=True)
        scenario = _scenario_for_step(scenarios, bi)
        x_sat, _ = apply_sat_channel_for_scenario(x, scenario, args, gen=gen, return_meta=False)
        x_rep = adapter(x_sat)
        z, logits = _feature_forward(model, x_rep, str(args.feature_name))
        n = int(x.shape[0])
        feature_buf.append(z.detach().cpu().float().numpy())
        logit_buf.append(logits.detach().cpu().float().numpy())
        labels.extend([int(v) for v in y.detach().cpu().reshape(-1).tolist()])
        domains.extend([int(v) for v in d.detach().cpu().reshape(-1).tolist()])
        txs.extend(_meta_to_list(meta, "tx", n))
        rxs.extend(_meta_to_list(meta, "rx", n))
        days.extend(_meta_to_list(meta, "day", n))
        eqs.extend(_meta_to_list(meta, "equalized", n))
        sigs.extend(_meta_to_list(meta, "sig_i", n))
        roles.extend([role] * n)
        views.extend(["iq_preadapter"] * n)
        scenario_buf.extend([scenario] * n)
    return {
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


def _concat(parts: Sequence[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {k: np.concatenate([p[k] for p in parts], axis=0) for k in parts[0].keys()}


def export_cell(args: argparse.Namespace, model: nn.Module, adapter: nn.Module, device: torch.device, cell: str, train_info: dict[str, Any]) -> Path:
    name, target_rx, unknown_tx = cell.split(":")
    scenarios = parse_sat_scenarios(str(args.sat_scenarios))
    source_ds, source_info = _dataset_for_role(args, role="source", pkl=str(args.wisig_pkl), tx_ids=str(args.source_tx_ids), rxs=str(args.source_rxs), seed_offset=101)
    proxy_ds, proxy_info = _dataset_for_role(args, role="proxy_unknown", pkl=str(args.new_wisig_pkl), tx_ids=str(args.proxy_unknown_tx_ids), rxs=str(args.proxy_unknown_rxs), seed_offset=211)
    target_old_ds, target_old_info = _dataset_for_role(args, role="target_old", pkl=str(args.wisig_pkl), tx_ids=str(args.target_old_tx_ids), rxs=target_rx, seed_offset=307)
    unknown_ds, unknown_info = _dataset_for_role(args, role="target_unknown", pkl=str(args.new_wisig_pkl), tx_ids=unknown_tx, rxs=target_rx, seed_offset=409)
    parts = []
    for role, ds, offset in [
        ("source", source_ds, 2001),
        ("proxy_unknown", proxy_ds, 2011),
        ("target_old", target_old_ds, 2021),
        ("target_unknown", unknown_ds, 2031),
    ]:
        loader = DataLoader(ds, batch_size=int(args.batch_size), shuffle=False, num_workers=0, drop_last=False)
        parts.append(_export_role(model, adapter, loader, args=args, device=device, role=role, scenarios=scenarios, seed=int(args.seed) + offset))
    payload = _concat(parts)
    manifest = {
        "payload_source": "phase1_iq_preadapter_satonly_features_v11",
        "feature_name": str(args.feature_name),
        "checkpoint": str(args.ckpt),
        "target_channel_view": "satellite/LEO",
        "channel_views": ["iq_preadapter"],
        "star_ground_channel_impl": str(args.star_ground_channel_impl),
        "sat_scenarios": scenarios,
        "source": source_info,
        "proxy_unknown": proxy_info,
        "target_old": target_old_info,
        "target_unknown": unknown_info,
        "uses_target_clean": False,
        "uses_target_labels_for_training": False,
        "uses_unknown_query_for_threshold": False,
        "adapter": train_info,
    }
    payload["manifest_json"] = np.asarray(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    out_dir = Path(args.runs_root) / name / str(args.out_subdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / str(args.out_name)
    np.savez(out_path, **payload)
    return out_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--wisig_pkl", default="./Dataset_WigSig/ManySig.pkl")
    p.add_argument("--new_wisig_pkl", default="./Dataset_WigSig/ManyTx.pkl")
    p.add_argument("--runs_root", type=Path, required=True)
    p.add_argument("--out_subdir", default="ADV3B02_CORE90_SOFT_E200_PHASE1_IQPRE_V11")
    p.add_argument("--out_name", default="features_iqpre_v11.npz")
    p.add_argument("--cells", required=True, help="semicolon-separated name:target_rx:unknown_tx_ids")
    p.add_argument("--feature_name", default="z_id")
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
    p.add_argument("--proxy_unknown_tx_ids", default="9-1,8-3,8-18,8-13,8-1,7-11,7-10,6-6,6-1,5-5,4-11,4-1,3-8,3-18,3-13,20-8")
    p.add_argument("--proxy_unknown_rxs", default="1-1,1-19,14-7,18-2,19-2,2-1")
    p.add_argument("--wisig_equalized", default="1")
    p.add_argument("--wisig_domain", default="rx_day")
    p.add_argument("--wisig_out_len", type=int, default=256)
    p.add_argument("--max_samples_per_combo", type=int, default=0)
    p.add_argument("--max_source_samples_per_tx", type=int, default=1000)
    p.add_argument("--max_export_samples_per_tx", type=int, default=200)
    p.add_argument("--num_old_classes", type=int, default=6)
    p.add_argument("--sat_scenarios", default="leo_clear_weak,leo_low_elev_weak,leo_rain_weak")
    p.add_argument("--star_ground_channel_impl", default="simplified_leo_residual", choices=["legacy_satellite", "simplified_leo_residual"])
    p.add_argument("--sat_fs_hz", type=float, default=25e6)
    p.add_argument("--sat_fc_hz", type=float, default=2.462e9)
    p.add_argument("--batch_size", type=int, default=384)
    p.add_argument("--epochs", type=int, default=45)
    p.add_argument("--hidden_dim", type=int, default=32)
    p.add_argument("--alpha", type=float, default=0.25)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--mse_weight", type=float, default=1.0)
    p.add_argument("--cos_weight", type=float, default=2.0)
    p.add_argument("--proto_ce_weight", type=float, default=0.6)
    p.add_argument("--logit_ce_weight", type=float, default=0.25)
    p.add_argument("--residual_weight", type=float, default=0.03)
    p.add_argument("--proto_temperature", type=float, default=0.07)
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--log_every", type=int, default=5)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=4070391)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    device = torch.device(str(args.device) if torch.cuda.is_available() else "cpu")
    source_loader, source_ds, _source_info = _make_source_loader(args)
    model = _build_model(args, source_ds, device)
    adapter, train_info = train_adapter(args, model, source_loader, device)
    exported = []
    for raw_cell in str(args.cells).split(";"):
        cell = raw_cell.strip()
        if not cell:
            continue
        exported.append(str(export_cell(args, model, adapter, device, cell, train_info)))
    summary = {
        "phase": "phase1_iq_preadapter_v11",
        "exported": exported,
        "train_info": train_info,
        "uses_target_clean": False,
        "uses_target_labels_for_training": False,
        "uses_unknown_query_for_threshold": False,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
