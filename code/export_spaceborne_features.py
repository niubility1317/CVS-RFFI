#!/usr/bin/env python
"""Export checkpoint features and WiSig metadata for spaceborne few-shot eval."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

CODE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = CODE_ROOT.parent
# Keep CODE_ROOT ahead of the legacy top-level package directory. When this
# script is executed directly, CODE_ROOT may already be sys.path[0]; remove both
# candidates first so the final order is deterministic.
for path in (str(CODE_ROOT), str(REPO_ROOT)):
    while path in sys.path:
        sys.path.remove(path)
for path in (str(REPO_ROOT), str(CODE_ROOT)):
    sys.path.insert(0, path)

from cvsrffi.wisig_fewshot_payload import assert_disjoint_tx_sets, canonical_tx_id, parse_tx_id_list
from cvsrffi.eval import apply_sat_channel_for_scenario
from cvsrffi.tensors import make_torch_generator
from dataset_wisig import WiSigCompactDataset, WiSigSubsetDataset, load_wisig_compact_pkl
from eval_feature_diagnosis import (
    build_model_from_ckpt,
    collect_feature_dict,
    infer_num_domains,
    load_state_dict_safely,
    parse_items,
    strip_module_prefix,
)
from training_controls import parse_sat_scenarios, sat_channel_config_for_scenario

SIMPLIFIED_LEO_RESIDUAL_SCENARIOS = {"leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak"}


def _validate_star_ground_impl(impl: str, scenarios: Sequence[str], *, field: str) -> None:
    impl_text = str(impl or "legacy_satellite")
    if impl_text != "simplified_leo_residual":
        return
    unexpected = sorted(set(str(s) for s in scenarios) - SIMPLIFIED_LEO_RESIDUAL_SCENARIOS)
    if unexpected:
        allowed = ",".join(sorted(SIMPLIFIED_LEO_RESIDUAL_SCENARIOS))
        raise ValueError(
            f"{field} uses star_ground_channel_impl=simplified_leo_residual but includes "
            f"non-simplified scenarios={unexpected}; allowed={allowed}"
        )


def _resolve_tx_indices(tx_list: Sequence[Any], spec: str | None, *, field: str) -> tuple[list[int], list[str]]:
    requested = parse_tx_id_list(spec)
    if not requested:
        raise ValueError(f"{field} must not be empty")
    labels = [canonical_tx_id(v) for v in tx_list]
    out_idx: list[int] = []
    out_labels: list[str] = []
    for item in requested:
        found = None
        try:
            raw_i = int(item)
        except Exception:
            raw_i = None
        if raw_i is not None and 0 <= raw_i < len(labels):
            found = raw_i
        else:
            for i, label in enumerate(labels):
                if label == item:
                    found = i
                    break
        if found is None:
            raise ValueError(f"cannot resolve {field} item {item!r} from tx_list={labels}")
        out_idx.append(int(found))
        out_labels.append(labels[int(found)])
    return out_idx, out_labels


def _resolve_indices(name_list: Sequence[Any], spec: str | None) -> list[int] | None:
    items = parse_items(spec)
    if items is None:
        return None
    labels = [canonical_tx_id(v) for v in name_list]
    out: list[int] = []
    for item in items:
        found = None
        if isinstance(item, int) and 0 <= item < len(labels):
            found = int(item)
        else:
            s = canonical_tx_id(item)
            for i, label in enumerate(labels):
                if label == s:
                    found = i
                    break
        if found is None:
            raise ValueError(f"cannot resolve {item!r} from {labels}")
        out.append(int(found))
    return sorted(set(out))


def _cap_dataset_per_tx(ds: WiSigCompactDataset, max_samples_per_tx: int, seed: int, split_source: str):
    if int(max_samples_per_tx) <= 0:
        return ds
    by_tx: dict[int, list[int]] = {}
    for i, it in enumerate(ds.index):
        by_tx.setdefault(int(it.tx_i), []).append(int(i))
    rng = np.random.default_rng(int(seed))
    selected: list[int] = []
    for tx_i in sorted(by_tx):
        idx = by_tx[tx_i]
        if len(idx) > int(max_samples_per_tx):
            pick = rng.permutation(len(idx))[: int(max_samples_per_tx)].tolist()
            idx = sorted(idx[int(i)] for i in pick)
        selected.extend(idx)
    return WiSigSubsetDataset(ds, sorted(selected), split_source=split_source)


def _build_wisig_dataset(
    *,
    pkl_path: str,
    tx_spec: str,
    role: str,
    equalized: str,
    out_len: int,
    domain: str,
    days: str | None,
    rxs: str | None,
    max_samples_per_combo: int,
    max_samples_per_tx: int,
    seed: int,
):
    ds = load_wisig_compact_pkl(pkl_path)
    tx_idx, tx_labels = _resolve_tx_indices(ds.get("tx_list", []), tx_spec, field=f"{role}_tx_ids")
    day_keep = _resolve_indices(ds.get("capture_date_list", []), days)
    rx_keep = _resolve_indices(ds.get("rx_list", []), rxs)
    base = WiSigCompactDataset(
        ds,
        out_len=int(out_len),
        equalized=("both" if str(equalized).lower() == "both" else int(equalized)),
        tx_keep=tx_idx,
        day_keep=day_keep,
        rx_keep=rx_keep,
        domain=str(domain),
        max_samples_per_combo=(None if int(max_samples_per_combo) <= 0 else int(max_samples_per_combo)),
        sample_strategy="random",
        seed=int(seed),
        build_index=True,
    )
    capped = _cap_dataset_per_tx(base, int(max_samples_per_tx), int(seed), split_source=f"{role}_max_per_tx")
    info = {
        "pkl": str(pkl_path),
        "role": role,
        "tx_idx": tx_idx,
        "tx_labels": tx_labels,
        "days": days,
        "rxs": rxs,
        "size": len(capped),
    }
    return capped, info


def _meta_to_list(meta: Any, key: str, n: int) -> list[str]:
    if not isinstance(meta, dict) or key not in meta:
        return [""] * int(n)
    value = meta[key]
    if torch.is_tensor(value):
        return [canonical_tx_id(v) for v in value.detach().cpu().reshape(-1).tolist()]
    if isinstance(value, np.ndarray):
        return [canonical_tx_id(v) for v in value.reshape(-1).tolist()]
    if isinstance(value, (list, tuple)):
        return [canonical_tx_id(v) for v in value]
    return [canonical_tx_id(value)] * int(n)


@torch.no_grad()
def extract_features_with_metadata(
    model,
    loader,
    *,
    device: torch.device,
    feature_name: str,
    role: str,
    channel_view: str = "clean",
    sat_scenarios: Sequence[str] | None = None,
    sat_args: argparse.Namespace | None = None,
    sat_seed: int = 0,
):
    feature_buf: list[np.ndarray] = []
    tx_logit_buf: list[np.ndarray] = []
    tx_buf: list[str] = []
    rx_buf: list[str] = []
    day_buf: list[str] = []
    eq_buf: list[str] = []
    sig_buf: list[str] = []
    role_buf: list[str] = []
    channel_view_buf: list[str] = []
    sat_scenario_buf: list[str] = []
    label_buf: list[int] = []
    domain_buf: list[int] = []
    model.eval()
    view = str(channel_view or "clean").lower()
    scenarios = list(sat_scenarios or [])
    sat_gen = make_torch_generator(device, int(sat_seed)) if view == "satellite" else None
    if view == "satellite" and not scenarios:
        raise ValueError(f"satellite feature export requires at least one scenario for role={role}")
    for bi, batch in enumerate(loader):
        if len(batch) == 4:
            x, y, d, meta = batch
        else:
            raise ValueError("WiSig feature export expects batches shaped (x, y, d, meta)")
        x = x.to(device, non_blocking=True)
        scenario = ""
        if view == "satellite":
            scenario = str(scenarios[int(bi) % len(scenarios)])
            if sat_args is None:
                raise ValueError("sat_args is required for satellite feature export")
            x, _ = apply_sat_channel_for_scenario(x, scenario, sat_args, gen=sat_gen, return_meta=False)
        out = model(x, y_tx=None, grl_lambda=1.0, return_aux=True)
        feats = collect_feature_dict(out)
        if feature_name not in feats:
            raise KeyError(f"feature {feature_name!r} not found; available={sorted(feats.keys())}")
        z = feats[feature_name].detach().cpu().float().numpy()
        logits_obj = out.get("tx_logits", out.get("logits")) if isinstance(out, dict) else None
        if logits_obj is None:
            raise KeyError("model output does not include tx_logits/logits for Phase1 classifier audit")
        tx_logits = logits_obj.detach().cpu().float().numpy()
        n = int(z.shape[0])
        feature_buf.append(z)
        tx_logit_buf.append(tx_logits)
        label_buf.extend([int(v) for v in y.detach().cpu().reshape(-1).tolist()])
        domain_buf.extend([int(v) for v in d.detach().cpu().reshape(-1).tolist()])
        tx_buf.extend(_meta_to_list(meta, "tx", n))
        rx_buf.extend(_meta_to_list(meta, "rx", n))
        day_buf.extend(_meta_to_list(meta, "day", n))
        eq_buf.extend(_meta_to_list(meta, "equalized", n))
        sig_buf.extend(_meta_to_list(meta, "sig_i", n))
        role_buf.extend([role] * n)
        channel_view_buf.extend([view] * n)
        sat_scenario_buf.extend([scenario] * n)
    if not feature_buf:
        raise ValueError(f"dataset role={role} produced no features")
    return {
        "features": np.concatenate(feature_buf, axis=0).astype(np.float32),
        "tx_logits": np.concatenate(tx_logit_buf, axis=0).astype(np.float32),
        "raw_labels": np.asarray(label_buf, dtype=np.int64),
        "domain_labels": np.asarray(domain_buf, dtype=np.int64),
        "tx_ids": np.asarray(tx_buf),
        "rx_ids": np.asarray(rx_buf),
        "day_ids": np.asarray(day_buf),
        "eq_ids": np.asarray(eq_buf),
        "sig_ids": np.asarray(sig_buf),
        "dataset_role": np.asarray(role_buf),
        "channel_views": np.asarray(channel_view_buf),
        "sat_scenarios": np.asarray(sat_scenario_buf),
    }


def _concat_payloads(payloads: Sequence[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = list(payloads[0].keys())
    return {key: np.concatenate([p[key] for p in payloads], axis=0) for key in keys}


def _write_synthetic(out_npz: Path) -> None:
    features = []
    tx_ids = []
    for tx, base in {
        "old_a": [1.0, 0.0],
        "old_b": [0.0, 1.0],
        "new_c": [-1.0, 0.0],
        "unknown_d": [0.0, -1.0],
    }.items():
        for i in range(80):
            features.append([base[0] + 0.001 * i, base[1] - 0.001 * i])
            tx_ids.append(tx)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_npz,
        features=np.asarray(features, dtype=np.float32),
        tx_ids=np.asarray(tx_ids),
        dataset_role=np.asarray(["synthetic"] * len(tx_ids)),
        manifest_json=json.dumps({"payload_source": "dry_run_synthetic_export"}, ensure_ascii=True),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, default=None)
    parser.add_argument("--wisig_pkl", type=str, default="./Dataset_WigSig/ManySig.pkl")
    parser.add_argument("--new_wisig_pkl", type=str, default=None)
    parser.add_argument("--out_npz", type=Path, required=True)
    parser.add_argument("--feature_name", default="z_id")
    parser.add_argument("--dataset", default="wisig")
    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument("--model_size", default=None)
    parser.add_argument("--model_variant", default=None)
    parser.add_argument("--branch_ablation", default=None)
    parser.add_argument("--sample_rate_hz", type=float, default=None)
    parser.add_argument("--source_tx_ids", required=False)
    parser.add_argument("--target_old_tx_ids", default=None)
    parser.add_argument("--new_tx_ids", required=False)
    parser.add_argument("--unknown_tx_ids", default=None)
    parser.add_argument("--proxy_unknown_tx_ids", default=None)
    parser.add_argument("--wisig_equalized", default="1")
    parser.add_argument("--wisig_domain", default="rx_day")
    parser.add_argument("--wisig_out_len", type=int, default=256)
    parser.add_argument("--source_days", default=None)
    parser.add_argument("--source_rxs", default=None)
    parser.add_argument("--target_old_days", default=None)
    parser.add_argument("--target_old_rxs", default=None)
    parser.add_argument("--new_days", default=None)
    parser.add_argument("--new_rxs", default=None)
    parser.add_argument("--proxy_unknown_days", default=None)
    parser.add_argument("--proxy_unknown_rxs", default=None)
    parser.add_argument("--max_samples_per_combo", type=int, default=0)
    parser.add_argument("--max_samples_per_tx", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--target_new_channel_view", default="clean", choices=["clean", "satellite"])
    parser.add_argument("--target_new_sat_scenarios", default="clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit")
    parser.add_argument("--target_new_sat_seed", type=int, default=None)
    parser.add_argument("--target_old_channel_view", default=None, choices=["clean", "satellite"])
    parser.add_argument("--target_old_sat_scenarios", default=None)
    parser.add_argument("--target_old_sat_seed", type=int, default=None)
    parser.add_argument("--proxy_unknown_channel_view", default="clean", choices=["clean", "satellite"])
    parser.add_argument("--proxy_unknown_sat_scenarios", default=None)
    parser.add_argument("--proxy_unknown_sat_seed", type=int, default=None)
    parser.add_argument(
        "--star_ground_channel_impl",
        default="legacy_satellite",
        choices=["legacy_satellite", "simplified_leo_residual"],
        help="Audit/control field for the star-ground channel implementation used by satellite views.",
    )
    parser.add_argument("--sat_fs_hz", type=float, default=25e6)
    parser.add_argument("--sat_fc_hz", type=float, default=2.462e9)
    parser.add_argument("--dry_run_synthetic", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.dry_run_synthetic:
        _write_synthetic(args.out_npz)
        print(json.dumps({"out_npz": str(args.out_npz), "mode": "dry_run_synthetic"}, ensure_ascii=False))
        return 0
    if args.ckpt is None:
        raise ValueError("--ckpt is required unless --dry_run_synthetic is set")
    if not args.source_tx_ids:
        raise ValueError("--source_tx_ids is required for real WiSig export")
    requested_new_tx_ids = parse_tx_id_list(args.new_tx_ids)
    old_unknown_only = not requested_new_tx_ids
    if old_unknown_only and not parse_tx_id_list(args.target_old_tx_ids):
        raise ValueError("--target_old_tx_ids is required when --new_tx_ids is omitted")

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
        max_samples_per_tx=int(args.max_samples_per_tx),
        seed=int(args.seed),
    )
    new_pkl_path = str(args.new_wisig_pkl or args.wisig_pkl)
    new_raw = load_wisig_compact_pkl(new_pkl_path)
    resolved_new_labels: list[str] = []
    if requested_new_tx_ids:
        _, resolved_new_labels = _resolve_tx_indices(new_raw.get("tx_list", []), args.new_tx_ids, field="new_tx_ids")
    resolved_unknown_labels: list[str] = []
    if parse_tx_id_list(args.unknown_tx_ids):
        _, resolved_unknown_labels = _resolve_tx_indices(
            new_raw.get("tx_list", []),
            args.unknown_tx_ids,
            field="unknown_tx_ids",
        )
    resolved_proxy_unknown_labels: list[str] = []
    if parse_tx_id_list(args.proxy_unknown_tx_ids):
        _, resolved_proxy_unknown_labels = _resolve_tx_indices(
            new_raw.get("tx_list", []),
            args.proxy_unknown_tx_ids,
            field="proxy_unknown_tx_ids",
        )
    overlap_audit = assert_disjoint_tx_sets(
        source_tx_ids=source_info["tx_labels"],
        new_tx_ids=resolved_new_labels,
        unknown_tx_ids=resolved_unknown_labels,
    )
    proxy_overlap_audit = {
        "proxy_unknown_overlaps_source": sorted(set(resolved_proxy_unknown_labels).intersection(source_info["tx_labels"])),
        "proxy_unknown_overlaps_target_unknown": sorted(
            set(resolved_proxy_unknown_labels).intersection(resolved_unknown_labels)
        ),
        "proxy_unknown_overlaps_target_new": sorted(set(resolved_proxy_unknown_labels).intersection(resolved_new_labels)),
    }
    if proxy_overlap_audit["proxy_unknown_overlaps_source"]:
        raise ValueError(f"proxy_unknown_tx_ids overlap source_tx_ids: {proxy_overlap_audit}")
    new_ds = None
    new_info = None
    if resolved_new_labels:
        new_ds, new_info = _build_wisig_dataset(
            pkl_path=new_pkl_path,
            tx_spec=",".join(resolved_new_labels),
            role="target_new",
            equalized=str(args.wisig_equalized),
            out_len=int(args.wisig_out_len),
            domain=str(args.wisig_domain),
            days=args.new_days,
            rxs=args.new_rxs,
            max_samples_per_combo=int(args.max_samples_per_combo),
            max_samples_per_tx=int(args.max_samples_per_tx),
            seed=int(args.seed) + 17,
        )
    unknown_ds = None
    unknown_info = None
    if resolved_unknown_labels:
        unknown_days = args.new_days
        unknown_rxs = args.new_rxs
        if old_unknown_only:
            unknown_days = args.target_old_days or args.new_days
            unknown_rxs = args.target_old_rxs or args.new_rxs
        unknown_ds, unknown_info = _build_wisig_dataset(
            pkl_path=new_pkl_path,
            tx_spec=",".join(resolved_unknown_labels),
            role="target_unknown",
            equalized=str(args.wisig_equalized),
            out_len=int(args.wisig_out_len),
            domain=str(args.wisig_domain),
            days=unknown_days,
            rxs=unknown_rxs,
            max_samples_per_combo=int(args.max_samples_per_combo),
            max_samples_per_tx=int(args.max_samples_per_tx),
            seed=int(args.seed) + 31,
        )
    proxy_unknown_ds = None
    proxy_unknown_info = None
    if resolved_proxy_unknown_labels:
        proxy_unknown_ds, proxy_unknown_info = _build_wisig_dataset(
            pkl_path=new_pkl_path,
            tx_spec=",".join(resolved_proxy_unknown_labels),
            role="proxy_unknown",
            equalized=str(args.wisig_equalized),
            out_len=int(args.wisig_out_len),
            domain=str(args.wisig_domain),
            days=args.proxy_unknown_days or args.new_days,
            rxs=args.proxy_unknown_rxs or args.source_rxs,
            max_samples_per_combo=int(args.max_samples_per_combo),
            max_samples_per_tx=int(args.max_samples_per_tx),
            seed=int(args.seed) + 43,
        )

    ckpt = torch.load(args.ckpt, map_location="cpu")
    if "args" not in ckpt or "model" not in ckpt:
        raise KeyError("checkpoint must contain 'args' and 'model'")
    ckpt_args = ckpt["args"]
    state = strip_module_prefix(ckpt["model"])
    num_domains = infer_num_domains(source_ds, state=state, split_info={}, ckpt_args=ckpt_args, cli_num_domains=None)
    device = torch.device(str(args.device) if torch.cuda.is_available() else "cpu")
    model = build_model_from_ckpt(ckpt_args, args, num_domains=num_domains, input_len=int(args.wisig_out_len), device=device)
    missing, unexpected, skipped_mismatch = load_state_dict_safely(model, state)

    source_loader = DataLoader(source_ds, batch_size=int(args.batch_size), shuffle=False, num_workers=0, drop_last=False)
    new_loader = (
        DataLoader(new_ds, batch_size=int(args.batch_size), shuffle=False, num_workers=0, drop_last=False)
        if new_ds is not None
        else None
    )
    star_ground_impl = str(args.star_ground_channel_impl)
    target_new_view = str(args.target_new_channel_view).lower()
    target_new_scenarios = parse_sat_scenarios(str(args.target_new_sat_scenarios)) if target_new_view == "satellite" else []
    if not old_unknown_only:
        _validate_star_ground_impl(star_ground_impl, target_new_scenarios, field="target_new_sat_scenarios")
    source_payload = extract_features_with_metadata(
        model,
        source_loader,
        device=device,
        feature_name=str(args.feature_name),
        role="source",
        channel_view="clean",
    )
    proxy_unknown_payload = None
    proxy_unknown_channel_profile = None
    if proxy_unknown_ds is not None:
        proxy_unknown_view = str(args.proxy_unknown_channel_view).lower()
        proxy_unknown_scenarios = (
            parse_sat_scenarios(str(args.proxy_unknown_sat_scenarios))
            if proxy_unknown_view == "satellite"
            else []
        )
        _validate_star_ground_impl(star_ground_impl, proxy_unknown_scenarios, field="proxy_unknown_sat_scenarios")
        proxy_unknown_seed = int(
            args.proxy_unknown_sat_seed if args.proxy_unknown_sat_seed is not None else int(args.seed) + 719
        )
        proxy_unknown_payload = extract_features_with_metadata(
            model,
            DataLoader(
                proxy_unknown_ds,
                batch_size=int(args.batch_size),
                shuffle=False,
                num_workers=0,
                drop_last=False,
            ),
            device=device,
            feature_name=str(args.feature_name),
            role="proxy_unknown",
            channel_view=proxy_unknown_view,
            sat_scenarios=proxy_unknown_scenarios,
            sat_args=args,
            sat_seed=proxy_unknown_seed,
        )
        proxy_unknown_channel_profile = {
            "view": proxy_unknown_view,
            "applied_roles": ["proxy_unknown"] if proxy_unknown_view == "satellite" else [],
            "downstream_roles": ["source_proxy_unknown_calibration"],
            "scenarios": proxy_unknown_scenarios,
            "scenario_configs": {name: sat_channel_config_for_scenario(name) for name in proxy_unknown_scenarios},
            "star_ground_channel_impl": star_ground_impl,
            "sat_seed": proxy_unknown_seed,
            "fs_hz": float(args.sat_fs_hz),
            "fc_hz": float(args.sat_fc_hz),
        }
    target_old_payload = None
    target_old_info = None
    target_old_channel_profile = None
    target_old_view = None
    target_old_scenarios: list[str] = []
    target_old_seed = None
    if parse_tx_id_list(args.target_old_tx_ids):
        target_old_ds, target_old_info = _build_wisig_dataset(
            pkl_path=str(args.wisig_pkl),
            tx_spec=str(args.target_old_tx_ids),
            role="target_old",
            equalized=str(args.wisig_equalized),
            out_len=int(args.wisig_out_len),
            domain=str(args.wisig_domain),
            days=args.target_old_days or args.new_days,
            rxs=args.target_old_rxs or args.new_rxs,
            max_samples_per_combo=int(args.max_samples_per_combo),
            max_samples_per_tx=int(args.max_samples_per_tx),
            seed=int(args.seed) + 29,
        )
        target_old_view = str(args.target_old_channel_view or args.target_new_channel_view).lower()
        target_old_scenarios = (
            parse_sat_scenarios(str(args.target_old_sat_scenarios or args.target_new_sat_scenarios))
            if target_old_view == "satellite"
            else []
        )
        _validate_star_ground_impl(star_ground_impl, target_old_scenarios, field="target_old_sat_scenarios")
        target_old_seed = int(args.target_old_sat_seed if args.target_old_sat_seed is not None else int(args.seed) + 811)
        target_old_payload = extract_features_with_metadata(
            model,
            DataLoader(target_old_ds, batch_size=int(args.batch_size), shuffle=False, num_workers=0, drop_last=False),
            device=device,
            feature_name=str(args.feature_name),
            role="target_old",
            channel_view=target_old_view,
            sat_scenarios=target_old_scenarios,
            sat_args=args,
            sat_seed=target_old_seed,
        )
        target_old_channel_profile = {
            "view": target_old_view,
            "applied_roles": ["target_old"] if target_old_view == "satellite" else [],
            "downstream_roles": ["target_old_support", "target_old_query"] if target_old_view == "satellite" else [],
            "scenarios": target_old_scenarios,
            "scenario_configs": {name: sat_channel_config_for_scenario(name) for name in target_old_scenarios},
            "star_ground_channel_impl": star_ground_impl,
            "sat_seed": target_old_seed,
            "fs_hz": float(args.sat_fs_hz),
            "fc_hz": float(args.sat_fc_hz),
            "implementation": "cvsrffi.eval.apply_sat_channel_for_scenario -> sat_channel.apply_sat_gnd_channel_batch",
        }
    new_payload = None
    if new_loader is not None:
        new_payload = extract_features_with_metadata(
            model,
            new_loader,
            device=device,
            feature_name=str(args.feature_name),
            role="target_new",
            channel_view=target_new_view,
            sat_scenarios=target_new_scenarios,
            sat_args=args,
            sat_seed=int(args.target_new_sat_seed if args.target_new_sat_seed is not None else int(args.seed) + 911),
        )
    unknown_payload = None
    target_unknown_view = target_new_view
    target_unknown_scenarios = target_new_scenarios
    target_unknown_seed = int(args.target_new_sat_seed if args.target_new_sat_seed is not None else int(args.seed) + 913)
    if old_unknown_only and target_old_view is not None:
        target_unknown_view = str(target_old_view)
        target_unknown_scenarios = list(target_old_scenarios)
        target_unknown_seed = int(args.target_old_sat_seed if args.target_old_sat_seed is not None else int(args.seed) + 913)
    if unknown_ds is not None:
        unknown_payload = extract_features_with_metadata(
            model,
            DataLoader(unknown_ds, batch_size=int(args.batch_size), shuffle=False, num_workers=0, drop_last=False),
            device=device,
            feature_name=str(args.feature_name),
            role="target_unknown",
            channel_view=target_unknown_view,
            sat_scenarios=target_unknown_scenarios,
            sat_args=args,
            sat_seed=target_unknown_seed,
        )
    payload_parts = [source_payload]
    if proxy_unknown_payload is not None:
        payload_parts.append(proxy_unknown_payload)
    if target_old_payload is not None:
        payload_parts.append(target_old_payload)
    if new_payload is not None:
        payload_parts.append(new_payload)
    if unknown_payload is not None:
        payload_parts.append(unknown_payload)
    payload = _concat_payloads(payload_parts)
    target_new_channel_profile = None
    if not old_unknown_only:
        target_new_channel_profile = {
            "view": target_new_view,
            "applied_roles": ["target_new"] if target_new_view == "satellite" else [],
            "downstream_roles": ["new_support", "new_query"] if target_new_view == "satellite" else [],
            "scenarios": target_new_scenarios,
            "scenario_configs": {name: sat_channel_config_for_scenario(name) for name in target_new_scenarios},
            "star_ground_channel_impl": star_ground_impl,
            "sat_seed": int(args.target_new_sat_seed if args.target_new_sat_seed is not None else int(args.seed) + 911),
            "fs_hz": float(args.sat_fs_hz),
            "fc_hz": float(args.sat_fc_hz),
            "implementation": "cvsrffi.eval.apply_sat_channel_for_scenario -> sat_channel.apply_sat_gnd_channel_batch",
        }
    target_unknown_channel_profile = None
    if unknown_info is not None:
        target_unknown_channel_profile = {
            "view": target_unknown_view,
            "applied_roles": ["target_unknown"] if target_unknown_view == "satellite" else [],
            "downstream_roles": ["unknown_query"] if target_unknown_view == "satellite" else [],
            "scenarios": target_unknown_scenarios,
            "scenario_configs": {name: sat_channel_config_for_scenario(name) for name in target_unknown_scenarios},
            "star_ground_channel_impl": star_ground_impl,
            "sat_seed": target_unknown_seed,
            "fs_hz": float(args.sat_fs_hz),
            "fc_hz": float(args.sat_fc_hz),
            "implementation": "cvsrffi.eval.apply_sat_channel_for_scenario -> sat_channel.apply_sat_gnd_channel_batch",
        }
    manifest = {
        "feature_name": str(args.feature_name),
        "checkpoint": str(args.ckpt),
        "target_new_channel_view": "disabled" if old_unknown_only else target_new_view,
        "target_unknown_channel_view": target_unknown_view,
        "target_channel_view": "satellite/LEO" if target_unknown_view == "satellite" else "clean",
        "star_ground_channel_impl": star_ground_impl,
        "target_channel_scenarios": target_unknown_scenarios,
        "deployment_primary_view": (
            "satellite/LEO target view" if target_unknown_view == "satellite" else "clean control/source reference"
        ),
        "source": source_info,
        "target_old": target_old_info,
        "target_new": new_info,
        "target_unknown": unknown_info,
        "proxy_unknown": proxy_unknown_info,
        "source_tx_ids": source_info["tx_labels"],
        "target_old_tx_ids": [] if target_old_info is None else target_old_info["tx_labels"],
        "new_tx_ids": resolved_new_labels,
        "unknown_tx_ids": resolved_unknown_labels,
        "proxy_unknown_tx_ids": resolved_proxy_unknown_labels,
        "requested_source_tx_ids": parse_tx_id_list(args.source_tx_ids),
        "requested_target_old_tx_ids": parse_tx_id_list(args.target_old_tx_ids),
        "requested_new_tx_ids": parse_tx_id_list(args.new_tx_ids),
        "requested_unknown_tx_ids": parse_tx_id_list(args.unknown_tx_ids),
        "requested_proxy_unknown_tx_ids": parse_tx_id_list(args.proxy_unknown_tx_ids),
        "overlap_audit": overlap_audit,
        "proxy_overlap_audit": proxy_overlap_audit,
        "channel_profile": {
            "source": {"view": "clean", "applied_roles": []},
            "proxy_unknown": proxy_unknown_channel_profile,
            "target_old": target_old_channel_profile,
            "target_new": target_new_channel_profile,
            "target_unknown": target_unknown_channel_profile,
        },
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
        "skipped_mismatch": len(skipped_mismatch),
    }
    payload["manifest_json"] = np.asarray(json.dumps(manifest, ensure_ascii=True))
    args.out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out_npz, **payload)
    print(json.dumps({"out_npz": str(args.out_npz), "manifest": manifest}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
