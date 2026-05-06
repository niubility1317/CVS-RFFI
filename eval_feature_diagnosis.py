import os
import json
import math
import argparse
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from dataset import WiFiRFFIDataset
from dataset_wisig import (
    load_wisig_compact_pkl,
    make_day123_trainval_day4_test,
    WiSigCompactDataset,
    WiSigSubsetDataset,
    WiSigConcatDataset,
)
from model_dual_cvsincnet import build_dual_model
from DataAugmentation import build_augmentor


# =============================
# basic utils
# =============================
def set_seed(seed: int = 1337):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_numpy(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def l2_normalize_np(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, eps)


def normalized_accuracy(acc_pct: float, n_cls: int) -> float:
    if n_cls <= 1:
        return 0.0
    chance = 100.0 / float(n_cls)
    return max(0.0, (float(acc_pct) - chance) / max(1e-6, 100.0 - chance))


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            out[k[7:]] = v
        else:
            out[k] = v
    return out


def parse_days(s: Optional[str]):
    if s is None:
        return None
    s = str(s).strip()
    if s == "":
        return None
    out = []
    for item in s.split(","):
        item = item.strip()
        if item == "":
            continue
        try:
            out.append(int(item))
        except Exception:
            out.append(item)
    return out if out else None


def parse_items(s: Optional[str]):
    if s is None:
        return None
    s = str(s).strip()
    if s == "":
        return None
    out = []
    for item in s.split(","):
        item = item.strip()
        if item == "":
            continue
        try:
            out.append(int(item))
        except Exception:
            out.append(item)
    return out if out else None


def resolve_keep_indices(name_list, items: Optional[list], field_name: str):
    if items is None:
        return None
    out = []
    for it in items:
        if isinstance(it, int):
            if 0 <= it < len(name_list):
                out.append(int(it))
                continue
            raise ValueError(f"{field_name} index out of range: {it}; valid=0..{len(name_list)-1}")
        s = str(it)
        found = None
        for i, v in enumerate(name_list):
            if str(v) == s:
                found = i
                break
        if found is None:
            raise ValueError(f"Cannot resolve {field_name}={it!r} from {list(name_list)}")
        out.append(found)
    return sorted(set(out))


class EmptyDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 0

    def __getitem__(self, idx):
        raise IndexError("EmptyDataset")


def _filter_wisig_match(it, keep_days=None, keep_rxs=None):
    if keep_days is not None and int(it.day_i) not in keep_days:
        return False
    if keep_rxs is not None and int(it.rx_i) not in keep_rxs:
        return False
    return True


def filter_wisig_dataset(ds, keep_days=None, keep_rxs=None, split_tag: str = "filtered"):
    keep_days = None if keep_days is None else set(int(v) for v in keep_days)
    keep_rxs = None if keep_rxs is None else set(int(v) for v in keep_rxs)

    if isinstance(ds, WiSigSubsetDataset):
        local_keep = [i for i, it in enumerate(ds.index) if _filter_wisig_match(it, keep_days, keep_rxs)]
        selected = ds.selected[local_keep].tolist() if len(local_keep) > 0 else []
        return WiSigSubsetDataset(ds.base, selected, split_source=f"{ds.split_source}|{split_tag}", transform=ds.transform)

    if isinstance(ds, WiSigCompactDataset):
        selected = [i for i, it in enumerate(ds.index) if _filter_wisig_match(it, keep_days, keep_rxs)]
        return WiSigSubsetDataset(ds, selected, split_source=split_tag, transform=None)

    if isinstance(ds, WiSigConcatDataset):
        parts = []
        for child in ds.datasets:
            sub = filter_wisig_dataset(child, keep_days=keep_days, keep_rxs=keep_rxs, split_tag=split_tag)
            if len(sub) > 0:
                parts.append(sub)
        if len(parts) == 0:
            return EmptyDataset()
        if len(parts) == 1:
            return parts[0]
        return WiSigConcatDataset(parts)

    if isinstance(ds, Subset):
        root = unwrap_dataset(ds)
        if isinstance(root, (WiSigSubsetDataset, WiSigCompactDataset, WiSigConcatDataset)):
            base_filtered = filter_wisig_dataset(root, keep_days=keep_days, keep_rxs=keep_rxs, split_tag=split_tag)
            if len(base_filtered) == 0:
                return EmptyDataset()
            max_idx = len(base_filtered)
            selected = [int(i) for i in ds.indices if 0 <= int(i) < max_idx]
            return Subset(base_filtered, selected)

    return ds


def build_wisig_explicit_test_dataset(ds_w, ckpt_args: Dict[str, Any], cli_args):
    day_items = parse_items(getattr(cli_args, 'eval_test_days', None))
    rx_items = parse_items(getattr(cli_args, 'eval_test_rxs', None))

    default_test_days = parse_days(ckpt_args.get('wisig_test_days', '3'))
    day_keep = resolve_keep_indices(ds_w.get('capture_date_list', []), day_items, 'test day') if day_items is not None else resolve_keep_indices(ds_w.get('capture_date_list', []), default_test_days, 'test day')
    rx_keep = resolve_keep_indices(ds_w.get('rx_list', []), rx_items, 'test rx') if rx_items is not None else None

    equalized = ("both" if str(ckpt_args.get('wisig_equalized', '1')).lower() == 'both' else int(ckpt_args.get('wisig_equalized', '1')))
    out_len = int(cli_args.wisig_out_len or ckpt_args.get('wisig_out_len', 256))
    domain = str(cli_args.wisig_domain or ckpt_args.get('wisig_domain', 'day'))
    max_test = (None if int(ckpt_args.get('wisig_max_test_per_combo', 0)) <= 0 else int(ckpt_args.get('wisig_max_test_per_combo', 0)))

    test_ds = WiSigCompactDataset(
        ds_w,
        out_len=out_len,
        crop_mode='center',
        normalize=True,
        equalized=equalized,
        tx_keep=None,
        rx_keep=rx_keep,
        day_keep=day_keep,
        domain=domain,
        transform=None,
        max_samples_per_combo=max_test,
        seed=int(ckpt_args.get('seed', 1337)),
        build_index=True,
    )

    info = {
        'eval_test_days_idx': day_keep,
        'eval_test_days_label': [ds_w.get('capture_date_list', [])[i] for i in day_keep],
        'eval_test_rxs_idx': rx_keep if rx_keep is not None else list(range(len(ds_w.get('rx_list', [])))),
        'eval_test_rxs_label': ([ds_w.get('rx_list', [])[i] for i in rx_keep] if rx_keep is not None else list(ds_w.get('rx_list', []))),
        'eval_test_size': len(test_ds),
    }
    return test_ds, info


def maybe_subset_dataset(ds, max_samples: int, seed: int):
    if max_samples is None or max_samples <= 0 or len(ds) <= max_samples:
        return ds
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), size=max_samples, replace=False)
    idx = np.sort(idx).tolist()
    return Subset(ds, idx)


def unwrap_dataset(ds):
    cur = ds
    visited = set()
    while True:
        oid = id(cur)
        if oid in visited:
            break
        visited.add(oid)
        if hasattr(cur, 'dataset'):
            cur = cur.dataset
            continue
        break
    return cur


# =============================
# data helpers
# =============================
def unpack_batch(batch):
    x = batch[0]
    y = batch[1]
    extra = batch[2:] if isinstance(batch, (tuple, list)) and len(batch) > 2 else ()
    return x, y, extra


def extract_domain_from_extra(extra, device) -> Optional[torch.Tensor]:
    if extra is None or len(extra) == 0:
        return None
    d0 = extra[0]
    if torch.is_tensor(d0):
        return d0.to(device, non_blocking=True).view(-1)
    try:
        return torch.as_tensor(d0, device=device).view(-1)
    except Exception:
        return None


def infer_num_domains_from_state(state: Dict[str, torch.Tensor]) -> Optional[int]:
    keys = [
        'dom_head.net.3.bias',
        'dom_head.net.3.weight',
        'adv_head.net.3.bias',
        'adv_head.net.3.weight',
    ]
    for k in keys:
        v = state.get(k, None)
        if v is None:
            continue
        try:
            if v.ndim == 1:
                return int(v.shape[0])
            if v.ndim >= 2:
                return int(v.shape[0])
        except Exception:
            continue
    return None


def infer_num_domains_from_dataset(ds) -> Optional[int]:
    # Unwrap torch.utils.data.Subset chains first.
    root = unwrap_dataset(ds)

    candidates = [ds]
    if root is not ds:
        candidates.append(root)
    if hasattr(root, 'base'):
        candidates.append(root.base)

    for obj in candidates:
        if hasattr(obj, 'index') and hasattr(obj, '_domain_lut'):
            try:
                doms = sorted({int(obj._domain_lut[(it.rx_i, it.day_i)]) for it in obj.index})
                if len(doms) > 0:
                    return max(1, len(doms))
            except Exception:
                pass
        if hasattr(obj, 'index') and hasattr(obj, 'base') and hasattr(obj.base, '_domain_lut'):
            try:
                doms = sorted({int(obj.base._domain_lut[(it.rx_i, it.day_i)]) for it in obj.index})
                if len(doms) > 0:
                    return max(1, len(doms))
            except Exception:
                pass
    return None


def infer_num_domains(
    ds,
    state: Optional[Dict[str, torch.Tensor]] = None,
    split_info: Optional[Dict[str, Any]] = None,
    ckpt_args: Optional[Dict[str, Any]] = None,
    cli_num_domains: Optional[int] = None,
) -> int:
    if cli_num_domains is not None and int(cli_num_domains) > 0:
        return int(cli_num_domains)

    n_state = infer_num_domains_from_state(state or {})
    if n_state is not None and n_state > 0:
        return int(n_state)

    n_ds = infer_num_domains_from_dataset(ds)
    if n_ds is not None and n_ds > 0:
        return int(n_ds)

    # Last-resort fallback from split metadata for common WiSig cases.
    domain = None
    if ckpt_args is not None:
        domain = ckpt_args.get('wisig_domain', None)
    if split_info is not None and domain is not None:
        try:
            if domain == 'day' and 'train_days_idx' in split_info:
                return max(1, len(split_info['train_days_idx']))
            if domain == 'rx_day' and 'train_days_idx' in split_info and ckpt_args is not None:
                ds_tmp = load_wisig_compact_pkl(ckpt_args.get('wisig_pkl', './Dataset_WigSig/ManySig.pkl'))
                n_rx = len(ds_tmp.get('rx_list', []))
                if n_rx > 0:
                    return max(1, n_rx * len(split_info['train_days_idx']))
        except Exception:
            pass
    return 1


# =============================
# linear probe / NCM
# =============================
def one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    out = np.zeros((y.shape[0], num_classes), dtype=np.float32)
    out[np.arange(y.shape[0]), y.astype(np.int64)] = 1.0
    return out


def ridge_multiclass_train(X: np.ndarray, y: np.ndarray, num_classes: int, reg: float = 1e-2) -> np.ndarray:
    X = X.astype(np.float32)
    y = y.astype(np.int64)
    Xb = np.concatenate([X, np.ones((X.shape[0], 1), dtype=np.float32)], axis=1)
    Y = one_hot(y, num_classes)
    d = Xb.shape[1]
    A = Xb.T @ Xb + reg * np.eye(d, dtype=np.float32)
    B = Xb.T @ Y
    W = np.linalg.solve(A, B)
    return W


def ridge_multiclass_predict(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    Xb = np.concatenate([X.astype(np.float32), np.ones((X.shape[0], 1), dtype=np.float32)], axis=1)
    logits = Xb @ W
    return logits.argmax(axis=1).astype(np.int64)


def acc_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    return 100.0 * float((y_true == y_pred).mean())


def ncm_train(X: np.ndarray, y: np.ndarray) -> Dict[int, np.ndarray]:
    Xn = l2_normalize_np(X)
    cents = {}
    for c in np.unique(y):
        cents[int(c)] = l2_normalize_np(Xn[y == c].mean(axis=0, keepdims=True))[0]
    return cents


def ncm_predict(X: np.ndarray, cents: Dict[int, np.ndarray]) -> np.ndarray:
    Xn = l2_normalize_np(X)
    keys = sorted(cents.keys())
    C = np.stack([cents[k] for k in keys], axis=0)
    sim = Xn @ C.T
    idx = sim.argmax(axis=1)
    return np.asarray([keys[i] for i in idx], dtype=np.int64)


# =============================
# feature diagnostics
# =============================
def fisher_ratio(X: np.ndarray, y: np.ndarray) -> float:
    if X.shape[0] <= 1:
        return float("nan")
    mu = X.mean(axis=0, keepdims=True)
    sw = 0.0
    sb = 0.0
    for c in np.unique(y):
        Xc = X[y == c]
        muc = Xc.mean(axis=0, keepdims=True)
        sw += float(((Xc - muc) ** 2).sum())
        sb += float(Xc.shape[0] * ((muc - mu) ** 2).sum())
    return float(sb / max(sw, 1e-12))


def mean_same_tx_cross_domain_cos(X: np.ndarray, y: np.ndarray, d: Optional[np.ndarray]) -> float:
    if d is None:
        return float("nan")
    Xn = l2_normalize_np(X)
    vals = []
    for c in np.unique(y):
        mask_c = (y == c)
        doms = np.unique(d[mask_c])
        if doms.size < 2:
            continue
        cents = []
        for dom in doms:
            md = mask_c & (d == dom)
            if md.sum() == 0:
                continue
            cents.append(l2_normalize_np(Xn[md].mean(axis=0, keepdims=True))[0])
        if len(cents) < 2:
            continue
        C = np.stack(cents, axis=0)
        S = C @ C.T
        iu = np.triu_indices(S.shape[0], k=1)
        vals.extend(S[iu].tolist())
    return float(np.mean(vals)) if vals else float("nan")


def mean_train_test_class_centroid_cos(X_src: np.ndarray, y_src: np.ndarray, X_tgt: np.ndarray, y_tgt: np.ndarray) -> float:
    Xs = l2_normalize_np(X_src)
    Xt = l2_normalize_np(X_tgt)
    vals = []
    common = sorted(set(np.unique(y_src).tolist()) & set(np.unique(y_tgt).tolist()))
    for c in common:
        cs = l2_normalize_np(Xs[y_src == c].mean(axis=0, keepdims=True))[0]
        ct = l2_normalize_np(Xt[y_tgt == c].mean(axis=0, keepdims=True))[0]
        vals.append(float(np.dot(cs, ct)))
    return float(np.mean(vals)) if vals else float("nan")


def mean_clean_perturbed_cos(X_clean: np.ndarray, X_pert: np.ndarray) -> float:
    X1 = l2_normalize_np(X_clean)
    X2 = l2_normalize_np(X_pert)
    return float(np.mean(np.sum(X1 * X2, axis=1)))


def relative_shift(X_clean: np.ndarray, X_pert: np.ndarray) -> float:
    num = np.linalg.norm(X_pert - X_clean, axis=1)
    den = np.linalg.norm(X_clean, axis=1)
    return float(np.mean(num / np.maximum(den, 1e-12)))


def summarize_score(tx_test_ncm: float, dom_val: float, inv_cos: float, centroid_cos: float, num_classes: int, num_domains: int) -> float:
    tx_term = normalized_accuracy(tx_test_ncm, num_classes)
    dom_term = normalized_accuracy(dom_val, max(2, num_domains)) if not math.isnan(dom_val) else 0.0
    inv_term = 0.0 if math.isnan(inv_cos) else max(0.0, min(1.0, inv_cos))
    drift_term = 0.0 if math.isnan(centroid_cos) else max(0.0, min(1.0, centroid_cos))
    score = 100.0 * (0.45 * tx_term + 0.20 * inv_term + 0.20 * drift_term + 0.15 * (1.0 - dom_term))
    return float(score)


# =============================
# feature collection
# =============================
def collect_feature_dict(out: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    feats = {
        "z_id": out["z_id"],
        "z_dom": out["z_dom"],
    }
    aux_id = out.get("aux_id", {})
    aux_dom = out.get("aux_dom", {})

    id_keys = [
        "feat_cls", "feat_dac", "feat_pa", "feat_imp", "feat_joint", "feat_con",
        "base", "t_emb", "f_emb", "dac_local", "pa_local", "dac_stats", "pa_stats"
    ]
    dom_keys = [
        "feat_cls", "feat_dac", "feat_pa", "feat_imp", "feat_joint", "feat_con",
        "base", "t_emb", "f_emb", "dac_local", "pa_local", "dac_stats", "pa_stats"
    ]
    for k in id_keys:
        v = aux_id.get(k, None)
        if torch.is_tensor(v):
            feats[f"id_{k}"] = v
    for k in dom_keys:
        v = aux_dom.get(k, None)
        if torch.is_tensor(v):
            feats[f"dom_{k}"] = v
    return feats


@torch.no_grad()
def extract_split_features(model, loader, device, feature_names: List[str], max_batches: int = 0):
    model.eval()
    feat_buf = {k: [] for k in feature_names}
    y_buf, d_buf = [], []
    for bi, batch in enumerate(loader):
        x, y, extra = unpack_batch(batch)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        d = extract_domain_from_extra(extra, device)

        out = model(x, y_tx=None, grl_lambda=1.0, return_aux=True)
        feats = collect_feature_dict(out)

        for k in feature_names:
            if k not in feats:
                raise KeyError(f"Requested feature '{k}' not found. Available keys: {sorted(feats.keys())}")
            feat_buf[k].append(feats[k].detach().cpu().float())
        y_buf.append(y.detach().cpu().long())
        if d is not None:
            d_buf.append(d.detach().cpu().long())
        if max_batches > 0 and (bi + 1) >= max_batches:
            break

    feat_np = {k: torch.cat(v, dim=0).numpy() for k, v in feat_buf.items()}
    y_np = torch.cat(y_buf, dim=0).numpy()
    d_np = torch.cat(d_buf, dim=0).numpy() if len(d_buf) > 0 else None
    return feat_np, y_np, d_np


@torch.no_grad()
def extract_perturbed_features(model, loader, device, feature_names: List[str], perturb: str, strength: float, max_batches: int = 0):
    model.eval()
    augmentor = build_augmentor(p_dac=0.0, p_pa=0.0, p_time_shift=0.0, p_amp_scale=0.0, p_phase_rot=0.0,
                                p_cfo=0.0, p_phase_noise=0.0, p_awgn=0.0, p_multipath=0.0,
                                p_dc_offset=0.0, p_bandedge_taper=0.0, defect_apply_channel=False)

    feat_clean = {k: [] for k in feature_names}
    feat_pert = {k: [] for k in feature_names}
    y_buf = []
    d_buf = []

    for bi, batch in enumerate(loader):
        x, y, extra = unpack_batch(batch)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        d = extract_domain_from_extra(extra, device)

        out_clean = model(x, y_tx=None, grl_lambda=1.0, return_aux=True)
        feats_clean = collect_feature_dict(out_clean)

        B = x.size(0)
        s = torch.full((B, 1), float(strength), device=x.device, dtype=x.dtype)
        if perturb == "dac":
            x_pert, _ = augmentor.simulate_dac(x, labels=y, dac_strength=s)
        elif perturb == "pa":
            x_pert, _ = augmentor.simulate_pa(x, labels=y, pa_strength=s)
        else:
            raise ValueError("perturb must be 'dac' or 'pa'")

        out_pert = model(x_pert, y_tx=None, grl_lambda=1.0, return_aux=True)
        feats_pert = collect_feature_dict(out_pert)

        for k in feature_names:
            feat_clean[k].append(feats_clean[k].detach().cpu().float())
            feat_pert[k].append(feats_pert[k].detach().cpu().float())

        y_buf.append(y.detach().cpu().long())
        if d is not None:
            d_buf.append(d.detach().cpu().long())
        if max_batches > 0 and (bi + 1) >= max_batches:
            break

    clean_np = {k: torch.cat(v, dim=0).numpy() for k, v in feat_clean.items()}
    pert_np = {k: torch.cat(v, dim=0).numpy() for k, v in feat_pert.items()}
    y_np = torch.cat(y_buf, dim=0).numpy()
    d_np = torch.cat(d_buf, dim=0).numpy() if len(d_buf) > 0 else None
    return clean_np, pert_np, y_np, d_np


# =============================
# dataset/model builders
# =============================
def build_datasets_from_ckpt(ckpt_args: Dict[str, Any], cli_args):
    dataset = cli_args.dataset or ckpt_args.get("dataset", "wisig")
    if dataset == "wisig":
        pkl_path = cli_args.wisig_pkl or ckpt_args.get("wisig_pkl", "./Dataset_WigSig/ManySig.pkl")
        ds_w = load_wisig_compact_pkl(pkl_path)
        train_ds, val_ds, test_ds, split_info = make_day123_trainval_day4_test(
            ds_w,
            equalized=("both" if str(ckpt_args.get("wisig_equalized", "1")).lower() == "both" else int(ckpt_args.get("wisig_equalized", "1"))),
            out_len=int(cli_args.wisig_out_len or ckpt_args.get("wisig_out_len", 256)),
            domain=str(cli_args.wisig_domain or ckpt_args.get("wisig_domain", "day")),
            normalize=True,
            crop_mode="center",
            transform_train=None,
            transform_eval=None,
            train_ratio=float(ckpt_args.get("wisig_train_ratio", 0.8)),
            guard_gap=int(ckpt_args.get("wisig_guard_gap", 8)),
            train_days=parse_days(ckpt_args.get("wisig_train_days", "0,1,2")),
            test_days=parse_days(ckpt_args.get("wisig_test_days", "3")),
            max_samples_per_combo_day123=(None if int(ckpt_args.get("wisig_max_day123_per_combo", 0)) <= 0 else int(ckpt_args.get("wisig_max_day123_per_combo", 0))),
            max_samples_per_combo_test=(None if int(ckpt_args.get("wisig_max_test_per_combo", 0)) <= 0 else int(ckpt_args.get("wisig_max_test_per_combo", 0))),
            max_samples_per_combo_train=(None if int(ckpt_args.get("wisig_max_train_per_combo", 0)) <= 0 else int(ckpt_args.get("wisig_max_train_per_combo", 0))),
            max_samples_per_combo_val=(None if int(ckpt_args.get("wisig_max_val_per_combo", 0)) <= 0 else int(ckpt_args.get("wisig_max_val_per_combo", 0))),
            seed=int(ckpt_args.get("seed", 1337)),
        )

        # Optional evaluation-time filtering for WiSig.
        # This keeps train/val aligned with the checkpoint split, and lets the user explicitly choose
        # which days / RXs are used for TEST in evaluation.
        eval_test_days = parse_items(getattr(cli_args, 'eval_test_days', None))
        eval_test_rxs = parse_items(getattr(cli_args, 'eval_test_rxs', None))
        if eval_test_days is not None or eval_test_rxs is not None:
            test_ds, extra_info = build_wisig_explicit_test_dataset(ds_w, ckpt_args, cli_args)
            split_info = dict(split_info)
            split_info.update(extra_info)
        input_len = int(cli_args.wisig_out_len or ckpt_args.get("wisig_out_len", 256))
        return dataset, train_ds, val_ds, test_ds, split_info, input_len

    train_ds = WiFiRFFIDataset(
        cli_args.dataset_dir or ckpt_args.get("dataset_dir", "./Dataset_ORALCE"),
        mode="train",
        run_name=cli_args.run_name or ckpt_args.get("run_name", "run1"),
    )
    test_ds = WiFiRFFIDataset(
        cli_args.dataset_dir or ckpt_args.get("dataset_dir", "./Dataset_ORALCE"),
        mode="test",
        run_name=cli_args.run_name or ckpt_args.get("run_name", "run1"),
    )
    val_ds = test_ds
    try:
        x0, _ = train_ds[0]
        input_len = int(x0.shape[-1])
    except Exception:
        input_len = 1024
    return dataset, train_ds, val_ds, test_ds, {"mode": "oralce_val_equals_test"}, input_len


def build_model_from_ckpt(ckpt_args: Dict[str, Any], cli_args, num_domains: int, input_len: int, device):
    dataset = cli_args.dataset or ckpt_args.get("dataset", "wisig")
    num_classes = int(cli_args.num_classes or ckpt_args.get("num_classes", 16))
    model_size = str(cli_args.model_size or ckpt_args.get("model_size", "M"))
    model_variant = str(getattr(cli_args, "model_variant", None) or ckpt_args.get("model_variant", "base"))
    branch_ablation = str(getattr(cli_args, "branch_ablation", None) or ckpt_args.get("branch_ablation", "none"))
    sample_rate_hz = float(cli_args.sample_rate_hz or ckpt_args.get("sample_rate_hz", 25e6 if dataset == "wisig" else 5e6))

    model = build_dual_model(
        num_classes=num_classes,
        num_domains=num_domains,
        model_size=model_size,
        dataset=dataset,
        input_len=input_len,
        sample_rate_hz=sample_rate_hz,
        model_variant=model_variant,
        branch_ablation=branch_ablation,
    ).to(device)
    return model


def load_state_dict_safely(model: torch.nn.Module, state: Dict[str, torch.Tensor]):
    model_state = model.state_dict()
    filtered = {}
    unexpected = []
    skipped_mismatch = []

    for k, v in state.items():
        if k not in model_state:
            unexpected.append(k)
            continue
        if tuple(model_state[k].shape) != tuple(v.shape):
            skipped_mismatch.append((k, tuple(v.shape), tuple(model_state[k].shape)))
            continue
        filtered[k] = v

    missing = sorted(set(model_state.keys()) - set(filtered.keys()))
    model.load_state_dict(filtered, strict=False)
    return missing, unexpected, skipped_mismatch


# =============================
# main diagnosis
# =============================
def diagnose_feature_set(feature_name: str,
                         train_feats: np.ndarray,
                         val_feats: np.ndarray,
                         test_feats: np.ndarray,
                         y_train: np.ndarray,
                         y_val: np.ndarray,
                         y_test: np.ndarray,
                         d_train: Optional[np.ndarray],
                         d_val: Optional[np.ndarray],
                         clean_test_feats: Optional[np.ndarray],
                         dac_test_feats: Optional[np.ndarray],
                         pa_test_feats: Optional[np.ndarray],
                         num_classes: int,
                         num_domains: int) -> Dict[str, Any]:
    result = {"feature": feature_name}

    # linear probe / NCM
    W_tx = ridge_multiclass_train(train_feats, y_train, num_classes=num_classes, reg=1e-2)
    result["tx_ridge_val_acc"] = acc_score(y_val, ridge_multiclass_predict(val_feats, W_tx))
    result["tx_ridge_test_acc"] = acc_score(y_test, ridge_multiclass_predict(test_feats, W_tx))

    cents = ncm_train(train_feats, y_train)
    result["tx_ncm_val_acc"] = acc_score(y_val, ncm_predict(val_feats, cents))
    result["tx_ncm_test_acc"] = acc_score(y_test, ncm_predict(test_feats, cents))

    # class separability / invariance
    result["fisher_train"] = fisher_ratio(train_feats, y_train)
    result["fisher_test"] = fisher_ratio(test_feats, y_test)
    result["src_same_tx_cross_dom_cos"] = mean_same_tx_cross_domain_cos(train_feats, y_train, d_train)
    result["train_test_centroid_cos"] = mean_train_test_class_centroid_cos(train_feats, y_train, test_feats, y_test)

    # known-domain probe on source-style val only
    if d_train is not None and d_val is not None and num_domains > 1:
        seen = np.unique(d_train)
        tr_mask = np.isin(d_train, seen)
        va_mask = np.isin(d_val, seen)
        if tr_mask.sum() > 0 and va_mask.sum() > 0:
            d_map = {int(v): i for i, v in enumerate(sorted(seen.tolist()))}
            yd_train = np.asarray([d_map[int(v)] for v in d_train[tr_mask]], dtype=np.int64)
            yd_val = np.asarray([d_map[int(v)] for v in d_val[va_mask]], dtype=np.int64)
            W_dom = ridge_multiclass_train(train_feats[tr_mask], yd_train, num_classes=len(d_map), reg=1e-2)
            result["dom_ridge_val_acc"] = acc_score(yd_val, ridge_multiclass_predict(val_feats[va_mask], W_dom))
        else:
            result["dom_ridge_val_acc"] = float("nan")
    else:
        result["dom_ridge_val_acc"] = float("nan")

    # perturbation sensitivity / robustness
    if clean_test_feats is not None and dac_test_feats is not None:
        result["clean_dac_cos"] = mean_clean_perturbed_cos(clean_test_feats, dac_test_feats)
        result["clean_dac_rel_shift"] = relative_shift(clean_test_feats, dac_test_feats)
        result["tx_ncm_test_under_dac_acc"] = acc_score(y_test, ncm_predict(dac_test_feats, cents))
    else:
        result["clean_dac_cos"] = float("nan")
        result["clean_dac_rel_shift"] = float("nan")
        result["tx_ncm_test_under_dac_acc"] = float("nan")

    if clean_test_feats is not None and pa_test_feats is not None:
        result["clean_pa_cos"] = mean_clean_perturbed_cos(clean_test_feats, pa_test_feats)
        result["clean_pa_rel_shift"] = relative_shift(clean_test_feats, pa_test_feats)
        result["tx_ncm_test_under_pa_acc"] = acc_score(y_test, ncm_predict(pa_test_feats, cents))
    else:
        result["clean_pa_cos"] = float("nan")
        result["clean_pa_rel_shift"] = float("nan")
        result["tx_ncm_test_under_pa_acc"] = float("nan")

    if not math.isnan(result["clean_dac_rel_shift"]) and not math.isnan(result["clean_pa_rel_shift"]):
        result["dac_minus_pa_shift"] = result["clean_dac_rel_shift"] - result["clean_pa_rel_shift"]
    else:
        result["dac_minus_pa_shift"] = float("nan")

    inv_cos = np.nanmean([result.get("clean_dac_cos", np.nan), result.get("clean_pa_cos", np.nan)])
    result["heuristic_cross_domain_score"] = summarize_score(
        tx_test_ncm=result["tx_ncm_test_acc"],
        dom_val=result["dom_ridge_val_acc"],
        inv_cos=inv_cos,
        centroid_cos=result["train_test_centroid_cos"],
        num_classes=num_classes,
        num_domains=num_domains,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description="Diagnose which features help TX classification and cross-domain generalization.")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--dataset", type=str, default=None, choices=[None, "wisig", "oralce"], nargs="?")
    parser.add_argument("--dataset_dir", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--wisig_pkl", type=str, default=None)
    parser.add_argument("--wisig_domain", type=str, default=None)
    parser.add_argument("--wisig_out_len", type=int, default=None)
    parser.add_argument("--eval_test_days", type=str, default=None, help="Override WiSig test days for evaluation only, e.g. 3 or 0,1,2 or date labels")
    parser.add_argument("--eval_test_rxs", type=str, default=None, help="Override WiSig test RXs for evaluation only, e.g. 0,1,2 or rx labels like 1-1,1-19")
    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument("--num_domains", type=int, default=None)
    parser.add_argument("--model_size", type=str, default=None)
    parser.add_argument("--model_variant", type=str, default=None, choices=[None, "base", "lite_a", "lite_b", "lite_c"], nargs="?")
    parser.add_argument("--branch_ablation", type=str, default=None)
    parser.add_argument("--sample_rate_hz", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--max_train_samples", type=int, default=30000)
    parser.add_argument("--max_val_samples", type=int, default=15000)
    parser.add_argument("--max_test_samples", type=int, default=15000)
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--perturb_strength", type=float, default=0.6)
    parser.add_argument("--feature_names", type=str,
                        default="z_id,z_dom,id_feat_cls,id_feat_joint,id_feat_dac,id_feat_pa,id_feat_imp,dom_feat_imp,dom_feat_pa,dom_feat_dac")
    parser.add_argument("--out_dir", type=str, default="./feature_diagnosis_out")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    if "args" not in ckpt:
        raise KeyError("Checkpoint must contain saved training args under key 'args'.")
    ckpt_args = ckpt["args"]

    dataset_name, train_ds_full, val_ds_full, test_ds_full, split_info, input_len = build_datasets_from_ckpt(ckpt_args, args)
    state = strip_module_prefix(ckpt["model"])
    num_domains = infer_num_domains(
        train_ds_full,
        state=state,
        split_info=split_info,
        ckpt_args=ckpt_args,
        cli_num_domains=args.num_domains,
    )

    train_ds = maybe_subset_dataset(train_ds_full, args.max_train_samples, args.seed + 1)
    val_ds = maybe_subset_dataset(val_ds_full, args.max_val_samples, args.seed + 2)
    test_ds = maybe_subset_dataset(test_ds_full, args.max_test_samples, args.seed + 3)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)

    model = build_model_from_ckpt(ckpt_args, args, num_domains=num_domains, input_len=input_len, device=device)
    missing, unexpected, skipped_mismatch = load_state_dict_safely(model, state)
    model.eval()

    feature_names = [s.strip() for s in args.feature_names.split(",") if s.strip()]

    print(f"[INFO] dataset={dataset_name} input_len={input_len} num_domains(train)={num_domains}")
    print(f"[INFO] dataset_sizes(full/subset): train={len(train_ds_full)}/{len(train_ds)} val={len(val_ds_full)}/{len(val_ds)} test={len(test_ds_full)}/{len(test_ds)}")
    print(f"[INFO] missing_keys={len(missing)} unexpected_keys={len(unexpected)} skipped_mismatch={len(skipped_mismatch)}")
    if missing:
        print(f"[INFO] missing sample: {missing[:8]}")
    if unexpected:
        print(f"[INFO] unexpected sample: {unexpected[:8]}")
    if skipped_mismatch:
        print(f"[INFO] skipped mismatch sample: {skipped_mismatch[:6]}")
    print(f"[INFO] features={feature_names}")
    print(f"[INFO] split_info={split_info}")

    train_feat, y_train, d_train = extract_split_features(model, train_loader, device, feature_names, max_batches=args.max_batches)
    val_feat, y_val, d_val = extract_split_features(model, val_loader, device, feature_names, max_batches=args.max_batches)
    test_feat, y_test, d_test = extract_split_features(model, test_loader, device, feature_names, max_batches=args.max_batches)

    clean_test_feat, dac_test_feat, _, _ = extract_perturbed_features(model, test_loader, device, feature_names, perturb="dac", strength=args.perturb_strength, max_batches=args.max_batches)
    _, pa_test_feat, _, _ = extract_perturbed_features(model, test_loader, device, feature_names, perturb="pa", strength=args.perturb_strength, max_batches=args.max_batches)

    all_results = []
    for name in feature_names:
        res = diagnose_feature_set(
            feature_name=name,
            train_feats=train_feat[name],
            val_feats=val_feat[name],
            test_feats=test_feat[name],
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            d_train=d_train,
            d_val=d_val,
            clean_test_feats=clean_test_feat[name],
            dac_test_feats=dac_test_feat[name],
            pa_test_feats=pa_test_feat[name],
            num_classes=int(args.num_classes or ckpt_args.get("num_classes", 16)),
            num_domains=num_domains,
        )
        all_results.append(res)

    all_results.sort(key=lambda z: z["heuristic_cross_domain_score"], reverse=True)

    json_path = os.path.join(args.out_dir, "feature_diagnosis_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "checkpoint": args.ckpt,
            "dataset": dataset_name,
            "split_info": split_info,
            "feature_names": feature_names,
            "results": all_results,
        }, f, ensure_ascii=False, indent=2)

    csv_path = os.path.join(args.out_dir, "feature_diagnosis_summary.csv")
    keys = [
        "feature",
        "heuristic_cross_domain_score",
        "tx_ridge_val_acc", "tx_ridge_test_acc",
        "tx_ncm_val_acc", "tx_ncm_test_acc",
        "dom_ridge_val_acc",
        "fisher_train", "fisher_test",
        "src_same_tx_cross_dom_cos", "train_test_centroid_cos",
        "clean_dac_cos", "clean_dac_rel_shift", "tx_ncm_test_under_dac_acc",
        "clean_pa_cos", "clean_pa_rel_shift", "tx_ncm_test_under_pa_acc",
        "dac_minus_pa_shift",
    ]
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for row in all_results:
            vals = []
            for k in keys:
                v = row.get(k, "")
                if isinstance(v, float):
                    if math.isnan(v):
                        vals.append("nan")
                    else:
                        vals.append(f"{v:.6f}")
                else:
                    vals.append(str(v))
            f.write(",".join(vals) + "\n")

    print("\n===== Top features by heuristic_cross_domain_score =====")
    for i, row in enumerate(all_results[: min(8, len(all_results))], start=1):
        print(
            f"[{i}] {row['feature']:<18} | score={row['heuristic_cross_domain_score']:.2f} "
            f"| tx_test_ncm={row['tx_ncm_test_acc']:.2f}% | dom_val={row['dom_ridge_val_acc']:.2f}% "
            f"| centroid_cos={row['train_test_centroid_cos']:.4f} "
            f"| clean_dac_cos={row['clean_dac_cos']:.4f} | clean_pa_cos={row['clean_pa_cos']:.4f}"
        )

    print(f"\nSaved JSON: {json_path}")
    print(f"Saved CSV : {csv_path}")


if __name__ == "__main__":
    main()
