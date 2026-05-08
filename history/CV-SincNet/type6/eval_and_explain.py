# eval_and_explain.py
# -*- coding: utf-8 -*-
"""
WiFi RFFI: evaluate CV-SincNet weights and generate analysis outputs.
- Confusion matrix + classification report (NO sklearn required)
- Optional confusion-matrix image (PNG if matplotlib exists, else SVG)
- Optional branch-only eval (time-only / freq-only) if model has t_proj/f_proj/fuse/cls_head
- Saliency (time + freq + mirror asymmetry) (NO matplotlib required; if available, also save PNG)
- NEW: Clustering distribution plots (PCA-2D + KMeans) (PNG if matplotlib exists, else SVG)

Run:
  python eval_and_explain.py
or:
  python eval_and_explain.py --weights ./weight/xxx.pth --data ./Dataset_ORALCE --out ./eval_out
"""

import os
import argparse
import inspect
from typing import Dict, Tuple, Optional, Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import WiFiRFFIDataset
from model import CVSincNet


# ===================== 修改区（你只需要改这几项） =====================
WEIGHTS_PATH = "./best_model.pth"   # ✅ 权重地址
DATASET_DIR  = "./Dataset_ORALCE"                            # ✅ 数据集根目录
OUT_DIR      = "./eval_out"
BATCH_SIZE   = 128
NUM_WORKERS  = 4
MAX_SALIENCY_SAMPLES = 4096
SAMPLE_RATE_HZ = 5e6

# NEW: 聚类分布图参数
MAX_CLUSTER_SAMPLES = 6000      # 建议 2k~10k，太大会很慢/文件很大
CLUSTER_K = 16                 # 默认=类别数（可命令行覆盖）
CLUSTER_SEED = 0
KMEANS_ITERS = 30
# =====================================================================


# -------- optional deps (matplotlib) --------
_HAS_MPL = False
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False


# -------------------- small utilities --------------------
def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def _strip_module_prefix(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not any(k.startswith("module.") for k in sd.keys()):
        return sd
    return {k.replace("module.", "", 1): v for k, v in sd.items()}


def _load_state_dict(weights_path: str) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(weights_path, map_location="cpu")

    if isinstance(ckpt, dict):
        for key in ["state_dict", "model", "model_state", "net", "weights"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                sd = ckpt[key]
                return _strip_module_prefix(sd)

        # pure state_dict
        if any(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return _strip_module_prefix(ckpt)

    raise ValueError(f"Unrecognized checkpoint format: {weights_path}")


def _find_key(sd: Dict[str, torch.Tensor], candidates) -> Optional[str]:
    for k in candidates:
        if k in sd:
            return k
    return None


def _find_head_weight_key(sd: Dict[str, torch.Tensor]) -> Optional[str]:
    """Find the classification head weight key in a checkpoint state_dict.

    This repo has used multiple head naming conventions over time:
      - Linear head:        cls_head.weight / classifier.weight / head.weight
      - CosFace head:       cls_head.head.weight / classifier.head.weight
    We use this key to infer (num_classes, emb_dim) and to sanity-check loading.
    """
    # Most common / preferred candidates first
    candidates = [
        "cls_head.head.weight",
        "classifier.head.weight",
        "cls_head.weight",
        "classifier.weight",
        "head.weight",
    ]
    k = _find_key(sd, candidates)
    if k is not None:
        return k

    # Fallback: search any reasonable 2D weight that looks like a head
    for kk, vv in sd.items():
        try:
            nd = vv.ndim
        except Exception:
            continue
        if nd != 2:
            continue
        if not kk.endswith("weight"):
            continue
        # heuristics: contains head-ish path
        if ("cls_head" in kk or "classifier" in kk or kk.split(".")[0] in ("head", "fc", "linear")) and (
            ".head." in kk or kk.endswith(".weight")
        ):
            return kk

    return None



def _infer_model_hparams(sd: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """
    从 checkpoint 反推模型超参（以 checkpoint 为准）

    注意：你的 model.py 使用的是 DACAwareClassifier + CosFaceHead，
    因此分类头权重 key 通常是 `cls_head.head.weight`，而不是 `cls_head.weight`。
    本函数会兼容多种历史命名，避免 eval 因 KeyError 直接中断。
    """
    head_w_key = _find_head_weight_key(sd)
    if head_w_key is None:
        raise KeyError(
            "Cannot find head weight in checkpoint. Tried keys like "
            "`cls_head.head.weight`, `cls_head.weight`, `classifier.weight`, `head.weight`."
        )

    # head weight: (num_classes, emb_dim)
    if sd[head_w_key].ndim != 2:
        raise KeyError(f"Found head key={head_w_key} but it is not a 2D weight tensor: shape={tuple(sd[head_w_key].shape)}")

    num_classes = int(sd[head_w_key].shape[0])
    emb_dim = int(sd[head_w_key].shape[1])

    sinc_out_key = _find_key(sd, ["sinc.low_hz_", "sinc.band_hz_", "sinc.low_hz", "sinc.band_hz"])
    if sinc_out_key is None:
        raise KeyError("Cannot find sinc.low_hz_ / sinc.band_hz_ in checkpoint to infer sinc_out.")
    sinc_out = int(sd[sinc_out_key].shape[0])

    tf_key = _find_key(sd, ["time_fuse.0.weight", "time_fuse.weight"])
    if tf_key is None:
        for k, v in sd.items():
            if "time_fuse" in k and k.endswith("weight") and v.ndim == 3 and v.shape[-1] == 1 and v.shape[0] == v.shape[1]:
                tf_key = k
                break
    if tf_key is None:
        raise KeyError("Cannot find time_fuse conv weight in checkpoint.")
    time_in = int(sd[tf_key].shape[1])

    sinc_kernel = None
    for k in ["sinc.window_", "sinc.t_", "sinc.kernel_", "sinc.filters_"]:
        if k in sd and hasattr(sd[k], "ndim") and sd[k].ndim >= 1:
            sinc_kernel = int(sd[k].shape[-1])
            break
    if sinc_kernel is None:
        sinc_kernel = 129

    extra = 4
    basis_factor = (time_in - extra) / max(1, sinc_out)

    return {
        "num_classes": num_classes,
        "emb_dim": emb_dim,
        "sinc_out": sinc_out,
        "sinc_kernel": sinc_kernel,
        "time_in": time_in,
        "basis_factor_x": float(basis_factor),
        "head_w_key": head_w_key,
    }



def _filter_kwargs_for_init(cls, kwargs: Dict) -> Dict:
    sig = inspect.signature(cls.__init__)
    accepted = set(sig.parameters.keys()) - {"self"}
    return {k: v for k, v in kwargs.items() if k in accepted}


def _align_state_dict_keys(sd: Dict[str, torch.Tensor], model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """Align checkpoint keys to current model keys for common head naming differences."""
    mkeys = set(model.state_dict().keys())
    sd2 = dict(sd)

    # ---- Linear head: cls_head <-> classifier ----
    if "cls_head.weight" in sd2 and "cls_head.weight" not in mkeys and "classifier.weight" in mkeys:
        sd2["classifier.weight"] = sd2.pop("cls_head.weight")
        if "cls_head.bias" in sd2 and "classifier.bias" in mkeys:
            sd2["classifier.bias"] = sd2.pop("cls_head.bias")

    if "classifier.weight" in sd2 and "classifier.weight" not in mkeys and "cls_head.weight" in mkeys:
        sd2["cls_head.weight"] = sd2.pop("classifier.weight")
        if "classifier.bias" in sd2 and "cls_head.bias" in mkeys:
            sd2["cls_head.bias"] = sd2.pop("classifier.bias")

    # ---- CosFace head: cls_head.head <-> classifier.head ----
    if "cls_head.head.weight" in sd2 and "cls_head.head.weight" not in mkeys and "classifier.head.weight" in mkeys:
        sd2["classifier.head.weight"] = sd2.pop("cls_head.head.weight")

    if "classifier.head.weight" in sd2 and "classifier.head.weight" not in mkeys and "cls_head.head.weight" in mkeys:
        sd2["cls_head.head.weight"] = sd2.pop("classifier.head.weight")

    return sd2



def build_model_from_weights(weights_path: str, device: torch.device) -> CVSincNet:
    sd = _load_state_dict(weights_path)
    hp = _infer_model_hparams(sd)

    candidates = {
        "num_classes": hp["num_classes"],
        "n_classes": hp["num_classes"],
        "classes": hp["num_classes"],

        "emb_dim": hp["emb_dim"],
        "embed_dim": hp["emb_dim"],

        "sinc_out": hp["sinc_out"],
        "sinc_channels": hp["sinc_out"],

        "sinc_kernel": hp["sinc_kernel"] if hp["sinc_kernel"] > 0 else None,
        "kernel_size": hp["sinc_kernel"] if hp["sinc_kernel"] > 0 else None,

        "sample_rate": SAMPLE_RATE_HZ,
        "fs": SAMPLE_RATE_HZ,
    }
    candidates = {k: v for k, v in candidates.items() if v is not None}
    init_kwargs = _filter_kwargs_for_init(CVSincNet, candidates)

    print(
        f"[MODEL] inferred num_classes={hp['num_classes']} emb_dim={hp['emb_dim']} "
        f"sinc_out={hp['sinc_out']} sinc_kernel={hp['sinc_kernel']} time_in={hp['time_in']} "
        f"basis_factor≈{hp['basis_factor_x']:.2f}",
        flush=True,
    )

    model = CVSincNet(**init_kwargs)

    sd = _align_state_dict_keys(sd, model)

    # 安全加载：只加载 key 存在且 shape 一致的权重
    model_sd = model.state_dict()
    filtered = {}
    skipped = []
    for k, v in sd.items():
        if k in model_sd and tuple(v.shape) == tuple(model_sd[k].shape):
            filtered[k] = v
        elif k in model_sd:
            skipped.append((k, tuple(v.shape), tuple(model_sd[k].shape)))

    # sanity check: ensure classification head weights are loaded if the model has a head
    head_keys_in_model = [k for k in [
        "cls_head.head.weight",
        "classifier.head.weight",
        "cls_head.weight",
        "classifier.weight",
        "head.weight",
    ] if k in model_sd]

    if head_keys_in_model:
        ok = any(k in filtered for k in head_keys_in_model)
        if not ok:
            shapes = {k: tuple(model_sd[k].shape) for k in head_keys_in_model}
            raise RuntimeError(
                "Head weight not loaded. "
                f"Detected head keys in model: {head_keys_in_model} with shapes={shapes}. "
                "This usually means checkpoint and model head naming / shape mismatch."
            )

    missing, unexpected = model.load_state_dict(filtered, strict=False)

    print(f"[LOAD] loaded={len(filtered)} keys, skipped_shape_mismatch={len(skipped)}", flush=True)
    if skipped:
        print("[LOAD] first 10 skipped keys (ckpt_shape -> model_shape):", flush=True)
        for item in skipped[:10]:
            print("   ", item, flush=True)

    if missing or unexpected:
        print(f"[WARN] load_state_dict missing={missing} unexpected={unexpected}", flush=True)

    model.to(device)
    model.eval()

    print(
        f"[MODEL] inferred num_classes={hp['num_classes']} emb_dim={hp['emb_dim']} "
        f"sinc_out={hp['sinc_out']} time_in={hp['time_in']}",
        flush=True,
    )
    return model


def make_test_loader(data_dir: str, batch_size: int, num_workers: int, device: torch.device) -> DataLoader:
    ds = WiFiRFFIDataset(data_dir, mode="test")
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    return dl


# -------------------- metrics (numpy only) --------------------
def confusion_matrix_np(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def classification_report_np(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> str:
    eps = 1e-12
    lines = []
    header = f"{'class':>8} {'precision':>10} {'recall':>10} {'f1':>10} {'support':>10}"
    lines.append(header)
    lines.append("-" * len(header))

    supports = []
    precisions = []
    recalls = []
    f1s = []

    for c in range(num_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        sup = np.sum(y_true == c)

        prec = tp / (tp + fp + eps)
        rec  = tp / (tp + fn + eps)
        f1   = 2 * prec * rec / (prec + rec + eps)

        supports.append(sup)
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

        lines.append(f"{c:>8d} {prec:>10.4f} {rec:>10.4f} {f1:>10.4f} {sup:>10d}")

    supports = np.array(supports, dtype=np.float64)
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    f1s = np.array(f1s)
    total = supports.sum() + eps

    acc = float(np.mean(y_true == y_pred))
    macro = (precisions.mean(), recalls.mean(), f1s.mean())
    weighted = (
        float((precisions * supports).sum() / total),
        float((recalls * supports).sum() / total),
        float((f1s * supports).sum() / total),
    )

    lines.append("")
    lines.append(f"accuracy: {acc:.4f}")
    lines.append(f"macro avg: precision={macro[0]:.4f} recall={macro[1]:.4f} f1={macro[2]:.4f}")
    lines.append(f"weighted avg: precision={weighted[0]:.4f} recall={weighted[1]:.4f} f1={weighted[2]:.4f}")
    return "\n".join(lines)



def _extract_logits(out: Any) -> torch.Tensor:
    """Robustly extract logits tensor from common model output conventions."""
    if torch.is_tensor(out):
        return out
    if isinstance(out, dict):
        for k in ("logits", "pred", "y_pred", "out"):
            if k in out and torch.is_tensor(out[k]):
                return out[k]
        raise RuntimeError(f"Dict output has no logits tensor. Keys={list(out.keys())}")
    if isinstance(out, (tuple, list)):
        if len(out) == 0:
            raise RuntimeError("Empty tuple/list output from model.")
        if torch.is_tensor(out[0]):
            return out[0]
        raise RuntimeError(f"Tuple/list output[0] is not a tensor: type(out[0])={type(out[0])}")
    raise RuntimeError(f"Unsupported model output type: {type(out)}")


# -------------------- evaluation --------------------
@torch.no_grad()
def eval_full(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    y_true, y_pred = [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        out = model(xb)
        logits = _extract_logits(out)
        pred = logits.argmax(dim=1)
        y_true.append(yb.detach().cpu().numpy())
        y_pred.append(pred.detach().cpu().numpy())
    return np.concatenate(y_true), np.concatenate(y_pred)


def _get_module(model: torch.nn.Module, names) -> Optional[torch.nn.Module]:
    for n in names:
        if hasattr(model, n):
            return getattr(model, n)
    return None


def _get_embeddings(model: torch.nn.Module, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    cache = {}
    t_proj = _get_module(model, ["t_proj", "time_proj", "proj_t"])
    f_proj = _get_module(model, ["f_proj", "freq_proj", "proj_f"])
    if t_proj is None or f_proj is None:
        raise AttributeError("Model has no t_proj/f_proj (cannot do branch-only eval).")

    def hook_t(_m, _inp, out): cache["t"] = out
    def hook_f(_m, _inp, out): cache["f"] = out

    ht = t_proj.register_forward_hook(hook_t)
    hf = f_proj.register_forward_hook(hook_f)
    _ = model(x)
    ht.remove(); hf.remove()

    return cache["t"], cache["f"]


@torch.no_grad()
def eval_branch_only(model: torch.nn.Module, loader: DataLoader, device: torch.device, branch: str) -> Tuple[np.ndarray, np.ndarray]:
    assert branch in ["time", "freq"]
    fuse = _get_module(model, ["fuse", "fusion", "feat_fuse"])
    head = _get_module(model, ["cls_head", "classifier", "head"])
    if fuse is None or head is None:
        raise AttributeError("Model has no fuse/head (cannot do branch-only eval).")

    y_true, y_pred = [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        # get t_emb / f_emb from forward hooks
        t_emb, f_emb = _get_embeddings(model, xb)
        if branch == "time":
            f_emb = torch.zeros_like(f_emb)
        else:
            t_emb = torch.zeros_like(t_emb)

        # some models expect an extra rho feature (circularity coefficient) in fuse input
        rho = None
        if getattr(model, "use_circularity", False) and hasattr(model, "_mirror_compressed_features"):
            try:
                _feat_f, rho = model._mirror_compressed_features(xb)
            except Exception:
                rho = None

        if rho is not None and torch.is_tensor(rho):
            base_in = torch.cat([t_emb, f_emb, rho], dim=1)
        else:
            base_in = torch.cat([t_emb, f_emb], dim=1)

        feat = fuse(base_in)

        out = head(feat)  # could be logits or (logits, dac_pred, ...)
        logits = _extract_logits(out)
        pred = logits.argmax(dim=1)

        y_true.append(yb.detach().cpu().numpy())
        y_pred.append(pred.detach().cpu().numpy())

    return np.concatenate(y_true), np.concatenate(y_pred)


# -------------------- plotting helpers (PNG if mpl else SVG) --------------------
_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
    "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#393b79", "#637939",
    "#8c6d31", "#8DFF582A", "#7b4173", "#3182bd", "#31a354", "#756bb1",
    "#636363", "#e6550d", "#969696", "#9c9ede", "#cedb9c", "#F2944313"
]


def _svg_escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def save_confusion_matrix_image(cm: np.ndarray, out_path_base: str, title: str = "Confusion Matrix"):
    """
    Save confusion matrix heatmap:
    - if matplotlib available -> PNG
    - else -> SVG
    """
    n = cm.shape[0]
    cm_max = float(cm.max()) if cm.size else 1.0
    if cm_max <= 0:
        cm_max = 1.0

    if _HAS_MPL:
        plt.figure(figsize=(9, 8))
        plt.imshow(cm, aspect="auto")
        plt.title(title)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.colorbar()
        step = max(1, n // 16)
        ticks = np.arange(0, n, step)
        plt.xticks(ticks, ticks)
        plt.yticks(ticks, ticks)
        plt.tight_layout()
        plt.savefig(out_path_base + ".png", dpi=200)
        plt.close()
        return

    # SVG fallback
    W, H = 900, 820
    pad_l, pad_t, pad_r, pad_b = 80, 60, 40, 60
    grid_w = W - pad_l - pad_r
    grid_h = H - pad_t - pad_b
    cell_w = grid_w / n
    cell_h = grid_h / n

    def gray(v):
        # map 0..cm_max -> 255..0
        t = max(0.0, min(1.0, v / cm_max))
        g = int(round(255 * (1 - t)))
        return f"rgb({g},{g},{g})"

    parts = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}">')
    parts.append(f'<rect x="0" y="0" width="{W}" height="{H}" fill="white"/>')
    parts.append(f'<text x="{W/2}" y="30" text-anchor="middle" font-size="18" font-family="Arial">{_svg_escape(title)}</text>')

    # cells
    for i in range(n):
        for j in range(n):
            x = pad_l + j * cell_w
            y = pad_t + i * cell_h
            parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{cell_w:.2f}" height="{cell_h:.2f}" fill="{gray(float(cm[i,j]))}" stroke="white" stroke-width="0.5"/>')

    parts.append(f'<text x="{W/2}" y="{H-20}" text-anchor="middle" font-size="14" font-family="Arial">Predicted</text>')
    parts.append(f'<text x="20" y="{H/2}" text-anchor="middle" font-size="14" font-family="Arial" transform="rotate(-90 20 {H/2})">True</text>')
    parts.append("</svg>")

    with open(out_path_base + ".svg", "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


def save_scatter_pca_svg(xy: np.ndarray, color_idx: np.ndarray, out_svg: str, title: str):
    """
    Minimal SVG scatter (no external deps).
    xy: (N,2)
    color_idx: (N,) integers -> palette
    """
    W, H = 1100, 700
    pad = 60

    x = xy[:, 0]
    y = xy[:, 1]
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    if xmax - xmin < 1e-12:
        xmax = xmin + 1.0
    if ymax - ymin < 1e-12:
        ymax = ymin + 1.0

    def tx(v):
        return pad + (v - xmin) / (xmax - xmin) * (W - 2 * pad)

    def ty(v):
        # invert y for screen
        return H - pad - (v - ymin) / (ymax - ymin) * (H - 2 * pad)

    parts = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}">')
    parts.append(f'<rect x="0" y="0" width="{W}" height="{H}" fill="white"/>')
    parts.append(f'<text x="{W/2}" y="30" text-anchor="middle" font-size="18" font-family="Arial">{_svg_escape(title)}</text>')
    parts.append(f'<rect x="{pad}" y="{pad}" width="{W-2*pad}" height="{H-2*pad}" fill="none" stroke="#333" stroke-width="1"/>')

    # points
    N = xy.shape[0]
    r = 2.0 if N <= 6000 else 1.5
    op = 0.55 if N <= 6000 else 0.35
    for i in range(N):
        c = _PALETTE[int(color_idx[i]) % len(_PALETTE)]
        parts.append(f'<circle cx="{tx(x[i]):.2f}" cy="{ty(y[i]):.2f}" r="{r}" fill="{c}" fill-opacity="{op}"/>')

    parts.append("</svg>")

    with open(out_svg, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


def save_scatter_pca_png(xy: np.ndarray, color_idx: np.ndarray, out_png: str, title: str):
    if not _HAS_MPL:
        return
    plt.figure(figsize=(11, 6))
    # simple palette mapping
    colors = [ _PALETTE[int(i) % len(_PALETTE)] for i in color_idx ]
    plt.scatter(xy[:, 0], xy[:, 1], s=6, c=colors, alpha=0.55, linewidths=0)
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# -------------------- saliency --------------------
def _pos_hz_axis(L: int, fs: float) -> np.ndarray:
    return np.arange(L // 2 + 1) * (fs / L)


def saliency_explain(model: torch.nn.Module, loader: DataLoader, device: torch.device, max_samples: int, out_dir: str):
    """
    保存：
      - time_importance_full.npy
      - freq_importance_fullfft_full.npy
      - mirror_asymmetry_full.npy
    如果 matplotlib 可用，再额外保存 png 图。
    """
    time_sum = None
    freq_sum = None
    seen = 0

    for xb, yb in loader:
        if seen >= max_samples:
            break

        xb = xb.to(device)
        yb = yb.to(device)

        take = min(xb.size(0), max_samples - seen)
        xb = xb[:take].contiguous().detach()
        yb = yb[:take].contiguous()

        xb.requires_grad_(True)
        model.zero_grad(set_to_none=True)

        logits = model(xb)
        sel = logits.gather(1, yb.view(-1, 1)).sum()
        sel.backward()

        g = xb.grad.detach()  # (B,2,L)
        g_abs = g.abs().sum(dim=1)               # (B,L)
        t_imp = g_abs.sum(dim=0).cpu().numpy()   # (L,)

        g_c = torch.complex(g[:, 0, :], g[:, 1, :])  # (B,L)
        G = torch.fft.fft(g_c, dim=-1)               # (B,L)
        f_imp = torch.abs(G).sum(dim=0).cpu().numpy()  # (L,)

        if time_sum is None:
            time_sum = t_imp
            freq_sum = f_imp
        else:
            time_sum += t_imp
            freq_sum += f_imp

        seen += take

    time_avg = time_sum / max(1, seen)
    freq_avg = freq_sum / max(1, seen)

    np.save(os.path.join(out_dir, "time_importance_full.npy"), time_avg)
    np.save(os.path.join(out_dir, "freq_importance_fullfft_full.npy"), freq_avg)

    L = len(freq_avg)
    ks = np.arange(1, L // 2)
    pos = freq_avg[ks]
    neg = freq_avg[L - ks]
    asym = np.abs(pos - neg) / (pos + neg + 1e-12)
    np.save(os.path.join(out_dir, "mirror_asymmetry_full.npy"), asym)

    print(f"[SAL] saved npy (samples_used={seen})", flush=True)

    if _HAS_MPL:
        # time plot
        plt.figure(figsize=(11, 3))
        plt.plot(time_avg)
        plt.title("Time importance (|dlogit/dx|) [full]")
        plt.xlabel("sample index"); plt.ylabel("avg importance")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "time_importance_full.png"), dpi=200)
        plt.close()

        # freq plot (pos)
        f_pos = freq_avg[: (L // 2 + 1)]
        hz_mhz = _pos_hz_axis(L, SAMPLE_RATE_HZ) / 1e6
        plt.figure(figsize=(11, 3))
        plt.plot(hz_mhz, f_pos)
        plt.title("Frequency importance (|FFT(grad)|) [full]")
        plt.xlabel("frequency (MHz)"); plt.ylabel("avg importance")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "freq_importance_full.png"), dpi=200)
        plt.close()

        # asym plot
        plt.figure(figsize=(11, 3))
        plt.plot(hz_mhz[1:L//2], asym)
        plt.title("Mirror-frequency asymmetry |pos-neg|/(pos+neg) [full]")
        plt.xlabel("frequency (MHz)"); plt.ylabel("asymmetry")
        plt.ylim(0, 1.0)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "mirror_asymmetry_full.png"), dpi=200)
        plt.close()
    else:
        print("[SAL] matplotlib not found -> skip png plotting (only .npy saved).", flush=True)


# -------------------- NEW: clustering distribution --------------------
def _pca_2d(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    PCA to 2D using SVD. Returns (coords2d, components2xD)
    """
    X = X.astype(np.float64, copy=False)
    Xc = X - X.mean(axis=0, keepdims=True)
    # SVD: Xc = U S Vt
    # components are rows of Vt
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    comps = Vt[:2, :]
    coords = Xc @ comps.T
    return coords.astype(np.float32), comps.astype(np.float32)


def _kmeans_np(X: np.ndarray, k: int, iters: int = 30, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple KMeans (L2) in numpy.
    Returns: (assignments N,), (centers kxD)
    """
    rng = np.random.default_rng(seed)
    N, D = X.shape
    k = int(max(1, min(k, N)))

    # init centers: choose random points
    idx = rng.choice(N, size=k, replace=False)
    centers = X[idx].copy()

    for _ in range(iters):
        # assign
        # dist^2 = ||x||^2 + ||c||^2 - 2 x·c
        x2 = np.sum(X * X, axis=1, keepdims=True)         # Nx1
        c2 = np.sum(centers * centers, axis=1)[None, :]   # 1xk
        dist = x2 + c2 - 2.0 * (X @ centers.T)            # Nxk
        a = np.argmin(dist, axis=1)

        # update
        new_centers = np.zeros_like(centers)
        counts = np.zeros((k,), dtype=np.int64)
        for i in range(N):
            new_centers[a[i]] += X[i]
            counts[a[i]] += 1
        for j in range(k):
            if counts[j] > 0:
                new_centers[j] /= counts[j]
            else:
                # re-init empty cluster
                new_centers[j] = X[rng.integers(0, N)]
        centers = new_centers

    # final assign
    x2 = np.sum(X * X, axis=1, keepdims=True)
    c2 = np.sum(centers * centers, axis=1)[None, :]
    dist = x2 + c2 - 2.0 * (X @ centers.T)
    a = np.argmin(dist, axis=1)
    return a.astype(np.int32), centers.astype(np.float32)


@torch.no_grad()
def _extract_features_logits_preds(
    model: torch.nn.Module, x: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Try to extract penultimate features:
    - if model has cls_head/classifier/head: capture its input via pre-hook as feature
    - else fallback feature = logits
    returns: (feat, logits, pred)
    """
    head = _get_module(model, ["cls_head", "classifier", "head"])
    cache = {}

    h = None
    if head is not None:
        def pre_hook(_m, inp):
            # inp is tuple, first is feature
            cache["feat"] = inp[0].detach()
        h = head.register_forward_pre_hook(pre_hook)

    logits = model(x)
    if h is not None:
        h.remove()

    feat = cache.get("feat", logits.detach())
    pred = logits.argmax(dim=1)
    return feat.detach(), logits.detach(), pred.detach()


def clustering_distribution(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    out_dir: str,
    max_samples: int,
    k: int,
    seed: int,
    iters: int,
):
    """
    Outputs:
      - cluster_features.npy (N,D)
      - cluster_pca2d.npy (N,2)
      - cluster_true.npy / cluster_pred.npy / cluster_id.npy
      - cluster_counts_true.csv (C x K)
      - cluster_summary.txt
      - cluster_pca_true.(png/svg)
      - cluster_pca_pred.(png/svg)
      - cluster_pca_cluster.(png/svg)
    """
    feats_list = []
    y_list = []
    p_list = []
    used = 0

    for xb, yb in loader:
        if used >= max_samples:
            break
        take = min(xb.size(0), max_samples - used)
        xb = xb[:take].to(device, non_blocking=True)
        yb = yb[:take].to(device, non_blocking=True)

        feat, _logits, pred = _extract_features_logits_preds(model, xb)

        feats_list.append(feat.cpu().numpy())
        y_list.append(yb.cpu().numpy())
        p_list.append(pred.cpu().numpy())
        used += take

    X = np.concatenate(feats_list, axis=0)
    y = np.concatenate(y_list, axis=0).astype(np.int64)
    p = np.concatenate(p_list, axis=0).astype(np.int64)

    # normalize features helps clustering
    Xn = X.astype(np.float32, copy=False)
    norm = np.linalg.norm(Xn, axis=1, keepdims=True) + 1e-12
    Xn = Xn / norm

    coords, comps = _pca_2d(Xn)
    cid, centers = _kmeans_np(Xn, k=int(k), iters=int(iters), seed=int(seed))

    np.save(os.path.join(out_dir, "cluster_features.npy"), Xn)
    np.save(os.path.join(out_dir, "cluster_pca2d.npy"), coords)
    np.save(os.path.join(out_dir, "cluster_true.npy"), y)
    np.save(os.path.join(out_dir, "cluster_pred.npy"), p)
    np.save(os.path.join(out_dir, "cluster_id.npy"), cid)
    np.save(os.path.join(out_dir, "cluster_pca_components.npy"), comps)
    np.save(os.path.join(out_dir, "cluster_centers.npy"), centers)

    C = int(max(y.max(), p.max()) + 1)
    K = int(max(cid.max() + 1, 1))

    # counts: C x K
    counts_true = np.zeros((C, K), dtype=np.int64)
    counts_pred = np.zeros((C, K), dtype=np.int64)
    np.add.at(counts_true, (y, cid), 1)
    np.add.at(counts_pred, (p, cid), 1)

    np.savetxt(os.path.join(out_dir, "cluster_counts_true.csv"), counts_true, fmt="%d", delimiter=",")
    np.savetxt(os.path.join(out_dir, "cluster_counts_pred.csv"), counts_pred, fmt="%d", delimiter=",")

    # cluster purity summary
    lines = []
    lines.append(f"samples_used={used}")
    lines.append(f"feature_dim={Xn.shape[1]}")
    lines.append(f"kmeans_k={K} iters={iters} seed={seed}")
    lines.append("")
    lines.append("Cluster summary (by TRUE label):")
    total_correct = 0
    for j in range(K):
        idx = np.where(cid == j)[0]
        if idx.size == 0:
            lines.append(f"  cluster {j:02d}: size=0")
            continue
        vals, cnts = np.unique(y[idx], return_counts=True)
        best_i = int(np.argmax(cnts))
        maj = int(vals[best_i])
        maj_cnt = int(cnts[best_i])
        purity = maj_cnt / max(1, idx.size)
        total_correct += maj_cnt
        lines.append(f"  cluster {j:02d}: size={idx.size:5d}  majority_true={maj:2d}  majority_cnt={maj_cnt:5d}  purity={purity:.3f}")
    overall_purity = total_correct / max(1, used)
    lines.append("")
    lines.append(f"Overall purity (sum majority / N): {overall_purity:.4f}")
    with open(os.path.join(out_dir, "cluster_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[CLUSTER] saved clustering outputs (samples_used={used}, K={K}, purity={overall_purity:.4f})", flush=True)

    # plots
    # true label plot
    if _HAS_MPL:
        save_scatter_pca_png(coords, y, os.path.join(out_dir, "cluster_pca_true.png"), "PCA(2D) colored by TRUE label")
        save_scatter_pca_png(coords, p, os.path.join(out_dir, "cluster_pca_pred.png"), "PCA(2D) colored by PRED label")
        save_scatter_pca_png(coords, cid, os.path.join(out_dir, "cluster_pca_cluster.png"), "PCA(2D) colored by KMeans cluster")
    else:
        save_scatter_pca_svg(coords, y, os.path.join(out_dir, "cluster_pca_true.svg"), "PCA(2D) colored by TRUE label")
        save_scatter_pca_svg(coords, p, os.path.join(out_dir, "cluster_pca_pred.svg"), "PCA(2D) colored by PRED label")
        save_scatter_pca_svg(coords, cid, os.path.join(out_dir, "cluster_pca_cluster.svg"), "PCA(2D) colored by KMeans cluster")
        print("[CLUSTER] matplotlib not found -> saved SVG plots instead of PNG.", flush=True)


# -------------------- main --------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=WEIGHTS_PATH)
    parser.add_argument("--data", type=str, default=DATASET_DIR)
    parser.add_argument("--out", type=str, default=OUT_DIR)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--max_saliency", type=int, default=MAX_SALIENCY_SAMPLES)

    # NEW: clustering args
    parser.add_argument("--max_cluster", type=int, default=MAX_CLUSTER_SAMPLES)
    parser.add_argument("--cluster_k", type=int, default=CLUSTER_K)
    parser.add_argument("--cluster_seed", type=int, default=CLUSTER_SEED)
    parser.add_argument("--kmeans_iters", type=int, default=KMEANS_ITERS)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _ensure_dir(args.out)

    print(f"[CFG] weights={args.weights}", flush=True)
    print(f"[CFG] data={args.data}", flush=True)
    print(f"[CFG] out={args.out}", flush=True)
    print(f"[CFG] device={device}", flush=True)

    model = build_model_from_weights(args.weights, device)
    test_loader = make_test_loader(args.data, args.batch_size, args.workers, device)

    # 1) Full evaluation
    y_true, y_pred = eval_full(model, test_loader, device)
    num_classes = int(max(y_true.max(), y_pred.max()) + 1)
    cm = confusion_matrix_np(y_true, y_pred, num_classes)
    acc = float((y_true == y_pred).mean() * 100.0)

    print(f"[FULL] Acc = {acc:.2f}%", flush=True)
    print("[FULL] Confusion matrix:\n", cm, flush=True)

    report = classification_report_np(y_true, y_pred, num_classes)
    print("[FULL] Classification report:\n", report, flush=True)

    np.save(os.path.join(args.out, "confusion_matrix_full.npy"), cm)
    np.savetxt(os.path.join(args.out, "confusion_matrix_full.txt"), cm, fmt="%d")
    with open(os.path.join(args.out, "classification_report_full.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    # Confusion matrix image (NEW)
    save_confusion_matrix_image(
        cm,
        out_path_base=os.path.join(args.out, "confusion_matrix_full"),
        title=f"Confusion Matrix (Acc={acc:.2f}%)"
    )

    # 2) Branch-only eval (optional)
    try:
        for branch in ["time", "freq"]:
            yt, yp = eval_branch_only(model, test_loader, device, branch=branch)
            nc = int(max(yt.max(), yp.max()) + 1)
            cm_b = confusion_matrix_np(yt, yp, nc)
            acc_b = float((yt == yp).mean() * 100.0)
            print(f"[{branch.upper()}-ONLY] Acc = {acc_b:.2f}%", flush=True)

            np.save(os.path.join(args.out, f"confusion_{branch}_only.npy"), cm_b)
            np.savetxt(os.path.join(args.out, f"confusion_{branch}_only.txt"), cm_b, fmt="%d")
            rep_b = classification_report_np(yt, yp, nc)
            with open(os.path.join(args.out, f"classification_report_{branch}_only.txt"), "w", encoding="utf-8") as f:
                f.write(rep_b)

            # optional image
            save_confusion_matrix_image(
                cm_b,
                out_path_base=os.path.join(args.out, f"confusion_{branch}_only"),
                title=f"{branch.upper()}-ONLY Confusion (Acc={acc_b:.2f}%)"
            )
    except Exception as e:
        print(f"[WARN] branch-only eval skipped: {e}", flush=True)

    # 3) Saliency explanation (full)
    saliency_explain(model, test_loader, device, args.max_saliency, args.out)

    # 4) NEW: Clustering distribution plots
    clustering_distribution(
        model=model,
        loader=test_loader,
        device=device,
        out_dir=args.out,
        max_samples=args.max_cluster,
        k=args.cluster_k,
        seed=args.cluster_seed,
        iters=args.kmeans_iters,
    )

    print("[DONE] All results saved to:", os.path.abspath(args.out), flush=True)


if __name__ == "__main__":
    main()
