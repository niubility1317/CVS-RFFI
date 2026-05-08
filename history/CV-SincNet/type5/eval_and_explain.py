# eval_and_explain.py
# -*- coding: utf-8 -*-
"""
WiFi RFFI: evaluate CV-SincNet weights and generate analysis outputs.
- Confusion matrix + classification report (NO sklearn required)
- Branch-only eval (time-only / freq-only) if model has t_proj/f_proj/fuse/cls_head
- Saliency (time + freq + mirror asymmetry) (NO matplotlib required; if available, also save PNG)

Run:
  python eval_and_explain.py
or:
  python eval_and_explain.py --weights ./weight/xxx.pth --data ./Dataset_ORALCE --out ./eval_out
"""

import os
import argparse
import inspect
from typing import Dict, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import WiFiRFFIDataset
from model import CVSincNet


# ===================== 修改区（你只需要改这几项） =====================
WEIGHTS_PATH = "./weight/last_model_run1_Aug_CL3_silm.pth"   # ✅ 权重地址
DATASET_DIR  = "./Dataset_ORALCE"                            # ✅ 数据集根目录
OUT_DIR      = "./eval_out"
BATCH_SIZE   = 256
NUM_WORKERS  = 4
MAX_SALIENCY_SAMPLES = 4096
SAMPLE_RATE_HZ = 5e6
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


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def _strip_module_prefix(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not any(k.startswith("module.") for k in sd.keys()):
        return sd
    return {k.replace("module.", "", 1): v for k, v in sd.items()}


def _load_state_dict(weights_path: str) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(weights_path, map_location="cpu")

    # 常见封装：{"state_dict":...} / {"model":...}
    if isinstance(ckpt, dict):
        for key in ["state_dict", "model", "model_state", "net", "weights"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                sd = ckpt[key]
                return _strip_module_prefix(sd)

        # 纯 state_dict
        if any(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return _strip_module_prefix(ckpt)

    raise ValueError(f"Unrecognized checkpoint format: {weights_path}")


def _find_key(sd: Dict[str, torch.Tensor], candidates) -> Optional[str]:
    for k in candidates:
        if k in sd:
            return k
    return None


def _infer_model_hparams(sd: Dict[str, torch.Tensor]) -> Dict[str, int]:
    """
    从 checkpoint 反推模型超参（以 checkpoint 为准）
    核心修复：sinc_out 必须从 sinc.low_hz_ / sinc.band_hz_ 的shape读出，
    不能再用 time_in 推 (你这个是 6*sinc_out+4 的结构)。
    """
    # --- head ---
    head_w_key = _find_key(sd, ["cls_head.weight", "classifier.weight", "head.weight"])
    if head_w_key is None:
        raise KeyError("Cannot find head weight (e.g. cls_head.weight) in checkpoint.")
    num_classes = int(sd[head_w_key].shape[0])
    emb_dim = int(sd[head_w_key].shape[1])

    # --- sinc_out: from checkpoint param shape ---
    sinc_out_key = _find_key(sd, ["sinc.low_hz_", "sinc.band_hz_", "sinc.low_hz", "sinc.band_hz"])
    if sinc_out_key is None:
        raise KeyError("Cannot find sinc.low_hz_ / sinc.band_hz_ in checkpoint to infer sinc_out.")
    sinc_out = int(sd[sinc_out_key].shape[0])

    # --- time_in: from time_fuse weight shape ---
    tf_key = _find_key(sd, ["time_fuse.0.weight", "time_fuse.weight"])
    if tf_key is None:
        for k, v in sd.items():
            if "time_fuse" in k and k.endswith("weight") and v.ndim == 3 and v.shape[-1] == 1 and v.shape[0] == v.shape[1]:
                tf_key = k
                break
    if tf_key is None:
        raise KeyError("Cannot find time_fuse conv weight in checkpoint.")
    time_in = int(sd[tf_key].shape[1])

    # --- sinc_kernel (optional) ---
    sinc_kernel = None
    for k in ["sinc.window_", "sinc.t_", "sinc.kernel_", "sinc.filters_"]:
        if k in sd and sd[k].ndim >= 1:
            sinc_kernel = int(sd[k].shape[-1])
            break
    if sinc_kernel is None:
        sinc_kernel = 129  # fallback

    # --- infer how many bases used in time branch (just for printing/debug) ---
    extra = 4
    basis_factor = (time_in - extra) / max(1, sinc_out)

    return {
        "num_classes": num_classes,
        "emb_dim": emb_dim,
        "sinc_out": sinc_out,
        "sinc_kernel": sinc_kernel,
        "time_in": time_in,
        "basis_factor_x": basis_factor,  # 例如你这里应该接近 6.0
    }


def _filter_kwargs_for_init(cls, kwargs: Dict) -> Dict:
    """
    关键修复点：自动过滤 CVSincNet.__init__ 不接受的参数（比如 freq_bins_keep）
    """
    sig = inspect.signature(cls.__init__)
    accepted = set(sig.parameters.keys()) - {"self"}
    return {k: v for k, v in kwargs.items() if k in accepted}


def _align_state_dict_keys(sd: Dict[str, torch.Tensor], model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """
    兼容不同版本命名：cls_head <-> classifier
    """
    mkeys = set(model.state_dict().keys())
    sd2 = dict(sd)

    # cls_head -> classifier
    if "cls_head.weight" in sd2 and "cls_head.weight" not in mkeys and "classifier.weight" in mkeys:
        sd2["classifier.weight"] = sd2.pop("cls_head.weight")
        if "cls_head.bias" in sd2:
            sd2["classifier.bias"] = sd2.pop("cls_head.bias")

    # classifier -> cls_head
    if "classifier.weight" in sd2 and "classifier.weight" not in mkeys and "cls_head.weight" in mkeys:
        sd2["cls_head.weight"] = sd2.pop("classifier.weight")
        if "classifier.bias" in sd2:
            sd2["cls_head.bias"] = sd2.pop("classifier.bias")

    return sd2


def build_model_from_weights(weights_path: str, device: torch.device) -> CVSincNet:
    sd = _load_state_dict(weights_path)
    hp = _infer_model_hparams(sd)

    # 这里准备一批“候选参数名”，让不同版本 __init__ 都能吃到
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

    # 去掉 None
    candidates = {k: v for k, v in candidates.items() if v is not None}

    # ✅ 自动过滤掉不被 __init__ 接受的参数（解决你现在的 freq_bins_keep 报错）
    init_kwargs = _filter_kwargs_for_init(CVSincNet, candidates)

    print(
    f"[MODEL] inferred num_classes={hp['num_classes']} emb_dim={hp['emb_dim']} "
    f"sinc_out={hp['sinc_out']} sinc_kernel={hp['sinc_kernel']} time_in={hp['time_in']} "
    f"basis_factor≈{hp['basis_factor_x']:.2f}",
    flush=True,
    )


    model = CVSincNet(**init_kwargs)

    sd = _align_state_dict_keys(sd, model)
    # --- 安全加载：只加载 key 存在且 shape 一致的权重，避免 size mismatch 直接崩 ---
    model_sd = model.state_dict()
    sd = _align_state_dict_keys(sd, model)

    filtered = {}
    skipped = []
    for k, v in sd.items():
        if k in model_sd and tuple(v.shape) == tuple(model_sd[k].shape):
            filtered[k] = v
        elif k in model_sd:
            skipped.append((k, tuple(v.shape), tuple(model_sd[k].shape)))

    if ("cls_head.weight" in model_sd) and ("cls_head.weight" not in filtered) and ("classifier.weight" not in filtered):
        raise RuntimeError("Head weight not loaded (cls_head/classifier). Model structure likely still mismatched.")

    missing, unexpected = model.load_state_dict(filtered, strict=False)

    print(f"[LOAD] loaded={len(filtered)} keys, skipped_shape_mismatch={len(skipped)}", flush=True)
    if len(skipped) > 0:
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
    ds = WiFiRFFIDataset(data_dir, mode="test")  # ✅ 你 dataset.py 是 mode，不是 split
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    return dl


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


@torch.no_grad()
def eval_full(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    y_true, y_pred = [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        logits = model(xb)
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
    """
    用 forward hook 抓 t_emb / f_emb（要求模型里有 t_proj / f_proj 这样的层）
    """
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

        t_emb, f_emb = _get_embeddings(model, xb)
        if branch == "time":
            f_emb = torch.zeros_like(f_emb)
        else:
            t_emb = torch.zeros_like(t_emb)

        feat = fuse(torch.cat([t_emb, f_emb], dim=1))
        logits = head(feat)
        pred = logits.argmax(dim=1)

        y_true.append(yb.detach().cpu().numpy())
        y_pred.append(pred.detach().cpu().numpy())

    return np.concatenate(y_true), np.concatenate(y_pred)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=WEIGHTS_PATH)
    parser.add_argument("--data", type=str, default=DATASET_DIR)
    parser.add_argument("--out", type=str, default=OUT_DIR)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--max_saliency", type=int, default=MAX_SALIENCY_SAMPLES)
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
    except Exception as e:
        print(f"[WARN] branch-only eval skipped: {e}", flush=True)

    # 3) Saliency explanation (full)
    saliency_explain(model, test_loader, device, args.max_saliency, args.out)

    print("[DONE] All results saved to:", os.path.abspath(args.out), flush=True)


if __name__ == "__main__":
    main()
