import argparse
import os
from typing import Optional, Tuple, Dict, Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from dataset import WiFiRFFIDataset
from model import build_model as build_single_model
from model_dual_cvsincnet import build_dual_model
from DataAugmentation_v2 import apply_receiver_dg

try:
    from dataset_wisig_modified import load_wisig_compact_pkl, make_day123_randomsplit_plus_day4_test
except Exception:
    try:
        from dataset_wisig import load_wisig_compact_pkl, make_day123_randomsplit_plus_day4_test
    except Exception:
        load_wisig_compact_pkl = None
        make_day123_randomsplit_plus_day4_test = None


class NpyRFFIDataset(Dataset):
    def __init__(self, x_path: str, y_path: str):
        self.x_path = os.path.realpath(x_path)
        self.y_path = os.path.realpath(y_path)
        self.X = np.load(self.x_path, mmap_mode="r")
        self.Y = np.load(self.y_path)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        signal = torch.from_numpy(self.X[idx].copy()).float()
        label = torch.tensor(int(self.Y[idx]), dtype=torch.long)
        energy = torch.sqrt(torch.sum(signal ** 2))
        if energy > 1e-8:
            signal = signal / energy
        return signal, label


def unpack_batch_xy(batch):
    if isinstance(batch, (tuple, list)):
        return batch[0], batch[1]
    raise TypeError(f"Unsupported batch type: {type(batch)}")


def _strip_module_prefix(sd: dict) -> dict:
    if not any(k.startswith("module.") for k in sd.keys()):
        return sd
    return {k.replace("module.", "", 1): v for k, v in sd.items()}


def _load_state_dict(weights_path: str, map_location="cpu") -> dict:
    ckpt = torch.load(weights_path, map_location=map_location)
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model", "model_state", "net", "weights"):
            if key in ckpt and isinstance(ckpt[key], dict):
                sd = ckpt[key]
                if "state_dict" in sd and isinstance(sd["state_dict"], dict):
                    sd = sd["state_dict"]
                return _strip_module_prefix(sd)
        if any(torch.is_tensor(v) for v in ckpt.values()):
            return _strip_module_prefix(ckpt)
    raise ValueError(f"Unrecognized checkpoint format: {weights_path}")


def _safe_load_into_model(model: torch.nn.Module, sd: dict, strict: bool = True) -> None:
    try:
        missing, unexpected = model.load_state_dict(sd, strict=strict)
        if len(missing) == 0 and len(unexpected) == 0:
            print("[OK] load_state_dict exact match.")
        else:
            print(f"[WARN] load_state_dict missing={len(missing)} unexpected={len(unexpected)}")
        return
    except RuntimeError as e:
        if not strict:
            raise
        print("[WARN] strict load failed, fallback to shape-safe partial load.")
        print("       ", str(e).splitlines()[0])

    model_sd = model.state_dict()
    kept = {}
    dropped_missing = 0
    dropped_shape = 0
    for k, v in sd.items():
        if k not in model_sd:
            dropped_missing += 1
            continue
        if hasattr(model_sd[k], "shape") and hasattr(v, "shape") and tuple(model_sd[k].shape) != tuple(v.shape):
            dropped_shape += 1
            continue
        kept[k] = v
    print(
        f"[INFO] filtered state_dict kept={len(kept)}/{len(sd)} "
        f"drop_not_in_model={dropped_missing} drop_shape_mismatch={dropped_shape}"
    )
    model.load_state_dict(kept, strict=False)


def _infer_num_classes_from_sd(sd: Dict[str, torch.Tensor], default_nc: int) -> int:
    candidates = [
        "id_backbone.cls_head.head.weight",
        "id_backbone.cls_head.weight",
        "cls_head.head.weight",
        "cls_head.weight",
        "classifier.weight",
    ]
    for k in candidates:
        if k in sd and torch.is_tensor(sd[k]) and sd[k].ndim == 2:
            return int(sd[k].shape[0])
    return int(default_nc)


def _infer_num_domains_from_sd(sd: Dict[str, torch.Tensor], default_nd: int) -> int:
    candidates = [
        "rx_head.net.3.weight", "rx_head.net.3.bias", "rx_head.net.0.weight",
        "probe_rx_on_tx.net.3.weight", "probe_rx_on_tx.net.3.bias",
        "adv_rx_on_tx.net.3.weight", "adv_rx_on_tx.net.3.bias",
        "dom_head.net.3.weight", "dom_head.net.3.bias", "dom_head.net.0.weight",
    ]
    for k in candidates:
        if k in sd and torch.is_tensor(sd[k]) and sd[k].ndim >= 1:
            return int(sd[k].shape[0])
    return int(default_nd)


def _is_dual_state_dict(sd: Dict[str, torch.Tensor]) -> bool:
    return any(k.startswith("id_backbone.") or k.startswith("dom_backbone.") for k in sd.keys())


def _get_logits(model_out: Any) -> torch.Tensor:
    if torch.is_tensor(model_out):
        return model_out
    if isinstance(model_out, (tuple, list)) and len(model_out) >= 1:
        return model_out[0]
    if isinstance(model_out, dict):
        for k in ("tx_logits", "logits", "out", "pred", "y_pred"):
            if k in model_out and torch.is_tensor(model_out[k]):
                return model_out[k]
    raise RuntimeError(f"Unsupported model output type: {type(model_out)}")


def _parse_days(s: str):
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
    return out if len(out) > 0 else None


def build_test_dataset(args, device: torch.device):
    if args.dataset == "wisig":
        if load_wisig_compact_pkl is None or make_day123_randomsplit_plus_day4_test is None:
            raise RuntimeError("WiSig dataset loader import failed; cannot build WISIG dataset.")
        ds_w = load_wisig_compact_pkl(args.wisig_pkl)
        eq2 = "both" if str(args.wisig_equalized).lower() == "both" else int(args.wisig_equalized)
        max_tr = None if int(args.wisig_max_train_per_combo) <= 0 else int(args.wisig_max_train_per_combo)
        max_te = None if int(args.wisig_max_test_per_combo) <= 0 else int(args.wisig_max_test_per_combo)
        train_ds, test_ds, split_info = make_day123_randomsplit_plus_day4_test(
            ds_w,
            equalized=eq2,
            out_len=int(args.wisig_out_len),
            domain=str(args.wisig_domain),
            normalize=True,
            crop_mode="center",
            transform=None,
            train_ratio=float(args.wisig_train_ratio),
            train_days=_parse_days(args.wisig_train_days),
            full_test_days=_parse_days(args.wisig_full_test_days),
            max_samples_per_combo_train=max_tr,
            max_samples_per_combo_test=max_te,
            seed=int(args.seed),
        )
        del train_ds
        input_len = int(args.wisig_out_len)
        infer_nc = len(ds_w.get("tx_list", []))
        if infer_nc > 0 and int(args.num_classes) != infer_nc:
            print(f"[WISIG] override num_classes {args.num_classes} -> {infer_nc}")
            args.num_classes = infer_nc
        print(f"[WISIG] pkl={args.wisig_pkl} out_len={input_len} domain={args.wisig_domain} split={split_info}")
        return test_ds, input_len

    if args.run2_dir:
        x_path = os.path.join(args.run2_dir, "X_test.npy")
        y_path = os.path.join(args.run2_dir, "Y_test.npy")
        if os.path.exists(x_path) and os.path.exists(y_path):
            print(f"[ORALCE] using run2 npy: {x_path}, {y_path}")
            ds = NpyRFFIDataset(x_path, y_path)
            return ds, int(args.input_len_oralce)
        print(f"[WARN] run2 npy not found under {args.run2_dir}; fallback to WiFiRFFIDataset.")

    ds = WiFiRFFIDataset(args.dataset_dir, mode="test", run_name=args.run_name)
    input_len = int(args.input_len_oralce)
    try:
        x0, _ = ds[0]
        input_len = int(x0.shape[-1])
    except Exception:
        pass
    print(f"[ORALCE] dataset_dir={args.dataset_dir} run_name={args.run_name} input_len={input_len}")
    return ds, input_len


def build_model_for_eval(args, device: torch.device, input_len: int) -> Tuple[torch.nn.Module, int]:
    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"weights not found: {args.weights}")
    sd = _load_state_dict(args.weights, map_location="cpu")

    is_dual = _is_dual_state_dict(sd)
    if args.force_single:
        is_dual = False
    if args.force_dual:
        is_dual = True

    if float(args.sample_rate_hz) <= 0.0:
        sample_rate_hz = 25e6 if args.dataset == "wisig" else 5e6
    else:
        sample_rate_hz = float(args.sample_rate_hz)

    if is_dual:
        num_classes = _infer_num_classes_from_sd(sd, args.num_classes)
        num_domains = _infer_num_domains_from_sd(sd, args.num_domains)
        model = build_dual_model(
            num_classes=num_classes,
            num_domains=num_domains,
            model_size=args.model_size,
            dataset=args.dataset,
            input_len=input_len,
            sample_rate_hz=sample_rate_hz,
            detach_imp_gate=bool(args.detach_imp_gate),
            disable_freq_stats_to_shared=bool(args.disable_freq_stats_to_shared),
        ).to(device)
        print(f"[MODEL] dual | num_classes={num_classes} num_domains={num_domains} model_size={args.model_size}")
    else:
        num_classes = _infer_num_classes_from_sd(sd, args.num_classes)
        model = build_single_model(
            num_classes=num_classes,
            model_size=args.model_size,
            dataset=args.dataset,
            input_len=input_len,
            sample_rate_hz=sample_rate_hz,
            detach_imp_gate=bool(args.detach_imp_gate),
            disable_freq_stats_to_shared=bool(args.disable_freq_stats_to_shared),
        ).to(device)
        print(f"[MODEL] single | num_classes={num_classes} model_size={args.model_size}")

    for k, v in list(sd.items()):
        if torch.is_tensor(v):
            sd[k] = v.to(device)
    if args.strict:
        model.load_state_dict(sd, strict=True)
        print("[OK] strict load_state_dict success.")
    else:
        _safe_load_into_model(model, sd, strict=True)
    model.eval()
    return model, int(num_classes)


def evaluate(model, loader, device, num_classes: int, args, sample_rate_hz: float):
    model.eval()
    correct = 0
    total = 0
    conf = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.no_grad():
        for batch in loader:
            x, y = unpack_batch_xy(batch)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            if bool(args.dc_block):
                x = x - x.mean(dim=-1, keepdim=True)
                rms = torch.sqrt(torch.mean(x ** 2, dim=(1, 2), keepdim=True) + 1e-12)
                x = x / rms

            if bool(args.tta):
                n_tta = max(1, int(args.tta_n))
                envs = max(1, int(args.tta_envs))
                logits_acc = None
                for _ in range(n_tta):
                    env_id = torch.randint(0, envs, (x.size(0),), device=x.device)
                    x_aug = apply_receiver_dg(
                        x,
                        fs=float(sample_rate_hz),
                        env_id=env_id,
                        p_lowpass=float(args.tta_p_lowpass),
                        p_multipath=float(args.tta_p_multipath),
                    )
                    rms = torch.sqrt(torch.mean(x_aug ** 2, dim=(1, 2), keepdim=True) + 1e-12)
                    x_aug = x_aug / rms
                    logits_i = _get_logits(model(x_aug))
                    logits_acc = logits_i if logits_acc is None else (logits_acc + logits_i)
                logits = logits_acc / float(n_tta)
            else:
                logits = _get_logits(model(x))

            pred = torch.argmax(logits, dim=1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())

            y_np = y.detach().cpu().numpy()
            p_np = pred.detach().cpu().numpy()
            for t, p in zip(y_np, p_np):
                if 0 <= int(t) < num_classes and 0 <= int(p) < num_classes:
                    conf[int(t), int(p)] += 1

    acc = 100.0 * correct / max(total, 1)
    per_class_total = conf.sum(axis=1)
    per_class_correct = np.diag(conf)
    per_class_acc = (per_class_correct / np.maximum(per_class_total, 1)) * 100.0
    return acc, conf, per_class_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="wisig", choices=["wisig", "oralce"])
    parser.add_argument("--weights", type=str, default="./best_model_v10.pth")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_classes", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--model_size", type=str, default="M")
    parser.add_argument("--num_domains", type=int, default=1)
    parser.add_argument("--force_single", action="store_true")
    parser.add_argument("--force_dual", action="store_true")
    parser.add_argument("--sample_rate_hz", type=float, default=0.0)
    parser.add_argument("--detach_imp_gate", dest="detach_imp_gate", action="store_true")
    parser.add_argument("--no_detach_imp_gate", dest="detach_imp_gate", action="store_false")
    parser.set_defaults(detach_imp_gate=True)
    parser.add_argument("--disable_freq_stats_to_shared", dest="disable_freq_stats_to_shared", action="store_true")
    parser.add_argument("--enable_freq_stats_to_shared", dest="disable_freq_stats_to_shared", action="store_false")
    parser.set_defaults(disable_freq_stats_to_shared=True)
    parser.add_argument("--save_conf", action="store_true")

    parser.add_argument("--dc_block", action="store_true")
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--tta_n", type=int, default=8)
    parser.add_argument("--tta_envs", type=int, default=8)
    parser.add_argument("--tta_p_lowpass", type=float, default=0.7)
    parser.add_argument("--tta_p_multipath", type=float, default=0.7)

    parser.add_argument("--dataset_dir", type=str, default="./Dataset_ORALCE")
    parser.add_argument("--run_name", type=str, default="run1")
    parser.add_argument("--run2_dir", type=str, default="./Dataset_ORALCE/run2")
    parser.add_argument("--input_len_oralce", type=int, default=1024)

    parser.add_argument("--wisig_pkl", type=str, default="./Dataset_WigSig/ManySig.pkl")
    parser.add_argument("--wisig_equalized", type=str, default="1")
    parser.add_argument("--wisig_domain", type=str, default="day", choices=["day", "rx", "rx_day"])
    parser.add_argument("--wisig_out_len", type=int, default=256)
    parser.add_argument("--wisig_train_ratio", type=float, default=0.8)
    parser.add_argument("--wisig_train_days", type=str, default="0,1,2")
    parser.add_argument("--wisig_full_test_days", type=str, default="3")
    parser.add_argument("--wisig_max_train_per_combo", type=int, default=0)
    parser.add_argument("--wisig_max_test_per_combo", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    req = str(args.device).strip().lower()
    if req.startswith("cuda") and torch.cuda.is_available():
        try:
            device = torch.device(args.device)
        except Exception:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    sample_rate_hz = float(args.sample_rate_hz) if float(args.sample_rate_hz) > 0 else (25e6 if args.dataset == "wisig" else 5e6)

    ds, input_len = build_test_dataset(args, device)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model, num_classes = build_model_for_eval(args, device, input_len=input_len)
    acc, conf, per_class_acc = evaluate(model, loader, device, num_classes=num_classes, args=args, sample_rate_hz=sample_rate_hz)

    print(f"[TEST] dataset={args.dataset} weights={args.weights} acc={acc:.2f}%")
    for i, a in enumerate(per_class_acc):
        print(f"  class {i:02d}: {a:.2f}%   (n={conf[i].sum()})")

    if args.save_conf:
        tag = f"conf_{args.dataset}"
        np.save(f"{tag}.npy", conf)
        with open(f"{tag}.txt", "w", encoding="utf-8") as f:
            f.write(np.array2string(conf, separator=", "))
        print(f"[TEST] confusion saved: {tag}.npy, {tag}.txt")


if __name__ == "__main__":
    main()
