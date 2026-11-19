"""Train and predict the two paired SatIQ CE arms without importing target truth."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .prepare import SCHEMA, STARTS, WIDTH, write_json


ARMS = ("backbone_ce", "pamf_ce")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def source_contract(root):
    contract = read(root / "contract.json")
    labels = contract["classes"]
    if (contract["schema"] != SCHEMA or contract["source_station"] != "Oxford, UK"
            or contract["target_station"] != "Thun, Switzerland"
            or contract["sample_rate_hz"] != 25_000_000
            or contract["window_starts"] != list(STARTS) or contract["window_samples"] != WIDTH
            or [entry["class_id"] for entry in labels] != list(range(len(labels)))
            or len({(entry["satellite_id"], entry["beam_id"]) for entry in labels}) != len(labels)):
        raise ValueError("Unexpected SatIQ source contract")
    return contract


def source_data(root, contract):
    datasets, identities = {}, {}
    for role, count in contract["source_counts"].items():
        with np.load(root / f"{role}.npz", allow_pickle=False) as d:
            if set(d.files) != {"ids", "iq", "y"}:
                raise ValueError("Unexpected labeled source fields")
            ids, iq, y = d["ids"], d["iq"], d["y"]
        check_iq(iq, ids, count)
        nch = len(contract["classes"])
        if (y.dtype != np.int64 or y.shape != (count,) or (y < 0).any() or (y >= nch).any()
                or np.bincount(y, minlength=nch).tolist()
                != [item[f"uk_{role}"] for item in contract["classes"]]):
            raise ValueError("Source labels and class map differ")
        # N607's PyTorch 2.1 was compiled against NumPy 1.x. DLPack shares
        # CPU arrays with NumPy 2.x without using the legacy C ABI.
        datasets[role] = tuple(torch.utils.dlpack.from_dlpack(a) for a in (iq, y))
        identities[role] = set(ids.tolist())
    if identities["train"] & identities["val"]:
        raise ValueError("Source train/validation physical IDs overlap")
    return datasets


def check_iq(iq, ids, count):
    if (iq.dtype != np.float32 or iq.shape != (count, len(STARTS), 2, WIDTH)
            or ids.dtype.kind != "U" or ids.shape != (count,)
            or len(set(ids.tolist())) != count):
        raise ValueError("Invalid IQ or non-unique physical IDs")


class Identity(nn.Module):
    """SatIQ identity readout using the native backbone and optional SIG-TIME."""

    def __init__(self, core, sequence_pool):
        super().__init__()
        self.core = core
        self.sequence_pool = sequence_pool

    def features(self, x):
        if self.sequence_pool is None:
            features = self.core.features(x)
        else:
            t2 = []
            handle = self.core.id_backbone.t2.register_forward_hook(
                lambda module, args, output: t2.append(output))
            try:
                features = self.core.features(x)
            finally:
                handle.remove()
            features = features + self.sequence_pool(t2[0])
        return features

    def forward(self, x):
        return self.core.id_backbone.cls_head.classify(self.features(x))


def build_model(runtime, seed, nclasses, arm):
    from experiments.pamf_phase3_open_world.source_development import _activate_code

    _activate_code(runtime)
    from experiments.cvs_residual_identity.model import build as build_core
    from experiments.cvs_rdrop_sinc_units.coordinate import CoordinateSinc
    from experiments.cvs_rdrop_sinc_units.x3_features import CompositePool

    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        backbone = build_core("residual_fusion")
        core = backbone.id_backbone
        core.cls_head.weight = nn.Parameter(torch.empty(nclasses, 160))
        nn.init.xavier_uniform_(core.cls_head.weight)
        core.sinc = CoordinateSinc(core.sinc, "CENTER_25K")
        signature = (CompositePool(core.t2.pw.out_channels, "SIG_TIME_X3", seed)
                     if arm == "pamf_ce" else None)
        model = Identity(backbone, signature)
    expected = 163265 + nclasses * 160 + (55968 if arm == "pamf_ce" else 0)
    if sum(p.numel() for p in model.parameters()) != expected:
        raise ValueError(f"{arm}: unexpected native backbone parameter count")
    return model


def packet_logits(model, raw):
    batch, windows = raw.shape[:2]
    # Preserve the relative level of the four windows. This normalization uses
    # only the current physical message and is identical in both stations.
    rms = raw.square().sum(dim=2).mean(dim=(1, 2), keepdim=True).sqrt().unsqueeze(2)
    x = (raw / rms.clamp_min(1e-8)).reshape(-1, 2, WIDTH)
    logits = model(x)
    return logits.reshape(batch, windows, -1).mean(dim=1)


@torch.no_grad()
def evaluate(model, x, device, batch):
    model.eval()
    parts = []
    for offset in range(0, len(x), batch):
        # The query memory map is read-only. DLPack needs a writable CPU view.
        array = np.array(x[offset:offset + batch], copy=True)
        tensor = torch.utils.dlpack.from_dlpack(array).to(device)
        parts.append(np.from_dlpack(packet_logits(model, tensor).cpu()))
    logits = np.concatenate(parts)
    if not np.isfinite(logits).all():
        raise ValueError("Non-finite inference logits")
    return logits


@torch.no_grad()
def validate(model, x, y, device, batch):
    model.eval()
    logits = torch.cat([packet_logits(model, a.to(device)).cpu() for a in x.split(batch)])
    loss = F.cross_entropy(logits, y)
    if not loss.isfinite():
        raise ValueError("Non-finite source validation loss")
    return {"accuracy": float((logits.argmax(dim=1) == y).float().mean()), "ce": float(loss)}


def load_weights(model, path):
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)


def run_train(args, config):
    contract = source_contract(args.source)
    data = source_data(args.source, contract)
    nclasses = len(contract["classes"])
    device = torch.device("cuda:0")
    args.out.mkdir(exist_ok=False)
    write_json(args.out / "resolved_config.json",
               dict(config, arm=args.arm, classes=nclasses, input_source=str(args.source),
                    initialization="scratch", selected_checkpoint="fixed_final_epoch",
                    output=str(args.out)))
    torch.manual_seed(config["model_seed"])
    model = build_model(config["runtime"], config["model_seed"], nclasses, args.arm)
    model = model.to(device)
    initial = args.out / "scratch_e00.pt"
    torch.save(model.state_dict(), initial)
    load_weights(model, initial)
    smoke = validate(model, *[a[:2] for a in data["train"]], device, config["batch_size"])
    print(json.dumps({"event": "no_query_checkpoint_smoke", "status": "PASS", **smoke}), flush=True)
    train_x, train_y = data["train"]
    val_x, val_y = data["val"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"],
                                  weight_decay=config["weight_decay"])
    clock = time.monotonic()
    steps = 0
    rng = torch.Generator().manual_seed(config["shuffle_seed"])
    with (args.out / "epochs.jsonl").open("x", encoding="utf-8") as log:
        for epoch in range(1, config["epochs"] + 1):
            model.train()
            total_loss = 0.0
            for indices in torch.randperm(len(train_y), generator=rng).split(config["batch_size"]):
                optimizer.zero_grad(set_to_none=True)
                logits = packet_logits(model, train_x[indices].to(device))
                loss = F.cross_entropy(logits, train_y[indices].to(device))
                if not loss.isfinite():
                    raise ValueError("Non-finite source training loss")
                loss.backward()
                optimizer.step()
                steps += 1
                total_loss += float(loss.detach()) * len(indices)
            row = {"epoch": epoch, "updates": steps, "uk_train_ce": total_loss / len(train_y),
                   "uk_val": validate(model, val_x, val_y, device, config["batch_size"]),
                   "seconds": time.monotonic() - clock,
                   "peak_gpu_allocated_mib": torch.cuda.max_memory_allocated() / 2**20}
            line = json.dumps(row, allow_nan=False)
            log.write(line + "\n")
            log.flush()
            print(line, flush=True)
    torch.save(model.state_dict(), args.out / "source_final.pt")


def run_predict(args, config):
    contract = source_contract(args.source)
    device = torch.device("cuda:0")
    model = build_model(config["runtime"], config["model_seed"], len(contract["classes"]), args.arm)
    load_weights(model, args.out / "source_final.pt")
    model = model.to(device)
    if {p.name for p in args.query.iterdir()} != {"ids.npy", "iq.npy"}:
        raise ValueError("Unexpected fields in Swiss public input")
    iq = np.load(args.query / "iq.npy", mmap_mode="r", allow_pickle=False)
    ids = np.load(args.query / "ids.npy", allow_pickle=False)
    check_iq(iq, ids, contract["target_physical_records"])
    if set(ids.tolist()) & (set(np.load(args.source / "train.npz")["ids"].tolist())
                           | set(np.load(args.source / "val.npz")["ids"].tolist())):
        raise ValueError("Swiss input overlaps a UK physical record")
    fpath = args.out / "predictions.npz"
    if fpath.exists():
        raise FileExistsError(fpath)
    logits = evaluate(model, iq, device, config["batch_size"])
    with fpath.open("xb") as f:
        np.savez(f, ids=ids, logits=logits)
    print(json.dumps({"event": "prediction_fixed", "arm": args.arm, "records": len(ids)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "predict"))
    parser.add_argument("config", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("arm", choices=ARMS)
    parser.add_argument("--query", type=Path)
    args = parser.parse_args()
    if (args.mode == "predict") != (args.query is not None):
        parser.error("--query is required only for prediction")
    config = read(args.config)
    (run_train if args.mode == "train" else run_predict)(args, config)


if __name__ == "__main__":
    main()
