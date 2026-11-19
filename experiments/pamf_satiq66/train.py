"""Scratch training and fixed-checkpoint prediction for SatIQ Oxford 66."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from .data import CLASSES, SCHEMA, SOURCE_COUNTS, read_json, write_json
from .model import build_model, capture, normalized, views
from .objectives import (centers, day_relation, masked_mean, packet_contrast, partners,
                         protected_backward, prototype_loss, prototypes, style_extrapolation, supervised)


ROWS = ("S0", "S1", "U0", "U1")


def row_data(root, role, representation):
    with np.load(root / f"{role}_meta.npz", allow_pickle=False) as z:
        fields = {"ids", "day", "y"} if role in {"L", "V"} else {"ids", "day"} if role == "U" else {"ids"}
        if set(z.files) != fields:
            raise ValueError(f"Forbidden or missing columns in {role}")
        ids = z["ids"]
        meta = {key: z[key] for key in fields - {"ids"}}
    iq = np.load(root / f"{role}_iq.npy", mmap_mode="r", allow_pickle=False)
    if (iq.shape != (len(ids), 2, 2, 256) or iq.dtype != np.float32
            or len(set(ids.tolist())) != len(ids) or any(len(a) != len(ids) for a in meta.values())):
        raise ValueError(f"Invalid physical records in {role}")
    if role in SOURCE_COUNTS and len(ids) != SOURCE_COUNTS[role]:
        raise ValueError(f"Unexpected count for {role}")
    if "day" in meta:
        legal = range(8, 11) if role == "V" else range(8)
        if not np.isin(meta["day"], legal).all():
            raise ValueError(f"Unexpected source date in {role}")
    if "y" in meta and not np.isin(meta["y"], range(66)).all():
        raise ValueError(f"Unexpected class index in {role}")
    return iq[:, representation], ids, meta


def check_contract(root, counts):
    c = read_json(root / "contract.json")
    if (c["schema"] != SCHEMA or c["classes"] != list(CLASSES) or c["true_receiver_count"] != 1
            or c["iq_rate_hz"] != [25000000, 800000] or (counts and c["counts"] != counts)):
        raise ValueError("Wrong SatIQ Oxford data contract")
    return c


class Batches:
    def __init__(self, metadata, seed):
        self.rng = np.random.default_rng(seed)
        self.groups = np.array([[np.flatnonzero((metadata["y"] == y) & (metadata["day"] == d))
                                 for d in range(8)] for y in range(66)])
        if self.groups.shape != (66, 8, 20):
            raise ValueError("Each satellite and source day must supply exactly 20 labeled records")

    def next(self):
        d0 = self.rng.integers(8, size=66)
        d1 = (d0 + self.rng.integers(1, 8, size=66)) % 8
        r0, r1 = self.rng.integers(20, size=(2, 66))
        y = np.arange(66)
        return self.rng.permutation(np.concatenate((self.groups[y, d0, r0], self.groups[y, d1, r1])))


def batch(data, indices, device):
    x, _, meta = data
    return (normalized(device_array(x[indices], device)),
            {k: device_array(a[indices], device) for k, a in meta.items()})


def device_array(array, device):
    # The N607 PyTorch 2.1 build uses the NumPy 1.x ABI. Writable copies
    # and DLPack bridge its tensors to the installed NumPy 2.x arrays.
    return torch.utils.dlpack.from_dlpack(np.array(array, copy=True)).to(device)


def cuda_device():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA worker requested but CUDA is unavailable")
    return torch.device("cuda:0")


def metrics(matrix):
    diagonal = np.diag(matrix)
    actual = matrix.sum(1)
    pred = matrix.sum(0)
    return {"accuracy": float(diagonal.sum() / max(1, matrix.sum())),
            "macro_f1": float((2 * diagonal / np.maximum(actual + pred, 1)).mean()),
            "macro_recall": float((diagonal / np.maximum(actual, 1)).mean()),
            "records": int(matrix.sum())}


@torch.inference_mode()
def predict(model, data, device):
    model.eval()
    x = data[0]
    parts = []
    for start in range(0, len(x), 256):
        t = normalized(device_array(x[start:start + 256], device))
        parts.append(np.from_dlpack(model(t).cpu()))
    logits = np.concatenate(parts)
    if logits.shape != (len(x), 66) or not np.isfinite(logits).all():
        raise ValueError("Prediction must cover all records and 66 classes with finite scores")
    return logits


def validate(model, data, device):
    logits = predict(model, data, device)
    target = data[2]["y"]
    matrix = np.bincount(66 * target + logits.argmax(1), minlength=66**2).reshape(66, 66)
    return metrics(matrix)


def strength(epoch, start):
    return .25 * max(0., (50 - epoch) / (50 - start))


def unlabeled_loss(model, raw_l, lmeta, u, umeta, ids, augment, rate, epoch, hybrid):
    model.eval()
    with torch.no_grad():
        if hybrid:
            raw_eval = capture(model, u)
            confidence, pseudo = raw_eval[0].softmax(1).max(1)
            _, candidate = views(u, augment, rate, strength(epoch, 21))
            bank, available = centers(raw_l, lmeta["y"], lmeta["day"])
            voted = prototypes(bank, raw_eval[2], umeta["day"]).argmax(2) == pseudo[:, None]
            altered = capture(model, candidate)[2]
            voted_alt = prototypes(bank, altered, umeta["day"]).argmax(2) == pseudo[:, None]
            nonzero = (candidate != u).any(dim=(1, 2))
        else:
            confidence, pseudo = model(u).softmax(1).max(1)
    model.train()
    accepted = confidence >= .95
    out = {"u_seen": len(u), "u_accepted": accepted.sum()}
    if not hybrid:
        return masked_mean(F.cross_entropy(model(u), pseudo, reduction="none"), accepted), out

    logits, fused, branches = capture(model, u)
    certified = accepted & available[umeta["day"]] & (voted.sum(1) >= 4) & (voted_alt.sum(1) >= 4)
    selected = certified & nonzero
    row = torch.where(selected[:, None, None], candidate, u)
    pl_raw = F.cross_entropy(logits, pseudo, reduction="none")
    pos, paired = partners(pseudo, umeta["day"], ids, accepted)
    contextual = model.classify_features(style_extrapolation(fused, pos, paired))
    pl_context = F.cross_entropy(contextual, pseudo, reduction="none")
    pl = masked_mean(pl_raw + .25 * (pl_context - pl_raw) / 1.25, accepted)
    if bool(selected.any()):
        logits_p, fused_p, branches_p = capture(model, row)
        pl_phys = F.cross_entropy(logits_p, pseudo, reduction="none")
        pl = pl + .25 * masked_mean(pl_phys - pl_raw, selected) / 1.25
        p_fused, p_anchors = packet_contrast(fused, fused_p, pseudo, accepted, selected)
        p_branches, _ = packet_contrast(branches, branches_p, pseudo, accepted, selected)
        proto = prototype_loss(prototypes(bank, branches_p, umeta["day"]), pseudo, selected)
    else:
        p_fused = p_branches = proto = pl * 0
        p_anchors = selected.sum()
    relation, days = day_relation(branches, pseudo, umeta["day"], ids, accepted)
    out.update(u_certified=certified.sum(), u_nonzero=selected.sum(), u_paired=paired.sum(),
               u_packet_anchors=p_anchors, u_day_anchors=days, u_proto=proto.detach(),
               u_packet_fusion=p_fused.detach(), u_packet_branches=p_branches.detach(),
               u_day_relation=relation.detach())
    return pl + .05 * (p_fused + p_branches + proto) + .025 * relation, out


def train(root, out, config, arm, representation, pilot=False, smoke=False):
    contract = check_contract(root, SOURCE_COUNTS)
    device = cuda_device()
    torch.manual_seed(config["model_seed"])
    ldata = row_data(root, "L", representation)
    vdata = row_data(root, "V", representation)
    if set(ldata[1]) & set(vdata[1]):
        raise ValueError("L and V overlap")
    uses_u = arm.startswith("U") and not smoke
    if uses_u:
        udata = row_data(root, "U", representation)
        if set(udata[1]) & (set(ldata[1]) | set(vdata[1])):
            raise ValueError("U overlaps a labeled source role")
    model = build_model(config["runtime"], config["model_seed"], representation).to(device)
    out.mkdir(parents=True, exist_ok=False)
    rate = contract["iq_rate_hz"][representation]
    resolved = dict(config, arm=arm, role="smoke" if smoke else "input_pilot" if pilot else "formal",
                    representation=representation, rate_hz=rate, source=str(root), output=str(out),
                    initialization="scratch", device=str(device), source_counts=contract["counts"],
                    uses_u=uses_u, checkpoint_policy="fixed_endpoint", classes=list(CLASSES))
    write_json(out / "resolved_config.json", resolved)
    scratch = out / "scratch_e00.pt"
    torch.save(model.state_dict(), scratch)
    model.load_state_dict(torch.load(scratch, map_location=device, weights_only=True), strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    labeled_sampler = Batches(ldata[2], config["data_seed"])
    u_rng = np.random.default_rng(config["data_seed"] + 1)
    augment_l = torch.Generator(device=device).manual_seed(config["augmentation_seed"])
    augment_u = torch.Generator(device=device).manual_seed(config["augmentation_seed"] + 1)
    epochs = 1 if smoke else config["input_pilot_epochs"] if pilot else config["epochs"]
    steps = 1 if smoke else config["steps_per_epoch"]
    hybrid_l = arm in {"S1", "U0", "U1"}
    if smoke:
        # Test the full supervised objective and each affected path from
        # a freshly serialized scratch checkpoint, using L only.
        hybrid_l = True
    start = time.monotonic()
    with (out / "epochs.jsonl").open("x", encoding="utf-8") as log:
        for epoch in range(1, epochs + 1):
            model.train()
            totals = {}
            for step in range(1, steps + 1):
                optimizer.zero_grad(set_to_none=True)
                indices = labeled_sampler.next()
                x, meta = batch(ldata, indices, device)
                ids = device_array(indices, device)
                ordinary, physical = views(x, augment_l, rate, strength(epoch, 1))
                raw = capture(model, x)
                raw_o = model(ordinary)
                raw_p = capture(model, physical)
                delta = (physical - x).square().mean(dim=(1, 2))
                loss_l, record = supervised(raw, raw_o, raw_p, meta["y"], meta["day"], ids, hybrid_l, delta > 0)
                record.update(l_seen=len(x), perturbation_rms=delta.sqrt().mean().detach())
                if epoch < 50:
                    record["l_added_snr_db"] = (-10 * (2 * delta).clamp_min(1e-12).log10()).mean()
                if not loss_l.isfinite():
                    raise ValueError("Non-finite labeled loss")
                loss = loss_l
                if uses_u and epoch >= 14:
                    u_indices = u_rng.choice(len(udata[1]), config["u_batch_size"], replace=False)
                    u, umeta = batch(udata, u_indices, device)
                    hybrid_u = arm == "U1" and epoch >= 21
                    loss_u, u_record = unlabeled_loss(model, raw[2].detach(), meta, u, umeta,
                                                        device_array(u_indices, device),
                                                        augment_u, rate, epoch, hybrid_u)
                    if not loss_u.isfinite():
                        raise ValueError("Non-finite hidden-U objective")
                    record.update(u_record, u_loss=loss_u.detach())
                    loss = loss_l + loss_u
                    if hybrid_u:
                        record.update(protected_backward(model, loss_l, loss_u))
                if not (arm == "U1" and epoch >= 21 and uses_u):
                    loss.backward()
                if smoke:
                    for key in ("packet_anchors", "day_anchors", "prototype_anchors"):
                        if int(record[key]) != len(x):
                            raise ValueError(f"Scratch smoke did not activate {key}")
                    for name, parameter in model.named_parameters():
                        if parameter.grad is not None and not parameter.grad.isfinite().all():
                            raise ValueError(f"Non-finite gradient for {name}")
                    core = model.core.id_backbone
                    paths = [core.t_proj, core.f_proj, core.pa_proj, model.sequence_pool, model.response_projection]
                    for path in paths:
                        if not any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in path.parameters()):
                            raise ValueError(f"Missing supervised gradient in {type(path).__name__}")
                optimizer.step()
                for k, value in record.items():
                    totals[k] = totals.get(k, 0) + value
            report = dict(epoch=epoch, updates=epoch * steps,
                          train={k: float(v) if k.endswith(("seen", "anchors", "accepted", "certified", "nonzero",
                                                           "paired", "conflict", "cap", "zero")) else float(v) / steps
                                 for k, v in totals.items()},
                          seconds=time.monotonic() - start)
            if not smoke and (epoch == epochs or (not pilot and epoch % 10 == 0)):
                report["v"] = validate(model, vdata, device)
            if device.type == "cuda":
                report["peak_gpu_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
            line = json.dumps(report, allow_nan=False)
            log.write(line + "\n")
            log.flush()
            print(line, flush=True)
    torch.save(model.state_dict(), out / "source_final.pt")
    write_json(out / "complete.json", report)


def prediction(root, out, query, config):
    check_contract(root, SOURCE_COUNTS)
    check_contract(query, None)
    c = read_json(out / "resolved_config.json")
    if c["role"] != "formal" or read_json(out / "complete.json")["updates"] != 5000:
        raise ValueError("Prediction requires the fixed formal endpoint")
    device = cuda_device()
    model = build_model(config["runtime"], config["model_seed"], c["representation"]).to(device)
    model.load_state_dict(torch.load(out / "source_final.pt", map_location=device, weights_only=True), strict=True)
    q = row_data(query, "Q", c["representation"])
    with (out / "predictions.npz").open("xb") as f:
        np.savez(f, ids=q[1], logits=predict(model, q, device))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "pilot", "smoke", "predict"))
    parser.add_argument("config", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--arm", choices=ROWS, default="S0")
    parser.add_argument("--representation", choices=(0, 1), type=int, default=0)
    parser.add_argument("--query", type=Path)
    a = parser.parse_args()
    c = read_json(a.config)
    if a.mode == "predict":
        if a.query is None:
            parser.error("predict requires --query")
        prediction(a.source, a.out, a.query, c)
    else:
        train(a.source, a.out, c, a.arm, a.representation, a.mode == "pilot", a.mode == "smoke")
