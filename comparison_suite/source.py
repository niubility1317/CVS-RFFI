"""Scratch, source-only training with exact physical-role matching.

No target loader, target score or target-dependent checkpoint selection exists in
this entry point. All native DA losses belong to Phase2, not source initialization.
"""
from __future__ import annotations
import argparse
import copy
import csv
import json
import random
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from comparison_suite.models import METHODS, factory, embedding, classifier_logits, architecture_metadata


def _json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def source_args(config):
    data = config.get("data", {})
    pkl = config.get("wisig_pkl", config.get("dataset", data.get("wisig_pkl", data.get("path"))))
    if not pkl:
        raise ValueError("data.wisig_pkl is required")
    if any(config.get(k) for k in ("checkpoint", "resume", "initial_checkpoint", "teacher_checkpoint")):
        raise ValueError("Source initialization must be scratch; checkpoint inheritance is prohibited")
    return SimpleNamespace(source_contract=config["source_contract"], use_source_ssl_split=True,
        wisig_pkl=pkl, output_dir=config["output_root"], wisig_train_rxs="1,3,4,6,8",
        wisig_train_days="1,2,3", wisig_test_rxs="0,2,5,7,9,10,11",
        wisig_labeled_ratio=.07, wisig_unlabeled_ratio=.63, wisig_source_val_ratio=.30,
        wisig_split_seed=392005, wisig_equalized=1, wisig_out_len=256,
        wisig_rms_normalize=True, wisig_domain="rx")


def _groups(dataset):
    # Physical-role indexes are inspected without reading U_s truth or IQ.
    groups = {}
    for i, physical_index in enumerate(dataset.indices):
        y = int(dataset.base.index[physical_index].tx_i)
        groups.setdefault(y, []).append(i)
    return groups


def source_episode(dataset, groups, *, rng, n_way, k_shot, query_per_class):
    from baselines.common.cvs_data import collate_cvs_dict
    available = [y for y, rows in groups.items() if len(rows) >= k_shot + query_per_class]
    if len(available) < n_way:
        raise ValueError("Insufficient disjoint labeled source support/query for episode")
    support, query = [], []
    for y in rng.sample(available, n_way):
        ids = rng.sample(groups[y], k_shot + query_per_class)
        support.extend(dataset[i] for i in ids[:k_shot])
        query.extend(dataset[i] for i in ids[k_shot:])
    return collate_cvs_dict(support), collate_cvs_dict(query)


def source_loss(model, method, x, y, receiver, config, outputs=None):
    outputs = model(x) if outputs is None else outputs
    if method == "feature_separation":
        from paper_reproduction.feature_separation_crossrx.losses import feature_separation_loss
        return feature_separation_loss(outputs, y, receiver,
            lambda_similarity=float(config.get("lambda_similarity", 1.)),
            lambda_tx_entropy=float(config.get("lambda_tx_entropy", 1.)),
            lambda_rx_entropy=float(config.get("lambda_rx_entropy", 1.)))
    if method == "orthogonal":
        from paper_reproduction.orthogonal_incremental_sei.losses import base_training_loss
        from paper_reproduction.orthogonal_incremental_sei.pseudo_targets import perturb_pseudo_targets
        targets = model.pseudo_targets
        assigned = {c: targets[c] for c in range(len(model.classifier_weight))}
        perturbed = perturb_pseudo_targets(targets, noise_range=float(config.get("pseudo_noise", .01)))
        return base_training_loss(outputs["features"], y, assigned, targets, perturbed)
    ce = F.cross_entropy(outputs["logits" if method == "dadda" else "tx_logits"], y)
    return ce, {"source_ce": ce.detach()}


@torch.no_grad()
def collect_prototypes(model, method, loader, num_classes, device):
    model.eval()
    sums, counts = None, torch.zeros(num_classes, device=device)
    for batch in loader:
        z = embedding(model, method, batch["iq"].to(device))
        y = batch["label"].to(device)
        if sums is None:
            sums = z.new_zeros(num_classes, z.shape[1])
        sums.index_add_(0, y, z)
        counts.index_add_(0, y, torch.ones_like(y, dtype=z.dtype))
    if sums is None or bool((counts == 0).any()):
        raise ValueError("Every source class must have labeled source samples")
    return sums / counts[:, None]


@torch.no_grad()
def validate(model, method, loader, device):
    model.eval()
    correct = count = 0
    total = 0.
    for batch in loader:
        y = batch["label"].to(device)
        logits = classifier_logits(model, method, batch["iq"].to(device))
        total += float(F.cross_entropy(logits, y, reduction="sum"))
        correct += int((logits.argmax(1) == y).sum())
        count += len(y)
    return {"source_val_loss": total / count if count else None,
            "source_val_accuracy": correct / count if count else None, "source_val_count": count}


def _fisher(model, method, loader, device):
    """Explicit CSIL corefix: empirical per-example supervised source Fisher.

    Author exp(gradient²) overflowed even in FP64. This named numerical
    correction uses mean CE gradient², with no exp, clamp or identity fallback.
    """
    model = copy.deepcopy(model).double().eval()
    count = 0
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    for b in loader:
        x, y = b["iq"].to(device=device, dtype=torch.float64), b["label"].to(device)
        for i in range(len(y)):
            model.zero_grad(set_to_none=True)
            F.cross_entropy(classifier_logits(model, method, x[i:i+1]), y[i:i+1]).backward()
            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    fisher[name].add_(parameter.grad.detach().square())
            count += 1
    if count == 0:
        raise ValueError("Empty source Fisher role")
    fisher = {name: (value / count).cpu() for name, value in fisher.items()}
    if any(not bool(torch.isfinite(value).all()) for value in fisher.values()):
        raise FloatingPointError("CSIL empirical per-example source Fisher is nonfinite; no clamping or fallback")
    model.zero_grad(set_to_none=True)
    return fisher


def train(config):
    from baselines.common.practical_source import build_contract_split, PracticalResidualAugment
    from baselines.common.cvs_data import make_cvs_loader
    method = config["method"]
    if method not in METHODS:
        raise ValueError(method)
    out = Path(config["output_root"])
    if (out / "last.pt").exists() or (out / "epoch_metrics.jsonl").exists():
        raise FileExistsError("Existing run artifacts may not be overwritten")
    out.mkdir(parents=True, exist_ok=True)
    args = source_args(config)
    seed = int(config["model_seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(config.get("device", "cpu"))
    split = build_contract_split(args)
    loaders = {role: make_cvs_loader(ds, batch_size=int(config.get("batch_size", 128)),
        shuffle=role == "train", num_workers=int(config.get("num_workers", 0)), device=device,
        drop_last=role == "train") for role, ds in (("train", split.train), ("export", split.train), ("val", split.val))}
    model_kwargs = config.get("model_kwargs", {})
    model = factory(method, split.num_classes, split.num_receivers, **model_kwargs).to(device)
    initialization = {"status": "SCRATCH", "checkpoint": None, "ancestors": [], "model_seed": seed,
        "source_contract": str(config["source_contract"]), "physical_roles": "EXACT_MATCH",
        "selection": "fixed_last_epoch", "target_access": False, "scratch_only": True,
        "checkpoint_sources": [], "target_contact": False}
    _json(out / "initialization.json", initialization)
    resolved = dict(config, architecture=architecture_metadata(method), source_counts=split.split_info["counts"],
        source_roles={"L_s": "TX and receiver supervised training", "U_s": "TX hidden; unused by these source initializers", "V": "source evaluation only"},
        augmentation={"route": "residual_noeq", "mode": "post_sync", "equalization_enabled": False,
        "schedule": "E1-40 high .30; E41-90 mid/urban .60; E91-200 all .80", "sat_ce_start": 80, "lambda_sat_cls": .68},
        optimizer=config.get("optimizer", "Adam"), total_parameters=sum(p.numel() for p in model.parameters()))
    resolved.update(epochs=int(config.get("epochs", 200)), batch_size=int(config.get("batch_size", 128)),
        lr=float(config.get("lr", .001)), grad_clip=config.get("grad_clip", None),
        num_workers=int(config.get("num_workers", 0)), augmentation_seed=int(config.get("augmentation_seed", seed)),
        receiver_seed=int(config.get("receiver_seed", 2027)), data_seed=int(config.get("data_seed", seed)),
        source_mechanism_weights={"source_ce": 1., "satellite_ce": .68,
            "similarity": float(config.get("lambda_similarity", 1.)) if method == "feature_separation" else None,
            "tx_entropy": float(config.get("lambda_tx_entropy", 1.)) if method == "feature_separation" else None,
            "rx_entropy": float(config.get("lambda_rx_entropy", 1.)) if method == "feature_separation" else None},
        episodic={"n_way": int(config.get("n_way", split.num_classes)), "k_shot": int(config.get("k_shot", 3)),
            "query_per_class": int(config.get("query_per_class", 2)), "steps_per_epoch": int(config.get("steps_per_epoch", 50)),
            "metric": "euclidean"} if method == "protonet" else None)
    _json(out / "resolved_config.json", resolved)
    lr = float(config.get("lr", .001))
    if str(config.get("optimizer", "Adam")).lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=float(config.get("momentum", .9)), weight_decay=float(config.get("weight_decay", 0.)))
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=float(config.get("weight_decay", 0.)))
    augment = PracticalResidualAugment(fs_hz=float(config.get("fs_hz", 25e6)), seed=int(config.get("augmentation_seed", seed)),
        receiver_seed=int(config.get("receiver_seed", 2027)))
    epochs = int(config.get("epochs", 200))
    groups = _groups(split.train) if method == "protonet" else None
    rng = random.Random(int(config.get("data_seed", seed)))
    step_total = 0
    log = (out / "training.log").open("w", encoding="utf-8")
    log.write("RESOLVED_CONFIG " + json.dumps(resolved, ensure_ascii=False) + "\n")
    print("RESOLVED_CONFIG " + json.dumps(resolved, ensure_ascii=False), flush=True)
    try:
        with (out / "epoch_metrics.jsonl").open("w", encoding="utf-8") as epoch_log, (out / "step_metrics.jsonl").open("w", encoding="utf-8") as step_log, (out / "epoch_metrics.csv").open("w", newline="", encoding="utf-8") as csvfile:
            writer = None
            for epoch in range(1, epochs + 1):
                tic = time.perf_counter()
                augment.set_epoch(epoch)
                model.train()
                sums, steps = {}, 0
                source_exposure, satellite_exposure = 0, 0
                if method == "protonet":
                    batches = range(int(config.get("steps_per_epoch", 50)))
                else:
                    batches = loaders["train"]
                for b in batches:
                    optimizer.zero_grad(set_to_none=True)
                    if method == "protonet":
                        from paper_reproduction.protonet_cda.model import prototypical_nll
                        support, query = source_episode(split.train, groups, rng=rng,
                            n_way=int(config.get("n_way", split.num_classes)), k_shot=int(config.get("k_shot", 3)), query_per_class=int(config.get("query_per_class", 2)))
                        sx, sy, qx, qy = (support["iq"].to(device), support["label"].to(device), query["iq"].to(device), query["label"].to(device))
                        # One concatenated clean/satellite forward keeps the source view contract explicit.
                        satellite = augment(qx, metadata=query["meta"])
                        zs, zq, zsat = torch.split(embedding(model, method, torch.cat((sx, qx, satellite))), (len(sx), len(qx), len(qx)))
                        clean, _ = prototypical_nll(zs, sy, zq, qy, metric="euclidean")
                        sat, _ = prototypical_nll(zs, sy, zsat, qy, metric="euclidean")
                        terms = {"episode_nll": clean.detach()}
                        batch_source_rows, batch_satellite_rows = len(sx)+len(qx), len(qx)
                    else:
                        x, y = b["iq"].to(device), b["label"].to(device)
                        receiver = b["receiver"].to(device)
                        satellite = augment(x, metadata=b["meta"])
                        # Satellite CE is the only satellite training objective.
                        combined = model(torch.cat((x, satellite)))
                        clean_out = {key: value[:len(x)] for key, value in combined.items() if isinstance(value, torch.Tensor)}
                        clean, terms = source_loss(model, method, x, y, receiver, config, outputs=clean_out)
                        sat = F.cross_entropy(combined["logits" if method == "dadda" else "tx_logits"][len(x):], y)
                        batch_source_rows, batch_satellite_rows = len(x), len(x)
                    sat_weight = .68 if epoch >= 80 else 0.
                    loss = clean + sat_weight * sat
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError("Nonfinite source training loss")
                    loss.backward()
                    clipping = config.get("grad_clip")
                    grad = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf") if clipping is None else float(clipping))
                    optimizer.step()
                    steps += 1
                    step_total += 1
                    source_exposure += batch_source_rows
                    satellite_exposure += batch_satellite_rows
                    record = {"epoch": epoch, "step": step_total, "loss": float(loss.detach()), "satellite_ce": float(sat.detach()),
                        "source_samples": batch_source_rows, "satellite_samples": batch_satellite_rows,
                        "satellite_weight": sat_weight, "learning_rate": optimizer.param_groups[0]["lr"], "gradient_norm": float(grad),
                        "method": method, "mechanism_state": "episodic" if method == "protonet" else "native_source_objective",
                        **{k: float(v) for k, v in terms.items()}}
                    step_log.write(json.dumps(record, allow_nan=False) + "\n")
                    line = "STEP " + json.dumps(record, allow_nan=False)
                    log.write(line + "\n")
                    if bool(config.get("print_steps", False)):
                        print(line, flush=True)
                    for key, value in record.items():
                        if isinstance(value, (int, float)) and key not in ("epoch", "step"):
                            sums[key] = sums.get(key, 0.) + value
                if steps == 0:
                    raise ValueError("No source optimizer steps")
                if method == "protonet":
                    proto = collect_prototypes(model, method, loaders["export"], split.num_classes, device)
                    model.source_prototypes.copy_(proto)
                    model.prototypes_ready.fill_(True)
                metrics = dict(epoch=epoch, optimizer_steps=steps, optimizer_steps_total=step_total,
                    source_sample_exposure=source_exposure, satellite_sample_exposure=satellite_exposure,
                    **{k: v / steps for k, v in sums.items()}, **validate(model, method, loaders["val"], device),
                    elapsed_seconds=time.perf_counter() - tic, peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None)
                epoch_log.write(json.dumps(metrics, allow_nan=False) + "\n")
                epoch_log.flush()
                if writer is None:
                    writer = csv.DictWriter(csvfile, fieldnames=list(metrics))
                    writer.writeheader()
                writer.writerow(metrics)
                csvfile.flush()
                line = "EPOCH " + json.dumps(metrics, allow_nan=False)
                print(line, flush=True)
                log.write(line + "\n")
                log.flush()
            prototypes = collect_prototypes(model, method, loaders["export"], split.num_classes, device).cpu()
            base_state = {"method": method, "prototypes": prototypes, "old_prototypes": prototypes,
                "source_role": "L_s", "source_contract": str(config["source_contract"]), "query_access": False,
                "source_class_ids": list(range(split.num_classes)), "computed_before_target_adaptation": True}
            if method == "csil":
                base_state.update(fc_weight=model.fc_bf_fp.weight.detach().cpu(), fc_bias=model.fc_bf_fp.bias.detach().cpu(),
                    old_fingerprints=model.fingerprints.detach().cpu(), fisher=_fisher(model, method, loaders["export"], device))
                base_state["fisher_objective"] = "empirical_source_label_CE_grad_squared"
                base_state["fisher_precision"] = "float64"
                base_state["method_version"] = "csil_official_repo_corefix_empirical_fisher"
                base_state["fisher_summary"] = {"finite": True,
                    "max": max(float(v.max()) for v in base_state["fisher"].values()),
                    "min": min(float(v.min()) for v in base_state["fisher"].values())}
                log.write("SOURCE_FISHER " + json.dumps(base_state["fisher_summary"]) + "\n")
            if method == "mopc_hr":
                base_state.update(classifier_weight=model.fc.weight.detach().cpu(), classifier_bias=model.fc.bias.detach().cpu())
            if method == "orthogonal":
                base_state.update(pseudo_targets=model.pseudo_targets.detach().cpu(), classifier_weight=model.classifier_weight.detach().cpu())
            torch.save(base_state, out / "base_state.pt")
            contract = json.loads((out / "source_contract.json").read_text(encoding="utf-8"))
            checkpoint = {"model": model.state_dict(), "state_dict": model.state_dict(), "method": method,
                "model_kwargs": model_kwargs, "num_classes": split.num_classes, "num_receivers": split.num_receivers,
                "classes": contract["classes"], "class_mapping": {c: i for i, c in enumerate(contract["classes"])},
                "epoch": epochs, "config": resolved, "initialization": initialization, "source_contract": contract,
                "selection": "fixed_last_epoch", "source_prototypes": prototypes, "architecture": architecture_metadata(method)}
            torch.save(checkpoint, out / "last.pt")
            _json(out / "completion.json", {"status": "SOURCE_TRAINED", "epoch": epochs,
                "steps": step_total, "checkpoint": str(out / "last.pt"), "target_access": False, "target_evaluated": False})
    finally:
        log.close()
    return out / "last.pt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    train(json.loads(Path(args.config).read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
