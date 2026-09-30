"""Inductive support-only adapters; held-out query is accepted only by score().

MRIOR/DADDA are supervised extensions of their original UDA objectives. RadioNet
ports the author's *alternating* ADA update, not a generic GRL replacement.
All changes to the paper task (256 IQ, K-shot target support, scratch source
training and final rather than selected weights) belong in the caller's record.
"""
from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from paper_reproduction.cvs_aligned.supervised_da import (
    dadda_sda_objective, mrior_sda_batch_step,
)
from paper_reproduction.receiver_agnostic_twostage_uda.losses import (
    dann_loss, stage2_lmmd_objective,
)


def _scalar(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu()) if value.numel() == 1 else value.detach().cpu().tolist()
    return value


def _output(model, x, **kwargs):
    out = model(x, **kwargs)
    if isinstance(out, torch.Tensor):
        if not hasattr(model, "embedding"):
            raise ValueError("tensor-output models must expose embedding(x)")
        return {"tx_logits": out, "features": model.embedding(x)}
    return out


def features(model, x):
    """Extract the task embedding without any query-dependent fitting."""
    if hasattr(model, "embedding"):
        return model.embedding(x)
    out = model(x)
    for key in ("features", "tx_features", "local_features", "embedding"):
        if key in out:
            return out[key]
    raise ValueError("model does not expose a task embedding")


def _logits(out):
    return out["tx_logits"] if "tx_logits" in out else out["logits"]


def _same_pool(x, size, stride):
    padding = max((math.ceil(x.shape[-1] / stride) - 1) * stride + size - x.shape[-1], 0)
    return F.max_pool1d(F.pad(x, (padding // 2, padding - padding // 2), value=-float("inf")), size, stride)


class RadioNetADAHead(nn.Module):
    """af_model.py conv classifier/discriminator, returning pre-activation logits."""
    def __init__(self, out_dim):
        super().__init__()
        self.conv = nn.Conv1d(1, 64, 4, padding="same")
        self.bn1 = nn.BatchNorm1d(64, eps=1e-3, momentum=.01)
        self.dense = nn.Linear(64, 128)
        self.bn2 = nn.BatchNorm1d(128, eps=1e-3, momentum=.01)
        self.final = nn.Linear(128, out_dim)
        self.bn_last = nn.BatchNorm1d(out_dim, eps=1e-3, momentum=.01)
        self.drop = nn.Dropout(.4)
        self.apply(_keras_initialize)

    def forward(self, z):
        x = F.softsign(self.bn1(self.conv(z.unsqueeze(1))))
        x = self.drop(_same_pool(x, 4, 1)).mean(-1)
        x = self.drop(F.softsign(self.bn2(self.dense(x))))
        return self.bn_last(self.final(x))


def _keras_initialize(module):
    if isinstance(module, (nn.Conv1d, nn.Linear)):
        nn.init.xavier_uniform_(module.weight)
        nn.init.zeros_(module.bias)


class RadioNetADANet(nn.Module):
    """Author ADA DF topology (BN, GAP and FeaturesVec), distinct from DF FT."""
    def __init__(self, num_classes, input_len=256, embedding_dim=64):
        super().__init__()
        self.input_len = input_len
        self.blocks = nn.ModuleList()
        widths = (2, 32, 64, 128, 256)
        for i in range(4):
            activation = nn.ELU if i == 0 else nn.ReLU
            self.blocks.append(nn.Sequential(
                nn.Conv1d(widths[i], widths[i+1], 8, padding="same"), activation(),
                nn.Conv1d(widths[i+1], widths[i+1], 8, padding="same"), activation(),
                nn.BatchNorm1d(widths[i+1], eps=1e-3, momentum=.01),
            ))
        self.drop = nn.Dropout(.1)
        self.projection = nn.Linear(256, embedding_dim)
        self.tx_classifier = RadioNetADAHead(num_classes)
        self.apply(_keras_initialize)

    def embedding(self, x):
        for i, block in enumerate(self.blocks):
            x = _same_pool(block(x), 8, 4)
            if i < 3:
                x = self.drop(x)
        return self.projection(x.mean(-1))

    def forward(self, x):
        z = self.embedding(x)
        return {"features": z, "tx_logits": self.tx_classifier(z)}


class _ActiveClasses(nn.Module):
    """Select source head columns, retaining native feature/estimate branches."""
    def __init__(self, model, columns):
        super().__init__()
        self.base = model
        self.register_buffer("columns", torch.tensor(columns, dtype=torch.long))

    @property
    def estimate_network(self):
        return self.base.estimate_network

    def embedding(self, x):
        return features(self.base, x)

    def forward(self, x, **kwargs):
        out = dict(_output(self.base, x, **kwargs))
        for name in ("tx_logits", "logits"):
            if name in out:
                out[name] = out[name].index_select(1, self.columns)
        return out


@dataclass
class AdaptationState:
    model: nn.Module
    classes: tuple[int, ...]
    decision: str = "native"
    support_features: torch.Tensor | None = None
    support_labels: torch.Tensor | None = None
    weight: torch.Tensor | None = None
    knn_k: int = 1
    resources: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)
    snapshot: dict[str, torch.Tensor] = field(default_factory=dict)

    @torch.no_grad()
    def score(self, query_x):
        """Per-sample competition over ALL classes; never changes fitted state."""
        self.model.eval()
        device = next(self.model.parameters()).device
        x = query_x.to(device=device, dtype=torch.float32)
        if self.decision == "native":
            return _logits(_output(self.model, x))
        z = features(self.model, x)
        if self.decision == "ncm":
            return F.normalize(z, dim=1) @ self.weight.T
        if self.decision == "protonet":
            return -torch.cdist(z, self.weight).square()
        if self.decision == "ridge":
            return torch.cat([z, torch.ones_like(z[:, :1])], 1) @ self.weight
        if self.decision == "linear":
            return F.linear(z, self.weight[0], self.weight[1])
        if self.decision == "knn":
            # Author default: cosine, distance weights, k=nShot. Exact zero
            # distances exclusively vote, as sklearn's distance-weighted kNN.
            distances = (1 - F.normalize(z, dim=1) @ F.normalize(self.support_features, dim=1).T).clamp_min(0)
            d, indices = distances.topk(self.knn_k, dim=1, largest=False, sorted=True)
            zero = d == 0
            weights = torch.where(zero.any(1, keepdim=True), zero.to(d.dtype), d.clamp_min(1e-12).reciprocal())
            votes = torch.zeros(z.shape[0], len(self.classes), device=device)
            votes.scatter_add_(1, self.support_labels[indices], weights)
            return votes
        raise RuntimeError(f"unknown frozen decision: {self.decision}")


def radionet_ada_batch_step(model, discriminator, source_x, source_y, support_x,
                            optimizer_ec, optimizer_d, *, class_weight=4., domain_weight=4.):
    """Port af_classifier.py alternating labels/weight-restoration semantics.

    E/C minimizes true domain labels while D is frozen. D minimizes the flipped
    labels while E/C is frozen. Reversing the domain-name convention yields
    the usual adversarial update; joint same-label minimization does not.
    """
    model.train()
    discriminator.train()
    optimizer_ec.zero_grad(set_to_none=True)
    for p in discriminator.parameters():
        p.requires_grad_(False)
    # Author train_on_batch runs dropout/BN in training mode and afterwards
    # restores *all* frozen-network weights, including BN moving statistics.
    # eval() here would substitute running-stat inference for that batch path.
    frozen_discriminator_buffers = {name: value.detach().clone() for name, value in discriminator.named_buffers()}
    xs = torch.cat([source_x, support_x], 0)
    outputs = _output(model, xs)
    ns = len(source_x)
    z = outputs["features"]
    labels = torch.cat([torch.zeros(ns, device=xs.device), torch.ones(len(support_x), device=xs.device)])
    source_ce = F.cross_entropy(_logits(outputs)[:ns], source_y)
    domain_ec = F.binary_cross_entropy_with_logits(discriminator(z).flatten(), labels)
    # Author class sample weights are 1 on source, 0 on target, averaged over
    # the full combined batch, giving ns/(ns+nt) rather than a full source mean.
    combined_loss = class_weight * source_ce * ns / len(xs) + domain_weight * domain_ec
    combined_loss.backward()
    ec_grad = _grad_norm(model)
    optimizer_ec.step()
    with torch.no_grad():
        for name, value in discriminator.named_buffers():
            value.copy_(frozen_discriminator_buffers[name])
    for p in discriminator.parameters():
        p.requires_grad_(True)
    discriminator.train()
    optimizer_d.zero_grad(set_to_none=True)
    model.train()
    frozen_encoder_buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
    with torch.no_grad():
        z = features(model, xs)
        for name, value in model.named_buffers():
            value.copy_(frozen_encoder_buffers[name])
    domain_d = F.binary_cross_entropy_with_logits(discriminator(z).flatten(), 1 - labels)
    domain_d.backward()
    d_grad = _grad_norm(discriminator)
    optimizer_d.step()
    return {"loss": combined_loss.detach(), "source_ce": source_ce.detach(),
            "domain_ec": domain_ec.detach(), "domain_discriminator": domain_d.detach(),
            "ec_gradient_norm": ec_grad, "discriminator_gradient_norm": d_grad,
            "class_weight": class_weight, "domain_weight": domain_weight}


def _grad_norm(model):
    return math.sqrt(sum(float(p.grad.detach().square().sum()) for p in model.parameters() if p.grad is not None))


def _indices(n, batch_size, generator, device):
    if n <= 0:
        raise ValueError("empty fit pool")
    # Repeated support examples are valid K-shot minibatch sampling, never query.
    return torch.randint(n, (max(2, min(batch_size, n)),), generator=generator).to(device)


def _map_labels(y, classes, *, require_all=False):
    y = y.to(dtype=torch.long)
    mapped = torch.full_like(y, -1)
    for i, value in enumerate(classes):
        mapped[y == value] = i
    if (mapped < 0).any():
        raise ValueError("labels outside registered classes")
    if require_all and set(mapped.detach().cpu().tolist()) != set(range(len(classes))):
        raise ValueError("every registered class requires support")
    return mapped


def _freeze_bn(model):
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def fit_adaptation(method, base_model, support_x, support_y, classes, config,
                   source_x=None, source_y=None, metadata=None):
    """Fit one paired old-support task; source replay only for explicit methods.

    support_y and source_y are GLOBAL physical class identifiers. classes fixes
    score-column order; config.source_classes fixes source checkpoint head order.
    Epoch/iteration budgets are fixed before invocation. No early selection.
    """
    aliases = {"radionet_ada": "radionet_ada_knn", "twostage": "twostage_sda",
               "feature_separation": "feature_separation_ft", "linear_probe": "linear"}
    method = aliases.get(method, method)
    accepted = {"mrior_sda", "dadda_sda", "radionet_ada_knn", "radionet_ada_ft_knn",
                "twostage_sda", "feature_separation_ft", "ncm", "protonet_cda",
                "ridge", "linear", "sft", "adabn", "coral_sft"}
    if method not in accepted:
        raise ValueError(f"unimplemented adapter: {method}")
    classes = tuple(int(c) for c in classes)
    if not classes or len(set(classes)) != len(classes):
        raise ValueError("classes must be nonempty and unique")
    config = dict(config)
    device = torch.device(config.get("device", next(base_model.parameters()).device))
    generator = torch.Generator().manual_seed(int(config.get("seed", 0)))
    # fork RNG avoids changing the caller's source-training / next-row RNG.
    rng_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=rng_devices):
        torch.manual_seed(int(config.get("seed", 0)))
        return _fit(method, base_model, support_x, support_y, classes, config,
                    source_x, source_y, metadata, device, generator)


def _fit(method, base, sx, sy, classes, cfg, source_x, source_y, metadata, device, generator):
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    sx = sx.to(device=device, dtype=torch.float32)
    sy = _map_labels(sy.to(device), classes, require_all=True)
    if len(sx) != len(sy) or not torch.isfinite(sx).all():
        raise ValueError("support input/labels invalid")
    model = copy.deepcopy(base).to(device)
    model.eval()
    with torch.no_grad():
        width = _logits(_output(model, sx[:2])).shape[1]
    source_classes = tuple(int(c) for c in cfg.get("source_classes", range(width)))
    if len(source_classes) != width or len(set(source_classes)) != width:
        raise ValueError("source_classes must describe every source classifier column")
    missing = set(classes) - set(source_classes)
    if missing:
        raise ValueError("adaptation API is old-class only; new-class registration uses registration API")
    model = _ActiveClasses(model, [source_classes.index(c) for c in classes]).to(device)
    for p in model.parameters():
        p.requires_grad_(True)
    replay_methods = {"mrior_sda", "dadda_sda", "radionet_ada_knn", "radionet_ada_ft_knn", "twostage_sda", "coral_sft"}
    uses_source = method in replay_methods
    if uses_source:
        if source_x is None or source_y is None:
            raise ValueError(f"{method} requires explicitly permitted source labeled IQ")
        source_y = source_y.to(device=device, dtype=torch.long)
        keep = torch.zeros_like(source_y, dtype=torch.bool)
        for c in classes:
            keep |= source_y == c
        source_x = source_x.to(device=device, dtype=torch.float32)[keep]
        source_y = _map_labels(source_y[keep], classes, require_all=True)
        if not torch.isfinite(source_x).all():
            raise ValueError("source input invalid")
    state = AdaptationState(model, classes, metadata=dict(metadata or {}))
    state.metadata.update({"method": method, "query_used_for_fit": False,
        "query_used_for_selection": False, "target_input_scope": "registered_old_support_only",
        "source_replay": uses_source, "cpl_enabled": False,
        "state_training": "independent_fit_from_source_checkpoint", "checkpoint_policy": "last_iteration",
        "paper_faithful_claim_allowed": False})
    batch_size = int(cfg.get("batch_size", 128))
    epochs = int(cfg.get("epochs", 20))
    if batch_size <= 0 or epochs < 0:
        raise ValueError("invalid batch size/epochs")
    lr = float(cfg.get("lr", .0006 if method == "mrior_sda" else .001))
    logger = cfg.get("logger")
    state.metadata["resolved_config"] = {
        **{key: value for key, value in cfg.items() if key != "logger"},
        "epochs": epochs, "batch_size": batch_size, "lr": lr,
        "classes": list(classes), "source_classes": list(source_classes),
        "support_count": len(sx), "source_replay_count": len(source_x) if uses_source else 0,
        "steps_per_epoch": int(cfg.get("steps_per_epoch", max(1, math.ceil(len(sx)/batch_size)))),
    }
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    def emit(row):
        row = {key: _scalar(value) for key, value in row.items()}
        row.update({"method": method, "learning_rate": lr,
                    "elapsed_seconds": time.perf_counter() - started})
        state.history.append(row)
        if logger:
            logger(dict(row))

    if method in {"ncm", "protonet_cda", "ridge", "linear"}:
        for p in model.parameters():
            p.requires_grad_(False)
        model.eval()
        with torch.no_grad():
            z = features(model, sx)
        if method in {"ncm", "protonet_cda"}:
            means = torch.stack([z[sy == c].mean(0) for c in range(len(classes))])
            state.decision = "ncm" if method == "ncm" else "protonet"
            state.weight = F.normalize(means, dim=1) if method == "ncm" else means
            total_trainable = 0
        elif method == "ridge":
            state.decision = "ridge"
            design = torch.cat([z, torch.ones_like(z[:, :1])], 1).double()
            onehot = F.one_hot(sy, len(classes)).double()
            penalty = torch.eye(design.shape[1], device=device, dtype=torch.float64) * float(cfg.get("ridge", 1.))
            penalty[-1, -1] = 0
            state.weight = torch.linalg.solve(design.T @ design + penalty, design.T @ onehot).float()
            total_trainable = 0
        else:
            head = nn.Linear(z.shape[1], len(classes)).to(device)
            optimizer = torch.optim.Adam(head.parameters(), lr=lr)
            for epoch in range(epochs):
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(head(z), sy)
                loss.backward()
                gradient = _grad_norm(head)
                optimizer.step()
                emit({"epoch": epoch + 1, "loss": loss, "support_ce": loss, "gradient_norm": gradient})
            state.decision = "linear"
            state.weight = (head.weight.detach().clone(), head.bias.detach().clone())
            total_trainable = sum(p.numel() for p in head.parameters())
    elif method == "adabn":
        for p in model.parameters():
            p.requires_grad_(False)
        model.eval()
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.train()
        with torch.no_grad():
            for epoch in range(epochs):
                _output(model, sx)
                emit({"epoch": epoch + 1, "loss": None, "gradient_norm": None, "bn_support_update": True})
        total_trainable = 0
    else:
        if method == "mrior_sda":
            estimate_parameters = list(model.estimate_network.parameters())
            estimate_ids = {id(p) for p in estimate_parameters}
            optimizer_t = torch.optim.Adam(estimate_parameters, lr=float(cfg.get("estimate_lr", lr)))
            optimizer = torch.optim.Adam([p for p in model.parameters() if id(p) not in estimate_ids], lr=lr)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        if method.startswith("radionet_ada"):
            # Default budgets/weights directly from author generate_params().
            lr = float(cfg.get("lr", .00001))
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            discriminator = RadioNetADAHead(1).to(device)
            optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=float(cfg.get("discriminator_lr", .00001)))
            total_trainable += sum(p.numel() for p in discriminator.parameters())
            iterations = int(cfg.get("iterations", 10000))
            if iterations < 0:
                raise ValueError("iterations must be non-negative")
            state.metadata["resolved_config"].update({"iterations": iterations, "lr": lr,
                "discriminator_lr": float(cfg.get("discriminator_lr", .00001)),
                "class_weight": float(cfg.get("class_weight", 4)), "domain_weight": float(cfg.get("domain_weight", 4))})
            state.metadata.update({"ada_update": "author_alternating_flipped_discriminator_labels",
                "knn_metric": "cosine", "knn_weights": "distance", "knn_k_policy": "nShot",
                "support_ft_epochs": int(cfg.get("finetune_epochs", 30)) if method.endswith("ft_knn") else 0})
            for iteration in range(iterations):
                si = _indices(len(source_x), batch_size, generator, device)
                ti = _indices(len(sx), batch_size, generator, device)
                terms = radionet_ada_batch_step(model, discriminator, source_x[si], source_y[si], sx[ti], optimizer, optimizer_d,
                    class_weight=float(cfg.get("class_weight", 4)), domain_weight=float(cfg.get("domain_weight", 4)))
                if not math.isfinite(float(terms["loss"])):
                    raise FloatingPointError("nonfinite RadioNet ADA fit loss")
                emit({"iteration": iteration + 1, "stage": "ADA", **terms})
            if method.endswith("ft_knn"):
                # The author test route replaces the classifier and fine-tunes
                # all layers for 30 support-only epochs before freezing kNN.
                with torch.no_grad():
                    dim = features(model.eval(), sx[:2]).shape[1]
                head = nn.Linear(dim, len(classes)).to(device)
                optimizer_ft = torch.optim.Adam(list(model.parameters()) + list(head.parameters()), lr=float(cfg.get("finetune_lr", .001)))
                for epoch in range(int(cfg.get("finetune_epochs", 30))):
                    model.train()
                    _freeze_bn(model) if len(sx) < 2 else None
                    optimizer_ft.zero_grad(set_to_none=True)
                    loss = F.cross_entropy(head(features(model, sx)), sy)
                    loss.backward()
                    gradient = _grad_norm(model)
                    optimizer_ft.step()
                    emit({"epoch": epoch + 1, "stage": "support_ft", "loss": loss, "support_ce": loss, "gradient_norm": gradient})
            model.eval()
            with torch.no_grad():
                state.support_features = features(model, sx).detach().clone()
            state.support_labels = sy.detach().clone()
            counts = torch.bincount(sy, minlength=len(classes))
            if not bool((counts == counts[0]).all()) and "knn_k" not in cfg:
                raise ValueError("author nShot kNN requires equal support counts or explicit preregistered knn_k")
            state.knn_k = int(cfg.get("knn_k", int(counts[0])))
            if not 1 <= state.knn_k <= len(sx):
                raise ValueError("kNN k must be within support size")
            state.decision = "knn"
        else:
            stages = [("fit", epochs)]
            if method == "twostage_sda":
                stages = [("dann", int(cfg.get("dann_epochs", epochs))),
                          ("lmmd", int(cfg.get("lmmd_epochs", epochs))),
                          ("support_ft", int(cfg.get("finetune_epochs", 20)))]
            for stage, stage_epochs in stages:
                for epoch in range(stage_epochs):
                    epoch_terms = []
                    steps = int(cfg.get("steps_per_epoch", max(1, math.ceil(len(sx) / batch_size))))
                    for _ in range(steps):
                        ti = _indices(len(sx), batch_size, generator, device)
                        tx, ty = sx[ti], sy[ti]
                        if uses_source:
                            si = _indices(len(source_x), batch_size, generator, device)
                            rx, ry = source_x[si], source_y[si]
                        if method == "mrior_sda":
                            terms = mrior_sda_batch_step(model, rx, ry, tx, ty, optimizer_t=optimizer_t, optimizer_ec=optimizer,
                                estimate_steps=int(cfg.get("estimate_steps", 7)), target_ce_weight=float(cfg.get("target_ce_weight", 1)),
                                dvkl_weight=float(cfg.get("dvkl_weight", .005)), mu=float(cfg.get("mu", .5)))
                            terms["gradient_norm"] = _grad_norm(model)
                        else:
                            model.train()
                            optimizer.zero_grad(set_to_none=True)
                            if method == "dadda_sda":
                                terms = dadda_sda_objective(model(rx), model(tx), source_labels=ry, target_support_labels=ty,
                                    target_ce_weight=float(cfg.get("target_ce_weight", 1)), alignment_weight=float(cfg.get("alignment_weight", 1)))
                            elif method == "twostage_sda" and stage == "dann":
                                terms = dann_loss(model(rx), model(tx), ry, domain_weight=float(cfg.get("domain_weight", 1)))
                            elif method == "twostage_sda" and stage == "lmmd":
                                terms = stage2_lmmd_objective(model(rx, return_activations=True), model(tx, return_activations=True), ry,
                                    num_classes=len(classes), lmmd_lambda=float(cfg.get("lmmd_weight", 1)),
                                    lmmd_layers=str(cfg.get("lmmd_layers", "activations")))
                            else:
                                output = _output(model, tx)
                                ce = F.cross_entropy(_logits(output), ty)
                                terms = {"loss": ce, "support_ce": ce}
                                if method == "coral_sft":
                                    source_output = model(rx)
                                    a, b = source_output["features"], output["features"]
                                    a, b = a-a.mean(0), b-b.mean(0)
                                    cov_a = a.T@a/max(len(a)-1, 1)
                                    cov_b = b.T@b/max(len(b)-1, 1)
                                    coral = (cov_a-cov_b).square().mean()/4
                                    source_ce = F.cross_entropy(_logits(source_output), ry)
                                    terms = {"loss": ce+source_ce+float(cfg.get("coral_weight", 1))*coral,
                                             "support_ce": ce, "source_ce": source_ce, "coral": coral}
                            terms["loss"].backward()
                            terms["gradient_norm"] = _grad_norm(model)
                            optimizer.step()
                        if not math.isfinite(float(terms["loss"].detach())):
                            raise FloatingPointError(f"nonfinite {method} fit loss")
                        epoch_terms.append({key: _scalar(v) for key, v in terms.items() if not isinstance(v, torch.Tensor) or v.numel() == 1})
                    merged = {key: sum(row[key] for row in epoch_terms)/len(epoch_terms) for key in epoch_terms[0]}
                    emit({"stage": stage, "epoch": epoch+1, "steps": steps, **merged})
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    state.snapshot = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model_bytes = sum(p.numel()*p.element_size() for p in model.parameters()) + sum(b.numel()*b.element_size() for b in model.buffers())
    decision_bytes = sum(v.numel()*v.element_size() for v in (state.support_features, state.support_labels) if v is not None)
    if isinstance(state.weight, torch.Tensor):
        decision_bytes += state.weight.numel()*state.weight.element_size()
    elif isinstance(state.weight, tuple):
        decision_bytes += sum(v.numel()*v.element_size() for v in state.weight)
    state.resources = {"trainable_parameters_during_fit": total_trainable,
        "fit_seconds": time.perf_counter()-started,
        "peak_cuda_allocated_mb": torch.cuda.max_memory_allocated(device)/2**20 if device.type == "cuda" else None,
        "resident_model_bytes": model_bytes, "resident_decision_bytes": decision_bytes,
        "new_transfer_bytes": None, "cpu_peak_memory_mb": None,
        "hardware": str(torch.cuda.get_device_name(device)) if device.type == "cuda" else "CPU",
        "measurement_scope": "fit+model_copy+decision_state; CUDA global allocator peak"}
    return state
