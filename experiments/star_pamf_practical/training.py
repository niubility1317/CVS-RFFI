"""Train the Smooth STAR E dual-BN, capped-pseudo-label recipe from scratch."""
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from experiments.cric_dura_comparison.io import write
from experiments.practical_full_2x2.cache import probability
from experiments.practical_full_2x2.compat import tensor
from .data import CLASSES, SOURCE, load_pairs, load_source
from .model import Domain_Classifier, Ni_model_2part
from .schedule import StratifiedCycleSampler, smooth_lr_factor, update_inference_ema
from .temperature import fit_temperature
from .view_bn import ViewBatchNorm, encode_view, view_context


@dataclass(frozen=True)
class Recipe:
    epochs: int = 100
    label_batch: int = 32
    unlabeled_batch: int = 256
    frozen_teacher_epoch: int = 5
    labeled_ce_ramp_end: int = 20
    labeled_ce_floor: float = 0.10
    satellite_ce_weight: float = 0.68
    pseudo_threshold: float = 0.90
    pseudo_weight: float = 9.0
    pseudo_sg_weight: float = 0.45
    pseudo_budget: float = 3.87260055
    smooth_batch_seed: int = 92001
    practical_application_seed: int = 91001
    smooth_lr_warmup: int = 5
    smooth_lr_initial: float = 0.20
    smooth_lr_start: int = 15
    smooth_lr_end: int = 100
    smooth_lr_floor: float = 0.10
    average_from: int = 76
    average_decay: float = 0.99
    lr_model: float = 0.001
    lr_domain: float = 0.01
    grad_clip: float = 10.0
    domain_ce_weight: float = 0.10
    domain_uniform_weight: float = 0.10
    orthogonal_weight: float = 0.01


RECIPE = Recipe()


def encode_iq(x):
    # The native model expects two 256-point time slots and their padded FFTs.
    # The locked ManySig view supplies one record. The unused slot is zero.
    iq = torch.complex(x[:, 0], x[:, 1])
    padded = F.pad(iq, (0, 256))
    spectrum = torch.fft.fft(padded, dim=-1, norm="ortho")
    missing = torch.zeros_like(padded)
    encoded = torch.cat((padded, missing, spectrum, missing), dim=-1)
    return torch.stack((encoded.real, encoded.imag), dim=1)


def forward(model, x, bank="clean", stochastic=False):
    features = encode_view(model.encoder, encode_iq(x), bank)
    return features, model.classifier(features[:, 128:], stochastic=stochastic)


def paired_forward(model, clean, practical):
    clean_features = encode_view(model.encoder, encode_iq(clean), "clean")
    practical_features = encode_view(model.encoder, encode_iq(practical), "sg")
    features = torch.cat((clean_features, practical_features))
    logits = model.classifier(features[:, 128:], stochastic=True)
    return clean_features, logits.chunk(2)


def batch(pair, indices, rng, p, device):
    clean, sg, scene = pair
    active = rng.random(len(indices)) < p
    selected = np.where(active, scene[indices], -1)
    satellite = np.where(active[:, None, None], sg[indices], clean[indices])
    return tensor(clean[indices], device), tensor(satellite, device), np.bincount(selected + 1, minlength=4)


def pseudo_loss(clean, practical, teacher, cfg):
    confidence, identity = clean.detach().softmax(1).max(1)
    accepted = torch.minimum(confidence, teacher.amax(1)) >= cfg.pseudo_threshold
    weight = accepted.to(clean.dtype)
    mass = weight.mean() * (1.0 + cfg.pseudo_sg_weight) * 0.5
    cap = (cfg.pseudo_budget / (cfg.pseudo_weight * mass).clamp_min(1e-12)).clamp_max(1.0)
    clean_ce = F.cross_entropy(clean, identity, reduction="none")
    sg_ce = F.cross_entropy(practical, identity, reduction="none")
    loss = cap * (weight * (clean_ce + cfg.pseudo_sg_weight * sg_ce)).mean() * 0.5
    return loss, accepted.sum()


def objective(model, discriminator, labeled, unlabeled, teacher, temperature, epoch, cfg):
    lx, ls, label = labeled
    ux, us, domain = unlabeled
    _, (clean_l, practical_l) = paired_forward(model, lx, ls)
    feature, (clean_u, practical_u) = paired_forward(model, ux, us)
    ce = F.cross_entropy(clean_l, label) + cfg.satellite_ce_weight * F.cross_entropy(practical_l, label)
    relevant, invariant = feature[:, :128], feature[:, 128:]
    domain_ce = F.cross_entropy(discriminator(relevant), domain)
    uniform = torch.full((len(ux), len(SOURCE)), 1.0 / len(SOURCE), device=ux.device)
    domain_kl = F.kl_div(discriminator(invariant).log_softmax(1), uniform, reduction="batchmean")
    gram = F.normalize(invariant, dim=1).T @ F.normalize(relevant, dim=1)
    orth = gram.square().sum() / invariant.numel()
    pseudo = ce.new_zeros(())
    accepted = torch.zeros((), dtype=torch.long, device=ce.device)
    scale = 1.0
    if epoch > cfg.frozen_teacher_epoch:
        progress = min(1.0, (epoch - cfg.frozen_teacher_epoch) /
                       (cfg.labeled_ce_ramp_end - cfg.frozen_teacher_epoch))
        scale -= (1.0 - cfg.labeled_ce_floor) * progress
        with torch.no_grad():
            teacher_prob = (forward(teacher, ux)[1] / temperature).softmax(1)
        pseudo, accepted = pseudo_loss(clean_u, practical_u, teacher_prob, cfg)
    loss = (scale * ce + cfg.pseudo_weight * pseudo + cfg.domain_ce_weight * domain_ce
            + cfg.domain_uniform_weight * domain_kl + cfg.orthogonal_weight * orth)
    return loss, accepted


@torch.no_grad()
def reestimate_bn(model, pair, cfg, device):
    clean, sg, _ = pair
    order = np.random.default_rng(cfg.smooth_batch_seed).permutation(len(clean))
    banks = [layer for layer in model.encoder.modules() if isinstance(layer, ViewBatchNorm)]
    model.eval()
    for view, iq in (("clean", clean), ("sg", sg)):
        with view_context(model.encoder, view):
            for layer in banks:
                layer.reset_bank()
                layer.momentum = None
                layer.train()
            for start in range(0, len(order), 100):
                model.encoder(encode_iq(tensor(iq[order[start:start + 100]], device)))
    for layer in banks:
        layer.momentum = 0.1
    model.eval()


@torch.no_grad()
def calibrate(model, validation, device):
    iq, meta = validation
    logits = [forward(model, tensor(iq[i:i + 128], device))[1].cpu()
              for i in range(0, len(iq), 128)]
    return fit_temperature(torch.cat(logits), tensor(meta["y"]))


def new_model(device):
    return Ni_model_2part(len(CLASSES)).to(device)


def train(c, device="cuda:0"):
    from copy import deepcopy
    from time import monotonic

    cfg = RECIPE
    torch.manual_seed(c["model_seed"])
    source = load_source(c["source_cache"])
    pairs = load_pairs(c["pairs"], source)
    root = Path(c["source_output"])
    root.mkdir(parents=True, exist_ok=False)
    write(root / "resolved_config.json", dict(c, recipe=asdict(cfg), checkpoint_init="scratch",
                                             domain_classes=list(SOURCE)))
    model = new_model(device)
    discriminator = Domain_Classifier(len(SOURCE)).to(device)
    model_optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr_model)
    domain_optimizer = torch.optim.Adam(discriminator.parameters(), lr=cfg.lr_domain)
    meta_l, meta_u = source["L_s"][1], source["U_s"][1]
    strata = np.column_stack((meta_l["y"], meta_l["domain"], meta_l["rx_i"]))
    sampler = StratifiedCycleSampler(tensor(strata), cfg.smooth_batch_seed)
    teacher = average = None
    temperature = 1.0
    steps = 0
    for epoch in range(1, cfg.epochs + 1):
        started = monotonic()
        model.train()
        sampler.set_epoch(epoch)
        indices_l = np.asarray(list(sampler))
        indices_u = np.random.default_rng(cfg.smooth_batch_seed + epoch).permutation(len(pairs["U_s"][0]))
        augmentation = np.random.default_rng(cfg.practical_application_seed + epoch)
        p = probability((epoch - 1) / cfg.epochs)
        domain_seen = 0
        selected = torch.zeros((), dtype=torch.long, device=device)
        total_loss = torch.zeros((), device=device)
        l_scenes = np.zeros(4, dtype=np.int64)
        u_scenes = np.zeros(4, dtype=np.int64)
        batches = 0
        factor = smooth_lr_factor(epoch, cfg)
        model_optimizer.param_groups[0]["lr"] = cfg.lr_model * factor
        domain_optimizer.param_groups[0]["lr"] = cfg.lr_domain * factor
        for step, start in enumerate(range(0, len(indices_l), cfg.label_batch)):
            li = indices_l[start:start + cfg.label_batch]
            ui = indices_u[step * cfg.unlabeled_batch:(step + 1) * cfg.unlabeled_batch]
            if len(ui) != cfg.unlabeled_batch:
                raise ValueError("Not enough distinct U_s examples for the fixed schedule")
            lx, ls, lcount = batch(pairs["L_s"], li, augmentation, p, device)
            ux, us, ucount = batch(pairs["U_s"], ui, augmentation, p, device)
            model_optimizer.zero_grad(set_to_none=True)
            domain_optimizer.zero_grad(set_to_none=True)
            loss, accepted = objective(
                model, discriminator, (lx, ls, tensor(meta_l["y"][li], device)),
                (ux, us, tensor(meta_u["domain"][ui], device)), teacher, temperature, epoch, cfg)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip, error_if_nonfinite=True)
            nn.utils.clip_grad_norm_(discriminator.parameters(), cfg.grad_clip, error_if_nonfinite=True)
            model_optimizer.step()
            domain_optimizer.step()
            if epoch >= cfg.average_from:
                average = update_inference_ema(average, model, cfg.average_decay)
            selected += accepted
            total_loss += loss.detach()
            domain_seen += len(ui)
            l_scenes += lcount
            u_scenes += ucount
            batches += 1
        steps += batches
        telemetry = dict(epoch=epoch, updates=steps, L_seen=len(indices_l), U_seen=domain_seen,
                         U_accepted=int(selected), L_practical=l_scenes.tolist(),
                         U_practical=u_scenes.tolist(), loss=float(total_loss / batches),
                         seconds=monotonic() - started)
        write(root / f"epoch_{epoch:03}.json", telemetry)
        print("STAR_TRAIN_EPOCH", telemetry, flush=True)
        if epoch == cfg.frozen_teacher_epoch:
            teacher = deepcopy(model).eval().requires_grad_(False)
            reestimate_bn(teacher, pairs["L_s"], cfg, device)
            temperature, details = calibrate(teacher, source["V"], device)
            write(root / "source_teacher_calibration.json", details)
    if average is None:
        raise RuntimeError("Final fixed E100 inference average is missing")
    reestimate_bn(average, pairs["L_s"], cfg, device)
    checkpoint = root / "final_e100_ema_source_bn.pth"
    torch.save(dict(state_dict=average.state_dict(), run_id=c["run_id"], row_id=c["row_id"],
                    model_seed=c["model_seed"], epoch=cfg.epochs, updates=steps), checkpoint)
    write(root / "complete.json", dict(status="SOURCE_TRAINED_FROZEN", checkpoint=str(checkpoint),
                                      epoch=cfg.epochs, model_seed=c["model_seed"], updates=steps,
                                      target_access=False, U_TX_truth_read=False))
