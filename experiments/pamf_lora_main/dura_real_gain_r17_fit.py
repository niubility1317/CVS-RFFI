"""Fit an R17-guided LoRa frequency fusion using lawful frozen E104."""

from pathlib import Path
from time import monotonic
import argparse

import numpy as np
import torch
from torch.nn import functional as F

from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_lora_main.dura_real_gain_fit import (
    DEVICE, LORA_ACTION, LORA_GAIN, activate_wisig_archive, as_complex, lora_models,
)
from experiments.pamf_lora_main.dura_real_gain_r17 import R17Gain


SEED, STEPS, ROLE_BATCH = 2026112861, 4000, 4


def boot(rt):
    from experiments.cric_dura_comparison.bootstrap import activate, numerical_policy
    numerical_policy()
    activate(rt)
    project = Path(rt['project'])
    release = project / 'releases' / f'{LORA_GAIN}-release'
    plan = read(release / 'runtime.json')
    if plan['run_id'] != LORA_GAIN:
        raise ValueError('Expected the same-contract LoRa real-gain release')
    activate_wisig_archive(release)
    activate_wisig_archive(Path(plan['w24_release_root']))
    _, _, prior = lora_models(project)
    from experiments.pamf_lora_main import w24_joint_pamf as joint
    from experiments.pamf_lora_main import w24_native_training as native
    old = read(project / 'runs' / LORA_GAIN / 'frequency_inphase_native' / 'fit-inputs.json')
    contract, reader, _, tracker = joint.boot(old, DEVICE)
    r17 = native.dura_model(project / 'runs' / LORA_ACTION, contract, DEVICE).eval().requires_grad_(False)
    return joint, native, old, contract, reader.eval().requires_grad_(False), tracker, prior, r17


def fit(rt):
    joint, native, old, contract, reader, tracker, prior, r17 = boot(rt)
    source = Path(old['source_root'])
    labeled = joint.source_role(source, contract, 'L_s')[0]
    unlabeled = joint.source_role(source, contract, 'U_s')[0]
    labels = joint.training_labels(source, contract)
    rng = np.random.default_rng(SEED)
    generator = torch.Generator(device=DEVICE).manual_seed(SEED + 1)
    torch.manual_seed(SEED)
    model = R17Gain(1024).to(DEVICE).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS, eta_min=1e-5)
    accepted = 0
    started = monotonic()
    for step in range(1, STEPS + 1):
        li = rng.choice(len(labeled), ROLE_BATCH, replace=False)
        ui = rng.choice(len(unlabeled), ROLE_BATCH, replace=False)
        raw = as_complex(np.concatenate((labeled[li], unlabeled[ui])))
        clean, observed, _, _ = native.source_pair(raw, generator)
        with torch.no_grad():
            p, action = prior(observed)
            r = r17(observed)
            clean_p, clean_action = prior(clean)
            clean_r = r17(clean)
            clean_aperture, _ = joint.normalize_aperture(tracker.aperture(clean))
            teacher = reader(clean_aperture)[0].softmax(-1)
            confidence, pseudo = teacher[ROLE_BATCH:].max(-1)
            active = confidence >= .95
        restored = model(observed, p, r, action)
        preserved = model(clean, clean_p, clean_r, clean_action)
        if step == 1:
            torch.testing.assert_close(restored, p, rtol=0, atol=0)
        aperture, _ = joint.normalize_aperture(tracker.aperture(restored))
        logits = reader(aperture)[0]
        target = torch.as_tensor(labels[li], device=DEVICE, dtype=torch.long)
        ce = F.cross_entropy(logits[:ROLE_BATCH], target)
        pseudo_loss = (F.cross_entropy(logits[ROLE_BATCH:], pseudo, reduction='none') * active).mean()
        wave = (restored - clean).abs().square().mean()
        keep = (preserved - clean_p).abs().square().mean()
        loss = ce + pseudo_loss + .3 * wave + 1.5 * keep
        opt.zero_grad(set_to_none=True)
        if step == 1:
            grad = torch.autograd.grad(ce, model.net[-1].weight, retain_graph=True)[0].flatten(1).norm(dim=1)
            if not torch.isfinite(grad).all() or not bool((grad > 0).all()):
                raise ValueError('Frozen E104 must supervise both physical and R17 real-gain directions')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite source-only restoration loss')
        opt.step()
        schedule.step()
        accepted += int(active.sum())
        if step == 1 or step % 50 == 0:
            print('LORA_R17_GAIN_FIT', step, *[float(x.detach()) for x in (ce, pseudo_loss, wave, keep, norm)], flush=True)
    torch.cuda.synchronize()
    proof = dict(status='SOURCE_FROZEN', dataset='lora', seed=SEED, updates=STEPS,
                 labeled_exposures=STEPS * ROLE_BATCH, unlabeled_exposures=STEPS * ROLE_BATCH,
                 accepted_pseudolabels=accepted, pseudo_threshold=.95, F_frozen=True,
                 U_truth_reads=0, V_reads=0, target_reads=0,
                 params=sum(p.numel() for p in model.parameters()), seconds=monotonic()-started,
                 peak_cuda_bytes=torch.cuda.max_memory_allocated())
    root = Path(rt['run_root'])
    with (root / 'gain.pt').open('xb') as stream:
        torch.save(dict(weights=model.state_dict(), proof=proof), stream)
    write(root / 'fit.json', proof)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runtime', type=Path)
    args = parser.parse_args()
    fit(read(args.runtime))
