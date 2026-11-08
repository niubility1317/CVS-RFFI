"""Fixed-budget conditional diffusion training on source-only paired IQ."""
import argparse
import os
from pathlib import Path
import time
import numpy as np
import torch

from experiments.pamf_phase1_flow import experiment as core
from experiments.pamf_phase1_flow.anchor_source_smoke import frozen_dura
from experiments.phase1_dura_orthogonal.controller import training_environments
from experiments.phase1_joint_particle.train import _data, _loss, BATCH, EPOCHS
from experiments.phase1_residual_diffusion.model import (ResidualDiffusion, forward_noise, scale_of,
    WIDTH, STEPS, PARTICLES, LATENT_SEED)

SCHEMA = 'phase1_residual_diffusion_source_v1'
MODEL_SEED = 2026110817
ORDER_SEED = 2026110813
TIME_SEED = 2026110818
NOISE_SEED = 2026110819


def run(root, out, commit, cache_path, *, device='cuda:0', smoke=False):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    training_environments(device, os.environ)
    if device != 'cuda:0' or not torch.cuda.is_available():
        raise ValueError('One explicitly selected CUDA GPU required')
    if out.exists():
        raise FileExistsError('A new source output root is required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(MODEL_SEED)
    values, fit, counts = _data(root, device, smoke, cache_path)
    received, anchor, innovation, variance, clean = values
    if smoke:
        dura = frozen_dura(root, device)
        if any(p.requires_grad for p in dura.parameters()):
            raise ValueError('Original DURA must be frozen')
        with torch.no_grad():
            context = dura.estimate_context(received[:BATCH])
            torch.testing.assert_close(context[2], variance[:BATCH], rtol=1e-4, atol=1e-8)
    model = ResidualDiffusion(width=WIDTH).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.001)
    order_rng = np.random.default_rng(ORDER_SEED)
    time_rng = torch.Generator(device=device).manual_seed(TIME_SEED)
    noise_rng = torch.Generator(device=device).manual_seed(NOISE_SEED)
    out.mkdir(parents=True)
    started, losses = time.monotonic(), []
    for epoch in range(1, (2 if smoke else EPOCHS+1)):
        permutation = order_rng.permutation(len(received))
        if smoke:
            permutation = np.tile(permutation, 3)
        total, visits, steps = 0., 0, 0
        for start in range(0, len(permutation), BATCH):
            index = torch.tensor(permutation[start:start+BATCH].tolist(), dtype=torch.long, device=device)
            y, base, direction, v, target = (value[index] for value in values)
            scale = scale_of(base)
            time_values = torch.rand((len(index),), generator=time_rng, device=device)
            noise = torch.randn((len(index), 2, 256), generator=noise_rng, device=device)
            noisy = forward_noise((target-base)/scale, time_values, noise)
            estimate = model.estimate(y, base, direction, v, noisy, time_values)
            loss = _loss(base+scale*estimate, base, target)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite source diffusion loss')
            if steps == 0 and (model.output.weight.grad is None or not model.output.weight.grad.abs().sum() > 0):
                raise ValueError('Initial diffusion gradient must be finite and nonzero')
            optim.step()
            total += float(loss.detach())*len(index)
            visits += len(index)
            steps += 1
        losses.append(total/visits)
        print('DIFFUSION_SOURCE_EPOCH', epoch, 'loss', losses[-1], 'steps', steps,
              'elapsed', round(time.monotonic()-started, 1), flush=True)
    common = dict(schema=SCHEMA, commit=commit, epochs=1 if smoke else EPOCHS, fit_ids=len(fit),
                  actual_source_view_count=len(received), views_per_record=3, width=WIDTH,
                  steps=STEPS, inference_particles=PARTICLES, latent_seed=LATENT_SEED,
                  model_seed=MODEL_SEED, order_seed=ORDER_SEED, time_seed=TIME_SEED, noise_seed=NOISE_SEED,
                  source_loss='normalized IQ + .02 normalized DURA change at random diffusion time',
                  TX_labels_read=0, F_weights_read=0, F_outputs_read=0, V_reads=0,
                  target_reads=0, DURA_frozen=True, source_counts=counts,
                  source_counts_scope='complete_cached_three_views', losses=losses,
                  elapsed_seconds=time.monotonic()-started)
    with (out/('smoke.pt' if smoke else 'final.pt')).open('xb') as handle:
        torch.save(dict(**common, model=model.state_dict()), handle)
    status = 'SOURCE_TRAINING_SMOKE_PASS' if smoke else 'SOURCE_TRAINED'
    core.write(out/('smoke.json' if smoke else 'source_summary.json'), dict(status=status, **common))
    print('DIFFUSION_'+status, commit, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--smoke', action='store_true')
    a = parser.parse_args()
    run(a.root, a.out, a.commit, a.cache, device=a.device, smoke=a.smoke)


if __name__ == '__main__':
    main()
