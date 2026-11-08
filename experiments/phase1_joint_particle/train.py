"""Fixed-budget, F-free source training for joint stochastic IQ proposals."""
import argparse
import os
from pathlib import Path
import time

import numpy as np
import torch

from experiments.pamf_phase1_flow import experiment as core
from experiments.pamf_phase1_flow.anchor_source_smoke import frozen_dura
from experiments.phase1_dura_innovation_probe.source_probe import source_inputs
from experiments.phase1_dura_orthogonal.controller import training_environments
from experiments.phase1_dura_orthogonal.source_context_cache import load_cache
from experiments.phase1_dura_orthogonal.train_source import load_source_training, WIDE_FIT_SIZE
from experiments.phase1_joint_particle.particle import JointParticleRecovery, PARTICLES, STEPS

SCHEMA = 'phase1_joint_particle_source_v1'
MODEL_SEED = 2026110812
ORDER_SEED = 2026110813
NOISE_SEED = 2026110814
EPOCHS = 20
BATCH = 32
WIDTH = 192
GUARD_WEIGHT = .02


def _data(root, device, smoke, cache_path):
    ids, _, _, _, _, _ = source_inputs(root)
    source_cache, _, _, fit = load_source_training(root, ids, fit_size=WIDE_FIT_SIZE)
    values, counts = load_cache(cache_path, fit, BATCH, device, deterministic=True)
    if smoke:
        roles = np.load(source_cache / 'train/roles.npy', allow_pickle=False)
        role_fit = roles[fit]
        positions = np.concatenate((np.flatnonzero(role_fit == 'L_s')[:32],
                                    np.flatnonzero(role_fit == 'U_s')[:32]))
        if len(positions) != 64:
            raise ValueError('Mixed legal source roles required for smoke')
        index = torch.tensor(positions.tolist(), dtype=torch.long)
        values = tuple(value[index] for value in values)
    return tuple(value.to(device) for value in values), fit, counts


@torch.no_grad()
def _actions(dura, received, variance):
    """Cache DURA action tensors for exactly the fixed source observations."""
    fields = ([], [], [])
    for start in range(0, len(received), BATCH):
        context = dura.estimate_context(received[start:start+BATCH])
        if start == 0:
            torch.testing.assert_close(context[2], variance[start:start+BATCH], rtol=1e-4, atol=1e-8)
        action = context[1]
        for dest, tensor in zip(fields, (action.kernel, action.main, action.image)):
            dest.append(tensor.detach())
        if (start+BATCH) % 4096 == 0:
            print('JOINT_SOURCE_ACTION_PREP', min(start+BATCH, len(received)), flush=True)
    return tuple(torch.cat(parts, 0) for parts in fields)


def _loss(output, anchor, clean):
    energy = clean.square().mean((1, 2)).clamp_min(1e-8)
    error = (output-clean).square().mean((1, 2))/energy
    change = (output-anchor).square().mean((1, 2))/energy
    return (error + GUARD_WEIGHT*change).mean()


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
    dura = frozen_dura(root, device)
    if any(p.requires_grad for p in dura.parameters()):
        raise ValueError('Original DURA must be frozen')
    started = time.monotonic()
    action_fields = _actions(dura, received, variance)
    from experiments.pamf_action_restore.operators import Action
    model = JointParticleRecovery(width=WIDTH).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.001)
    order_rng = np.random.default_rng(ORDER_SEED)
    noise_rng = torch.Generator(device=device).manual_seed(NOISE_SEED)
    out.mkdir(parents=True)
    losses = []
    for epoch in range(1, (2 if smoke else EPOCHS+1)):
        permutation = order_rng.permutation(len(received))
        if smoke:
            permutation = np.tile(permutation, 3)
        total, visits, steps = 0., 0, 0
        for start in range(0, len(permutation), BATCH):
            index = torch.tensor(permutation[start:start+BATCH].tolist(), dtype=torch.long, device=device)
            y, base, direction, v, target = (value[index] for value in values)
            action = Action(*(field[index] for field in action_fields))
            latent = torch.randn((len(index), 1, 2, 256), generator=noise_rng, device=device)
            output = model(y, base, direction, v, action, latent=latent, validate=False)
            loss = _loss(output, base, target)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite source loss')
            if steps == 0 and (model.output.weight.grad is None or
                               not model.output.weight.grad.abs().sum() > 0):
                raise ValueError('Initial source gradient must be finite and nonzero')
            optim.step()
            total += float(loss.detach())*len(index)
            visits += len(index)
            steps += 1
        losses.append(total/visits)
        print('JOINT_SOURCE_EPOCH', epoch, 'loss', losses[-1], 'steps', steps,
              'elapsed', round(time.monotonic()-started, 1), flush=True)
    common = dict(schema=SCHEMA, commit=commit, epochs=1 if smoke else EPOCHS,
                  fit_ids=len(fit), views_per_record=3, width=WIDTH, steps=STEPS,
                  inference_particles=PARTICLES, training_particles=1,
                  model_seed=MODEL_SEED, order_seed=ORDER_SEED, noise_seed=NOISE_SEED,
                  source_loss='normalized IQ + .02 normalized DURA change',
                  TX_labels_read=0, F_weights_read=0, F_outputs_read=0, V_reads=0,
                  target_reads=0, DURA_frozen=True, source_counts=counts,
                  source_counts_scope='complete_cached_three_views',
                  actual_source_view_count=len(received),
                  elapsed_seconds=time.monotonic()-started, losses=losses)
    if smoke:
        with (out/'smoke.pt').open('xb') as handle:
            torch.save(dict(**common, model=model.state_dict()), handle)
        core.write(out/'smoke.json', dict(status='SOURCE_TRAINING_SMOKE_PASS', **common))
        print('JOINT_SOURCE_TRAINING_SMOKE_PASS', flush=True)
        return
    with (out/'final.pt').open('xb') as handle:
        torch.save(dict(**common, model=model.state_dict()), handle)
    core.write(out/'source_summary.json', dict(status='SOURCE_TRAINED', **common))
    print('JOINT_SOURCE_TRAINED', commit, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    run(args.root, args.out, args.commit, args.cache, device=args.device, smoke=args.smoke)


if __name__ == '__main__':
    main()
