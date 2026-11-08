"""Train joint IQ recovery with a fixed source-only rotating channel schedule."""
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
from experiments.phase1_dura_orthogonal.source_diverse_views import diverse_epochs
from experiments.phase1_dura_orthogonal.train_source import load_source_training, prepare, WIDE_FIT_SIZE
from experiments.phase1_joint_particle.particle import JointParticleRecovery, PARTICLES, STEPS
from experiments.phase1_joint_particle.train import (SCHEMA, MODEL_SEED, ORDER_SEED, NOISE_SEED,
    EPOCHS, BATCH, WIDTH, _actions, _loss)


def run(root, out, commit, *, device='cuda:0', smoke=False):
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
    pilot, _, _, _, _, _ = source_inputs(root)
    cache, fresh, clean, fit_all = load_source_training(root, pilot, fit_size=WIDE_FIT_SIZE)
    fit = fit_all
    if smoke:
        roles = np.load(cache/'train/roles.npy', allow_pickle=False)[fit_all]
        positions = np.concatenate((np.flatnonzero(roles == 'L_s')[:32],
                                    np.flatnonzero(roles == 'U_s')[:32]))
        if len(positions) != 64:
            raise ValueError('Mixed legal source roles required for smoke')
        fit = fit_all[positions]
    dura = frozen_dura(root, device)
    if any(p.requires_grad for p in dura.parameters()):
        raise ValueError('Original DURA must be frozen')
    from experiments.pamf_action_restore.operators import Action
    model = JointParticleRecovery(width=WIDTH).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.001)
    order_rng = np.random.default_rng(ORDER_SEED)
    noise_rng = torch.Generator(device=device).manual_seed(NOISE_SEED)
    out.mkdir(parents=True)
    started = time.monotonic()
    losses, counts_by_epoch, schedule = [], [], []
    for epoch in range(1, (2 if smoke else EPOCHS+1)):
        epochs = diverse_epochs(epoch)
        data, counts = prepare(root, dura, clean, fresh, fit, device, BATCH, source_epochs=epochs)
        values = tuple(value.to(device) for value in data)
        received, anchor, innovation, variance, clean_target = values
        action_fields = _actions(dura, received, variance)
        schedule.append(list(epochs))
        counts_by_epoch.append(counts)
        print('JOINT_DIVERSE_SOURCE_VIEWS', epoch, epochs, len(received), flush=True)
        permutation = order_rng.permutation(len(received))
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
        print('JOINT_DIVERSE_SOURCE_EPOCH', epoch, 'loss', losses[-1], 'steps', steps,
              'elapsed', round(time.monotonic()-started, 1), flush=True)
    common = dict(schema=SCHEMA, commit=commit, epochs=1 if smoke else EPOCHS,
                  fit_ids=len(fit_all), actual_fit_ids=len(fit), views_per_record=3,
                  source_view_schedule=schedule, source_counts_by_epoch=counts_by_epoch,
                  width=WIDTH, steps=STEPS, inference_particles=PARTICLES, training_particles=1,
                  model_seed=MODEL_SEED, order_seed=ORDER_SEED, noise_seed=NOISE_SEED,
                  source_loss='normalized IQ + .02 normalized DURA change',
                  TX_labels_read=0, F_weights_read=0, F_outputs_read=0, V_reads=0,
                  target_reads=0, DURA_frozen=True, elapsed_seconds=time.monotonic()-started,
                  losses=losses)
    name = 'smoke.pt' if smoke else 'final.pt'
    with (out/name).open('xb') as handle:
        torch.save(dict(**common, model=model.state_dict()), handle)
    status = 'SOURCE_TRAINING_SMOKE_PASS' if smoke else 'SOURCE_TRAINED'
    core.write(out/('smoke.json' if smoke else 'source_summary.json'), dict(status=status, **common))
    print('JOINT_DIVERSE_'+status, commit, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    run(args.root, args.out, args.commit, device=args.device, smoke=args.smoke)


if __name__ == '__main__':
    main()
