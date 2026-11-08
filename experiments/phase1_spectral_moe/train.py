"""F-free source-only training on records held out of both frozen experts."""
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
from experiments.phase1_dura_orthogonal.signal_gate_oof_train import select_oof_ids
from experiments.phase1_dura_orthogonal.train_source import load_source_training, prepare
from experiments.phase1_joint_particle.particle import JointParticleRecovery
from experiments.phase1_joint_particle.train import SCHEMA as JOINT_SCHEMA, _loss
from experiments.phase1_residual_diffusion.model import ResidualDiffusion
from experiments.phase1_residual_diffusion.train import SCHEMA as DIFFUSION_SCHEMA
from experiments.phase1_residual_diffusion.source_probe import load_model, JOINT, JOINT_COMMIT, DIFFUSION, DIFFUSION_COMMIT
from experiments.phase1_spectral_moe.model import SpectralMixture

SCHEMA = 'phase1_spectral_three_expert_oof_v1'
EPOCHS = 20
WIDTH = 128
MODEL_SEED = 2026110823
ORDER_SEED = 2026110824

@torch.no_grad()
def cache_experts(dura, joint, diffusion, values):
    collected = [[], []]
    for start in range(0, len(values[0]), 8):
        y, anchor, direction, variance = (v[start:start+8].to('cuda:0') for v in values[:4])
        context = dura.estimate_context(y)
        torch.testing.assert_close(context[2], variance, rtol=1e-4, atol=1e-8)
        for dest, result in zip(collected, (joint(y, anchor, direction, variance, context[1]),
                                            diffusion(y, anchor, direction, variance))):
            if not torch.isfinite(result).all():
                raise FloatingPointError('Finite frozen expert output required')
            dest.append(result.cpu())
        if (start+len(y)) % 4096 == 0:
            print('SPECTRAL_EXPERT_CACHE', start+len(y), flush=True)
    return tuple(torch.cat(parts).to('cuda:0') for parts in collected)

def run(root, out, commit, smoke=False):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    training_environments('cuda:0', os.environ)
    if out.exists() or not torch.cuda.is_available():
        raise ValueError('Unique output and selected GPU required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    pilot, _, _, _, _, _ = source_inputs(root)
    cache, fresh, clean, parent_fit = load_source_training(root, pilot, fit_size=32768)
    rx = np.load(cache/'train/rx.npy', allow_pickle=False)
    day = np.load(cache/'train/day.npy', allow_pickle=False)
    roles = np.load(cache/'train/roles.npy', allow_pickle=False)
    fit = select_oof_ids(rx, day, roles, parent_fit, pilot)
    if len(parent_fit) != 32768 or len(fit) != 17632:
        raise ValueError('Fixed parent fit and disjoint source records required')
    dura = frozen_dura(root, 'cuda:0')
    joint = load_model(root, JOINT, JOINT_SCHEMA, JOINT_COMMIT, 3, JointParticleRecovery)
    diffusion = load_model(root, DIFFUSION, DIFFUSION_SCHEMA, DIFFUSION_COMMIT, 8, ResidualDiffusion)
    selected = fit[:64] if smoke else fit
    out.mkdir(parents=True)
    started = time.monotonic()
    values, counts = prepare(root, dura, clean, fresh, selected, 'cuda:0', 32, source_epochs=(21, 22, 23))
    experts = cache_experts(dura, joint, diffusion, values)
    del dura, joint, diffusion
    values = tuple(v.to('cuda:0') for v in values)
    torch.manual_seed(MODEL_SEED)
    model = SpectralMixture(WIDTH).to('cuda:0').train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.001)
    rng = np.random.default_rng(ORDER_SEED)
    losses = []
    for epoch in range(1, 2 if smoke else EPOCHS+1):
        order = rng.permutation(len(values[0]))
        total = torch.zeros((), device='cuda:0')
        for start in range(0, len(order), 128):
            index = torch.tensor(order[start:start+128].tolist(), dtype=torch.long, device='cuda:0')
            y, anchor, _, variance, clean_iq = (v[index] for v in values)
            output = model(y, anchor, experts[0][index], experts[1][index], variance)
            loss = _loss(output, anchor, clean_iq)
            if not torch.isfinite(loss): raise FloatingPointError('Finite source loss required')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            total += loss.detach()*len(index)
        torch.cuda.synchronize()
        losses.append(float(total.cpu())/len(order))
        print('SPECTRAL_SOURCE_EPOCH', epoch, 'loss', losses[-1], 'elapsed', round(time.monotonic()-started, 1), flush=True)
    common = dict(schema=SCHEMA, commit=commit, epochs=1 if smoke else EPOCHS, fit_ids=17632,
        actual_fit_ids=len(selected), actual_source_view_count=len(values[0]), views_per_record=3,
        width=WIDTH, joint=JOINT, joint_commit=JOINT_COMMIT, diffusion=DIFFUSION,
        diffusion_commit=DIFFUSION_COMMIT, parent_fit_disjoint=True, pilot_disjoint=True,
        TX_labels_read=0, F_weights_read=0, F_outputs_read=0, V_reads=0, target_reads=0,
        DURA_frozen=True, source_counts=counts, losses=losses, elapsed_seconds=time.monotonic()-started)
    with (out/('smoke.pt' if smoke else 'final.pt')).open('xb') as f:
        torch.save(dict(**common, model=model.eval().cpu().state_dict()), f)
    core.write(out/('smoke.json' if smoke else 'source_summary.json'), dict(
        status='SOURCE_TRAINING_SMOKE_PASS' if smoke else 'SOURCE_TRAINED', **common))
    print('SPECTRAL_SOURCE_DONE', commit, flush=True)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--commit', required=True)
    p.add_argument('--device', choices=('cuda:0',), default='cuda:0')
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    run(args.root, args.out, args.commit, args.smoke)

if __name__ == '__main__': main()
