"""F-free DURA/cumulant recovery with matched-budget 30-view source rotation."""
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
from experiments.phase1_dura_orthogonal.train_source import load_source_training, prepare
from experiments.phase1_joint_particle.train import _loss
from experiments.phase1_cumulant_context.model import CumulantContextRecovery, cumulant_loss, CUMULANT_WEIGHT, LAGS
from experiments.phase1_cumulant_context.train import EPOCHS, BATCH, WIDTH, LAYERS, MODEL_SEED, ORDER_SEED

SCHEMA = 'phase1_cumulant_diverse_source_v1'
SCHEDULE = tuple(diverse_epochs(epoch) for epoch in range(1, EPOCHS+1))


def schedule_contract(schedule=SCHEDULE):
    if (len(schedule) != EPOCHS or any(len(group) != 3 for group in schedule)
            or sorted(v for group in schedule for v in group) != sorted(list(range(1,31))*2)):
        raise ValueError('Twenty three-view epochs must use each of 30 views twice')
    return schedule


def run(root, out, commit, smoke=False):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    training_environments('cuda:0', os.environ)
    if out.exists() or not torch.cuda.is_available():
        raise ValueError('Unique output and selected GPU required')
    schedule_contract()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    pilot, _, _, _, _, _ = source_inputs(root)
    source_cache, fresh, clean, fit = load_source_training(root, pilot, fit_size=32768)
    if smoke:
        roles = np.load(source_cache/'train/roles.npy', allow_pickle=False)
        selected = np.concatenate((fit[roles[fit] == 'L_s'][:32], fit[roles[fit] == 'U_s'][:32]))
        if len(selected) != 64:
            raise ValueError('Mixed legal source roles required for smoke')
    else:
        selected = fit
    dura = frozen_dura(root, 'cuda:0')
    if any(p.requires_grad for p in dura.parameters()):
        raise ValueError('Frozen original DURA required')
    torch.manual_seed(MODEL_SEED)
    model = CumulantContextRecovery(width=WIDTH, layers=LAYERS).to('cuda:0').train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.001)
    rng = np.random.default_rng(ORDER_SEED)
    cpu_groups, group_counts, losses = {}, {}, []
    out.mkdir(parents=True)
    started = time.monotonic()
    for epoch in range(1, 2 if smoke else EPOCHS+1):
        group = SCHEDULE[epoch-1]
        if group not in cpu_groups:
            values, counts = prepare(root, dura, clean, fresh, selected, 'cuda:0', BATCH,
                                     architecture='large_context_stable', source_epochs=group)
            if len(values[0]) != len(selected)*3:
                raise ValueError('Three matched source views required')
            cpu_groups[group] = values
            group_counts[str(group)] = counts
        values = tuple(v.to('cuda:0') for v in cpu_groups[group])
        order = rng.permutation(len(values[0]))
        total = torch.zeros((), device='cuda:0')
        for start in range(0, len(order), BATCH):
            idx = torch.tensor(order[start:start+BATCH].tolist(), dtype=torch.long, device='cuda:0')
            received, anchor, direction, variance, target = (v[idx] for v in values)
            result = model(received, anchor, direction, variance)
            loss = _loss(result, anchor, target) + CUMULANT_WEIGHT*cumulant_loss(result, target)
            if not torch.isfinite(loss):
                raise FloatingPointError('Finite source loss required')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if start == 0 and (model.out.weight.grad is None or model.out.weight.grad.abs().sum() <= 0):
                raise ValueError('Finite nonzero source gradient required')
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            total += loss.detach()*len(idx)
        torch.cuda.synchronize()
        losses.append(float(total.cpu())/len(order))
        print('CUMULANT_DIVERSE_SOURCE_EPOCH', epoch, 'views', group, 'loss', losses[-1],
              'elapsed', round(time.monotonic()-started, 1), flush=True)
        del values
    common = dict(schema=SCHEMA, commit=commit, epochs=1 if smoke else EPOCHS,
        fit_ids=len(fit), actual_fit_ids=len(selected), actual_source_view_count=3*len(selected),
        views_per_record=3, source_epochs_schedule=[list(x) for x in (SCHEDULE[:1] if smoke else SCHEDULE)],
        width=WIDTH, layers=LAYERS, model_seed=MODEL_SEED, order_seed=ORDER_SEED,
        lags=list(LAGS), cumulant_weight=CUMULANT_WEIGHT,
        trainable_parameters=sum(p.numel() for p in model.parameters()),
        source_loss='normalized IQ + .02 DURA change + .1 bounded moment',
        TX_labels_read=0, F_weights_read=0, F_outputs_read=0, V_reads=0, target_reads=0,
        DURA_frozen=True, source_counts_by_view_group=group_counts,
        elapsed_seconds=time.monotonic()-started, losses=losses)
    with (out/('smoke.pt' if smoke else 'final.pt')).open('xb') as f:
        torch.save(dict(**common, model=model.eval().cpu().state_dict()), f)
    core.write(out/('smoke.json' if smoke else 'source_summary.json'), dict(
        status='SOURCE_TRAINING_SMOKE_PASS' if smoke else 'SOURCE_TRAINED', **common))
    print('CUMULANT_DIVERSE_SOURCE_DONE', commit, flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--commit', required=True)
    p.add_argument('--device', choices=('cuda:0',), default='cuda:0')
    p.add_argument('--smoke', action='store_true')
    a=p.parse_args()
    run(a.root,a.out,a.commit,a.smoke)

if __name__ == '__main__':
    main()
