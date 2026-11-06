"""Compare fixed and held-out action probes for frozen source DURA models."""
import argparse
import os
from pathlib import Path
import time

import numpy as np
import torch

from experiments.pamf_phase1_flow import experiment as core
from experiments.pamf_phase1_flow.anchor_source_smoke import frozen_dura
from experiments.phase1_dura_innovation_probe.source_probe import source_inputs
from experiments.phase1_dura_orthogonal.model import InternalDuraRecovery
from experiments.phase1_dura_orthogonal.oracle_action_source_probe import true_action
from experiments.phase1_dura_orthogonal.train_source import SCHEMA, TRAIN_EPOCHS, FIT_SIZE

SOURCE = '20261106-phase1-dura-sim-action-source-s2026092701-r01'
SOURCE_COMMIT = '3df74f4faa51b34fbfff408ca50bf76198d93c40'
TRAIN_PROBE_SEED = 2026110614
HOLDOUT_PROBE_SEED = 2026110615
HOLDOUT_DRAWS = 8


def per_record_probe_error(estimated, reference, probe):
    values = []
    for value in (probe, 1j*probe):
        actual, target = estimated(value), reference(value)
        scale = target.abs().square().mean(-1).clamp_min(1e-6)
        values.append(torch.log1p((actual-target).abs().square().mean(-1)/scale))
    return torch.stack(values, dim=1).mean(1)


def load_candidate(root, device):
    folder = root/'runs'/SOURCE
    summary = core.read(folder/'source_summary.json')
    state = torch.load(folder/'final.pt', map_location='cpu', weights_only=False)
    expected = dict(commit=SOURCE_COMMIT, architecture='dura_internal', objective='sim_action_guard',
                    epochs=TRAIN_EPOCHS, candidate_dura_trainable=True, source_simulator_action_used=True,
                    source_simulator_action_weight=.01, source_probe_seed=TRAIN_PROBE_SEED,
                    TX_labels_read=0, F_weights_read=0, F_outputs_read=0, V_reads=0,
                    target_reads=0, DURA_frozen=True)
    if (summary.get('status') != 'SOURCE_TRAINED' or summary.get('fit_ids') != FIT_SIZE or
            state.get('schema') != SCHEMA or len(state.get('fit_ids', [])) != FIT_SIZE or
            any(summary.get(k) != v or state.get(k) != v for k, v in expected.items())):
        raise ValueError('Matching fixed source-only action-guided candidate required')
    model = InternalDuraRecovery(frozen_dura(root, device))
    model.load_state_dict(state['model'], strict=True)
    return model.to(device).eval().requires_grad_(False).dura


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Unique source diagnostic output required')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    if len(visible) != 1 or not visible[0].strip() or not torch.cuda.is_available():
        raise ValueError('Exactly one GPU must be selected')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ids, rx, scene, _, received, _ = source_inputs(args.root)
    folder = args.root/'runs/20261019-phase1-restoration-learned-action-source-s2026092701-r07/cache/epochs/19'
    meta = core.read(folder/'complete.json')
    if (meta.get('count') != core.COUNT or meta.get('namespace') != 'learned-action/train/epoch19'
            or any(meta.get(k) != 0 for k in ('TX_labels_read', 'F_outputs_read', 'target_reads'))):
        raise ValueError('Approved matching source simulator cache required')
    kernel = np.load(folder/'kernel.npy', mmap_mode='r', allow_pickle=False)
    coefficients = np.load(folder/'coefficients.npy', mmap_mode='r', allow_pickle=False)
    if (kernel.shape != (core.COUNT, 2, 64) or coefficients.shape != (core.COUNT, 4, 256)
            or kernel.dtype != np.float32 or coefficients.dtype != np.float32):
        raise ValueError('Complete matching source actions required')
    original, candidate = frozen_dura(args.root, 'cuda:0'), load_candidate(args.root, 'cuda:0')
    train_generator = torch.Generator().manual_seed(TRAIN_PROBE_SEED)
    fixed_parts = torch.randn((1, 2, 256), generator=train_generator).to('cuda:0')
    fixed = torch.complex(fixed_parts[:, 0], fixed_parts[:, 1])
    holdout_generator = torch.Generator().manual_seed(HOLDOUT_PROBE_SEED)
    chunks = []
    started = time.monotonic()
    with torch.no_grad():
        for start in range(0, len(ids), 32):
            selected = ids[start:start+32]
            x = core.tensor(received[selected], 'cuda:0')
            h, c = core.tensor(kernel[selected], 'cuda:0'), core.tensor(coefficients[selected], 'cuda:0')
            baseline_action = original.estimate_context(x)[1]
            candidate_action = candidate.estimate_context(x)[1]
            reference = true_action(type(baseline_action), h, c)
            fixed_probe = fixed.expand(len(x), -1)
            a_fixed = per_record_probe_error(baseline_action, reference, fixed_probe)
            b_fixed = per_record_probe_error(candidate_action, reference, fixed_probe)
            a_holdout, b_holdout = [], []
            for _ in range(HOLDOUT_DRAWS):
                parts = torch.randn((len(x), 2, 256), generator=holdout_generator).to('cuda:0')
                probe = torch.complex(parts[:, 0], parts[:, 1])
                a_holdout.append(per_record_probe_error(baseline_action, reference, probe))
                b_holdout.append(per_record_probe_error(candidate_action, reference, probe))
            values = torch.stack((a_fixed, b_fixed, torch.stack(a_holdout).mean(0),
                                  torch.stack(b_holdout).mean(0)), dim=1)
            if not torch.isfinite(values).all():
                raise FloatingPointError('Finite source action errors required')
            if start == 0:
                args.out.mkdir(parents=True)
                print('ACTION_HOLDOUT_SOURCE_SMOKE_PASS', flush=True)
            chunks.append(values.cpu())
    values = torch.cat(chunks)
    groups = {}
    for group, receivers in (('RX4_6', (4, 6)), ('RX8', (8,))):
        for view in core.VIEWS:
            mask = np.isin(rx, receivers) & (scene == view)
            if mask.sum() != 48:
                raise ValueError('Complete source receiver/view cells required')
            cell = values[torch.tensor(mask.tolist(), dtype=torch.bool)]
            groups[group+'/'+view] = dict(count=len(cell), mean_log_response_error=cell.mean(0).tolist(),
                improved_fixed=int((cell[:, 1] < cell[:, 0]).sum()),
                improved_holdout=int((cell[:, 3] < cell[:, 2]).sum()))
    core.write(args.out/'result.json', dict(status='ACTION_HOLDOUT_SOURCE_COMPLETE', commit=args.commit,
        records=len(ids), order=['original_fixed', 'candidate_fixed', 'original_holdout', 'candidate_holdout'],
        train_probe_seed=TRAIN_PROBE_SEED, holdout_probe_seed=HOLDOUT_PROBE_SEED,
        holdout_draws_per_record=HOLDOUT_DRAWS, groups=groups, elapsed_seconds=time.monotonic()-started,
        TX_labels_read=0, F_weights_read=0, F_outputs_read=0, V_reads=0, target_IQ_reads=0,
        target_truth_reads=0, model_updates=0,
        caveat='Source simulator action response only; held-out probes are not held-out data or identity accuracy.'))
    print('ACTION_HOLDOUT_SOURCE_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
