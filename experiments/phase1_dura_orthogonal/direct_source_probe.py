"""Compare received, direct model and DURA on frozen source IQ only."""
import argparse
import os
from pathlib import Path
import time

import numpy as np
import torch

from experiments.pamf_phase1_flow import experiment as core
from experiments.pamf_phase1_flow.anchor_source_smoke import frozen_dura
from experiments.phase1_dura_innovation_probe.source_probe import source_inputs
from experiments.phase1_dura_orthogonal.model import DualDomainRecovery
from experiments.phase1_dura_orthogonal.train_source import SCHEMA, DUAL_DOMAIN_EPOCHS, FIT_SIZE

RUN = '20261106-phase1-direct-dualdomain-source-s2026092701-r01'
SOURCE_COMMIT = 'cd21119f3b026986a7a650dadbde1e45481c5bb8'


def pair_errors(received, restored, anchor, clean):
    if not (received.shape == restored.shape == anchor.shape == clean.shape and
            received.ndim == 3 and received.shape[1:] == (2, 256)):
        raise ValueError('Matching source IQ windows required')
    energy = clean.square().mean((1, 2)).clamp_min(1e-8)
    return torch.stack([((value-clean).square().mean((1, 2))/energy)
                        for value in (received, restored, anchor)], dim=1)


def load_direct(root, device):
    folder = root/'runs'/RUN
    summary = core.read(folder/'source_summary.json')
    state = torch.load(folder/'final.pt', map_location='cpu', weights_only=False)
    expected = dict(commit=SOURCE_COMMIT, architecture='dual_domain', objective='iq',
                    epochs=DUAL_DOMAIN_EPOCHS, DURA_training_reads=0, TX_labels_read=0,
                    F_weights_read=0, F_outputs_read=0, V_reads=0, target_reads=0, DURA_frozen=True)
    if (summary.get('status') != 'SOURCE_TRAINED' or summary.get('fit_ids') != FIT_SIZE or
            state.get('schema') != SCHEMA or len(state.get('fit_ids', [])) != FIT_SIZE or
            any(summary.get(k) != v or state.get(k) != v for k, v in expected.items())):
        raise ValueError('The fixed source-only direct checkpoint is required')
    model = DualDomainRecovery(width=state['width'])
    model.load_state_dict(state['model'], strict=True)
    return model.to(device).eval().requires_grad_(False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Unique source output required')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    if len(visible) != 1 or not visible[0].strip() or not torch.cuda.is_available():
        raise ValueError('Exactly one GPU must be selected')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = load_direct(args.root, 'cuda:0')
    dura = frozen_dura(args.root, 'cuda:0')
    ids, rx, scene, _, received, clean = source_inputs(args.root)
    chunks = []
    started = time.monotonic()
    with torch.no_grad():
        for start in range(0, len(ids), 32):
            selected = ids[start:start+32]
            x, y = core.tensor(received[selected], 'cuda:0'), core.tensor(clean[selected], 'cuda:0')
            values = pair_errors(x, model(x), dura(x), y)
            if not torch.isfinite(values).all():
                raise FloatingPointError('Finite real-checkpoint source outputs required')
            if start == 0:
                args.out.mkdir(parents=True)
                print('DIRECT_SOURCE_SMOKE_PASS', flush=True)
            chunks.append(values.cpu())
    values = torch.cat(chunks)
    groups = {}
    for group, receivers in (('RX4_6', (4, 6)), ('RX8', (8,))):
        for view in core.VIEWS:
            mask = np.isin(rx, receivers) & (scene == view)
            if mask.sum() != 48:
                raise ValueError('Complete source cells required')
            cell = values[torch.tensor(mask.tolist(), dtype=torch.bool)]
            groups[group+'/'+view] = dict(count=len(cell), mean_nmse=cell.mean(0).tolist(),
                direct_better_than_received=int((cell[:, 1] < cell[:, 0]).sum()),
                direct_better_than_dura=int((cell[:, 1] < cell[:, 2]).sum()))
    core.write(args.out/'result.json', dict(status='DIRECT_SOURCE_DIAGNOSTIC_COMPLETE', commit=args.commit,
        records=len(ids), order=['received', 'direct', 'dura'], groups=groups,
        elapsed_seconds=time.monotonic()-started, TX_labels_read=0, F_weights_read=0, F_outputs_read=0,
        V_reads=0, target_IQ_reads=0, target_truth_reads=0, model_updates=0,
        caveat='Previously examined source IQ; not independent of DURA and not identity accuracy.'))
    print('DIRECT_SOURCE_DIAGNOSTIC_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
