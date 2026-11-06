"""Source-only ideal line probe between frozen DURA and a direct recovery model."""
import argparse
import os
from pathlib import Path
import time

import numpy as np
import torch

from experiments.pamf_phase1_flow import experiment as core
from experiments.pamf_phase1_flow.anchor_source_smoke import frozen_dura
from experiments.phase1_dura_innovation_probe.metrics import oracle_line_probe
from experiments.phase1_dura_innovation_probe.source_probe import source_inputs
from experiments.phase1_dura_orthogonal.direct_source_probe import load_direct


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
    model, dura = load_direct(args.root, 'cuda:0'), frozen_dura(args.root, 'cuda:0')
    ids, rx, scene, _, received, clean = source_inputs(args.root)
    chunks = {}
    started = time.monotonic()
    with torch.no_grad():
        for start in range(0, len(ids), 32):
            selected = ids[start:start+32]
            x, y = core.tensor(received[selected], 'cuda:0'), core.tensor(clean[selected], 'cuda:0')
            anchor, direct = dura(x), model(x)
            values = oracle_line_probe(anchor, y, direct-anchor, limit=.5)
            if start == 0:
                args.out.mkdir(parents=True)
                print('COMPLEMENT_SOURCE_SMOKE_PASS', flush=True)
            for name, value in values.items():
                chunks.setdefault(name, []).append(value.cpu())
    arrays = {key: torch.cat(value) for key, value in chunks.items()}
    groups = {}
    for group, receivers in (('RX4_6', (4, 6)), ('RX8', (8,))):
        for view in core.VIEWS:
            mask = np.isin(rx, receivers) & (scene == view)
            if mask.sum() != 48:
                raise ValueError('Complete source cells required')
            idx = torch.tensor(mask.tolist(), dtype=torch.bool)
            base, positive, signed, alpha = (arrays[k][idx] for k in
                    ('anchor_error', 'positive_error', 'signed_error', 'optimal_alpha'))
            groups[group+'/'+view] = dict(count=int(mask.sum()), anchor_nmse=float(base.mean()),
                oracle_positive_nmse=float(positive.mean()), oracle_signed_nmse=float(signed.mean()),
                positive_improved=int((positive < base-1e-9).sum()),
                negative_optimal=int((alpha < -1e-8).sum()), mean_optimal_alpha=float(alpha.mean()))
    core.write(args.out/'result.json', dict(status='COMPLEMENT_SOURCE_DIAGNOSTIC_COMPLETE', commit=args.commit,
        records=len(ids), groups=groups, elapsed_seconds=time.monotonic()-started,
        direction='direct IQ minus DURA IQ, normalized to DURA RMS', optimal_step_bound=.5,
        TX_labels_read=0, F_weights_read=0, F_outputs_read=0, V_reads=0, target_IQ_reads=0,
        target_truth_reads=0, model_updates=0,
        caveat='Per-record step uses source clean IQ and is not deployable; not an identity metric or a general upper bound.'))
    print('COMPLEMENT_SOURCE_DIAGNOSTIC_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
