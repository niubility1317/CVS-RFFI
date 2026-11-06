"""Source-only line-oracle comparison for independent filter directions."""
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
from experiments.phase1_dura_orthogonal.complex_gain_source_probe import load_adaptive
from experiments.phase1_dura_orthogonal.timevarying_inverse import TimeVaryingInverseRecovery
from experiments.phase1_dura_orthogonal.train_source import SCHEMA, DUAL_DOMAIN_EPOCHS, FIT_SIZE

SOURCE = '20261106-phase1-timevarying-inverse-source-s2026092701-r01'
SOURCE_COMMIT = 'afe7e06202c58629416d7571466afc58356a4298'


def load_timevarying(root, device):
    folder = root/'runs'/SOURCE
    summary = core.read(folder/'source_summary.json')
    state = torch.load(folder/'final.pt', map_location='cpu', weights_only=False)
    expected = dict(commit=SOURCE_COMMIT, architecture='timevarying_inverse', objective='iq',
                    epochs=DUAL_DOMAIN_EPOCHS, adaptive_filter_taps=9, DURA_training_reads=0,
                    TX_labels_read=0, F_weights_read=0, F_outputs_read=0, V_reads=0,
                    target_reads=0, DURA_frozen=True)
    if (summary.get('status') != 'SOURCE_TRAINED' or summary.get('fit_ids') != FIT_SIZE or
            state.get('schema') != SCHEMA or len(state.get('fit_ids', [])) != FIT_SIZE or
            any(summary.get(k) != v or state.get(k) != v for k, v in expected.items())):
        raise ValueError('Fixed source-only time-varying checkpoint required')
    model = TimeVaryingInverseRecovery(width=state['width'], taps=9)
    model.load_state_dict(state['model'], strict=True)
    return model.to(device).eval().requires_grad_(False)


def summarize_line(values):
    base, positive, signed, alpha = (values[k] for k in ('anchor_error', 'positive_error', 'signed_error', 'optimal_alpha'))
    if not all(torch.isfinite(v).all() for v in (base, positive, signed, alpha)):
        raise FloatingPointError('Finite source line metrics required')
    return dict(count=len(base), anchor_nmse=float(base.mean()), oracle_positive_nmse=float(positive.mean()),
                oracle_signed_nmse=float(signed.mean()), positive_improved=int((positive < base-1e-9).sum()),
                negative_optimal=int((alpha < -1e-8).sum()), mean_optimal_alpha=float(alpha.mean()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--commit', required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError('Unique source output required')
    if len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')) != 1 or not torch.cuda.is_available():
        raise ValueError('Exactly one CUDA GPU required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ids, rx, scene, _, received, clean = source_inputs(args.root)
    dura, fixed, varying = frozen_dura(args.root, 'cuda:0'), load_adaptive(args.root, 'cuda:0'), load_timevarying(args.root, 'cuda:0')
    chunks = {'fixed': {}, 'varying': {}}
    started = time.monotonic()
    with torch.no_grad():
        for start in range(0, len(ids), 32):
            chosen = ids[start:start+32]
            x, y = core.tensor(received[chosen], 'cuda:0'), core.tensor(clean[chosen], 'cuda:0')
            anchor = dura(x)
            for name, model in (('fixed', fixed), ('varying', varying)):
                values = oracle_line_probe(anchor, y, model(x)-anchor, limit=.5)
                for key, value in values.items():
                    chunks[name].setdefault(key, []).append(value.cpu())
            if start == 0:
                args.out.mkdir(parents=True)
                print('TIMEVARYING_COMPLEMENT_SOURCE_SMOKE_PASS', flush=True)
    arrays = {name: {key: torch.cat(value) for key, value in chunk.items()} for name, chunk in chunks.items()}
    groups = {}
    for group, receivers in (('RX4_6', (4, 6)), ('RX8', (8,))):
        for view in core.VIEWS:
            mask = np.isin(rx, receivers) & (scene == view)
            if mask.sum() != 48:
                raise ValueError('Complete source receiver/view groups required')
            idx = torch.tensor(mask.tolist(), dtype=torch.bool)
            groups[group+'/'+view] = {name: summarize_line({key: value[idx] for key, value in data.items()})
                                      for name, data in arrays.items()}
    core.write(args.out/'result.json', dict(status='TIMEVARYING_COMPLEMENT_SOURCE_COMPLETE', commit=args.commit,
        records=len(ids), groups=groups, elapsed_seconds=time.monotonic()-started,
        direction='independent output minus DURA, normalized to DURA RMS', optimal_step_bound=.5,
        model_updates=0, TX_labels_read=0, F_weights_read=0, F_outputs_read=0, V_reads=0,
        target_IQ_reads=0, target_truth_reads=0,
        caveat='Per-record oracle step uses same-record source clean IQ and is not deployable; not identity accuracy or a general upper bound.'))
    print('TIMEVARYING_COMPLEMENT_SOURCE_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
