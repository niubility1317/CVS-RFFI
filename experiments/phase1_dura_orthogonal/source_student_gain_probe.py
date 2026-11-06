"""Source-only scalar-gain oracle diagnostics for fixed direct students."""
import argparse
import time
from pathlib import Path

import numpy as np
import torch

from experiments.pamf_phase1_flow import experiment as core
from experiments.pamf_phase1_flow.anchor_source_smoke import frozen_dura
from experiments.phase1_dura_innovation_probe.source_probe import source_inputs
from experiments.phase1_dura_orthogonal.complex_gain_source_probe import gain_metrics
from experiments.phase1_dura_orthogonal.direct_spectral_student import DirectSpectralStudent
from experiments.phase1_dura_orthogonal.train_source import SCHEMA, DUAL_DOMAIN_EPOCHS

MODELS = (
    ('student_8192', '20261106-phase1-direct-spectral-distill-source-s2026092701-r01',
     'd98f85fe9c01bebe6091537311b7e6d110e08fdd', 'distill_iq', 8192),
    ('student_32768', '20261106-phase1-direct-spectral-wide-distill-source-s2026092701-r01',
     '18f348500c4db11a7538cb61f6971cf54a41379e', 'wide_source_distill', 32768),
)


def load_student(root, run, commit, objective, fit_count, pilot_ids):
    folder = root/'runs'/run
    summary = core.read(folder/'source_summary.json')
    state = torch.load(folder/'final.pt', map_location='cpu', weights_only=False)
    expected = dict(commit=commit, architecture='direct_spectral_student', objective=objective,
        epochs=DUAL_DOMAIN_EPOCHS, DURA_training_reads=1, source_iq_teacher='dura_v1_R65_E20_4940',
        source_iq_teacher_weight=.5, inference_received_iq_only=True, TX_labels_read=0,
        F_weights_read=0, F_outputs_read=0, V_reads=0, target_reads=0, DURA_frozen=True)
    if (summary.get('status') != 'SOURCE_TRAINED' or summary.get('fit_ids') != fit_count or
            state.get('schema') != SCHEMA or len(state.get('fit_ids', [])) != fit_count or
            any(summary.get(k) != v or state.get(k) != v for k, v in expected.items()) or
            np.intersect1d(pilot_ids, state['fit_ids']).size):
        raise ValueError('Fixed source-only student and disjoint pilot required')
    model = DirectSpectralStudent(width=state['width'], taps=9)
    model.load_state_dict(state['model'], strict=True)
    return model.eval().requires_grad_(False)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--commit', required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError('Unique source diagnostic output required')
    torch.set_num_threads(2)
    ids, rx, scene, _, received, clean = source_inputs(args.root)
    models = [load_student(args.root, run, commit, objective, count, ids)
              for _, run, commit, objective, count in MODELS]
    dura = frozen_dura(args.root, 'cpu')
    chunks = []
    started = time.monotonic()
    for offset in range(0, len(ids), 16):
        selected = ids[offset:offset+16]
        x, y = core.tensor(received[selected], 'cpu'), core.tensor(clean[selected], 'cpu')
        metrics = torch.stack([gain_metrics(v, y) for v in (x, models[0](x), models[1](x), dura(x))], 1)
        if not torch.isfinite(metrics).all():
            raise FloatingPointError('Finite source oracle metrics required')
        if (metrics[:, :, 2] > metrics[:, :, 1]+1e-5).any() or (metrics[:, :, 1] > metrics[:, :, 0]+1e-5).any():
            raise ValueError('Nested source oracle minima must hold')
        if offset == 0:
            print('STUDENT_GAIN_SOURCE_SMOKE_PASS', len(selected), flush=True)
        chunks.append(metrics.cpu())
    values = torch.cat(chunks)
    groups = {}
    for name, receivers in (('RX4_6', (4, 6)), ('RX8', (8,))):
        for view in core.VIEWS:
            mask = np.isin(rx, receivers) & (scene == view)
            if mask.sum() != 48:
                raise ValueError('48 records per source receiver/view required')
            cell = values[torch.tensor(mask.tolist(), dtype=torch.bool)]
            groups[name+'/'+view] = dict(count=len(cell), mean_nmse=cell.mean(0).tolist())
    args.out.mkdir(parents=True)
    core.write(args.out/'result.json', dict(status='STUDENT_GAIN_SOURCE_COMPLETE', commit=args.commit,
        model_order=['received', 'student_8192', 'student_32768', 'dura'],
        metric_order=['unaligned', 'oracle_phase_only', 'oracle_complex_gain'],
        records=len(ids), groups=groups, elapsed_seconds=time.monotonic()-started,
        TX_labels_read=0, F_weights_read=0, F_outputs_read=0, V_reads=0, target_reads=0, model_updates=0,
        caveat='Per-record phase/gain uses source clean IQ and cannot be deployed; IQ error is not identity accuracy.'))
    print('STUDENT_GAIN_SOURCE_COMPLETE', len(ids), flush=True)


if __name__ == '__main__':
    main()
