"""Same-packet source screen on a fresh RX-specific physical subset."""

import argparse
import os
from pathlib import Path

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_lora_main.dura_real_gain_fit import DEVICE, as_complex
from experiments.pamf_lora_main.dura_real_gain_r17 import R17Gain
from experiments.pamf_lora_main.dura_real_gain_r17_fit import SEED, STEPS, boot
from experiments.pamf_lora_main.dura_real_gain_screen import packet_scale, paired


COUNT = 300
VIEWS = ('clean', 'practical_low_urban')
MODES = ('direct', 'lora_native_r17', 'real_gain', 'fusion')


def select_unexposed(rx, previous_rows):
    available = np.setdiff1d(np.flatnonzero(rx == 0), previous_rows, assume_unique=False)
    if len(available) < COUNT:
        raise ValueError('Insufficient fresh physical samples on the preselected receiver')
    return np.sort(np.random.default_rng(SEED + 2).choice(available, COUNT, replace=False))


@torch.no_grad()
def predict(rt):
    joint, _, old, contract, reader, _, prior, r17 = boot(rt)
    root, source = Path(rt['run_root']), Path(old['source_root'])
    checkpoint = torch.load(root / 'gain.pt', map_location=DEVICE, weights_only=True)
    proof = read(root / 'fit.json')
    if (checkpoint['proof'] != proof or (proof['status'], proof['seed'], proof['updates']) !=
            ('SOURCE_FROZEN', SEED, STEPS) or any(proof[k] for k in ('U_truth_reads', 'V_reads', 'target_reads'))):
        raise ValueError('Expected fixed lawful source-trained R17 fusion')
    gain = R17Gain(1024).to(DEVICE).eval().requires_grad_(False)
    gain.load_state_dict(checkpoint['weights'], strict=True)

    def classify(z):
        cpu = np.asarray(z.contiguous().view(torch.float32).tolist(), dtype=np.float32).view(np.complex64)
        return joint.classify_aperture(reader, joint.received_aperture(cpu, DEVICE))

    def decisions(raw):
        y = as_complex(raw)
        scale = packet_scale(y)
        normalized = y / scale
        p, action = prior(normalized)
        previous = prior(y)[0]
        return np.stack([classify(z) for z in (y, r17(y), previous,
                         gain(normalized, p, r17(normalized), action) * scale)])

    sample = joint.source_role(source, contract, 'U_s')[0][:2]
    if decisions(sample).shape != (len(MODES), 2):
        raise ValueError('Frozen source checkpoint/reader smoke failed')
    previous, _, _ = joint.panel(source, contract)
    with np.load(source / 'V_metadata.npz', allow_pickle=False) as metadata:
        ids = metadata['ids'].copy()
        rx = metadata['rx_i'].copy()
    if ids.tolist() != contract['role_ids']['V']:
        raise ValueError('Existing source V physical order differs')
    rows = select_unexposed(rx, previous)
    raw = joint.native(source, 'V', contract['counts']['V'])[rows]
    matrix = {}
    for view in VIEWS:
        signals = np.stack([joint.source_receive(view, (int(index), joint.SOURCE[0], value))
                            for index, value in zip(rows, raw, strict=True)])
        matrix[view] = np.concatenate([decisions(signals[i:i+8]) for i in range(0, COUNT, 8)], axis=1)
        print('LORA_R17_GAIN_SCREEN', view, COUNT, flush=True)
    with (root / 'screen.npz').open('xb') as stream:
        np.savez(stream, ids=ids[rows], **matrix)
    write(root / 'screen-frozen.json', dict(status='PREDICTIONS_FROZEN', modes=list(MODES),
          views=list(VIEWS), count=COUNT, source_RX='b200_1', first_screen_overlap=0,
          source_truth_reads=0, target_reads=0))


def score(rt):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Independent CPU scorer required')
    root = Path(rt['run_root'])
    frozen = read(root / 'screen-frozen.json')
    if (frozen['status'], frozen['modes'], frozen['views'], frozen['count'], frozen['source_truth_reads']) != (
            'PREDICTIONS_FROZEN', list(MODES), list(VIEWS), COUNT, 0):
        raise ValueError('All registered source predictions must freeze before truth')
    with np.load(root / 'screen.npz', allow_pickle=False) as archive:
        ids = archive['ids'].copy()
        outputs = {view: archive[view].copy() for view in VIEWS}
    if len(ids) != COUNT or any(x.shape != (len(MODES), COUNT) for x in outputs.values()):
        raise ValueError('Incomplete prediction matrix')
    project = Path(rt['project'])
    old = read(project / 'runs' / '20261128-phase1-lora-raw8192-frequency-quadrature-s2026113037-r01'
               / 'frequency_inphase_native' / 'fit-inputs.json')
    with np.load(Path(old['source_root']) / 'V_metadata.npz', allow_pickle=False) as metadata:
        truth = dict(zip(metadata['ids'].tolist(), metadata['y'].tolist(), strict=True))
    labels = np.asarray([truth[x] for x in ids], dtype=np.int64)
    permutation = np.random.default_rng(SEED + 3).permutation(COUNT)
    hard, clean = outputs[VIEWS[-1]], outputs['clean']
    comparisons = {name: paired(hard[-1], hard[index], labels, permutation, 2)
                   for index, name in enumerate(MODES[:-1])}
    clean_deltas = {name: int(((clean[-1] == labels).astype(np.int16)
                              - (clean[index] == labels).astype(np.int16)).sum())
                    for index, name in enumerate(MODES[:-1])}
    promote = all(value['passes'] for value in comparisons.values()) and min(clean_deltas.values()) >= -2
    write(root / 'screen-score.json', dict(status='SOURCE_SCREEN_SCORED', truth_last=True,
          count=COUNT, modes=list(MODES), correct={v: (x == labels).sum(1).tolist()
          for v, x in outputs.items()}, low_urban_pairs=comparisons, clean_net=clean_deltas,
          promote=promote, missing_controls=['same-contract LoRa DURA v1'], formal_acceptance_complete=False,
          exposure='Different physical V IDs from first screen; W24 model selection saw source V historically.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('predict', 'score'))
    parser.add_argument('runtime', type=Path)
    args = parser.parse_args()
    (predict if args.action == 'predict' else score)(read(args.runtime))
