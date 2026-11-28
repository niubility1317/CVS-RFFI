"""Freeze a two-view, one-receiver source screen before independent scoring."""

import argparse
import importlib
import os
from pathlib import Path

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, tensor, write
from experiments.pamf_lora_main.dura_real_gain_fit import (
    DEVICE, LORA_ACTION, LORA_GAIN, SEED, WIDTH, activate_wisig_archive, as_complex, lora_models, wisig_models, wisig_real_prior,
)
from experiments.pamf_lora_main.dura_real_gain_fusion import PhysicsGain, RealSpectralGain
from experiments.pamf_restoration.models import complex_iq, pair


VIEWS = ('clean', 'practical_low_urban')
MODES = {'lora': ('direct', 'lora_native_r17', 'real_gain', 'fusion'),
         'wisig': ('direct', 'wisig_r17', 'real_gain', 'fusion')}
LORA_RX, WISIG_RX = 0, 1


def load_gain(folder, dataset, *, real=False):
    name = 'wisig_real_gain' if real else dataset
    checkpoint = torch.load(folder / name / 'gain.pt', map_location=DEVICE, weights_only=True)
    proof = read(folder / name / 'fit.json')
    if (checkpoint['proof'] != proof or proof['status'] != 'SOURCE_FROZEN' or proof['dataset'] != name
            or proof['seed'] != SEED or proof['updates'] != 2000
            or any(proof[k] != 0 for k in ('V_reads', 'target_reads', 'newly_read_TX_labels'))):
        raise ValueError('Expected the fixed source-only gain checkpoint')
    model = (RealSpectralGain(32) if real else PhysicsGain(WIDTH[dataset])).to(DEVICE).eval()
    model.load_state_dict(checkpoint['weights'], strict=True)
    return model.requires_grad_(False)


def packet_scale(y):
    return y.abs().square().mean(-1, keepdim=True).clamp_min(1e-12).sqrt()


@torch.no_grad()
def predict_lora(rt, root):
    project = Path(rt['project'])
    from experiments.cric_dura_comparison.bootstrap import activate
    activate(rt)
    parent_release = read(project / 'releases' / f'{LORA_GAIN}-release' / 'runtime.json')
    if parent_release['run_id'] != LORA_GAIN:
        raise ValueError('Expected registered LoRa phase-aware parent release')
    # Import the reader's whole dependency tree from its original W24 release.
    activate_wisig_archive(project / 'releases' / f'{LORA_GAIN}-release')
    activate_wisig_archive(Path(parent_release['w24_release_root']))
    _, _, base = lora_models(project)
    from experiments.pamf_lora_main import w24_joint_pamf as joint
    from experiments.pamf_lora_main import w24_native_training as native

    old = read(project / 'runs' / native.PHASE_RUN / 'frequency_inphase_native' / 'fit-inputs.json')
    source = Path(old['source_root'])
    contract, reader, _, _ = joint.boot(old, DEVICE)
    prior = native.task_model(None, DEVICE)
    prior.core.innovation = native.spectral_innovation('frequency_inphase_native', DEVICE)
    native.load_stage(project / 'runs' / native.PHASE_RUN, 'frequency_inphase_native', contract, prior.core, DEVICE)
    prior.eval().requires_grad_(False)
    dura = native.dura_model(project / 'runs' / LORA_ACTION, contract, DEVICE).eval().requires_grad_(False)
    gain = load_gain(Path(rt['lora_frozen_root']), 'lora')
    def classify(z):
        cpu = np.asarray(z.contiguous().view(torch.float32).tolist(), dtype=np.float32).view(np.complex64)
        return joint.classify_aperture(reader, joint.received_aperture(cpu, DEVICE))

    def decisions(raw_packets):
        y = as_complex(raw_packets)
        scale = packet_scale(y)
        p, action = base(y / scale)
        values = (y, dura(y), prior(y), gain(y / scale, p, action) * scale)
        return np.stack([classify(z) for z in values])

    # Real-checkpoint and reader smoke uses U_s only, before opening V.
    sample = joint.source_role(source, contract, 'U_s')[0][:2]
    if decisions(sample).shape != (len(MODES['lora']), 2):
        raise ValueError('Real LoRa checkpoint and reader smoke failed')
    indices, receivers, physical_ids = joint.panel(source, contract)
    chosen = np.flatnonzero(receivers == LORA_RX)
    if len(chosen) != 60:
        raise ValueError('Expected 60 fixed LoRa source-V packets on the first source RX')
    rows = indices[chosen]
    raw = joint.native(source, 'V', contract['counts']['V'])[rows]
    ids = np.asarray(physical_ids)[chosen]
    results = {}
    for view in VIEWS:
        signals = np.stack([joint.source_receive(view, (int(i), joint.SOURCE[LORA_RX], x))
                            for i, x in zip(rows, raw, strict=True)])
        results[view] = np.concatenate([decisions(signals[i:i+8]) for i in range(0, len(ids), 8)], axis=1)
        print('SCREEN_FROZEN_VIEW', 'lora', view, len(ids), flush=True)
    return ids, results


@torch.no_grad()
def predict_wisig(rt, root):
    project = Path(rt['project'])
    _, _, base = wisig_models(project, rt['frozen_sources'])
    original = importlib.import_module('experiments.pamf_unrolled_restore.wisig_anchored_window_source')
    config = read(project / 'releases' / '20261125-phase1-wisig-native-fir-source-m4-r01-release' / 'runtime.json')
    reader = original.tri.reader_factory(config).eval().requires_grad_(False)
    gain = load_gain(root, 'wisig')
    control = load_gain(root, 'wisig', real=True)
    packet = wisig_real_prior(project)

    def classify(z):
        return np.asarray(original.classify(reader, z).argmax(1).cpu().tolist(), dtype=np.int16)

    def decisions(batch):
        y = as_complex(batch)
        scale = packet_scale(y)
        p, action = base(y / scale)
        normal, estimate, group_scale = packet(y / scale)
        real = ((estimate + control(normal, estimate)) * group_scale).flatten(-2) * scale
        dura, _ = base(y)
        fused = gain(y / scale, p, action) * scale
        return np.stack([classify(z) for z in (y, dura, real, fused)])

    source, _, calibration = original.core.source_iq(Path(config['source_root']))
    if decisions(source[calibration[:2]]).shape != (len(MODES['wisig']), 2):
        raise ValueError('Real WiSig checkpoint and reader smoke failed')
    cache, _, _, _, ids, rx, _ = original.core.baseline.handles(config)
    selected = np.flatnonzero(rx == WISIG_RX)
    if len(selected) != 360:
        raise ValueError('Expected 360 fixed WiSig source-V packets on RX 1')
    results = {}
    for view in VIEWS:
        signals = np.load(cache / 'validation_full' / f'{view}.npy', mmap_mode='r', allow_pickle=False)
        results[view] = np.concatenate([decisions(signals[selected[i:i+48]])
                                        for i in range(0, len(selected), 48)], axis=1)
        print('SCREEN_FROZEN_VIEW', 'wisig', view, len(selected), flush=True)
    return ids[selected], results


def predict(rt, dataset):
    from experiments.cric_dura_comparison.bootstrap import numerical_policy
    numerical_policy()
    root = Path(rt['run_root'])
    ids, values = (predict_lora if dataset == 'lora' else predict_wisig)(rt, root)
    folder = root / dataset
    with (folder / 'screen.npz').open('xb') as stream:
        np.savez(stream, ids=ids, **values)
    write(folder / 'screen-frozen.json', dict(status='PREDICTIONS_FROZEN', views=list(VIEWS),
          modes=list(MODES[dataset]), count=len(ids), RX=LORA_RX if dataset == 'lora' else WISIG_RX,
          source_truth_reads=0, target_reads=0))


def paired(candidate, baseline, labels, permutation, minimum):
    a, b = candidate == labels, baseline == labels
    delta = a.astype(np.int16) - b.astype(np.int16)
    rng = np.random.default_rng(SEED + 8)
    draws = rng.integers(0, len(delta), size=(2000, len(delta)))
    half = np.array_split(permutation, 2)
    halves = [int(delta[index].sum()) for index in half]
    return dict(repaired=int((a & ~b).sum()), harmed=int((~a & b).sum()), net=int(delta.sum()),
                half_nets=halves, ci95=np.quantile(delta[draws].sum(1), (.025, .975)).tolist(),
                passes=bool(min(halves) >= minimum and delta.sum() >= 2 * minimum))


def score(rt, dataset):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Only the independent CPU scorer opens source truth')
    root = Path(rt['run_root'])
    folder = root / dataset
    modes = MODES[dataset]
    mark = read(folder / 'screen-frozen.json')
    expected = 60 if dataset == 'lora' else 360
    if (mark['status'], mark['views'], mark['modes'], mark['count'], mark['source_truth_reads']) != (
            'PREDICTIONS_FROZEN', list(VIEWS), list(modes), expected, 0):
        raise ValueError('All two-view same-packet predictions must be fixed before truth')
    with np.load(folder / 'screen.npz', allow_pickle=False) as archive:
        ids = archive['ids'].copy()
        predictions = {view: archive[view].copy() for view in VIEWS}
    if len(ids) != expected or any(value.shape != (len(modes), expected) for value in predictions.values()):
        raise ValueError('Incomplete paired screen')
    project = Path(rt['project'])
    if dataset == 'lora':
        from experiments.pamf_lora_main import w24_joint_pamf as joint
        parent = read(project / 'runs' / joint.raw_train.PHASE_RUN / 'frequency_inphase_native' / 'fit-inputs.json')
        with np.load(Path(parent['source_root']) / 'V_metadata.npz', allow_pickle=False) as metadata:
            lookup = dict(zip(metadata['ids'].tolist(), metadata['y'].tolist(), strict=True))
        labels = np.asarray([lookup[x] for x in ids], dtype=np.int64)
    else:
        from experiments.cric_dura_comparison.bootstrap import activate
        activate(rt)
        release = project / 'releases' / '20261125-phase1-wisig-native-fir-source-m4-r01-release'
        activate_wisig_archive(release)
        original = importlib.import_module('experiments.pamf_unrolled_restore.wisig_anchored_window_source')
        config = read(release / 'runtime.json')
        _, _, source, _, all_ids, _, positions = original.core.baseline.handles(config)
        y = np.load(source / 'V' / 'labels.npy', allow_pickle=False)[positions]
        lookup = dict(zip(all_ids.tolist(), y.tolist(), strict=True))
        labels = np.asarray([lookup[x] for x in ids], dtype=np.int64)
    permutation = np.random.default_rng(SEED + 7).permutation(expected)
    minimum, max_clean_loss = (1, 1) if dataset == 'lora' else (2, 2)
    hard, clean = predictions[VIEWS[-1]], predictions['clean']
    comparisons = {name: paired(hard[-1], hard[i], labels, permutation, minimum)
                   for i, name in enumerate(modes[:-1])}
    clean_deltas = {name: int(((clean[-1] == labels).astype(np.int16)
                              - (clean[i] == labels).astype(np.int16)).sum())
                    for i, name in enumerate(modes[:-1])}
    promote = all(x['passes'] for x in comparisons.values()) and min(clean_deltas.values()) >= -max_clean_loss
    write(folder / 'screen-score.json', dict(status='SOURCE_SCREEN_SCORED', truth_last=True,
          count=expected, modes=list(modes), correct={v: (x == labels).sum(1).tolist()
          for v, x in predictions.items()}, low_urban_pairs=comparisons, clean_net=clean_deltas,
          promote=promote, missing_controls=(['same-contract LoRa DURA v1'] if dataset == 'lora'
                                             else ['DURA v1 comparison deferred to formal evaluation']),
          formal_acceptance_complete=False,
          caveat='source V receiver and historical benchmark already exposed'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('predict', 'score'))
    parser.add_argument('runtime', type=Path)
    parser.add_argument('dataset', choices=('lora', 'wisig'))
    args = parser.parse_args()
    (predict if args.action == 'predict' else score)(read(args.runtime), args.dataset)
