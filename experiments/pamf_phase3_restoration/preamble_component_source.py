"""Attribute frozen source errors to repeat, contrast, and outside-window IQ."""

import argparse
import os
from pathlib import Path

import numpy as np
import torch

from .cache import check_F0_state, load_F0, tensor
from .cost_probe import write
from .main_split_eval import valid_identity
from .phase_mixture_source import REFERENCE_RUN
from .source_action_factor_probe import references
from .source_inverse import R07, SCENES
from .source_pair import read, validate_source


RUN = '20261107-diagnostic-Restore-preamble-component-sourceV-s2026092701-r01'
EDGE_RUN = '20261107-diagnostic-Restore-preamble-edges-sourceV-s2026092701-r01'
COMPONENT_COMMIT = '7bb818da2e7a6084368e18693bfc4b102f58a780'
WINDOW = slice(80, 160)
COMPONENTS = ('repeat', 'contrast', 'outside')
EDGE_COMPONENTS = ('repeat', 'contrast', 'front', 'tail')
EDGE_HYBRIDS = tuple(range(4, 12))
EDGE_REUSED = (0, 1, 2, 3, 12, 13, 14, 15)


def components(base, clean):
    """Partition the oracle correction using phase from base alone."""
    estimate = base.double()
    error = clean.double() - estimate
    cycles = torch.complex(estimate[:, 0, WINDOW], estimate[:, 1, WINDOW]).reshape(-1, 4, 20)
    cross = (cycles[:, :-1].conj() * cycles[:, 1:]).sum((1, 2))
    shift = torch.exp(1j * torch.angle(cross)[:, None, None] *
                      torch.arange(4, device=base.device, dtype=torch.float64)[None, :, None])
    window = torch.complex(error[:, 0, WINDOW], error[:, 1, WINDOW]).reshape_as(cycles)
    coherent = ((window * shift.conj()).mean(1, keepdim=True) * shift).flatten(1)
    repeat = torch.zeros_like(error)
    repeat[:, 0, WINDOW] = coherent.real
    repeat[:, 1, WINDOW] = coherent.imag
    outside = error.clone()
    outside[:, :, WINDOW] = 0
    return repeat, error - outside - repeat, outside


def counterfactuals(base, clean, *, edge_split=False):
    pieces = components(base, clean)
    masks = range(1, 7)
    if edge_split:
        repeat, contrast, outside = pieces
        front = outside.clone()
        front[:, :, 80:] = 0
        pieces = (repeat, contrast, front, outside - front)
        masks = EDGE_HYBRIDS
    error = clean.double() - base.double()
    energy = torch.stack([p.square().sum((1, 2)) for p in pieces], 1)
    torch.testing.assert_close(energy.sum(1), error.square().sum((1, 2)), rtol=1e-12, atol=1e-12)
    iq = {}
    for mask in masks:
        selected = [p for bit, p in enumerate(pieces) if mask & (1 << bit)]
        iq[mask] = (base.double() + sum(selected)).to(base.dtype)
    return iq, energy


def logits(reader, iq):
    values = reader.classify_features(reader.features(iq))
    if values.shape != (len(iq), 6) or not torch.isfinite(values).all():
        raise ValueError('Invalid frozen F0 prediction')
    return values


def previous_predictions(folder, identity, ref, take):
    previous = read(folder/'results.json')
    if (read(folder/'complete.json')['status'] != 'ARTIFACTS_COMPLETE'
            or previous['status'] != 'SOURCE_PREAMBLE_COMPONENT_PREDICTIONS_COMPLETE'
            or previous['code_commit'] != COMPONENT_COMMIT or previous['F0_identity'] != identity
            or previous['source_V_records'] != len(take) or previous['views'] != list(SCENES)
            or previous['TX_labels_read'] != 0 or previous['target_inputs'] != 0):
        raise ValueError('Original frozen component predictions required')
    with np.load(folder/'reference.npz', allow_pickle=False) as file:
        if (not np.array_equal(file['source_index'], take)
                or not all(np.array_equal(file[key], ref[key]) for key in ('ids', 'rx', 'day', 'logits'))):
            raise ValueError('Original component panel differs')


def worker(args, out):
    inputs = validate_source(read(args.inputs))
    reader, identity = load_F0(inputs, 'cpu')
    valid_identity(identity)
    snapshot = {k: v.detach().clone() for k, v in reader.state_dict().items()}
    cache = args.root / 'runs' / R07 / 'cache'
    source = np.flatnonzero(np.load(cache/'train/roles.npy', allow_pickle=False) == 'L_s')[:2]
    if len(source) != 2:
        raise ValueError('Expected two real L_s records')
    x = np.load(cache/'train/clean.npy', mmap_mode='r', allow_pickle=False)
    y = np.load(cache/'epochs/01/received.npy', mmap_mode='r', allow_pickle=False)
    with torch.inference_mode():
        hybrids, _ = counterfactuals(tensor(y[source], 'cpu'), tensor(x[source], 'cpu'), edge_split=args.edge_split)
        logits(reader, torch.cat(list(hybrids.values())))
    masks = tuple(hybrids)
    write(out/'smoke.json', dict(status='PASS', real_L_s=2, F0_rows=2*len(masks), target_inputs=0))
    print('SOURCE_PREAMBLE_COMPONENT_REAL_SMOKE_PASS', flush=True)
    ref, take = references(args.root, cache, identity)
    clean = tensor(ref['clean'], 'cpu')
    with (out/'reference.npz').open('xb') as file:
        np.savez(file, **{key: ref[key] for key in ('ids', 'rx', 'day', 'logits')}, source_index=take)
    folder = args.root/'runs'/REFERENCE_RUN
    previous = args.root/'runs'/RUN
    if args.edge_split:
        previous_predictions(previous, identity, ref, take)
    with torch.inference_mode():
        for scene in SCENES:
            with np.load(folder/(scene+'.npz'), allow_pickle=False) as file:
                if not np.array_equal(file['ids'][take], ref['ids']):
                    raise ValueError('Source baseline physical IDs differ')
                base = tensor(file['physical__iq'][take].astype(np.float32), 'cpu')
                baseline = file['physical__prediction'][take]
                baseline_logits = file['physical__logits'][take]
            if not np.array_equal(baseline, baseline_logits.argmax(1)):
                raise ValueError('Source baseline predictions differ')
            first = logits(reader, base[:2])
            torch.testing.assert_close(first, tensor(baseline_logits[:2].astype(np.float32), 'cpu'),
                                       rtol=2e-5, atol=2e-5)
            hybrids, energy = counterfactuals(base, clean, edge_split=args.edge_split)
            predictions = {0: baseline, 7: ref['logits'].argmax(1)}
            if args.edge_split:
                with np.load(previous/(scene+'.npz'), allow_pickle=False) as file:
                    old = file['prediction']
                    if (not np.array_equal(file['ids'], ref['ids']) or old.shape != (8, len(take))
                            or not np.array_equal(old[0], baseline)
                            or not np.array_equal(old[7], predictions[7])):
                        raise ValueError('Original component controls differ')
                    split = energy.tolist()
                    before = file['component_energy']
                    np.testing.assert_allclose(np.asarray(split)[:, :2], before[:, :2], rtol=1e-12, atol=1e-12)
                    np.testing.assert_allclose(np.asarray(split)[:, 2:].sum(1), before[:, 2], rtol=1e-12, atol=1e-12)
                    predictions = dict(zip(EDGE_REUSED, old))
            for mask in masks:
                pred = [logits(reader, batch).argmax(1).tolist() for batch in hybrids[mask].split(32)]
                predictions[mask] = np.concatenate(pred)
            with (out/(scene+'.npz')).open('xb') as file:
                np.savez(file, ids=ref['ids'], prediction=np.stack([predictions[key] for key in sorted(predictions)]),
                         component_energy=np.asarray(energy.tolist()))
            print('SOURCE_PREAMBLE_COMPONENT_VIEW_COMPLETE', scene, len(take), flush=True)
    check_F0_state(reader, snapshot)
    write(out/'results.json', dict(status='SOURCE_PREAMBLE_COMPONENT_PREDICTIONS_COMPLETE',
        code_commit=args.commit, F0_identity=identity, views=list(SCENES), source_V_records=len(take),
        components=list(EDGE_COMPONENTS if args.edge_split else COMPONENTS), window=[80, 160],
        period_samples=20, hybrids=list(masks),
        F0_hybrid_rows=len(take)*len(SCENES)*len(masks), F0_smoke_rows=2*len(masks),
        F0_baseline_check_rows=14, TX_labels_read=0,
        target_inputs=0, OE_reads=0, optimizer_updates=0, new_deployment_candidate=False))


def main(*, edge_split=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    parser.set_defaults(edge_split=edge_split)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or (Path.cwd()/'release_commit.txt').read_text().strip() != args.commit:
        raise ValueError('Expected CPU-only release and registered code commit')
    os.nice(10)
    torch.set_num_threads(2)
    out = args.root/'runs'/(EDGE_RUN if args.edge_split else RUN)
    out.mkdir()
    write(out/'launch.json', dict(pid=os.getpid(), code_commit=args.commit, CPU_threads=2, CUDA_empty=True))
    worker(args, out)
    write(out/'complete.json', dict(status='ARTIFACTS_COMPLETE', TX_labels_read=0, target_inputs=0))


if __name__ == '__main__':
    main()
