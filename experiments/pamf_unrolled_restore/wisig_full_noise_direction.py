"""Compare known-noise direction and complete removal on frozen WiSig source IQ."""

import argparse
from pathlib import Path, PurePosixPath

import numpy as np

if __package__ is None:
    from experiments import pamf_unrolled_restore

    pamf_unrolled_restore.__path__.append(str(Path(__file__).resolve().parent))

from experiments.cric_dura_comparison.io import read, tensor, write
from experiments.pamf_unrolled_restore.wisig_native_fir_anchor_probe import PARENT, dispatch, reference
from experiments.pamf_unrolled_restore.wisig_waveform_audit import PANEL, PANEL_CHECK, RUN as WAVE_RUN, panel


RUN = '20261127-diagnostic-wisig-full-noise-direction-s2026112501-r01'
MODES = ('direct', 'r69', 'equal_both', 'exact_both')
METRIC_NAMES = ('center_r69_remaining', 'center_equal_remaining', 'center_edit_to_noise_rms',
                'edge_r69_remaining', 'edge_equal_remaining', 'edge_edit_to_noise_rms')
BOUNDARY = 24
CENTER = slice(BOUNDARY, 256-BOUNDARY)
EDGES = np.r_[0:BOUNDARY, 256-BOUNDARY:256]


def power(x):
    return np.square(np.abs(x)).sum(axis=-1, keepdims=True)


def interventions(received, restored, signal):
    if received.shape != restored.shape or received.shape != signal.shape or received.shape[-1] != 256:
        raise ValueError('Expected matched native 256-point WiSig observations')
    matched = restored.copy()
    for take in (CENTER, EDGES):
        y, g, s = (a[..., take] for a in (received, restored, signal))
        need = s - y
        p = power(need)
        factor = np.sqrt(np.divide(power(g-y), p, out=np.zeros_like(p), where=p > 1e-15))
        matched[..., take] = np.where(p > 1e-15, y + factor * need, g)
    exact = np.where(power(signal-received) > 1e-15, signal, restored)
    return matched, exact


def receipt(views, count):
    return dict(status='PREDICTIONS_FROZEN', views=list(views), modes=list(MODES), count=count,
                target_reads=0, source_truth_reads=0, route='full', edge_points_each=BOUNDARY,
                source_oracle_only=True, waveform_parent=WAVE_RUN)


def predict(rt, parent):
    import torch

    from experiments.pamf_unrolled_restore.wisig_anchored_window_source import classify, core, tri
    from experiments.pamf_unrolled_restore.wisig_uncertain_prior import as_complex

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root, wave = (Path(rt[k]) for k in ('run_root', 'wave_root'))
    cache, ids, chosen, _ = panel(rt, parent)
    views = core.baseline.VIEWS
    expected = read(wave/'waveforms-frozen.json')
    if (read(wave/'state.json')['status'] != 'SOURCE_SCORED' or expected['status'] != 'FROZEN'
            or expected['source_truth_reads'] or expected['target_reads'] or expected['simulator_reference_reads']
            or expected['modes'][:2] != list(MODES[:2]) or expected['views'] != list(views)
            or expected['packets'] != len(chosen)):
        raise ValueError('Expected the frozen receive-only WiSig waveform controls')
    device = 'cuda:0'
    reader = tri.reader_factory(parent).eval().requires_grad_(False)
    engine = core.baseline.original_r69(Path(parent['parent_root']))

    def decisions(iq):
        z = tensor(np.ascontiguousarray(iq, dtype=np.complex64).view(np.float32), device).view(torch.complex64)
        return np.asarray(classify(reader, z).argmax(-1).tolist(), dtype=np.uint8)

    with torch.no_grad():
        source, _, calibration = core.source_iq(cache)
        samples = tensor(source[calibration[:4]], device)
        (restored,) = engine(samples)
        pair = [np.from_dlpack(as_complex(z).cpu().contiguous()) for z in (samples, restored)]
        for z in (*pair, *interventions(*pair, pair[0])):
            decisions(z)
        write(root/'smoke.json', dict(status='PASS', source_u_count=4, target_reads=0, source_truth_reads=0,
                                      oracle_reads=0, input_points=256))
        oracle = Path(rt['panel_root'])
        observed = np.load(oracle/'received.npy', mmap_mode='r', allow_pickle=False)
        with np.load(oracle/'scorer_reference.npz', allow_pickle=False) as components:
            signal, noise = components['signal'], components['noise']
            if observed.shape != (24, len(chosen), 2, 256) or signal.shape != noise.shape or signal.shape != (
                    24, len(chosen), 256):
                raise ValueError('Unexpected registered Practical reference layout')
            all_predictions, all_metrics = [], []
            for index, scene in enumerate(views):
                with np.load(wave/f'full_{scene}.npz', allow_pickle=False) as old:
                    if not np.array_equal(old['ids'], ids[chosen]):
                        raise ValueError(f'Frozen physical ID order differs for {scene}')
                    raw, fixed = old['iq'][:2]
                    original = old['decisions'][:2]
                if index:
                    y = observed[index-1]
                    np.testing.assert_allclose(raw, y[:, 0] + 1j*y[:, 1], rtol=2e-5, atol=3e-6)
                    np.testing.assert_allclose(raw, signal[index-1] + noise[index-1], rtol=2e-5, atol=3e-6)
                    noiseless = signal[index-1].astype(np.complex64)
                else:
                    noiseless = raw
                matched, exact = interventions(raw, fixed, noiseless)
                pred = [np.concatenate([decisions(z[i:i+64]) for i in range(0, len(z), 64)])
                        for z in (raw, fixed, matched, exact)]
                if not np.array_equal(pred[:2], original):
                    raise ValueError(f'Direct and R69 must reproduce the frozen {scene} decisions')
                if index == 0 and not (np.array_equal(pred[2], original[1]) and np.array_equal(pred[3], original[1])):
                    raise ValueError('Clean must preserve the reference R69 waveform')
                metrics = []
                for take in (CENTER, EDGES):
                    a, b, x = (z[..., take] for z in (raw, fixed, noiseless))
                    valid = power(x-a) > 1e-15
                    edit, replaced = power(b-a), power(matched[..., take]-a)
                    np.testing.assert_allclose(replaced, edit, rtol=1e-4, atol=2e-8)
                    base = np.divide(power(b-x), power(x-a), out=np.zeros_like(edit), where=valid)
                    residual = np.divide(power(matched[..., take]-x), power(x-a), out=np.zeros_like(edit), where=valid)
                    ratio = np.sqrt(np.divide(edit, power(x-a), out=np.zeros_like(edit), where=valid))
                    metrics.extend((base, residual, ratio))
                all_predictions.append(pred)
                all_metrics.append(np.concatenate(metrics, axis=-1))
                print('WISIG_FULL_NOISE_PREDICTED', scene, len(chosen), flush=True)
    np.savez(root/'predictions.npz', decisions=np.asarray(all_predictions, dtype=np.uint8), ids=ids[chosen],
             metrics=np.stack(all_metrics))
    write(root/'predictions-frozen.json', receipt(views, len(chosen)))


def score(rt, parent):
    from experiments.pamf_unrolled_restore.wisig_anchored_window_source import core

    root = Path(rt['run_root'])
    cache, ids, chosen, rx = panel(rt, parent)
    views = core.baseline.VIEWS
    if read(root/'predictions-frozen.json') != receipt(views, len(chosen)):
        raise ValueError('Expected the complete frozen source-only readout')
    with np.load(root/'predictions.npz', allow_pickle=False) as bundle:
        pred, metrics = (bundle[name] for name in ('decisions', 'metrics'))
        if (not np.array_equal(bundle['ids'], ids[chosen]) or pred.shape != (len(views), len(MODES), len(chosen))
                or pred.dtype != np.uint8 or pred.max() >= 6
                or metrics.shape != (len(views), len(chosen), len(METRIC_NAMES))
                or not np.isfinite(metrics).all()):
            raise ValueError('Invalid source prediction matrix')
    _, _, label_root, _, bound, _, positions = core.baseline.handles(parent)
    if not np.array_equal(bound, ids):
        raise ValueError('Source scoring physical IDs differ')
    truth = np.load(label_root/'V'/'labels.npy', allow_pickle=False)[positions[chosen]]
    day = np.load(cache/'validation/day.npy', allow_pickle=False)[chosen]
    hits = pred == truth
    rng = np.random.default_rng(2026112502)
    groups = [np.flatnonzero((rx == r) & (day == d)) for r in np.unique(rx) for d in np.unique(day)]
    draws = np.concatenate([rng.choice(g, (2000, len(g))) for g in groups], axis=1)
    pairs = {}
    for a, b in ((1, 0), (2, 1), (3, 1), (3, 2)):
        delta = hits[:, a].astype(np.int16) - hits[:, b]
        blocks = dict(zip(views, delta, strict=True))
        blocks['six'] = delta[1:]
        result = {}
        for k, v in blocks.items():
            per_packet = v.reshape(-1, len(chosen)).sum(axis=0)
            result[k] = dict(repaired=int((v>0).sum()), harmed=int((v<0).sum()), net=int(v.sum()),
                             ci95=np.quantile(per_packet[draws].sum(axis=1), [.025, .975]).tolist())
        for name, field in (('receiver', rx), ('day', day), ('tx', truth)):
            result[name] = {str(g): dict(six=int(delta[1:, field==g].sum()), low=int(delta[-1, field==g].sum()))
                            for g in np.unique(field)}
        pairs[f'{MODES[a]}_vs_{MODES[b]}'] = result
    write(root/'scores.json', dict(status='ANALYZED', count=len(chosen), modes=list(MODES), views=list(views),
          target_reads=0, source_truth_last=True, correct=hits.sum(axis=-1).tolist(), pairs=pairs,
          metric_order=list(METRIC_NAMES), median_metrics=np.median(metrics, axis=1).tolist()))


def main():
    from experiments.pamf_lora_main.publish import release, release_commit

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    for action in ('dispatch','predict','score'):
        sub.add_parser(action).add_argument('runtime', type=Path)
    pub = sub.add_parser('publish')
    for key in ('repo','output','context','ssh-config'):
        pub.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'publish':
        ctx = read(args.context)
        rt = {k:ctx[k] for k in ('project','python','frozen_sources')}
        project = PurePosixPath(rt['project'])
        releases, runs = project/'releases', project/'runs'
        parent = releases/f'{PARENT}-release'
        rt.update(run_id=RUN, runner=__spec__.name, code_commit=release_commit(args.repo), reference=str(parent),
                  release_root=str(releases/f'{RUN}-release'), run_root=str(runs/RUN), panel_root=str(runs/PANEL),
                  wave_root=str(runs/WAVE_RUN), max_compute_processes=8,
                  external_releases=list(map(str, (parent, runs/PARENT, runs/PANEL, runs/PANEL_CHECK, runs/WAVE_RUN))))
        release(args.repo, args.output, rt, args.ssh_config)
    elif args.action == 'dispatch':
        dispatch(args.runtime, entry=__file__)
    else:
        from experiments.pamf_lora_main.runner import activate

        rt = read(args.runtime)
        original = reference(rt)
        activate(original)
        (predict if args.action == 'predict' else score)(rt, original)


if __name__ == '__main__':
    main()
