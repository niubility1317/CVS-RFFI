"""Cross the magnitude and phase of frozen WiSig restoration readouts."""

import argparse
from itertools import product
from pathlib import Path, PurePosixPath

import numpy as np

from experiments.cric_dura_comparison.io import read, write


RUN = '20261125-diagnostic-wisig-stage-polar-source-s2026112501-r01'
BASE = ('direct', 'r69', 'centered_map_1', 'centered_final')
MODES = (*BASE, 'fir_amp_r69_phase', 'r69_amp_fir_phase', 'final_amp_first_phase', 'first_amp_final_phase')
COMPARISONS = (('fir_amp_r69_phase', 'centered_final'), ('r69_amp_fir_phase', 'centered_final'),
               ('fir_amp_r69_phase', 'r69_amp_fir_phase'), ('r69', 'fir_amp_r69_phase'),
               ('r69', 'r69_amp_fir_phase'), ('final_amp_first_phase', 'centered_map_1'),
               ('first_amp_final_phase', 'centered_map_1'), ('centered_final', 'final_amp_first_phase'),
               ('centered_final', 'first_amp_final_phase'))


def polar_pair(first, second):
    """Keep each signal's amplitude and take phase from the other signal."""
    a, b = np.abs(first), np.abs(second)
    # Zero-valued donors have no phase, so keep the amplitude donor's sample there.
    phase_a = np.divide(first, a, out=np.zeros_like(first), where=a>0)
    phase_b = np.divide(second, b, out=np.zeros_like(second), where=b>0)
    return np.where(b>0, a*phase_b, first), np.where(a>0, b*phase_a, second)


def variants(wave):
    direct, r69, first, final = wave
    return direct, r69, first, final, *polar_pair(final, r69), *polar_pair(final, first)


def load_case(path, ids, indices):
    with np.load(path, allow_pickle=False) as file:
        wave, decisions = file['iq'][indices], file['decisions'][indices]
        if not np.array_equal(file['ids'], ids) or wave.shape != (len(BASE), len(ids), 256):
            raise ValueError('Expected the fixed WiSig source-only waveform packet order')
    return wave, decisions


def predict(rt, parent):
    import torch
    from experiments.cric_dura_comparison.io import tensor
    from experiments.pamf_unrolled_restore.wisig_anchored_window_source import classify, core, tri
    from experiments.pamf_unrolled_restore.wisig_native_fir_anchor_probe import models_from
    from experiments.pamf_unrolled_restore.wisig_uncertain_prior import as_complex
    from experiments.pamf_unrolled_restore.wisig_waveform_audit import MODES as ARCHIVE_MODES, ROUTES, panel

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root, source = Path(rt['run_root']), Path(rt['waveform_root'])
    cache, ids, chosen, _ = panel(rt, parent)
    archived = read(source/'waveforms-frozen.json')
    fixed = dict(status='FROZEN', modes=list(ARCHIVE_MODES), routes=list(ROUTES), views=list(core.baseline.VIEWS),
                 packets=len(chosen), source_truth_reads=0, target_reads=0, simulator_reference_reads=0,
                 all_archive_decisions_match=True)
    if archived != fixed or read(source/'state.json')['status'] != 'SOURCE_SCORED':
        raise ValueError('Complete earlier clean/practical waveform comparisons required')
    reader = tri.reader_factory(parent).eval().requires_grad_(False)
    engine = core.baseline.original_r69(Path(parent['parent_root']))
    (model, _), _ = models_from(rt)

    def decisions(waves):
        result = []
        for z in variants(waves):
            value = tensor(np.stack((z.real, z.imag), axis=1), 'cuda:0')
            result.append(classify(reader, as_complex(value)).argmax(-1).tolist())
        return np.asarray(result, dtype=np.uint8)

    with torch.no_grad():
        train, _, held = core.source_iq(cache)
        sample = tensor(train[held[:4]], 'cuda:0')
        observed = as_complex(sample)
        original, = engine(sample)
        stages, _ = model.stages(model.analyze(observed))
        smoke = [np.from_dlpack(z.cpu().contiguous()) for z in (observed, as_complex(original), stages[2], stages[4])]
        if decisions(smoke).shape != (len(MODES), len(sample)):
            raise ValueError('Complete real source U checkpoint smoke required')
        write(root/'smoke.json', dict(status='PASS', packets=len(sample), target_reads=0, source_V_reads=0,
                                    input_gate=False, original_model_calls_per_packet=2))
        indices = [ARCHIVE_MODES.index(name) for name in BASE]
        for route, view in product(ROUTES, core.baseline.VIEWS):
            key = f'{route}_{view}.npz'
            wave, original = load_case(source/key, ids[chosen], indices)
            pred = np.concatenate([decisions(wave[:, i:i+64]) for i in range(0,len(chosen),64)], axis=1)
            if not np.array_equal(pred[:len(BASE)], original) or pred.shape != (len(MODES), len(chosen)):
                raise ValueError(f'Frozen WiSig source decisions differ on {route}/{view}')
            with (root/key).open('xb') as file:
                np.savez(file, ids=ids[chosen], decision=pred)
            print('WISIG_POLAR_PREDICTED', route, view, len(chosen), flush=True)
    write(root/'predictions-frozen.json', dict(status='FROZEN', modes=list(MODES), routes=list(ROUTES),
          views=list(core.baseline.VIEWS), packets=len(chosen), source_truth_reads=0, target_reads=0,
          original_controls_exact=True, input_gate=False))


def score(rt, parent):
    from experiments.pamf_unrolled_restore.wisig_anchored_window_source import core
    from experiments.pamf_unrolled_restore.wisig_waveform_audit import ROUTES, panel

    _, ids, chosen, rx = panel(rt,parent)
    root = Path(rt['run_root'])
    views = core.baseline.VIEWS
    proof = read(root/'predictions-frozen.json')
    if proof != dict(status='FROZEN', modes=list(MODES), routes=list(ROUTES), views=list(views), packets=len(chosen),
                     source_truth_reads=0, target_reads=0, original_controls_exact=True, input_gate=False):
        raise ValueError('Both complete blind source prediction routes are required')
    decisions = {}
    for route in ROUTES:
        arrays = []
        for view in views:
            with np.load(root/f'{route}_{view}.npz', allow_pickle=False) as file:
                if not np.array_equal(file['ids'], ids[chosen]):
                    raise ValueError(f'Physical IDs differ on {route}/{view}')
                arrays.append(file['decision'])
        decisions[route] = np.stack(arrays)
        if decisions[route].shape != (len(views), len(MODES), len(chosen)) or (decisions[route]>5).any():
            raise ValueError('Unexpected WiSig source prediction shape or decision range')
    _, _, label_root, _, bound_ids, _, positions = core.baseline.handles(parent)
    if not np.array_equal(bound_ids, ids):
        raise ValueError('WiSig scoring contract has different source physical IDs')
    truth = np.load(label_root/'V'/'labels.npy', allow_pickle=False)[positions[chosen]]
    results = {}
    for route, pred in decisions.items():
        hit = pred==truth
        by_view = dict(zip(views, hit.sum(axis=-1).tolist(), strict=True))
        by_rx = {str(r): dict(zip(views, hit[:, :, rx==r].sum(axis=-1).tolist(), strict=True)) for r in np.unique(rx)}
        results[route] = dict(correct=by_view, correct_by_rx=by_rx,
                              paired=core.baseline.paired_intervals(pred, truth, rx, MODES, COMPARISONS))
    write(root/'source-scores.json', dict(status='SOURCE_SCORED', modes=list(MODES), packets=len(chosen),
          target_reads=0, source_truth_last=True, results=results))


def main():
    if __package__ is None:
        from experiments import pamf_unrolled_restore

        pamf_unrolled_restore.__path__.append(str(Path(__file__).resolve().parent))

    from experiments.pamf_lora_main.publish import release, release_commit
    from experiments.pamf_unrolled_restore.wisig_native_fir_anchor_probe import PARENT, dispatch, reference
    from experiments.pamf_unrolled_restore.wisig_waveform_audit import PANEL, PANEL_CHECK, RUN as WAVEFORM_RUN

    parser = argparse.ArgumentParser()
    actions = parser.add_subparsers(dest='action', required=True)
    for action in ('dispatch','predict','score'):
        actions.add_parser(action).add_argument('runtime', type=Path)
    pub = actions.add_parser('publish')
    for field in ('repo','output','context','ssh-config'):
        pub.add_argument('--'+field, type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'dispatch':
        dispatch(args.runtime, entry=__file__)
    elif args.action == 'publish':
        context = read(args.context)
        rt = {key: context[key] for key in ('project','python','frozen_sources')}
        project = PurePosixPath(rt['project'])
        releases, runs = project/'releases', project/'runs'
        parent = releases/f'{PARENT}-release'
        rt.update(run_id=RUN, runner=__spec__.name, code_commit=release_commit(args.repo), reference=str(parent),
                  release_root=str(releases/f'{RUN}-release'), run_root=str(runs/RUN), panel_root=str(runs/PANEL),
                  waveform_root=str(runs/WAVEFORM_RUN), max_compute_processes=8,
                  external_releases=list(map(str, (parent,runs/PARENT,runs/PANEL,runs/PANEL_CHECK,runs/WAVEFORM_RUN))))
        release(args.repo, args.output, rt, args.ssh_config)
    else:
        from experiments.pamf_lora_main.runner import activate

        rt = read(args.runtime)
        parent = reference(rt)
        activate(parent)
        (predict if args.action == 'predict' else score)(rt, parent)


if __name__ == '__main__':
    main()
