"""Fixed Phase1 paired-source FIR research diagnostic; not an inference method."""
import argparse
from pathlib import Path

import numpy as np
import torch

from . import anchor_balanced as anchor, experiment as core
from .dura_source_trisig_diagnostic import PARENT
from .feature_conditioned_experiment import source_labels
from .source_fir_oracle import fit_paired_fir

RUN = '20261101-diagnostic-phase1-source-fir-oracle-s2026092701-r01'
DURA_SHA = '14200ac0f8eef01f57aa82e8fb5122ff855fc81208983d9234d6c5caef2e28ee'
TAPS = (1, 9, 33)


def paired(reference, old, new, mask):
    old_correct, new_correct = old == reference, new == reference
    return dict(count=int(mask.sum()), before=int((mask & old_correct).sum()),
                after=int((mask & new_correct).sum()), repairs=int((mask & ~old_correct & new_correct).sum()),
                damages=int((mask & old_correct & ~new_correct).sum()))


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    if Path('release_commit.txt').read_text(encoding='ascii').strip() != args.commit:
        raise ValueError('Pinned diagnostic release required')
    torch.set_num_threads(2)
    teacher, contract = core.teacher(args.root, 'cpu')
    cache, clean, roles = core.data(args.root, contract)
    dura = anchor.frozen_dura(args.root, 'cpu')
    if anchor.sha(args.root / anchor.DURA) != DURA_SHA:
        raise ValueError('Exact frozen DURA checkpoint required')
    labels = source_labels(args.root, cache, clean, roles, contract)
    ids = np.load(cache / 'train/physical_ids.npy', allow_pickle=False)
    fixed = np.concatenate([np.flatnonzero(roles == role)[np.linspace(0, count - 1, n, dtype=int)]
                            for role, count, n in (('L_s', 6300, 128), ('U_s', 56700, 384))])
    if not np.all(labels[fixed[128:]] == -1):
        raise ValueError('U_s source labels must never be materialized')
    source = args.root / 'runs' / PARENT
    if core.read(source / 'launch.json').get('commit') != '7e2d22ed1fe0feeac1df8381be89e4bb527108c5':
        raise ValueError('Pinned source producer required')
    results = []
    for view in (1, 20):
        with np.load(source / f'DURA_default-{view}.npz', allow_pickle=False) as z:
            saved = {k: z[k] for k in ('source_positions', 'physical_ids', 'roles', 'clean', 'native', 'scene', 'state')}
        ix = saved['source_positions']
        if not (np.array_equal(ix, fixed) and np.array_equal(saved['physical_ids'], ids[ix])
                and np.array_equal(saved['roles'], roles[ix]) and np.array_equal(saved['clean'], clean[ix])):
            raise ValueError('Exact aligned Phase1 source rows required')
        received = np.load(cache / 'epochs' / f'{view:02d}' / 'received.npy', mmap_mode='r', allow_pickle=False)
        names = ['clean', 'raw', 'dura'] + [f'{base}_fir{n}' for base in ('raw', 'dura') for n in TAPS]
        logits = {name: [] for name in names}
        errors = {name: [] for name in names if name != 'clean'}
        for start in range(0, len(ix), 32):
            sl = slice(start, start + 32)
            x = core.tensor(received[ix[sl]], 'cpu')
            target = core.tensor(saved['clean'][sl], 'cpu')
            native = dura(x)
            torch.testing.assert_close(native, core.tensor(saved['native'][sl], 'cpu'), rtol=1e-4, atol=1e-5)
            values = {'clean': target, 'raw': x, 'dura': native}
            for base in ('raw', 'dura'):
                for n in TAPS:
                    values[f'{base}_fir{n}'] = fit_paired_fir(values[base], target, taps=n, widely_linear=True, ridge=1e-4)
            energy = target.square().mean((1, 2)).clamp_min(1e-4)
            for name, waveform in values.items():
                scores = teacher.classify_features(teacher.features(waveform))
                if not torch.isfinite(scores).all():
                    raise FloatingPointError('Finite frozen Tri-SIG-TIME scores required')
                logits[name].append(scores.argmax(1).cpu())
                if name != 'clean':
                    errors[name].append(((waveform-target).square().mean((1, 2))/energy).cpu())
        decisions = {key: np.from_dlpack(torch.cat(value)) for key, value in logits.items()}
        errors = {key: torch.cat(value) for key, value in errors.items()}
        masks = {'all': np.ones(len(ix), dtype=bool), 'blocked': saved['state'] == 'BLOCKED',
                 'labeled': saved['roles'] == 'L_s'}
        groups = []
        for group, mask in masks.items():
            for name in names:
                if name == 'clean':
                    continue
                selected = torch.tensor(mask.tolist(), dtype=torch.bool)
                clean_match = paired(decisions['clean'], decisions['dura'], decisions[name], mask)
                labeled_mask = mask & (saved['roles'] == 'L_s')
                label_match = paired(labels[ix], decisions['dura'], decisions[name], labeled_mask)
                groups.append(dict(group=group, method=name, clean_teacher=clean_match,
                                   labeled_truth=label_match, mean_iq_nmse=float(errors[name][selected].mean())))
        results.append(dict(source_view=view, count=len(ix), groups=groups))
        print(f'COMPLETED_SOURCE_VIEW {view}', flush=True)
    out = args.root / 'runs' / RUN
    out.mkdir()
    core.write(out / 'complete.json', dict(status='ANALYZED_SOURCE_ONLY', run=RUN, commit=args.commit,
        source_unique_records=512, source_observations=1024, source_views=[1, 20],
        materialized_L_s_labels=6300, evaluated_L_s_labels=128, U_s_labels_read=0, V_reads=0, test_IQ_reads=0, DURA_sha256=DURA_SHA,
        F_run=core.F_RUN, F_row=core.F_ROW, taps=list(TAPS), ridge=1e-4, widely_linear=True,
        privileged_clean_for_each_window=True, network_updates=0, results=results,
        interpretation='Noncausal paired clean oracle, not a deployable receiver or classifier-accuracy upper bound.'))


if __name__ == '__main__':
    main()
