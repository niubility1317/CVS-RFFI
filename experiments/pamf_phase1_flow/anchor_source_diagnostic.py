"""Measure why a fixed anchored field changes DURA on Phase1 source examples."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from . import anchor_balanced as anchor, experiment as core
from .dura_source_trisig_diagnostic import PARENT

RUN = '20261101-diagnostic-phase1-anchor-source-paired-s2026092701-r02'
MODEL_RUN = '20261101-phase1-dura-anchored-balanced-E30-s2026092701-r02'
MODEL_COMMIT = 'f16f514179428d16bd019a23aca56dd82aa93b1b'
MODEL_DURA_SHA = '14200ac0f8eef01f57aa82e8fb5122ff855fc81208983d9234d6c5caef2e28ee'


def paired_counts(reference, dura, candidate, mask):
    values = [np.asarray(v) for v in (reference, dura, candidate, mask)]
    if any(v.ndim != 1 or v.shape != values[0].shape for v in values) or values[3].dtype != np.bool_:
        raise ValueError('Aligned one-dimensional predictions and boolean mask required')
    ref, old, new, mask = values
    before, after = old == ref, new == ref
    return dict(count=int(mask.sum()), dura_matches=int((mask & before).sum()),
                candidate_matches=int((mask & after).sum()),
                repairs=int((mask & ~before & after).sum()), damages=int((mask & before & ~after).sum()))


def masked_mean(values, mask):
    # N607's NumPy bridge can reject ndarray even when NumPy itself is loaded.
    return float(values[torch.tensor(mask.tolist(), dtype=torch.bool)].mean())


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
    digest = anchor.sha(args.root / anchor.DURA)
    if digest != MODEL_DURA_SHA:
        raise ValueError('Exact frozen DURA checkpoint required')
    model = anchor.load(args.root / 'runs' / MODEL_RUN / 'flow/final.pt', 'cpu', MODEL_COMMIT, digest)
    source = args.root / 'runs' / PARENT
    if core.read(source / 'launch.json').get('commit') != '7e2d22ed1fe0feeac1df8381be89e4bb527108c5':
        raise ValueError('Pinned source producer required')
    ids = np.load(cache / 'train/physical_ids.npy', allow_pickle=False)
    fixed = np.concatenate([np.flatnonzero(roles == role)[np.linspace(0, count - 1, n, dtype=int)]
                            for role, count, n in (('L_s', 6300, 128), ('U_s', 56700, 384))])
    results = []
    for view in (1, 20):
        with np.load(source / f'DURA_default-{view}.npz', allow_pickle=False) as z:
            saved = {k: z[k] for k in ('source_positions', 'physical_ids', 'roles', 'clean', 'native', 'scene', 'state')}
        ix = saved['source_positions']
        if not (np.array_equal(ix, fixed) and np.array_equal(saved['physical_ids'], ids[ix])
                and np.array_equal(saved['roles'], roles[ix]) and np.array_equal(saved['clean'], clean[ix])):
            raise ValueError('Exact source physical alignment required')
        raw = np.load(cache / 'epochs' / f'{view:02d}' / 'received.npy', mmap_mode='r', allow_pickle=False)
        if raw.shape != (core.COUNT, 2, 256) or saved['native'].shape != (512, 2, 256):
            raise ValueError('Complete paired source IQ required')
        observed = {k: [] for k in ('clean_logits', 'dura_logits', 'candidate_logits', 'dura_nmse', 'candidate_nmse', 'candidate_delta')}
        for offset in range(0, 512, 32):
            sl = slice(offset, offset + 32)
            x = core.tensor(raw[ix[sl]], 'cpu')
            clean_batch = core.tensor(saved['clean'][sl], 'cpu')
            old = dura(x)
            # Confirm the cached comparison really represents this frozen DURA.
            torch.testing.assert_close(old, core.tensor(saved['native'][sl], 'cpu'), rtol=1e-4, atol=1e-5)
            new = model(x, old)
            energy = clean_batch.square().mean((1, 2)).clamp_min(1e-4)
            for key, iq in (('clean', clean_batch), ('dura', old), ('candidate', new)):
                logits = teacher.classify_features(teacher.features(iq))
                if not torch.isfinite(logits).all():
                    raise ValueError('Finite frozen teacher outputs required')
                observed[key + '_logits'].append(logits.cpu())
            observed['dura_nmse'].append(((old-clean_batch).square().mean((1, 2))/energy).cpu())
            observed['candidate_nmse'].append(((new-clean_batch).square().mean((1, 2))/energy).cpu())
            observed['candidate_delta'].append(((new-old).square().mean((1, 2))/energy).cpu())
        observed = {k: torch.cat(v) for k, v in observed.items()}
        ref = observed['clean_logits'].argmax(1).numpy()
        before = observed['dura_logits'].argmax(1).numpy()
        after = observed['candidate_logits'].argmax(1).numpy()
        p = F.softmax(observed['clean_logits'], dim=1)
        logp = F.log_softmax(observed['clean_logits'], dim=1)
        for k in ('dura', 'candidate'):
            observed[k + '_kl'] = (p * (logp - F.log_softmax(observed[k + '_logits'], dim=1))).sum(1)
        masks = {'all': np.ones(len(ix), dtype=bool)}
        for key in ('scene', 'state', 'roles'):
            for value in np.unique(saved[key]):
                masks[f'{key}:{value}'] = saved[key] == value
        groups = []
        for label, mask in masks.items():
            counts = paired_counts(ref, before, after, mask)
            if counts['candidate_matches'] - counts['dura_matches'] != counts['repairs'] - counts['damages']:
                raise AssertionError('Paired count identity violated')
            counts['group'] = label
            for metric in ('dura_nmse', 'candidate_nmse', 'candidate_delta', 'dura_kl', 'candidate_kl'):
                counts[metric] = masked_mean(observed[metric], mask)
            groups.append(counts)
        results.append(dict(source_view=view, count=len(ix), groups=groups))
    out = args.root / 'runs' / RUN
    if out.exists():
        raise FileExistsError('Unique diagnostic output required')
    out.mkdir()
    core.write(out / 'complete.json', dict(status='ANALYZED_SOURCE_ONLY', run=RUN, commit=args.commit,
        model_run=MODEL_RUN, model_commit=MODEL_COMMIT, DURA_sha256=digest, F_row=core.F_ROW,
        source_unique_records=512, source_observations=1024, source_views=[1,20], TX_labels_read=0,
        V_reads=0, test_IQ_reads=0, network_updates=0, results=results,
        interpretation='In-training source consistency to clean teacher is not TX accuracy or independent generalization.'))
    print(json.dumps(dict(status='ANALYZED_SOURCE_ONLY', run=RUN)), flush=True)


if __name__ == '__main__':
    main()
