"""Source-only paired-IQ representation diagnostic; not a TX classifier."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from experiments.pamf_phase1_flow import experiment as core
from experiments.pamf_phase1_flow.anchor_source_smoke import frozen_dura
from experiments.phase1_dura_innovation_probe.source_probe import source_inputs
from experiments.phase1_dura_orthogonal.controller import training_environments
from experiments.phase1_dura_orthogonal.joint_ssl_iq import SourceIQEncoder
from experiments.phase1_dura_orthogonal.multiscale_spectral import DuraMultiscaleSpectral
from experiments.phase1_dura_orthogonal.train_source import extract_context

SOURCE = '20261106-phase1-dura-joint-iq-ssl-source-s2026092701-r01'
SOURCE_COMMIT = '63ab9a08fe9f1fb076168467360e85cbc70ccafe'


def matching_metrics(view, clean):
    """Match a view to its own clean record among records of one public RX/scene."""
    if view.ndim != 2 or view.shape != clean.shape or len(view) < 2:
        raise ValueError('At least two paired source embeddings required')
    scores = view @ clean.T
    n = len(view)
    matched = scores.diagonal()
    other_mean = (scores.sum()-matched.sum())/(n*(n-1))
    return dict(count=n, matched_cosine=float(matched.mean()), other_cosine=float(other_mean),
                matched_minus_other=float(matched.mean()-other_mean),
                record_top1=float((scores.argmax(1) == torch.arange(n, device=scores.device)).float().mean()))


def feature_statistics(values):
    centered = values-values.mean(0, keepdim=True)
    eigen = torch.linalg.eigvalsh(centered.T @ centered/len(values)).clamp_min(0)
    total = eigen.sum()
    if total <= 0:
        return dict(feature_std_mean=0., covariance_effective_rank=0.)
    p = eigen/total
    rank = torch.exp(-(p[p > 0]*p[p > 0].log()).sum())
    return dict(feature_std_mean=float(centered.square().mean(0).sqrt().mean()),
                covariance_effective_rank=float(rank))


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    training_environments('cuda:0', os.environ)
    if args.out.exists():
        raise FileExistsError('Unique source representation diagnostic output required')
    args.out.mkdir(parents=True)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ids, rx, scene, _, received, clean = source_inputs(args.root)
    source = args.root/'runs'/SOURCE
    summary = core.read(source/'source_summary.json')
    expected = dict(status='SOURCE_TRAINED', commit=SOURCE_COMMIT, objective='ssl_joint_guard',
                    architecture='dura_multiscale', TX_labels_read=0, F_weights_read=0,
                    F_outputs_read=0, V_reads=0, target_reads=0, DURA_frozen=True)
    if any(summary.get(k) != v for k, v in expected.items()):
        raise ValueError('Fixed source-only joint SSL training required')
    state = torch.load(source/'final.pt', map_location='cpu', weights_only=False)
    if (state.get('commit') != SOURCE_COMMIT or state.get('objective') != 'ssl_joint_guard' or
            np.intersect1d(ids, state['fit_ids']).size):
        raise ValueError('Source pilot must be outside the fixed candidate fitting IDs')
    model = DuraMultiscaleSpectral(width=32, taps=9)
    model.load_state_dict(state['model'], strict=True)
    model = model.to('cuda:0').eval().requires_grad_(False)
    torch.manual_seed(2026110615)
    random_encoder = SourceIQEncoder().to('cuda:0').eval().requires_grad_(False)
    learned_encoder = SourceIQEncoder()
    learned_encoder.load_state_dict(state['source_ssl_aux_state'], strict=True)
    learned_encoder = learned_encoder.to('cuda:0').eval().requires_grad_(False)
    dura = frozen_dura(args.root, 'cuda:0')
    x = core.tensor(received[ids[:2]], 'cuda:0')
    anchor, direction, variance = extract_context(dura, x)
    if not torch.equal(anchor, dura(x)):
        raise ValueError('Fixed DURA context must match its native output')
    smoke = model(x, anchor, direction, variance)
    if not torch.isfinite(learned_encoder(smoke)).all():
        raise FloatingPointError('Finite real-checkpoint source representations required')
    print('SOURCE_REPRESENTATION_SMOKE_PASS', flush=True)
    start = time.monotonic()
    embeddings = {name: {view: [] for view in ('clean', 'received', 'dura', 'candidate')}
                  for name in ('random', 'learned')}
    for offset in range(0, len(ids), 32):
        selected = ids[offset:offset+32]
        x = core.tensor(received[selected], 'cuda:0')
        y = core.tensor(clean[selected], 'cuda:0')
        anchor, direction, variance = extract_context(dura, x)
        restored = model(x, anchor, direction, variance)
        for name, encoder in (('random', random_encoder), ('learned', learned_encoder)):
            for view, value in (('clean', y), ('received', x), ('dura', anchor), ('candidate', restored)):
                embeddings[name][view].append(encoder(value).cpu())
    for encoder in embeddings.values():
        for view in encoder:
            encoder[view] = torch.cat(encoder[view])
    results = {}
    public_groups = sorted({(int(r), str(s)) for r, s in zip(rx, scene)})
    for r, s in public_groups:
        selected = np.flatnonzero((rx == r) & (scene == s)).tolist()
        if len(selected) < 2:
            raise ValueError('At least two source records per RX and view required')
        for name, encoder in embeddings.items():
            for view in ('received', 'dura', 'candidate'):
                results[f'{name}/RX{r}/{s}/{view}'] = matching_metrics(
                    encoder[view][selected], encoder['clean'][selected])
    statistics = {name: feature_statistics(encoder['clean']) for name, encoder in embeddings.items()}
    core.write(args.out/'result.json', dict(status='SOURCE_REPRESENTATION_DIAGNOSTIC_COMPLETE',
        commit=args.commit, source_checkpoint=SOURCE, source_checkpoint_commit=SOURCE_COMMIT,
        records=len(ids), public_group_count=len(public_groups), source_pilot_not_candidate_fit=True,
        groups=results, clean_feature_statistics=statistics, elapsed_seconds=time.monotonic()-start,
        TX_labels_read=0, F_weights_read=0, F_outputs_read=0, V_reads=0, target_reads=0, model_updates=0,
        caveat='Record matching may exploit packet or content details; it is not TX identification or an identity accuracy metric. Source pilot was previously explored and is not independent of DURA.'))
    print('SOURCE_REPRESENTATION_DIAGNOSTIC_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
