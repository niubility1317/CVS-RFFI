"""Test whether time-resolved source-prior features retain record differences."""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from experiments.pamf_phase1_flow import experiment as core
from experiments.pamf_phase1_flow.anchor_source_smoke import frozen_dura
from experiments.phase1_dura_innovation_probe.source_probe import source_inputs
from experiments.phase1_dura_orthogonal.source_masked_prior import MaskedIQPrior, masked_inputs
from experiments.phase1_dura_orthogonal.source_masked_prior_compare import SOURCE, SOURCE_COMMIT
from experiments.phase1_dura_orthogonal.source_masked_representation_probe import MASK_START, raw_features
from experiments.phase1_dura_orthogonal.source_ssl_representation_probe import matching_metrics

BINS = (8, 32, 256)


def descriptors(model, iq):
    starts = torch.full((len(iq),), MASK_START, device=iq.device, dtype=torch.long)
    observed, mask, scale, _ = masked_inputs(iq, starts)
    h = F.silu(model.entry(torch.cat((observed/scale, mask[:, None].to(iq.dtype)), 1)))
    for block in model.blocks:
        h = h + .2*F.silu(block(h))
    global_features = torch.cat((h.mean(-1), h.std(-1, unbiased=False)), 1)
    centered = h-h.mean(-1, keepdim=True)
    result = {'global': F.normalize(global_features, dim=1)}
    for bins in BINS:
        result['temporal_'+str(bins)] = F.normalize(F.adaptive_avg_pool1d(centered, bins).flatten(1), dim=1)
    return result


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--commit', required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError('Unique source output required')
    torch.set_num_threads(2)
    ids, rx, scene, _, received, clean = source_inputs(args.root)
    folder = args.root/'runs'/SOURCE
    summary = core.read(folder/'result.json')
    state = torch.load(folder/'final.pt', map_location='cpu', weights_only=False)
    if (summary.get('status') != 'SOURCE_MASKED_PRIOR_COMPLETE' or
            summary.get('commit') != SOURCE_COMMIT or state.get('commit') != SOURCE_COMMIT or
            any(summary.get(k) != 0 for k in ('TX_labels_read', 'F_weights_read', 'F_outputs_read', 'V_reads', 'target_reads')) or
            len(state['fit_ids']) != 32768 or np.intersect1d(ids, state['fit_ids']).size):
        raise ValueError('Fixed source-only prior and disjoint pilot required')
    torch.manual_seed(2026110707)
    random = MaskedIQPrior().eval().requires_grad_(False)
    trained = MaskedIQPrior().eval().requires_grad_(False)
    trained.load_state_dict(state['model'], strict=True)
    dura = frozen_dura(args.root, 'cpu')
    x, y = core.tensor(received[ids[:8]], 'cpu'), core.tensor(clean[ids[:8]], 'cpu')
    if not torch.isfinite(dura(x)).all() or not all(torch.isfinite(v).all() for v in descriptors(trained, y).values()):
        raise FloatingPointError('Finite real-checkpoint source smoke required')
    print('MASKED_TEMPORAL_SOURCE_SMOKE_PASS 8', flush=True)
    started = time.monotonic()
    vectors = {}
    for offset in range(0, len(ids), 16):
        selected = ids[offset:offset+16]
        x, y = core.tensor(received[selected], 'cpu'), core.tensor(clean[selected], 'cpu')
        anchor = dura(x)
        for view, iq in (('clean', y), ('received', x), ('dura', anchor)):
            for name, model in (('masked_prior', trained), ('random_prior', random)):
                for descriptor, value in descriptors(model, iq).items():
                    vectors.setdefault((name, descriptor, view), []).append(value)
            vectors.setdefault(('raw_iq', 'visible', view), []).append(raw_features(iq))
    vectors = {key: torch.cat(value) for key, value in vectors.items()}
    groups = {}
    public_groups = sorted({(int(r), str(v)) for r, v in zip(rx, scene)})
    for receiver, scene_name in public_groups:
        selected = np.flatnonzero((rx == receiver) & (scene == scene_name)).tolist()
        if len(selected) < 2:
            raise ValueError('At least two records per source receiver and view required')
        for name, descriptor in sorted({(k[0], k[1]) for k in vectors}):
            for view in ('received', 'dura'):
                groups[f'{name}/{descriptor}/RX{receiver}/{scene_name}/{view}'] = matching_metrics(
                    vectors[(name, descriptor, view)][selected], vectors[(name, descriptor, 'clean')][selected])
    args.out.mkdir(parents=True)
    core.write(args.out/'result.json', dict(status='MASKED_TEMPORAL_SOURCE_COMPLETE', commit=args.commit,
        source_checkpoint=SOURCE, source_checkpoint_commit=SOURCE_COMMIT, records=len(ids),
        public_group_count=len(public_groups), groups=groups, temporal_bins=list(BINS),
        fixed_mask_start=MASK_START, elapsed_seconds=time.monotonic()-started,
        TX_labels_read=0, F_weights_read=0, F_outputs_read=0, V_reads=0, target_reads=0, model_updates=0,
        caveat='Source same-record retrieval may exploit packet, content or receiver cues; not TX identity accuracy.'))
    print('MASKED_TEMPORAL_SOURCE_COMPLETE', len(ids), flush=True)


if __name__ == '__main__':
    main()
