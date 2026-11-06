"""Source-only paired-record structure in a fixed masked-IQ prior."""
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
from experiments.phase1_dura_orthogonal.source_ssl_representation_probe import matching_metrics, feature_statistics

MASK_START = 112


def features(model, iq):
    starts = torch.full((len(iq),), MASK_START, device=iq.device, dtype=torch.long)
    observed, mask, scale, _ = masked_inputs(iq, starts)
    hidden = F.silu(model.entry(torch.cat((observed/scale, mask[:, None].to(iq.dtype)), 1)))
    for block in model.blocks:
        hidden = hidden + .2*F.silu(block(hidden))
    summary = torch.cat((hidden.mean(-1), hidden.std(-1, unbiased=False)), 1)
    return F.normalize(summary, dim=1)


def raw_features(iq):
    starts = torch.full((len(iq),), MASK_START, device=iq.device, dtype=torch.long)
    observed, _, _, _ = masked_inputs(iq, starts)
    return F.normalize(observed.flatten(1), dim=1)


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
    x = core.tensor(received[ids[:8]], 'cpu')
    y = core.tensor(clean[ids[:8]], 'cpu')
    if not all(torch.isfinite(z).all() for z in (features(trained, y), dura(x))):
        raise FloatingPointError('Finite real-checkpoint source smoke required')
    print('MASKED_REPRESENTATION_SOURCE_SMOKE_PASS 8', flush=True)
    started = time.monotonic()
    values = {name: {view: [] for view in ('clean', 'received', 'dura')}
              for name in ('masked_prior', 'random_prior', 'raw_iq')}
    for offset in range(0, len(ids), 16):
        selected = ids[offset:offset+16]
        x, y = core.tensor(received[selected], 'cpu'), core.tensor(clean[selected], 'cpu')
        anchor = dura(x)
        for view, iq in (('clean', y), ('received', x), ('dura', anchor)):
            for name, model in (('masked_prior', trained), ('random_prior', random)):
                values[name][view].append(features(model, iq))
            values['raw_iq'][view].append(raw_features(iq))
    for group in values.values():
        for view in group:
            group[view] = torch.cat(group[view])
    groups = {}
    public_groups = sorted({(int(r), str(v)) for r, v in zip(rx, scene)})
    for receiver, view in public_groups:
        selected = np.flatnonzero((rx == receiver) & (scene == view)).tolist()
        if len(selected) < 2:
            raise ValueError('At least two records per source receiver and view required')
        for name, vectors in values.items():
            for input_view in ('received', 'dura'):
                groups[f'{name}/RX{receiver}/{view}/{input_view}'] = matching_metrics(
                    vectors[input_view][selected], vectors['clean'][selected])
    stats = {name: feature_statistics(group['clean']) for name, group in values.items() if name != 'raw_iq'}
    args.out.mkdir(parents=True)
    core.write(args.out/'result.json', dict(status='MASKED_REPRESENTATION_SOURCE_COMPLETE', commit=args.commit,
        source_checkpoint=SOURCE, source_checkpoint_commit=SOURCE_COMMIT, records=len(ids),
        public_group_count=len(public_groups), groups=groups, clean_feature_statistics=stats,
        fixed_mask_start=MASK_START, elapsed_seconds=time.monotonic()-started,
        TX_labels_read=0, F_weights_read=0, F_outputs_read=0, V_reads=0, target_reads=0, model_updates=0,
        caveat='Same-record retrieval may use packet, content, or receiver cues; it is not TX identity accuracy. Source pilot previously explored.'))
    print('MASKED_REPRESENTATION_SOURCE_COMPLETE', len(ids), flush=True)


if __name__ == '__main__':
    main()
