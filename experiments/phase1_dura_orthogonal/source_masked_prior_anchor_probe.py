"""Source-only ideal-step probe of masked-prior reconstruction from DURA IQ."""
import argparse
import time
from pathlib import Path

import numpy as np
import torch

from experiments.pamf_phase1_flow import experiment as core
from experiments.pamf_phase1_flow.anchor_source_smoke import frozen_dura
from experiments.phase1_dura_innovation_probe.source_probe import source_inputs
from experiments.phase1_dura_orthogonal.source_masked_prior import MaskedIQPrior, masked_inputs, SPAN
from experiments.phase1_dura_orthogonal.source_masked_prior_compare import SOURCE, SOURCE_COMMIT

STARTS = (1, 33, 65, 97, 129, 161, 193, 223)


def prior_wave(model, anchor):
    """Eight masked model calls; preserve two endpoints and average overlaps."""
    total, count = torch.zeros_like(anchor), torch.zeros_like(anchor[:, :1])
    for start in STARTS:
        starts = torch.full((len(anchor),), start, device=anchor.device, dtype=torch.long)
        observed, mask, scale, _ = masked_inputs(anchor, starts)
        prediction = model(observed, mask, scale)
        total[:, :, start:start+SPAN] += prediction[:, :, start:start+SPAN]
        count[:, :, start:start+SPAN] += 1
    return torch.where(count > 0, total/count.clamp_min(1), anchor)


def ideal_probe(anchor, clean, prediction):
    direction = prediction-anchor
    energy = clean.square().mean((1, 2)).clamp_min(1e-8)
    denominator = direction.square().sum((1, 2)).clamp_min(1e-20)
    alpha = ((clean-anchor)*direction).sum((1, 2))/denominator
    def metric(step):
        return ((anchor+step[:, None, None]*direction-clean).square().mean((1, 2))/energy)
    return torch.stack((metric(torch.zeros_like(alpha)), metric(alpha.clamp(0, .5)),
                        metric(alpha.clamp(-.5, .5)), alpha), 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    if args.out.exists() or not torch.cuda.is_available():
        raise ValueError('Unique output and registered GPU required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ids, rx, scene, _, received, clean = source_inputs(args.root)
    folder = args.root/'runs'/SOURCE
    summary = core.read(folder/'result.json')
    state = torch.load(folder/'final.pt', map_location='cpu', weights_only=False)
    if (summary.get('status') != 'SOURCE_MASKED_PRIOR_COMPLETE' or
            summary.get('commit') != SOURCE_COMMIT or state.get('commit') != SOURCE_COMMIT or
            any(summary.get(k) != 0 for k in ('TX_labels_read', 'F_weights_read', 'F_outputs_read', 'V_reads', 'target_reads')) or
            len(state['fit_ids']) != 32768 or np.intersect1d(ids, state['fit_ids']).size):
        raise ValueError('Matching source-only prior and disjoint pilot required')
    model = MaskedIQPrior().to('cuda:0').eval()
    model.load_state_dict(state['model'], strict=True)
    dura = frozen_dura(args.root, 'cuda:0')
    def batch(selected):
        x, y = core.tensor(received[selected], 'cuda:0'), core.tensor(clean[selected], 'cuda:0')
        with torch.no_grad():
            anchor = dura(x)
            value = ideal_probe(anchor, y, prior_wave(model, anchor))
        if not torch.isfinite(value).all():
            raise FloatingPointError('Finite source-only ideal-step diagnostics required')
        return value.cpu()
    if batch(ids[:8]).shape != (8, 4):
        raise ValueError('Real checkpoint source smoke failed')
    print('MASKED_ANCHOR_SOURCE_SMOKE_PASS 8', flush=True)
    started = time.monotonic()
    values = np.from_dlpack(torch.cat([batch(ids[i:i+32]) for i in range(0, len(ids), 32)]).contiguous()).copy()
    groups = {}
    for name, receivers in (('RX4_6', (4, 6)), ('RX8', (8,))):
        for view in core.VIEWS:
            v = values[np.isin(rx, receivers) & (scene == view)]
            if len(v) != 48:
                raise ValueError('48 source pilot records per group required')
            groups[name+'/'+view] = dict(count=len(v), dura_nmse=float(v[:, 0].mean()),
                ideal_positive_nmse=float(v[:, 1].mean()), ideal_signed_nmse=float(v[:, 2].mean()),
                positive_improvement_fraction=float((v[:, 1] < v[:, 0]-1e-9).mean()),
                negative_step_fraction=float((v[:, 3] < 0).mean()))
    args.out.mkdir(parents=True)
    core.write(args.out/'result.json', dict(status='SOURCE_MASKED_ANCHOR_PROBE_COMPLETE', commit=args.commit,
        source_prior_commit=SOURCE_COMMIT, records=len(ids), starts=list(STARTS), groups=groups,
        elapsed_seconds=time.monotonic()-started, TX_labels_read=0, F_weights_read=0,
        F_outputs_read=0, V_reads=0, target_reads=0, model_updates=0,
        caveat='Source clean IQ supplies per-record ideal step, unavailable at deployment; not a performance bound or identity metric.'))
    print('MASKED_ANCHOR_SOURCE_COMPLETE', len(ids), flush=True)


if __name__ == '__main__':
    main()
