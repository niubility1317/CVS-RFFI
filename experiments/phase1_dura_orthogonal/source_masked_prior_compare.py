"""Compare a fixed source masked-IQ prior to a visible-only lag baseline."""
import argparse
from pathlib import Path

import numpy as np
import torch

from experiments.pamf_phase1_flow import experiment as core
from experiments.phase1_dura_innovation_probe.source_probe import source_inputs
from experiments.phase1_dura_orthogonal.source_masked_prior import MaskedIQPrior, masked_inputs, errors, SEED, SPAN

SOURCE = '20261107-phase1-masked-iq-prior-source-s2026092701-r01'
SOURCE_COMMIT = 'acdfd17ae205fa5e04b32a5ecdd5eea5d7bb21d4'
LAG = 40


def lag_prediction(observed, mask):
    """Use only visible IQ, estimating a common complex phase from visible pairs."""
    if observed.ndim != 3 or observed.shape[1:] != (2, 256) or mask.shape != (len(observed), 256):
        raise ValueError('Masked IQ input required')
    z = torch.complex(observed[:, 0], observed[:, 1])
    valid = ~mask[:, LAG:] & ~mask[:, :-LAG]
    cross = ((z[:, LAG:] * z[:, :-LAG].conj()) * valid).sum(-1)
    phase = torch.where(cross.abs() > 1e-10, cross/cross.abs().clamp_min(1e-10), torch.ones_like(cross))
    left = torch.cat((torch.zeros_like(z[:, :LAG]), z[:, :-LAG] * phase[:, None]), -1)
    right = torch.cat((z[:, LAG:] * phase.conj()[:, None], torch.zeros_like(z[:, :LAG])), -1)
    has_left = torch.cat((torch.zeros_like(mask[:, :LAG]), ~mask[:, :-LAG]), -1)
    has_right = torch.cat((~mask[:, LAG:], torch.zeros_like(mask[:, :LAG])), -1)
    count = has_left.to(z.real.dtype) + has_right.to(z.real.dtype)
    if bool((count[mask] == 0).any()):
        raise ValueError('Every masked sample needs a visible lag neighbour')
    result = (left*has_left + right*has_right)/count.clamp_min(1)
    return torch.stack((result.real, result.imag), 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--commit', required=True)
    args = p.parse_args()
    if args.out.exists() or not torch.cuda.is_available():
        raise ValueError('Unique output and registered GPU required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ids, rx, scene, _, _, clean = source_inputs(args.root)
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
    random = np.random.default_rng(SEED+1)
    starts = torch.tensor(random.integers(1, 256-SPAN, size=len(ids)).tolist(), device='cuda:0')
    values = []
    with torch.no_grad():
        for offset in range(0, len(ids), 32):
            y = core.tensor(clean[ids[offset:offset+32]], 'cuda:0')
            observed, mask, scale, linear = masked_inputs(y, starts[offset:offset+32])
            pred = model(observed, mask, scale)
            lag = lag_prediction(observed, mask)
            batch = torch.stack((errors(pred, y, mask), errors(lag, y, mask), errors(linear, y, mask)), 1)
            if not torch.isfinite(batch).all():
                raise FloatingPointError('Finite source predictions required')
            if offset == 0:
                print('MASKED_PRIOR_COMPARE_SMOKE_PASS', len(y), flush=True)
            values.append(batch.cpu())
    values = np.from_dlpack(torch.cat(values).contiguous()).copy()
    groups = {}
    for name, receivers in (('RX4_6', (4, 6)), ('RX8', (8,))):
        for view in core.VIEWS:
            selected = values[np.isin(rx, receivers) & (scene == view)]
            if len(selected) != 48:
                raise ValueError('48 source pilot records per group required')
            groups[name+'/'+view] = dict(count=len(selected), prior=float(selected[:, 0].mean()),
                lag40=float(selected[:, 1].mean()), linear=float(selected[:, 2].mean()),
                prior_beats_lag40=float((selected[:, 0] < selected[:, 1]).mean()))
    args.out.mkdir(parents=True)
    core.write(args.out/'result.json', dict(status='SOURCE_MASKED_PRIOR_COMPARE_COMPLETE', commit=args.commit,
        prior_commit=SOURCE_COMMIT, records=len(ids), lag=LAG, groups=groups,
        TX_labels_read=0, F_weights_read=0, F_outputs_read=0, V_reads=0, target_reads=0,
        caveat='Source clean masked IQ only; no satellite restoration or identity accuracy.'))
    print('MASKED_PRIOR_COMPARE_COMPLETE', len(ids), flush=True)


if __name__ == '__main__':
    main()
