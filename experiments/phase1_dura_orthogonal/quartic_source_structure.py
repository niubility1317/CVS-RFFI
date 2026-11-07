"""Frozen source-only structure comparison for the quartic DURA candidate."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.pamf_phase1_flow import experiment as core
from experiments.pamf_phase1_flow.anchor_source_smoke import frozen_dura
from experiments.phase1_dura_innovation_probe.source_probe import source_inputs
from experiments.phase1_dura_orthogonal.envelope_source_structure import physical_ranks, summarize_ranks
from experiments.phase1_dura_orthogonal.quartic_recovery import DuraQuarticRecovery, OFFSETS, QUARTIC_WEIGHT, BRANCH_SCALE
from experiments.phase1_dura_orthogonal.source_quartic_equal_dim import features
from experiments.phase1_dura_orthogonal.source_retrieval_geometry import descriptors
from experiments.phase1_dura_orthogonal.train_source import SCHEMA, WIDE_FIT_SIZE, extract_context

CANDIDATE = '20261107-phase1-dura-quartic-source-s2026092701-r02'
CANDIDATE_COMMIT = 'be39d46aea181caca21c7ce401d1a39d7eba7b4e'


def load_candidate(root, pilot, roles):
    folder = root/'runs'/CANDIDATE
    summary = core.read(folder/'source_summary.json')
    state = torch.load(folder/'final.pt', map_location='cpu', weights_only=False)
    fit = np.asarray(state.get('fit_ids'))
    expected = dict(commit=CANDIDATE_COMMIT, epochs=5, architecture='dura_quartic',
                    objective='wide_quartic_guard', source_fit_size=WIDE_FIT_SIZE,
                    DURA_training_reads=1, DURA_frozen=True, source_quartic_weight=QUARTIC_WEIGHT,
                    quartic_offsets=list(OFFSETS), quartic_branch_scale=BRANCH_SCALE,
                    TX_labels_read=0, F_weights_read=0, F_outputs_read=0, V_reads=0, target_reads=0)
    if (summary.get('status') != 'SOURCE_TRAINED' or summary.get('fit_ids') != WIDE_FIT_SIZE
            or state.get('schema') != SCHEMA or fit.shape != (WIDE_FIT_SIZE,) or fit.dtype.kind not in ('i', 'u')
            or len(np.unique(fit)) != WIDE_FIT_SIZE or fit.min() < 0 or fit.max() >= len(roles)
            or not np.isin(roles[fit], ('L_s', 'U_s')).all() or np.intersect1d(fit, pilot).size
            or any(obj.get(k) != v for obj in (summary, state) for k, v in expected.items())):
        raise ValueError('Matching source-only quartic candidate and disjoint source pilot required')
    model = DuraQuarticRecovery(width=state['width']).eval().requires_grad_(False)
    model.load_state_dict(state['model'], strict=True)
    return model, fit


def structure(iq):
    power, coupling = features(iq)
    _, amplitude = descriptors(iq)
    return power, amplitude, coupling


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Unique source diagnostic output required')
    torch.set_num_threads(2)
    ids, rx, scene, _, received, clean = source_inputs(args.root)
    complete = core.read(args.root/'runs'/core.F_RUN/'F'/core.F_ROW/'source/complete.json')
    source_contract = core.read(complete['config']['source_contract'])
    _, train_clean, roles = core.data(args.root, source_contract)
    candidate, fit = load_candidate(args.root, ids, roles)
    dura = frozen_dura(args.root, 'cpu')
    with torch.no_grad():
        norm = structure(core.tensor(train_clean[fit[:512]], 'cpu'))
        reference = structure(core.tensor(clean[ids], 'cpu'))
        stats = [(v.mean(0), v.std(0, unbiased=False).clamp_min(.001)) for v in norm]
        anchor_parts, candidate_parts = [], []
        for start in range(0, len(ids), 8):
            x = core.tensor(received[ids[start:start+8]], 'cpu')
            anchor, direction, variance = extract_context(dura, x)
            output = candidate(x, anchor, direction, variance)
            if not torch.isfinite(output).all():
                raise FloatingPointError('Finite source IQ output required')
            anchor_parts.append(anchor)
            candidate_parts.append(output)
            if start == 0:
                print('QUARTIC_STRUCTURE_REAL_DURA_CANDIDATE_SMOKE_PASS', flush=True)
            if (start+8) % 168 == 0:
                print('QUARTIC_STRUCTURE_PROGRESS', start+8, len(ids), flush=True)
        if any(p.requires_grad or p.grad is not None for model in (dura, candidate) for p in model.parameters()):
            raise ValueError('DURA and candidate must remain frozen')
        anchor, output = structure(torch.cat(anchor_parts)), structure(torch.cat(candidate_parts))
        ranks = {name: (physical_ranks(a, ref, *stat), physical_ranks(b, ref, *stat))
                 for name, a, b, ref, stat in zip(('power', 'amplitude', 'quartic'), anchor, output, reference, stats)}
        groups = {'all': np.ones(len(ids), dtype=bool)}
        for name, receivers in (('RX4_6', (4, 6)), ('RX8', (8,))):
            for view in core.VIEWS:
                mask = np.isin(rx, receivers) & (scene == view)
                if mask.sum() != 48:
                    raise ValueError('Complete fixed source groups required')
                groups[name+'/'+view] = mask
        results = {}
        for group, values in groups.items():
            mask = torch.tensor(values.tolist(), dtype=torch.bool)
            results[group] = {}
            for feature, (a, b) in ranks.items():
                results[group][feature] = dict(dura=summarize_ranks(a, mask), candidate=summarize_ranks(b, mask),
                    paired_top1_repairs=int(((a != 1) & (b == 1) & mask).sum()),
                    paired_top1_damages=int(((a == 1) & (b != 1) & mask).sum()))
    args.out.mkdir(parents=True)
    result = dict(status='QUARTIC_SOURCE_STRUCTURE_COMPLETE', commit=args.commit,
                  candidate=CANDIDATE, candidate_commit=CANDIDATE_COMMIT, source_records=len(ids),
                  source_normalization_fit_records=512, source_pilot_disjoint_from_candidate_fit=True,
                  physical_record_retrieval_not_transmitter_accuracy=True, TX_labels_read=0, F_weights_read=0,
                  F_outputs_read=0, V_reads=0, target_reads=0, weights_saved=False, results=results)
    (args.out/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf8')
    print('QUARTIC_SOURCE_STRUCTURE_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
