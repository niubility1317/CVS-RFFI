"""Source-only physical line search for a frozen DURA correction direction."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.pamf_phase1_flow import experiment as core
from experiments.pamf_phase1_flow.anchor_source_smoke import frozen_dura
from experiments.phase1_dura_innovation_probe.source_probe import source_inputs
from experiments.phase1_dura_orthogonal.large_context_stable import StableLargeContextDura
from experiments.phase1_dura_orthogonal.train_source import SCHEMA

SOURCE = '20261108-phase1-dura-large-context-stable-source-s2026092701-r01'
SOURCE_COMMIT = 'b873acec23758690e5484bd1c5f82162739c6655'


def bounded_step(residual, direction):
    """Minimize squared residual on [0, 1], choosing zero for a zero direction."""
    if residual.shape != direction.shape or residual.ndim != 2:
        raise ValueError('Matching complex residual and direction required')
    numerator = (direction.conj() * residual).real.sum(-1)
    denominator = direction.abs().square().sum(-1)
    return torch.where(denominator > 1e-16, numerator / denominator.clamp_min(1e-16), 0.).clamp(0, 1)


def metrics(received, anchor, candidate, clean, action):
    def c(x):
        return torch.complex(x[:, 0], x[:, 1])
    x, a, g, y = map(c, (received, anchor, candidate, clean))
    delta = g - a
    observed_direction = action(delta)
    observed_residual = x - action(a)
    physical = bounded_step(observed_residual, observed_direction)
    oracle = bounded_step(y - a, delta)
    reference = y.abs().square().sum(-1).clamp_min(1e-12)
    errors = torch.stack([(z-y).abs().square().sum(-1)/reference for z in
        (a, g, a + physical[:, None]*delta, a + oracle[:, None]*delta)], 1)
    return torch.cat((errors, physical[:, None], oracle[:, None]), 1)


def load_model(root, ids):
    state = torch.load(root/'runs'/SOURCE/'final.pt', map_location='cpu', weights_only=False)
    expected = dict(schema=SCHEMA, commit=SOURCE_COMMIT, architecture='large_context_stable',
        objective='wide_large_context_stable', epochs=20, TX_labels_read=0, F_weights_read=0,
        F_outputs_read=0, V_reads=0, target_reads=0, DURA_frozen=True)
    if any(state.get(k) != v for k, v in expected.items()) or len(state['fit_ids']) != 32768:
        raise ValueError('Matching source-only frozen model required')
    if np.intersect1d(ids, np.asarray(state['fit_ids'])).size:
        raise ValueError('Pilot records must not overlap candidate training records')
    model = StableLargeContextDura(width=state['width'], layers=state['large_context_layers'])
    model.load_state_dict(state['model'], strict=True)
    return model.eval().requires_grad_(False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--commit', required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError('Unique output required')
    torch.set_num_threads(2)
    ids, rx, scene, _, received, clean = source_inputs(args.root)
    model = load_model(args.root, ids)
    dura = frozen_dura(args.root, 'cpu')
    rows = []
    with torch.no_grad():
        for start in range(0, len(ids), 8):
            selected = ids[start:start+8]
            x = core.tensor(received[selected], 'cpu')
            y = core.tensor(clean[selected], 'cpu')
            context = dura.estimate_context(x)
            stages, _, action, _ = dura.stage_outputs(x, context)
            anchor = stages[-1]
            observed = torch.complex(x[:, 0], x[:, 1])
            a = torch.complex(anchor[:, 0], anchor[:, 1])
            innovation = action.adjoint(observed - action(a))
            innovation = torch.stack((innovation.real, innovation.imag), 1)
            candidate = model(x, anchor, innovation, context[2], validate=False)
            values = metrics(x, anchor, candidate, y, action)
            if not torch.isfinite(values).all():
                raise FloatingPointError('Finite source diagnostic values required')
            if start == 0:
                torch.testing.assert_close(anchor, dura(x), rtol=1e-5, atol=3e-6)
                print('PHYSICAL_LINE_SOURCE_SMOKE_PASS', len(x), flush=True)
            rows.append(values.cpu())
    values = np.from_dlpack(torch.cat(rows).contiguous()).copy()
    groups = {}
    for name, receivers in (('RX4_6', (4, 6)), ('RX8', (8,))):
        for view in core.VIEWS:
            mask = np.isin(rx, receivers) & (scene == view)
            cell = values[mask]
            if len(cell) != 48:
                raise ValueError('48 records per source group required')
            groups[name+'/'+view] = dict(count=len(cell), means=cell.mean(0).tolist(),
                physical_better=int((cell[:, 2] < cell[:, 0]).sum()),
                physical_worse=int((cell[:, 2] > cell[:, 0]).sum()))
    args.out.mkdir(parents=True)
    with (args.out/'values.npz').open('xb') as f:
        np.savez(f, ids=ids, rx=rx, scene=scene, values=values)
    result = dict(status='PHYSICAL_LINE_SOURCE_COMPLETE', commit=args.commit, source=SOURCE,
        source_commit=SOURCE_COMMIT, records=len(ids), groups=groups,
        columns=['dura_nmse','candidate_nmse','physical_line_nmse','clean_oracle_nmse','physical_alpha','oracle_alpha'],
        TX_labels_read=0, F_weights_read=0, F_outputs_read=0, V_reads=0, target_reads=0, model_updates=0,
        caveat='Exploratory previously used source pilot; line search uses estimated action and received only, oracle uses unavailable source clean. Waveform NMSE is not identity accuracy.')
    (args.out/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf8')
    print('PHYSICAL_LINE_SOURCE_COMPLETE', len(ids), flush=True)


if __name__ == '__main__':
    main()
