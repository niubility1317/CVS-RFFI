"""Frozen source-IQ diagnosis of mixture expert use; no target or TX labels."""
import argparse
import time
from pathlib import Path

import numpy as np
import torch

from experiments.pamf_phase1_flow import experiment as core
from experiments.pamf_phase1_flow.anchor_source_smoke import frozen_dura
from experiments.phase1_dura_innovation_probe.source_probe import source_inputs
from experiments.phase1_dura_orthogonal.dura_experts import DuraExperts, EXPERT_COUNT
from experiments.phase1_dura_orthogonal.train_source import SCHEMA, WIDE_FIT_SIZE, extract_context


def fixed_model(root, source, commit, pilot_ids, device):
    folder = root/'runs'/source
    summary = core.read(folder/'source_summary.json')
    state = torch.load(folder/'final.pt', map_location='cpu', weights_only=False)
    expected = dict(commit=commit, architecture='dura_experts', objective='wide_mixture_guard',
                    epochs=5, source_fit_size=WIDE_FIT_SIZE, source_expert_count=EXPERT_COUNT,
                    DURA_training_reads=1, DURA_frozen=True, TX_labels_read=0, F_weights_read=0,
                    F_outputs_read=0, V_reads=0, target_reads=0)
    if (summary.get('status') != 'SOURCE_TRAINED' or state.get('schema') != SCHEMA or
            any(obj.get(k) != v for obj in (summary, state) for k, v in expected.items())):
        raise ValueError('Fixed source-only expert checkpoint required')
    fit = np.asarray(state.get('fit_ids', []))
    if len(fit) != WIDE_FIT_SIZE or np.intersect1d(fit, pilot_ids).size:
        raise ValueError('Pilot IDs must be disjoint from source fit')
    model = DuraExperts(width=state['width'], taps=9)
    model.load_state_dict(state['model'], strict=True)
    return model.to(device).eval().requires_grad_(False)


@torch.no_grad()
def evaluate_batch(model, dura, x, clean):
    captured = {}
    def save(key):
        def hook(module, inputs, output):
            captured[key] = output.detach()
        return hook
    handles = [model.gate.register_forward_hook(save('gate'))]
    handles.extend(expert.register_forward_hook(save(str(i))) for i, expert in enumerate(model.experts))
    try:
        anchor, direction, variance = extract_context(dura, x)
        output = model(x, anchor, direction, variance)
    finally:
        for handle in handles:
            handle.remove()
    if set(captured) != {'gate', '0', '1', '2'}:
        raise ValueError('Three expert outputs and one gate required')
    weights = captured['gate'].softmax(1)
    components = torch.stack([captured[str(i)] for i in range(EXPERT_COUNT)], 1)
    energy = clean.square().mean((1, 2)).clamp_min(1e-8)
    anchor_err = (anchor-clean).square().mean((1, 2))/energy
    mixture_err = (output-clean).square().mean((1, 2))/energy
    expert_err = (components-clean[:, None]).square().mean((2, 3))/energy[:, None]
    stacked = torch.cat((weights, anchor_err[:, None], mixture_err[:, None], expert_err), 1)
    if not torch.isfinite(stacked).all() or not torch.allclose(weights.sum(1), torch.ones_like(energy)):
        raise FloatingPointError('Finite source-only diagnostic required')
    return np.from_dlpack(stacked.cpu().contiguous()).copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--candidate-commit', required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Unique source diagnostic output required')
    if args.device != 'cuda:0' or not torch.cuda.is_available():
        raise ValueError('Registered CUDA device required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ids, rx, scene, _, received, clean = source_inputs(args.root)
    model = fixed_model(args.root, args.source, args.candidate_commit, ids, args.device)
    dura = frozen_dura(args.root, args.device)
    def batch(start, end):
        x = core.tensor(received[ids[start:end]], args.device)
        y = core.tensor(clean[ids[start:end]], args.device)
        return evaluate_batch(model, dura, x, y)
    smoke = batch(0, 8)
    if smoke.shape != (8, 8):
        raise ValueError('Real-checkpoint source smoke required')
    print('EXPERT_SOURCE_SMOKE_PASS', len(smoke), flush=True)
    started = time.monotonic()
    values = np.concatenate([batch(start, min(start+32, len(ids))) for start in range(0, len(ids), 32)])
    if values.shape != (len(ids), 8):
        raise ValueError('All source pilot records required')
    groups = {}
    for name, receivers in (('RX4_6', (4, 6)), ('RX8', (8,))):
        for view in core.VIEWS:
            mask = np.isin(rx, receivers) & (scene == view)
            subset = values[mask]
            if len(subset) != 48:
                raise ValueError('48 source records per receiver/view group required')
            weights, errs = subset[:, :3], subset[:, 3:]
            groups[name+'/'+view] = dict(count=len(subset), mean_gate=weights.mean(0).tolist(),
                gate_entropy=float((-(weights*np.log(np.maximum(weights, 1e-12))).sum(1)).mean()),
                anchor_nmse=float(errs[:, 0].mean()), mixture_nmse=float(errs[:, 1].mean()),
                expert_nmse=errs[:, 2:].mean(0).tolist(),
                source_clean_oracle_expert_nmse=float(errs[:, 2:].min(1).mean()))
    args.out.mkdir(parents=True)
    core.write(args.out/'result.json', dict(status='SOURCE_DIAGNOSTIC_COMPLETE', commit=args.commit,
        candidate=args.source, candidate_commit=args.candidate_commit, count=len(ids), groups=groups,
        elapsed_seconds=time.monotonic()-started, F_weights_read=0, F_outputs_read=0,
        TX_labels_read=0, V_reads=0, target_reads=0,
        oracle_scope='per-record best expert uses source clean IQ; unavailable at deployment'))
    print('EXPERT_SOURCE_DIAGNOSTIC_COMPLETE', len(ids), flush=True)


if __name__ == '__main__':
    main()
