"""Source-only clean-referenced global phase potential, not deployable."""
import argparse
import math
import os
from pathlib import Path
import sys

from experiments.phase1_dura_orthogonal.benchmark_flow_n607_sync import wait_for_idle
from experiments.phase1_dura_orthogonal.conditional_phase_source_probe import SOURCE, SOURCE_COMMIT


def rotate(signal, angle):
    import torch
    c, s = angle.cos()[:, None], angle.sin()[:, None]
    real, imag = signal[:, 0], signal[:, 1]
    return torch.stack((c*real-s*imag, s*real+c*imag), dim=1)


def analyze_batch(model, received, anchor, innovation, variance, clean):
    import torch
    from experiments.phase1_dura_orthogonal.noise_gated_dura import noise_gate
    with torch.no_grad():
        ar, ai, cr, ci = anchor[:, 0], anchor[:, 1], clean[:, 0], clean[:, 1]
        inner_real = (cr*ar+ci*ai).sum(-1)
        inner_imag = (ci*ar-cr*ai).sum(-1)
        oracle = torch.atan2(inner_imag, inner_real).clamp(-math.pi/2, math.pi/2)
        power = anchor.square().mean((1, 2)).clamp_min(1e-12)
        params = model.summary(model.condition_features(received, anchor, innovation, variance, power))
        phase = (math.pi/2)*params[:, 0].tanh()
        gain = 1+.1*params[:, 1].tanh()
        mix = params[:, 2].sigmoid()
        weight = (1-mix)*noise_gate(anchor, variance)[:, 0, 0]
        effective = torch.atan2(weight*gain*phase.sin(), 1+weight*(gain*phase.cos()-1))
        angular_error = torch.atan2((oracle-effective).sin(), (oracle-effective).cos()).abs()
        energy = clean.square().mean((1, 2)).clamp_min(1e-8)
        phase_only = anchor+weight[:, None, None]*(gain[:, None, None]*rotate(anchor, phase)-anchor)
        values = torch.stack((oracle.abs(), effective.abs(), angular_error,
            (anchor-clean).square().mean((1, 2))/energy,
            (rotate(anchor, oracle)-clean).square().mean((1, 2))/energy,
            (phase_only-clean).square().mean((1, 2))/energy,
            (model(received, anchor, innovation, variance)-clean).square().mean((1, 2))/energy), 1)
        if not torch.isfinite(values).all() or (values[:, 4] > values[:, 3]+1e-5).any():
            raise ValueError('Finite bounded oracle must not worsen IQ NMSE')
        return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    uuid = wait_for_idle(args.out, args.gpu, allow_shared=True)
    if 'torch' in sys.modules:
        raise RuntimeError('Select GPU before importing torch')
    os.environ.update(CUDA_VISIBLE_DEVICES=uuid, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                      OPENBLAS_NUM_THREADS='2', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    import numpy as np
    import torch
    from experiments.pamf_phase1_flow import experiment as core
    from experiments.pamf_phase1_flow.anchor_source_smoke import frozen_dura
    from experiments.phase1_dura_innovation_probe.source_probe import source_inputs
    from experiments.phase1_dura_orthogonal.conditional_phase_dura import ConditionalPhaseDura
    from experiments.phase1_dura_orthogonal.train_source import extract_context, SCHEMA

    if not torch.cuda.is_available():
        raise ValueError('Registered GPU required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    folder = args.root/'runs'/SOURCE
    summary = core.read(folder/'source_summary.json')
    state = torch.load(folder/'final.pt', map_location='cpu', weights_only=False)
    expected = dict(commit=SOURCE_COMMIT, architecture='conditional_phase_dura', objective='wide_conditional_phase',
                    epochs=5, DURA_frozen=True, DURA_training_reads=1, TX_labels_read=0, F_weights_read=0,
                    F_outputs_read=0, V_reads=0, target_reads=0)
    if (summary.get('status') != 'SOURCE_TRAINED' or summary.get('fit_ids') != 32768
            or state.get('schema') != SCHEMA or len(state.get('fit_ids', [])) != 32768
            or any(obj.get(k) != v for obj in (summary, state) for k, v in expected.items())):
        raise ValueError('Matching fixed final source-only candidate required')
    ids, rx, scene, _, received, clean = source_inputs(args.root)
    if np.intersect1d(np.asarray(state['fit_ids']), ids).size:
        raise ValueError('Source diagnostic IDs must be excluded from candidate fit')
    model = ConditionalPhaseDura(width=state['width']).cuda().eval().requires_grad_(False)
    model.load_state_dict(state['model'], strict=True)
    dura = frozen_dura(args.root, 'cuda:0')
    def evaluate(chosen):
        x, y = core.tensor(received[chosen], 'cuda:0'), core.tensor(clean[chosen], 'cuda:0')
        anchor, innovation, variance = extract_context(dura, x)
        return analyze_batch(model, x, anchor, innovation, variance, y)
    smoke = evaluate(ids[:8])
    if smoke.shape != (8, 7) or any(p.requires_grad for net in (model, dura) for p in net.parameters()):
        raise ValueError('Frozen real checkpoint source smoke required')
    print('CONDITIONAL_PHASE_ORACLE_SMOKE_PASS', len(smoke), flush=True)
    values = torch.cat([evaluate(ids[i:i+16]).cpu() for i in range(0, len(ids), 16)])
    groups = {}
    for name, receivers in (('RX4_6', (4, 6)), ('RX8', (8,))):
        for view in core.VIEWS:
            mask = torch.as_tensor((np.isin(rx, receivers) & (scene == view)).tolist(), dtype=torch.bool)
            cell = values[mask]
            if len(cell) != 48:
                raise ValueError('48 source records per receiver/view group required')
            groups[name+'/'+view] = dict(count=len(cell), means=cell.mean(0).tolist(),
                learned_global_branch_beats_anchor=int((cell[:, 5] < cell[:, 3]).sum()),
                oracle_phase_beats_anchor=int((cell[:, 4] < cell[:, 3]).sum()))
    args.out.mkdir(parents=True)
    core.write(args.out/'result.json', dict(status='CONDITIONAL_PHASE_ORACLE_COMPLETE', commit=args.commit,
        source=SOURCE, source_commit=SOURCE_COMMIT, records=len(ids), groups=groups,
        metric_order=['oracle_abs_phase','learned_effective_abs_phase','angle_abs_error',
                      'anchor_nmse','bounded_oracle_phase_nmse','learned_global_branch_nmse','candidate_nmse'],
        TX_labels_read=0, F_weights_read=0, F_outputs_read=0, V_reads=0, target_reads=0, model_updates=0,
        physical_gpu=args.gpu, physical_gpu_uuid=uuid,
        caveat='Oracle uses source clean and cannot deploy; phase-only result is not a general recovery or identity upper bound.'))
    print('CONDITIONAL_PHASE_ORACLE_COMPLETE', len(ids), flush=True)


if __name__ == '__main__':
    main()
