"""Source-only inspection of a fixed conditional phase DURA correction."""
import argparse
import math
import os
from pathlib import Path
import sys

from experiments.phase1_dura_orthogonal.benchmark_flow_n607_sync import wait_for_idle

SOURCE = '20261107-phase1-dura-conditional-phase-source-s2026092701-r01'
SOURCE_COMMIT = 'afac42e77542d0f035996d1a2e8dde7decbd4ac5'


def analyze_batch(model, received, anchor, innovation, variance, clean):
    import torch
    from experiments.phase1_dura_orthogonal.noise_gated_dura import noise_gate
    with torch.no_grad():
        power = anchor.square().mean((1, 2)).clamp_min(1e-12)
        params = model.summary(model.condition_features(received, anchor, innovation, variance, power))
        phase = (math.pi/2)*params[:, 0:1].tanh()
        magnitude = 1+.1*params[:, 1:2].tanh()
        mix = params[:, 2:3].sigmoid()[:, :, None]
        real, imag = anchor[:, 0], anchor[:, 1]
        rotated = magnitude[:, None]*torch.stack((phase.cos()*real-phase.sin()*imag,
                                                    phase.sin()*real+phase.cos()*imag), 1)
        phase_part = (1-mix)*noise_gate(anchor, variance)*(rotated-anchor)
        local_part = mix*(model.local(received, anchor, innovation, variance)-anchor)
        output = anchor+phase_part+local_part
        if not torch.allclose(output, model(received, anchor, innovation, variance), rtol=1e-6, atol=1e-7):
            raise ValueError('Branch decomposition must reproduce fixed model output')
        energy = clean.square().mean((1, 2)).clamp_min(1e-8)
        vals = torch.stack((phase[:, 0].abs(), mix[:, 0, 0],
            phase_part.square().mean((1, 2))/energy, local_part.square().mean((1, 2))/energy,
            (phase_part*local_part).mean((1, 2))/energy,
            (output-clean).square().mean((1, 2))/energy,
            (anchor-clean).square().mean((1, 2))/energy), 1)
        if not torch.isfinite(vals).all():
            raise FloatingPointError('Finite source branch diagnostics required')
        return vals


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
    print('CONDITIONAL_PHASE_SOURCE_BRANCH_SMOKE_PASS', len(smoke), flush=True)
    values = torch.cat([evaluate(ids[i:i+16]).cpu() for i in range(0, len(ids), 16)])
    groups = {}
    for name, receivers in (('RX4_6', (4, 6)), ('RX8', (8,))):
        for view in core.VIEWS:
            mask = torch.as_tensor((np.isin(rx, receivers) & (scene == view)).tolist(), dtype=torch.bool)
            cell = values[mask]
            if len(cell) != 48:
                raise ValueError('48 source records per receiver/view group required')
            groups[name+'/'+view] = dict(count=len(cell), means=cell.mean(0).tolist())
    args.out.mkdir(parents=True)
    core.write(args.out/'result.json', dict(status='CONDITIONAL_PHASE_SOURCE_BRANCH_COMPLETE', commit=args.commit,
        source=SOURCE, source_commit=SOURCE_COMMIT, records=len(ids), groups=groups,
        metric_order=['abs_phase_radians','local_mix','phase_part_nmse','local_part_nmse',
                      'branch_cross_inner','candidate_nmse','dura_nmse'],
        TX_labels_read=0, F_weights_read=0, F_outputs_read=0, V_reads=0, target_reads=0, model_updates=0,
        physical_gpu=args.gpu, physical_gpu_uuid=uuid,
        caveat='Explored source IQ; clean is a diagnostic reference only, not deployed or identity accuracy.'))
    print('CONDITIONAL_PHASE_SOURCE_BRANCH_COMPLETE', len(ids), flush=True)


if __name__ == '__main__':
    main()
