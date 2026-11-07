"""Source-IQ response comparison of three previously frozen restorers."""
import argparse
import os
from pathlib import Path
import sys
import time

from experiments.phase1_dura_orthogonal.benchmark_flow_n607_sync import wait_for_idle

WIDE = '20261106-phase1-dura-wide-source-source-s2026092701-r01'
RESPONSE = '20261107-phase1-dura-counterfactual-response-source-s2026092701-r01'
SOURCES = ((WIDE, '2993157edf705d93f015e9a51e4eb2fa0681f3dc', 'wide_source_guard'),
           (RESPONSE, '2eadeeed2f24e22fe69c516549fe5272fca88c22', 'wide_counterfactual_guard'))


def fixed_models(root, pilot_ids, device):
    import numpy as np
    import torch
    from experiments.pamf_phase1_flow import experiment as core
    from experiments.phase1_dura_orthogonal.multiscale_spectral import DuraMultiscaleSpectral
    from experiments.phase1_dura_orthogonal.train_source import SCHEMA, WIDE_FIT_SIZE

    models, fit_ids = [], []
    for source, commit, objective in SOURCES:
        folder = root/'runs'/source
        summary = core.read(folder/'source_summary.json')
        state = torch.load(folder/'final.pt', map_location='cpu', weights_only=False)
        expected = dict(commit=commit, architecture='dura_multiscale', objective=objective,
                        epochs=5, source_fit_size=WIDE_FIT_SIZE, DURA_training_reads=1,
                        DURA_frozen=True, TX_labels_read=0, F_weights_read=0,
                        F_outputs_read=0, V_reads=0, target_reads=0)
        if (summary.get('status') != 'SOURCE_TRAINED' or summary.get('fit_ids') != WIDE_FIT_SIZE or
                state.get('schema') != SCHEMA or
                any(obj.get(k) != v for obj in (summary, state) for k, v in expected.items())):
            raise ValueError('Frozen source-only model provenance mismatch')
        if objective == 'wide_counterfactual_guard' and any(
                obj.get('source_response_weight') != .01 for obj in (summary, state)):
            raise ValueError('Fixed response loss weight required')
        fit = np.asarray(state.get('fit_ids', []))
        if fit.shape != (WIDE_FIT_SIZE,) or np.intersect1d(fit, pilot_ids).size:
            raise ValueError('Source pilot must be disjoint from both trained models')
        model = DuraMultiscaleSpectral(width=state['width'], taps=9)
        model.load_state_dict(state['model'], strict=True)
        models.append(model.to(device).eval().requires_grad_(False))
        fit_ids.append(fit)
    if not np.array_equal(*fit_ids):
        raise ValueError('Paired models must use the same ordered source fit IDs')
    return models


def evaluate_batch(dura, models, x, clean):
    import torch
    from experiments.phase1_dura_orthogonal.source_dura_perturbation_probe import perturbations, response_metrics
    from experiments.phase1_dura_orthogonal.train_source import extract_context

    with torch.no_grad():
        context = dura.estimate_context(x)
        anchor, innovation, variance = extract_context(dura, x)
        originals = [anchor] + [model(x, anchor, innovation, variance) for model in models]
        energy = clean.square().mean((1, 2)).clamp_min(1e-8)
        nmse = torch.stack([(value-clean).square().mean((1, 2))/energy for value in originals], 1)
        outcomes = []
        for delta in perturbations(clean):
            observed = context[1](delta)
            changed = x + torch.stack((observed.real, observed.imag), 1)
            a, innovation, variance = extract_context(dura, changed)
            outputs = [a] + [model(changed, a, innovation, variance) for model in models]
            values = []
            for original, output in zip(originals, outputs):
                d = output-original
                values.append(response_metrics(delta, torch.complex(d[:, 0], d[:, 1])))
            outcomes.append(torch.stack(values, 1))
        metrics = torch.stack(outcomes, 1)
        if not torch.isfinite(metrics).all() or not torch.isfinite(nmse).all():
            raise FloatingPointError('Finite source response and NMSE required')
        return metrics, nmse


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--commit', required=True)
    p.add_argument('--gpu', type=int, required=True)
    args = p.parse_args()
    uuid = wait_for_idle(args.out, args.gpu, allow_shared=True)
    if 'torch' in sys.modules:
        raise RuntimeError('GPU must be selected before importing torch')
    os.environ.update(CUDA_VISIBLE_DEVICES=uuid, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                      OPENBLAS_NUM_THREADS='2', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    import numpy as np
    import torch
    from experiments.pamf_phase1_flow import experiment as core
    from experiments.pamf_phase1_flow.anchor_source_smoke import frozen_dura
    from experiments.phase1_dura_innovation_probe.source_probe import source_inputs
    from experiments.phase1_dura_orthogonal.source_dura_perturbation_probe import DIRECTIONS, EPSILON
    from experiments.phase1_dura_orthogonal.train_source import extract_context

    if not torch.cuda.is_available():
        raise ValueError('Registered GPU required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    ids, rx, scene, _, received, clean = source_inputs(args.root)
    models = fixed_models(args.root, ids, 'cuda:0')
    dura = frozen_dura(args.root, 'cuda:0')
    x = core.tensor(received[ids[:8]], 'cuda:0')
    y = core.tensor(clean[ids[:8]], 'cuda:0')
    with torch.no_grad():
        torch.testing.assert_close(extract_context(dura, x)[0], dura(x), rtol=1e-5, atol=3e-6)
        smoke, smoke_nmse = evaluate_batch(dura, models, x, y)
    if smoke.shape != (8, 4, 3, 3) or smoke_nmse.shape != (8, 3):
        raise ValueError('Real-checkpoint source smoke has incorrect shape')
    print('FROZEN_RESPONSE_SOURCE_SMOKE_PASS', len(x), flush=True)
    values, errors = [], []
    for start in range(0, len(ids), 16):
        selected = ids[start:start+16]
        x, y = core.tensor(received[selected], 'cuda:0'), core.tensor(clean[selected], 'cuda:0')
        metric, nmse = evaluate_batch(dura, models, x, y)
        values.append(metric.cpu())
        errors.append(nmse.cpu())
    values = torch.cat(values).numpy()
    errors = torch.cat(errors).numpy()
    groups = {}
    for name, receivers in (('RX4_6', (4, 6)), ('RX8', (8,))):
        for view in core.VIEWS:
            mask = np.isin(rx, receivers) & (scene == view)
            if mask.sum() != 48:
                raise ValueError('48 records per source receiver/view group required')
            groups[name+'/'+view] = dict(count=int(mask.sum()), mean_metrics=values[mask].mean(0).tolist(),
                                          mean_nmse=errors[mask].mean(0).tolist())
    args.out.mkdir(parents=True)
    core.write(args.out/'result.json', dict(status='FROZEN_RESPONSE_SOURCE_COMPLETE', commit=args.commit,
        records=len(ids), directions=list(DIRECTIONS), epsilon_relative_rms=EPSILON,
        methods=['DURA', 'wide_source_guard', 'counterfactual_guard'],
        metric_order=['real_projection_gain', 'relative_squared_response_error', 'response_norm_ratio'],
        groups=groups, TX_labels_read=0, F_weights_read=0, F_outputs_read=0, V_reads=0,
        target_reads=0, model_updates=0, physical_gpu=args.gpu, physical_gpu_uuid=uuid,
        caveat='Synthetic perturbations and estimated operator are source proxies; models were selected for explanation after ground results, so this is not independent validation or a basis for target-tuned selection.'))
    print('FROZEN_RESPONSE_SOURCE_COMPLETE', len(ids), flush=True)


if __name__ == '__main__':
    main()
