"""Measure source-only input-noise sensitivity of fixed restoration networks."""
import argparse
import os
from pathlib import Path
import sys

from experiments.phase1_dura_orthogonal.benchmark_flow_n607_sync import wait_for_idle

SEED = 2026110731
DRAWS = 4
EPSILON = .01


def noise_deltas(x, random):
    import torch
    if x.ndim != 3 or x.shape[1:] != (2, 256) or random.shape != (len(x), DRAWS, 2, 256):
        raise ValueError('Matching IQ and four random noise draws required')
    random = random.to(device=x.device, dtype=x.dtype)
    scale = x.square().mean((1, 2), keepdim=True).sqrt().unsqueeze(1)
    norm = random.square().mean((2, 3), keepdim=True).sqrt()
    if not torch.isfinite(random).all() or (norm <= 0).any() or (scale <= 0).any():
        raise ValueError('Finite nonzero input and noise required')
    return EPSILON * scale * random / norm


def evaluate_batch(dura, models, x, clean, random):
    import torch
    from experiments.phase1_dura_orthogonal.train_source import extract_context

    with torch.no_grad():
        anchor, innovation, variance = extract_context(dura, x)
        originals = [anchor] + [model(x, anchor, innovation, variance) for model in models]
        clean_energy = clean.square().mean((1, 2)).clamp_min(1e-8)
        base_error = torch.stack([(out-clean).square().mean((1, 2))/clean_energy for out in originals], 1)
        results = []
        for delta in noise_deltas(x, random).unbind(1):
            changed = x + delta
            a, innovation, variance = extract_context(dura, changed)
            outputs = [a] + [model(changed, a, innovation, variance) for model in models]
            noise_energy = delta.square().mean((1, 2)).clamp_min(1e-20)
            gains = torch.stack([(out-base).square().mean((1, 2))/noise_energy
                                 for out, base in zip(outputs, originals)], 1)
            errors = torch.stack([(out-clean).square().mean((1, 2))/clean_energy for out in outputs], 1)
            results.append(torch.stack((gains, errors-base_error), -1))
        result = torch.stack(results, 1)
        if not torch.isfinite(result).all():
            raise FloatingPointError('Finite source noise responses required')
        return result


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
    from experiments.phase1_dura_orthogonal.frozen_response_compare import fixed_models
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
    generator = torch.Generator(device='cpu').manual_seed(SEED)
    random = torch.randn((len(ids), DRAWS, 2, 256), generator=generator)
    x = core.tensor(received[ids[:8]], 'cuda:0')
    y = core.tensor(clean[ids[:8]], 'cuda:0')
    with torch.no_grad():
        torch.testing.assert_close(extract_context(dura, x)[0], dura(x), rtol=1e-5, atol=3e-6)
        smoke = evaluate_batch(dura, models, x, y, random[:8])
    if smoke.shape != (8, DRAWS, 3, 2):
        raise ValueError('Real-checkpoint source smoke has incorrect shape')
    print('FROZEN_NOISE_SOURCE_SMOKE_PASS', len(x), flush=True)
    values = []
    for start in range(0, len(ids), 16):
        selected = ids[start:start+16]
        x, y = core.tensor(received[selected], 'cuda:0'), core.tensor(clean[selected], 'cuda:0')
        values.append(evaluate_batch(dura, models, x, y, random[start:start+16]).cpu())
    values = torch.cat(values)
    groups = {}
    for name, receivers in (('RX4_6', (4, 6)), ('RX8', (8,))):
        for view in core.VIEWS:
            mask = torch.as_tensor((np.isin(rx, receivers) & (scene == view)).tolist(), dtype=torch.bool)
            if int(mask.sum()) != 48:
                raise ValueError('48 records per source receiver/view group required')
            groups[name+'/'+view] = dict(count=48, mean_metrics=values[mask].mean((0, 1)).tolist())
    args.out.mkdir(parents=True)
    core.write(args.out/'result.json', dict(status='FROZEN_NOISE_SOURCE_COMPLETE', commit=args.commit,
        records=len(ids), draws=DRAWS, seed=SEED, epsilon_relative_received_rms=EPSILON,
        methods=['DURA', 'wide_source_guard', 'counterfactual_guard'],
        metric_order=['output_difference_energy_per_added_input_noise_energy', 'added_noise_paired_change_in_clean_nmse'],
        groups=groups, TX_labels_read=0, F_weights_read=0, F_outputs_read=0, V_reads=0,
        target_reads=0, model_updates=0, physical_gpu=args.gpu, physical_gpu_uuid=uuid,
        caveat='Added normalized independent Gaussian noise probes local response, not actual channel noise or target accuracy. Models were selected for explanation after ground results.'))
    print('FROZEN_NOISE_SOURCE_COMPLETE', len(ids), flush=True)


if __name__ == '__main__':
    main()
