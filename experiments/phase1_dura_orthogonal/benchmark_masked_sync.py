"""Compare checked and generated-mask training on the same source IQ and GPU."""
import argparse
import json
import os
from pathlib import Path

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import torch

from experiments.phase1_dura_orthogonal.benchmark_masked_training_full import load_inputs, train
from experiments.phase1_dura_orthogonal.masked_prior_constraint import masked_prior_loss
from experiments.phase1_dura_orthogonal.multiscale_spectral import DuraMultiscaleSpectral


def fast(prior, output, clean, starts):
    return masked_prior_loss(prior, output, clean, starts, starts_generated_in_range=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--teacher', type=Path, required=True)
    parser.add_argument('--initial-state', type=Path, required=True)
    parser.add_argument('--initial-commit', required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or not torch.cuda.is_available():
        raise ValueError('A unique output and GPU are required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    data, prior, initial = load_inputs(args.cache, args.teacher, args.initial_state, args.initial_commit)
    data = tuple(torch.cat([v[i:i+512] for i in (0, 32768, 65536)]) for v in data)
    smoke = DuraMultiscaleSpectral(width=32, taps=9).to('cuda:0').eval()
    smoke.load_state_dict(initial, strict=True)
    with torch.no_grad():
        if not torch.equal(smoke(*(v[:8] for v in data[:4])), data[1][:8]):
            raise ValueError('Real source checkpoint smoke did not reproduce DURA anchor')
    del smoke
    print('MASK_SYNC_REAL_SOURCE_SMOKE_PASS', flush=True)
    # Alternate order to expose order/load sensitivity rather than assuming a speedup.
    results = []
    for iteration, order in enumerate(((masked_prior_loss, fast), (fast, masked_prior_loss),
                                       (masked_prior_loss, fast), (fast, masked_prior_loss))):
        models = {}
        for fn in order:
            name = 'fast' if fn is fast else 'checked'
            model, losses, epochs, seconds = train(f'{iteration}-{name}', fn, prior, data, initial, 2)
            models[name] = (model, losses, epochs, seconds)
        left, right = models['checked'], models['fast']
        if not torch.equal(left[1], right[1]) or left[2] != right[2]:
            raise ValueError('Training losses differ')
        if any(not torch.equal(left[0].state_dict()[k], right[0].state_dict()[k]) for k in left[0].state_dict()):
            raise ValueError('Final model parameters differ')
        results.append({'checked_seconds': left[3], 'fast_seconds': right[3],
                        'steps': len(left[1]), 'losses_exact': True, 'parameters_exact': True})
        del models, left, right
    args.out.mkdir(parents=True)
    (args.out/'result.json').write_text(json.dumps(dict(status='MASK_SYNC_BENCHMARK_COMPLETE',
        commit=args.commit, source_ids=512, source_views=1536, epochs=2, trials=results,
        gpu=torch.cuda.get_device_name(), TX_labels_read=0, F_weights_read=0,
        F_outputs_read=0, V_reads=0, target_reads=0), indent=2)+'\n', encoding='utf8')
    print('MASK_SYNC_BENCHMARK_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
