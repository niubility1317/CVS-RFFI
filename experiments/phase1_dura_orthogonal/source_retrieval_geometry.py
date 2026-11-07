"""Source-only, TX-blind physical-record retrieval under three fixed IQ descriptors."""
import argparse
import json
import os
from pathlib import Path

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import torch
import experiments

from experiments.phase1_dura_orthogonal.benchmark_masked_training_full import load_inputs
from experiments.phase1_dura_orthogonal.train_source import extract_context


def descriptors(iq):
    """Normalized 32-band power and 16 amplitude quantiles; no learned state."""
    if iq.ndim != 3 or iq.shape[1:] != (2, 256):
        raise ValueError('IQ must have shape (N, 2, 256)')
    z = torch.complex(iq[:, 0], iq[:, 1])
    amplitude = z.abs()
    normalized = amplitude / amplitude.square().mean(1, keepdim=True).clamp_min(1e-12).sqrt()
    positions = torch.linspace(8, 247, 16, device=iq.device).long()
    quantiles = normalized.sort(dim=1).values[:, positions]
    spectrum = torch.fft.fftshift(torch.fft.fft(z, dim=1).abs().square(), dim=1)
    bands = spectrum.reshape(len(iq), 32, 8).mean(2)
    bands = torch.log((bands / bands.mean(1, keepdim=True).clamp_min(1e-12)).clamp_min(1e-8))
    return bands, quantiles


def retrieval(query, reference, mean, scale):
    """Retrieve matching row; ties receive the conservative worst rank."""
    if query.shape != reference.shape or query.ndim != 2 or len(query) < 5:
        raise ValueError('At least five paired record feature rows required')
    q, r = (query-mean)/scale, (reference-mean)/scale
    distance = torch.cdist(q, r)
    true_distance = distance.diagonal()
    ranks = (distance <= true_distance[:, None]).sum(1)
    return dict(n=len(query), top1=float((ranks == 1).float().mean()),
                top5=float((ranks <= 5).float().mean()),
                mean_reciprocal_rank=float((1/ranks.float()).mean()),
                mean_paired_squared_distance=float(true_distance.square().mean()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--teacher', type=Path, required=True)
    parser.add_argument('--initial-state', type=Path, required=True)
    parser.add_argument('--initial-commit', required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or not torch.cuda.is_available():
        raise ValueError('Unique output and CUDA GPU required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    experiments.__path__ = [str(args.root/'releases/pamf_stage2_inference_20261021_r01/experiments'), *experiments.__path__]
    from experiments.pamf_phase1_flow.anchor_source_smoke import frozen_dura
    data, prior, initial = load_inputs(args.cache, args.teacher, args.initial_state, args.initial_commit)
    del prior, initial
    dura = frozen_dura(args.root, 'cuda:0')
    with torch.no_grad():
        actual = extract_context(dura, data[0][:8])
        if not all(torch.isfinite(value).all() for value in actual) or any(p.requires_grad for p in dura.parameters()):
            raise ValueError('Frozen real DURA source smoke required')
        for offset in (32768, 65536):
            if not torch.equal(data[4][:1024], data[4][offset:offset+1024]):
                raise ValueError('Three views must refer to the same clean physical records')
        print('RETRIEVAL_REAL_DURA_SMOKE_PASS', flush=True)
        train = descriptors(data[4][:512])
        reference = descriptors(data[4][512:1024])
        feature_names = ('spectrum', 'amplitude', 'combined')
        fit = (*train, torch.cat(train, dim=1))
        ref = (*reference, torch.cat(reference, dim=1))
        stats = [(v.mean(0), v.std(0, unbiased=False).clamp_min(.001)) for v in fit]
        results = {}
        for view, offset in enumerate((0, 32768, 65536), start=1):
            results[str(view)] = {}
            for name, field in (('received', 0), ('dura', 1)):
                features = descriptors(data[field][offset+512:offset+1024])
                features = (*features, torch.cat(features, dim=1))
                results[str(view)][name] = {key: retrieval(q, r, *stat)
                    for key, q, r, stat in zip(feature_names, features, ref, stats)}
        if not all(torch.isfinite(v).all() for v in (*train, *reference)):
            raise ValueError('Finite source-only descriptors required')
    args.out.mkdir(parents=True)
    result = dict(status='SOURCE_RETRIEVAL_COMPLETE', commit=args.commit,
        normalization_source_ids=512, retrieval_source_ids=512, source_views=3,
        retrieval_chance_top1=1/512, retrieval_chance_top5=5/512,
        feature_definitions='32 normalized log-power bins, 16 normalized amplitude quantiles, concatenation',
        TX_labels_read=0, F_weights_read=0, F_outputs_read=0, V_reads=0, target_reads=0,
        weights_saved=False, results=results)
    (args.out/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf8')
    print('SOURCE_RETRIEVAL_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
