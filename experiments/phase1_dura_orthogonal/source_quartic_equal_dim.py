"""Source-only comparison of equally sized spectral-power and closure features."""
import argparse
import json
from pathlib import Path
import time

import torch

from experiments.phase1_dura_innovation_probe.source_probe import source_inputs
from experiments.phase1_dura_orthogonal.source_context_cache import load_cache
from experiments.phase1_dura_orthogonal.source_quartic_closure import closure_features, standardize
from experiments.phase1_dura_orthogonal.source_retrieval_geometry import retrieval
from experiments.phase1_dura_orthogonal.train_source import WIDE_FIT_SIZE, extract_context, load_source_training


def features(iq):
    """256 log-power bins and 256 pooled bounded quartic coupling values."""
    if iq.ndim != 3 or iq.shape[1:] != (2, 256):
        raise ValueError('IQ must have shape (N, 2, 256)')
    z = torch.complex(iq[:, 0], iq[:, 1])
    power = torch.fft.fft(z, dim=1, norm='ortho').abs().square()
    power = torch.log((power/power.mean(1, keepdim=True).clamp_min(1e-12)).clamp_min(1e-8))
    coupling = closure_features(iq).reshape(len(iq), 2, 4, 32, 8).mean(-1).flatten(1)
    return power, coupling


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Unique source diagnostic output required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    ids, _, _, _, _, _ = source_inputs(args.root)
    _, _, _, fit = load_source_training(args.root, ids, fit_size=WIDE_FIT_SIZE)
    data, _ = load_cache(args.cache, fit, 32, 'cuda:0', deterministic=True)
    from experiments.pamf_phase1_flow.anchor_source_smoke import frozen_dura
    dura = frozen_dura(args.root, 'cpu')
    with torch.no_grad():
        context = extract_context(dura, data[0][:8])
        if not all(torch.isfinite(v).all() for v in context) or any(p.requires_grad for p in dura.parameters()):
            raise ValueError('Real frozen DURA source smoke required')
        for offset in (WIDE_FIT_SIZE, 2*WIDE_FIT_SIZE):
            if not torch.equal(data[4][:1024], data[4][offset:offset+1024]):
                raise ValueError('Matching clean records across source views required')
        print('EQUAL_DIM_REAL_DURA_SMOKE_PASS', flush=True)
        started = time.monotonic()
        train, ref = features(data[4][:512]), features(data[4][512:1024])
        stats = [(v.mean(0), v.std(0, unbiased=False).clamp_min(.001)) for v in train]
        refs = [standardize(v, *stat) for v, stat in zip(ref, stats)]
        results = {}
        for view, offset in enumerate((0, WIDE_FIT_SIZE, 2*WIDE_FIT_SIZE), start=1):
            results[str(view)] = {}
            for name, field in (('received', 0), ('dura', 1)):
                current = features(data[field][offset+512:offset+1024])
                queries = [standardize(v, *stat) for v, stat in zip(current, stats)]
                if not all(torch.isfinite(v).all() for v in (*queries, *refs)):
                    raise FloatingPointError('Finite source features required')
                results[str(view)][name] = {key: retrieval(q, r, torch.zeros(256), torch.ones(256))
                    for key, q, r in zip(('power', 'closure'), queries, refs)}
    args.out.mkdir(parents=True)
    result = dict(status='SOURCE_QUARTIC_EQUAL_DIM_COMPLETE', commit=args.commit, source_views=3,
        normalization_source_ids=512, retrieval_source_ids=512, power_dimensions=256, closure_dimensions=256,
        feature_definition='256 normalized log-power bins vs mean-pooled 32x4x2 bounded quartic frequency products',
        retrieval_chance_top1=1/512, device='cpu', elapsed_seconds=time.monotonic()-started,
        TX_labels_read=0, F_weights_read=0, F_outputs_read=0, V_reads=0, target_reads=0, model_updates=0,
        caveat='Equal dimension does not equalize all estimator properties. Same-record retrieval can reflect content; not TX identity accuracy, target performance or deployable reference.',
        results=results)
    (args.out/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf8')
    print('SOURCE_QUARTIC_EQUAL_DIM_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
