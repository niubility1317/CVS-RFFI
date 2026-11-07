"""Source-only physical-record retrieval with fourth-order frequency closure."""
import argparse
import json
from pathlib import Path
import time

import torch

from experiments.phase1_dura_innovation_probe.source_probe import source_inputs
from experiments.phase1_dura_orthogonal.source_context_cache import load_cache
from experiments.phase1_dura_orthogonal.source_retrieval_geometry import descriptors, retrieval
from experiments.phase1_dura_orthogonal.train_source import WIDE_FIT_SIZE, extract_context, load_source_training

OFFSETS = (1, 2, 4, 8)


def closure_features(iq):
    """Return bounded fourth-order frequency products, not cumulant estimates."""
    if iq.ndim != 3 or iq.shape[1:] != (2, 256) or not iq.is_floating_point():
        raise ValueError('Floating-point IQ must have shape (N, 2, 256)')
    z = torch.complex(iq[:, 0], iq[:, 1])
    z = z/z.abs().square().mean(1, keepdim=True).clamp_min(1e-12).sqrt()
    spectrum = torch.fft.fft(z, dim=1, norm='ortho')
    closure = torch.stack([spectrum.roll(d, 1)*spectrum.roll(-d, 1)*spectrum.conj().square()
                           for d in OFFSETS], dim=1).flatten(1)
    closure = closure/(1+closure.abs())
    return torch.cat((closure.real, closure.imag), dim=1)


def standardize(features, mean, scale):
    return (features-mean)/scale/(features.shape[1]**.5)


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
        print('SOURCE_QUARTIC_REAL_DURA_SMOKE_PASS', flush=True)
        started = time.monotonic()
        train_clean, reference_clean = data[4][:512], data[4][512:1024]
        train = (descriptors(train_clean)[0], closure_features(train_clean))
        ref = (descriptors(reference_clean)[0], closure_features(reference_clean))
        stats = [(v.mean(0), v.std(0, unbiased=False).clamp_min(.001)) for v in train]
        refs = [standardize(v, *stat) for v, stat in zip(ref, stats)]
        results = {}
        for view, offset in enumerate((0, WIDE_FIT_SIZE, 2*WIDE_FIT_SIZE), start=1):
            results[str(view)] = {}
            for name, field in (('received', 0), ('dura', 1)):
                x = data[field][offset+512:offset+1024]
                features = (descriptors(x)[0], closure_features(x))
                queries = [standardize(v, *stat) for v, stat in zip(features, stats)]
                queries.append(torch.cat(queries, dim=1))
                reference = [*refs, torch.cat(refs, dim=1)]
                if not all(torch.isfinite(v).all() for v in (*queries, *reference)):
                    raise FloatingPointError('Finite source features required')
                results[str(view)][name] = {
                    key: retrieval(q, r, torch.zeros(q.shape[1]), torch.ones(q.shape[1]))
                    for key, q, r in zip(('power', 'closure', 'power_plus_closure'), queries, reference)}
    args.out.mkdir(parents=True)
    result = dict(status='SOURCE_QUARTIC_CLOSURE_COMPLETE', commit=args.commit, normalization_source_ids=512,
        retrieval_source_ids=512, source_views=3, offsets=list(OFFSETS), records_per_view=512,
        retrieval_chance_top1=1/512, device='cpu', elapsed_seconds=time.monotonic()-started,
        feature_definition='bounded FFT[f-d]*FFT[f+d]*conj(FFT[f])^2; separate equal-weight standardized power and closure blocks',
        TX_labels_read=0, F_weights_read=0, F_outputs_read=0, V_reads=0, target_reads=0, model_updates=0,
        caveat='Record retrieval can reflect signal content; this is not a cumulant estimate, TX identity accuracy, target performance or a deployable reference library.',
        results=results)
    (args.out/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf8')
    print('SOURCE_QUARTIC_CLOSURE_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
