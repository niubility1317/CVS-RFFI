"""Real F/G source smoke or frozen prediction; truth only in a separate CPU action."""
import argparse
from pathlib import Path

from experiments.cric_dura_comparison.io import read, write, tensor


def smoke(c):
    from .dura_reference import ready
    if not ready(c):
        raise ValueError('Reference source/raw owner has not finished')
    from experiments.cric_dura_comparison.bootstrap import activate, numerical_policy
    activate(c)
    numerical_policy()
    import numpy as np
    import torch
    from experiments.cric_dura_comparison.frozen_inference import load_F, load_G
    from experiments.pamf_stage2_inference.engine import InferenceEngine
    contract = read(c['source_contract'])
    reader, source = load_F(c['source_output'], contract, 'cuda:0')
    if (source['run_id'], source['row_id'], source['model_seed']) != (
            c['source_run_id'], c['source_row_id'], c['model_seed']):
        raise ValueError('Smoke F source identity differs')
    g = load_G(c['readouts']['DURA-LU'], contract, 'cuda:0')
    call = InferenceEngine(g, reader, reuse=True, graph=True, native=False)
    x = tensor(np.load(Path(c['source_cache']) / 'L_s_x.npy', mmap_mode='r', allow_pickle=False)[:2], 'cuda:0')
    with torch.inference_mode():
        restored, _, _, logits = call(x)
        direct = g(x)
        raw = reader.classify_features(reader.features(x))
        if (not torch.allclose(restored, direct, atol=1e-5, rtol=1e-5)
                or logits.shape != (2, 6) or raw.shape != (2, 6)
                or not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(raw).all())):
            raise ValueError('Real frozen F/G source-only smoke failed')
    write(Path(c['output_root']).parents[1] / 'smoke.json', dict(status='PASS',
          source_run_id=c['source_run_id'], source_row_id=c['source_row_id'], model_seed=c['model_seed'],
          target_reads=0, truth_reads=0, source_samples=2, parameters_G=266951,
          graph_native_IQ_tolerance=1e-5, backend=dict(reuse=True, graph=True, native=False),
          precision='FP32', TF32=False, immediate_continuation=True))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('smoke', 'predict', 'score'))
    p.add_argument('--config', required=True)
    a = p.parse_args()
    c = read(a.config)
    if a.action == 'smoke':
        smoke(c)
    elif a.action == 'predict':
        from .dura_reference import ready
        if not ready(c):
            raise ValueError('Original source/raw predictions not complete')
        from experiments.cric_dura_comparison.evaluation import predict
        predict(c)
    else:
        from experiments.cric_dura_comparison.evaluation import score
        score(c)


if __name__ == '__main__':
    main()
