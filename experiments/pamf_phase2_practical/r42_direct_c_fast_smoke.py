"""Numerical equivalence on source L_s only; target and query never opened."""
import argparse
from pathlib import Path
import numpy as np

from .io import read, require, write
from .milkr_screen import initialize, subset
from .teacher_features import AdaptiveResidualBridge
from .milkr.prior_fusion import head_columns
from .milkr.adaptive_residual import blend
from .milkr.r42_direct_c import fit_direct_c, DirectPair
from .milkr.r42_direct_c_cuda import CudaDirectC
from .r42_suite_worker import _restore, orbit_features, _decisions


def run(rt, seed, output):
    from experiments.cric_dura_comparison.f_data import SourceDataset
    from .fast_transfer import to_numpy
    local = dict(rt, model_seeds=[seed])
    data = SourceDataset(rt['source_cache'], 'L_s', read(rt['source_contract']))
    y = np.asarray(data.metadata['y'])
    chosen = [np.flatnonzero(y == c)[:7] for c in (0, 1)]
    require(all(len(a) == 7 for a in chosen), 'Insufficient legal source examples')
    positions = np.concatenate(chosen)
    # N607 PyTorch 2.1 / NumPy 2 cannot use a NumPy index as a Torch index.
    raw = to_numpy(data.x[positions.tolist()], np.float32)
    reader, engine = initialize(local)
    parent = Path(rt['source_parent_root']) / 'sources' / f's{seed}'
    bridge = AdaptiveResidualBridge.load(parent / 'source_orbit_cca.npz')
    columns = head_columns(read(rt['source_contract'])['classes'], rt['old_classes'])
    source = Path(rt['predecessor_root']) / 'sources' / f's{seed}'
    metadata = read(source / 'source_complete.json')
    with np.load(source / 'source_prototypes.npz', allow_pickle=False) as z:
        prototypes = {key: np.array(z[key], copy=True) for key in z.files}
    representations = []
    for restored in (False, True):
        iq = _restore(engine, raw, 'cuda:0', 7, fast_transfer=True) if restored else raw
        f, _ = orbit_features(reader, iq, 'cuda:0', 7, True)
        representations.append(blend(bridge.transform(f, restored=restored, old_columns=columns), 1.))
    max_error = 0.
    checked = 0
    for k in (1, 5):
        support = np.r_[np.arange(k), np.arange(7, 7+k)]
        source_probe = np.array([5, 6, 12, 13])
        for sp in (0, 1):
            modes = []
            for proto in (None, prototypes):
                state = fit_direct_c(subset(representations[sp], support), np.repeat([6, 7], k),
                    [f'source-{i}' for i in support], rule=metadata['original_rule'], source_prototypes=proto)
                modes.append(state)
                scorer = CudaDirectC(state, pair_budget=rt.get('direct_c_pair_budget', 262144))
                for qp in (0, 1):
                    sample = subset(representations[qp], source_probe)
                    cpu, gpu = state.score(sample), scorer.score(sample)
                    max_error = max(max_error, float(np.max(np.abs(cpu-gpu))))
                    require(np.allclose(cpu, gpu, rtol=1e-10, atol=1e-10), 'Source CPU/CUDA score mismatch')
                    classes = rt['old_classes'] + rt['new_classes'][:2]
                    a, ao = _decisions(cpu, classes)
                    b, bo = _decisions(gpu, classes)
                    require(np.array_equal(a,b) and np.array_equal(ao,bo), 'Source CPU/CUDA decision mismatch')
                    checked += len(cpu)
            pair = DirectPair(*modes)
            for qp in (0, 1):
                sample = subset(representations[qp], source_probe)
                for state, combined in zip(modes, pair.score(sample)):
                    separate = state.score(sample)
                    max_error = max(max_error, float(np.max(np.abs(separate-combined))))
                    require(np.allclose(separate, combined, rtol=1e-12, atol=1e-12), 'Shared CPU kernel differs')
                    classes = rt['old_classes'] + rt['new_classes'][:2]
                    a, ao = _decisions(separate, classes)
                    b, bo = _decisions(combined, classes)
                    require(np.array_equal(a,b) and np.array_equal(ao,bo), 'Shared CPU decision mismatch')
                    checked += len(separate)
    write(output, dict(status='PASS', seed=seed, source_only=True, source_probe_examples=14,
          comparison_rows=checked, max_abs_score_difference=max_error, target_reads=0, query_reads=0,
          F_updates=0, G_updates=0, optimizer_updates=0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(read(args.runtime), args.seed, args.output)


if __name__ == '__main__':
    main()
