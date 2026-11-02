"""Source-only prototype export and truth-blind direct-C predictions."""
import argparse
from pathlib import Path
import numpy as np

from .io import read, write, require
from .milkr_screen import initialize, subset
from .milkr.prior_fusion import head_columns
from .milkr.adaptive_residual import blend
from .milkr.r42_direct_c import fit_direct_c, DirectPair
from .milkr import orbit_ensemble
from .teacher_features import AdaptiveResidualBridge
from .r42_direct_c_support import load_new_support
from .r42_suite_worker import _restore, orbit_features, _decisions
from .episodic_experiment import split_rows


def cuda_for_support(rt, count):
    return bool(rt.get('direct_c_cuda', False) and count >= rt.get('direct_c_cuda_min_support', 256))


def source(rt, seed, root):
    from experiments.cric_dura_comparison.f_data import SourceDataset
    from .features import source_smoke
    from .fast_transfer import to_numpy
    local = dict(rt, model_seeds=[seed], **rt['source_reuse_by_seed'][str(seed)])
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    source_smoke(local, root / 'source_smoke.json')
    source_root = Path(rt['source_parent_root']) / 'sources' / f's{seed}'
    parent = read(source_root / 'source_selection.json')
    require(parent['status'] == 'SOURCE_FROZEN' and parent['target_eligible'], 'Source encoder not frozen')
    require(parent['source_new_optimizer_updates'] == 0 and parent['target_reads'] == 0 and parent['query_reads'] == 0, 'Source provenance mismatch')
    contract = read(rt['source_contract'])
    columns = head_columns(contract['classes'], rt['old_classes'])
    data = SourceDataset(rt['source_cache'], 'L_s', contract)
    labels = np.asarray(data.metadata['y'], dtype=np.int64)
    require(set(labels.tolist()) == set(range(6)), 'Wrong source labels')
    reader, _ = initialize(local)
    bridge = AdaptiveResidualBridge.load(source_root / 'source_orbit_cca.npz')
    sums = None
    counts = np.zeros(6, dtype=np.int64)
    batch = int(rt.get('batch_size', 128))
    for start in range(0, len(data), batch):
        raw = to_numpy(data.x[start:start + batch], np.float32)
        features, _ = orbit_features(reader, raw, 'cuda:0', batch, True)
        mapped = bridge.transform(features, restored=False, old_columns=columns)
        normalized = orbit_ensemble.normalize(blend(mapped, 1.))
        if sums is None:
            sums = {name: np.zeros((6,) + array.shape[1:], dtype=np.float64) for name, array in normalized.items()}
        batch_y = labels[start:start + len(raw)]
        for old_col, source_col in enumerate(columns):
            mask = batch_y == source_col
            counts[old_col] += int(mask.sum())
            for name, values in normalized.items():
                sums[name][old_col] += values[mask].sum(axis=0)
    require((counts > 0).all(), 'Missing legal source class samples')
    prototypes = {name: value / counts.reshape((-1,) + (1,) * (value.ndim - 1)) for name, value in sums.items()}
    orbit_ensemble.normalize(prototypes)
    np.savez_compressed(root / 'source_prototypes.npz', **prototypes)
    write(root / 'source_complete.json', dict(status='SOURCE_READY', source_role='L_s',
          source_class_counts=counts.tolist(), source_rows=int(counts.sum()), model_seed=seed,
          target_reads=0, query_reads=0, optimizer_updates=0,
          original_rule=parent['K1_rule'], prototype_policy='mean_normalized_clean_source_L_s_features'))


def predict(rt, job):
    seed = int(job['seed'])
    local = dict(rt, model_seeds=[seed])
    root = Path(job['output'])
    root.mkdir(parents=True, exist_ok=False)
    source_root = Path(rt['run_root']) / 'sources' / f's{seed}'
    source_info = read(source_root / 'source_complete.json')
    require(source_info['status'] == 'SOURCE_READY' and source_info['model_seed'] == seed, 'Source not ready')
    with np.load(source_root / 'source_prototypes.npz', allow_pickle=False) as z:
        prototypes = {name: np.array(z[name], copy=True) for name in z.files}
    reader, engine = initialize(local)
    parent = Path(rt['source_parent_root']) / 'sources' / f's{seed}'
    bridge = AdaptiveResidualBridge.load(parent / 'source_orbit_cca.npz')
    columns = head_columns(read(rt['source_contract'])['classes'], rt['old_classes'])
    cap = Path(rt['data_root']) / job['capsule']
    manifest = read(cap / 'manifest.json')
    require(manifest['protocol_schema'] == 'p2_min_v1' and manifest['phase2_data_status'] == 'VALIDATED_ONCE', 'Wrong data handles')
    specs = {(s['k'], s['new_count']): s for s in (read(p) for p in (cap / 'splits').glob('*.json'))}
    points = sorted(set(tuple(p) for p in job['points']))
    require(points and all(k in (1, 5, 10, 20) and n in (2, 5, 10, 20) for k, n in points), 'Wrong work points')
    states = []
    batch = int(rt.get('batch_size', 128))
    # All fits close before any query sample or query feature is opened.
    for k in sorted({p[0] for p in points}):
        max_n = max(n for kk, n in points if kk == k)
        task = load_new_support(cap, specs[k, max_n]['split_id'])
        for sp in (0, 1):
            raw = task['support_x']
            if sp:
                raw = _restore(engine, raw, 'cuda:0', k, fast_transfer=True)
            features, _ = orbit_features(reader, raw, 'cuda:0', k, True)
            mapped = blend(bridge.transform(features, restored=bool(sp), old_columns=columns), 1.)
            for n in sorted(n for kk, n in points if kk == k):
                count = n * k
                actual = np.asarray(specs[k, n]['support_indices'], dtype=np.int64)[6*k:]
                prefix = np.asarray(task['split']['support_indices'], dtype=np.int64)[6*k:6*k+count]
                require(np.array_equal(actual, prefix), 'New support prefix differs between N')
                labels = task['support_y'][:count]
                ids = task['support_ids'][:count]
                for mode in ('head', 'source_prototype'):
                    state = fit_direct_c(subset(mapped, slice(0, count)), labels, ids,
                        rule=source_info['original_rule'], source_prototypes=prototypes if mode == 'source_prototype' else None)
                    states.append((mode, k, n, sp, state))
                    write(root / f'{mode}-K{k}-N{n}-S{sp}-fit.json', state.audit)
    bank = np.load(cap / 'received.npy', mmap_mode='r', allow_pickle=False)
    rows = []
    scorers = {}
    # K-specific query banks exclude the allocated old/new support for that K.
    # The same physical record can be query at another K under the fixed split.
    for k in sorted({p[0] for p in points}):
      selected = [s for s in states if s[1] == k]
      query_by_n = {n: split_rows(cap, specs[k, n]) for kk, n in points if kk == k}
      allrows = np.unique(np.concatenate(list(query_by_n.values())))
      support = np.asarray(specs[k, max(query_by_n)]['support_indices'], dtype=np.int64)
      require(not np.isin(allrows, support).any(), 'Query bank contains allocated support for this K')
      positions = np.full(len(bank), -1, dtype=np.int64)
      positions[allrows] = np.arange(len(allrows))
      for qp in (0, 1):
        raw = np.asarray(bank[allrows])
        if qp:
            raw = _restore(engine, raw, 'cuda:0', batch, fast_transfer=True)
        features, _ = orbit_features(reader, raw, 'cuda:0', batch, True)
        mapped = blend(bridge.transform(features, restored=bool(qp), old_columns=columns), 1.)
        groups = sorted({(n, sp) for _, _, n, sp, _ in selected})
        for n, sp in groups:
            spec = specs[k, n]
            query = query_by_n[n]
            pos = positions[query]
            require((pos >= 0).all(), 'Missing query positions')
            pair_states = {mode: state for mode, _, nn, ss, state in selected if nn == n and ss == sp}
            require(set(pair_states) == {'head', 'source_prototype'}, 'Missing paired states')
            pieces = {mode: [] for mode in pair_states}
            restricted = {mode: [] for mode in pair_states}
            use_cuda = cuda_for_support(rt, len(pair_states['head'].support['z_id']))
            if use_cuda:
                from .milkr.r42_direct_c_cuda import CudaDirectC
                for mode, state in pair_states.items():
                    key = (mode, k, n, sp)
                    if key not in scorers:
                        scorers[key] = CudaDirectC(state, 'cuda:0', pair_budget=rt.get('direct_c_pair_budget', 262144))
            else:
                pair = DirectPair(pair_states['head'], pair_states['source_prototype'])
            for start in range(0, len(pos), int(rt.get('decision_batch', 512))):
                sample = subset(mapped, pos[start:start + int(rt.get('decision_batch', 512))])
                if use_cuda:
                    values = [scorers[(mode, k, n, sp)].score(sample) for mode in ('head', 'source_prototype')]
                else:
                    values = pair.score(sample)
                for mode, scores in zip(('head', 'source_prototype'), values):
                    pred, old = _decisions(scores, spec['registered_classes'])
                    pieces[mode].append(pred)
                    restricted[mode].append(old)
            for mode in ('head', 'source_prototype'):
                predicted = np.concatenate(pieces[mode]).astype(np.int16)
                old_restricted = np.concatenate(restricted[mode]).astype(np.int16)
                name = f'{mode}-C-K{k}-N{n}-S{sp}Q{qp}.npz'
                np.savez_compressed(root / name, query_indices=query, predicted_indices=predicted, old_restricted_indices=old_restricted)
                rows.append(dict(arm='R42_direct_' + mode, stage='C', k=k, new_count=n,
                    input_path=f'S{sp}Q{qp}', split_id=spec['split_id'], registered_classes=spec['registered_classes'],
                    prediction=name, actual_target_old_labels=0, actual_target_new_labels=n*k,
                    execution_backend='cuda_fp64' if use_cuda else 'cpu_fp64_shared_kernel'))
    require(len(rows) == len(points) * 8, 'Incomplete direct-C prediction matrix')
    write(root / 'predictions.json', dict(capsule=job['capsule'], capsule_id=manifest['capsule_id'], model_seed=seed, rows=rows))
    write(root / 'complete.json', dict(status='PREDICTIONS_COMPLETE', capsule_id=manifest['capsule_id'],
          rows=len(rows), truth_read=False, query_fit=False, target_old_support_read=False,
          query_scope='per_K_query_union; records may be query for another K'))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('source', 'predict'))
    p.add_argument('--runtime', required=True)
    p.add_argument('--seed', type=int)
    p.add_argument('--job')
    args = p.parse_args()
    rt = read(args.runtime)
    if args.action == 'source':
        require(args.seed in rt['model_seeds'], 'Wrong model seed')
        source(rt, args.seed, Path(rt['run_root']) / 'sources' / f's{args.seed}')
    else:
        predict(rt, read(args.job))


if __name__ == '__main__':
    main()
