"""Post-scoring GPU0 resource profile; no training, truth, or accuracy output.

Run independently, after the frozen full suite closes:
python -m experiments.pamf_phase2_practical.r42_suite_profile --runtime ... --output ...
"""
import argparse
from collections.abc import Mapping
import csv
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from .io import read, write, require
from .group1_resources import Measure, tensor_bytes
from .r42_suite_worker import array_bytes

SEED = 2026092701
CAPSULE = 'capsules/P01/MIX01/1-1'
REPEATS = 3
QUERY_LIMIT = 256
WARMUP_RECORDS = 10
RTOL, ATOL = 2e-4, 2e-5


def gpu0_snapshot():
    """Observe other owners; never stop, signal, or alter their processes."""
    uuid = subprocess.check_output(['nvidia-smi', '-i', '0', '--query-gpu=uuid',
        '--format=csv,noheader,nounits'], text=True).strip()
    lines = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory',
        '--format=csv,noheader,nounits'], text=True)
    processes = [dict(gpu_uuid=row[0].strip(), pid=int(row[1]), process_name=row[2].strip(),
                      used_gpu_memory_MiB=row[3].strip())
                 for row in csv.reader(lines.splitlines()) if row and row[0].strip() == uuid]
    others = [p for p in processes if p['pid'] != os.getpid()]
    return dict(observed_at=time.time(), physical_GPU=0, GPU_uuid=uuid, processes=processes,
                other_compute_processes=others, isolation='shared_GPU' if others else 'observed_isolated_GPU')


def timed(function):
    import torch
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = function()
    torch.cuda.synchronize()
    return result, time.perf_counter() - start


def latency_summary(records, batch):
    require(records and all(r['sample_count'] > 0 for r in records), 'Missing measured latency calls')
    result = dict(sample_count=sum(r['sample_count'] for r in records), batch_count=len(records),
        requested_batch=batch, observed_batch_sizes=[r['sample_count'] for r in records],
        quantile_unit='individual_record_call' if batch == 1 else 'whole_batch_call',
        stages={}, calls=records)
    for key in ('pipeline_seconds', 'G_seconds', 'F_seconds', 'NN_seconds', 'head_seconds'):
        values = np.asarray([r[key] for r in records], np.float64)
        result['stages'][key] = dict(total_seconds=float(values.sum()),
            p50_seconds=float(np.quantile(values, .50)), p95_seconds=float(np.quantile(values, .95)),
            p99_seconds=float(np.quantile(values, .99)),
            amortized_seconds_per_record=float(values.sum() / result['sample_count']))
    return result


def _leaves(value, prefix=''):
    """Deployment geometry and coefficients, excluding timing/selection audits."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, np.ndarray) or isinstance(value, (str, float, int, bool, type(None))):
        return {prefix: value}
    if isinstance(value, Mapping):
        return {p: v for k, child in value.items()
                if k not in ('selection', 'audit', 'training_summary', 'correlation_summary')
                for p, v in _leaves(child, prefix + '/' + str(k)).items()}
    if isinstance(value, (tuple, list)):
        return {p: v for i, child in enumerate(value) for p, v in _leaves(child, prefix + '/' + str(i)).items()}
    if hasattr(value, '__dict__'):
        return _leaves(vars(value), prefix)
    raise TypeError('Unsupported deployed state member: ' + type(value).__name__)


def compare_states(cold, incremental):
    a, b = _leaves(cold), _leaves(incremental)
    failures = []
    maximum = 0.
    for key in sorted(set(a) | set(b)):
        if key not in a or key not in b:
            failures.append(key)
            continue
        x, y = a[key], b[key]
        if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
            if not (isinstance(x, np.ndarray) and isinstance(y, np.ndarray) and x.shape == y.shape):
                failures.append(key)
                continue
            if x.dtype.kind in 'fiu' and y.dtype.kind in 'fiu':
                if x.size:
                    maximum = max(maximum, float(np.max(np.abs(x.astype(np.float64) - y.astype(np.float64)))))
                same = np.allclose(x, y, rtol=RTOL, atol=ATOL, equal_nan=False)
            else:
                same = np.array_equal(x, y)
        elif isinstance(x, (float, int)) and isinstance(y, (float, int)):
            maximum = max(maximum, abs(float(x) - float(y)))
            exact = key.endswith(('/ridge', '/strength')) or '/config/' in key
            same = x == y if exact else bool(np.isclose(x, y, rtol=RTOL, atol=ATOL))
        else:
            same = x == y
        if not same:
            failures.append(key)
    return dict(passed=not failures, rtol=RTOL, atol=ATOL, compared_leaves=len(a),
                maximum_absolute_difference=maximum, failed_members=failures,
                scope='all deployed feature geometry/kernel state, coefficients, config, and residual strength')


def merge_support(old, new, old_ids, current_ids, current_labels):
    mask = np.asarray(current_labels) < 6
    require(np.array_equal(np.asarray(old_ids), np.asarray(current_ids)[mask]), 'B/C old support IDs differ')
    require(np.array_equal(mask, np.arange(len(mask)) < int(mask.sum())), 'Unexpected noncanonical support order')
    require(set(old) == set(new), 'B/new feature schemas differ')
    return {key: np.concatenate((old[key], new[key]), axis=0) for key in old}


def support_input(cap, spec, bank):
    indices = np.asarray(spec['support_indices'], np.int32)
    ids = np.load(cap / 'ids.npy', mmap_mode='r', allow_pickle=False)
    labels = np.load(cap / 'support_labels' / (spec['split_id'] + '.npy'), allow_pickle=False)
    require(np.array_equal(labels, np.repeat(np.arange(6 + spec['new_count']), 5)), 'Wrong K5 support labels')
    return dict(x=np.asarray(bank[indices], np.float32), ids=np.asarray(ids[indices]), y=labels)


def fresh_engine(engine):
    from experiments.pamf_stage2_inference.engine import InferenceEngine
    return InferenceEngine(engine.pipeline.model, reuse=True, graph=True, native=False)


def extract_path(reader, engine, bridge, columns, arm, raw, restored, batch):
    from .milkr_screen import extract as baseline
    from .teacher_features import extract_orbit_ensemble
    from .r42_suite_worker import _restore
    times = dict(G_seconds=0., F_seconds=0., NN_seconds=0.)
    if restored:
        raw, times['G_seconds'] = timed(lambda: _restore(engine, raw, 'cuda:0', batch))
    extractor = extract_orbit_ensemble if arm == 'R42' else baseline
    (features, _), times['F_seconds'] = timed(lambda: extractor(reader, raw, 'cuda:0', None, batch=batch))
    if arm == 'R42':
        features, times['NN_seconds'] = timed(lambda: bridge.transform(features, restored=restored, old_columns=columns))
    return features, times


def fit_head(arm, features, support, spec, old_classes, rule):
    if arm == 'R42':
        from .milkr.adaptive_residual import fit
        return fit(features, support['y'], support['ids'], fixed=rule)
    from .milkr.d92_branch_local_ridge import fit_branch_local_ridge
    return fit_branch_local_ridge(**features, support_labels=np.asarray(support['y'], np.int64),
        support_ids=list(map(str, support['ids'])), classes=spec['registered_classes'],
        old_classes=old_classes, arm='local_ridge')


def query_call(reader, engine, bridge, columns, arm, state, raw, path):
    """Actual F/FFT or C8 RF plus actual four-phase-head payload; no proxy."""
    from .milkr.prior_fusion import fuse
    start = time.perf_counter()
    stages = dict(G_seconds=0., F_seconds=0., NN_seconds=0., head_seconds=0.)
    evaluations = []
    for restored in ((False, True) if path == 'fusion' else (path == 'G-only',)):
        features, parts = extract_path(reader, engine, bridge, columns, arm, raw, restored, len(raw))
        for key, seconds in parts.items():
            stages[key] += seconds
        def evaluate():
            if arm == 'R42':
                return state.evaluate(features)
            return state.score(**features), None
        values, seconds = timed(evaluate)
        stages['head_seconds'] += seconds
        evaluations.append(values)
    def final_decision():
        scores = fuse(evaluations[0], evaluations[1])[0] if path == 'fusion' else evaluations[0][0]
        require(np.isfinite(scores).all() and scores.ndim == 2, 'Nonfinite all-class profile scores')
        return scores.argmax(1).shape
    shape, seconds = timed(final_decision)
    stages['head_seconds'] += seconds
    return dict(sample_count=len(raw), pipeline_seconds=time.perf_counter() - start,
                decision_shape=list(shape), **stages)


def support_profiles(output, report, reader, engine, bridge, columns, supports, specs, rule):
    heads = {}
    for arm in ('R42', 'LocalRidge'):
        for sp in (0, 1):
            for repeat in range(REPEATS):
                b_features = cold_features = cold_state = b_engine = None
                for stage in ('B_cold', 'C_from_A_cold', 'C_after_B_incremental'):
                    label = f'{arm}-S{sp}-r{repeat}-{stage}'
                    snapshot = gpu0_snapshot()
                    incremental = stage == 'C_after_B_incremental'
                    spec = specs[0 if stage == 'B_cold' else 20]
                    support = supports[spec['new_count']]
                    current_engine = b_engine if incremental else fresh_engine(engine)
                    raw = support['x'][support['y'] >= 6] if incremental else support['x']
                    with Measure(output / (label + '-memory.json'),
                        dict(arm=arm, support_path=sp, repeat=repeat, stage=stage,
                             concurrency=snapshot['isolation'], native_optimizer_steps=0), reader) as probe:
                        probe.result['concurrency'] = snapshot['isolation']
                        features, parts = extract_path(reader, current_engine, bridge, columns,
                                                       arm, raw, bool(sp), 5)
                        assembly_seconds = 0.
                        if incremental:
                            features, assembly_seconds = timed(lambda: merge_support(b_features, features,
                                supports[0]['ids'], support['ids'], support['y']))
                        state, fit_seconds = timed(lambda: fit_head(arm, features, support, spec,
                            specs[0]['registered_classes'], rule))
                        probe.result.update(**parts, feature_assembly_seconds=assembly_seconds,
                            CPU_fit_seconds=fit_seconds, head_state_bytes=array_bytes(state),
                            new_support_record_count=len(raw), total_fit_support_records=len(support['y']),
                            old_support_features_reused=incremental, B_coefficients_inherited=False,
                            full_kernel_refit=True, graph_setup=current_engine.setup())
                        if incremental:
                            comparison = compare_states(cold_state, state)
                            feature_comparison = compare_states(cold_features, features)
                            probe.result.update(cold_C_comparison=comparison, cold_C_feature_comparison=feature_comparison)
                            require(comparison['passed'] and feature_comparison['passed'], 'C-after-B geometry differs from cold C')
                    report['support'].append(dict(label=label, memory_file=label + '-memory.json',
                        telemetry=probe.result, GPU_before=snapshot, GPU_after=gpu0_snapshot()))
                    if stage == 'B_cold':
                        b_features, b_engine = features, current_engine
                    elif stage == 'C_from_A_cold':
                        cold_features, cold_state = features, state
                        heads[arm, sp] = state
    return heads


def query_profiles(output, report, reader, engine, bridge, columns, heads, raw):
    for arm in ('R42', 'LocalRidge'):
        for sp in (0, 1):
            for path in (('raw', 'G-only', 'fusion') if arm == 'R42' else ('raw', 'G-only')):
                for batch in (1, 128):
                    label = f'{arm}-S{sp}-{path}-B{batch}'
                    active = fresh_engine(engine)
                    function = lambda chunk: query_call(reader, active, bridge, columns, arm, heads[arm, sp], chunk, path)
                    chunks = [raw[i:i + batch] for i in range(0, len(raw), batch)]
                    snapshot = gpu0_snapshot()
                    with Measure(output / (label + '-memory.json'),
                        dict(arm=arm, support_path=sp, query_path=path, batch=batch,
                             concurrency=snapshot['isolation'], native_optimizer_steps=0), reader) as probe:
                        probe.result['concurrency'] = snapshot['isolation']
                        # Each measured batch shape receives its first call outside
                        # warmup and measured quantiles, including graph capture.
                        cold = [function(raw[:size]) for size in sorted({len(x) for x in chunks}, reverse=True)]
                        cold_setup = active.setup()
                        warmup = [function(raw[i:min(i + batch, WARMUP_RECORDS)])
                                  for i in range(0, WARMUP_RECORDS, batch)]
                        measured = [function(chunk) for chunk in chunks]
                        summary = latency_summary(measured, batch)
                        probe.result.update(measured_sample_count=summary['sample_count'],
                            measured_batch_count=summary['batch_count'], cold_first_calls=cold,
                            cold_graph_setup=cold_setup, warmup_records=WARMUP_RECORDS,
                            warmup_calls=warmup, warmup_excluded_from_latency=True,
                            measured_summary=summary, graph_setup_after=active.setup())
                    report['query'].append(dict(label=label, telemetry=probe.result,
                        memory_file=label + '-memory.json', GPU_before=snapshot, GPU_after=gpu0_snapshot()))


def run(rt, output):
    """Use the complete suite runtime, with immutable frozen source artifacts."""
    import torch
    from .milkr_screen import initialize
    from .teacher_features import AdaptiveResidualBridge
    from .milkr.prior_fusion import head_columns
    from .episodic_experiment import split_rows
    root = Path(rt['run_root'])
    require(read(root / 'complete.json')['status'] == 'SCORED_COMPLETE', 'Profile runs only after scientific scoring completes')
    require(read(root / 'predictions_complete.json')['status'] == 'ALL_PREDICTIONS_FIXED', 'Suite predictions are not fixed')
    require(rt['kernel_method'] == 'adaptive_residual_ensemble' and rt['arm'] == 'R42'
            and rt['model_seeds'] == [2026092701, 2026092702, 2026092703, 2026092704]
            and CAPSULE in rt['capsules'], 'Wrong frozen four-source suite runtime')
    source_files = []
    for seed in rt['model_seeds']:
        source = root / 'sources' / f's{seed}'
        done = read(source / 'suite_source_complete.json')
        require(done['status'] == 'R42_SOURCE_READY' and done['model_seed'] == seed
                and done['source_new_optimizer_updates'] == 0, 'Source reuse is not frozen')
        source_files.append(dict(seed=seed, source=str(source), new_training_updates=0))
    require(not torch.cuda.is_initialized(), 'Use a fresh profile process before CUDA is initialized')
    output = Path(output)
    require(output.resolve() != root.resolve() and root.resolve() not in output.resolve().parents,
            'Resource profile must use an independent output directory outside the scientific suite')
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status='RUNNING', capsule=CAPSULE, F_seed=SEED, K=5, N=20, support_repeats=REPEATS,
        source_reuse=source_files, training_updates=0, query_fit=False, truth_reads=0,
        new_accuracy_artifacts=False, MACs=None, FLOPs=None, energy_joules=None,
        query_latency_stage='C_from_A_26_registered_classes',
        unmeasured_reason='operation counts and attributable energy were not measured',
        cold_scope='new graph cache with existing frozen model weights; includes first-call capture; not OS/process cold',
        latency_scope='resident received IQ through GPU transfers, G, actual F/FFT, frozen source NN/CCA, and CPU head',
        F_scope=dict(R42='eight actual RF forwards and four actual default-head phases per path',
                     LocalRidge='one actual original F/default-head forward and FFT per path'),
        LocalRidge_fusion='not applicable: original control has only raw/G query paths',
        memory_scope='actual profiler-process peaks including shared F/G/bridge, CPU state snapshots and retained heads; not isolated per-cell deployment peaks',
        model_bytes_scope='parameter/buffer or deployed array storage; graph workspace is reflected in measured process CUDA memory',
        support=[], query=[], GPU_observations=[])
    before = g_before = reader = engine = None
    try:
        report['GPU_observations'].append(gpu0_snapshot())
        # Bind the physical nvidia-smi GPU0 UUID, independent of CUDA ordinal ordering.
        os.environ['CUDA_VISIBLE_DEVICES'] = report['GPU_observations'][0]['GPU_uuid']
        local = dict(rt, model_seeds=[SEED])
        reader, engine = initialize(local)
        before = {key: value.detach().cpu().clone() for key, value in reader.state_dict().items()}
        g_before = {key: value.detach().cpu().clone() for key, value in engine.pipeline.model.state_dict().items()}
        source = root / 'sources' / f's{SEED}'
        selection = read(source / 'source_selection.json')
        require(selection['source_new_optimizer_updates'] == 0, 'Frozen source NN would require new fitting')
        bridge = AdaptiveResidualBridge.load(source / 'source_orbit_cca.npz')
        cap = Path(rt['data_root']) / CAPSULE
        manifest = read(cap / 'manifest.json')
        require(manifest['protocol_schema'] == 'p2_min_v1'
                and manifest['phase2_data_status'] == 'VALIDATED_ONCE', 'Wrong capsule handles')
        specs = {s['new_count']: s for s in (read(p) for p in (cap / 'splits').glob('*.json'))
                 if s['k'] == 5 and s['new_count'] in (0, 20)}
        require(set(specs) == {0, 20} and all(s['capsule_id'] == manifest['capsule_id'] for s in specs.values()),
                'Missing representative K5/N20 splits')
        columns = head_columns(read(rt['source_contract'])['classes'], specs[0]['registered_classes'])
        bank = np.load(cap / 'received.npy', mmap_mode='r', allow_pickle=False)
        supports = {n: support_input(cap, spec, bank) for n, spec in specs.items()}
        query_indices = split_rows(cap, specs[20])[:QUERY_LIMIT]
        raw = np.asarray(bank[query_indices], np.float32)
        require(len(raw) >= WARMUP_RECORDS and not np.isin(query_indices, specs[20]['support_indices']).any(),
                'Insufficient legal profile query records')
        report.update(capsule_id=manifest['capsule_id'], split_ids={str(n): s['split_id'] for n, s in specs.items()},
            query_indices=query_indices.tolist(), query_sample_rule='first min(256, legal C query count), original index order',
            model_state_bytes=dict(F=tensor_bytes(reader.state_dict()), G=tensor_bytes(engine.pipeline.model.state_dict()),
                frozen_source_NN=array_bytes([e.neural for e in bridge.encoders]),
                frozen_source_CCA=array_bytes([e.parameters for e in bridge.encoders])))
        heads = support_profiles(output, report, reader, engine, bridge, columns, supports, specs, selection['K1_rule'])
        report['model_state_bytes']['heads'] = {f'{arm}-S{sp}': array_bytes(state) for (arm, sp), state in heads.items()}
        query_profiles(output, report, reader, engine, bridge, columns, heads, raw)
        report['status'] = 'VERIFIED'
    except Exception as error:
        report.update(status='FAILED', error=repr(error))
        raise
    finally:
        if before is not None:
            report['F_state_unchanged'] = all(torch.equal(before[k], v.detach().cpu()) for k, v in reader.state_dict().items())
        if g_before is not None:
            report['G_state_unchanged'] = all(torch.equal(g_before[k], v.detach().cpu()) for k, v in engine.pipeline.model.state_dict().items())
        if before is not None and g_before is not None:
            if not (report['F_state_unchanged'] and report['G_state_unchanged']):
                report.update(status='FAILED', error='Frozen F/G state changed during resource profile')
        try:
            report['GPU_observations'].append(gpu0_snapshot())
        except Exception as error:
            report.update(status='FAILED', GPU_observation_error=repr(error))
        observations = report['GPU_observations'] + [entry[key] for name in ('support', 'query')
            for entry in report[name] for key in ('GPU_before', 'GPU_after')]
        report['concurrency'] = ('UNKNOWN' if report.get('GPU_observation_error') else
            'shared_GPU_nonisolated' if any(o['other_compute_processes'] for o in observations) else 'observed_isolated_GPU0_snapshots')
        write(output / 'profile.json', report)
    require(report['status'] == 'VERIFIED', report.get('error', 'Resource profile failed'))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(read(args.runtime), args.output)
