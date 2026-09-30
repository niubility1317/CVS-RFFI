"""Support-only registration diagnostics for the unchanged frozen LocalRidge."""
import argparse
from copy import deepcopy
import csv
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'code'), str(ROOT / 'tools')]
from cvsrffi import d92_branch_interaction as interaction
from cvsrffi.d92_branch_local_ridge import FROZEN_CONFIG, fit_branch_local_ridge
from cvsrffi.d92_branch_support_probe import FROZEN_CONFIG as CACHE_VALIDATION_CONFIG
from d92_registration_score_diagnostics import diagnose_registration
from evaluate_d92_branch_support_probe import load_support, read, write, check, scalars, csv_record, peak_rss
from evaluate_d92_branch_local_ridge_probe import stage_csv_from_jsonl
from run_d92_registration_diagnostic import DIAGNOSTIC_CONFIG

SCOPE = 'SUPPORT_ONLY_LOCAL_RIDGE_REGISTRATION_MECHANISM_NOT_QUERY_OR_ADAPTATION'
STATUS = 'REGISTRATION_DIAGNOSTIC_COMPLETE'
CHANNEL = dict(route='residual', mode='post_sync', equalization_enabled=False, fs_hz=25000000)
SCENARIOS = ['practical_high', 'practical_mid', 'practical_low_urban']
BRANCHES = ('z_id', 'fft', 't_emb', 'f_emb', 'pa_local')
IDENTITY_FIELDS = ('split_id', 'receiver', 'scenario', 'k', 'support_seed', 'registered_classes', 'new_count')
KS = [1, 5, 10, 20]
NEW_COUNTS = [0, 2, 5, 10, 20]
METRICS = ('B0_old_accuracy', 'C0_old_columns_accuracy', 'C0_old_accuracy', 'C0_new_accuracy',
           'C0_h', 'C0_new_minus_old', 'C0_abs_new_old_gap', 'old_order_change',
           'new_competition_loss', 'total_old_accuracy_drop', 'old_order_recovery_fraction',
           'total_recovery_fraction', 'old_winner_changed_fraction', 'old_displaced_by_new_fraction')


def split_identity(split, old_classes):
    result = {key: deepcopy(split[key]) for key in IDENTITY_FIELDS if key != 'new_count'}
    result['new_count'] = len(split['registered_classes']) - len(old_classes)
    return result


def validate_selection(selection):
    """Validate the explicit forty-parent metadata plan, without opening data."""
    check(set(selection) == {'receiver_scenes', 'support_seed', 'ks', 'new_counts', 'splits'}, 'Unexpected selection fields')
    pairs = selection['receiver_scenes']
    check(isinstance(pairs, list) and len(pairs) == 2
          and all(isinstance(p, (list, tuple)) and len(p) == 2 and all(isinstance(v, str) and v for v in p) for p in pairs)
          and len({tuple(p) for p in pairs}) == 2, 'Exactly two explicit receiver/scene pairs required')
    check(all(p[1] in SCENARIOS for p in pairs), 'Selection must retain practical residual scenes')
    check(type(selection['support_seed']) is int and selection['support_seed'] >= 0
          and selection['ks'] == KS and selection['new_counts'] == NEW_COUNTS, 'Fixed pilot axes mismatch')
    expected = {(rx, scene, k, new, selection['support_seed']) for rx, scene in pairs for k in KS for new in NEW_COUNTS}
    ids, cells = set(), set()
    for item in selection['splits']:
        check(set(item) == set(IDENTITY_FIELDS), 'Selected split identity schema mismatch')
        cell = tuple(item[key] for key in ('receiver', 'scenario', 'k', 'new_count', 'support_seed'))
        check(isinstance(item['split_id'], str) and item['split_id'] and item['split_id'] not in ids
              and cell in expected and cell not in cells, 'Duplicate or unexpected selected split')
        classes = item['registered_classes']
        check(isinstance(classes, list) and classes and all(isinstance(v, str) and v for v in classes)
              and len(classes) == len(set(classes)), 'Invalid selected class registry')
        ids.add(item['split_id']); cells.add(cell)
    check(cells == expected and len(ids) == 40, 'Incomplete explicit forty-parent selection')
    return ids


def selected_tasks(tasks, selection, old):
    wanted = validate_selection(selection)
    lookup = {split['split_id']: (split, positions, labels) for split, positions, labels in tasks}
    check(wanted <= set(lookup), 'Selected split missing from complete producer plan')
    chosen = []
    old_maps = {}
    for identity in selection['splits']:
        task = lookup[identity['split_id']]; split = task[0]
        check(split_identity(split, old) == identity, 'Selected split identity differs from producer metadata')
        classes = split['registered_classes']
        check(set(old) <= set(classes), 'Missing old registry')
        mapping = {pid: classes[label] for pid, label in zip(split['support_ids'], split['support_labels'])}
        old_map = {pid: cls for pid, cls in mapping.items() if cls in old}
        key = tuple(split[v] for v in ('receiver', 'scenario', 'k', 'support_seed'))
        if key in old_maps:
            check(old_maps[key] == old_map, 'Old support physical identity/label mismatch across new counts')
        old_maps[key] = old_map
        chosen.append(task)
    return chosen


def measured_metrics(diagnostic):
    n = diagnostic['old_held_count']
    new, old = diagnostic['Cnew_acc'], diagnostic['C_full_acc']
    result = dict(B0_old_accuracy=diagnostic['B_acc'], C0_old_columns_accuracy=diagnostic['C_old_columns_acc'],
        C0_old_accuracy=old, C0_new_accuracy=new, C0_h=diagnostic['C_harmonic_mean'],
        C0_new_minus_old=None if new is None else new-old, C0_abs_new_old_gap=None if new is None else abs(new-old),
        old_order_recovery_fraction=diagnostic['transitions']['old_order_change']['incorrect_to_correct']/n,
        total_recovery_fraction=diagnostic['transitions']['total']['incorrect_to_correct']/n,
        old_winner_changed_fraction=diagnostic['old_winner_changed_count']/n,
        old_displaced_by_new_fraction=diagnostic['old_winner_displaced_by_new_count']/n)
    result.update({name: diagnostic['decomposition'][name]['accuracy_fraction'] for name in
                   ('old_order_change', 'new_competition_loss', 'total_old_accuracy_drop')})
    return result


def parent_mean(trials):
    return {key: None if trials[0]['metrics'][key] is None else
            sum(trial['metrics'][key] for trial in trials)/len(trials) for key in METRICS}


def diagnose_evidence(evidence, old):
    return diagnose_registration(**{key: evidence[key] for key in
        ('b_scores', 'b_classes', 'b_ids', 'c_scores', 'c_classes', 'c_ids', 'held_labels')}, old_classes=old)


def _safe(value):
    if isinstance(value, dict): return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_safe(v) for v in value]
    if isinstance(value, np.ndarray): return _safe(value.tolist())
    if isinstance(value, np.generic): return _safe(value.item())
    if isinstance(value, float) and not np.isfinite(value): return str(value)
    return value


def probe_registration(*, z_id, fft, t_emb, f_emb, pa_local, support_labels, support_ids,
                       classes, old_classes, log_callback=None):
    """Fresh B0/C0 fits per fold; no state or training rows cross a path."""
    started = time.perf_counter()
    b, a, labels, ids, canonical, _, old, k = interaction._prepare(
        z_id, fft, t_emb, f_emb, pa_local, support_labels, support_ids, classes, old_classes)
    check(bool(old), 'Old registry required')
    classes, old = list(canonical), sorted(old)
    order = np.asarray(sorted(range(len(ids)), key=lambda i: support_ids[i]))
    raw = {name: np.asarray(value)[order] for name, value in zip(BRANCHES, (z_id, fft, t_emb, f_emb, pa_local))}
    label_by_id = {pid: classes[int(label)] for pid, label in zip(ids, labels)}
    is_old = np.asarray([label_by_id[pid] in old for pid in ids])
    folds = 0 if k == 1 else min(k, 3)
    result = dict(classes=classes, old_classes=old, support_count=len(ids), old_class_count=len(old),
        new_class_count=len(classes)-len(old), fold_count=folds, physical_fold_assignment=[],
        numerical=interaction._diagonal_stats(b, a), folds=[], oof=None, oneshot_proxy=None,
        sequence_paths=0, head_fit_count=0, factorization_count=0, optimizer_steps=0,
        persistent_state_bytes=0, heldout_unavailable_reason='K1_NO_INDEPENDENT_PHYSICAL_HOLDOUT' if k == 1 else None)
    if k == 1:
        result['fit_seconds'] = time.perf_counter()-started
        return result
    positions = [np.flatnonzero(labels == i) for i in range(len(classes))]
    assignments = np.full(len(ids), -1, dtype=int)
    for indices in positions: assignments[indices] = np.arange(k) % folds
    result['physical_fold_assignment'] = [dict(physical_id=pid, class_id=label_by_id[pid], fold=int(assignments[i])) for i, pid in enumerate(ids)]
    completed = []
    completed_path_evidence = []

    def path(keep, scope, index):
        entry = dict(scope=scope, fold=index if scope == 'support_oof' else None,
            trial=index if scope == 'support_oneshot_proxy' else None, parent_k=k,
            train_k=int(keep.sum())//len(classes), held_k=int((~keep).sum())//len(classes),
            b_training_ids=[pid for i, pid in enumerate(ids) if keep[i] and is_old[i]],
            c_training_ids=[pid for i, pid in enumerate(ids) if keep[i]],
            b_ids=[pid for i, pid in enumerate(ids) if not keep[i] and is_old[i]],
            c_ids=[pid for i, pid in enumerate(ids) if not keep[i]], b_classes=old, c_classes=classes,
            held_labels={pid: label_by_id[pid] for i, pid in enumerate(ids) if not keep[i]},
            stages=[], c_reuses_b0=len(classes) == len(old))
        try:
            for name, train_mask, held_mask, registry in (
                    ('B0', keep & is_old, ~keep & is_old, old), ('C0', keep, ~keep, classes)):
                if name == 'C0' and entry['c_reuses_b0']:
                    entry['c_scores'] = deepcopy(entry['b_scores'])
                    continue
                training_ids = [pid for i, pid in enumerate(ids) if train_mask[i]]
                y = np.asarray([registry.index(label_by_id[pid]) for pid in training_ids], dtype=np.int64)
                stage_started = time.perf_counter()
                state = fit_branch_local_ridge(**{key: value[train_mask] for key, value in raw.items()},
                    support_labels=y, support_ids=training_ids, classes=registry, old_classes=old, arm='local_ridge')
                audit = state.audit_dict()['final_fit']
                audit.update(state=name, scope=scope, fold=entry['fold'], trial=entry['trial'], parent_k=k,
                    training_physical_ids=training_ids, train_k=entry['train_k'],
                    all_states_estimated_from_trainfold_only=True)
                entry['stages'].append(audit)
                result['head_fit_count'] += 1
                result['factorization_count'] += audit['factorization_calls']
                score_started = time.perf_counter()
                scores = state.score(**{key: value[held_mask] for key, value in raw.items()})
                audit.update(score_seconds=time.perf_counter()-score_started,
                    fit_and_score_seconds=time.perf_counter()-stage_started,
                    held_physical_count=int(held_mask.sum()), class_count=len(registry),
                    learning_rate=None, learning_rate_reason='Closed-form frozen LocalRidge; no optimizer',
                    source_validation=None, source_validation_reason='No source samples or features accessed')
                entry['b_scores' if name == 'B0' else 'c_scores'] = scores.tolist()
                if log_callback: log_callback(audit)
            entry['diagnostic'] = diagnose_evidence(entry, old)
            entry['metrics'] = measured_metrics(entry['diagnostic'])
        except Exception as exc:
            details = exc.audit_dict() if callable(getattr(exc, 'audit_dict', None)) else {}
            exc.registration_context = _safe(dict(current_path=entry, failed_fit=details,
                completed_stages=completed, completed_paths=result['sequence_paths'],
                completed_path_evidence=completed_path_evidence))
            raise
        completed.extend(deepcopy(entry['stages']))
        completed_path_evidence.append(entry)
        result['sequence_paths'] += 1
        return entry

    b_rows, c_rows = {}, {}
    for fold in range(folds):
        entry = path(assignments != fold, 'support_oof', fold)
        result['folds'].append(entry)
        b_rows.update(zip(entry['b_ids'], entry['b_scores']))
        c_rows.update(zip(entry['c_ids'], entry['c_scores']))
    evidence = dict(b_ids=sorted(b_rows), b_classes=old, b_scores=[b_rows[v] for v in sorted(b_rows)],
        c_ids=sorted(c_rows), c_classes=classes, c_scores=[c_rows[v] for v in sorted(c_rows)], held_labels=label_by_id)
    # Every physical held row appears exactly once; unequal-sized folds are not averaged.
    diagnostic = diagnose_evidence(evidence, old)
    result['oof'] = dict(diagnostic=diagnostic, metrics=measured_metrics(diagnostic), aggregation='one_record_per_physical_held_id')
    trials = []
    for trial in range(k):
        keep = np.zeros(len(ids), dtype=bool)
        for indices in positions: keep[indices[trial]] = True
        trials.append(path(keep, 'support_oneshot_proxy', trial))
    result['oneshot_proxy'] = dict(trials=trials, trial_count=k, parent_mean_metrics=parent_mean(trials),
        aggregation='all_anchors_mean_within_parent_then_equal_parent', proxy_train_k=1)
    result['fit_seconds'] = time.perf_counter()-started
    return result


def compact_record(record):
    keys = ('split_id', 'receiver', 'scenario', 'k', 'support_seed', 'new_count', 'support_count',
            'old_class_count', 'new_class_count', 'fold_count', 'sequence_paths', 'head_fit_count',
            'factorization_count', 'optimizer_steps', 'fit_seconds', 'heldout_unavailable_reason')
    result = {key: record[key] for key in keys}
    result.update(oof=None if record['oof'] is None else record['oof']['metrics'],
        proxy=None if record['oneshot_proxy'] is None else record['oneshot_proxy']['parent_mean_metrics'],
        proxy_anchor_count=0 if record['oneshot_proxy'] is None else record['oneshot_proxy']['trial_count'],
        oof_old_held_count=0 if record['oof'] is None else record['oof']['diagnostic']['old_held_count'],
        oof_new_held_count=0 if record['oof'] is None else record['oof']['diagnostic']['additional_new_held_count'],
        scope=SCOPE, query_rows_used=0, source_rows_used=0)
    result['path_count_ranges'] = {}
    groups = dict(oof=record['folds'], proxy=[] if record['oneshot_proxy'] is None else record['oneshot_proxy']['trials'])
    for name, paths in groups.items():
        result['path_count_ranges'][name] = None if not paths else {
            key: dict(min=min(values), max=max(values)) for key, values in {
                'train_k': [p['train_k'] for p in paths], 'held_k': [p['held_k'] for p in paths],
                'b_train_count': [len(p['b_training_ids']) for p in paths],
                'c_train_count': [len(p['c_training_ids']) for p in paths],
                'b_held_count': [len(p['b_ids']) for p in paths],
                'c_held_count': [len(p['c_ids']) for p in paths]}.items()}
    return result


def evaluate(*, support_features, capsule, output, config, expected_capsule_id,
             expected_checkpoint_sha256, expected_model_seed):
    out = Path(output)
    if out.exists(): raise FileExistsError(out)
    check(set(config) == {'algorithm', 'producer_matrix', 'selection'} and config['algorithm'] == DIAGNOSTIC_CONFIG,
          'Frozen registration diagnostic configuration mismatch')
    check(DIAGNOSTIC_CONFIG['base_algorithm'] == FROZEN_CONFIG, 'Frozen LocalRidge code/config mismatch')
    validate_selection(config['selection'])
    manifest = read(Path(capsule) / 'manifest.json')
    check(all(manifest.get('channel', {}).get(key) == value for key, value in CHANNEL.items())
          and manifest.get('scenarios') == SCENARIOS, 'Practical residual post_sync/noeq capsule metadata mismatch')
    begin = time.perf_counter()
    arrays, tasks, old, producer, extraction, provenance = load_support(
        support_features=support_features, capsule=capsule, expected_capsule_id=expected_capsule_id,
        expected_checkpoint_sha256=expected_checkpoint_sha256, expected_model_seed=expected_model_seed,
        config=dict(algorithm=CACHE_VALIDATION_CONFIG, matrix=config['producer_matrix']))
    chosen = selected_tasks(tasks, config['selection'], old)
    payload = dict(feature_cache_reused=True, checkpoint_loaded=False, native_physical_forward_count_this_run=0,
        feature_extraction_this_run_seconds=0., new_source_payload_bytes=0, new_ground_statistics_bytes=0,
        model_incremental_transfer_bytes=0, support_feature_array_bytes=producer['feature_array_bytes'],
        support_feature_file_bytes=producer['feature_file_bytes'], cache_load_seconds=time.perf_counter()-begin)
    binding = dict(capsule_id=expected_capsule_id, checkpoint_sha256=expected_checkpoint_sha256,
        model_seed=expected_model_seed, channel=deepcopy(manifest['channel']), scenarios=manifest['scenarios'])
    startup = dict(binding, scope=SCOPE, config=config, episodes=len(chosen), producer_episodes=len(tasks),
        support_features=str(support_features), capsule_manifest=str(Path(capsule)/'manifest.json'),
        provenance=provenance, payload_audit=payload, query_rows_used=0, source_rows_used=0,
        query_iq_access=False, truth_read=False, checkpoint_loaded=False, encoder_updated=False,
        adapted_state_inherited=False, optimizer_steps=0, adapter_training=False,
        actual_A=None, adapted_B=None, unavailable_reason='Only original LocalRidge B0/C0; no ground A or adapted B is evaluated',
        source_validation=None, source_validation_reason='No source samples or feature banks accessed',
        learning_rate=None, learning_rate_reason='Closed-form LocalRidge; no optimizer',
        argv=sys.argv, pid=os.getpid(), python=sys.executable,
        blas_environment={key: os.environ.get(key) for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')})
    out.mkdir(parents=True, exist_ok=False); write(out/'startup.json', startup)
    print(json.dumps(dict(event='STARTUP', **startup), allow_nan=False), flush=True)
    totals = dict(episodes=0, k1_episodes=0, oof_episodes=0, proxy_anchor_count=0,
                  sequence_paths=0, head_fit_count=0, factorization_count=0)
    try:
        with (out/'fit_trace.jsonl').open('x', encoding='utf-8') as trace, \
             (out/'compact.jsonl').open('x', encoding='utf-8') as compact, \
             (out/'compact.csv').open('x', encoding='utf-8', newline='') as csvfile, \
             (out/'fit_stages.jsonl').open('x', encoding='utf-8') as stagefile:
            writer = None
            for split, positions, labels in chosen:
                current_stages = []
                def log(stage):
                    row = dict(scalars(stage), split_id=split['split_id'])
                    current_stages.append(row)
                    stagefile.write(json.dumps(row, allow_nan=False)+'\n'); stagefile.flush()
                    print(json.dumps(dict(event='LOCAL_RIDGE_REGISTRATION_FIT', **row), allow_nan=False), flush=True)
                try:
                    audit = probe_registration(**{key: value[positions] for key, value in arrays.items()},
                        support_labels=labels, support_ids=split['support_ids'], classes=split['registered_classes'],
                        old_classes=old, log_callback=log)
                except Exception as exc:
                    write(out/'probe_failed.json', _safe(dict(status='REGISTRATION_DIAGNOSTIC_FAILED', **binding,
                        split_identity=split_identity(split, old), error_type=type(exc).__name__, error=str(exc),
                        failure_context=getattr(exc, 'registration_context', {}), completed_episodes=totals['episodes'],
                        completed_current_stages=current_stages, trace_files=['fit_trace.jsonl', 'fit_stages.jsonl'],
                        query_rows_used=0, source_rows_used=0)))
                    raise
                record = dict(audit, **split_identity(split, old), scope=SCOPE, query_rows_used=0, source_rows_used=0)
                trace.write(json.dumps(record, allow_nan=False)+'\n'); trace.flush()
                small = compact_record(record)
                compact.write(json.dumps(small, allow_nan=False)+'\n'); compact.flush()
                if writer is None:
                    writer = csv.DictWriter(csvfile, fieldnames=list(small)); writer.writeheader()
                writer.writerow(csv_record(small)); csvfile.flush()
                for key in ('sequence_paths', 'head_fit_count', 'factorization_count'): totals[key] += audit[key]
                totals['episodes'] += 1; totals['k1_episodes'] += int(split['k'] == 1)
                totals['oof_episodes'] += int(split['k'] > 1); totals['proxy_anchor_count'] += small['proxy_anchor_count']
                print(json.dumps(dict(event='SUPPORT_PARENT_COMPLETE', **small, completed=totals['episodes'], total=len(chosen)), allow_nan=False), flush=True)
    finally:
        stage_csv_from_jsonl(out/'fit_stages.jsonl', out/'fit_stages.csv')
    marker = dict(binding, **totals, status=STATUS, scope=SCOPE, algorithm=DIAGNOSTIC_CONFIG,
        selection=config['selection'], producer_matrix=config['producer_matrix'], payload_audit=payload,
        query_rows_used=0, source_rows_used=0, truth_read=False, optimizer_steps=0, persistent_state_bytes=0,
        wall_seconds=time.perf_counter()-begin, peak_process_rss_bytes=peak_rss())
    write(out/'probe_complete.json', marker)
    return marker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('support-features', 'capsule', 'output', 'config', 'expected-capsule-id', 'expected-checkpoint-sha256'):
        parser.add_argument('--'+key, required=True)
    parser.add_argument('--expected-model-seed', type=int, required=True)
    args = vars(parser.parse_args()); args['config'] = read(args['config']); evaluate(**args)


if __name__ == '__main__': main()
