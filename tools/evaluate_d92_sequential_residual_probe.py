"""Fixed support-only three-path sequential residual pilot; no query input API."""
import argparse
from copy import deepcopy
import csv
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'code'), str(ROOT/'tools')]
from cvsrffi import d92_branch_interaction as interaction
from cvsrffi.d92_branch_local_ridge import fit_branch_local_ridge
from cvsrffi.d92_branch_support_probe import FROZEN_CONFIG as CACHE_VALIDATION_CONFIG
from cvsrffi.d92_sequential_residual_head import (
    FROZEN_CONFIG as RESIDUAL_CONFIG, prepare_residual_training, train_residual_head,
)
from evaluate_d92_registration_diagnostic import (
    CHANNEL, SCENARIOS, BRANCHES, split_identity, selected_tasks, validate_selection,
    diagnose_evidence, measured_metrics as baseline_metrics, _safe,
)
from evaluate_d92_branch_support_probe import load_support, read, write, check, scalars, csv_record, peak_rss
from evaluate_d92_branch_local_ridge_probe import stage_csv_from_jsonl
from run_d92_sequential_residual_probe import PROBE_CONFIG, STATUS

SCOPE = 'SUPPORT_ONLY_SEQUENTIAL_RESIDUAL_OOF_AND_PROXY_NOT_QUERY_EVALUATION'
PATHS = ('R0', 'R_reset', 'R_seq')
METRICS = ('A_old_accuracy', 'B0_old_accuracy', 'B_old_accuracy', 'C_old_columns_accuracy',
    'C_old_accuracy', 'C_new_accuracy', 'C_h', 'adaptation_gain_B_minus_A',
    'support_adaptation_B_minus_B0', 'old_order_change', 'new_competition_loss',
    'total_old_accuracy_drop', 'C_abs_new_old_gap', 'C_new_minus_old',
    'C_old_minus_R0', 'C_new_minus_R0', 'old_order_recovery_fraction',
    'total_recovery_fraction', 'old_winner_changed_fraction', 'old_displaced_by_new_fraction') + tuple(
        'correctness_B_Cold_C_'+bits for bits in ('000', '100', '010', '110', '011', '111'))
COUNTERS = ('episodes', 'k1_episodes', 'oof_episodes', 'proxy_anchor_count',
    'sequence_paths', 'head_fit_count', 'factorization_count', 'residual_training_stages', 'optimizer_steps')


def path_metrics(diagnostic, reference):
    old, new = diagnostic['C_full_acc'], diagnostic['Cnew_acc']
    base = baseline_metrics(diagnostic)
    result = dict(A_old_accuracy=None, B0_old_accuracy=reference['B_acc'], B_old_accuracy=diagnostic['B_acc'],
        C_old_columns_accuracy=diagnostic['C_old_columns_acc'], C_old_accuracy=old, C_new_accuracy=new,
        C_h=diagnostic['C_harmonic_mean'], adaptation_gain_B_minus_A=None,
        support_adaptation_B_minus_B0=diagnostic['B_acc']-reference['B_acc'],
        C_abs_new_old_gap=None if new is None else abs(new-old), C_new_minus_old=None if new is None else new-old,
        C_old_minus_R0=old-reference['C_full_acc'],
        C_new_minus_R0=None if new is None else new-reference['Cnew_acc'],
        **{key: base[key] for key in ('old_order_change', 'new_competition_loss', 'total_old_accuracy_drop',
            'old_order_recovery_fraction', 'total_recovery_fraction', 'old_winner_changed_fraction', 'old_displaced_by_new_fraction')})
    for bits in ('000', '100', '010', '110', '011', '111'):
        result['correctness_B_Cold_C_'+bits] = sum(
            ''.join(str(int(row[key])) for key in ('B_correct', 'C_old_columns_correct', 'C_full_correct')) == bits
            for row in diagnostic['old_records']) / diagnostic['old_held_count']
    return result


def assess_paths(evidence, old):
    diagnostics = {name: diagnose_evidence(value, old) for name, value in evidence.items()}
    return {name: dict(diagnostic=value, metrics=path_metrics(value, diagnostics['R0']))
            for name, value in diagnostics.items()}


def parent_mean(trials):
    return {name: {metric: None if trials[0]['paths'][name]['metrics'][metric] is None else
        sum(t['paths'][name]['metrics'][metric] for t in trials)/len(trials) for metric in METRICS} for name in PATHS}


def pooled_assess(entries, labels, classes, old):
    evidence = {}
    for name in PATHS:
        brows, crows = {}, {}
        for entry in entries:
            check(not set(brows).intersection(entry['b_ids']) and not set(crows).intersection(entry['c_ids']),
                  'Duplicate physical OOF held ID')
            brows.update(zip(entry['b_ids'], entry['paths'][name]['b_scores']))
            crows.update(zip(entry['c_ids'], entry['paths'][name]['c_scores']))
        evidence[name] = dict(b_ids=sorted(brows), b_classes=old, b_scores=[brows[i] for i in sorted(brows)],
            c_ids=sorted(crows), c_classes=classes, c_scores=[crows[i] for i in sorted(crows)], held_labels=labels)
    return assess_paths(evidence, old)


def probe_sequential_residual(*, z_id, fft, t_emb, f_emb, pa_local, support_labels,
                             support_ids, classes, old_classes, log_callback=None, step_callback=None,
                             start_callback=None):
    started = time.perf_counter()
    b, a, labels, ids, canonical, _, old, k = interaction._prepare(
        z_id, fft, t_emb, f_emb, pa_local, support_labels, support_ids, classes, old_classes)
    check(bool(old), 'Old registry required')
    classes, old = list(canonical), sorted(old)
    order = np.asarray(sorted(range(len(ids)), key=lambda i: support_ids[i]))
    raw = {key: np.asarray(value)[order] for key, value in zip(BRANCHES, (z_id, fft, t_emb, f_emb, pa_local))}
    label_by_id = {pid: classes[int(y)] for pid, y in zip(ids, labels)}
    is_old = np.asarray([label_by_id[pid] in old for pid in ids])
    folds = 0 if k == 1 else min(k, 3)
    result = dict(classes=classes, old_classes=old, support_count=len(ids), old_class_count=len(old),
        new_class_count=len(classes)-len(old), fold_count=folds, physical_fold_assignment=[],
        numerical=interaction._diagonal_stats(b, a), folds=[], oof=None, oneshot_proxy=None,
        sequence_paths=0, head_fit_count=0, factorization_count=0, residual_training_stages=0,
        optimizer_steps=0, persistent_state_bytes=0,
        heldout_unavailable_reason='K1_NO_INDEPENDENT_PHYSICAL_HOLDOUT' if k == 1 else None)
    if k == 1:
        result['fit_seconds'] = time.perf_counter()-started
        return result
    positions = [np.flatnonzero(labels == i) for i in range(len(classes))]
    assignments = np.full(len(ids), -1, dtype=int)
    for indices in positions: assignments[indices] = np.arange(k) % folds
    result['physical_fold_assignment'] = [dict(physical_id=pid, class_id=label_by_id[pid], fold=int(assignments[i])) for i, pid in enumerate(ids)]
    completed = []

    def path(keep, scope, index):
        reuse = classes == old
        entry = dict(scope=scope, fold=index if scope == 'support_oof' else None,
            trial=index if scope == 'support_oneshot_proxy' else None, parent_k=k,
            train_k=int(keep.sum())//len(classes), held_k=int((~keep).sum())//len(classes),
            b_training_ids=[pid for i, pid in enumerate(ids) if keep[i] and is_old[i]],
            c_training_ids=[pid for i, pid in enumerate(ids) if keep[i]],
            b_ids=[pid for i, pid in enumerate(ids) if not keep[i] and is_old[i]],
            c_ids=[pid for i, pid in enumerate(ids) if not keep[i]], b_classes=old, c_classes=classes,
            held_labels={pid: label_by_id[pid] for i, pid in enumerate(ids) if not keep[i]},
            stages=[], residual_stages=[], c_reuses_b0=reuse, c_reuses_b_residual=reuse, paths={})
        coords = {key: entry[key] for key in ('scope', 'fold', 'trial', 'parent_k', 'train_k')}
        bases, prepared, residuals = {}, {}, {}
        try:
            for name, train, held, registry in (
                    ('B0', keep & is_old, ~keep & is_old, old), ('C0', keep, ~keep, classes)):
                if name == 'C0' and reuse:
                    bases[name] = bases['B0']
                    continue
                tids = [pid for i, pid in enumerate(ids) if train[i]]
                ys = np.asarray([registry.index(label_by_id[pid]) for pid in tids], dtype=np.int64)
                fit_start = time.perf_counter()
                state = fit_branch_local_ridge(**{key: value[train] for key, value in raw.items()},
                    support_labels=ys, support_ids=tids, classes=registry, old_classes=old, arm='local_ridge')
                base_audit = state.audit_dict()
                audit = base_audit['final_fit']
                audit['persistent_state_bytes'] = base_audit['persistent_state_bytes']
                result['head_fit_count'] += 1; result['factorization_count'] += audit['factorization_calls']
                score_start = time.perf_counter()
                training_scores = state.score(**{key: value[train] for key, value in raw.items()})
                train_score_seconds = time.perf_counter()-score_start
                score_start = time.perf_counter()
                held_scores = state.score(**{key: value[held] for key, value in raw.items()})
                held_score_seconds = time.perf_counter()-score_start
                # All consumers receive the same immutable numerical base arrays.
                training_scores.setflags(write=False); held_scores.setflags(write=False)
                bases[name] = dict(state=state, training_scores=training_scores, held_scores=held_scores,
                    train=train, held=held, registry=registry, labels=ys, ids=tids, audit=audit)
                audit.update(state=name, **coords, training_physical_ids=tids,
                    all_states_estimated_from_trainfold_only=True, held_physical_count=int(held.sum()),
                    class_count=len(registry), score_seconds=train_score_seconds+held_score_seconds,
                    train_score_seconds=train_score_seconds, held_score_seconds=held_score_seconds,
                    fit_and_score_seconds=time.perf_counter()-fit_start, learning_rate=None,
                    learning_rate_reason='Closed-form frozen LocalRidge; no optimizer',
                    source_validation=None, source_validation_reason='No source samples or features accessed')
                entry['stages'].append(audit)
                if log_callback: log_callback(dict(event='BASE_FIT', **audit))
            for name, base_name, inherited in (('B', 'B0', None), ('C_reset', 'C0', None), ('C_seq', 'C0', 'B')):
                if name != 'B' and reuse:
                    residuals[name] = residuals['B']
                    continue
                base = bases[base_name]
                context = dict(state=name, **coords, inheritance_source=inherited,
                    training_physical_ids=base['ids'], train_physical_count=len(base['ids']),
                    held_physical_count=int(base['held'].sum()), class_count=len(base['registry']))
                if inherited:
                    old_rows_in_c = [i for i, pid in enumerate(base['ids']) if label_by_id[pid] in old]
                    check([base['ids'][i] for i in old_rows_in_c] == bases['B0']['ids'],
                          'Inherited old physical ID order differs from B')
                    for key, value in raw.items():
                        check(np.array_equal(value[base['train']][old_rows_in_c], value[bases['B0']['train']]),
                              'Inherited old support feature mismatch: '+key)
                context['entry_old_feature_equality_checked'] = bool(inherited)
                def on_step(row):
                    if step_callback: step_callback(dict(row, **context, source_validation=None,
                        source_validation_reason='SOURCE_ACCESS_FORBIDDEN'))
                fresh_preparation = base_name not in prepared
                if fresh_preparation:
                    prepared[base_name] = prepare_residual_training(z_id=raw['z_id'][base['train']],
                        base_scores=base['training_scores'], support_labels=base['labels'],
                        support_ids=base['ids'], classes=base['registry'])
                if start_callback:
                    start_callback(dict(context, event='RESIDUAL_STAGE_START', q=prepared[base_name].q,
                        U_shape=[160, 8], V_shape=[8, len(base['registry'])],
                        zero_initialized_new_columns=len(base['registry'])-(len(old) if inherited else 0),
                        bandwidth_tau=base['audit']['bandwidth_tau'],
                        interaction_centered_trace=base['audit']['interaction_centered_trace'],
                        config=PROBE_CONFIG, source_validation=None, source_validation_reason='SOURCE_ACCESS_FORBIDDEN'))
                state = train_residual_head(prepared[base_name],
                    inherited=None if inherited is None else residuals[inherited]['state'],
                    context=context, log_callback=on_step)
                audit = state.audit_dict()
                audit.update(**context, all_states_estimated_from_trainfold_only=True, source_validation=None,
                    source_validation_reason='No source samples or features accessed', score_seconds=None,
                    actual_prepare_seconds=prepared[base_name].prepare_seconds if fresh_preparation else 0.,
                    prepare_shared_with=None if fresh_preparation else 'C_reset')
                # Commit successful training accounting before held inference can fail.
                residuals[name] = dict(state=state, scores=None)
                entry['residual_stages'].append(audit)
                result['residual_training_stages'] += 1
                result['optimizer_steps'] += audit['optimizer_steps']
                result['persistent_state_bytes'] = max(result['persistent_state_bytes'], audit['persistent_state_bytes'])
                score_start = time.perf_counter()
                scores = state.score(z_id=raw['z_id'][base['held']], base_scores=base['held_scores'])
                audit['score_seconds'] = time.perf_counter()-score_start
                residuals[name]['scores'] = scores
                if log_callback: log_callback(dict(event='RESIDUAL_FIT', **audit))
            common = {key: entry[key] for key in ('b_ids', 'b_classes', 'c_ids', 'c_classes', 'held_labels')}
            evidence = {
                'R0': dict(common, b_scores=bases['B0']['held_scores'].tolist(), c_scores=bases['C0']['held_scores'].tolist()),
                'R_reset': dict(common, b_scores=residuals['B']['scores'].tolist(), c_scores=residuals['C_reset']['scores'].tolist()),
                'R_seq': dict(common, b_scores=residuals['B']['scores'].tolist(), c_scores=residuals['C_seq']['scores'].tolist())}
            assessments = assess_paths(evidence, old)
            entry['paths'] = {name: dict(b_scores=value['b_scores'], c_scores=value['c_scores'], **assessments[name])
                              for name, value in evidence.items()}
        except Exception as exc:
            exc.registration_context = _safe(dict(current_path=entry,
                failed_fit=exc.audit_dict() if callable(getattr(exc, 'audit_dict', None)) else {},
                completed_residual_states={name: dict(U=value['state'].U, V=value['state'].V,
                    q=value['state'].q, classes=value['state'].classes, audit=value['state'].audit_dict())
                    for name, value in residuals.items()},
                completed_paths=completed, counters={key: result[key] for key in
                    ('sequence_paths', 'head_fit_count', 'factorization_count', 'residual_training_stages', 'optimizer_steps')}))
            raise
        result['sequence_paths'] += 1
        completed.append(entry)
        return entry

    for fold in range(folds): result['folds'].append(path(assignments != fold, 'support_oof', fold))
    result['oof'] = dict(paths=pooled_assess(result['folds'], label_by_id, classes, old),
                         aggregation='one_record_per_physical_held_id')
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
        'factorization_count', 'residual_training_stages', 'optimizer_steps', 'fit_seconds',
        'persistent_state_bytes', 'heldout_unavailable_reason')
    return dict({key: record[key] for key in keys},
        oof=None if record['oof'] is None else {name: record['oof']['paths'][name]['metrics'] for name in PATHS},
        proxy=None if record['oneshot_proxy'] is None else record['oneshot_proxy']['parent_mean_metrics'],
        proxy_anchor_count=0 if record['oneshot_proxy'] is None else record['oneshot_proxy']['trial_count'],
        scope=SCOPE, query_rows_used=0, source_rows_used=0)


def evaluate(*, support_features, capsule, output, config, expected_capsule_id,
             expected_checkpoint_sha256, expected_model_seed):
    out = Path(output)
    if out.exists(): raise FileExistsError(out)
    check(set(config) == {'algorithm', 'producer_matrix', 'selection'} and config['algorithm'] == PROBE_CONFIG,
          'Frozen sequential residual configuration mismatch')
    check(PROBE_CONFIG == RESIDUAL_CONFIG, 'Frozen residual code/config mismatch')
    validate_selection(config['selection'])
    manifest = read(Path(capsule)/'manifest.json')
    check(all(manifest.get('channel', {}).get(key) == value for key, value in CHANNEL.items())
          and manifest.get('scenarios') == SCENARIOS, 'Practical residual post_sync/noeq metadata mismatch')
    begin = time.perf_counter()
    arrays, tasks, old, producer, extraction, provenance = load_support(
        support_features=support_features, capsule=capsule, expected_capsule_id=expected_capsule_id,
        expected_checkpoint_sha256=expected_checkpoint_sha256, expected_model_seed=expected_model_seed,
        config=dict(algorithm=CACHE_VALIDATION_CONFIG, matrix=config['producer_matrix']))
    check(len(old) == 6, 'Fixed pilot requires six old classes')
    chosen = selected_tasks(tasks, config['selection'], old)
    payload = dict(feature_cache_reused=True, checkpoint_loaded=False, native_physical_forward_count_this_run=0,
        feature_extraction_this_run_seconds=0., new_source_payload_bytes=0, new_ground_statistics_bytes=0,
        model_incremental_transfer_bytes=0, support_feature_array_bytes=producer['feature_array_bytes'],
        support_feature_file_bytes=producer['feature_file_bytes'], cache_load_seconds=time.perf_counter()-begin,
        residual_deployment_package_bytes=None, residual_incremental_transfer_bytes=None,
        unmeasured_reason='Support pilot retains audit outputs; no deployment package serialized or transmitted')
    binding = dict(capsule_id=expected_capsule_id, checkpoint_sha256=expected_checkpoint_sha256,
        model_seed=expected_model_seed, channel=deepcopy(manifest['channel']), scenarios=manifest['scenarios'])
    startup = dict(binding, scope=SCOPE, config=config, episodes=len(chosen), producer_episodes=len(tasks),
        support_features=str(support_features), provenance=provenance, payload_audit=payload,
        query_rows_used=0, source_rows_used=0, query_iq_access=False, truth_read=False,
        checkpoint_loaded=False, encoder_updated=False, adapted_state_inherited=True, adapter_training=True,
        actual_A=None, actual_A_unavailable_reason='No legal matched ground predictor scores in support pilot',
        source_validation=None, source_validation_reason='No source samples or feature banks accessed',
        argv=sys.argv, pid=os.getpid(), python=sys.executable, hardware=dict(platform=platform.platform(),
        processor=platform.processor(), cpu_count=os.cpu_count(), dtype='float64', gpu_use=False),
        blas_environment={key: os.environ.get(key) for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')})
    out.mkdir(parents=True, exist_ok=False); write(out/'startup.json', startup)
    print(json.dumps(dict(event='STARTUP', **startup), allow_nan=False), flush=True)
    totals = dict.fromkeys(COUNTERS, 0); peak_state = 0
    try:
        with (out/'fit_trace.jsonl').open('x', encoding='utf-8') as trace, \
             (out/'compact.jsonl').open('x', encoding='utf-8') as compact, \
             (out/'compact.csv').open('x', encoding='utf-8', newline='') as csvfile, \
             (out/'fit_stages.jsonl').open('x', encoding='utf-8') as stages, \
             (out/'training_steps.jsonl').open('x', encoding='utf-8') as steps, \
             (out/'training_steps_compact.jsonl').open('x', encoding='utf-8') as smallsteps, \
             (out/'training.log').open('x', encoding='utf-8') as textlog:
            writer = None
            for split, positions, labels in chosen:
                def log(stage):
                    row = dict(scalars(stage), split_id=split['split_id'])
                    stages.write(json.dumps(row, allow_nan=False)+'\n'); stages.flush()
                    line = json.dumps(row, allow_nan=False)
                    textlog.write(line+'\n'); textlog.flush(); print(line, flush=True)
                def step(row):
                    full = dict(row, split_id=split['split_id'])
                    steps.write(json.dumps(full, allow_nan=False)+'\n'); steps.flush()
                    compact_step = scalars(full)
                    smallsteps.write(json.dumps(compact_step, allow_nan=False)+'\n'); smallsteps.flush()
                    line = 'SUPPORT_RESIDUAL_STEP '+json.dumps(compact_step, allow_nan=False)
                    textlog.write(line+'\n'); textlog.flush(); print(line, flush=True)
                def stage_start(row):
                    line = json.dumps(dict(row, split_id=split['split_id']), allow_nan=False)
                    textlog.write(line+'\n'); textlog.flush(); print(line, flush=True)
                try:
                    audit = probe_sequential_residual(**{key: value[positions] for key, value in arrays.items()},
                        support_labels=labels, support_ids=split['support_ids'], classes=split['registered_classes'],
                        old_classes=old, log_callback=log, step_callback=step, start_callback=stage_start)
                except Exception as exc:
                    write(out/'probe_failed.json', _safe(dict(status='SEQUENTIAL_RESIDUAL_PROBE_FAILED', **binding,
                        split_identity=split_identity(split, old), error_type=type(exc).__name__, error=str(exc),
                        failure_context=getattr(exc, 'registration_context', {}), completed_episodes=totals['episodes'],
                        query_rows_used=0, source_rows_used=0)))
                    raise
                record = dict(audit, **split_identity(split, old), scope=SCOPE, query_rows_used=0, source_rows_used=0)
                trace.write(json.dumps(record, allow_nan=False)+'\n'); trace.flush()
                small = compact_record(record)
                compact.write(json.dumps(small, allow_nan=False)+'\n'); compact.flush()
                if writer is None:
                    writer = csv.DictWriter(csvfile, fieldnames=list(small)); writer.writeheader()
                writer.writerow(csv_record(small)); csvfile.flush()
                for key in COUNTERS[4:]: totals[key] += audit[key]
                totals['episodes'] += 1; totals['k1_episodes'] += int(split['k'] == 1)
                totals['oof_episodes'] += int(split['k'] > 1); totals['proxy_anchor_count'] += small['proxy_anchor_count']
                peak_state = max(peak_state, audit['persistent_state_bytes'])
                print(json.dumps(dict(event='SUPPORT_PARENT_COMPLETE', **small, completed=totals['episodes'], total=len(chosen)), allow_nan=False), flush=True)
    finally:
        for name in ('fit_stages', 'training_steps_compact'):
            if (out/(name+'.jsonl')).exists(): stage_csv_from_jsonl(out/(name+'.jsonl'), out/(name+'.csv'))
    marker = dict(binding, **totals, status=STATUS, scope=SCOPE, algorithm=PROBE_CONFIG,
        selection=config['selection'], producer_matrix=config['producer_matrix'], payload_audit=payload,
        query_rows_used=0, source_rows_used=0, truth_read=False, persistent_state_bytes=peak_state,
        persistent_state_scope='maximum_one_residual_state_numeric_arrays_not_concurrent_audit_or_base_state',
        peak_gpu_memory_bytes=None, gpu_memory_reason='CPU-only run',
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
