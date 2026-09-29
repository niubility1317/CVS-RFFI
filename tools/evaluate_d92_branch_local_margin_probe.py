"""Evaluate four fresh fixed arms from existing physical support-only caches."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'code'), str(ROOT / 'tools')]
from cvsrffi.d92_branch_local_margin import FROZEN_CONFIG, probe_branch_local_margin
from cvsrffi.d92_branch_support_probe import FROZEN_CONFIG as CACHE_VALIDATION_CONFIG
from evaluate_d92_branch_support_probe import (
    load_support, read, write, check, scalars, csv_record, peak_rss, BRANCHES,
)

SCOPE = 'SUPPORT_OOF_AND_SUPPORT_ONESHOT_PROXY_NOT_QUERY_EVALUATION'
ARMS = ('branch_ridge', 'interaction_ridge', 'local_ridge', 'local_margin')
PAIRS = ('local_ridge_minus_branch_ridge', 'local_ridge_minus_interaction_ridge',
         'interaction_ridge_minus_branch_ridge', 'local_margin_minus_branch_ridge',
         'local_margin_minus_interaction_ridge', 'local_margin_minus_local_ridge')


def compact_proxy(audit, split_id):
    """Keep exact physical trial evidence and add parent-equal scalar means."""
    proxy = audit['oneshot_proxy']
    audit['proxy_anchor_count'] = 0 if proxy is None else proxy['trial_count']
    if proxy is not None:
        proxy.update(parent_split_id=split_id, diagnostic='support_oneshot_proxy',
                     evidence_schema='anchor_confusion_class_nll_and_exact_physical_mapping_v1')
        proxy['parent_mean_metrics'] = {
            arm: {key: (sum(t['oof'][arm]['metrics'][key] for t in proxy['trials'])
                        / len(proxy['trials'])
                        if proxy['trials'][0]['oof'][arm]['metrics'][key] is not None else None)
                  for key in ('accuracy', 'macro_accuracy', 'old_accuracy',
                              'new_accuracy', 'h', 'macro_nll')}
            for arm in ARMS}
    return audit


def stage_csv_from_jsonl(path, destination):
    """Use all measured stage columns, including diagnostics first seen on failure."""
    columns = set()
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            columns.update(json.loads(line))
    with destination.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted(columns))
        if columns:
            writer.writeheader()
            with path.open(encoding='utf-8') as source:
                for line in source:
                    row = json.loads(line)
                    writer.writerow(csv_record({key: row.get(key) for key in columns}))


def evaluate(*, support_features, capsule, output, config, expected_capsule_id,
             expected_checkpoint_sha256, expected_model_seed):
    out = Path(output)
    if out.exists():
        raise FileExistsError(out)
    check(set(config) == {'algorithm', 'matrix'} and config['algorithm'] == FROZEN_CONFIG,
          'Frozen local margin configuration mismatch')
    started = time.perf_counter()
    arrays, tasks, old, producer, extraction, provenance = load_support(
        support_features=support_features, capsule=capsule,
        expected_capsule_id=expected_capsule_id,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_model_seed=expected_model_seed,
        config=dict(algorithm=CACHE_VALIDATION_CONFIG, matrix=config['matrix']))
    load_seconds = time.perf_counter() - started
    out.mkdir(parents=True, exist_ok=False)
    payload = dict(existing_model_file_bytes=producer['model_file_bytes'],
        model_file_bytes_scope='Existing full training checkpoint package; not loaded or transferred by this diagnostic',
        support_feature_array_bytes=producer['feature_array_bytes'],
        support_feature_file_bytes=producer['feature_file_bytes'], cache_load_seconds=load_seconds,
        feature_cache_reused=True, feature_extraction_this_run_seconds=0.,
        native_physical_forward_count_this_run=0, new_source_payload_bytes=0,
        new_ground_statistics_bytes=0, model_incremental_transfer_bytes=0,
        checkpoint_loaded=False)
    startup = dict(scope=SCOPE, argv=sys.argv, python=sys.executable, pid=os.getpid(),
        config=config, capsule_id=expected_capsule_id,
        checkpoint_sha256=expected_checkpoint_sha256, model_seed=expected_model_seed,
        provenance=provenance, support_features=str(support_features),
        capsule_manifest=str(Path(capsule) / 'manifest.json'), episodes=len(tasks),
        query_rows_used=0, source_rows_used=0, query_iq_access=False, truth_read=False,
        adapted_state_inherited=False, cross_row_adapted_state_reuse=False,
        checkpoint_loaded=False, encoder_updated=False, payload_audit=payload,
        source_validation=None, source_validation_reason='No source samples or source feature banks accessed',
        learning_rate=None, epoch=None, optimizer_steps=0, device='cpu',
        learning_rate_reason='Exact row-block minimization; no gradient learning rate',
        epoch_reason='Candidate iterations are measured as complete sweeps',
        unavailable_reason='No source validation or deployment head in support diagnostic',
        blas_environment={k: os.environ.get(k) for k in
                          ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')})
    write(out / 'startup.json', startup)
    print(json.dumps(dict(event='STARTUP', **startup), allow_nan=False), flush=True)
    keys = ('support_count', 'fold_count', 'numerical', 'oof', 'paired', 'fit_seconds',
            'factorization_count', 'standard_factorization_count', 'proxy_anchor_count',
            'oneshot_proxy', 'optimizer_steps', 'persistent_state_bytes',
            'heldout_unavailable_reason')
    fields = ['split_id', 'receiver', 'scenario', 'k', 'new_count', 'support_seed',
              'completed', 'total', 'scope', 'classes', *keys, 'fit_call_seconds',
              'log_write_seconds', 'total_seconds', 'peak_process_rss_bytes',
              'query_rows_used', 'source_rows_used']
    factorizations = standard_factorizations = proxy_anchors = k1_count = optimizer_steps = 0
    try:
        with (out / 'fit_trace.jsonl').open('x', encoding='utf-8') as trace, \
             (out / 'compact.jsonl').open('x', encoding='utf-8') as compact, \
             (out / 'compact.csv').open('x', encoding='utf-8', newline='') as csvfile, \
             (out / 'fit_stages.jsonl').open('x', encoding='utf-8') as stages, \
             (out / 'solver_sweeps.jsonl').open('x', encoding='utf-8') as sweeps, \
             (out / 'solver_sweeps.log').open('x', encoding='utf-8') as sweep_text:
            writer = csv.DictWriter(csvfile, fieldnames=fields)
            writer.writeheader()

            def log_stages(values, split_id):
                for stage in values:
                    row = dict(scalars(stage), split_id=split_id)
                    stages.write(json.dumps(row, allow_nan=False) + '\n')
                    stages.flush()
                    print(json.dumps(dict(event='ANALYTICAL_SUPPORT_FIT', **row),
                                     allow_nan=False), flush=True)

            for number, (split, positions, labels) in enumerate(tasks, 1):
                begin = time.perf_counter()
                def log_sweep(value):
                    row = dict(scalars(value), split_id=split['split_id'])
                    row.update(learning_rate=None,
                        learning_rate_reason='Exact row-block minimization; no gradient learning rate',
                        source_validation=None, source_validation_reason='No source sample permission')
                    sweeps.write(json.dumps(row, allow_nan=False) + '\n')
                    sweeps.flush()
                    line = ('[CVS LocalMargin] split={split_id} scope={scope} sweep={sweep} '
                            'steps={optimizer_steps} margin_loss={loss_data} rkhs_penalty={loss_ridge} '
                            'P={primal_objective} D={dual_objective} gap={relative_duality_gap} '
                            'KKT={relative_kkt_residual} LR=N/A source_validation=N/A').format_map(
                                {key: row.get(key, 'N/A') for key in ('split_id','scope','sweep',
                                 'optimizer_steps','loss_data','loss_ridge','primal_objective',
                                 'dual_objective','relative_duality_gap','relative_kkt_residual')})
                    line += ' measured=' + json.dumps(row, sort_keys=True, allow_nan=False)
                    sweep_text.write(line + '\n')
                    sweep_text.flush()
                    print(line, flush=True)
                try:
                    audit = probe_branch_local_margin(
                        **{key: value[positions] for key, value in arrays.items()},
                        support_labels=labels, support_ids=split['support_ids'],
                        classes=split['registered_classes'], old_classes=old, log_callback=log_sweep)
                except Exception as error:
                    details = error.audit_dict() if callable(getattr(error, 'audit_dict', None)) else {}
                    failure = dict(details, status='SUPPORT_PROBE_FAILED', scope=SCOPE,
                        split_id=split['split_id'], parent_k=split['k'],
                        error_type=type(error).__name__, error=str(error),
                        query_rows_used=0, source_rows_used=0,
                        optimizer_steps_completed_episodes=optimizer_steps,
                        completed_episodes=number - 1)
                    # Preserve the core's scope and physical failure context verbatim.
                    failure['failure_context'] = details
                    failure.setdefault('completed_stages', [])
                    write(out / 'probe_failed.json', failure)
                    log_stages(failure['completed_stages'], split['split_id'])
                    print(json.dumps(dict(event='SUPPORT_PROBE_FAILED', **failure),
                                     allow_nan=False), flush=True)
                    raise
                elapsed = time.perf_counter() - begin
                k = split['k']
                folds = 0 if k == 1 else min(k, 3)
                check(audit['k'] == k and audit['support_count'] == len(positions)
                      and set(audit['classes']) == set(split['registered_classes'])
                      and audit['fold_count'] == folds and len(audit['folds']) == folds
                      and audit['factorization_count'] >= 0
                      and type(audit['optimizer_steps']) is int and audit['optimizer_steps'] >= 0
                      and audit['persistent_state_bytes'] == 0, 'Local margin core audit mismatch')
                if k == 1:
                    check(audit['oof'] is None and audit['paired'] is None
                          and audit['oneshot_proxy'] is None
                          and audit['factorization_count'] == 0, 'K1 fabricated holdout or fit')
                    k1_count += 1
                else:
                    check(set(audit['oof']) == set(ARMS) and set(audit['paired']) == set(PAIRS),
                          'Local margin arm/pair mismatch')
                audit = compact_proxy(audit, split['split_id'])
                factorizations += audit['factorization_count']
                optimizer_steps += audit['optimizer_steps']
                standard_factorizations += audit['standard_factorization_count']
                proxy_anchors += audit['proxy_anchor_count']
                coords = {key: split[key] for key in
                          ('split_id', 'receiver', 'scenario', 'k', 'support_seed')}
                coords['new_count'] = len(split['registered_classes']) - len(old)
                record = dict(audit, **{key: value for key, value in coords.items() if key != 'k'},
                    scope=SCOPE, registered_classes=split['registered_classes'],
                    fit_call_seconds=elapsed, query_rows_used=0, source_rows_used=0)
                writing = time.perf_counter()
                trace.write(json.dumps(record, allow_nan=False) + '\n')
                trace.flush()
                all_stages = [stage for fold in audit['folds'] for stage in fold['stages']]
                if audit['oneshot_proxy']:
                    all_stages += [stage for trial in audit['oneshot_proxy']['trials']
                                   for stage in trial['stages']]
                log_stages(all_stages, split['split_id'])
                small = dict(coords, completed=number, total=len(tasks), scope=SCOPE,
                    classes=len(audit['classes']), **{key: scalars(audit[key]) for key in keys},
                    fit_call_seconds=elapsed, log_write_seconds=time.perf_counter() - writing,
                    total_seconds=time.perf_counter() - begin, peak_process_rss_bytes=peak_rss(),
                    query_rows_used=0, source_rows_used=0)
                compact.write(json.dumps(small, allow_nan=False) + '\n')
                compact.flush()
                writer.writerow(csv_record(small))
                csvfile.flush()
                print(json.dumps(dict(event='SUPPORT_EPISODE_COMPLETE', **small),
                                 allow_nan=False), flush=True)
    finally:
        stage_csv_from_jsonl(out / 'fit_stages.jsonl', out / 'fit_stages.csv')
        stage_csv_from_jsonl(out / 'solver_sweeps.jsonl', out / 'solver_sweeps.csv')
    marker = dict(status='SUPPORT_PROBE_COMPLETE', scope=SCOPE, capsule_id=expected_capsule_id,
        checkpoint_sha256=expected_checkpoint_sha256, model_seed=expected_model_seed,
        episodes=len(tasks), k1_episodes=k1_count, oof_episodes=len(tasks) - k1_count,
        algorithm=FROZEN_CONFIG, matrix=config['matrix'], query_rows_used=0, source_rows_used=0,
        truth_read=False, factorization_count=factorizations,
        standard_factorization_count=standard_factorizations, proxy_anchor_count=proxy_anchors,
        optimizer_steps=optimizer_steps, persistent_state_bytes=0, payload_audit=payload,
        wall_seconds=time.perf_counter() - started, peak_process_rss_bytes=peak_rss(),
        peak_process_rss_reason='Linux process high-water RSS; null on unsupported platform')
    write(out / 'probe_complete.json', marker)
    return marker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('support-features', 'capsule', 'output', 'config', 'expected-capsule-id',
                'expected-checkpoint-sha256'):
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--expected-model-seed', type=int, required=True)
    args = vars(parser.parse_args())
    args['config'] = read(args['config'])
    evaluate(**args)


if __name__ == '__main__':
    main()
