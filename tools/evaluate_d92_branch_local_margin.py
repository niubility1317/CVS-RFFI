"""Frozen BranchLocalMargin: support-only fit, independent all-class query scores."""
import argparse
import csv
import importlib
import json
import os
from pathlib import Path
import sys
import time
import types

import numpy as np
from export_d92_branch_features import (CACHE_SCHEMA, FEATURE_CONTRACT, local_core as cache_core, load_features,
    validate_capsule, read, write, check, peak_process_rss)
from d92_orbit_feature_cache import validate_split

ROOT = Path(__file__).resolve().parents[1]
MODE = 'd92_branch_local_margin_registration'
ANALYTIC_REASON = 'Fixed local-kernel margin objective; exact row-block minimization has no gradient learning rate; iterations are full sweeps; no CV or parameter selection'
SOURCE_REASON = 'Auxiliary source samples and source feature banks are prohibited by design'


def local_core():
    """Resolve the frozen local core independently of deployed native imports."""
    name = '_d92_branch_local_margin_prediction_local'
    if name not in sys.modules:
        package = types.ModuleType(name); package.__path__ = [str(ROOT/'code/cvsrffi')]
        sys.modules[name] = package
    return importlib.import_module(name+'.d92_branch_local_margin')


def stable_predictions(scores, classes):
    order = np.asarray(sorted(range(len(classes)), key=lambda index: classes[index]))
    return order[np.argmax(scores[:, order], axis=1)]


def scalar_record(value):
    """Keep measured scalar/statistic trees, omitting physical ID lists/arrays."""
    return {k:scalar_record(v) if isinstance(v, dict) else v for k,v in value.items()
            if isinstance(v, dict) or v is None or isinstance(v, (str, int, float, bool))}


def csv_record(value):
    return {k:'N/A' if v is None else json.dumps(v, sort_keys=True, allow_nan=False) if isinstance(v, (dict, list)) else v for k,v in value.items()}


def write_stage_csv(source, destination):
    columns = set()
    with source.open(encoding='utf-8') as stream:
        for line in stream:
            columns.update(json.loads(line))
    with destination.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted(columns))
        if columns:
            writer.writeheader()
            with source.open(encoding='utf-8') as records:
                for line in records:
                    record = json.loads(line)
                    writer.writerow(csv_record({key:record.get(key) for key in columns}))


def predict(*, row_root, capsule, output, config, expected_capsule_id, expected_checkpoint_sha256, branch_features):
    out, capsule = Path(output), Path(capsule)
    if out.exists(): raise FileExistsError(out)
    core = local_core()
    check(set(config) == {'algorithm'} and config['algorithm'] == core.FROZEN_CONFIG, 'Frozen BranchLocalMargin config mismatch')
    loading = time.perf_counter()
    # The unchanged producer was registered under BranchRidge. Its algorithm is
    # used only to bind the raw frozen cache, never to choose or fit this head.
    arrays, ids, producer, previous, provenance = load_features(branch_features=branch_features, capsule=capsule,
        row_root=row_root, expected_capsule_id=expected_capsule_id, expected_checkpoint_sha256=expected_checkpoint_sha256,
        config=dict(algorithm=cache_core().FROZEN_CONFIG))
    cache_load_seconds = time.perf_counter()-loading
    manifest = validate_capsule(capsule, expected_capsule_id); old = producer['classes']
    paths = sorted((capsule/'splits').glob('*.json'))
    check(len(paths) == manifest['split_count'] and paths, 'Incomplete split matrix')
    splits = [read(path) for path in paths]
    check(len({v['split_id'] for v in splits}) == len(splits) and all(p.stem == v['split_id'] for p,v in zip(paths,splits)), 'Split ID mismatch')
    tasks = [validate_split(split, manifest, ids, old) for split in splits]
    out.mkdir(parents=True, exist_ok=False)
    payload = dict(new_source_payload_bytes=0, new_ground_statistics_bytes=0,
        model_file_bytes=producer['model_file_bytes'], model_file_bytes_scope='Existing full training checkpoint package; not loaded or transferred in this run',
        model_already_deployed=None, model_incremental_transfer_bytes=0, model_deployment_unknown_reason='Deployment state unknown; existing feature cache reused',
        feature_array_bytes=producer['feature_array_bytes'], feature_file_bytes=producer['feature_file_bytes'],
        existing_feature_extraction_timing=producer['timing'],
        existing_native_physical_forward_count=producer['native_physical_forward_count'],
        feature_storage_scope='Existing raw frozen received features; no fitted target state',
        feature_cache_reused=True, cache_load_seconds=cache_load_seconds,
        feature_extraction_this_run_seconds=0., native_physical_forward_count=0,
        native_batch_calls=0, smoke_forward_count=0, smoke_batch_calls=0,
        native_total_physical_forward_count=0, native_total_batch_calls=0,
        support_cache_reuse_count=0,
        checkpoint_loaded=False)
    startup = dict(argv=sys.argv, python=sys.executable, pid=os.getpid(), config=config, capsule_id=expected_capsule_id,
        checkpoint_sha256=expected_checkpoint_sha256, model_seed=previous['seed'], provenance=provenance,
        feature_cache_schema=CACHE_SCHEMA, feature_contract=FEATURE_CONTRACT, branch_features=str(branch_features),
        query_fit_access=False, truth_read=False, source_data_access=False, ground_summary_access=False,
        adapted_state_inherited=False, cross_row_adapted_state_reuse=False, encoder_updated=False,
        payload_audit=payload, source_validation=None, source_validation_reason=SOURCE_REASON,
        learning_rate=None, epoch=None, gradient_norm=None, unavailable_reason=ANALYTIC_REASON,
        startup_gradient_reason='Not yet fitted; each sweep and final_fit report measured dual gradient and KKT residual',
        prediction_tie_policy='physical_class_id_ascending', selection='fixed_no_selection',
        device='cpu', prediction_scope='independent_received_sample_all_registered_classes',
        fit_scope='current_split_support_only',
        new_source_payload_bytes=0, new_ground_statistics_bytes=0)
    write(out/'startup.json', startup); print(json.dumps(dict(event='STARTUP', **startup), allow_nan=False), flush=True)
    fields = ['split_id', 'receiver', 'scenario', 'k', 'new_count', 'support_seed', 'completed', 'total', 'classes',
        'selected', 'selection', 'fold_count', 'oof', 'factorization_count', 'optimizer_steps', 'fit_seconds',
        'head_bytes', 'persistent_state_bytes', 'final_fit', 'fit_call_seconds', 'query_score_seconds',
        'prediction_write_seconds', 'total_seconds', 'peak_process_rss_bytes', 'source_rows_used_for_fit',
        'query_rows_used_for_fit', 'source_validation', 'source_validation_reason', 'learning_rate', 'epoch', 'gradient_norm', 'unavailable_reason']
    completed_stages = []
    completed_splits = factorization_count = optimizer_steps = sweep_count = 0
    split = None
    current_fit_audit = None
    try:
        with (out/'predictions.jsonl').open('x', encoding='utf-8') as predictions, \
             (out/'fit_trace.jsonl').open('x', encoding='utf-8') as trace, \
             (out/'compact.jsonl').open('x', encoding='utf-8') as compact, \
             (out/'compact.csv').open('x', encoding='utf-8', newline='') as csvfile, \
             (out/'fit_stages.jsonl').open('x', encoding='utf-8') as stages, \
             (out/'solver_sweeps.jsonl').open('x', encoding='utf-8') as sweeps, \
             (out/'solver_sweeps.log').open('x', encoding='utf-8') as sweep_text:
            writer = csv.DictWriter(csvfile, fieldnames=fields); writer.writeheader()
            for number, (split, (support, query, labels)) in enumerate(zip(splits, tasks), 1):
                started = time.perf_counter()
                def log_sweep(value):
                    event = dict(scalar_record(value), split_id=split['split_id'],
                        source_validation=None, source_validation_reason=SOURCE_REASON,
                        learning_rate=None, learning_rate_reason=ANALYTIC_REASON)
                    sweeps.write(json.dumps(event, allow_nan=False)+'\n'); sweeps.flush()
                    text = ('[CVS LocalMargin] split={split_id} scope={scope} sweep={sweep} '
                            'steps={optimizer_steps} margin_loss={loss_data} rkhs_penalty={loss_ridge} '
                            'P={primal_objective} D={dual_objective} gap={relative_duality_gap} '
                            'KKT={relative_kkt_residual} LR=N/A source_validation=N/A').format_map(
                                {key: event.get(key, 'N/A') for key in ('split_id','scope','sweep',
                                 'optimizer_steps','loss_data','loss_ridge','primal_objective',
                                 'dual_objective','relative_duality_gap','relative_kkt_residual')})
                    text += ' measured=' + json.dumps(event, sort_keys=True, allow_nan=False)
                    sweep_text.write(text+'\n'); sweep_text.flush(); print(text, flush=True)
                state = core.fit_branch_local_margin(**{k:v[support] for k,v in arrays.items()}, support_labels=labels,
                    support_ids=ids[support], classes=split['registered_classes'], old_classes=old, arm='local_margin',
                    log_callback=log_sweep)
                fit_call_seconds = time.perf_counter()-started
                check(list(state.classes) == split['registered_classes'], 'Output class order mismatch')
                audit = state.audit_dict()
                check(audit['fold_count'] == 0 and audit['folds'] == [] and audit['oof'] is None
                      and type(audit['optimizer_steps']) is int and audit['optimizer_steps'] >= 0
                      and audit['factorization_count'] == 0
                      and audit['final_fit']['certified'] is True
                      and audit['optimizer_steps'] == audit['final_fit']['optimizer_steps']
                      and audit['selected'] == 'local_margin' and audit['selection'] == 'fixed_no_selection'
                      and audit['config'] == core.FROZEN_CONFIG, 'Frozen one-fit protocol mismatch')
                check(audit['final_fit']['training_physical_ids'] == sorted(ids[support].tolist())
                      and audit['final_fit']['train_physical_count'] == len(support)
                      and audit['final_fit']['all_states_estimated_from_current_support_only'] is True,
                      'Current support fit binding mismatch')
                factorization_count += audit['factorization_count']
                optimizer_steps += audit['optimizer_steps']
                sweep_count += audit['sweep_count']
                current_fit_audit = dict(split_id=split['split_id'], **audit['final_fit'])
                # Completed full traces are append-only on disk. Retain only
                # scalar summaries across splits, plus this one in-flight fit.
                completed_stages.append(scalar_record(current_fit_audit))
                small_stage = dict(split_id=split['split_id'], **scalar_record(audit['final_fit']))
                stages.write(json.dumps(small_stage, allow_nan=False)+'\n'); stages.flush()
                print(json.dumps(dict(event='CERTIFIED_MARGIN_FIT', **small_stage), allow_nan=False), flush=True)
                stage = time.perf_counter()
                # Enforce one received observation at this boundary as well as in
                # the core; no query pool can affect another query's prediction.
                scores = np.empty((len(query), len(state.classes)), dtype=np.float64)
                for position, physical_index in enumerate(query):
                    one = state.score(**{k:v[physical_index:physical_index+1] for k,v in arrays.items()})
                    check(one.shape == (1,len(state.classes)) and np.isfinite(one).all(), 'Invalid individual query scores')
                    scores[position] = one[0]
                query_score_seconds = time.perf_counter()-stage
                check(scores.shape == (len(query), len(state.classes)) and np.isfinite(scores).all(), 'Invalid query scores')
                stage = time.perf_counter()
                record = {key:split[key] for key in ('split_id', 'capsule_id', 'receiver', 'scenario', 'k', 'support_seed')}
                record.update(mode=MODE, classes=list(state.classes), query_ids=ids[query].tolist(), scores=scores.tolist(),
                              predicted_indices=stable_predictions(scores, state.classes).tolist())
                predictions.write(json.dumps(record, allow_nan=False)+'\n'); predictions.flush()
                prediction_write_seconds = time.perf_counter()-stage
                coordinates = {key:split[key] for key in ('split_id', 'receiver', 'scenario', 'k', 'support_seed')}
                coordinates['new_count'] = len(state.classes)-len(old)
                audit.update({k:v for k,v in coordinates.items() if k != 'k'}, registered_classes=list(state.classes),
                    support_records=sorted([dict(physical_id=str(pid), class_id=state.classes[int(label)])
                        for pid,label in zip(ids[support], labels)], key=lambda item:item['physical_id']),
                    fit_call_seconds=fit_call_seconds, query_score_seconds=query_score_seconds,
                    prediction_write_seconds=prediction_write_seconds, query_rows_used_for_fit=0, source_rows_used_for_fit=0)
                trace.write(json.dumps(audit, allow_nan=False)+'\n'); trace.flush()
                current_fit_audit = None
                small = dict(coordinates, completed=number, total=len(splits), classes=len(state.classes),
                    **{key:audit[key] for key in ('selected', 'selection', 'fold_count', 'oof', 'factorization_count',
                        'optimizer_steps', 'fit_seconds', 'head_bytes', 'persistent_state_bytes')},
                    final_fit=scalar_record(audit['final_fit']), fit_call_seconds=fit_call_seconds,
                    query_score_seconds=query_score_seconds, prediction_write_seconds=prediction_write_seconds,
                    total_seconds=time.perf_counter()-started, peak_process_rss_bytes=peak_process_rss(),
                    source_rows_used_for_fit=0, query_rows_used_for_fit=0, source_validation=None,
                    source_validation_reason=SOURCE_REASON, learning_rate=None, epoch=audit['final_fit']['sweeps'],
                    gradient_norm=audit['final_fit']['gradient_norm'], unavailable_reason=ANALYTIC_REASON)
                compact.write(json.dumps(small, allow_nan=False)+'\n'); compact.flush()
                writer.writerow(csv_record(small)); csvfile.flush()
                completed_splits += 1
                print(json.dumps(dict(event='ROW_COMPLETE', **small), allow_nan=False), flush=True)
        write_stage_csv(out/'fit_stages.jsonl', out/'fit_stages.csv')
        write_stage_csv(out/'solver_sweeps.jsonl', out/'solver_sweeps.csv')
        marker = dict(status='PREDICTIONS_COMPLETE', split_count=len(splits), predictions=len(splits),
            capsule_id=expected_capsule_id, checkpoint_sha256=expected_checkpoint_sha256, truth_read=False,
            query_used_for_fitting=False, source_data_access=False, payload_audit=payload,
            algorithm=core.FROZEN_CONFIG, mode=MODE,
            factorization_count=factorization_count, optimizer_steps=optimizer_steps, sweep_count=sweep_count,
            new_source_payload_bytes=0, new_ground_statistics_bytes=0,
            peak_process_rss_bytes=peak_process_rss(), peak_process_rss_reason='Linux process high-water RSS; null when unavailable')
        write(out/'predictions_complete.json', marker)
        return marker
    except Exception as exc:
        details = exc.audit_dict() if callable(getattr(exc, 'audit_dict', None)) else {}
        write(out/'technical_failure.json', dict(status='TECHNICAL_FAILURE', error_type=type(exc).__name__,
            error=str(exc), split_id=split['split_id'] if split is not None else None,
            completed_splits=completed_splits, completed_stages=completed_stages,
            current_fit=current_fit_audit,
            completed_evidence=dict(fit_trace=str(out/'fit_trace.jsonl'),
                fit_stages=str(out/'fit_stages.jsonl'), solver_sweeps=str(out/'solver_sweeps.jsonl'),
                solver_text=str(out/'solver_sweeps.log'), predictions=str(out/'predictions.jsonl')),
            completed_factorization_count=factorization_count, failure_context=details,
            factorization_count=factorization_count + details.get('factorization_count', details.get('factorization_calls', 0)),
            query_rows_used_for_fit=0, source_rows_used_for_fit=0,
            completed_optimizer_steps=optimizer_steps,
            optimizer_steps=optimizer_steps + details.get('optimizer_steps', 0)))
        if (out/'fit_stages.jsonl').exists() and not (out/'fit_stages.csv').exists():
            write_stage_csv(out/'fit_stages.jsonl', out/'fit_stages.csv')
        if (out/'solver_sweeps.jsonl').exists() and not (out/'solver_sweeps.csv').exists():
            write_stage_csv(out/'solver_sweeps.jsonl', out/'solver_sweeps.csv')
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('row-root', 'capsule', 'output', 'config', 'expected-capsule-id', 'expected-checkpoint-sha256', 'branch-features'):
        parser.add_argument('--'+key, required=True)
    args = vars(parser.parse_args()); args['config'] = read(args['config']); predict(**args)


if __name__ == '__main__': main()
