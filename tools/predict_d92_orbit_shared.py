"""OSC support-only fitting from an existing frozen received-orbit cache."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'code'), str(ROOT / 'tools')]
import numpy as np
from cvsrffi.stage2_d92_orbit_shared import FROZEN_CONFIG, fit_orbit_shared
from d92_orbit_feature_cache import (FEATURE_CONTRACT, CACHE_SCHEMA, CACHE_NAME, read, write,
    validate_capsule, validate_split, load_features, peak_process_rss)

SOURCE_VALIDATION_REASON = 'Not run by design: user prohibits auxiliary source-sample and source-feature-cache access'
ANALYTIC_REASON = 'Analytical medoid/alignment/covariance/linear solve; no optimizer, learning rate, gradient or epochs'
RSS_REASON = 'Linux ru_maxrss process high-water memory; null when unavailable on this platform'
STAGE_FIELDS = ['split_id', 'scope', 'fold', 'train_k', 'train_physical_count', 'physical_df', 'shrinkage',
    'covariance_trace', 'feature_energy', 'zero_residual_tolerance', 'status', 'condition_bound',
    'medoid_seconds', 'covariance_seconds', 'solve_seconds', 'fit_seconds', 'optimizer_steps',
    'all_states_estimated_from_trainfold_only', 'covariance_matrix_bytes', 'covariance_shape_bytes',
    'training_feature_bytes', 'aligned_feature_bytes', 'template_bytes', 'cholesky_bytes',
    'coefficient_bytes', 'intercept_bytes', 'persistent_state_bytes']


def stable_predictions(scores, classes):
    order = np.asarray(sorted(range(len(classes)), key=lambda index: classes[index]))
    return order[np.argmax(scores[:, order], axis=1)]


def scalar_record(record):
    return {key: value for key, value in record.items() if value is None or isinstance(value, (str, int, float, bool))}


def csv_record(record):
    return {key: 'N/A' if value is None else json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
        for key, value in record.items()}


def predict(*, row_root, capsule, output, orbit_features, expected_capsule_id, expected_checkpoint_sha256, config):
    if set(config) != {'algorithm'} or config['algorithm'] != FROZEN_CONFIG:
        raise ValueError('Expected frozen OSC configuration')
    row_root, capsule, out = map(Path, (row_root, capsule, output))
    if out.exists():
        raise FileExistsError(out)
    manifest = validate_capsule(capsule, expected_capsule_id)
    identity, fft, ids, marker, previous = load_features(orbit_features=orbit_features, capsule=capsule,
        row_root=row_root, expected_capsule_id=expected_capsule_id,
        expected_checkpoint_sha256=expected_checkpoint_sha256, algorithm=config['algorithm'])
    old_classes = marker['classes']; paths = sorted((capsule / 'splits').glob('*.json'))
    if not paths or len(paths) != manifest['split_count']:
        raise ValueError('Incomplete split matrix')
    splits = [read(path) for path in paths]
    if len({split['split_id'] for split in splits}) != len(splits):
        raise ValueError('Duplicate split IDs')
    for split in splits:
        validate_split(split, manifest, ids, old_classes)
    out.mkdir(parents=True, exist_ok=False)
    payload = dict(new_ground_statistics_bytes=0, new_source_payload_bytes=0,
        model_file_bytes=marker['model_file_bytes'], model_already_deployed=None, model_incremental_transfer_bytes=None,
        model_deployment_unknown_reason='Model deployment state is unknown; incremental transfer bytes cannot be inferred',
        model_file_bytes_scope='Complete training checkpoint package on disk, not a minimal inference-only model package',
        received_feature_array_bytes=identity.nbytes + fft.nbytes, received_feature_file_bytes=marker['feature_file_bytes'],
        received_cache_scope='Existing locally computed received-only cache; not a ground transmission payload',
        reused_frozen_cache=True, new_feature_extraction_seconds=0, checkpoint_loaded=False)
    startup = dict(argv=sys.argv, python=sys.executable, pid=os.getpid(), config=config,
        checkpoint_sha256=expected_checkpoint_sha256, capsule_id=expected_capsule_id, model_seed=previous['seed'],
        query_fit_access=False, truth_read=False, source_data_access=False, ground_summary_access=False,
        source_validation=None, source_validation_reason=SOURCE_VALIDATION_REASON,
        orbit_features=str(orbit_features), baseline_row_root=str(row_root), payload_audit=payload,
        feature_cache_schema=CACHE_SCHEMA, feature_cache_name=CACHE_NAME, feature_contract=FEATURE_CONTRACT,
        feature_cache_precision='float32', downstream_unit_norm_floor=1e-12,
        adapted_state_inherited=False, cross_row_adapted_state_reuse=False, encoder_updated=False,
        prediction_tie_policy='physical_class_id_ascending', selection='none_cv_diagnostic_only',
        training_log_unit='analytical fit stages; OOF diagnostics do not select or calibrate the model',
        learning_rate=None, gradient_norm=None, epoch=None, unavailable_reason=ANALYTIC_REASON,
        new_ground_statistics_bytes=0, new_source_payload_bytes=0)
    write(out / 'startup.json', startup)
    print(json.dumps(dict(event='STARTUP', **startup), allow_nan=False), flush=True)
    fields = ['split_id', 'completed', 'total', 'receiver', 'scenario', 'support_seed', 'new_count', 'k', 'classes',
        'selected', 'selection', 'candidate_count', 'fold_count', 'oof', 'final_fit', 'fit_seconds',
        'head_bytes', 'persistent_state_bytes', 'optimizer_steps', 'fit_call_seconds', 'query_score_seconds',
        'prediction_write_seconds', 'total_seconds', 'peak_process_rss_bytes', 'peak_process_rss_reason',
        'source_rows_used_for_fit', 'query_rows_used_for_fit', 'source_validation', 'source_validation_reason',
        'learning_rate', 'gradient_norm', 'epoch', 'unavailable_reason']
    with (out / 'predictions.jsonl').open('x', encoding='utf-8') as predictions, \
            (out / 'fit_trace.jsonl').open('x', encoding='utf-8') as trace, \
            (out / 'compact.jsonl').open('x', encoding='utf-8') as compact, \
            (out / 'compact.csv').open('x', encoding='utf-8', newline='') as csvfile, \
            (out / 'fit_stages.jsonl').open('x', encoding='utf-8') as stages_json, \
            (out / 'fit_stages.csv').open('x', encoding='utf-8', newline='') as stages_csv:
        writer = csv.DictWriter(csvfile, fieldnames=fields); writer.writeheader()
        stage_writer = csv.DictWriter(stages_csv, fieldnames=STAGE_FIELDS); stage_writer.writeheader()
        for index, split in enumerate(splits):
            support, query, labels = validate_split(split, manifest, ids, old_classes)
            started = time.perf_counter()
            state = fit_orbit_shared(support_identity_views=identity[support], support_fft=fft[support],
                support_labels=labels, support_ids=ids[support], classes=split['registered_classes'], old_classes=old_classes)
            fit_call_seconds = time.perf_counter() - started
            if list(state.classes) != split['registered_classes']:
                raise ValueError('OSC class order mismatch')
            audit = state.audit_dict()
            query_identity, query_fft = identity[query], fft[query]
            score_started = time.perf_counter(); scores = state.score(query_identity, query_fft)
            query_score_seconds = time.perf_counter() - score_started
            if scores.shape != (len(query), len(state.classes)) or not np.isfinite(scores).all():
                raise ValueError('Invalid query scores')
            write_started = time.perf_counter()
            record = {key: split[key] for key in ('split_id', 'capsule_id', 'receiver', 'scenario', 'k', 'support_seed')}
            record.update(mode='d92_osc_registration', classes=list(state.classes), query_ids=ids[query].tolist(),
                predicted_indices=stable_predictions(scores, state.classes).tolist(), scores=scores.tolist())
            predictions.write(json.dumps(record, allow_nan=False) + '\n'); predictions.flush()
            prediction_write_seconds = time.perf_counter() - write_started
            coordinates = {key: split[key] for key in ('receiver', 'scenario', 'support_seed')}
            coordinates['new_count'] = len(state.classes) - len(old_classes)
            audit.update(registered_classes=list(state.classes), old_classes=list(old_classes), **coordinates,
                fit_call_seconds=fit_call_seconds, query_score_seconds=query_score_seconds,
                prediction_write_seconds=prediction_write_seconds)
            trace.write(json.dumps(dict(split_id=split['split_id'], **audit), allow_nan=False) + '\n'); trace.flush()
            for stage in [fold['training'] for fold in audit['folds']] + [audit['final_fit']]:
                small_stage = dict(split_id=split['split_id'], **scalar_record(stage))
                stages_json.write(json.dumps(small_stage, allow_nan=False) + '\n')
                stage_writer.writerow(csv_record({key: small_stage.get(key) for key in STAGE_FIELDS}))
                print(json.dumps(dict(event='ANALYTICAL_FIT', **small_stage), allow_nan=False), flush=True)
            stages_json.flush(); stages_csv.flush()
            small = dict(split_id=split['split_id'], completed=index + 1, total=len(splits), k=split['k'],
                classes=len(state.classes), **coordinates,
                **{key: audit[key] for key in ('selected', 'selection', 'candidate_count', 'fold_count', 'fit_seconds',
                    'head_bytes', 'persistent_state_bytes', 'optimizer_steps')},
                oof=scalar_record(audit['oof']) if audit['oof'] is not None else None,
                final_fit=scalar_record(audit['final_fit']), fit_call_seconds=fit_call_seconds,
                query_score_seconds=query_score_seconds, prediction_write_seconds=prediction_write_seconds,
                total_seconds=time.perf_counter() - started, peak_process_rss_bytes=peak_process_rss(),
                peak_process_rss_reason=RSS_REASON, source_rows_used_for_fit=0, query_rows_used_for_fit=0,
                source_validation=None, source_validation_reason=SOURCE_VALIDATION_REASON,
                learning_rate=None, gradient_norm=None, epoch=None, unavailable_reason=ANALYTIC_REASON)
            compact.write(json.dumps(small, allow_nan=False) + '\n'); compact.flush()
            writer.writerow(csv_record(small)); csvfile.flush()
            print(json.dumps(dict(event='ROW_COMPLETE', **small), allow_nan=False), flush=True)
    write(out / 'predictions_complete.json', dict(status='PREDICTIONS_COMPLETE', split_count=len(splits), predictions=len(splits),
        capsule_id=expected_capsule_id, checkpoint_sha256=expected_checkpoint_sha256, truth_read=False,
        source_data_access=False, query_used_for_fitting=False, payload_audit=payload,
        peak_process_rss_bytes=peak_process_rss(), peak_process_rss_reason=RSS_REASON,
        new_ground_statistics_bytes=0, new_source_payload_bytes=0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('row-root', 'capsule', 'output', 'orbit-features', 'config', 'expected-capsule-id', 'expected-checkpoint-sha256'):
        parser.add_argument('--' + name, required=True)
    args = vars(parser.parse_args()); args['config'] = read(args['config']); predict(**args)


if __name__ == '__main__':
    main()
