"""Frozen summary normalization and row-local support fitting; no truth interface."""
import argparse
import csv
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'code'), str(ROOT / 'tools')]
import numpy as np
from cvs_d92_matched import registered_features
from predict_d92_support_cv import validate_split
from predict_d92_sourcefree_head import plain, read
from cvsrffi.d92_ground_summary import load_ground_summary
from cvsrffi.stage2_d92_summary_joint import (
    FROZEN_CONFIG, build_summary_operator, fit_summary_joint,
)
SOURCE_VALIDATION_REASON = 'Not run by design: user prohibits auxiliary source-sample and source-feature-cache access'


def stable_predictions(scores, classes):
    """Exact ties choose physical class ID, independently of registry column order."""
    order = np.asarray(sorted(range(len(classes)), key=lambda i: classes[i]))
    return order[np.argmax(scores[:, order], axis=1)]


def peak_process_rss():
    # Linux ru_maxrss is measured process high-water memory, not per-head memory.
    if sys.platform.startswith('linux'):
        import resource
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    return None


def predict(*, row_root, capsule, output, expected_capsule_id,
            expected_checkpoint_sha256, config):
    if (set(config) != {'algorithm', 'summary_already_deployed'}
            or config['algorithm'] != FROZEN_CONFIG
            or type(config['summary_already_deployed']) is not bool):
        raise ValueError('Expected the frozen SGJoint formula and explicit delivery state')
    row_root, capsule, out = map(Path, (row_root, capsule, output))
    if out.exists():
        raise FileExistsError(out)
    manifest = read(capsule / 'manifest.json')
    if (manifest.get('protocol_schema') != 'p2_min_v1'
            or manifest.get('phase2_data_status') != 'VALIDATED_ONCE'
            or manifest.get('capsule_id') != expected_capsule_id):
        raise ValueError('Capsule mismatch')
    previous = read(row_root / 'd92_startup.json')
    feature_root = row_root / 'received_features'
    marker = read(feature_root / 'features_complete.json')
    provenance = read(feature_root / 'checkpoint_provenance.json')
    if (previous.get('checkpoint_sha256') != expected_checkpoint_sha256
            or previous.get('capsule') != str(capsule)
            or previous.get('features') != str(feature_root / 'received_features.npz')
            or previous.get('query_fit_access') is not False
            or previous.get('truth_read') is not False
            or marker.get('status') != 'FROZEN_FEATURES_COMPLETE'
            or marker.get('capsule_id') != expected_capsule_id
            or marker.get('query_used_for_fitting') is not False
            or provenance.get('target_access_before_freeze') is not False
            or provenance.get('checkpoint_inheritance') != []):
        raise ValueError('Frozen feature provenance mismatch')
    old_classes = provenance['classes']
    summary = load_ground_summary(row_root / 'ground',
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_classes=old_classes, already_deployed=config['summary_already_deployed'])
    operator = build_summary_operator(summary)
    payload_audit = plain(summary.payload_audit)
    # Only the frozen normalization operator survives. No reconstructed source bank.
    del summary
    with np.load(feature_root / 'received_features.npz', allow_pickle=False) as d:
        if set(d.files) != {'identity160', 'logits', 'ids'}:
            raise ValueError('Unexpected feature members')
        identity, ids = d['identity160'], d['ids'].astype(str)
    with np.load(capsule / 'received.npz', allow_pickle=False) as d:
        iq, capsule_ids = d['iq'], d['ids'].astype(str)
    if (not np.array_equal(ids, capsule_ids) or len(set(ids)) != len(ids)
            or identity.shape != (len(ids), 160) or not np.isfinite(identity).all()
            or iq.ndim != 3 or iq.shape[0] != len(ids) or iq.shape[1:] != (2, 256)
            or not np.isfinite(iq).all()):
        raise ValueError('Feature/received IQ/physical ID alignment mismatch')
    paths = sorted((capsule / 'splits').glob('*.json'))
    if len(paths) != manifest['split_count']:
        raise ValueError('Incomplete split matrix')
    splits = [read(p) for p in paths]
    for split in splits:
        validate_split(split, manifest, ids, old_classes)
    started_features = time.perf_counter()
    features = registered_features(iq, identity)
    feature_seconds = time.perf_counter() - started_features
    del iq, identity
    out.mkdir(parents=True, exist_ok=False)
    startup = dict(argv=sys.argv, config=config,
        checkpoint_sha256=expected_checkpoint_sha256, capsule_id=expected_capsule_id,
        query_fit_access=False, truth_read=False, source_data_access=False,
        source_validation=None, source_validation_reason=SOURCE_VALIDATION_REASON,
        model_seed=previous['seed'], payload_audit=payload_audit,
        received_feature_seconds=feature_seconds,
        received_feature_scope='Deterministic FFT plus cached frozen identity; excludes encoder extraction',
        feature_transform_scope='Independent normalization and same-row FFT; no cross-query statistics',
        prediction_tie_policy='physical_class_id_ascending',
        new_ground_statistics_bytes=0, cross_row_adapted_state_reuse=False)
    (out / 'startup.json').write_text(json.dumps(startup, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(event='STARTUP', **startup)), flush=True)
    fields = ['split_id', 'completed', 'total', 'k', 'classes', 'selected',
        'selection', 'candidate_count', 'fold_count', 'selected_objective',
        'selected_macro_nll', 'selected_old_nll', 'selected_new_nll',
        'fit_seconds', 'persistent_state_bytes', 'summary_operator_bytes',
        'total_seconds', 'peak_process_rss_bytes', 'query_rows_used_for_fit',
        'source_rows_used_for_fit', 'learning_rate', 'gradient', 'unavailable_reason']
    fields += ['source_validation', 'source_validation_reason']
    with (out / 'predictions.jsonl').open('x', encoding='utf-8') as pred, \
            (out / 'fit_trace.jsonl').open('x', encoding='utf-8') as trace, \
            (out / 'compact.jsonl').open('x', encoding='utf-8') as compact, \
            (out / 'compact.csv').open('x', encoding='utf-8', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for index, split in enumerate(splits):
            s, q, y = validate_split(split, manifest, ids, old_classes)
            started = time.perf_counter()
            state = fit_summary_joint(support_features=features[s], support_labels=y,
                support_ids=ids[s], classes=split['registered_classes'],
                old_classes=old_classes, summary_operator=operator)
            audit = state.audit_dict()
            scores = state.score(features[q])
            if scores.shape != (len(q), len(state.classes)) or not np.isfinite(scores).all():
                raise ValueError('Invalid query scores')
            record = {k: split[k] for k in ('split_id', 'capsule_id', 'receiver', 'scenario', 'k', 'support_seed')}
            record.update(mode='d92_sgjoint_registration', classes=split['registered_classes'],
                query_ids=ids[q].tolist(), predicted_indices=stable_predictions(scores, state.classes).tolist(),
                scores=scores.tolist())
            pred.write(json.dumps(record, allow_nan=False) + '\n')
            pred.flush()
            trace.write(json.dumps(dict(split_id=split['split_id'], **audit), allow_nan=False) + '\n')
            trace.flush()
            selected_risk = audit['selected_oof_risk'] or {}
            small = dict(split_id=split['split_id'], completed=index + 1, total=len(splits),
                k=split['k'], classes=len(state.classes), selected=audit['selected'],
                selection=audit['selection'], candidate_count=len(audit['candidate_trace']),
                fold_count=audit['fold_count'], selected_objective=selected_risk.get('objective'),
                selected_macro_nll=selected_risk.get('macro_nll'), selected_old_nll=selected_risk.get('old_nll'),
                selected_new_nll=selected_risk.get('new_nll'),
                fit_seconds=audit['fit_seconds'], persistent_state_bytes=audit['persistent_state_bytes'],
                summary_operator_bytes=audit['summary_operator_bytes'],
                total_seconds=time.perf_counter() - started, peak_process_rss_bytes=peak_process_rss(),
                query_rows_used_for_fit=0, source_rows_used_for_fit=0, learning_rate=None, gradient=None,
                source_validation=None, source_validation_reason=SOURCE_VALIDATION_REASON,
                unavailable_reason='Closed-form LDA and scalar temperature search; no learning rate/gradient; process peak RSS Linux-only')
            compact.write(json.dumps(small, allow_nan=False) + '\n')
            compact.flush()
            csv_row = {key: 'N/A' if value is None else value for key, value in small.items()}
            csv_row['selected'] = json.dumps(small['selected'], sort_keys=True)
            writer.writerow(csv_row)
            csvfile.flush()
            # Full measured candidate/fold terms are kept on disk; no large arrays in console.
            print(json.dumps(dict(event='ROW_COMPLETE', **small), allow_nan=False), flush=True)
            for entry in audit.get('candidate_trace', []):
                print(json.dumps(dict(event='SUPPORT_CV_CANDIDATE', split_id=split['split_id'], **entry), allow_nan=False), flush=True)
            del state
    complete = dict(status='PREDICTIONS_COMPLETE', split_count=len(splits), predictions=len(splits),
        capsule_id=expected_capsule_id, truth_read=False, source_data_access=False,
        payload_audit=payload_audit, peak_process_rss_bytes=peak_process_rss(),
        peak_memory_scope='Whole predictor process including input feature arrays, linear algebra, and logging; not satellite hardware measurement')
    (out / 'predictions_complete.json').write_text(json.dumps(complete, indent=2) + '\n', encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('row-root', 'capsule', 'output', 'config', 'expected-capsule-id', 'expected-checkpoint-sha256'):
        p.add_argument('--' + name, required=True)
    a = p.parse_args()
    predict(row_root=a.row_root, capsule=a.capsule, output=a.output, config=read(a.config),
        expected_capsule_id=a.expected_capsule_id, expected_checkpoint_sha256=a.expected_checkpoint_sha256)


if __name__ == '__main__':
    main()
