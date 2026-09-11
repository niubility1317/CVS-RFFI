"""Retrospective paired robustness of sealed predictions; no model or fitting.

Each model supplies a manifest (core90_conditional_prediction_manifest_v1) and
JSONL predictions. All four scenes must close on identical opaque IDs before
the truth callable/file is opened. No result is an untouched-target claim.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

SCENES = ('clean', 'leo_clear_weak', 'leo_low_elev_weak', 'leo_rain_weak')


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _rows(path):
    with Path(path).open(encoding='utf-8-sig') as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict): raise ValueError('JSONL row must be an object')
                yield row


def _integer(value, minimum=0):
    return type(value) is int and value >= minimum


def load_closed_predictions(path, manifest_path, *, _validate_all=True):
    m = _json(manifest_path)
    declared = None
    if m.get('schema') == 'core90_target_frozen_15_v1':
        declared = m.get('rows')
        completion_path = Path(manifest_path).parent/'predictions_complete.json'
        if not completion_path.exists(): raise ValueError('declared target prediction closure missing')
        completion = _json(completion_path)
        if (not isinstance(declared, list) or not declared or len(set(declared)) != len(declared)
                or any(not isinstance(r, str) or not r or Path(r).name != r for r in declared)
                or completion.get('complete') is not True or completion.get('rows') != declared
                or completion.get('truth_accessed_for_scoring') is not False
                or m.get('all_rows_predicted_before_truth') is not True
                or m.get('fitting') is not False or m.get('selection') is not False):
            raise ValueError('declared target manifest/closure inconsistent')
        if Path(path).name not in [r+'_predictions.jsonl' for r in declared]:
            raise ValueError('prediction model not declared')
        if completion.get('counts') != {s:m.get('samples_per_scene') for s in SCENES}:
            raise ValueError('declared scene counts incomplete')
        m = dict(m, schema='core90_conditional_prediction_manifest_v1', complete=True,
                 prediction_file_closed_before_scoring=True, truth_used_for_prediction=False,
                 prediction_scope='all_registered_classes_argmax', prediction_count=m.get('samples_per_scene', 0)*4)
    if m.get('schema') not in ('core90_conditional_prediction_manifest_v1', 'core90_game_source_predictions_v1'):
        raise ValueError('unsupported prediction manifest schema')
    if (m.get('complete') is not True or m.get('prediction_file_closed_before_scoring') is not True
            or m.get('truth_used_for_prediction') is not False
            or m.get('prediction_scope') != 'all_registered_classes_argmax'):
        raise ValueError('manifest does not establish complete truth-free prediction closure')
    if m['schema'] == 'core90_game_source_predictions_v1':
        counts = m.get('counts', {})
        if m.get('source_only') is not True or set(counts) != set(SCENES) or not all(_integer(n, 1) for n in counts.values()) or len(set(counts.values())) != 1:
            raise ValueError('source prediction manifest counts incomplete')
        m = dict(m, scenes=list(SCENES), samples_per_scene=counts['clean'], prediction_count=sum(counts.values()))
    n, classes = m.get('samples_per_scene'), m.get('num_classes')
    if not _integer(n, 1) or not _integer(classes, 1) or m.get('prediction_count') != n * 4:
        raise ValueError('invalid manifest count or class count')
    if len(m.get('scenes', [])) != 4 or set(m['scenes']) != set(SCENES):
        raise ValueError('all four scenes required')
    result = {scene: {} for scene in SCENES}
    for row in _rows(path):
        if set(row) & {'truth', 'y', 'label', 'tx_i', 'capture_group', 'role'}:
            raise ValueError('prediction contains truth/role metadata')
        sid, scene, pred = row.get('sample_id'), row.get('scene'), row.get('prediction')
        if not isinstance(sid, str) or not sid or scene not in result or not _integer(pred) or pred >= classes:
            raise ValueError('invalid opaque ID, scene or registered prediction')
        if sid in result[scene]: raise ValueError('duplicate scene/ID')
        result[scene][sid] = pred
    ids = set(result['clean'])
    if len(ids) != n or any(set(r) != ids for r in result.values()):
        raise ValueError('prediction scene/ID closure incomplete')
    if declared and _validate_all:
        for name in declared:
            other = Path(manifest_path).parent/(name+'_predictions.jsonl')
            if not other.exists(): raise ValueError('declared prediction file missing: '+name)
            if other.resolve() == Path(path).resolve(): continue
            values, _ = load_closed_predictions(other, manifest_path, _validate_all=False)
            if set(values['clean']) != ids: raise ValueError('declared models have mismatched opaque IDs')
    return result, m


def closed_source_metrics(predictions_path, manifest_path, truth_path):
    """Supplement legacy source scores with dense confusion and RX x TX counts."""
    predictions, manifest = load_closed_predictions(predictions_path, manifest_path)
    if manifest.get('source_only') is not True:
        raise ValueError('source metrics require source-only manifest')
    truth = {}
    for row in _rows(truth_path):
        sid, y = row.get('sample_id'), row.get('truth')
        if sid not in predictions['clean'] or sid in truth or not _integer(y) or y >= manifest['num_classes'] or 'rx_i' not in row:
            raise ValueError('invalid source truth join')
        truth[sid] = row
    if set(truth) != set(predictions['clean']): raise ValueError('source truth incomplete')
    result = {}
    for scene, values in predictions.items():
        confusion = [[0]*manifest['num_classes'] for _ in range(manifest['num_classes'])]
        groups = {}
        for sid, pred in values.items():
            y = truth[sid]['truth']
            confusion[y][pred] += 1
            key = str(truth[sid]['rx_i'])+':'+str(y)
            group = groups.setdefault(key, dict(count=0, correct=0))
            group['count'] += 1; group['correct'] += pred == y
        for group in groups.values(): group['accuracy'] = group['correct']/group['count']
        result[scene] = dict(confusion_matrix=confusion,
                            prediction_histogram=[sum(row[c] for row in confusion) for c in range(manifest['num_classes'])],
                            per_rx_tx=groups, worst_rx_tx_accuracy=min(g['accuracy'] for g in groups.values()))
    return result


def analyze_pair(a_predictions, a_manifest, b_predictions, b_manifest, truth_records):
    a, ma = load_closed_predictions(a_predictions, a_manifest)
    b, mb = load_closed_predictions(b_predictions, b_manifest)
    ids = set(a['clean'])
    if ids != set(b['clean']) or ma['num_classes'] != mb['num_classes']:
        raise ValueError('paired models have different opaque IDs or classes')
    if ma.get('augmentation_seed') != mb.get('augmentation_seed'):
        raise ValueError('paired view augmentation seeds differ')
    # Deliberately invoke a truth provider only AFTER both complete files validate.
    records = truth_records() if callable(truth_records) else _rows(truth_records)
    truth = {}
    for row in records:
        sid, y = row.get('sample_id'), row.get('truth')
        if sid not in ids or sid in truth or not _integer(y) or y >= ma['num_classes'] or 'rx_i' not in row:
            raise ValueError('truth IDs/classes/RX metadata invalid')
        truth[sid] = row
    if set(truth) != ids: raise ValueError('truth join incomplete')
    shared = {sid for sid in ids if a['clean'][sid] == b['clean'][sid] == truth[sid]['truth']}
    def counts(group, subset):
        return dict(sorted(Counter(str(truth[sid][group]) for sid in subset).items()))
    def summarize(scene, subset):
        n = len(subset)
        ae = {sid for sid in subset if a[scene][sid] != truth[sid]['truth']}
        be = {sid for sid in subset if b[scene][sid] != truth[sid]['truth']}
        return dict(count=n, a_errors=len(ae), b_errors=len(be),
                    a_error_rate=len(ae)/n if n else None, b_error_rate=len(be)/n if n else None,
                    a_wrong_b_right=len(ae-be), a_right_b_wrong=len(be-ae), both_wrong=len(ae & be))
    scenes = {}
    for scene in SCENES[1:]:
        summary = summarize(scene, shared)
        summary['overall'] = summarize(scene, ids)
        for group, name in [('truth', 'per_tx'), ('rx_i', 'per_rx')]:
            summary[name] = {g: summarize(scene, {sid for sid in shared if str(truth[sid][group]) == g})
                             for g in counts(group, ids)}
        scenes[scene] = summary
    return dict(schema='core90_conditional_robustness_v1', status='SEALED_PREDICTIONS_ANALYZED',
                scientific_scope='RETROSPECTIVE_CONTACTED_BENCHMARK_NO_SELECTION_OR_FITTING',
                total_count=len(ids), shared_clean_correct=dict(count=len(shared), fraction=len(shared)/len(ids),
                    per_tx_counts=counts('truth', shared), per_rx_counts=counts('rx_i', shared),
                    full_per_tx_counts=counts('truth', ids), full_per_rx_counts=counts('rx_i', ids)),
                scenes=scenes, replaces_overall_metrics=False)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('a-predictions', 'a-manifest', 'b-predictions', 'b-manifest', 'truth', 'output'):
        p.add_argument('--'+key, required=True)
    args = p.parse_args(argv)
    result = analyze_pair(args.a_predictions, args.a_manifest, args.b_predictions, args.b_manifest, args.truth)
    with Path(args.output).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
