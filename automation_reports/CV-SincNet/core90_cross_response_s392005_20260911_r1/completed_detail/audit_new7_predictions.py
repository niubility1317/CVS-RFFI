"""Read-only complete prediction audit, with truth opened after all scene validation."""
import json
import sys
from pathlib import Path
sys.path.insert(0, '/home/szu2070436088/2510044040/CV-SincNet/releases/core90_cross_response_392005_54afc2bc/code')
from cvsrffi.cross_response.final_scoring import _load_complete_predictions, _accuracy
root = Path('/home/szu2070436088/2510044040/CV-SincNet/runs/core90_cross_response_s392005_20260911_r1')
result = {}
baseline_manifest = json.loads((root/"U0"/"final_predictions"/"prediction_manifest.json").read_text())
for variant in ("U1","U3","U4_additive","U4_bilinear","U5","head_only","permanent_detach"):
    directory = root/variant/'final_predictions'
    manifest, predictions = _load_complete_predictions(directory)
    if baseline_manifest is None:
        baseline_manifest = manifest
    assert all(manifest[k] == baseline_manifest[k] for k in
               ('record_ids', 'named_ids', 'record_metadata', 'scene_seeds', 'registered_class_count'))
    truth_rows = [json.loads(line) for line in (directory/'scoring_truth.jsonl').read_text().splitlines()]
    truth = {row['record_id']:row['label'] for row in truth_rows}
    assert len(truth_rows) == len(truth) == manifest['record_count']
    assert set(truth) == set(manifest['record_ids'])
    stored = json.loads((directory/'independent_scores.json').read_text())
    groups_checked = 0
    confusion = {}
    for scene, pred in predictions.items():
        aggregate = stored['test'] if scene == 'clean' else stored['sat_test_named'][scene]['aggregate']
        named = stored['named_test'] if scene == 'clean' else stored['sat_test_named'][scene]['named']
        groups = [(manifest['record_ids'], aggregate)]
        groups += [(ids, named[name]) for name, ids in manifest['named_ids'].items()]
        for rx, score in stored['per_receiver'][scene].items():
            groups.append(([rid for rid, meta in manifest['record_metadata'].items() if str(meta['receiver']) == rx], score))
        for cls, score in stored['per_class'][scene].items():
            groups.append(([rid for rid, y in truth.items() if str(y) == cls], score))
        for ids, score in groups:
            checked = _accuracy(pred, truth, ids)
            assert checked['tx_correct'] == score['tx_correct'] and checked['tx_total'] == score['tx_total']
            assert abs(checked['tx_acc'] - score['tx_acc']) < 1e-10
            groups_checked += 1
        matrix = [[0]*manifest['registered_class_count'] for _ in range(manifest['registered_class_count'])]
        for rid, y in truth.items():
            matrix[y][pred[rid]] += 1
        confusion[scene] = matrix
    result[variant] = dict(status='VERIFIED', records=manifest['record_count'], prediction_rows=manifest['rows_written'],
        registered_classes=manifest['registered_class_count'], compared_score_groups=groups_checked,
        same_physical_records_roles_and_scene_seeds=True, all_predictions_validated_before_truth_read=True,
        confusion_matrices=confusion, confusion_orientation='rows=true, columns=predicted', no_remote_writes=True)
print(json.dumps(result))
