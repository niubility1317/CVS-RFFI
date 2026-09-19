"""One-shot CPU audit of fixed, already scheduled practical predictions."""
import argparse
import datetime
import json
import os
from pathlib import Path
import time

SCENES = ('clean', 'practical_high', 'practical_mid', 'practical_low_urban')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def recount(row):
    source = Path(row['source_output'])
    evaluation = source / 'target_epochs' / ('E' + str(row['epochs']))
    scope = read(evaluation / 'evaluation_scope.json')
    checkpoint = source / ('epoch_' + str(row['epochs']) + '_ssdg.pth')
    assert scope['checkpoint'] == str(checkpoint) and checkpoint.is_file()
    assert scope['feeds_training'] is False and scope['record_count'] == 672000
    initialization = read(source / 'initialization.json')
    assert initialization['scratch_only'] and not initialization['checkpoint_sources']
    assert initialization['target_training_contact'] is False
    assert initialization['source_roles'] == 'EXACT_MATCH'
    config = read(source / 'resolved_config.json')
    for key, value in row['channel_config'].items():
        assert config[key] == value, (key, config[key], value)
    pred = read(evaluation / 'predictions.json')
    assert pred['checkpoint'] == str(checkpoint) and pred['record_count'] == 672000
    assert len(pred['records']) == 672000
    # Truth is opened only after the complete prediction artifact and scope exist.
    old_score = read(evaluation / 'score.json')
    truth = read(config['a1_periodic_target_truth'])
    labels = {r['sample_id']: r['label'] for r in truth['records']}
    assert len(labels) == len(truth['records']) == 168000
    seen = {s: set() for s in SCENES}
    matrices = {s: [[0] * 6 for _ in range(6)] for s in SCENES}
    for r in pred['records']:
        scene, sid, yp = r['scenario'], r['sample_id'], r['predicted_class']
        assert r['row_id'] == row['source_row_id'] + '_E' + str(row['epochs'])
        assert r['run_id'] == row['source_run_id']
        y = labels[sid]
        assert sid not in seen[scene] and 0 <= y < 6 and 0 <= yp < 6
        seen[scene].add(sid)
        matrices[scene][y][yp] += 1
    metrics = {}
    for scene, cm in matrices.items():
        assert seen[scene] == set(labels)
        correct = sum(cm[i][i] for i in range(6))
        total = sum(map(sum, cm))
        f1 = [2 * cm[i][i] / (sum(cm[i]) + sum(r[i] for r in cm)) for i in range(6)]
        assert correct == old_score['metrics'][scene]['correct']
        assert total == old_score['metrics'][scene]['total']
        metrics[scene] = dict(accuracy=correct / total, macro_f1=sum(f1) / 6,
                              correct=correct, total=total, per_class_f1=f1,
                              confusion_matrix_true_rows=cm)
    return dict(status='VERIFIED', epoch=row['epochs'], checkpoint=str(checkpoint),
                prediction_path=str(evaluation / 'predictions.json'),
                channel_config=row['channel_config'], metrics=metrics,
                coverage_exact=True, original_score_matches=True,
                scope='existing frozen predictions; no new inference; no training feedback; not blind confirmation')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', required=True)
    args = parser.parse_args()
    spec = read(args.spec)
    root = Path(spec['execution']['remote_run_root'])
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'owner.json').open('x', encoding='utf-8') as f:
        json.dump(dict(pid=os.getpid(), cwd=os.getcwd(), spec=args.spec), f)
    states = {}
    deadline = time.time() + 24 * 3600
    while True:
        for row in spec['rows']:
            key = row['row_id']
            if states.get(key, {}).get('status') in ('VERIFIED', 'FAILED'):
                continue
            evaluation = Path(row['source_output']) / 'target_epochs' / ('E' + str(row['epochs']))
            if all((evaluation / f).is_file() for f in ('predictions.json', 'score.json', 'evaluation_scope.json')):
                try:
                    result = recount(row)
                    with (root / (key + '.json')).open('x', encoding='utf-8') as f:
                        json.dump(result, f, ensure_ascii=False, indent=2)
                    states[key] = result
                    print(key, 'VERIFIED', json.dumps(result['metrics']), flush=True)
                except Exception as exc:
                    states[key] = dict(status='FAILED', error=repr(exc))
                    print(key, repr(exc), flush=True)
            else:
                proc = Path('/proc') / str(row['training_pid']) / 'cmdline'
                alive = proc.exists() and row['source_output'] in proc.read_bytes().decode(errors='replace')
                states[key] = dict(status='WAITING_FOR_SCHEDULED_TEST' if alive else 'FAILED',
                                   epoch=row['epochs'], evaluation=str(evaluation))
        complete = all(s['status'] == 'VERIFIED' for s in states.values())
        terminal = all(s['status'] in ('VERIFIED', 'FAILED') for s in states.values())
        status = 'COMPLETE' if complete else 'PARTIAL' if terminal or time.time() > deadline else 'RUNNING'
        payload = dict(status=status, timestamp=datetime.datetime.now().astimezone().isoformat(), rows=states)
        temporary = root / 'state.tmp'
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(root / 'state.json')
        if status != 'RUNNING':
            with (root / 'completion.json').open('x', encoding='utf-8') as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            return
        time.sleep(60)


if __name__ == '__main__':
    main()
