"""Project identities for completed ABC analysis, without fitting or scoring.

Reads validated capsule IDs/splits and completed frozen-DG JSONL records.
JSONL parsing necessarily encounters their stored score fields; these are
discarded, never computed on, returned, or supplied to a predictor. No truth,
IQ member, feature values, source sample, or checkpoint weights are read.
"""
import argparse
import inspect
import json
from pathlib import Path
import subprocess


def check(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def terminal(spec):
    root = Path(spec['execution']['remote_run_root'])
    start, done = read(root/'startup.json'), read(root/'complete.json')
    check(start['spec'] == spec, 'Actual startup spec differs from requested run')
    check(done.get('status') == 'SCORED' and done.get('selection_feedback_forbidden') is True,
          'Only completed immutable scored runs may be projected')
    check(done.get('commit') == start.get('commit') and bool(start.get('commit')),
          'Completion/startup commit mismatch')
    count = spec['confirmation']['expected_split_count']
    dg = len(spec['data']['target_receivers'])*len(spec['data']['scenarios'])
    check(done.get('model_rows') == len(spec['rows']) == 4
          and done.get('records') == 4*(2*count+dg), 'Incomplete model or record matrix')
    check(spec['confirmation']['candidate_method'] in
          ('D92-BranchLocalRidge-v1', 'D92-BranchLocalMargin-v1'), 'Unsupported candidate')


def collect_identity(spec):
    import numpy as np
    conf = spec['confirmation']
    cap = Path(conf['capsule'])
    manifest = read(cap/'manifest.json')
    cid = spec['data']['capsule_id']
    count = conf['expected_split_count']
    check(manifest.get('protocol_schema') == 'p2_min_v1'
          and manifest.get('phase2_data_status') == 'VALIDATED_ONCE'
          and manifest.get('capsule_id') == cid == conf['reuse_validated_capsule_id']
          and manifest.get('split_count') == count, 'Validated capsule binding mismatch')
    with np.load(cap/'received.npz', allow_pickle=False) as arrays:
        ids = arrays['ids'].astype(str).tolist()
    check(len(ids) == len(set(ids)) == manifest['received_count'], 'Physical ID inventory mismatch')
    splits = {}
    for path in sorted((cap/'splits').glob('*.json')):
        row = read(path)
        sid = row['split_id']
        check(sid == path.stem and sid not in splits and row['capsule_id'] == cid,
              'Split identity/capsule mismatch')
        normalized = {k: row[k] for k in ('split_id','capsule_id','receiver','scenario',
                       'k','support_seed','registered_classes','support_labels')}
        for source, destination in (('support_indices','support_ids'),('query_indices','query_ids')):
            indices = row[source]
            check(isinstance(indices, list) and all(type(i) is int and 0 <= i < len(ids) for i in indices),
                  'Invalid physical record index')
            normalized[destination] = [ids[i] for i in indices]
        splits[sid] = normalized
    check(len(splits) == count, 'Incomplete physical split matrix')
    models = []
    expected_dg = {(s['receiver'], s['scenario']) for s in splits.values()}
    for row in spec['rows']:
        origin = Path(row['reuse_row_root'])
        previous = read(origin/'d92_startup.json')
        ground = read(origin/'ground_provenance.json')
        marker = read(origin/'predictions_complete.json')
        sha, seed = row['expected_checkpoint_sha256'], row['seeds']['model']
        check(previous.get('checkpoint_sha256') == ground.get('checkpoint_sha256') == sha
              and previous.get('seed') == seed and ground.get('source_root') == row['source_root']
              and previous.get('capsule') == str(cap)
              and previous.get('query_fit_access') is False and previous.get('truth_read') is False,
              'Frozen DG source/checkpoint/model binding mismatch')
        check(marker.get('status') == 'PREDICTIONS_COMPLETE' and marker.get('capsule_id') == cid
              and marker.get('split_count') == count and marker.get('truth_read') is False,
              'Frozen DG producer predictions incomplete')
        dg, seen, main, lines = [], set(), set(), 0
        with (origin/'predictions.jsonl').open(encoding='utf-8') as stream:
            for line in stream:
                record = json.loads(line)
                lines += 1
                if record.get('mode') != 'frozen_dg':
                    sid = record['split_id']
                    check(record.get('mode') == 'd92_registration' and sid in splits and sid not in main,
                          'Invalid producer registration identity coverage')
                    main.add(sid)
                    continue
                projected = {k: record[k] for k in ('split_id','capsule_id','receiver','scenario',
                                                   'classes','query_ids')}
                sid = projected['split_id']
                check(sid in splits, 'DG references unknown physical split')
                source = splits[sid]
                key = (projected['receiver'], projected['scenario'])
                check(key not in seen and projected['capsule_id'] == cid
                      and projected['classes'] == conf['old_classes'] == source['registered_classes']
                      and projected['query_ids'] == source['query_ids']
                      and all(projected[k] == source[k] for k in ('receiver','scenario')),
                      'Frozen DG physical query/class binding mismatch')
                seen.add(key)
                dg.append(projected)
        check(seen == expected_dg and main == set(splits)
              and lines == marker.get('predictions') == count+len(expected_dg),
              'Frozen DG producer identity coverage incomplete')
        models.append(dict(model_seed=seed, row_id=row['row_id'], checkpoint_sha256=sha,
                           source_root=row['source_root'], reuse_row_root=row['reuse_row_root'],
                           splits=list(splits.values()), frozen_dg=dg))
    return dict(schema='d92_adaptation_registration_identity_v1', run_id=spec['run_id'],
                capsule_id=cid, candidate_method=conf['candidate_method'], models=models)


def collect_all(specs):
    check(len(specs) == 2 and len({s['run_id'] for s in specs}) == 2, 'Two distinct completed cohorts required')
    for spec in specs:
        terminal(spec)
    return [collect_identity(spec) for spec in specs]


def remote_script(specs):
    return ('import json\nfrom pathlib import Path\n'
            +'\n'.join(inspect.getsource(f) for f in (check, read, terminal, collect_identity, collect_all))
            +'\nprint(json.dumps(collect_all('+repr(specs)+'),allow_nan=False))\n')


def main():
    from read_d92_run import FLAGS
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec', type=Path, action='append', required=True)
    p.add_argument('--output', type=Path, required=True, help='New directory containing one identity JSON per run')
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    specs = [read(path) for path in args.spec]
    script = remote_script(specs)
    compile(script, 'read_only_abc_identity_projection', 'exec')
    result = subprocess.run(['ssh',*FLAGS,'-T','N607',
        '/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -'], input=script.encode('utf-8'),
        capture_output=True, check=True)
    values = json.loads(result.stdout)
    check(len(values) == len(specs), 'Incomplete projection output')
    for value, spec in zip(values, specs):
        check(value['run_id'] == spec['run_id'] and len(value['models']) == 4, 'Projection binding mismatch')
    args.output.mkdir(parents=True, exist_ok=False)
    for value in values:
        path = args.output/(value['run_id']+'.json')
        with path.open('x', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False)
            stream.write('\n')
    print(json.dumps(dict(status='VERIFIED_IDENTITY_PROJECTION', runs=[v['run_id'] for v in values],
        output=str(args.output), fitting=False, scoring=False, truth_read=False,
        source_samples_read=False, numeric_prediction_values_exported=False)))


if __name__ == '__main__':
    main()
