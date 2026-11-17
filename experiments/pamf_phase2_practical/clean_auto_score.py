"""Wait for the whole retained clean comparison and score it truth-last."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import time

import numpy as np

from .clean_comparison_plan import tasks
from .clean_scheduler import completion_matches
from .clean_score import expected_rows, key, score_job, verify_all
from .episodic_io import query_complement
from .io import atomic, read, require, write

DA = ('protonet_cda', 'mrior_sda', 'dadda_sda', 'twostage', 'poster_ft', 'tweak_m1')
CIL = ('mopc_hr', 'orthogonal', 'lora_triplet_knn', 'issl_fixed_v1')
CORE = ('mopc_hr', 'orthogonal', 'lora')
EXCLUDED_PLANS = {'20261112-phase2-clean-pilot-r01', '20261112-phase2-clean-pilot-r02',
                  '20261113-phase2-clean-pilot-r03', '20261114-phase2-clean-ready-full-m4-r01'}


def unit(group, job):
    return group, job.get('profile', 'R42'), job['capsule'], int(job['seed'])


def expected_units(task_list, mopc_stabilization='none'):
    require(mopc_stabilization == 'none', 'Retired MoPC versions cannot be scored')
    result = {}
    for task in task_list:
        for group, profiles in (('pamf', ('R42',)), ('native_da', DA),
                                ('native_cil', CIL), ('hybrid', DA)):
            for profile in profiles:
                job = dict(task, profile=profile)
                if group=='native_cil' and profile=='mopc_hr':
                    job['mopc_stabilization']=mopc_stabilization
                if group == 'hybrid':
                    job['methods'] = ['orthogonal','lora']
                    if profile == 'protonet_cda':
                        job['mopc_bridge'] = 'proto_affine_v1'
                u = unit(group, job)
                require(u not in result, 'Duplicate expected unit')
                result[u] = expected_rows(group, job)
                if group=='hybrid':
                    mopc_job=dict(job,methods=['mopc_hr'],mopc_stabilization=mopc_stabilization)
                    result[u] |= expected_rows(group,mopc_job)
    return result


def coverage(expected, available):
    missing_units = missing_rows = 0
    for u, rows in expected.items():
        actual = available.get(u, set())
        if not rows.issubset(actual):
            missing_units += 1
            missing_rows += len(rows - actual)
    return dict(ready=missing_units == 0, missing_units=missing_units, missing_rows=missing_rows)


def merge_rows(rows):
    chosen = {}
    for row in rows:
        identity = key(row)
        if identity in chosen:
            old = chosen[identity]
            require(all(old.get(field) == row.get(field) for field in
                        ('split_id', 'registered_classes', 'query_count', 'input_condition')), 'Conflicting row metadata')
            if old['prediction'] != row['prediction']:
                with np.load(old['prediction'], allow_pickle=False) as a, np.load(row['prediction'], allow_pickle=False) as b:
                    require(set(a.files) == set(b.files) and all(np.array_equal(a[k], b[k]) for k in a.files),
                            'Conflicting duplicate predictions')
        else:
            chosen[identity] = row
    return list(chosen.values())


def mark_incomplete(path, unstable):
    path = Path(path)
    stamp = (path.stat().st_size, path.stat().st_mtime_ns)
    last, count = unstable.get(str(path), (None, 0))
    count = count + 1 if stamp == last else 1
    unstable[str(path)] = (stamp, count)
    require(count < 3, 'Malformed JSON remained unchanged')


def collect(preparations, expected, unstable=None, mopc_stabilization='none'):
    if unstable is None:
        unstable = {}
    available, entries = {}, {}
    configs=list(Path(preparations).glob('*phase2*clean*/scheduler.json'))
    for config_path in sorted(configs):
        if config_path.parent.name in EXCLUDED_PLANS:
            continue
        try:
            config = read(config_path)
        except json.JSONDecodeError:
            mark_incomplete(config_path, unstable)
            continue
        unstable.pop(str(config_path), None)
        require(config.get('input_condition') == 'clean_ground_pre_satellite_normalized',
                'Unexpected clean plan condition')
        for entry in config['jobs']:
            if not Path(entry['completion']).is_file():
                continue
            job = read(entry['job'])
            u = unit(entry['group'], job)
            if u not in expected:
                continue
            has_mopc=(entry['group']=='native_cil' and job.get('profile')=='mopc_hr') or \
                     (entry['group']=='hybrid' and 'mopc_hr' in job.get('methods',[]))
            if has_mopc and job.get('mopc_stabilization','none')!=mopc_stabilization:
                if entry['group']=='hybrid':
                    require(job.get('methods')==['mopc_hr'], 'Cannot skip mixed-version MoPC job')
                continue
            if u[:2] == ('hybrid','protonet_cda') and job.get('mopc_bridge','original') == 'original' \
               and 'mopc_hr' in job.get('methods',[]):
                require(job['methods'] == ['mopc_hr'], 'Legacy MoPC is mixed with other methods in one job')
                continue
            rows = expected_rows(entry['group'], job)
            require(rows and rows.issubset(expected[u]), 'Completed job differs from retained matrix')
            try:
                valid = completion_matches(entry)
            except json.JSONDecodeError:
                mark_incomplete(entry['completion'], unstable)
                continue
            unstable.pop(entry['completion'], None)
            require(valid, 'Invalid completed prediction')
            available.setdefault(u, set()).update(rows)
            entries.setdefault(u, []).append(entry)
    return available, entries


def assemble_unit(payload):
    u, candidates, wanted = payload
    verified = verify_all({'jobs': candidates}, {'jobs': len(candidates)})
    rows = []
    roots = set()
    for item in verified:
        roots.add((str(Path(item['rt']['data_root']).resolve()), str(Path(item['rt']['clean_root']).resolve())))
        for raw in item['payload']['rows']:
            row = dict(raw, prediction=str((item['folder'] / raw['prediction']).resolve()))
            rows.append(row)
    require(len(roots) == 1, 'Mixed data bindings')
    rows = merge_rows(rows)
    require({key(row) for row in rows} == wanted, 'Incomplete combined predictions')
    sample = verified[0]
    sample['payload'] = dict(sample['payload'], rows=rows)
    return sample


def validate_before_truth(expected, entries):
    for u, wanted in expected.items():
        validate_prediction_rows(assemble_unit((u, entries[u], wanted)))


def validate_prediction_rows(item):
    from .mopc_identity import require_author_mopc_row
    cap = Path(item['rt']['data_root']) / item['job']['capsule']
    specs = {s['split_id']: s for s in (read(p) for p in (cap / 'splits').glob('*.json'))}
    registered = {}
    for row in item['payload']['rows']:
        require_author_mopc_row(row)
        spec = specs[row['split_id']]
        require((spec['k'], spec['new_count']) == (row['k'], row['new_count']) and
                spec['registered_classes'] == row['registered_classes'], 'Wrong split/K/N/classes')
        n = int(spec['new_count'])
        if n not in registered:
            registered[n] = np.load(cap / 'stage_rows' / f'N{n:02d}.npy', allow_pickle=False)
        expected = query_complement(registered[n], np.asarray(spec['support_indices']))
        with np.load(row['prediction'], allow_pickle=False) as prediction:
            q = prediction['query_indices']
            values = prediction['predicted_indices']
            require(q.ndim == 1 and np.issubdtype(q.dtype, np.integer) and
                    np.array_equal(q, expected) and len(q) == spec['query_count'], 'Wrong query rows')
            require(values.ndim == 1 and len(values) == len(q) and np.issubdtype(values.dtype, np.integer) and
                    (values >= 0).all() and (values < len(spec['registered_classes'])).all(),
                    'Wrong predicted classes')
            if 'old_restricted_indices' in prediction:
                restricted = prediction['old_restricted_indices']
                require(restricted.ndim == 1 and len(restricted) == len(q) and
                        np.issubdtype(restricted.dtype, np.integer) and
                        (restricted >= 0).all() and (restricted < 6).all(), 'Wrong old-class predictions')


def score_unit(payload):
    u, candidates, wanted, output = payload
    sample = assemble_unit((u, candidates, wanted))
    target = Path(output) / ('--'.join((u[0], u[1], u[2].replace('/', '_'), str(u[3]))))
    score_job(sample, target)
    return len(sample['payload']['rows'])


def run(preparations, output, interval, workers, mopc_stabilization='none'):
    require(interval >= 60 and 1 <= workers <= 4, 'Wrong polling or scoring resources')
    root = Path(output).resolve()
    require(not root.exists(), 'Automatic scoring output already exists')
    root.mkdir(parents=True)
    write(root / 'owner.json', dict(pid=os.getpid(), cwd=str(Path.cwd()),
          preparations=str(Path(preparations).resolve()), workers=workers, mopc_stabilization=mopc_stabilization))
    opened = False
    try:
        expected = expected_units(tasks(),mopc_stabilization)
        unstable = {}
        while True:
            available, entries = collect(preparations, expected, unstable, mopc_stabilization)
            status = coverage(expected, available)
            atomic(root / 'state.json', dict(status='READY' if status['ready'] else 'WAITING',
                   units=len(expected), covered_units=len(available), truth_opened=False, **status))
            if status['ready']:
                break
            time.sleep(interval)
        validate_before_truth(expected, entries)
        payloads = ((u, entries[u], rows, str(root / 'scores')) for u, rows in expected.items())
        opened = True
        atomic(root / 'state.json', dict(status='SCORING', units=len(expected), truth_opened=True))
        with ProcessPoolExecutor(max_workers=workers) as pool:
            counts = list(pool.map(score_unit, payloads, chunksize=1))
        write(root / 'complete.json', dict(status='SCORED_COMPLETE', units=len(counts), rows=sum(counts),
              truth_last=True, target_feedback=False))
        atomic(root / 'state.json', dict(status='SCORED_COMPLETE', units=len(counts), rows=sum(counts), truth_opened=True))
    except Exception as exc:
        atomic(root / 'state.json', dict(status='FAILED', error=str(exc), truth_opened=opened))
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preparations', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--interval', type=int, default=300)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--mopc-stabilization', choices=['none'], default='none')
    p.add_argument('--check-only', action='store_true', help='Inspect coverage without opening truth or starting a watcher')
    a = p.parse_args()
    if a.check_only:
        expected = expected_units(tasks(),a.mopc_stabilization)
        available, _ = collect(a.preparations, expected, mopc_stabilization=a.mopc_stabilization)
        print(json.dumps(dict(units=len(expected), covered_units=len(available),
                              truth_opened=False, **coverage(expected, available))))
        return
    run(a.preparations, a.output, a.interval, a.workers, a.mopc_stabilization)


if __name__ == '__main__':
    main()
