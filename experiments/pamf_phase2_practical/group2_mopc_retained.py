"""Classify a drained MoPC owner without changing retained artifacts.

This is an inventory helper only: it never starts work or opens the scoring
barrier. The caller must separately establish that a continuation is safe.
"""
from pathlib import Path

from .io import read, require
from .group2_schedule import closed_prediction


def partition_retained_mopc(previous, state, desired):
    previous = Path(previous)
    require(state['status'] == 'FAILED' and not state['active'], 'Parent has not drained')
    require(state['total'] == len(desired) and bool(state['failed']), 'Wrong parent task count')
    require(len({job['key'] for job in desired}) == len(desired), 'Duplicate task keys')
    failed = {item['index']: item for item in state['failed']}
    require(len(failed) == len(state['failed']) and all(0 <= i < len(desired) for i in failed),
            'Invalid failed task indices')
    result = dict(completed=[], quarantined=[], pending=[])
    expected_outputs, expected_launches = set(), set()
    fields = ('key', 'profile', 'seed', 'capsule', 'methods', 'points', 'load_B_for_mopc', 'trace_registration')
    for i, intended in enumerate(desired):
        actual = read(previous/'jobs'/f'{i:05d}.json')
        key = intended['key']
        output = previous/'predictions'/key
        require(all(actual.get(field) == intended.get(field) for field in fields) and
                actual['methods'] == ['mopc_hr'] and bool(actual.get('load_B_for_mopc')) and
                Path(actual['output']) == output, 'Parent or saved B binding differs')
        launch = previous/'launches'/f'worker-{i:05d}.json'
        if i in failed:
            item = failed[i]
            require(item['key'] == key and item.get('exit_code') != 0 and launch.is_file() and
                    output.is_dir() and not (output/'complete.json').exists(), 'Invalid retained failure')
            log = (previous/'logs'/f'worker-{i:05d}.log').read_text(encoding='utf-8')
            require('REGISTRATION_FAILURE' in log and 'nonfinite' in log.lower(), 'Unclassified retained failure')
            result['quarantined'].append(actual)
            expected_outputs.add(key); expected_launches.add(launch.name)
        elif (output/'complete.json').is_file():
            require(launch.is_file(), 'Completed task was not launched')
            closed_prediction(actual, len(actual['points']))
            index = read(output/'predictions.json')
            require(all(row['resources'].get('B_state_ref') ==
                        str(Path(actual['load_B_for_mopc'])/f"B-K{int(row['k'])}.pt")
                        for row in index['rows']), 'Completed saved B binding differs')
            result['completed'].append(actual)
            expected_outputs.add(key); expected_launches.add(launch.name)
        else:
            log = previous/'logs'/f'worker-{i:05d}.log'
            require(not output.exists() and not launch.exists() and not log.exists(), 'Unclassified started task')
            result['pending'].append(intended)
    require(len(result['completed']) == state['completed'], 'Completed task count differs')
    require({path.name for path in (previous/'predictions').iterdir()} == expected_outputs and
            {path.name for path in (previous/'launches').glob('worker-*.json')} == expected_launches,
            'Unclassified retained output or launch')
    return result
