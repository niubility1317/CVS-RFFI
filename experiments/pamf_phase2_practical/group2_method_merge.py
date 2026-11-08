"""Check method-level prediction indices before a later immutable merge.

This module does not open query, truth or scores, and never declares a global
prediction barrier. Physical materialization is a separate later operation.
"""
from pathlib import Path
import os

import numpy as np

from .io import read, require, write


def combine_indices(job, subset_index, mopc_index, subset_output):
    for index in (subset_index, mopc_index):
        require(index['capsule'] == job['capsule'] and index['model_seed'] == job['seed'] and
                index['profile'] == job['profile'], 'Method index job binding differs')
    require(subset_index['capsule_id'] == mopc_index['capsule_id'], 'Method capsule binding differs')
    points = {(int(k), int(n)) for k, n in job['points']}
    profile = job['profile']
    expected_subset = {(k, n, profile+'__'+method) for k, n in points for method in ('orthogonal', 'lora')}
    expected_mopc = {(k, n, profile+'__mopc_hr') for k, n in points}
    combined = []
    seen_points = {}
    for index, expected in ((subset_index, expected_subset), (mopc_index, expected_mopc)):
        rows = index['rows']
        actual = [(int(r['k']), int(r['new_count']), r['arm']) for r in rows]
        require(len(actual) == len(expected) and set(actual) == expected, 'Method rows incomplete or duplicated')
        for row in rows:
            point = (int(row['k']), int(row['new_count']))
            expected_b = str(Path(subset_output)/f"B-K{point[0]}.pt")
            require(row['resources'].get('B_state_ref') == expected_b, 'Method did not use the same B state')
            point_binding = (row['split_id'], tuple(row['registered_classes']), int(row['query_count']))
            require(point not in seen_points or seen_points[point] == point_binding,
                    'Method split or query binding differs')
            seen_points[point] = point_binding
            combined.append(dict(row))
    return dict(capsule=job['capsule'], capsule_id=subset_index['capsule_id'], profile=profile,
                model_seed=job['seed'], rows=combined)


def materialize_merge(job, subset_output, mopc_output, output):
    """Link three already-fixed methods for one job; never open target truth."""
    from .group2_schedule import closed_prediction

    require(job['methods'] == ['mopc_hr', 'orthogonal', 'lora'] and not job.get('save_B_for_mopc')
            and not job.get('load_B_for_mopc'), 'Merge requires a full-matrix job')
    subset_output, mopc_output, output = map(Path, (subset_output, mopc_output, output))
    subset_job = dict(job, output=str(subset_output), methods=['orthogonal', 'lora'], save_B_for_mopc=True)
    mopc_job = dict(job, output=str(mopc_output), methods=['mopc_hr'], load_B_for_mopc=str(subset_output))
    closed_prediction(subset_job, len(job['points'])*2)
    closed_prediction(mopc_job, len(job['points']))
    left, right = read(subset_output/'predictions.json'), read(mopc_output/'predictions.json')
    combined = combine_indices(job, left, right, subset_output)
    names = [row['prediction'] for row in combined['rows']]
    require(len(names) == len(set(names)) and all(Path(n).name == n for n in names),
            'Invalid or duplicate prediction filename')
    query_indices = {}
    for source, index in ((subset_output, left), (mopc_output, right)):
        for row in index['rows']:
            point = (int(row['k']), int(row['new_count']))
            with np.load(source/row['prediction'], allow_pickle=False) as data:
                actual = data['query_indices']
                require(len(actual) == int(row['query_count']), 'Prediction query count differs')
                if point in query_indices:
                    require(np.array_equal(query_indices[point], actual), 'Prediction query indices differ')
                else:
                    query_indices[point] = actual.copy()
    output.mkdir(parents=True, exist_ok=False)
    for source, index in ((subset_output, left), (mopc_output, right)):
        for row in index['rows']:
            os.link(source/row['prediction'], output/row['prediction'])
    write(output/'predictions.json', combined)
    write(output/'complete.json', dict(status='PREDICTIONS_COMPLETE', capsule_id=combined['capsule_id'],
          rows=len(combined['rows']), truth_read=False, query_fit=False, profile=job['profile'], model_seed=job['seed']))
