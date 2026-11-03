"""Exact pilot prediction reuse and source-auxiliary lookup for the fixed matrix."""
from pathlib import Path
import copy
import os
import shutil

import numpy as np

from .io import read, require, write


def hu_aux_folder(rt, c_method, seed):
    roots = rt.get('hu_aux_roots_by_seed')
    root = roots.get(str(seed)) if roots is not None else rt.get('hu_aux_run_root')
    require(bool(root), 'Missing seed-bound Hu source auxiliary root')
    return Path(root)/'source'/f'{c_method}-s{seed}'


def reuse_pilot_prediction(rt, job, manifest, spec, expected_query, output):
    """Reuse only the fixed pilot row after checking its scientific inputs and query order."""
    original = Path(job['reuse_prediction'])
    done, index = read(original/'complete.json'), read(original/'predictions.json')
    require(done['status'] == 'PREDICTIONS_COMPLETE' and done['truth_read'] is False and
            done['query_fit'] is False and done['rows'] == 1, 'Pilot prediction not frozen')
    require(index['capsule_id'] == done['capsule_id'] == manifest['capsule_id'] and
            index['capsule'] == job['capsule'] and index['profile'] == job['b_method'] and
            index['c_method'] == job['c_method'] and index['model_seed'] == job['seed'] and
            len(index['rows']) == 1, 'Pilot prediction binding mismatch')
    row = copy.deepcopy(index['rows'][0])
    arm = job['b_method']+'__'+job['c_method']+'__C_SOURCE'
    require((row['k'], row['new_count']) == (5, 20) and row['split_id'] == spec['split_id'] and
            row['registered_classes'] == spec['registered_classes'] and row['query_count'] == spec['query_count'] and
            row['stage'] == 'C' and row['arm'] == arm and row['input_path'] == 'P0', 'Pilot split binding mismatch')
    expected_checkpoint = Path(rt['c_source_roots'][job['c_method']])/'source'/f'{job["c_method"]}-s{job["seed"]}'/'source.pt'
    require(row['resources']['B_and_C']['source_checkpoint'] == str(expected_checkpoint), 'Pilot source checkpoint mismatch')
    name = row['prediction']
    require(Path(name).name == name and (original/name).is_file(), 'Missing pilot prediction')
    with np.load(original/name, allow_pickle=False) as data:
        query, predicted, old = data['query_indices'], data['predicted_indices'], data['old_restricted_indices']
        require(np.array_equal(query, expected_query) and len(query) == row['query_count'], 'Pilot query mismatch')
        require(predicted.shape == old.shape == query.shape and np.issubdtype(predicted.dtype, np.integer) and
                np.issubdtype(old.dtype, np.integer) and bool((predicted >= 0).all()) and
                bool((predicted < len(row['registered_classes'])).all()) and bool((old >= 0).all()) and
                bool((old < 6).all()), 'Invalid pilot prediction values')
    try:
        os.link(original/name, output/name)
    except OSError:
        shutil.copyfile(original/name, output/name)
    row['reuse'] = dict(prediction_source=str(original), score_source=job['reuse_score'],
                        B_state_across_N='not_verified_against_new_B')
    row['resources']['B_fit_shared_across_N'] = None
    row['resources']['B_state_scope'] = 'pilot_single_N_state_not_preserved'
    return row


def score_with_pilot_reuse(rt, job):
    """Score only new frozen rows and merge the independently scored pilot row."""
    from .episodic_score import run
    from .milkr_r42_suite import common_query_score

    root = Path(rt['run_root'])
    require(read(root/'predictions_complete.json')['status'] == 'ALL_PREDICTIONS_FIXED',
            'Truth barrier not closed')
    prediction = Path(job['output'])
    index = read(prediction/'predictions.json')
    complete = read(prediction/'complete.json')
    require(complete['status'] == 'PREDICTIONS_COMPLETE' and complete['truth_read'] is False and
            complete['rows'] == len(index['rows']), 'Prediction not frozen')
    reused = [r for r in index['rows'] if r.get('reuse')]
    pending = [r for r in index['rows'] if not r.get('reuse')]
    require(len(reused) == 1 and pending and '/MIX' in job['capsule'], 'Wrong pilot score reuse matrix')
    reused_row = reused[0]
    require(reused_row['reuse']['score_source'] == job['reuse_score'] and
            reused_row['reuse']['prediction_source'] == job['reuse_prediction'], 'Wrong pilot score path')
    oldroot = Path(job['reuse_score'])
    olddone = read(oldroot/'complete.json')
    natural, common = read(oldroot/'metrics.json'), read(oldroot/'common_query_metrics.json')
    require(olddone['status'] == 'SCORED_COMPLETE' and olddone['rows'] == 1 and olddone['truth_last'] is True and
            natural['truth_last'] is True and natural['target_feedback'] is False and
            common['truth_last'] is True and common['target_feedback'] is False and
            natural['capsule_id'] == index['capsule_id'] and natural['model_seed'] == index['model_seed'] and
            len(natural['rows']) == len(common['rows']) == 1, 'Pilot score not closed')
    keys = ('split_id', 'stage', 'input_path', 'arm', 'prediction', 'k', 'new_count', 'registered_classes', 'query_count')
    for old in (natural['rows'][0], common['rows'][0]):
        require(all(old[k] == reused_row[k] for k in keys), 'Pilot score/prediction mismatch')
    partial = root/'score_inputs'/job['key']
    partial.mkdir(parents=True, exist_ok=False)
    for row in pending:
        name = row['prediction']
        require(Path(name).name == name, 'Invalid prediction name')
        try:
            os.link(prediction/name, partial/name)
        except OSError:
            shutil.copyfile(prediction/name, partial/name)
    filtered = dict(index, rows=pending)
    write(partial/'predictions.json', filtered)
    write(partial/'complete.json', dict(complete, rows=len(pending)))
    output = root/'scores'/job['key']
    run(rt, dict(prediction=str(partial), output=str(output)))
    common_query_score(rt, job, partial, filtered, output)
    fresh, fresh_common = read(output/'metrics.json'), read(output/'common_query_metrics.json')
    require(fresh_common['common_query_counts'] == common['common_query_counts'], 'Pilot common-query mismatch')
    def merge(fresh_rows, old_row):
        merged = {(r['k'], r['new_count']): r for r in fresh_rows}
        require(len(merged) == len(pending), 'Duplicate fresh score row')
        old = dict(old_row, reuse=reused_row['reuse'], resources=reused_row['resources'])
        key = old['k'], old['new_count']
        require(key not in merged, 'Duplicate pilot score row')
        merged[key] = old
        require(set(merged) == {(r['k'], r['new_count']) for r in index['rows']}, 'Incomplete merged scores')
        return [merged[r['k'], r['new_count']] for r in index['rows']]
    fresh['rows'] = merge(fresh['rows'], natural['rows'][0])
    fresh_common['rows'] = merge(fresh_common['rows'], common['rows'][0])
    write(output/'metrics.json', fresh, exclusive=False)
    write(output/'common_query_metrics.json', fresh_common, exclusive=False)
    write(output/'complete.json', dict(status='SCORED_COMPLETE', rows=len(index['rows']),
        truth_last=True, target_feedback=False, reused_rows=1), exclusive=False)
