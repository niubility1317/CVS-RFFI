"""Focused capacity, freeze, checkpoint and clean truth-last routing checks."""
import argparse
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from . import design as d


def rejected(callable_, exceptions=(ValueError, FileNotFoundError)):
    try:
        callable_()
    except exceptions:
        return
    raise AssertionError('Invalid state accepted')


def checks():
    import numpy as np
    from . import dispatch as q, evaluate as e
    from .publish import REMOTE, INSPECT
    rows = d.rows()
    assert len(rows) == 12 and len({r['row_id'] for r in rows}) == 12
    assert set(d.ARMS) == {'C0', 'C1', 'C2', 'C3'} and len(d.SEEDS) == 3
    for row in rows:
        d.validate(d.config(row))
    assert q.choose_gpu({0: dict(pids={1, 2, 3, 4}, free_mb=50000)}) is None
    assert q.choose_gpu({0: dict(pids={1, 2, 3}, free_mb=12000)}) == 0
    assert q.choose_gpu({0: dict(pids=set(), free_mb=11999)}) is None
    assert q.choose_gpu({0: dict(pids={1, 2}, free_mb=12000), 1: dict(pids={3}, free_mb=20000)}) == 1
    assert e.VIEWS == ('clean',) and e.COUNT == 168000
    with tempfile.TemporaryDirectory() as folder:
        with patch.object(d, 'BASE', Path(folder)):
            row = rows[0]
            c = d.config(row)
            rejected(lambda: e.snapshot(row['row_id']))
            d.write(d.BASE / row['row_id'] / 'source_frozen.json',
                    dict(status='SOURCE_FROZEN', config=c, target_access=False))
            assert e.snapshot(row['row_id']) == [c]
            ck = dict(epoch=40, run_id=c['run_id'], candidate_id=c['row_id'], config=c,
                      parent_checkpoint=c['parent_checkpoint'], checkpoint_selection='final_only', model={}, baseline_args={})
            e.checkpoint_provenance(c, ck)
            for key, invalid in [('epoch', 39), ('parent_checkpoint', 'different'), ('checkpoint_selection', 'best_target')]:
                rejected(lambda key=key, invalid=invalid: e.checkpoint_provenance(c, dict(ck, **{key: invalid})))
            ids = np.asarray(['test-' + str(i) for i in range(e.COUNT)])
            with patch.object(e, 'physical_ids', lambda: ids):
                rejected(lambda: e.prediction_preflight([c]))
                out = d.BASE / c['row_id'] / 'prediction'
                d.write(out / 'complete.json', dict(status='PREDICTIONS_COMPLETE', count=e.COUNT,
                    views=['clean'], truth_read=False, query_fit=False))
                d.write(out / 'provenance.json', dict(config=c, parent_checkpoint=c['parent_checkpoint']))
                np.savez(out / 'predictions.npz', ids=ids, clean=np.zeros(e.COUNT, np.int64))
                assert np.array_equal(e.prediction_preflight([c]), ids)
                np.savez(out / 'predictions.npz', ids=ids[::-1], clean=np.zeros(e.COUNT, np.int64))
                rejected(lambda: e.prediction_preflight([c]))
            # Scorer must not touch target truth when prediction preflight rejects.
            with patch.object(e, 'prediction_preflight', side_effect=ValueError('incomplete')), patch.object(e.d, 'read', wraps=d.read) as read:
                rejected(lambda: e.score(c['row_id']))
                assert not any(call.args[0] == d.TRUTH for call in read.call_args_list)
    # Complete twelve-row synthetic score exercises real stratification, recount,
    # report creation, three-seed goal assessment, and parent pairing together.
    score_matrix_check()
    fixture = []
    for arm, value in [('P0', .81), ('C0', .82), ('C1', .825), ('C2', .83), ('C3', .84)]:
        for seed in d.SEEDS:
            for dimension in ('overall', 'receiver'):
                fixture.append(dict(row_id=arm + '-s' + str(seed), arm=arm, model_seed=seed,
                    dimension=dimension, accuracy=value, macro_f1=value, view='clean'))
    summary = e.summarize(fixture)
    assert len(summary) == 5 and all(r['complete_three_seeds'] for r in summary)
    pair = next(r for r in e.paired_results(fixture) if r['treatment'] == 'C3' and r['control'] == 'C0')
    assert abs(pair['mean_difference'] - .02) < 1e-12 and pair['seed_count'] == 3
    partial = e.summarize([r for r in fixture if r['model_seed'] != d.SEEDS[-1]])
    assert not any(r['complete_three_seeds'] for r in partial)
    scored = dict(status='SCORED_COMPLETE', models=12, views=1, metric_records=168, day_metric_records=48,
                  query_count_per_view=168000, prediction_count=2016000, truth_last=True, independent_recount='VERIFIED')
    q.require_scored(scored, 12)
    rejected(lambda: q.require_scored(dict(scored, views=7), 12))
    compile(REMOTE.replace('CONFIG', repr(dict(project='example'))), 'remote-release', 'exec')
    compile(INSPECT.replace('CONFIG', repr(dict(project='example'))), 'remote-inspect', 'exec')
    return dict(status='PASS', rows=12, seeds=3, per_gpu_limit=4, clean_only=True,
        truth_last_rejection=True, checkpoint_parent_and_budget=True, no_target_feedback=True,
        paired_summary=True, partial_seed_not_complete=True, synthetic_full_matrix_scored=True)


def score_matrix_check():
    import numpy as np
    from . import evaluate as e, dispatch as q
    labels, receivers, dates = [], [], []
    for label in range(6):
        for receiver in e.HELDOUT_RX:
            for date in e.TARGET_DAYS:
                labels.extend([label] * 1000)
                receivers.extend([receiver] * 1000)
                dates.extend([date] * 1000)
    ids = np.asarray(['fixture-' + str(i) for i in range(e.COUNT)])
    y = np.asarray(labels, dtype=np.int64)
    truth = {pid: dict(label=labels[i], receiver=receivers[i], day=dates[i]) for i, pid in enumerate(ids)}
    with tempfile.TemporaryDirectory() as folder, patch.object(d, 'BASE', Path(folder)):
        configs = [d.config(r) for r in d.rows()]
        for c in configs:
            # Only synthetic paths are changed; production configuration stays fixed.
            c['parent_prediction'] = str(d.BASE / 'parent-fixture' / str(c['model_seed']) / 'predictions.npz')
            out = d.BASE / c['row_id'] / 'prediction'
            done = dict(status='PREDICTIONS_COMPLETE', count=e.COUNT, views=['clean'], truth_read=False, query_fit=False)
            d.write(out / 'complete.json', done)
            d.write(out / 'provenance.json', dict(config=c, parent_checkpoint=c['parent_checkpoint']))
            np.savez(out / 'predictions.npz', ids=ids, clean=y)
            parent = Path(c['parent_prediction'])
            if not parent.exists():
                d.write(parent.parent / 'complete.json', done)
                d.write(parent.parent / 'provenance.json', dict(status='VERIFIED', checkpoint=c['parent_checkpoint'],
                    config=dict(model_seed=c['model_seed'], arm='native'), truth_read=False, query_fit=False))
                np.savez(parent, ids=ids, clean=y)
        original_read = d.read
        def read(path):
            return truth if path == d.TRUTH else original_read(path)
        with patch.object(e, 'snapshot', return_value=configs), patch.object(e, 'physical_ids', return_value=ids), \
                patch.object(e, 'source_resources', return_value=dict(scope='synthetic test fixture')), patch.object(d, 'read', side_effect=read):
            e.score()
        q.require_scored(original_read(d.BASE / 'scoring_complete.json'), 12)
        assert len(original_read(d.BASE / 'parent_scores.json')['results']) == 42
        goal = original_read(d.BASE / 'goal_assessment.json')
        assert goal['performance_status'] == 'MET' and goal['mechanism_increment_status'] == 'NOT_POSITIVE'
        summary = original_read(d.BASE / 'summary.json')['results']
        assert len(summary) == 5 and all(r['seed_count'] == 3 and r['accuracy_mean'] == 1 for r in summary)
        assert (d.BASE / 'analysis.md').is_file()


def remote_preflight():
    import torch
    from .common import verify_parent, build_model
    result = checks()
    torch.set_num_threads(2)
    parents = []
    for seed in d.SEEDS:
        c = d.config(next(r for r in d.rows() if r['model_seed'] == seed and r['arm'] == 'C0'))
        provenance = verify_parent(c)
        model = build_model(c, torch.device('cpu'), checkpoint=Path(c['parent_checkpoint']))
        model.eval()
        with torch.no_grad():
            logits = model(torch.zeros(2, 2, 256))
        if logits.shape != (2, 6) or not torch.isfinite(logits).all():
            raise ValueError('Parent strict-load CPU inference failed')
        parents.append(dict(model_seed=seed, checkpoint=c['parent_checkpoint'], strict_load=True,
                            no_query_forward=True, verified=provenance))
        del model
    result.update(parents=parents, target_access=False, remote_parent_preflight='VERIFIED')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--remote-preflight', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = remote_preflight() if args.remote_preflight else checks()
    if args.output:
        d.write(args.output, result)
    print(json.dumps(result))
