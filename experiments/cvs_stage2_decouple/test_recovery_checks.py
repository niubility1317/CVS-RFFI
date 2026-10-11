"""Synthetic recovery-state checks; never read source IQ or target/truth."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from . import design as d
from . import test_recovery as r


def rejects(fn, exceptions=(ValueError, RuntimeError, FileExistsError)):
    try:
        fn()
    except exceptions:
        return
    raise AssertionError('Unsafe recovery state accepted')


def failed_metadata(c):
    out = d.BASE / c['row_id'] / 'prediction'
    d.write(out / 'provenance.json', dict(config=c, checkpoint=str(Path(c['output_root']) / 'final_ssdg.pth'),
        truth_read=False, query_fit=False))
    d.write(out / 'resolved_config.json', dict(row_id=c['row_id']))
    return out


def source_fixture():
    d.write(d.BASE / 'dispatcher.json', dict(pid=11, start_ticks=100))
    launches = []
    for index, row in enumerate(d.rows()):
        c = d.config(row)
        launches.append(dict(kind='train', identifier=c['row_id'], pid=100 + index, start_ticks=1000 + index))
        d.write(d.BASE / 'configs' / (c['row_id'] + '.json'), c)
        source = Path(c['output_root'])
        d.write(source / 'completion.json', dict(status='SOURCE_TRAINED', config=c, epoch=40,
            optimizer_steps=8880, logged_steps=8880, target_access=False, checkpoint_selection='final_only'))
        (source / 'final_ssdg.pth').write_bytes(b'fixture checkpoint; never loaded')
        d.write(d.BASE / c['row_id'] / 'source_frozen.json', dict(status='SOURCE_FROZEN', config=c,
            epoch=40, optimizer_steps=8880, target_access=False, checkpoint=str(source / 'final_ssdg.pth')))
    d.write(d.BASE / 'launch.json', dict(rows=launches))


def run():
    with patch.object(r, 'process', return_value=dict(start_ticks=10)):
        rejects(lambda: r.require_exited(dict(pid=1, start_ticks=10)))
        assert r.require_exited(dict(pid=1, start_ticks=9))['status'] == 'EXITED_PID_REUSED'
    with patch.object(r, 'process', return_value=None):
        assert r.require_exited(dict(pid=1, start_ticks=10))['status'] == 'EXITED'
    with tempfile.TemporaryDirectory() as temp:
        base = Path(temp)
        with patch.object(d, 'BASE', base / 'source_run'), patch.object(r, 'RECOVERY_BASE', base / 'recovery_run'):
            r.RECOVERY_BASE.mkdir()
            source_fixture()
            with patch.object(r, 'process', return_value=None):
                assert r.preflight()['rows'] == 12
                row = d.rows()[0]
                c = d.config(row)
                path = Path(c['output_root']) / 'completion.json'
                old = d.read(path)
                d.write(path, dict(old, optimizer_steps=8879))
                rejects(r.preflight)
                d.write(path, old)
            out = failed_metadata(c)
            before = {p.name: p.read_bytes() for p in out.iterdir()}
            assert r.reconcile_prediction(c) == 'ARCHIVED_METADATA_ONLY'
            archive = r.RECOVERY_BASE / c['row_id'] / 'recovery_archive/prediction-r02-failed'
            assert not out.exists() and before == {p.name: p.read_bytes() for p in archive.iterdir()}
            assert r.reconcile_prediction(c) == 'MISSING'
            out = failed_metadata(c)
            rejects(lambda: r.reconcile_prediction(c))  # immutable archive collision
            (out / 'predictions.npz').write_bytes(b'partial prediction preserved')
            rejects(lambda: r.reconcile_prediction(c))
            assert (out / 'predictions.npz').read_bytes() == b'partial prediction preserved'
            d.write(out / 'complete.json', {'fixture': True})
            with patch.object(r.evaluate, 'prediction_preflight') as validation:
                assert r.reconcile_prediction(c) == 'REUSED_COMPLETE'
                validation.assert_called_once_with([c])
            # A fresh synthetic row follows predict -> validation -> CPU scorer.
            c2 = d.config(d.rows()[1])
            failed_metadata(c2)
            original_failure = d.BASE / c2['row_id'] / 'failure.json'
            d.write(original_failure, dict(status='FAILED', error='original bridge TypeError'))
            original_bytes = original_failure.read_bytes()
            events = []
            def predict(row_id):
                assert row_id == c2['row_id']
                events.append('predict')
            def predictions(configs):
                assert configs == [c2]
                events.append('fixed_predictions')
            def score(row_id):
                assert events == ['predict', 'fixed_predictions']
                events.append('independent_score')
            with patch.object(r, 'lock', side_effect=lambda *a, **k: nullcontext()), \
                    patch.object(r, 'await_capacity'), patch.object(r, 'commit', return_value='fixture-commit'), \
                    patch.object(r.evaluate, 'predict', side_effect=predict), \
                    patch.object(r.evaluate, 'prediction_preflight', side_effect=predictions), \
                    patch.object(r, 'score_complete', return_value=False), \
                    patch.object(r, 'independent_score', side_effect=score), patch.object(r, 'record'):
                r.worker(c2['row_id'])
            assert events == ['predict', 'fixed_predictions', 'independent_score']
            assert original_failure.read_bytes() == original_bytes
            resolution = d.read(d.BASE / c2['row_id'] / 'failure_resolution.json')
            assert resolution['status'] == 'RESOLVED' and resolution['original_failure_preserved']
            assert d.read(r.RECOVERY_BASE / c2['row_id'] / 'completion.json')['execution_commit'] == 'fixture-commit'
            # GPU jobs use the common capacity policy, including external PIDs.
            assert r.choose_gpu({0: dict(pids={1, 2, 3, 4}, free_mb=20000)}) is None
            assert r.choose_gpu({0: dict(pids={1, 2, 3}, free_mb=20000)}) == 0
            # Registration is copied once, with immutable inputs and current commit.
            release = base / 'release'
            (release / 'recovery_registration').mkdir(parents=True)
            d.write(release / 'recovery_registration/experiment.json', dict(run_id=r.RECOVERY_RUN))
            for name in ('report.md', 'events.jsonl'):
                (release / 'recovery_registration' / name).write_text('fixture\n', encoding='utf-8')
            with patch.object(d, 'ROOT', release):
                r.initialize_registration()
                assert d.read(r.RECOVERY_BASE / 'experiment.json')['run_id'] == r.RECOVERY_RUN
    return dict(status='PASS', scope='synthetic_state_only', target_access=False,
        old_owners_must_exit=True, pid_reuse_preserved=True, all_twelve_sources_fixed_E40=True,
        metadata_archive_preserves_bytes=True, partial_or_collision_never_overwritten=True,
        complete_prediction_reused=True, prediction_before_independent_scoring=True,
        original_failures_preserved=True, four_process_capacity=True,
        independent_recovery_registration=True, current_release_commit_recorded=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = run()
    if args.output:
        d.write(args.output, result)
    print(json.dumps(result))
