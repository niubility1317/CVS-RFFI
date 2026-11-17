import json
from pathlib import Path
from unittest.mock import patch

import pytest

from experiments.pamf_repl_factorial_recovery import plan, runner


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def test_scoped_configuration_uses_new_run_default(tmp_path):
    with plan.scoped():
        with pytest.raises(FileNotFoundError):
            runner.base_plan.configuration({}, plan.candidate_rows()[0], tmp_path, 'commit')


def test_recovery_reuses_only_exact_frozen_rows(tmp_path):
    project = tmp_path / 'project'
    old = project / 'runs' / plan.PREVIOUS_RUN
    root = project / 'runs' / plan.RUN
    rows = plan.candidate_rows()
    statuses = ['COMPLETE', 'COMPLETE', 'FAILED', 'QUEUED']
    state = {'status': 'PARTIAL_TECHNICAL_FAILURE',
             'jobs': {'train__' + row['row_id']: {'status': status}
                      for row, status in zip(rows, statuses)}}
    put(old / 'pipeline_state.json', state)
    old_sources = {}
    for row in rows[:2]:
        rid = row['row_id']; folder = old / 'F' / rid / 'source'
        cfg = {'row_id': rid, 'run_id': plan.PREVIOUS_RUN,
               'tuning_recipe': plan.RECIPES[row['recipe']]}
        put(folder / 'complete.json', {'status': 'SOURCE_FROZEN', 'epoch': 50,
            'target_access': False, 'U_truth_read': False, 'config': cfg})
        (folder / 'final_ssdg.pth').touch()
        for epoch in (30, 50):
            path = folder / f'segment{epoch}' / 'U_pseudo_observations.jsonl.gz'
            path.parent.mkdir(); path.touch()
        old_sources[rid] = {'source_output': str(folder)}
    put(old / 'sources.json', old_sources)

    def fake_prepare(rt):
        jobs = []; sources = {}
        for row in rows:
            rid = row['row_id']; folder = root / 'F' / rid / 'source'
            sources[rid] = {'source_output': str(folder)}
            for kind in ('train', 'predict', 'u_score'):
                cfg = root / 'configs' / (kind + '__' + rid + '.json')
                put(cfg, {'source_output': str(folder)})
                jobs.append({'id': kind + '__' + rid, 'kind': 'score' if kind == 'u_score' else kind,
                             'config': str(cfg), 'dependencies': []})
        put(root / 'sources.json', sources); put(root / 'jobs.json', jobs)
        return jobs

    with patch.object(runner.scored, 'prepare', side_effect=fake_prepare):
        jobs = runner.prepare({'project': str(project), 'run_root': str(root)})
    ids = {j['id'] for j in jobs}
    assert all('train__' + row['row_id'] not in ids for row in rows[:2])
    assert all('train__' + row['row_id'] in ids for row in rows[2:])
    for row in rows[:2]:
        rid = row['row_id']; source = old_sources[rid]['source_output']
        for kind in ('predict', 'u_score'):
            job = next(j for j in jobs if j['id'] == kind + '__' + rid)
            assert json.loads(Path(job['config']).read_text())['source_output'] == source
            assert job['dependencies'] == [str(Path(source) / 'complete.json')]

    done_path = Path(old_sources[rows[0]['row_id']]['source_output']) / 'complete.json'
    done = json.loads(done_path.read_text()); done['target_access'] = True; put(done_path, done)
    with patch.object(runner.scored, 'prepare', side_effect=fake_prepare):
        with pytest.raises(ValueError, match='source-only'):
            runner.prepare({'project': str(project), 'run_root': str(root)})
