import json
from pathlib import Path
import pytest
from experiments.pamf_physics_route import prepare as module
from experiments.pamf_physics_route.plan import RUN, REPLICATION_RUN, candidate_rows, configuration
from experiments.pamf_physics_route.source import validate_source
from experiments.pamf_physics_route.runner import worker


@pytest.mark.parametrize('run_id', [RUN, REPLICATION_RUN])
def test_practical_plan_has_no_weak_dependency_and_scores_after_predictions(tmp_path, monkeypatch, run_id):
    def put(path, obj):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj), encoding='utf-8')

    previous = tmp_path / 'runs' / module.PRIOR
    put(previous / 'pipeline_state.json', {'status': 'ANALYZED'})
    matched = tmp_path / 'matched'
    put(matched / 'complete.json', {'status': 'SCORED_COMPLETE'})
    put(previous / 'sources.json', {'ptbr-beta05-s2026092701':
        {'source_output': 'frozen', 'predictions': 'frozen', 'scores': str(matched)}})
    tests = {t: {'views_root': 'available'} for t in module.TASKS}
    ctx = {'existing': {},
           'handles': {'F0': {'source_contract': 'contract', 'classes': list('ABCDEF'), 'tests': tests}},
           'caches': {'F0': {'L': {'path': 'frozen'}, 'LU': {'path': 'frozen'}}}}
    rows = candidate_rows(run_id)
    expected = {'ptbr-beta05-s2026092701'} if run_id == RUN else set()
    for row in rows:
        seed = row['model_seed']
        baseline = tmp_path / f'baseline-{seed}'
        tri = (tmp_path / 'runs/20261027-phase1-pamf-repl-shuffled-views-m16-r02/readouts'
               / f'main_split-repl-shuffled_time-3view-s{seed}')
        for folder in (baseline, tri):
            put(folder / 'scores/complete.json', {'status': 'SCORED_COMPLETE'})
            put(folder / 'predictions/complete.json', {'status': 'ALL_PREDICTIONS_COMPLETE', 'truth_read': False})
        ctx['existing'][f'F0-ordinary_PL-s{seed}'] = dict(predictions=str(baseline / 'predictions'),
                                                          scores=str(baseline / 'scores'))
        expected.update((f'baseline_pl-s{seed}', f'tri_sigtime_ssl-s{seed}', row['row_id']))
    context = tmp_path / 'context.json'
    put(context, ctx)
    run = tmp_path / 'runs' / run_id
    monkeypatch.setattr(module, 'configuration', lambda context, row, project, commit, *, run_id: {
        'output_root': str(run / 'F' / row['row_id'] / 'source'), 'paper_row': {'family': 'factorial'},
        'source_cache': 'source-only'})
    jobs = module.prepare({'context': str(context), 'run_root': str(run), 'project': str(tmp_path),
                           'run_id': run_id, 'code_commit': 'test', 'frozen_sources': {}})
    kinds = [j['kind'] for j in jobs]
    if run_id == RUN:
        assert kinds == ['train', 'source_diagnostic', 'predict', 'score']
        assert jobs[1]['nonblocking']
    else:
        assert kinds == ['train', 'predict', 'score'] * 2
    predict = {j['completion'] for j in jobs if j['kind'] == 'predict'}
    for score in (j for j in jobs if j['kind'] == 'score'):
        assert set(score['dependencies']) == predict
    sources = json.loads((run / 'sources.json').read_text(encoding='utf-8'))
    assert set(sources) == expected
    assert not any('weak' in json.dumps(x).lower() for x in jobs)


def test_replications_use_their_own_paper_rows_and_e20_prefixes(monkeypatch):
    from experiments.pamf_paper_completion import plan as paper
    monkeypatch.setattr(paper, 'configuration', lambda context, row, project, commit:
                        dict(model_seed=row['model_seed'], paper_row=row))
    rows = candidate_rows(REPLICATION_RUN)
    context = {'sources': {str(r['model_seed']): {'F21': f'/parent/s{r["model_seed"]}'} for r in rows}}
    for row in rows:
        seed = row['model_seed']
        c = configuration(context, row, '/project', 'commit', run_id=REPLICATION_RUN)
        assert c['model_seed'] == c['paper_row']['model_seed'] == seed
        assert c['shared_prefix'] == f'/parent/s{seed}/segment20'
        assert c['output_root'].endswith(f'/F/ptbr-route-s{seed}/source')


def test_source_rejects_target_handles_and_wrong_route():
    row = candidate_rows()[0]
    c = dict(physical_row=row, run_id=RUN, row_id=row['row_id'], model_seed=row['model_seed'],
             source_rxs=row['source_rxs'], source_days=row['source_days'], num_classes=6,
             epochs=50, optimizer_updates=11100, ablation='F21', U_use=True, own_EMA=False,
             native_U_gate=False, posthoc_head_fit=False,
             paper_row=dict(structure='full', permuted=False, signature_views=3),
             hard_pl=dict(method='HPL2', threshold=.95, weight=1.), hard_pl_label_smoothing=0.,
             physical_relation=dict(beta=.5, l_weight=.05, u_weight=.05),
             reference_route_rule='reference_delay_stable_only', algorithm_id='pamf_ptbr_route_v1',
             output_root='/project/runs/' + RUN + '/F/' + row['row_id'] + '/source')
    assert validate_source(c) == row
    with pytest.raises(ValueError):
        validate_source(dict(c, reference_route_rule='always'))
    with pytest.raises(ValueError, match='target'):
        validate_source(dict(c, target_truth='/forbidden'))
    with pytest.raises(ValueError, match='practical'):
        worker({}, 'weak_predict')
