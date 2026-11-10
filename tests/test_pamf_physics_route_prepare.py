import json
from pathlib import Path
import pytest
from experiments.pamf_physics_route import prepare as module
from experiments.pamf_physics_route.plan import RUN, candidate_rows
from experiments.pamf_physics_route.source import validate_source
from experiments.pamf_physics_route.runner import worker


def test_practical_plan_has_no_weak_dependency_and_scores_after_predictions(tmp_path, monkeypatch):
    def put(path, obj):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj), encoding='utf-8')

    previous = tmp_path / 'runs' / module.PRIOR
    put(previous / 'pipeline_state.json', {'status': 'ANALYZED'})
    matched, baseline = tmp_path / 'matched', tmp_path / 'baseline'
    for p in (matched, baseline):
        put(p / 'complete.json', {'status': 'SCORED_COMPLETE'})
    tri = (tmp_path / 'runs/20261027-phase1-pamf-repl-shuffled-views-m16-r02/readouts'
           / 'main_split-repl-shuffled_time-3view-s2026092701')
    put(tri / 'scores/complete.json', {'status': 'SCORED_COMPLETE'})
    put(tri / 'predictions/complete.json', {'status': 'ALL_PREDICTIONS_COMPLETE', 'truth_read': False})
    put(previous / 'sources.json', {'ptbr-beta05-s2026092701':
        {'source_output': 'frozen', 'predictions': 'frozen', 'scores': str(matched)}})
    tests = {t: {'views_root': 'available'} for t in module.TASKS}
    ctx = {'existing': {'F0-ordinary_PL-s2026092701':
             {'source_output': 'frozen', 'predictions': 'frozen', 'scores': str(baseline)}},
           'handles': {'F0': {'source_contract': 'contract', 'classes': list('ABCDEF'), 'tests': tests}},
           'caches': {'F0': {'L': {'path': 'frozen'}, 'LU': {'path': 'frozen'}}}}
    context = tmp_path / 'context.json'
    put(context, ctx)
    run = tmp_path / 'runs' / RUN
    monkeypatch.setattr(module, 'configuration', lambda context, row, project, commit: {
        'output_root': str(run / 'F' / row['row_id'] / 'source'), 'paper_row': {'family': 'factorial'},
        'source_cache': 'source-only'})
    jobs = module.prepare({'context': str(context), 'run_root': str(run), 'project': str(tmp_path),
                           'code_commit': 'test', 'frozen_sources': {}})
    assert [j['kind'] for j in jobs] == ['train', 'source_diagnostic', 'predict', 'score']
    assert next(j for j in jobs if j['kind'] == 'source_diagnostic')['nonblocking']
    score = next(j for j in jobs if j['kind'] == 'score')
    assert len(score['dependencies']) == 1 and Path(score['dependencies'][0]).parts[-2:] == ('predictions', 'complete.json')
    sources = json.loads((run / 'sources.json').read_text(encoding='utf-8'))
    assert set(sources) == {'baseline_pl-s2026092701', 'ptbr-beta05-s2026092701',
                            'tri_sigtime_ssl-s2026092701', candidate_rows()[0]['row_id']}
    assert not any('weak' in json.dumps(x).lower() for x in jobs)


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
