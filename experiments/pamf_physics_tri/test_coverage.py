import torch

from experiments.pamf_physics_tri.coverage_objective import RollingSourceMemory, certificate_tiers
from experiments.pamf_physics_tri.recovery import branch_scores


def test_tiers_require_both_views_and_available_cross_day_reference():
    pseudo = torch.zeros(4, dtype=torch.long)
    raw = torch.zeros(4, 5, 2)
    changed = torch.zeros_like(raw)
    raw[:, :, 0] = 1
    changed[:, :, 0] = 1
    raw[1, :2, 1] = 2
    changed[1, :2, 1] = 2
    changed[2, :3, 1] = 2
    have = torch.tensor([True, True, True, False])
    strong, weak, _, _ = certificate_tiers(raw, changed, have, pseudo)
    assert strong.tolist() == [True, False, False, False]
    assert weak.tolist() == [False, True, False, False]


def test_memory_capacity_and_cross_day_exclusion():
    memory = RollingSourceMemory(capacity=2)
    memory.add(torch.arange(5.).reshape(5, 1), torch.tensor([0, 0, 0, 1, 1]),
               torch.tensor([1, 1, 1, 1, 2]))
    assert [x.item() for x in memory.groups[(0, 1)]] == [1., 2.]
    bank = torch.tensor([[1., 0.], [0., 1.], [1., 0.], [0., 1.]])
    query = torch.tensor([[1., 0.]])
    branches = {name: torch.cat((bank, query)) for name in ('time', 'frequency', 'pa', 'path', 'reference')}
    labels = torch.tensor([0, 1, 0, 1])
    days = torch.tensor([1, 1, 2, 2])
    raw, _, have = branch_scores(branches, branches, labels, days, torch.tensor([1]), 2)
    assert have.tolist() == [True]
    assert raw.argmax(2).eq(0).all()
    _, _, no_reference = branch_scores(branches, branches, labels, torch.ones(4, dtype=torch.long),
                                       torch.tensor([1]), 2)
    assert no_reference.tolist() == [False]


def test_fixed_factorial_uses_three_seeds_and_reuses_baseline():
    from experiments.pamf_physics_tri import coverage_plan as plan
    rows = plan.candidate_rows()
    assert len(rows) == 9
    assert {(row['reference_memory'], row['tiered']) for row in rows} == {(True, False), (False, True), (True, True)}
    assert {row['model_seed'] for row in rows} == set(plan.SEEDS)
    assert all(plan.baseline_row(row)['arm'] == 'branch_prototype' for row in rows)


def test_gate3_pairing_does_not_wait_for_unfinished_control(tmp_path, monkeypatch):
    import json
    from experiments.pamf_physics_tri import coverage_runner, target_tuning_report

    root = tmp_path / 'coverage'
    (root / 'analysis').mkdir(parents=True)
    old = tmp_path / 'runs' / '20261112-phase1-pamf-prototype-fullsource-tuning-m33-r01'
    old.mkdir(parents=True)
    seeds = (2026092701, 2026092702, 2026092703)
    sources = {}
    rows = []
    for seed in seeds:
        folder = old / str(seed)
        folder.mkdir()
        if seed != seeds[-1]:
            (folder / 'complete.json').write_text('{}', encoding='utf-8')
        sources[f'ten_class-prototype-gate3-s{seed}'] = {'scores': str(folder)}
        rows.append({'row_id': f'candidate-s{seed}', 'model_seed': seed,
                     'current': {'clean': .8, 'practical_mean': .6, 'counts': {'clean': 10}}})
    (old / 'sources.json').write_text(json.dumps(sources), encoding='utf-8')
    (root / 'analysis' / 'results.json').write_text(json.dumps({'rows': rows}), encoding='utf-8')
    monkeypatch.setattr(target_tuning_report, 'analyze', lambda _: None)
    monkeypatch.setattr(target_tuning_report, 'load_score', lambda _: {
        'clean': .7, 'practical_mean': .5, 'counts': {'clean': 10}})
    coverage_runner.analyze({'run_root': str(root), 'project': str(tmp_path)})
    result = json.loads((root / 'analysis' / 'gate3_comparison.json').read_text(encoding='utf-8'))
    assert len(result['pairs']) == 2
    assert result['pending'] == ['ten_class-prototype-gate3-s2026092703']
    assert abs(result['pairs'][0]['clean_delta'] - .1) < 1e-9
