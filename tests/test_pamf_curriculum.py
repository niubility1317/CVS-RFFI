from pathlib import Path

from experiments.pamf_physics_tri import curriculum_plan
from experiments.pamf_physics_route import paired_report


def test_curriculum_prespecifies_two_arms_and_three_seeds():
    rows = curriculum_plan.candidate_rows()
    assert len(rows) == 6
    assert {r['arm'] for r in rows} == {'day_physics', 'branch_certificate'}
    assert {r['protocol_id'] for r in rows} == {'m1-natural'}
    assert {r['model_seed'] for r in rows} == {2026092701, 2026092702, 2026092703}


def test_paired_report_includes_same_seed_branch_control(monkeypatch, tmp_path):
    row, seed, pid = 'candidate', 2026092701, 'm1-natural'
    sources = {row: 'candidate-entry', f'baseline_pl-{pid}-s{seed}': 'pl-entry',
               f'tri_sigtime_ssl-{pid}-s{seed}': 'tri-entry', f'branch-certified-{pid}-s{seed}': 'branch-entry'}
    cfg = {'model_seed': seed, 'protocol_id': pid, 'views': list(paired_report.VIEWS)}
    monkeypatch.setattr(paired_report, 'read', lambda path: sources if str(path).endswith('sources.json')
                        else cfg if 'score__' in str(path) else {'inputs': True})
    monkeypatch.setattr(paired_report, 'frozen_row', lambda entry, seed, runs, inputs: entry)
    _, entries = paired_report.prepare_row(Path(tmp_path), row)
    assert entries['Branch-certified'] == 'branch-entry'
