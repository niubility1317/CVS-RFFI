from collections import Counter

import torch

from experiments.pamf_physics_tri.hybrid_confidence import HybridConfidence


def test_every_confident_sample_reaches_hybrid_regardless_of_original_physical_half():
    objective = HybridConfidence.__new__(HybridConfidence)
    objective.c = {'practical_gate': {'mode': 'all', 'strength': .25, 'schedule': 'taper', 'all_confident': True}}
    objective.labeled = (torch.zeros(2, 1), torch.tensor([0, 1]), torch.tensor([1, 2]))
    objective.physical_selection = None
    objective.as_calls = objective.as_offered = objective.as_certified = objective.as_active_certified = 0
    objective.as_used = objective.as_physical_used = 0
    objective.strength_weighted_uses = 0.
    objective.accepted_classes = Counter()

    class Student:
        def _cd_practical(self, x, metadata, epoch, step):
            return x + 1

    x = torch.zeros(6, 1)
    accepted = torch.tensor([True, False, True, True, False, True])
    physical = torch.tensor([False, True, False, True, False, True])
    mixed = objective.mixed_view(Student(), x, torch.full_like(x, 2.), physical,
                                 {'day_i': torch.ones(6, dtype=torch.long)}, 21, 1,
                                 torch.tensor([0, 1, 1, 1, 0, 0]), accepted)
    assert torch.equal(objective.physical_selection, accepted)
    assert torch.allclose(mixed[accepted], torch.full((4, 1), .25))
    assert objective.as_offered == objective.as_used == 4
    assert objective.as_physical_used == 2


def test_ten_configurations_have_three_matched_hybrid_baselines():
    from experiments.pamf_physics_tri import hybrid_confidence_plan as plan
    rows = plan.candidate_rows()
    assert len(rows) == 30
    assert len({row['setting'] for row in rows}) == 10
    assert all(plan.baseline_row(row)['arm'] == 'hybrid' and
               plan.baseline_row(row)['model_seed'] == row['model_seed'] for row in rows)


def test_staged_hybrid_training_waits_for_real_checkpoint_smoke(tmp_path, monkeypatch):
    import json
    from experiments.pamf_physics_tri import hybrid_confidence_runner
    root = tmp_path / 'run'
    jobs = [dict(id='train__row', kind='train', config='train.json', completion='source.json',
                 successful_statuses=['SOURCE_FROZEN'], dependencies=[])]
    monkeypatch.setattr(hybrid_confidence_runner, 'prepare', lambda _: jobs)
    hybrid_confidence_runner.stage({'run_root': str(root), 'runtime_file': 'runtime.json', 'release_root': 'release',
                                    'code_commit': 'commit', 'launch_owner': 'owner'})
    staged = json.loads((root / 'jobs.json').read_text())
    state = json.loads((root / 'pipeline_state.json').read_text())
    assert state['status'] == 'STAGED'
    assert staged[0]['id'] == 'smoke'
    assert staged[1]['dependencies'] == [str(root / 'smoke.json')]
