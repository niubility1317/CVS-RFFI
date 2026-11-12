from collections import Counter

import torch

from experiments.pamf_physics_tri import confidence_only


def test_all_confident_samples_are_routed_even_outside_original_physical_half(monkeypatch):
    objective = confidence_only.ConfidenceOnlyPrototype.__new__(confidence_only.ConfidenceOnlyPrototype)
    objective.c = {'practical_gate': {'mode': 'all', 'strength': .25, 'schedule': 'taper', 'all_confident': True}}
    objective.labeled = (torch.zeros(2, 1), torch.tensor([0, 1]), torch.tensor([1, 2]))
    objective.physical_selection = None
    objective.as_calls = objective.as_offered = objective.as_certified = objective.as_active_certified = 0
    objective.as_used = objective.as_physical_used = 0
    objective.strength_weighted_uses = 0.
    objective.accepted_classes = Counter()
    monkeypatch.setattr(confidence_only, 'capture_source_bank', lambda *_: (torch.zeros(2, 1), {'time': torch.zeros(2, 1)}))

    class Student:
        def _cd_practical(self, x, metadata, epoch, step):
            return x + 1

    x = torch.zeros(6, 1)
    ordinary = torch.full_like(x, 2.)
    physical = torch.tensor([False, True, False, True, False, True])
    accepted = torch.tensor([True, False, True, True, False, True])
    pseudo = torch.tensor([0, 1, 1, 1, 0, 0])
    mixed = objective.mixed_view(Student(), x, ordinary, physical, {'day_i': torch.ones(6, dtype=torch.long)},
                                 epoch=21, step=1, pseudo=pseudo, accepted=accepted)
    assert torch.equal(objective.physical_selection, accepted)
    assert torch.allclose(mixed[accepted], torch.full((4, 1), .25))
    assert mixed[1].item() == 0. and mixed[4].item() == 2.
    assert objective.as_offered == objective.as_used == 4
    assert objective.as_physical_used == 2


def test_nine_rows_use_matching_original_baselines():
    from experiments.pamf_physics_tri import confidence_plan as plan
    rows = plan.candidate_rows()
    assert len(rows) == 9
    assert {row['setting'] for row in rows} == {'ten_class', 'labeled_1pct', 'source_rx_6'}
    assert {row['model_seed'] for row in rows} == set(plan.SEEDS)
    assert all(plan.baseline_row(row)['model_seed'] == row['model_seed'] for row in rows)


def test_every_selected_packet_gets_prototype_loss_when_cross_day_classes_are_missing():
    query = torch.tensor([[0., 1.], [1., 0.]], requires_grad=True)
    raw = torch.tensor([[1., 0.], [0., 1.]])
    bank = torch.tensor([[1., 0.], [0., 1.]])
    loss, missing_positive, no_negative = confidence_only.complete_prototype_loss(
        query, raw, bank, torch.tensor([0, 1]), torch.tensor([1, 2]), torch.tensor([1, 2]),
        torch.tensor([0, 0]), torch.tensor([True, True]), 3, .2)
    loss.backward()
    assert missing_positive == 1 and no_negative == 1
    assert torch.isfinite(loss)
    assert query.grad[0].abs().sum() > 0
    assert query.grad[1].abs().sum() > 0


def test_staged_training_waits_for_real_checkpoint_smoke(tmp_path, monkeypatch):
    import json
    from experiments.pamf_physics_tri import confidence_runner
    root = tmp_path / 'run'
    jobs = [dict(id='train__row', kind='train', config='train.json', completion='source.json',
                 successful_statuses=['SOURCE_FROZEN'], dependencies=[])]
    monkeypatch.setattr(confidence_runner, 'prepare', lambda _: jobs)
    confidence_runner.stage({'run_root': str(root), 'runtime_file': 'runtime.json', 'release_root': 'release',
                             'code_commit': 'commit', 'launch_owner': 'owner'})
    staged = json.loads((root / 'jobs.json').read_text())
    state = json.loads((root / 'pipeline_state.json').read_text())
    assert state['status'] == 'STAGED'
    assert staged[0]['id'] == 'smoke'
    assert staged[1]['dependencies'] == [str(root / 'smoke.json')]
