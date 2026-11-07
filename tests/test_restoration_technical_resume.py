import copy
import json

import numpy as np
import pytest
import torch

from experiments.pamf_phase3_restoration.technical_resume import (
    restore_training_state,
    validate_history,
)
from experiments.pamf_phase3_restoration import response_reliability_resume_trial as resume


def _payload(model, optimizer):
    return {
        'config': {'run_id': 'parent', 'source': 'fixed'},
        'code_commit': 'parent_commit',
        'epoch': 7,
        'updates': 1729,
        'state': copy.deepcopy(model.state_dict()),
        'optimizer': copy.deepcopy(optimizer.state_dict()),
        'torch_rng_cpu': torch.get_rng_state().clone(),
        'torch_rng_cuda': None,
        'selection_use': 'technical_recovery_only/not_source_or_target_model_selection',
    }


def test_restored_optimizer_model_and_rng_continue_same_update():
    torch.manual_seed(47)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    for _ in range(1729):
        optimizer.zero_grad()
        model(torch.ones(1, 3)).square().sum().backward()
        optimizer.step()
    saved = _payload(model, optimizer)

    def step(m, opt):
        opt.zero_grad()
        m(torch.randn(1, 3)).square().sum().backward()
        opt.step()
        return copy.deepcopy(m.state_dict())

    expected = step(model, optimizer)
    restored = torch.nn.Linear(3, 2)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.01)
    restore_training_state(saved, restored, restored_optimizer,
                           {'run_id': 'parent', 'source': 'fixed'}, 'parent_commit', 7)
    actual = step(restored, restored_optimizer)
    assert all(torch.equal(actual[k], expected[k]) for k in expected)


def test_rejects_wrong_recovery_identity_before_changing_model():
    torch.manual_seed(3)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    saved = _payload(model, optimizer)
    before = copy.deepcopy(model.state_dict())
    saved['code_commit'] = 'different'
    with pytest.raises(ValueError, match='recovery identity'):
        restore_training_state(saved, model, optimizer,
                               {'run_id': 'parent', 'source': 'fixed'}, 'parent_commit', 7)
    assert all(torch.equal(model.state_dict()[k], before[k]) for k in before)


def test_history_requires_the_complete_original_schedule_and_seven_epochs():
    orders = np.tile(np.arange(63000, dtype=np.int32), (30, 1))
    draws = np.array(list(range(1, 21)) + list(range(1, 11)))
    rows = [dict(epoch=i, draw=i, updates=247 * i, steps=247,
                 count=63000, L_s=6300, U_s=56700) for i in range(1, 8)]
    validate_history(rows, orders, draws, 7, draws)
    broken = copy.deepcopy(rows)
    broken[-1]['updates'] -= 1
    with pytest.raises(ValueError, match='history'):
        validate_history(broken, orders, draws, 7, draws)
    orders[7, 0] = orders[7, 1]
    with pytest.raises(ValueError, match='schedule'):
        validate_history(rows, orders, draws, 7, draws)


def test_only_stopped_parent_without_target_predictions_can_resume(tmp_path):
    parent = tmp_path / 'runs' / resume.original.RUN
    folder = parent / 'rows' / resume.ARM / 'training'
    (folder / 'recovery').mkdir(parents=True)
    (folder / 'recovery' / 'E07.pt').write_bytes(b'checkpoint')
    (folder / 'smoke.json').write_text(json.dumps(dict(status='PASS', TX_labels_read=0,
        V_reads=0, OE_reads=0, target_inputs=0)), encoding='utf-8')
    (parent / 'resource_stop.json').write_text(json.dumps(dict(
        status='STOPPED_OWN_PROCESSES_RESOURCE_CAP', run_id=resume.original.RUN,
        last_complete_epoch=7, own_controller_pid=999999998, own_child_pid=999999999)), encoding='utf-8')
    assert resume.parent_training(tmp_path) == folder
    (parent / ('submitted-' + resume.ARM + '-cross_rx.json')).write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='target predictions'):
        resume.parent_training(tmp_path)


def test_resume_keeps_original_source_recipe_and_budget():
    old = resume.original.configuration(resume.ARM)
    new = resume.configuration(resume.ARM)
    for key in ('source_draws', 'objective', 'source_response_weighting',
                'source_action_deployment_input', 'source_noise_deployment_input'):
        assert new[key] == old[key]
    assert len(new['source_draws']) == 30
    assert new['technical_recovery_epoch'] == 7
