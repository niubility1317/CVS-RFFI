import numpy as np
import pytest
import torch

from experiments.pamf_phase1_flow.resume_state import claim_resume_source, load_state, save_state, validate_resume_source


def _step(model, optimizer, rng):
    x = torch.tensor(rng.normal(size=(4, 3)), dtype=torch.float32)
    x = x + torch.randn_like(x)
    optimizer.zero_grad()
    model(x).square().mean().backward()
    optimizer.step()


def test_epoch_checkpoint_restores_model_optimizer_and_random_streams(tmp_path):
    torch.manual_seed(17)
    rng = np.random.default_rng(23)
    model = torch.nn.Linear(3, 2)
    opt = torch.optim.AdamW(model.parameters(), lr=0.01)
    _step(model, opt, rng)
    path = tmp_path / 'epoch-01.pt'
    contract = {'source': 'phase1', 'teacher': 'frozen', 'commit': 'abc'}
    save_state(path, model, opt, rng, epoch=1, updates=1, contract=contract, device='cpu')
    _step(model, opt, rng)
    expected = {name: value.clone() for name, value in model.state_dict().items()}

    resumed = torch.nn.Linear(3, 2)
    resumed_opt = torch.optim.AdamW(resumed.parameters(), lr=0.01)
    resumed_rng = np.random.default_rng(99)
    epoch, updates = load_state(path, resumed, resumed_opt, resumed_rng, contract=contract, device='cpu')
    assert (epoch, updates) == (1, 1)
    _step(resumed, resumed_opt, resumed_rng)
    for name, value in resumed.state_dict().items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)


def test_checkpoint_rejects_wrong_contract_and_overwrite(tmp_path):
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    rng = np.random.default_rng(1)
    path = tmp_path / 'epoch-01.pt'
    save_state(path, model, optimizer, rng, epoch=1, updates=1, contract={'run': 'a'}, device='cpu')
    with pytest.raises(FileExistsError):
        save_state(path, model, optimizer, rng, epoch=1, updates=1, contract={'run': 'a'}, device='cpu')
    with pytest.raises(ValueError, match='contract'):
        load_state(path, model, optimizer, rng, contract={'run': 'b'}, device='cpu')


def test_resume_requires_failed_stopped_source_without_test_predictions(tmp_path, monkeypatch):
    import json
    from experiments.pamf_phase1_flow import resume_state
    source = tmp_path / 'runs' / 'source'
    checkpoint = source / 'flow/checkpoints/epoch-01.pt'
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b'checkpoint')
    (source / 'complete.json').write_text(json.dumps({'status': 'FAILED_TECHNICAL_PRESERVED'}))
    (source / 'submitted.json').write_text(json.dumps({'pid': 1234}))
    monkeypatch.setattr(resume_state, 'process_exists', lambda pid: True)
    with pytest.raises(ValueError, match='worker'):
        validate_resume_source(tmp_path, checkpoint)
    monkeypatch.setattr(resume_state, 'process_exists', lambda pid: False)
    assert validate_resume_source(tmp_path, checkpoint) == checkpoint.resolve()
    (source / 'flow/test_prediction').mkdir()
    with pytest.raises(ValueError, match='prediction'):
        validate_resume_source(tmp_path, checkpoint)


def test_resume_claim_blocks_duplicate_live_or_tested_descendant(tmp_path, monkeypatch):
    import json
    import sys
    import types
    from experiments.pamf_phase1_flow import resume_state
    monkeypatch.setitem(sys.modules, 'fcntl', types.SimpleNamespace(LOCK_EX=1, LOCK_UN=2, flock=lambda *a: None))
    source = tmp_path / 'runs/source'
    checkpoint = source / 'flow/checkpoints/epoch-01.pt'
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b'checkpoint')
    (source / 'complete.json').write_text(json.dumps({'status': 'FAILED_TECHNICAL_PRESERVED'}))
    (source / 'submitted.json').write_text(json.dumps({'pid': 100}))
    monkeypatch.setattr(resume_state, 'process_exists', lambda pid: pid != 100)
    claim_resume_source(tmp_path, checkpoint, 'next', owner_pid=101)
    with pytest.raises(ValueError, match='owner'):
        claim_resume_source(tmp_path, checkpoint, 'other', owner_pid=102)
    monkeypatch.setattr(resume_state, 'process_exists', lambda pid: False)
    descendant = tmp_path / 'runs/next'
    (descendant / 'flow/test_prediction').mkdir(parents=True)
    with pytest.raises(ValueError, match='prediction'):
        claim_resume_source(tmp_path, checkpoint, 'other', owner_pid=102)


def test_resume_claim_blocks_a_branch_when_a_grandchild_is_active(tmp_path, monkeypatch):
    import json
    import sys
    import types
    from experiments.pamf_phase1_flow import resume_state
    monkeypatch.setitem(sys.modules, 'fcntl', types.SimpleNamespace(LOCK_EX=1, LOCK_UN=2, flock=lambda *a: None))
    root = tmp_path / 'runs/root'
    first = root / 'flow/checkpoints/epoch-01.pt'
    first.parent.mkdir(parents=True)
    first.write_bytes(b'checkpoint')
    (root / 'complete.json').write_text(json.dumps({'status': 'FAILED_TECHNICAL_PRESERVED'}))
    (root / 'submitted.json').write_text(json.dumps({'pid': 100}))
    monkeypatch.setattr(resume_state, 'process_exists', lambda pid: False)
    claim_resume_source(tmp_path, first, 'child', owner_pid=101)
    child = tmp_path / 'runs/child'
    second = child / 'flow/checkpoints/epoch-02.pt'
    second.parent.mkdir(parents=True)
    second.write_bytes(b'checkpoint')
    (child / 'resume_parent.json').write_text(json.dumps({'lineage_root': 'root'}))
    (child / 'complete.json').write_text(json.dumps({'status': 'FAILED_TECHNICAL_PRESERVED'}))
    (child / 'submitted.json').write_text(json.dumps({'pid': 102}))
    claim_resume_source(tmp_path, second, 'grandchild', owner_pid=103)
    monkeypatch.setattr(resume_state, 'process_exists', lambda pid: pid == 103)
    with pytest.raises(ValueError, match='owner'):
        claim_resume_source(tmp_path, first, 'branch', owner_pid=104)


def test_boundary_training_resumes_from_completed_epoch(tmp_path, monkeypatch):
    from experiments.pamf_phase1_flow import boundary_student_experiment as exp

    cache = tmp_path / 'cache'
    clean = np.arange(4, dtype=np.float32).reshape(4, 1)
    for view in (1, 2):
        folder = cache / 'epochs' / f'{view:02d}'
        folder.mkdir(parents=True)
        np.save(folder / 'received.npy', clean + view)
    monkeypatch.setattr(exp.core, 'COUNT', 4)
    monkeypatch.setattr(exp.core, 'BATCH', 2)
    monkeypatch.setattr(exp.core, 'EPOCHS', 2)
    monkeypatch.setattr(exp.core, 'STEPS', 4)
    monkeypatch.setattr(exp.core, 'teacher', lambda root, device: (object(), {}))
    monkeypatch.setattr(exp.core, 'data', lambda root, contract: (cache, clean, np.array(['L_s'] * 4)))
    monkeypatch.setattr(exp.core, 'lr', lambda update: 0.01)
    monkeypatch.setattr(exp.prior, 'snapshot', lambda net: {})
    monkeypatch.setattr(exp.prior, 'unchanged', lambda net, state: True)
    monkeypatch.setattr(exp.prior, 'frozen_dura', lambda root, device: object())
    monkeypatch.setattr(exp.prior, 'sha', lambda path: 'dura-hash')
    monkeypatch.setattr(exp, 'make_student', lambda device: torch.nn.Linear(1, 1).to(device))

    def step(student, teacher, dura, received, target, optimizer):
        optimizer.zero_grad()
        loss = (student(received) + torch.randn_like(received) - target).square().mean()
        loss.backward()
        optimizer.step()
        return {'loss': float(loss.detach()), 'wrong': 0, 'oracle_reached': 0}
    monkeypatch.setattr(exp, 'step', step)
    full, interrupted, resumed = (tmp_path / name for name in ('full', 'interrupted', 'resumed'))
    for folder in (full, interrupted, resumed):
        folder.mkdir()
    torch.manual_seed(31)
    exp.train(tmp_path, full, 'commit', device='cpu')
    expected = torch.load(full / 'flow/final.pt', weights_only=False)['model']
    original_save = exp.save_state

    def interrupt_after_first(*args, **kwargs):
        original_save(*args, **kwargs)
        if kwargs['epoch'] == 1:
            raise RuntimeError('simulated interruption')
    monkeypatch.setattr(exp, 'save_state', interrupt_after_first)
    torch.manual_seed(31)
    with pytest.raises(RuntimeError, match='simulated interruption'):
        exp.train(tmp_path, interrupted, 'commit', device='cpu')
    monkeypatch.setattr(exp, 'save_state', original_save)
    checkpoint = interrupted / 'flow/checkpoints/epoch-01.pt'
    exp.train(tmp_path, resumed, 'commit', device='cpu', resume_checkpoint=checkpoint)
    actual = torch.load(resumed / 'flow/final.pt', weights_only=False)['model']
    for name, value in actual.items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
