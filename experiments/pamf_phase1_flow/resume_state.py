"""Immutable epoch-boundary state for source-only restoration training."""
import json
import os
from pathlib import Path
import re

import torch


def process_exists(pid):
    return (Path('/proc') / str(pid)).exists()


def validate_resume_source(root, checkpoint):
    root = Path(root).resolve()
    checkpoint = Path(checkpoint).resolve()
    if (not checkpoint.is_file() or not checkpoint.is_relative_to(root / 'runs')
            or checkpoint.parent.name != 'checkpoints' or checkpoint.parent.parent.name != 'flow'
            or not re.fullmatch(r'epoch-\d{2}\.pt', checkpoint.name)):
        raise ValueError('Existing registered source epoch checkpoint required')
    source = checkpoint.parent.parent.parent
    complete = source / 'complete.json'
    submitted = source / 'submitted.json'
    if not complete.is_file() or not submitted.is_file():
        raise ValueError('Terminal source status and worker ownership required')
    if json.loads(complete.read_text())['status'] != 'FAILED_TECHNICAL_PRESERVED':
        raise ValueError('Only a failed source run may be resumed')
    pid = json.loads(submitted.read_text())['pid']
    if not isinstance(pid, int) or pid <= 0 or process_exists(pid):
        raise ValueError('Source worker must be confirmed stopped')
    if (source / 'flow/test_prediction').exists():
        raise ValueError('A source with test prediction activity cannot be resumed')
    return checkpoint


def claim_resume_source(root, checkpoint, destination, *, owner_pid=None):
    import fcntl
    root = Path(root).resolve()
    if not destination or Path(destination).name != destination or destination in ('.', '..'):
        raise ValueError('Valid unique destination run required')
    checkpoint = validate_resume_source(root, checkpoint)
    source = checkpoint.parent.parent.parent
    if root / 'runs' / destination == source:
        raise ValueError('Resume destination must be a new run')
    parent = source / 'resume_parent.json'
    lineage = source
    if parent.exists():
        name = json.loads(parent.read_text(encoding='utf8'))['lineage_root']
        if not isinstance(name, str) or Path(name).name != name or name in ('.', '..'):
            raise ValueError('Invalid resume lineage root')
        lineage = root / 'runs' / name
        prior_claim = lineage / 'flow/resume_claims' / (source.name + '.json')
        if not prior_claim.is_file() or json.loads(prior_claim.read_text(encoding='utf8'))['destination'] != source.name:
            raise ValueError('Resume lineage claim is missing')
    claims = lineage / 'flow/resume_claims'
    claims.mkdir(exist_ok=True)
    with (lineage / 'flow/resume_claim.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        validate_resume_source(root, checkpoint)
        for item in claims.glob('*.json'):
            claim = json.loads(item.read_text(encoding='utf8'))
            target_name = claim['destination']
            if not isinstance(target_name,str) or Path(target_name).name != target_name or target_name in ('.','..'):
                raise ValueError('Invalid prior resume owner')
            target = root / 'runs' / target_name
            if (target / 'flow/test_prediction').exists():
                raise ValueError('Prior resume owner started test prediction')
            if process_exists(claim['owner_pid']):
                raise ValueError('Prior resume owner is still active')
            if target.exists():
                complete = target / 'complete.json'
                if not complete.is_file() or json.loads(complete.read_text())['status'] != 'FAILED_TECHNICAL_PRESERVED':
                    raise ValueError('Prior resume owner status is not a confirmed failure')
                submitted = target / 'submitted.json'
                if submitted.exists() and process_exists(json.loads(submitted.read_text())['pid']):
                    raise ValueError('Prior resume worker is still active')
        path = claims / (destination + '.json')
        with path.open('x', encoding='utf8') as f:
            json.dump(dict(destination=destination,owner_pid=owner_pid or os.getpid(),checkpoint=str(checkpoint)),f)
            f.flush()
            os.fsync(f.fileno())
        fcntl.flock(lock, fcntl.LOCK_UN)
    return checkpoint, lineage.name


def save_state(path, model, optimizer, rng, *, epoch, updates, contract, device):
    path = Path(path)
    if epoch < 1 or updates < 1 or not contract:
        raise ValueError('Positive progress and an exact training contract are required')
    if path.exists():
        raise FileExistsError(path)
    device = torch.device(device)
    state = dict(schema='phase1_source_epoch_v1', epoch=epoch, updates=updates,
                 contract=contract, model=model.state_dict(), optimizer=optimizer.state_dict(),
                 numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                 cuda_rng=torch.cuda.get_rng_state(device) if device.type == 'cuda' else None,
                 device_type=device.type)
    tmp = path.with_name(path.name + '.partial')
    created = False
    try:
        with tmp.open('xb') as f:
            created = True
            torch.save(state, f)
            f.flush()
            os.fsync(f.fileno())
        os.link(tmp, path)
    finally:
        if created:
            tmp.unlink()


def load_state(path, model, optimizer, rng, *, contract, device):
    state = torch.load(path, map_location='cpu', weights_only=False)
    device = torch.device(device)
    if (state.get('schema') != 'phase1_source_epoch_v1' or state.get('contract') != contract
            or state.get('device_type') != device.type):
        raise ValueError('Checkpoint training contract or device type does not match')
    epoch, updates = state.get('epoch'), state.get('updates')
    if not isinstance(epoch, int) or not isinstance(updates, int) or epoch < 1 or updates < 1:
        raise ValueError('Checkpoint progress must be positive')
    model.load_state_dict(state['model'], strict=True)
    optimizer.load_state_dict(state['optimizer'])
    rng.bit_generator.state = state['numpy_rng']
    torch.set_rng_state(state['torch_rng'])
    if device.type == 'cuda':
        torch.cuda.set_rng_state(state['cuda_rng'], device)
    return epoch, updates
