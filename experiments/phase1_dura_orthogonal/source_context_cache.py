"""Reusable source-only tensors for the fixed DURA context preparation."""
from pathlib import Path
import os

import torch

SCHEMA = 'phase1_fixed_dura_source_context_v1'
FIELDS = ('received', 'anchor', 'direction', 'variance', 'clean')


def _contract(fit, batch_size, device, deterministic=False):
    if device != 'cuda:0' or batch_size != 32:
        raise ValueError('Cache is registered for FP32 CUDA preparation with batch size 32')
    contract = dict(schema=SCHEMA, fit_ids=[int(x) for x in fit], source_epochs=[21, 22, 23],
                batch_size=batch_size, device=device, precision='FP32_TF32off',
                dura='dura_v1_R65_E20_4940', TX_labels_read=0, F_weights_read=0,
                F_outputs_read=0, V_reads=0, target_reads=0)
    if deterministic:
        contract['deterministic_algorithms'] = True
    return contract


def _validate_data(data, counts, n):
    if len(data) != 5 or not isinstance(counts, dict) or sum(counts.values()) != n:
        raise ValueError('Complete five-field source context and scene counts required')
    for name, tensor in zip(FIELDS, data):
        shape = (n,) if name == 'variance' else (n, 2, 256)
        if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32 or tuple(tensor.shape) != shape:
            raise ValueError('Invalid source context field: '+name)


def save_cache(path, data, counts, fit, batch_size, device, *, deterministic=False):
    path = Path(path)
    contract = _contract(fit, batch_size, device, deterministic)
    _validate_data(data, counts, 3*len(fit))
    if path.exists():
        raise FileExistsError('Source context cache is immutable')
    tmp = path.with_name(path.name+'.partial')
    with tmp.open('xb') as handle:
        torch.save(dict(contract=contract, data=data, counts=counts), handle)
    os.replace(tmp, path)


def load_cache(path, fit, batch_size, device, *, deterministic=False):
    payload = torch.load(Path(path), map_location='cpu', weights_only=True)
    if payload.get('contract') != _contract(fit, batch_size, device, deterministic):
        raise ValueError('Source context cache does not match fixed training input')
    data, counts = payload['data'], payload['counts']
    _validate_data(data, counts, 3*len(fit))
    return data, counts
