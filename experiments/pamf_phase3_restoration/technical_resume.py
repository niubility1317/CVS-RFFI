"""Validate and restore an interrupted source-only training boundary."""

import numpy as np
import torch


def restore_training_state(payload, model, optimizer, expected_config, expected_commit, epoch):
    required = {'config', 'code_commit', 'epoch', 'updates', 'state', 'optimizer',
                'torch_rng_cpu', 'torch_rng_cuda', 'selection_use'}
    if (set(payload) != required or payload['config'] != expected_config
            or payload['code_commit'] != expected_commit or payload['epoch'] != epoch
            or payload['updates'] != 247 * epoch
            or payload['selection_use'] != 'technical_recovery_only/not_source_or_target_model_selection'):
        raise ValueError('Invalid technical recovery identity')
    state = payload['state']
    opt = payload['optimizer']
    if (set(state) != set(model.state_dict())
            or any(not torch.isfinite(value).all() for value in state.values())
            or not opt['state']
            or {float(value['step']) for value in opt['state'].values() if 'step' in value} != {float(247 * epoch)}
            or any(not torch.isfinite(value).all() for group in opt['state'].values()
                   for value in group.values() if torch.is_tensor(value))):
        raise ValueError('Invalid technical recovery model or optimizer state')
    if not torch.is_tensor(payload['torch_rng_cpu']):
        raise ValueError('Missing technical recovery RNG state')
    uses_cuda = next(model.parameters()).is_cuda
    if uses_cuda and not torch.is_tensor(payload['torch_rng_cuda']):
        raise ValueError('Missing technical recovery CUDA RNG state')
    model.load_state_dict(state, strict=True)
    optimizer.load_state_dict(opt)
    torch.set_rng_state(payload['torch_rng_cpu'])
    if uses_cuda:
        torch.cuda.set_rng_state(payload['torch_rng_cuda'])


def validate_history(rows, orders, draws, epoch, expected_draws):
    if (not 1 <= epoch < 30 or len(rows) != epoch or orders.shape != (30, 63000)
            or draws.shape != (30,) or not np.array_equal(draws, np.asarray(expected_draws))):
        raise ValueError('Invalid technical recovery history or schedule')
    for index, row in enumerate(rows, 1):
        if (row['epoch'], row['draw'], row['updates'], row['steps'], row['count'],
                row['L_s'], row['U_s']) != (index, int(draws[index - 1]), 247 * index,
                                            247, 63000, 6300, 56700):
            raise ValueError('Invalid technical recovery history')
    reference = np.arange(63000)
    if any(not np.array_equal(np.sort(order), reference) for order in orders):
        raise ValueError('Invalid technical recovery schedule')
