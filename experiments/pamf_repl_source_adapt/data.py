"""Source-only fold binding. The held-out validation loader is separate."""
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from .folds import ROLES, fold_groups, fold_positions


def registered_fold(config):
    from experiments.cric_dura_comparison.io import read

    contract = read(config['source_contract'])
    groups = fold_groups(contract['source_rxs'], contract['source_days'])
    name = config['source_fold']
    if name not in groups:
        raise ValueError('Unknown source fold for this data contract')
    metadata = {}
    for role in ROLES:
        with np.load(Path(config['source_cache']) / f'{role}_metadata.npz', allow_pickle=False) as values:
            if role == 'U_s' and 'y' in values.files:
                raise ValueError('Only TX-blind source U is allowed')
            if values['ids'].tolist() != contract['role_ids'][role]:
                raise ValueError(f'{role} metadata does not match the physical source contract')
            metadata[role] = {field: values[field] for field in ('rx_i', 'day_i')}
            if role == 'L_s':
                metadata[role]['y'] = values['y']
    partition = fold_positions(metadata, groups[name])
    if partition is None:
        raise ValueError(f'Insufficient source training or validation coverage for {name}')
    return contract, partition


@contextmanager
def installed_training_fold(config, partition):
    from experiments.cric_dura_comparison import f_data, f_train

    original, alias = f_data.context, f_train.context

    def context(current):
        if any(current[key] != config[key] for key in ('source_fold', 'source_cache', 'source_contract')):
            raise ValueError('Unexpected source contract or fold in the training worker')
        keep = partition['train']
        value = original(current, source_positions=keep)
        source = value['source_contract']
        value['source_contract'] = dict(source, source_fold=current['source_fold'],
            counts={role: len(positions) for role, positions in keep.items()},
            role_ids={role: [source['role_ids'][role][i] for i in positions] for role, positions in keep.items()})
        return value

    f_data.context = f_train.context = context
    try:
        yield
    finally:
        f_data.context, f_train.context = original, alias
