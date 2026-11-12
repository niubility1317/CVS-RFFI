"""Deterministic, class-stratified development partition of source L only."""
from contextlib import contextmanager
from pathlib import Path

import numpy as np


def split_labeled_ids(ids, labels, *, seed, fraction=0.2):
    ids, labels = np.asarray(ids), np.asarray(labels)
    if (ids.ndim != 1 or labels.shape != ids.shape or not len(ids)
            or len(set(ids.tolist())) != len(ids) or not 0 < fraction < 1):
        raise ValueError('Unique aligned source-L IDs and labels are required')
    rng = np.random.default_rng(seed)
    selected = np.zeros(len(ids), dtype=bool)
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        if len(indices) < 2:
            raise ValueError('Every source class needs both training and development samples')
        count = max(1, min(len(indices) - 1, int(round(fraction * len(indices)))))
        selected[rng.choice(indices, size=count, replace=False)] = True
    return ids[~selected].tolist(), ids[selected].tolist()


class _LabeledSubset:
    def __init__(self, source, positions):
        self.source = source
        self.positions = list(positions)
        self.x = source.x[self.positions]
        self.metadata = {key: values[self.positions] for key, values in source.metadata.items()}
        self.role = source.role
        self.iq_binding = dict(source.iq_binding, subset='source_labeled_development_partition',
                               count=len(self.positions))

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, i):
        return self.source[self.positions[i]]


def partition_context(ctx, train_ids, dev_ids, *, model_seed):
    """Restrict the actual L training loader; U and V retain their roles."""
    from experiments.cric_dura_comparison.f_data import FixedEpochLoader

    source = ctx['train_loader'].dataset
    ids = source.metadata['ids'].tolist()
    train_ids, dev_ids = list(train_ids), list(dev_ids)
    train_set, dev_set = set(train_ids), set(dev_ids)
    if (source.role != 'L_s' or len(set(ids)) != len(ids)
            or len(train_set) != len(train_ids) or len(dev_set) != len(dev_ids)
            or train_set & dev_set or train_set | dev_set != set(ids) or not dev_set):
        raise ValueError('Exact disjoint source-L training/development IDs required')
    train = _LabeledSubset(source, [i for i, value in enumerate(ids) if value in train_set])
    dev = _LabeledSubset(source, [i for i, value in enumerate(ids) if value in dev_set])
    result = dict(ctx)
    result['train_loader'] = FixedEpochLoader(train, 128, model_seed, drop_last=True)
    result['split_info'] = dict(ctx['split_info'], mode='source_L_train_development',
                                labeled_size=len(train), labeled_development_size=len(dev))
    return result, dev


def verify_parent_partition(current, parent):
    """Reject full-L or differently partitioned checkpoints for development."""
    partition = current.get('source_development')
    if (not partition or partition != parent.get('source_development')
            or current.get('source_contract') != parent.get('source_contract')):
        raise ValueError('CHECKPOINT_DATA_CONTRACT_MISMATCH: source development partition')


@contextmanager
def installed(c):
    """Install the source-only partition for one training segment."""
    from experiments.cric_dura_comparison import f_train
    from experiments.cric_dura_comparison.io import read

    if c.get('parent_source'):
        parent = read(Path(c['parent_source']) / 'complete.json')
        verify_parent_partition(c, parent['config'])
    partition = c['source_development']
    original = f_train.context

    def context(cfg):
        if cfg.get('source_development') != partition:
            raise ValueError('Source development partition changed within training segment')
        result, _ = partition_context(original(cfg), partition['train_ids'], partition['dev_ids'],
                                      model_seed=cfg['model_seed'])
        return result

    f_train.context = context
    try:
        yield
    finally:
        f_train.context = original
