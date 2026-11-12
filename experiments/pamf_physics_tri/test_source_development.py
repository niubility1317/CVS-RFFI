import numpy as np
import pytest
import torch

from experiments.pamf_physics_tri.source_development import partition_context, split_labeled_ids, verify_parent_partition, installed
from experiments.cric_dura_comparison.f_data import FixedEpochLoader


def test_split_is_reproducible_disjoint_and_preserves_each_class():
    ids = np.array([f"sample-{i:03d}" for i in range(60)])
    labels = np.repeat(np.arange(3), 20)
    train, dev = split_labeled_ids(ids, labels, seed=20261112)
    train2, dev2 = split_labeled_ids(ids, labels, seed=20261112)
    assert train == train2 and dev == dev2
    assert not set(train) & set(dev)
    assert set(train) | set(dev) == set(ids)
    lookup = dict(zip(ids, labels))
    assert [sum(lookup[x] == y for x in dev) for y in range(3)] == [4, 4, 4]


def test_split_rejects_duplicate_ids_and_classes_without_two_samples():
    with pytest.raises(ValueError):
        split_labeled_ids(np.array(['a', 'a', 'c']), np.array([0, 0, 1]), seed=1)
    with pytest.raises(ValueError):
        split_labeled_ids(np.array(['a', 'b', 'c']), np.array([0, 0, 1]), seed=1)


def test_training_loader_cannot_access_development_records():
    class Labeled(torch.utils.data.Dataset):
        role = 'L_s'
        iq_binding = {'role': 'L_s'}
        metadata = {'ids': np.array([f'id-{i}' for i in range(300)]),
                    'y': np.repeat(np.arange(3), 100)}
        x = torch.arange(300).reshape(300, 1)

        def __len__(self):
            return len(self.x)

        def __getitem__(self, i):
            return self.x[i], self.metadata['y'][i]

    dataset = Labeled()
    train_ids, dev_ids = split_labeled_ids(dataset.metadata['ids'], dataset.metadata['y'], seed=20261112)
    original_u, original_v = object(), object()
    ctx = {'train_loader': FixedEpochLoader(dataset, 128, 7),
           'unlabeled_loader': original_u, 'val_loader': original_v, 'split_info': {}}
    result, dev = partition_context(ctx, train_ids, dev_ids, model_seed=7)
    assert set(result['train_loader'].dataset.metadata['ids']) == set(train_ids)
    assert set(dev.metadata['ids']) == set(dev_ids)
    assert not set(result['train_loader'].dataset.metadata['ids']) & set(dev.metadata['ids'])
    assert result['unlabeled_loader'] is original_u and result['val_loader'] is original_v
    with pytest.raises(ValueError):
        partition_context(ctx, train_ids + dev_ids[:1], dev_ids, model_seed=7)


def test_parent_requires_exact_same_development_partition():
    current = {'source_contract': 'source.json', 'source_development': {'train_ids': ['a'], 'dev_ids': ['b']}}
    verify_parent_partition(current, dict(current))
    with pytest.raises(ValueError):
        verify_parent_partition(current, {'source_contract': 'source.json'})
    with pytest.raises(ValueError):
        verify_parent_partition(current, {'source_contract': 'source.json',
                                           'source_development': {'train_ids': ['b'], 'dev_ids': ['a']}})


def test_installed_scope_enforces_matching_parent_before_loading(tmp_path):
    import json
    from experiments.cric_dura_comparison import f_train
    parent = tmp_path / 'parent'
    parent.mkdir()
    (parent / 'complete.json').write_text(json.dumps({'config': {'source_contract': 's.json'}}))
    c = {'source_contract': 's.json', 'source_development': {'train_ids': ['a'], 'dev_ids': ['b']},
         'parent_source': str(parent)}
    original = f_train.context
    with pytest.raises(ValueError):
        with installed(c):
            pass
    assert f_train.context is original
