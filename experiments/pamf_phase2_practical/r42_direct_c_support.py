"""Direct-C access to target new-class support, without opening query IQ."""
from pathlib import Path
import numpy as np
from .io import read, require


def load_new_support(capsule, split_id):
    root = Path(capsule)
    require(len(split_id) == 24 and all(c in '0123456789abcdef' for c in split_id), 'Invalid split ID')
    manifest = read(root / 'manifest.json')
    split = read(root / 'splits' / (split_id + '.json'))
    require(manifest['storage'] == 'episodic-pool-complement-v1', 'Unsupported storage')
    for value in (manifest, split):
        require(value['protocol_schema'] == 'p2_min_v1' and value['phase2_data_status'] == 'VALIDATED_ONCE', 'Wrong data handles')
    require(split['capsule_id'] == manifest['capsule_id'] and split['split_id'] == split_id, 'Wrong capsule or split')
    k, n = split['k'], split['new_count']
    require(k in (1, 5, 10, 20) and n in (2, 5, 10, 20), 'Wrong K or new class count')
    require(len(split['registered_classes']) == 6 + n, 'Wrong class count')
    support = np.asarray(split['support_indices'], dtype=np.int64)
    require(support.shape == (k * (6 + n),) and len(np.unique(support)) == len(support), 'Wrong support positions')
    require((support >= 0).all() and (support < manifest['received_count']).all(), 'Support positions outside bank')
    # The established physical split reserves old support from query for paired
    # comparison, but its old IQ and labels are never used by this method.
    new_positions = support[6 * k:]
    labels_file = np.load(root / 'support_labels' / (split_id + '.npy'), mmap_mode='r', allow_pickle=False)
    require(labels_file.shape == support.shape, 'Wrong support labels shape')
    labels = np.asarray(labels_file[6 * k:], dtype=np.int64)
    require(np.array_equal(labels, np.repeat(np.arange(6, 6 + n), k)), 'Wrong new support labels')
    bank = np.load(root / 'received.npy', mmap_mode='r', allow_pickle=False)
    ids = np.load(root / 'ids.npy', mmap_mode='r', allow_pickle=False)
    require(bank.dtype == np.float32 and bank.shape == (manifest['received_count'], 2, 256), 'Wrong received bank')
    require(ids.shape == (len(bank),), 'Wrong ID bank')
    new_ids = np.asarray(ids[new_positions])
    require(len(np.unique(new_ids)) == len(new_ids), 'Repeated new support IDs')
    return dict(manifest=manifest, split=split, support_x=bank[new_positions], support_ids=new_ids,
                support_y=labels, actual_target_old_labels=0, actual_target_new_labels=len(labels))
