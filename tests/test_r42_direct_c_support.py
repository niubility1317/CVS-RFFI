import json
import numpy as np
import pytest

from experiments.pamf_phase2_practical.r42_direct_c_support import load_new_support


def test_direct_c_reads_only_new_support_and_preserves_registered_identity(tmp_path):
    root = tmp_path
    (root / 'splits').mkdir()
    (root / 'support_labels').mkdir()
    split_id = 'a' * 24
    manifest = {'storage': 'episodic-pool-complement-v1', 'protocol_schema': 'p2_min_v1',
                'phase2_data_status': 'VALIDATED_ONCE', 'capsule_id': 'cap', 'received_count': 10}
    split = {'capsule_id': 'cap', 'split_id': split_id, 'protocol_schema': 'p2_min_v1',
             'phase2_data_status': 'VALIDATED_ONCE', 'k': 1, 'new_count': 2,
             'registered_classes': list(range(8)), 'support_indices': list(range(8)),
             'query_count': 2}
    (root / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    (root / 'splits' / (split_id + '.json')).write_text(json.dumps(split), encoding='utf-8')
    x = np.broadcast_to(np.arange(10, dtype=np.float32)[:, None, None], (10, 2, 256)).copy()
    np.save(root / 'received.npy', x)
    np.save(root / 'ids.npy', np.arange(10))
    # Unusable old labels demonstrate they are not used or checked by this loader.
    np.save(root / 'support_labels' / (split_id + '.npy'), np.array([-99] * 6 + [6, 7]))
    result = load_new_support(root, split_id)
    assert result['support_y'].tolist() == [6, 7]
    assert result['support_ids'].tolist() == [6, 7]
    assert result['support_x'][:, 0, 0].tolist() == [6., 7.]
    assert result['actual_target_old_labels'] == 0
    assert result['actual_target_new_labels'] == 2
    assert 'query_x' not in result


def test_direct_c_rejects_bad_new_labels(tmp_path):
    root = tmp_path
    (root / 'splits').mkdir()
    (root / 'support_labels').mkdir()
    split_id = 'b' * 24
    manifest = {'storage': 'episodic-pool-complement-v1', 'protocol_schema': 'p2_min_v1',
                'phase2_data_status': 'VALIDATED_ONCE', 'capsule_id': 'cap', 'received_count': 8}
    split = {'capsule_id': 'cap', 'split_id': split_id, 'protocol_schema': 'p2_min_v1',
             'phase2_data_status': 'VALIDATED_ONCE', 'k': 1, 'new_count': 2,
             'registered_classes': list(range(8)), 'support_indices': list(range(8))}
    (root / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    (root / 'splits' / (split_id + '.json')).write_text(json.dumps(split), encoding='utf-8')
    np.save(root / 'received.npy', np.zeros((8, 2, 256), dtype=np.float32))
    np.save(root / 'ids.npy', np.arange(8))
    np.save(root / 'support_labels' / (split_id + '.npy'), np.array([0, 1, 2, 3, 4, 5, 6, 6]))
    with pytest.raises(ValueError, match='new support labels'):
        load_new_support(root, split_id)
