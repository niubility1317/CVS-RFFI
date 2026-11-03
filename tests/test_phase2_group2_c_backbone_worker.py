import json

import numpy as np
import pytest
import torch
from torch import nn

from experiments.pamf_phase2_practical.group2_c_backbone_worker import support_only, score_indexed


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def make_capsule(tmp_path):
    split_id = 'a' * 24
    manifest = {'storage': 'episodic-pool-complement-v1', 'protocol_schema': 'p2_min_v1',
                'phase2_data_status': 'VALIDATED_ONCE', 'capsule_id': 'valid', 'received_count': 20}
    split = dict(protocol_schema='p2_min_v1', phase2_data_status='VALIDATED_ONCE',
                 capsule_id='valid', split_id=split_id, k=1, new_count=0,
                 registered_classes=list(range(6)), support_indices=list(range(6)))
    write_json(tmp_path/'manifest.json', manifest)
    write_json(tmp_path/'splits'/(split_id+'.json'), split)
    (tmp_path/'support_labels').mkdir()
    np.save(tmp_path/'support_labels'/(split_id+'.npy'), np.arange(6, dtype=np.int16))
    np.save(tmp_path/'received.npy', np.arange(20*2*256, dtype=np.float32).reshape(20, 2, 256))
    return split_id


def test_support_loader_returns_only_permitted_support(tmp_path):
    split_id = make_capsule(tmp_path)
    x, y, positions, split = support_only(tmp_path, split_id)
    assert x.shape == (6, 2, 256)
    assert y.tolist() == positions.tolist() == list(range(6))
    assert split['new_count'] == 0
    with pytest.raises(ValueError):
        support_only(tmp_path, '../invalid')


def test_indexed_query_prediction_is_batched_and_does_not_mutate_state():
    class State:
        def __init__(self):
            self.model = nn.Linear(1, 1)
            self.batch_sizes = []
        def score(self, x):
            self.batch_sizes.append(len(x))
            return torch.stack([x[:, 0, 0] + i for i in range(8)], dim=1)

    bank = np.zeros((30, 2, 256), dtype=np.float32)
    state = State()
    pred, old = score_indexed(state, bank, np.arange(17), count=8, old_count=6,
                              batch_size=5, device='cpu')
    assert pred.tolist() == [7]*17 and old.tolist() == [5]*17
    assert state.batch_sizes == [5, 5, 5, 2]


def test_worker_fits_only_support_then_writes_fixed_prediction(tmp_path, monkeypatch):
    from experiments.pamf_phase2_practical import io, group1, group1_source, group2_c_backbone_fit
    from experiments.pamf_phase2_practical.group2_c_backbone_worker import predict

    cap = tmp_path/'capsules'/'P01'/'MIX01'/'1-1'
    cap.mkdir(parents=True)
    old_id = make_capsule(cap)
    new_id = 'b'*24
    write_json(cap/'splits'/(new_id+'.json'),
        dict(protocol_schema='p2_min_v1', phase2_data_status='VALIDATED_ONCE', capsule_id='valid',
             split_id=new_id, k=1, new_count=2, registered_classes=list(range(8)),
             support_indices=list(range(8)), query_count=8))
    np.save(cap/'support_labels'/(new_id+'.npy'), np.arange(8, dtype=np.int16))
    (cap/'stage_rows').mkdir()
    np.save(cap/'stage_rows'/'N02.npy', np.arange(16, dtype=np.int32))
    model = nn.Linear(1, 1)
    monkeypatch.setattr(io, 'activate', lambda rt: None)
    monkeypatch.setattr(group1, 'source_scope', lambda rt: {})
    monkeypatch.setattr(group1_source, 'load_source', lambda *args: dict(model=model))
    stages = []
    def fit(rt, bundle, method, x, y, **kwargs):
        assert len(x) == 6 and y.tolist() == list(range(6))
        stages.append('B_support_only')
        return dict(c_state={}, metadata={'C_inherits_adapted_encoder': True})
    class State:
        def __init__(self):
            self.model = nn.Linear(1, 1)
        def score(self, x):
            assert stages == ['B_support_only', 'C_new_support_only']
            return torch.zeros(len(x), 8)
    def register(rt, c_state, x, y, new_ids, **kwargs):
        assert len(x) == 2 and y.tolist() == [6, 7] and list(new_ids) == [6, 7]
        stages.append('C_new_support_only')
        return State()
    monkeypatch.setattr(group2_c_backbone_fit, 'fit_b_from_c_source', fit)
    monkeypatch.setattr(group2_c_backbone_fit, 'register_native_c', register)
    job = dict(b_method='protonet_cda', c_method='mopc_hr', seed=2026092701,
               capsule='capsules/P01/MIX01/1-1', points=[(1, 2)], output=str(tmp_path/'prediction'))
    rt = dict(device='cpu', c_source_roots={'mopc_hr': 'fixture'}, data_root=str(tmp_path), batch_size=3)
    predict(rt, job)
    index = json.loads((tmp_path/'prediction'/'predictions.json').read_text(encoding='utf-8'))
    assert stages == ['B_support_only', 'C_new_support_only']
    assert index['rows'][0]['query_count'] == 8
    assert json.loads((tmp_path/'prediction'/'complete.json').read_text(encoding='utf-8'))['truth_read'] is False
