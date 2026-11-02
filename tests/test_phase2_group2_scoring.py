import numpy as np
import torch
from torch import nn
import pytest

from experiments.pamf_phase2_practical.group2_worker import score_fixed


class Fixed:
    def __init__(self, mutate=False):
        self.model = nn.Linear(2, 4, bias=False).eval()
        self.mutate = mutate
    def score(self, x):
        if self.mutate:
            self.model.weight.add_(1)
        return self.model(x)


def test_batch_scoring_is_exact_and_restricted_output_is_old_only():
    state = Fixed()
    query = np.arange(18, dtype=np.float32).reshape(9, 2)
    pred, old = score_fixed(state, query, count=4, old_count=2, batch_size=3, device='cpu')
    with torch.no_grad():
        scores = state.model(torch.tensor(query))
    assert pred.tolist() == scores.argmax(1).tolist()
    assert old.tolist() == scores[:, :2].argmax(1).tolist()


def test_query_cannot_update_model_state():
    with pytest.raises(ValueError, match='updated'):
        score_fixed(Fixed(mutate=True), np.ones((2, 2), np.float32), count=4,
                    old_count=2, batch_size=2, device='cpu')


def test_missing_full_cross_product_never_marks_predictions_complete():
    from experiments.pamf_phase2_practical.group2_worker import require_all_rows
    row = {'k': 5, 'new_count': 2, 'arm': 'dadda_sda__lora'}
    require_all_rows([row], [(5, 2)], 'dadda_sda', ['lora'])
    with pytest.raises(ValueError, match='Incomplete'):
        require_all_rows([row], [(5, 2)], 'dadda_sda', ['lora', 'mopc_hr'])
    with pytest.raises(ValueError, match='Incomplete'):
        require_all_rows([row, row], [(5, 2)], 'dadda_sda', ['lora'])


def test_worker_fits_b_once_then_emits_every_requested_c_row(tmp_path, monkeypatch):
    import json
    from experiments.pamf_phase2_practical import io, group1, group1_source, group1_target, group2_cross_fit, episodic_io, episodic_experiment
    from experiments.pamf_phase2_practical.group2_worker import predict

    cap = tmp_path / 'capsules' / 'X'
    (cap / 'splits').mkdir(parents=True)
    (cap / 'manifest.json').write_text(json.dumps({'capsule_id':'c', 'protocol_schema':'p2_min_v1',
                                                  'phase2_data_status':'VALIDATED_ONCE'}))
    for n in (0, 2):
        (cap / 'splits' / f'{n}.json').write_text(json.dumps({'k':1, 'new_count':n, 'split_id':str(n),
            'registered_classes':list(range(6+n))}))
    def load(cap, sid, *, stage):
        n = int(sid)
        return {'support_x':np.ones((6+n, 2), np.float32), 'support_y':np.arange(6+n),
                'support_ids':np.arange(6+n), 'query_x':np.ones((3, 2), np.float32)}
    model = nn.Linear(2, 8).eval()
    calls = []
    monkeypatch.setattr(io, 'activate', lambda rt: None)
    monkeypatch.setattr(group1, 'row', lambda p: {'category':'DA', 'method':'dadda'})
    monkeypatch.setattr(group1, 'source_scope', lambda rt: {})
    monkeypatch.setattr(group1_source, 'load_source', lambda *args: {'model':model})
    monkeypatch.setattr(group1_target, 'fit_B', lambda rt, bundle, x, y, seed, logger: calls.append(('B', len(x))) or
                        {'model':model, 'score_fn':lambda m,v:m(v)[:,:6]})
    monkeypatch.setattr(episodic_io, 'load_task', load)
    monkeypatch.setattr(episodic_experiment, 'split_rows', lambda cap, spec: np.arange(3))
    class AnchorModel(nn.Module):
        def forward(self, value):
            return {'tx_logits':model(value)[:,:6]}
    anchor = type('Anchor', (), {'model':AnchorModel()})()
    monkeypatch.setattr(group2_cross_fit, 'prepare_b', lambda m, p, f, x, y, **kw: {'lora':None, 'mopc_hr':anchor})
    def register(prepared, x, y, *, seed, methods):
        calls.append(('C', len(x)))
        state = Fixed()
        state.model = nn.Linear(2, 8).eval()
        return {methods[0]:state}
    monkeypatch.setattr(group2_cross_fit, 'register_c', register)
    out = tmp_path / 'output'
    predict({'data_root':str(tmp_path), 'source_run_root':'source', 'device':'cpu'},
            {'profile':'dadda_sda', 'seed':7, 'capsule':'capsules/X', 'output':str(out),
             'points':[(1,2)], 'methods':['lora','mopc_hr']})
    assert calls == [('B', 6), ('C', 2), ('C', 2)]
    assert json.loads((out/'complete.json').read_text())['truth_read'] is False
    assert len(json.loads((out/'predictions.json').read_text())['rows']) == 2
