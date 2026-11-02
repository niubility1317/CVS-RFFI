import torch

from experiments.pamf_phase2_practical.group2_cross_fit import prepare_b, register_c
from test_phase2_group2_cil_bridge import DaddaLike


def feature(model, x):
    return model(x)['local_features']


def test_one_b_state_is_reused_for_multiple_independent_c_registrations(monkeypatch):
    from experiments.pamf_phase2_practical import group2_cil_bridge
    real_register = group2_cil_bridge.register
    monkeypatch.setattr(group2_cil_bridge, 'register',
                        lambda anchor, x, y, *, seed: real_register(anchor, x, y, seed=seed, epochs=1))
    calls = []
    real_anchor = group2_cil_bridge.make_anchor
    def count_anchor(*args, **kwargs):
        calls.append(kwargs['method'])
        return real_anchor(*args, **kwargs)
    monkeypatch.setattr(group2_cil_bridge, 'make_anchor', count_anchor)
    model = DaddaLike().eval()
    old_x = torch.tensor([[0., 0.], [1., 1.]])
    old_y = torch.tensor([0, 1])
    prepared = prepare_b(model, 'dadda_sda', feature, old_x, old_y, old_count=2)
    assert calls == ['mopc_hr']
    states = register_c(prepared, torch.tensor([[2., 2.], [3., 3.]]), torch.tensor([2, 3]),
                        seed=1, methods=('lora', 'mopc_hr', 'orthogonal'))
    assert set(states) == {'lora', 'mopc_hr', 'orthogonal'}
    assert all(state.score(old_x).shape == (2, 4) for state in states.values())
    assert prepared['mopc_hr'].model.fc.out_features == 2
    again = register_c(prepared, torch.tensor([[4., 4.], [5., 5.]]), torch.tensor([2, 3]),
                       seed=2, methods=('lora',))
    assert again['lora'].features.data_ptr() != prepared['lora'].features.data_ptr()
    assert torch.equal(again['lora'].features[:2], prepared['lora'].features)
