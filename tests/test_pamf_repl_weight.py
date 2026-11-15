from copy import deepcopy

import pytest

from experiments.pamf_repl_weight import plan


def baseline():
    return dict(hard_pl={'weight': 6.0},
        hard_pl_gradient_guard={'weight': 6.0, 'U_to_L_norm_cap': 1.0},
        hard_pl_joint_contrast={'weight': .05},
        hard_pl_receiver_extrapolation={'ordinary_CE_coefficient': .25, 'extrapolated_CE_coefficient': .25},
        hard_pl_mixstyle={'probability_per_level': .5})


def test_each_recipe_changes_only_the_intended_field():
    source = baseline()
    expected = {
        'weight3': {'hard_pl.weight', 'hard_pl_gradient_guard.weight'},
        'weight15': {'hard_pl.weight', 'hard_pl_gradient_guard.weight'},
        'shallow05': set(), 'joint025': {'hard_pl_joint_contrast.weight'},
        'extrap125': {'hard_pl_receiver_extrapolation.ordinary_CE_coefficient',
                      'hard_pl_receiver_extrapolation.extrapolated_CE_coefficient'},
        'style025': {'hard_pl_mixstyle.probability_per_level'},
        'guard05': {'hard_pl_gradient_guard.U_to_L_norm_cap'},
    }
    for name in plan.RECIPES:
        current = plan.apply_recipe(source, name)
        changed = {f'{k}.{sub}' for k, values in source.items() for sub in values if values[sub] != current[k][sub]}
        assert changed == expected[name]
        assert source == baseline()
        v = current['hard_pl_receiver_extrapolation']
        assert v['ordinary_CE_coefficient'] + v['extrapolated_CE_coefficient'] == .5


def test_matrix_is_fixed_and_clean_only():
    rows = plan.candidate_rows()
    assert len(rows) == 14 and len({x['row_id'] for x in rows}) == 14
    assert {r['setting'] for r in rows} == {'ten_class', 'source_rx_6'}
    assert all(r['model_seed'] == 2026092701 and r['views'] == ['clean'] for r in rows)


@pytest.mark.parametrize('key', ['target_inputs', 'target_truth', 'truth', 'tests', 'views_root', 'checkpoint'])
def test_training_rejects_target_handles(key):
    row = plan.candidate_rows()[0]
    c = dict(run_id=plan.RUN, row_id=row['row_id'], tuning_row=deepcopy(row),
             tuning_recipe=deepcopy(plan.RECIPES[row['recipe']]), model_seed=row['model_seed'],
             protocol_id=row['protocol_id'], output_root=f"/p/runs/{plan.RUN}/F/{row['row_id']}/source", **{key:'forbidden'})
    with pytest.raises(ValueError, match='Target or truth'):
        plan.validate(c)


def test_component_gradient_includes_parameters_unique_to_one_loss(monkeypatch):
    import math
    import torch
    from torch import nn
    from experiments.pamf_ssl_research.hard_pl_guard_objective import Guarded
    from experiments.pamf_repl_weight.objective import ObservedREPL

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.block = nn.Module()
            self.block.t2 = nn.ParameterDict({'a': nn.Parameter(torch.tensor(2.)), 'b': nn.Parameter(torch.tensor(3.))})
    net = Toy()
    obj = ObservedREPL.__new__(ObservedREPL)
    obj.calls = 100
    obj.metrics = {}
    obj.parts = {'joint': net.block.t2['a'] + net.block.t2['b']}
    monkeypatch.setattr(Guarded, 'guide', lambda self, labeled, student: labeled)
    obj.guide(net.block.t2['a'], net)
    assert obj.metrics['HPL_component_sample'] == 1
    assert obj.metrics['HPL_component_joint_t2_L_norm'] == pytest.approx(1.)
    assert obj.metrics['HPL_component_joint_t2_U_norm'] == pytest.approx(math.sqrt(2.))
    assert obj.metrics['HPL_component_joint_t2_cosine'] == pytest.approx(1/math.sqrt(2.))


def test_extrapolation_telemetry_uses_actual_coefficient(monkeypatch):
    import contextlib
    import torch
    from experiments.pamf_paper_completion import model
    from experiments.pamf_ssl_research.hard_pl_receiver_extrapolation_objective import ReceiverExtrapolation
    from experiments.pamf_repl_weight.objective import installed

    @contextlib.contextmanager
    def fake_scope(*args, **kwargs):
        yield
    monkeypatch.setattr(model, 'installed', fake_scope)
    monkeypatch.setattr(ReceiverExtrapolation, 'loss', lambda *args: torch.tensor(3.))
    holder = {}
    cfg = {'tuning_recipe': {'extrap_coefficient': .125}, 'hard_pl_smoke': True,
           'hard_pl': {'weight': 6.}, 'hard_pl_receiver_extrapolation': {'extrapolated_CE_coefficient': .125}}
    dummy = type('D', (), {'metrics': {}, 'parts': {}, '_observed_joint': torch.tensor(2.)})()
    with installed(cfg, holder):
        ReceiverExtrapolation.loss(dummy, None, None)
        assert dummy.metrics['HPL_RXE_reallocated_CE_coefficient'] == .125
