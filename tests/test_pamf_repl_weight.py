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


def test_full_training_parser_counts_exposures_once(tmp_path):
    import gzip
    import json
    from experiments.pamf_repl_weight.analyze_training import analyze
    root = tmp_path/'run'; source = root/'F'/'row'/'source'
    root.mkdir(); source.mkdir(parents=True)
    (root/'logs').mkdir()
    (root/'logs'/'train__row.log').write_text('training complete\n', encoding='utf-8')
    (root/'sources.json').write_text(json.dumps({'row': {'source_output': str(source)}}), encoding='utf-8')
    (source/'complete.json').write_text(json.dumps({'status':'SOURCE_FROZEN','target_access':False,'U_truth_read':False}), encoding='utf-8')
    for end, count, epochs in ((30, 2220, 10), (50, 4440, 20)):
        folder = source/f'segment{end}'; folder.mkdir()
        (folder/'repl_state.json').write_text(json.dumps({'U_truth_read':False,'calls':count,'seen':count*10,
            'accepted':count*8,'U_pseudo_observation':{'accepted_by_pseudo_class':{'0':count*8},
            'confidence_bins':{'below90':count*2,'99plus':count*8}}}),encoding='utf-8')
        (folder/'source_v_pseudo_proxy.json').write_text(json.dumps({'role':'V','U_truth_read':False,
            'used_for_training':False,'count':10,'accepted':9,'accepted_correct':8}),encoding='utf-8')
        (folder/'metrics_epoch.jsonl').write_text(''.join(json.dumps({'epoch':i,'val_tx_acc':90})+'\n'
            for i in range(1,epochs+1)),encoding='utf-8')
        (folder/'metrics_epoch.csv').write_text('epoch\n'+''.join(str(i)+'\n' for i in range(1,epochs+1)),encoding='utf-8')
        with gzip.open(folder/'step_metrics.jsonl.gz','wt',encoding='utf-8') as stream:
            for step in range(1,count+1):
                m = {'train/loss_tx_labeled':.5, 'HPL_component_sample':int(step%100==0)}
                for group in ('t2', 'cls_head', 'other'):
                    m.update({f'HPL_GP_{group}_L_norm':2., f'HPL_GP_{group}_U_norm':1.,
                              f'HPL_GP_{group}_dot_before':2. if step % 2 else -2.})
                if step%100==0: m['HPL_component_joint_t2_cosine']=.25
                stream.write(json.dumps({'step':step,'metrics':m})+'\n')
    result = analyze(root)['rows'][0]
    assert result['steps'] == 6660 and result['U_seen'] == 66600
    assert result['U_accepted'] == 53280 and result['U_acceptance'] == pytest.approx(.8)
    assert result['confidence_bins'] == {'below90':13320, '99plus':53280}
    assert result['epochs'][0]['means']['HPL_GP_t2_cosine_before'] == pytest.approx(0.)
    assert len(result['epochs']) == 30 and result['component_samples']['HPL_component_joint_t2_cosine'] == 66


def test_source_v_proxy_handles_versioned_computed_sinc_coordinates(tmp_path, monkeypatch):
    import torch
    from torch import nn
    from experiments.cric_dura_comparison import f_data
    from experiments.pamf_repl_weight.objective import source_validation_proxy

    class TinyV(torch.utils.data.Dataset):
        def __init__(self, root, role, contract):
            assert role == 'V'
        def __len__(self): return 2
        def __getitem__(self, i): return torch.tensor([1.]), i, 0, {'rx_i': 1}
    class CoordinateModel(nn.Module):
        def __init__(self):
            super().__init__(); self.p = nn.Parameter(torch.tensor(1.))
        def forward(self, x):
            coordinate = self.p * 25000. + 10.
            _ = coordinate._version
            return torch.cat((x*coordinate, -x*coordinate), 1)
    monkeypatch.setattr(f_data, 'SourceDataset', TinyV)
    contract = tmp_path/'contract.json'; contract.write_text('{}', encoding='utf-8')
    c = dict(source_cache=str(tmp_path), source_contract=str(contract), output_root=str(tmp_path))
    model = CoordinateModel().train()
    source_validation_proxy(c, model)
    assert model.training and (tmp_path/'source_v_pseudo_proxy.json').is_file()


def test_recover_e30_requires_complete_and_matching_observations(tmp_path):
    import gzip
    import json
    from experiments.pamf_repl_weight.recovery import recover_e30_telemetry
    (tmp_path/'progress.json').write_text(json.dumps({'PL_seen':100, 'PL_accepted':80}),encoding='utf-8')
    metrics = {'HPL_calls':2220, 'HPL_seen':100, 'HPL_accepted':80, 'HPL_student_views':600,
               'HPL_EMA_updates':0, 'HPL_pseudo_class_0':45, 'HPL_pseudo_class_1':35,
               'HPL_conf_bin_below90':10, 'HPL_conf_bin_90to95':10,
               'HPL_conf_bin_95to99':30, 'HPL_conf_bin_99plus':50}
    path=tmp_path/'step_metrics.jsonl.gz'
    with gzip.open(path,'wt',encoding='utf-8') as f:
        for step in range(1,2221): f.write(json.dumps({'step':step,'metrics':metrics})+'\n')
    result=recover_e30_telemetry(tmp_path,2)
    assert result['seen']==100 and result['accepted']==80 and result['recovered_from_complete_step_log']
    (tmp_path/'progress.json').write_text(json.dumps({'PL_seen':100,'PL_accepted':79}),encoding='utf-8')
    with pytest.raises(ValueError,match='counters'):
        recover_e30_telemetry(tmp_path,2)


def test_recovery_remote_preflight_compiles_and_checks_own_e30(monkeypatch):
    from experiments.pamf_branch_training import publish as base
    from experiments.pamf_repl_weight import publish
    seen = {}
    def capture(*args, **kwargs):
        seen['script'] = base.REMOTE
    monkeypatch.setattr(base, 'publish', capture)
    publish.publish('repo','output','context','ssh')
    assert plan.PREVIOUS_RUN in seen['script']
    assert 'for recipe in' in seen['script']
    compile(seen['script'].replace('PACKAGE', '{}'), '<remote preflight>', 'exec')


def test_paired_clean_counts_correct_repairs_and_damage():
    import numpy as np
    from experiments.pamf_repl_weight.analyze_paired import compare
    ids=np.asarray(['a','b','c'])
    truths={'a':{'label':0,'receiver':'r1','day':'d1'},'b':{'label':1,'receiver':'r1','day':'d2'},
            'c':{'label':1,'receiver':'r2','day':'d2'}}
    result=compare(ids,np.asarray([0,0,1]),np.asarray([1,1,1]),truths)
    assert result['overall']['gain']==1 and result['overall']['damage']==1 and result['overall']['net']==0
    assert result['receiver']['r1']['gain']==1 and result['receiver']['r1']['damage']==1
    with pytest.raises(ValueError,match='IDs'):
        compare(np.asarray(['a','a']),np.asarray([0,0]),np.asarray([0,0]),truths)
