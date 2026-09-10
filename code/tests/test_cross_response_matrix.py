"""Contract tests against generated rows and the actual training parser."""
import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.core90_cross_response_matrix import (
    DEFAULT_CONFIG, VARIANTS, build_matrix, load_config, resolve_variant,
    validate_baseline, validate_input_roles,
)

ROLES = dict(wisig_train_rxs='0,1,2,3', wisig_test_rxs='4,5',
             wisig_train_days='0,1', wisig_test_days='2,3')


def test_matrix_actual_switches_and_independent_outputs(tmp_path):
    rows = build_matrix(wisig_pkl='source.pkl', output_root=tmp_path, roles=ROLES, seeds=[1, 2])
    assert len(rows) == 20
    assert len({row['baseline_args']['output_dir'] for row in rows}) == 20
    by_id = {r['variant']: r['cross_response'] for r in rows}
    assert not by_id['U0']['enabled']
    assert by_id['U1']['enabled'] and not by_id['U1']['response_enabled']
    assert by_id['U2']['response_enabled'] and not by_id['U2']['decision_enabled']
    assert by_id['U3']['decision_enabled'] and not by_id['U3']['response_enabled']
    assert by_id['U4_additive']['predictor_mode'] == 'additive'
    assert by_id['U4_bilinear']['predictor_mode'] == 'bilinear'
    assert by_id['U5']['scheduler_mode'] == 'feedback'
    assert by_id['Ux']['identity_interaction_enabled']
    assert by_id['head_only']['head_only'] and not by_id['head_only']['decision_enabled']
    assert by_id['permanent_detach']['permanent_detach']
    for row in rows:
        validate_baseline(row['baseline_args'])
        assert row['launch'] is False
    paired = ['U1', 'U2', 'U3', 'U4_additive', 'U4_bilinear']
    for key in ('P', 'Q', 'K', 'data_seed', 'role_rotation', 'mixstyle_role_policy'):
        assert len({by_id[v][key] for v in paired}) == 1


@pytest.mark.parametrize('field,value', [('from_scratch', False), ('baseline_ckpt', 'old.pth'),
    ('labeled_ratio', .1), ('best_metric', 'joint_safe'), ('epochs', 201),
    ('checkpoint_selection', 'best'), ('phase2_export_checkpoint', 'old.pth')])
def test_rejects_protocol_drift(field, value):
    base = copy.deepcopy(load_config()['baseline_args'])
    base[field] = value
    with pytest.raises(ValueError):
        validate_baseline(base)


def test_requires_explicit_disjoint_physical_roles():
    for roles in ({}, dict(ROLES, wisig_train_rxs='0,0,1'), dict(ROLES, wisig_test_rxs='0,4')):
        with pytest.raises(ValueError):
            validate_input_roles(roles)


def test_unknown_variant_key_and_confounded_head_only_are_rejected():
    config = load_config()
    config['variants']['U2']['typo'] = True
    with pytest.raises(ValueError, match='unknown variant keys'):
        resolve_variant(config, 'U2')
    config = load_config()
    config['variants']['head_only']['decision_enabled'] = True
    with pytest.raises(ValueError, match='head_only'):
        resolve_variant(config, 'head_only')


def test_every_row_parses_in_actual_training_entrypoint(tmp_path):
    from SSDG.train_ssdg import build_arg_parser
    parser = build_arg_parser()
    rows = build_matrix(wisig_pkl='source.pkl', output_root=tmp_path, roles=ROLES, seeds=[5])
    for row in rows:
        args = parser.parse_args(row['argv'])
        assert args.cross_response_variant == row['variant']
        assert Path(args.cross_response_config) == DEFAULT_CONFIG
        assert args.epochs == 200 and args.from_scratch and not args.baseline_ckpt
        assert args.batch_size == 128
        assert args.model_variant == 'lite_d' and args.id_feature_key == 'feat_joint'
        assert args.lambda_fishr == .04
        assert args.proxy_unknown_core_quantile == .90
        assert args.proxy_unknown_accept_quantile == .85
        assert args.proxy_unknown_vaccept_cvar_alpha == .30
        assert args.use_concat_sat_channel_aug and args.concat_sat_ce_only
        assert args.checkpoint_selection == 'final_only'


@pytest.mark.parametrize('name', ['proxy_unknown_energy_loss', 'source_episode_three_sigma_loss'])
def test_historical_adapter_preserves_loss_gradient_and_single_update(name):
    import torch
    from cvsrffi.cross_response import baseline_compat as compat, legacy_losses as legacy
    torch.manual_seed(21)
    initial = torch.randn(24, 12)
    labels = torch.arange(4).repeat_interleave(6)
    domains = torch.arange(3).repeat(8)
    outcomes = []
    for adapted in (False, True):
        z = initial.clone().requires_grad_()
        torch.manual_seed(50)
        function = getattr(compat if adapted else legacy, name)
        if name == 'proxy_unknown_energy_loss':
            kwargs = dict(virtual_count=12, virtual_mode='hard', virtual_detach=False,
                          vaccept_weight=1., core_accept_weight=.45, component_gate_weight=.65,
                          core_quantile=.9, accept_quantile=.85, vaccept_cvar_alpha=.3)
            if adapted:
                kwargs.update(component_radius_mode='three_sigma', radius_budget_weight=0,
                              radius_budget_rad=.5)
            loss, _ = function(z, labels, **kwargs)
        else:
            kwargs = dict(mixup_weight=.75)
            if adapted:
                kwargs.update(radius_mode='three_sigma', local_component_compact_weight=0,
                              local_component_inter_margin_rad=.4)
            loss, _ = function(z, labels, domains, **kwargs)
        loss.backward()
        optimizer = torch.optim.SGD([z], lr=.01)
        grad = z.grad.clone()
        optimizer.step()
        outcomes.append((loss.detach(), grad, z.detach()))
    for reference, actual in zip(outcomes[0], outcomes[1]):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)


def test_historical_adapter_rejects_activated_later_mechanisms_and_unknowns():
    import torch
    from cvsrffi.cross_response.baseline_compat import proxy_unknown_energy_loss, source_episode_three_sigma_loss
    z, y = torch.randn(8, 4), torch.arange(4).repeat(2)
    with pytest.raises(ValueError, match='later mechanism'):
        proxy_unknown_energy_loss(z, y, bridge_accept_weight=.01)
    with pytest.raises(ValueError, match='three_sigma'):
        source_episode_three_sigma_loss(z, y, y, radius_mode='min_three_sigma_core')
    with pytest.raises(TypeError, match='unsupported'):
        proxy_unknown_energy_loss(z, y, unregistered_weight=0)


def test_historical_prototype_loss_and_resume_state_match():
    import torch
    from cvsrffi.cross_response.baseline_compat import PrototypeMemoryBank
    from cvsrffi.cross_response.legacy_losses import PrototypeMemoryBank as HistoricalBank
    torch.manual_seed(7)
    kwargs = dict(num_classes=3, num_domains=2, momentum=.95, margin=.15,
                  domain_align_weight=.1, push_weight=.1, min_count=2)
    old, new = HistoricalBank(**kwargs), PrototypeMemoryBank(**kwargs)
    y = torch.arange(3).repeat_interleave(4)
    d = torch.tensor([0, 0, 1, 1] * 3)
    for step in range(2):
        initial = torch.randn(12, 8)
        outcomes = []
        for bank in (old, new):
            z = initial.clone().requires_grad_()
            loss, _ = bank.loss(z, y, d)
            (loss + z.sum() * 0).backward()  # Cold historical memory returns an unconnected zero.
            outcomes.append((loss.detach(), z.grad.clone()))
            bank.update(z.detach(), y, d)
        for reference, actual in zip(outcomes[0], outcomes[1]):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    restored = PrototypeMemoryBank(**kwargs)
    restored.load_state_dict(new.state_dict())
    for key, value in restored.state_dict().items():
        torch.testing.assert_close(value, new.state_dict()[key], rtol=0, atol=0)
