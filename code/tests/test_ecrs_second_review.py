from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from cvsrffi.ecrs_config import validate_ecrs_resume_args, validate_ecrs_revision_args
from cvsrffi.ecrs_runtime import ECRSRuntime
from test_ecrs_runtime import TinyModel
from test_ecrs_revision_evaluation import IdentityOnly, batch
from cvsrffi.ecrs_evaluation import evaluate_revision_paths


@pytest.mark.parametrize('empty_kind', ['disabled', 'zero_weight', 'no_pairs'])
def test_empty_objectives_do_not_decay_momentum_update_or_tick(empty_kind):
    model = TinyModel()
    opt = torch.optim.AdamW([{'params': model.ecrs.encoder.parameters()},
                             {'params': model.raw_head.parameters()}], lr=.01, weight_decay=.1)
    runtime = ECRSRuntime(model, optimizer=opt, group_configs={0: {'total_steps': 20}})
    # Create nonzero optimizer momentum first, then exercise an empty objective.
    model.forward_response(torch.randn(4, 3))['z_resp'].square().sum().backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    before = deepcopy(model.ecrs.encoder.state_dict())
    state_before = deepcopy(opt.state_dict())
    runtime.set_epoch(1, {'enabled': True, 'resp_cls': False,
                         'cross_rx': empty_kind != 'disabled', 'u_pair': False})
    if empty_kind == 'zero_weight':
        runtime.weights['cross_rx'] = 0.
    x = torch.randn(4, 3)
    output = model.forward_response(x)
    # BN training forward is separate from optimizer behavior checked here.
    before = deepcopy(model.ecrs.encoder.state_dict())
    info = runtime.revision_losses(model, output, x, torch.zeros(4, dtype=torch.long),
        {'receiver_id': torch.zeros(4, dtype=torch.long), 'day_id': torch.zeros(4, dtype=torch.long)})
    assert not info['total'].requires_grad
    (model.raw_head(x).square().mean() + info['total']).backward()
    assert all(p.grad is None for p in model.ecrs.encoder.parameters())
    opt.step()
    step = runtime.after_optimizer_step(model, True)
    assert step == {'ema_updated': False, 'advanced_groups': []}
    for key, value in before.items():
        torch.testing.assert_close(value, model.ecrs.encoder.state_dict()[key], rtol=0, atol=0)
    for index in state_before['param_groups'][0]['params']:
        for key, value in state_before['state'][index].items():
            torch.testing.assert_close(value, opt.state_dict()['state'][index][key], rtol=0, atol=0)


@pytest.mark.parametrize('key,value', [('use_aug', False), ('aug_p_pa', .8),
    ('lambda_adv', 0.), ('label_smoothing', .2), ('concat_sat_seed', 999), ('stage1_epochs', 20)])
def test_resume_rejects_baseline_and_augmentation_changes(key, value):
    saved = {'use_aug': True, 'aug_p_pa': .18, 'lambda_adv': .5, 'label_smoothing': .01,
             'concat_sat_seed': 2027, 'stage1_epochs': 15}
    current = dict(saved, **{key: value})
    with pytest.raises(ValueError, match=key):
        validate_ecrs_resume_args(SimpleNamespace(**current), saved)


def test_resume_runtime_paths_and_missing_auxiliary_state_are_explicit():
    validate_ecrs_resume_args({'output_dir': 'new', 'best_save_path': 'new.pth', 'device': 'cuda'},
                             {'output_dir': 'old', 'best_save_path': 'old.pth', 'device': 'cpu'})
    with pytest.raises(ValueError, match='auxiliary state'):
        validate_ecrs_resume_args({'lambda_proto': .1}, {'lambda_proto': .1})
    runtime = ECRSRuntime(TinyModel())
    old = runtime.state_dict()
    old.pop('training_semantics')
    with pytest.raises(ValueError, match='semantics changed'):
        runtime.load_state_dict(old)


def test_source_rescue_harm_preserves_group_cancellations():
    result = evaluate_revision_paths(IdentityOnly(), [batch()], 'cpu')
    rx = result['gain_by_group']['per_rx']
    assert (rx['1']['rescue'], rx['1']['harm'], rx['1']['net']) == (1, 0, 1)
    assert (rx['3']['rescue'], rx['3']['harm'], rx['3']['net']) == (1, 1, 0)
    assert sum(c['rescue'] for c in rx.values()) == result['rescue']
    assert sum(c['harm'] for c in rx.values()) == result['harm']
    day = result['gain_by_group']['per_day']
    assert day['2']['rescue'] == day['2']['harm'] == 1


def test_missing_leo_augmentation_rejected_only_for_active_u_loss():
    args = SimpleNamespace(ecrs_version='v2', ecrs_u_pair_enabled=True,
                           use_concat_sat_channel_aug=False, lambda_ecrs_u_pair=.03)
    with pytest.raises(ValueError, match='synchronized'):
        validate_ecrs_revision_args(args)
    args.lambda_ecrs_u_pair = 0.
    validate_ecrs_revision_args(args)
