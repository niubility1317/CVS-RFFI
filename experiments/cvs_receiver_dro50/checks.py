import ast
import copy
import importlib
import json
from types import SimpleNamespace
import torch
from torch.nn import functional as F
from . import design as d
from .objective import group_risk
from .observer import Observer


def checks():
    torch.set_num_threads(2)
    generator = torch.Generator().manual_seed(2026101509)
    logits = torch.randn(11, 6, generator=generator, dtype=torch.float64, requires_grad=True)
    labels = torch.arange(11) % 6
    rx = torch.tensor([1] * 3 + [3] * 2 + [4] * 2 + [6] * 2 + [8] * 2)
    state = torch.zeros(5, dtype=torch.float64)
    raw = F.cross_entropy(logits, labels, label_smoothing=.01)
    erm, same, detail = group_risk(logits, labels, rx, state, 'ERM')
    assert torch.allclose(erm, raw, atol=1e-14, rtol=0) and torch.equal(state, same)
    balanced, _, b = group_risk(logits, labels, rx, state, 'BALANCED')
    means = torch.stack([F.cross_entropy(logits[rx == r], labels[rx == r], label_smoothing=.01) for r in d.RX])
    assert torch.allclose(balanced, means.mean(), atol=1e-14, rtol=0)
    dro, changed, info = group_risk(logits, labels, rx, state, 'DRO')
    expected_q = (.01 * means.detach()).softmax(0)
    assert torch.allclose(changed.softmax(0), expected_q, atol=1e-14, rtol=0)
    assert torch.allclose(dro, expected_q @ means, atol=1e-14, rtol=0)
    assert not changed.requires_grad and torch.equal(state, torch.zeros_like(state))
    assert int(changed.argmax()) == int(means.argmax())
    expected_gradient = torch.autograd.grad(expected_q @ means, logits, retain_graph=True)[0]
    actual_gradient = torch.autograd.grad(dro, logits, retain_graph=True)[0]
    assert torch.allclose(actual_gradient, expected_gradient, atol=1e-14, rtol=0)
    missing, ms, md = group_risk(logits[:3], labels[:3], rx[:3], state, 'DRO')
    assert torch.allclose(missing, F.cross_entropy(logits[:3], labels[:3], label_smoothing=.01), atol=1e-14)
    assert md['applied_weights'] == [1., 0., 0., 0., 0.] and torch.isfinite(ms).all()
    negatives = 0
    for role in ('U_s', 'V', 'query'):
        try:
            group_risk(logits, labels, rx, state, 'DRO', role=role)
        except ValueError:
            negatives += 1
        else:
            raise AssertionError('Forbidden receiver-risk role accepted')
    for receivers in (rx * 0, None):
        try:
            group_risk(logits, labels, receivers, state, 'DRO')
        except ValueError:
            negatives += 1
        else:
            raise AssertionError('Invalid receiver metadata accepted')
    for key, value in [('role', 'V'), ('step_size', .1), ('groups', [1, 3]), ('mode', 'OTHER')]:
        bad = d.config(d.training_rows()[0]); bad['receiver_risk'][key] = value
        try:
            d.validate(bad)
        except ValueError:
            negatives += 1
        else:
            raise AssertionError('Changed risk configuration accepted')
    # Execute the exact new native call site with an original CE graph. This
    # tests integration without reading IQ or a checkpoint.
    path = d.ROOT / 'experiments/adv3b02_xuc/code/SSDG/train_ssdg.py'
    tree = ast.parse(path.read_text(encoding='utf8'))
    hook = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                and ast.unparse(n.test) == '_labeled_identity_loss_hook is not None')
    compiled = compile(ast.Module(body=[hook], type_ignores=[]), '<actual-native-labelled-risk-hook>', 'exec')
    model = SimpleNamespace()
    observer = Observer(); observer.c = d.config(d.training_rows()[0]); observer.model = model
    observer.steps = 0; observer.logq = state.clone(); observer.params = []; observer.head = None
    from io import StringIO
    observer.stream = StringIO()
    args = SimpleNamespace(label_smoothing=.01, lambda_u=.16, lambda_ent=.01, ema_decay=.999)
    namespace = dict(_labeled_identity_loss_hook=observer.labeled, model=model, args=args,
        epoch=1, batch_idx=1, out_l={'tx_logits': logits}, y_l=labels, receiver_l_base=rx,
        labeled_clean_count=len(labels), loss_tx_l=raw)
    exec(compiled, namespace)
    assert namespace['loss_tx_l'] is raw and observer.steps == 1
    U = logits.sum() * 0
    assert observer.unlabeled(model=model, args=args, epoch=1, batch_idx=1, loss_l=raw,
        loss_u=U, loss_ent=U, active=False, logits=None, pseudo=None, mask=None) is U
    for row in d.training_rows():
        c = d.config(row); a = d.make_args(c, 'cpu')
        assert (a.epochs, a.label_epochs, a.pseudo_epochs, a.weight_decay) == (50, 33, 17, .0001)
        assert a.use_unlabeled and a.use_ema_teacher and not a.freeze_backbone
        assert c['receiver_risk']['mode'] in ('ERM', 'BALANCED', 'DRO')
    from .model import build
    initial = None
    for row in d.training_rows():
        candidate = build(d.config(row), 'cpu').eval()
        values = {k: v.clone() for k, v in candidate.state_dict().items()}
        if initial is None:
            initial = values
        else:
            assert set(initial) == set(values) and all(torch.equal(v, values[k]) for k, v in initial.items())
    candidate.train()
    x = torch.randn(10, 2, 256, generator=generator)
    scores = candidate.forward_identity_only(x)['tx_logits']
    loss, _, _ = group_risk(scores, torch.arange(10) % 6,
        torch.tensor([r for r in d.RX for _ in range(2)]), state, 'DRO')
    named = dict(candidate.named_parameters())
    objects = [named[n] for n in d.PARAMETERS]
    W = candidate.id_backbone.encoder.core.id_backbone.cls_head.weight
    gradients = torch.autograd.grad(loss, [*objects, W])
    assert W.shape == (6, 160) and all(torch.isfinite(g).all() and g.norm() > 1e-12 for g in gradients)
    for name in ('source', 'predict', 'score', 'feature_check', 'audit', 'preflight', 'dispatch', 'publish'):
        importlib.import_module(f'experiments.cvs_receiver_dro50.{name}')
    from .publish import configure, transport
    configure(); compile(transport.REMOTE, '<dro-remote>', 'exec'); compile(transport.INSPECT, '<dro-inspect>', 'exec')
    assert 'experiments.cvs_receiver_dro50.preflight' in transport.REMOTE
    assert 'experiments.cvs_receiver_dro50.dispatch' in transport.REMOTE
    return dict(status='PASS', protocol_and_config_negatives=negatives, exact_original_ERM_graph=True,
        known_group_mean_and_multiplicative_update=True, detached_adversarial_weights=True,
        missing_RX_handled=True, actual_native_hook_executed=True, original_U_loss_returned_unchanged=True,
        native_budgets_verified=True, three_actual_initial_states_equal=True,
        actual_t1_t2_SIG_and_default_W_joint_gradients_nonzero=True,
        real_IQ_or_checkpoint_read=False, head_fit_updates=0)


if __name__ == '__main__':
    print(json.dumps(checks()))
